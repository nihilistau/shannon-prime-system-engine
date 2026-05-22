/* arena.c — build the packed-weight arena (sp_engine/arena.h). Per-row Frobenius
 * Q8/Q4 over the model's matmul weights, quantized once at load. */
#include "sp_engine/arena.h"
#include "sp_engine/model.h"
#include "sp/frobenius_lift.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct sp_arena {
    sp_arena_tensor *t;
    int    n;
    int    precision;     /* 8 or 4 */
    size_t bytes;
    long   promoted;      /* Q4 rows stored Q8 */
    long   total_rows;
};

/* bytes occupied by `n` contiguous elements of a ggml weight row (matches the
 * forward-pass dequant: F32/F16/Q8_0). */
static size_t row_bytes(uint32_t type, int n) {
    switch (type) {
        case GGML_T_F32:  return (size_t)n * 4;
        case GGML_T_F16:  return (size_t)n * 2;
        case GGML_T_Q8_0: return (size_t)(n / 32) * 34;
        default:          return 0;
    }
}

/* Quantize one GGUF weight tensor W (ggml layout [ne0=in=cols, ne1=out=rows])
 * into the arena slot `out`. Returns 0 on success. */
static int build_tensor(const qwen3_model *m, const gguf_tensor *W, int prec,
                        float promote, sp_arena_tensor *out, long *promoted) {
    const uint8_t *base = (const uint8_t *)gguf_tensor_data(m->gguf, W);
    if (!base || W->n_dims < 2) return 1;
    int cols = (int)W->dims[0];     /* in  */
    int rows = (int)W->dims[1];     /* out */
    size_t rb = row_bytes(W->type, cols);
    if (rb == 0 || cols <= 0 || rows <= 0) return 1;

    float  *wrow = (float *)malloc((size_t)cols * sizeof(float));
    int8_t *tmp  = (int8_t *)malloc((size_t)cols);                 /* Q4 pack scratch */
    out->row_prec  = (uint8_t *)malloc((size_t)rows);
    out->row_scale = (float *)malloc((size_t)rows * sizeof(float));
    out->row_off   = (size_t *)malloc((size_t)rows * sizeof(size_t));
    out->codes     = (uint8_t *)malloc((size_t)rows * (size_t)cols);  /* upper bound (all Q8) */
    if (!wrow || !tmp || !out->row_prec || !out->row_scale || !out->row_off || !out->codes) {
        free(wrow); free(tmp); return 1;
    }
    out->rows = rows; out->cols = cols;
    snprintf(out->name, sizeof out->name, "%s", W->name);

    size_t off = 0;
    int rc = 0;
    for (int j = 0; j < rows && rc == 0; j++) {
        if (sp_dequant_row(base + (size_t)j * rb, W->type, cols, wrow)) { rc = 1; break; }
        float s = sp_frob_row_scale(wrow, cols);
        int p = prec;
        if (prec == 4 && sp_frob_q4_row_relerr(wrow, cols) > promote) { p = 8; (*promoted)++; }
        out->row_prec[j] = (uint8_t)p;
        out->row_scale[j] = s;
        out->row_off[j] = off;
        uint8_t *dst = out->codes + off;
        if (p == 8) {
            int8_t *q = (int8_t *)dst;
            for (int i = 0; i < cols; i++) q[i] = sp_frob_quant1(wrow[i], s);
            off += (size_t)cols;
        } else {
            for (int i = 0; i < cols; i++) tmp[i] = sp_frob_quant1_q4(wrow[i], s);
            sp_frob_q4_pack(tmp, cols, dst);
            off += (size_t)((cols + 1) / 2);
        }
    }
    free(wrow); free(tmp);
    if (rc) return 1;
    out->codes_bytes = off;
    uint8_t *shr = (uint8_t *)realloc(out->codes, off ? off : 1);   /* shrink to actual */
    if (shr) out->codes = shr;
    return 0;
}

sp_arena *sp_arena_build(const qwen3_model *m, int precision, float q4_promote,
                         int include_embed) {
    if (!m || (precision != 8 && precision != 4)) return NULL;
    const qwen3_config *c = &m->cfg;

    /* collect the matmul weight tensors: 7 per layer + LM head `output`
     * (only if untied — a tied output is the embedding) + optionally the
     * embedding itself (1b, needed before releasing the source). */
    int cap = (int)c->n_layers * 7 + 2;
    const gguf_tensor **src = (const gguf_tensor **)malloc((size_t)cap * sizeof(*src));
    if (!src) return NULL;
    int n = 0;
    for (uint32_t i = 0; i < c->n_layers; i++) {
        const qwen3_layer *L = &m->layers[i];
        src[n++] = L->attn_q;  src[n++] = L->attn_k; src[n++] = L->attn_v;
        src[n++] = L->attn_output;
        src[n++] = L->ffn_gate; src[n++] = L->ffn_up; src[n++] = L->ffn_down;
    }
    if (m->output && m->output != m->token_embd) src[n++] = m->output;
    if (include_embed && m->token_embd) src[n++] = m->token_embd;

    sp_arena *a = (sp_arena *)calloc(1, sizeof *a);
    if (!a) { free((void *)src); return NULL; }
    a->t = (sp_arena_tensor *)calloc((size_t)n, sizeof(sp_arena_tensor));
    if (!a->t) { free((void *)src); free(a); return NULL; }
    a->n = n; a->precision = precision;

    for (int k = 0; k < n; k++) {
        if (build_tensor(m, src[k], precision, q4_promote, &a->t[k], &a->promoted)) {
            free((void *)src); sp_arena_free(a); return NULL;
        }
        a->bytes += a->t[k].codes_bytes
                  + (size_t)a->t[k].rows * (sizeof(float) + sizeof(size_t) + 1);
        a->total_rows += a->t[k].rows;
    }
    free((void *)src);
    return a;
}

void sp_arena_free(sp_arena *a) {
    if (!a) return;
    for (int k = 0; k < a->n; k++) {
        free(a->t[k].row_prec); free(a->t[k].row_scale);
        free(a->t[k].row_off);  free(a->t[k].codes);
    }
    free(a->t);
    free(a);
}

const sp_arena_tensor *sp_arena_find(const sp_arena *a, const char *name) {
    if (!a || !name) return NULL;
    for (int k = 0; k < a->n; k++)
        if (strcmp(a->t[k].name, name) == 0) return &a->t[k];
    return NULL;
}

int sp_arena_dequant_row(const sp_arena_tensor *at, int r, float *dst) {
    if (!at || r < 0 || r >= at->rows) return 1;
    const uint8_t *rc = at->codes + at->row_off[r];
    if (at->row_prec[r] == 8) {
        const int8_t *cp = (const int8_t *)rc;
        float inv = at->row_scale[r] / 127.0f;
        for (int i = 0; i < at->cols; i++) dst[i] = (float)cp[i] * inv;
    } else {
        float inv = at->row_scale[r] / 7.0f;
        int8_t v;
        for (int i = 0; i < at->cols; i++) {
            uint8_t b = (i & 1) ? (uint8_t)(rc[i >> 1] >> 4) : (uint8_t)(rc[i >> 1] & 0xF);
            v = (int8_t)((b & 0x8) ? (int)b - 16 : (int)b);
            dst[i] = (float)v * inv;
        }
    }
    return 0;
}

size_t sp_arena_bytes(const sp_arena *a)      { return a ? a->bytes : 0; }
int    sp_arena_precision(const sp_arena *a)  { return a ? a->precision : 0; }
long   sp_arena_promoted(const sp_arena *a)   { return a ? a->promoted : 0; }
long   sp_arena_total_rows(const sp_arena *a) { return a ? a->total_rows : 0; }
