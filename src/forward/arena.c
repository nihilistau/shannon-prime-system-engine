/* arena.c — build the packed-weight arena (sp_engine/arena.h). Per-row Frobenius
 * Q8/Q4 over the model's matmul weights, quantized once at load. */
#include "sp_engine/arena.h"
#include "sp_engine/model.h"
#include "sp/frobenius_lift.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Format-lock (Piece 3, roadmap §8.2.2): the packed arena byte layout + dequant
 * contract are owned by the math core (sp/frobenius_lift.h); these asserts make a
 * silent drift of that contract a compile error on the engine side too. A change
 * REQUIRES bumping SP_FROB_ARENA_LAYOUT_VERSION (math core) + a cross-backend migration. */
_Static_assert(SP_FROB_ARENA_LAYOUT_VERSION == 1u, "arena layout v1 frozen; bump + migrate to change");
_Static_assert(SP_FROB_QMAX  == 127, "arena Q8 dequant: code in [-127,127], q*scale/127");
_Static_assert(SP_FROB_QMAX4 == 7,   "arena Q4 dequant: code in [-7,7], q*scale/7, two per byte");
_Static_assert(sizeof(float) == 4,   "arena per-row Frobenius scale is 4-byte IEEE-754 on the wire");

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

/* Row reader for sp_frob_pack_tensor: dequant GGUF source row `j` (F32/F16/Q8_0)
 * to `cols` f32. Keeps the GGUF format knowledge engine-side; the math-core packer
 * never sees a gguf_tensor and never materializes the whole tensor as f32. */
typedef struct {
    const uint8_t *base;   /* start of the tensor's data in the GGUF mapping */
    uint32_t       type;   /* ggml type of the source rows */
    size_t         rb;     /* bytes per source row */
    int            cols;
} gguf_row_ctx;

static int gguf_get_row(void *ctx, int j, float *dst) {
    const gguf_row_ctx *g = (const gguf_row_ctx *)ctx;
    return sp_dequant_row(g->base + (size_t)j * g->rb, g->type, g->cols, dst);
}

/* Pack one GGUF weight tensor W (ggml layout [ne0=in=cols, ne1=out=rows]) into the
 * arena slot `out` via the math-core mixed-precision packer. Returns 0 on success. */
static int build_tensor(const qwen3_model *m, const gguf_tensor *W, int prec,
                        float promote, sp_arena_tensor *out, long *promoted) {
    const uint8_t *base = (const uint8_t *)gguf_tensor_data(m->gguf, W);
    if (!base || W->n_dims < 2) return 1;
    int cols = (int)W->dims[0];     /* in  */
    int rows = (int)W->dims[1];     /* out */
    size_t rb = row_bytes(W->type, cols);
    if (rb == 0 || cols <= 0 || rows <= 0) return 1;
    snprintf(out->name, sizeof out->name, "%s", W->name);
    gguf_row_ctx ctx = { base, W->type, rb, cols };
    return sp_frob_pack_tensor(rows, cols, prec, promote, gguf_get_row, &ctx, &out->pt, promoted);
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
        a->bytes += sp_frob_packed_tensor_bytes(&a->t[k].pt);
        a->total_rows += a->t[k].pt.rows;
    }
    free((void *)src);
    return a;
}

sp_arena *sp_arena_from_packed(const sp_arena_tensor *ts, int n, int precision) {
    if (!ts || n < 0 || (precision != 8 && precision != 4)) return NULL;
    sp_arena *a = (sp_arena *)calloc(1, sizeof *a);
    if (!a) return NULL;
    a->t = (sp_arena_tensor *)calloc((size_t)(n ? n : 1), sizeof(sp_arena_tensor));
    if (!a->t) { free(a); return NULL; }
    a->n = n; a->precision = precision;
    for (int k = 0; k < n; k++) {
        a->t[k] = ts[k];                    /* shallow: arena now owns the pt buffers */
        a->bytes += sp_frob_packed_tensor_bytes(&a->t[k].pt);
        a->total_rows += a->t[k].pt.rows;
    }
    return a;
}

void sp_arena_free(sp_arena *a) {
    if (!a) return;
    for (int k = 0; k < a->n; k++) sp_frob_packed_free(&a->t[k].pt);
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
    return at ? sp_frob_packed_dequant_row(&at->pt, r, dst) : 1;
}

size_t sp_arena_bytes(const sp_arena *a)      { return a ? a->bytes : 0; }
int    sp_arena_precision(const sp_arena *a)  { return a ? a->precision : 0; }
long   sp_arena_promoted(const sp_arena *a)   { return a ? a->promoted : 0; }
long   sp_arena_total_rows(const sp_arena *a) { return a ? a->total_rows : 0; }
