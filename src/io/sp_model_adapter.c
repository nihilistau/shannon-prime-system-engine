/* sp_model_adapter.c — reconstruct a qwen3_model from a loaded .sp-model so the
 * existing gemma3_forward / qwen3_forward run unchanged (E_FMT_4). See sp_model.h.
 *
 * The forward reads weights through exactly three engine entry points:
 *   - matmul(m, W, ...)   -> sp_arena_find(m->arena, W->name)        (Q8 codes)
 *   - embed_row(m, tok)   -> sp_arena_find(m->arena, token_embd->name)
 *   - as_f32(m, t)        -> m->released path: norm_src[i]==t -> norm_buf[i]
 * so the adapter needs (a) a packed arena rebuilt from the on-disk OK_Q8 codes +
 * ".scale" siblings, keyed by tensor name; (b) synthetic gguf_tensor entries the
 * qwen3_layer pointers reference (only `name` is load-bearing for matmul; norms
 * are matched by pointer identity); (c) owned f32 norm buffers wired through the
 * released path. Codes/scales are memcpy'd out of the mmap into malloc'd buffers
 * so sp_arena_free can release them normally (the ABI handle stays zero-malloc;
 * only this test-adapter path copies). */
#define _CRT_SECURE_NO_WARNINGS
#include "sp_engine/sp_model.h"
#include "sp_engine/model.h"
#include "sp_engine/arena.h"
#include "sp/frobenius_lift.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

void sp_set_error(const char *msg);

/* Build a Q8 sp_frob_packed_tensor from the .sp-model OK_Q8 codes + ".scale"
 * sibling. dims on disk are [cols=in, rows=out] (GGUF ne0=in, ne1=out). */
static int rebuild_q8(const sp_model *m, const char *name, sp_arena_tensor *out) {
    const sp_tensor_entry *codes_e = sp_model_find_tensor(m, name);
    if (!codes_e) { sp_set_error("adapter: missing OK_Q8 tensor"); return 1; }
    if (codes_e->dtype_id != SP_DT_OK_Q8 || codes_e->n_dims < 2) {
        sp_set_error("adapter: tensor not OK_Q8 rank>=2"); return 1;
    }
    char sname[80];
    snprintf(sname, sizeof sname, "%s.scale", name);
    const sp_tensor_entry *scale_e = sp_model_find_tensor(m, sname);
    if (!scale_e || scale_e->dtype_id != SP_DT_FROBENIUS_SCALE_FP32) {
        sp_set_error("adapter: missing .scale sibling"); return 1;
    }
    int cols = (int)codes_e->dims[0];       /* in  */
    int rows = (int)codes_e->dims[1];       /* out */
    if (cols <= 0 || rows <= 0) { sp_set_error("adapter: bad Q8 dims"); return 1; }
    size_t ncode = (size_t)rows * cols;
    if (codes_e->size_bytes != ncode) { sp_set_error("adapter: Q8 size mismatch"); return 1; }
    if (scale_e->size_bytes != (size_t)rows * sizeof(float)) { sp_set_error("adapter: scale size mismatch"); return 1; }

    const void *cd = sp_model_tensor_data(m, codes_e);
    const void *sd = sp_model_tensor_data(m, scale_e);
    if (!cd || !sd) { sp_set_error("adapter: tensor data OOB"); return 1; }

    sp_frob_packed_tensor *pt = &out->pt;
    memset(out, 0, sizeof *out);
    snprintf(out->name, sizeof out->name, "%s", name);
    pt->rows = rows; pt->cols = cols; pt->codes_bytes = ncode;
    pt->codes     = (uint8_t *)malloc(ncode ? ncode : 1);
    pt->row_scale = (float   *)malloc((size_t)rows * sizeof(float));
    pt->row_off   = (size_t  *)malloc((size_t)rows * sizeof(size_t));
    pt->row_prec  = (uint8_t *)malloc((size_t)rows);
    if (!pt->codes || !pt->row_scale || !pt->row_off || !pt->row_prec) {
        sp_frob_packed_free(pt); sp_set_error("adapter: OOM packed tensor"); return 1;
    }
    memcpy(pt->codes, cd, ncode);
    memcpy(pt->row_scale, sd, (size_t)rows * sizeof(float));
    for (int r = 0; r < rows; r++) { pt->row_off[r] = (size_t)r * cols; pt->row_prec[r] = 8; }
    return 0;
}

/* Copy an on-disk F32 tensor into a fresh owned buffer. */
static float *copy_f32(const sp_model *m, const char *name, int *n_out) {
    const sp_tensor_entry *e = sp_model_find_tensor(m, name);
    if (!e || e->dtype_id != SP_DT_F32) { sp_set_error("adapter: missing F32 norm"); return NULL; }
    size_t n = e->size_bytes / sizeof(float);
    const void *d = sp_model_tensor_data(m, e);
    if (!d) { sp_set_error("adapter: norm data OOB"); return NULL; }
    float *b = (float *)malloc(n ? n * sizeof(float) : sizeof(float));
    if (!b) { sp_set_error("adapter: OOM norm"); return NULL; }
    memcpy(b, d, n * sizeof(float));
    if (n_out) *n_out = (int)n;
    return b;
}

struct qwen3_model *sp_model_to_qwen3(const sp_model *m) {
    if (!m) { sp_set_error("adapter: null sp_model"); return NULL; }
    const sp_model_header *h = sp_model_get_header(m);

    qwen3_model *q = (qwen3_model *)calloc(1, sizeof *q);
    if (!q) { sp_set_error("adapter: OOM model"); return NULL; }
    q->gguf = NULL; q->released = 1;          /* norms served from owned buffers */

    /* arch_struct payload is the engine qwen3_config (v0 §3 arch_struct). */
    if (h->arch_struct_size < sizeof(qwen3_config)) { sp_set_error("adapter: arch_struct too small"); free(q); return NULL; }
    memcpy(&q->cfg, h->arch_struct, sizeof(qwen3_config));
    const qwen3_config *c = &q->cfg;
    if (c->n_vocab != h->vocab_size) { sp_set_error("adapter: vocab mismatch arch_struct"); free(q); return NULL; }

    /* synthetic gguf_tensor entries: the layer pointers reference these. Layout:
     *   [0]                     token_embd
     *   [1]                     output_norm
     *   [2 + L*NPL + k]         per-layer tensors (k in [0,NPL))
     * where NPL covers every tensor a layer's pointers need. */
    const int NPL = 13;   /* attn_norm,q,k,v,output,q_norm,k_norm,post_attn_norm,ffn_norm,gate,up,down,post_ffw_norm */
    int n_t = 2 + (int)c->n_layers * NPL;
    gguf_tensor *T = (gguf_tensor *)calloc((size_t)n_t, sizeof(gguf_tensor));
    q->layers = (qwen3_layer *)calloc(c->n_layers, sizeof(qwen3_layer));
    /* owned norm wiring (released path): up to 6 norms per layer (gemma3:
     * attn_norm, ffn_norm, attn_q_norm, attn_k_norm, post_attn_norm, post_ffw_norm)
     * + output_norm. */
    int norm_cap = (int)c->n_layers * 6 + 1;
    q->norm_src = (const gguf_tensor **)malloc((size_t)norm_cap * sizeof(*q->norm_src));
    q->norm_buf = (float **)malloc((size_t)norm_cap * sizeof(*q->norm_buf));
    /* packed matmul tensors: 7/layer + embedding + (optional untied LM head) */
    int arena_cap = (int)c->n_layers * 7 + 2;
    sp_arena_tensor *ats = (sp_arena_tensor *)calloc((size_t)arena_cap, sizeof(sp_arena_tensor));
    if (!T || !q->layers || !q->norm_src || !q->norm_buf || !ats) {
        sp_set_error("adapter: OOM tables"); free(T); free(ats); qwen3_free(q); return NULL;
    }

    int ti = 0, ai = 0, ni = 0, rc = 0;
    #define NEW_T(nm) (snprintf(T[ti].name, sizeof T[ti].name, "%s", (nm)), &T[ti++])
    #define ADD_NORM(tp, nm) do { \
        int len = 0; float *b = copy_f32(m, (nm), &len); \
        if (!b) { rc = 1; } else { \
            gguf_tensor *gt = NEW_T(nm); gt->n_dims = 1; gt->dims[0] = (uint64_t)len; \
            gt->type = 0 /*F32*/; gt->n_elements = (uint64_t)len; \
            (tp) = gt; q->norm_src[ni] = gt; q->norm_buf[ni] = b; ni++; } } while (0)
    #define ADD_Q8(tp, nm) do { \
        gguf_tensor *gt = NEW_T(nm); \
        if (rebuild_q8(m, (nm), &ats[ai])) { rc = 1; } else { \
            const sp_frob_packed_tensor *pt = &ats[ai].pt; \
            gt->n_dims = 2; gt->dims[0] = (uint64_t)pt->cols; gt->dims[1] = (uint64_t)pt->rows; \
            gt->n_elements = (uint64_t)pt->cols * pt->rows; gt->type = 1 /*F16 placeholder*/; \
            (tp) = gt; ai++; } } while (0)

    /* embedding (token_embd) — packed Q8; tied LM head reuses it. */
    ADD_Q8(q->token_embd, "token_embd.weight");
    /* output_norm */
    ADD_NORM(q->output_norm, "output_norm.weight");
    /* tied LM head: m->output == token_embd (Gemma3). Untied models would carry
     * an "output.weight" Q8 tensor; handle if present. */
    if (sp_model_find_tensor(m, "output.weight")) {
        ADD_Q8(q->output, "output.weight");
        q->cfg.tied_embedding = 0;
    } else {
        q->output = q->token_embd; q->cfg.tied_embedding = 1;
    }

    char nm[96];
    for (uint32_t i = 0; i < c->n_layers && rc == 0; i++) {
        qwen3_layer *L = &q->layers[i];
        #define LN(suffix) (snprintf(nm, sizeof nm, "blk.%u." suffix, i), nm)
        ADD_NORM(L->attn_norm, LN("attn_norm.weight"));
        ADD_Q8(L->attn_q,      LN("attn_q.weight"));
        ADD_Q8(L->attn_k,      LN("attn_k.weight"));
        ADD_Q8(L->attn_v,      LN("attn_v.weight"));
        ADD_Q8(L->attn_output, LN("attn_output.weight"));
        ADD_NORM(L->ffn_norm,  LN("ffn_norm.weight"));
        ADD_Q8(L->ffn_gate,    LN("ffn_gate.weight"));
        ADD_Q8(L->ffn_up,      LN("ffn_up.weight"));
        ADD_Q8(L->ffn_down,    LN("ffn_down.weight"));
        if (c->arch == SP_ARCH_GEMMA3) {
            ADD_NORM(L->attn_q_norm,    LN("attn_q_norm.weight"));
            ADD_NORM(L->attn_k_norm,    LN("attn_k_norm.weight"));
            ADD_NORM(L->post_attn_norm, LN("post_attention_norm.weight"));
            ADD_NORM(L->post_ffw_norm,  LN("post_ffw_norm.weight"));
        } else if (c->has_qk_norm) {
            ADD_NORM(L->attn_q_norm, LN("attn_q_norm.weight"));
            ADD_NORM(L->attn_k_norm, LN("attn_k_norm.weight"));
        }
        #undef LN
    }
    q->n_norm = ni;
    #undef NEW_T
    #undef ADD_NORM
    #undef ADD_Q8

    if (rc) {
        /* free the packed tensors not yet adopted by an arena */
        for (int k = 0; k < ai; k++) sp_frob_packed_free(&ats[k].pt);
        free(T); free(ats); qwen3_free(q); return NULL;
    }

    /* Stash the synthetic tensor array on the model BEFORE adopting the arena, so
     * any later failure path frees it via qwen3_free. The layer pointers reference
     * T; qwen3_free releases it. */
    q->synth_tensors = T;

    q->arena = sp_arena_from_packed(ats, ai, 8);
    free(ats);   /* arena copied the descriptors; it now owns the inner buffers */
    if (!q->arena) {
        /* OOM only: the per-tensor pt buffers are leaked here (ats already freed),
         * but teardown stays double-free-safe. */
        sp_set_error("adapter: arena build");
        qwen3_free(q); return NULL;
    }
    return q;
}
