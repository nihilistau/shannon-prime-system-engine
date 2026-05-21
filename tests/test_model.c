/* test_model.c — 2-CPU.B step 1: Qwen3 config + weight binding + dequant.
 * Validates the bound model against the known Qwen3-0.6B architecture and
 * checks every weight tensor's shape is consistent with the config. */
#include "sp/sp_test.h"
#include "sp_engine/model.h"

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#ifndef SP_QWEN3_GGUF
#define SP_QWEN3_GGUF "Qwen3-0.6B-f16.gguf"
#endif

/* a tensor with ne0==in and ne1==out (ggml stores [in,out]) */
static int shape2(const gguf_tensor *t, uint64_t in, uint64_t out) {
    return t && t->n_dims == 2 && t->dims[0] == in && t->dims[1] == out;
}
static int shape1(const gguf_tensor *t, uint64_t n) {
    return t && t->n_dims == 1 && t->dims[0] == n;
}

static void MODEL_BIND(void) {
    qwen3_model *m = qwen3_load(SP_QWEN3_GGUF);
    SP_CHECK(m != NULL, "qwen3_load(Qwen3-0.6B)");
    if (!m) { fprintf(stderr, "    (model not found: %s)\n", SP_QWEN3_GGUF); return; }
    const qwen3_config *c = &m->cfg;

    /* config matches the known Qwen3-0.6B architecture */
    SP_CHECK_EQ_I64(c->n_layers, 28, "n_layers");
    SP_CHECK_EQ_I64(c->n_embd, 1024, "n_embd");
    SP_CHECK_EQ_I64(c->n_ff, 3072, "n_ff");
    SP_CHECK_EQ_I64(c->n_head, 16, "n_head");
    SP_CHECK_EQ_I64(c->n_head_kv, 8, "n_head_kv (GQA)");
    SP_CHECK_EQ_I64(c->head_dim, 128, "head_dim");
    SP_CHECK_EQ_I64(c->n_vocab, 151936, "n_vocab");
    SP_CHECK(c->rope_freq_base == 1000000.0f, "rope_freq_base 1e6");
    SP_CHECK(fabsf(c->rms_eps - 1e-6f) < 1e-12f, "rms_eps 1e-6");
    SP_CHECK(c->has_qk_norm == 1, "Qwen3 per-head QK-norm present");
    SP_CHECK(c->tied_embedding == 0, "untied output (separate output.weight)");

    const uint64_t E = c->n_embd, FF = c->n_ff, V = c->n_vocab;
    const uint64_t Q = (uint64_t)c->n_head * c->head_dim;     /* 2048 */
    const uint64_t KV = (uint64_t)c->n_head_kv * c->head_dim; /* 1024 */

    /* embeddings + head */
    SP_CHECK(shape2(m->token_embd, E, V), "token_embd [n_embd, n_vocab]");
    SP_CHECK(shape1(m->output_norm, E), "output_norm [n_embd]");
    SP_CHECK(shape2(m->output, E, V), "output [n_embd, n_vocab]");

    /* every layer's tensors bound with consistent shapes */
    int bind_ok = 1, shape_ok = 1;
    for (uint32_t i = 0; i < c->n_layers; i++) {
        const qwen3_layer *L = &m->layers[i];
        if (!L->attn_norm || !L->attn_q || !L->attn_k || !L->attn_v || !L->attn_output ||
            !L->attn_q_norm || !L->attn_k_norm || !L->ffn_norm || !L->ffn_gate ||
            !L->ffn_up || !L->ffn_down) { bind_ok = 0; break; }
        shape_ok &= shape1(L->attn_norm, E);
        shape_ok &= shape2(L->attn_q, E, Q);
        shape_ok &= shape2(L->attn_k, E, KV);
        shape_ok &= shape2(L->attn_v, E, KV);
        shape_ok &= shape2(L->attn_output, Q, E);
        shape_ok &= shape1(L->attn_q_norm, c->head_dim);
        shape_ok &= shape1(L->attn_k_norm, c->head_dim);
        shape_ok &= shape1(L->ffn_norm, E);
        shape_ok &= shape2(L->ffn_gate, E, FF);
        shape_ok &= shape2(L->ffn_up, E, FF);
        shape_ok &= shape2(L->ffn_down, FF, E);
        if (!shape_ok) break;
    }
    SP_CHECK(bind_ok, "all 28 layers fully bound");
    SP_CHECK(shape_ok, "all per-layer tensor shapes consistent with config");

    /* dequant a row of the f16 embedding table -> finite, sane range */
    float *row = (float *)malloc(E * sizeof(float));
    const void *src = gguf_tensor_data(m->gguf, m->token_embd); /* row 0 (token 0) */
    int rc = sp_dequant_row(src, m->token_embd->type, (int)E, row);
    SP_CHECK(rc == 0, "dequant embedding row");
    int finite = 1; double mag = 0;
    for (uint64_t i = 0; i < E; i++) { if (!isfinite(row[i])) { finite = 0; break; } mag += fabs(row[i]); }
    SP_CHECK(finite, "dequant row all finite");
    SP_CHECK(mag > 0.0, "dequant row non-zero");
    free(row);

    fprintf(stderr, "    qwen3-0.6B bound: %u layers, Q=%llu KV=%llu (GQA %u:%u), "
            "head_dim=%u, vocab=%llu\n", c->n_layers, (unsigned long long)Q,
            (unsigned long long)KV, c->n_head, c->n_head_kv, c->head_dim,
            (unsigned long long)V);
    qwen3_free(m);
}

int main(void) {
    SP_RUN(MODEL_BIND);
    return SP_DONE();
}
