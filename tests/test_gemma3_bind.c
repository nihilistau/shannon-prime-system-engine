/* test_gemma3_bind.c — GEMMA3_BIND: Gemma3-1B config parse + weight binding.
 * Mirrors test_model.c (Qwen3). Asserts the known Gemma3-1B architecture and that
 * the shared loader (arch-dispatched) binds every per-layer tensor incl. the
 * sandwich (post-attn / post-ffw) norms and the per-head QK norms, with the LM
 * head tied to token_embd. */
#define _CRT_SECURE_NO_WARNINGS
#include "sp/sp_test.h"
#include "sp_engine/model.h"
#include <stdio.h>

#ifndef SP_GEMMA3_GGUF
#define SP_GEMMA3_GGUF "gemma-3-1b-it-f16.gguf"
#endif

static void GEMMA3_BIND(void) {
    qwen3_model *m = qwen3_load(SP_GEMMA3_GGUF);   /* shared loader, arch-dispatched */
    SP_CHECK(m != NULL, "gemma3 load");
    if (!m) return;
    const qwen3_config *c = &m->cfg;
    SP_CHECK(c->arch == SP_ARCH_GEMMA3,        "arch == gemma3");
    SP_CHECK_EQ_I64(c->n_layers, 26,           "26 layers");
    SP_CHECK_EQ_I64(c->n_embd, 1152,           "n_embd 1152");
    SP_CHECK_EQ_I64(c->n_ff, 6912,             "n_ff 6912");
    SP_CHECK_EQ_I64(c->n_head, 4,              "4 q heads");
    SP_CHECK_EQ_I64(c->n_head_kv, 1,           "1 kv head");
    SP_CHECK_EQ_I64(c->head_dim, 256,          "head_dim 256");
    SP_CHECK_EQ_I64(c->n_vocab, 262144,        "vocab 262144");
    SP_CHECK_EQ_I64(c->sliding_window, 512,    "sliding window 512");
    SP_CHECK(c->tied_embedding,                "tied LM head");
    SP_CHECK(m->output == m->token_embd,       "output aliases token_embd");
    SP_CHECK(c->has_qk_norm,                   "per-head QK-norm present");
    int bound = 1;
    for (uint32_t i = 0; i < c->n_layers; i++) {
        const qwen3_layer *L = &m->layers[i];
        if (!L->attn_norm || !L->attn_q || !L->attn_k || !L->attn_v ||
            !L->attn_q_norm || !L->attn_k_norm || !L->attn_output ||
            !L->post_attn_norm || !L->ffn_norm || !L->ffn_gate ||
            !L->ffn_up || !L->ffn_down || !L->post_ffw_norm) { bound = 0; break; }
    }
    SP_CHECK(bound, "all 26 layers fully bound (incl. sandwich + QK norms)");
    qwen3_free(m);
}

int main(void) { SP_RUN(GEMMA3_BIND); return SP_DONE(); }
