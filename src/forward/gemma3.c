/* gemma3.c — Gemma3 f32 reference forward pass (CPU). M_GEMMA3_CPU.
 *
 * Built on the shared kernels (kernels.{h,c}); only the Gemma deltas vs Qwen3
 * live here. Constants pinned from the oracle reference
 * (shannon-prime-lattice-llama/src/models/gemma3.cpp + llama-hparams.{h,cpp}):
 *
 *   - Embedding scale: x *= sqrtf(n_embd)  (ggml_scale on the looked-up embd).
 *   - RMSNorm: plain LLM_NORM_RMS (x/rms·w) for EVERY norm site — the (1+w) is
 *     baked into the GGUF weights at conversion, so the shared rmsnorm is exact
 *     (adding 1 would double-count). Same for the per-head QK norm.
 *   - QK-norm: rmsnorm_head over head_dim, on Q and K, BEFORE RoPE. V is neither
 *     normed nor roped.
 *   - RoPE: NEOX, n_rot = head_dim. base = 1e6 (rope.freq_base) on GLOBAL layers,
 *     10000 (rope_freq_base_train_swa default; no GGUF override) on LOCAL/SWA.
 *   - Layer pattern: set_swa_pattern(6) => swa_layers[il] = (il % 6 < 5); a layer
 *     is GLOBAL (full causal, base 1e6) iff (il % 6 == 5) (il = 5,11,17,23), else
 *     LOCAL (sliding window 512, base 10000). 4 global / 22 local of 26.
 *   - Query scale: f_attention_scale = 1/sqrt(n_embd_head_k) = 1/sqrt(head_dim)
 *     (the non-27B branch), applied to scores; build_attn kq_scale = 1.0. This is
 *     the same 1/sqrt(head_dim) the engine's `ascale` already is.
 *   - Sandwich norms + residual: sa_out = x + post_attn_norm(wo·attn);
 *     out = sa_out + post_ffw_norm(ffn(ffn_norm(sa_out))).
 *   - FFN: GeGLU — gelu_tanh(ffn_gate·x) * (ffn_up·x), then ffn_down.
 *   - No final-logit softcap (absent => 0). Tied LM head (output == token_embd).
 */
#define _CRT_SECURE_NO_WARNINGS
#include "sp_engine/model.h"
#include "sp_engine/kernels.h"

#include <stdlib.h>
#include <math.h>

/* GELU, tanh approximation (Gemma FFN, LLM_FFN_GELU => ggml_gelu). The oracle
 * evaluates this through an F16 lookup table; the f32 closed form here matches it
 * to the precision floor (§8.6.1), which the distributional gate tolerates. */
static float gelu_tanh(float x) {
    const float k = 0.7978845608028654f;   /* sqrt(2/pi) */
    return 0.5f * x * (1.0f + tanhf(k * (x + 0.044715f * x * x * x)));
}

int gemma3_forward(const qwen3_model *m, const int32_t *tokens, int n_tok, float *logits) {
    const qwen3_config *c = &m->cfg;
    const int E = (int)c->n_embd, FF = (int)c->n_ff, HD = (int)c->head_dim;
    const int NH = (int)c->n_head, NKV = (int)c->n_head_kv;
    const int QD = NH * HD;          /* q proj width  (1024) */
    const int KVD = NKV * HD;        /* kv proj width (256)  */
    const int group = NH / NKV;      /* q-heads per kv-head  */
    const int V = (int)c->n_vocab;
    const int SW = (int)c->sliding_window;
    const float eps = c->rms_eps;
    const float gbase = c->rope_freq_base;   /* global layers */
    const float lbase = 10000.0f;            /* local/SWA layers */
    const float ascale = 1.0f / sqrtf((float)HD);
    const float embscale = sqrtf((float)E);

    sp_kernels_read_env();

    int rc = 1;
    float *x   = (float *)malloc((size_t)n_tok * E * sizeof(float));   /* residual stream */
    float *nx  = (float *)malloc((size_t)n_tok * E * sizeof(float));   /* normed / post-norm scratch */
    float *q   = (float *)malloc((size_t)n_tok * QD * sizeof(float));
    float *k   = (float *)malloc((size_t)n_tok * KVD * sizeof(float));
    float *vv  = (float *)malloc((size_t)n_tok * KVD * sizeof(float));
    float *ao  = (float *)malloc((size_t)n_tok * QD * sizeof(float));  /* attn out (concat heads) */
    float *ap  = (float *)malloc((size_t)n_tok * E * sizeof(float));   /* attn out proj */
    float *g   = (float *)malloc((size_t)n_tok * FF * sizeof(float));
    float *up  = (float *)malloc((size_t)n_tok * FF * sizeof(float));
    float *dn  = (float *)malloc((size_t)n_tok * E * sizeof(float));
    float *sc  = (float *)malloc((size_t)n_tok * sizeof(float));       /* attn scores */
    if (!x || !nx || !q || !k || !vv || !ao || !ap || !g || !up || !dn || !sc) goto done;

    /* embedding lookup, scaled by sqrt(n_embd) */
    for (int t = 0; t < n_tok; t++) {
        if (embed_row(m, tokens[t], E, x + (size_t)t * E)) goto done;
        float *xt = x + (size_t)t * E;
        for (int i = 0; i < E; i++) xt[i] *= embscale;
    }

    for (uint32_t L = 0; L < c->n_layers; L++) {
        const qwen3_layer *ly = &m->layers[L];
        const int global = ((L % 6) == 5);              /* global attn layer? */
        const float rbase = global ? gbase : lbase;
        const int win = global ? -1 : SW;               /* full causal vs sliding window */

        /* ── attention block ── */
        for (int t = 0; t < n_tok; t++)
            rmsnorm(x + (size_t)t * E, as_f32(m, ly->attn_norm), E, eps, nx + (size_t)t * E);

        if (matmul(m, ly->attn_q, nx, n_tok, E, QD, q)) goto done;
        if (matmul(m, ly->attn_k, nx, n_tok, E, KVD, k)) goto done;
        if (matmul(m, ly->attn_v, nx, n_tok, E, KVD, vv)) goto done;

        /* per-head QK-RMSNorm (over head_dim) then NEOX RoPE; V untouched */
        const float *qn = as_f32(m, ly->attn_q_norm);
        const float *kn = as_f32(m, ly->attn_k_norm);
        for (int t = 0; t < n_tok; t++) {
            for (int h = 0; h < NH; h++) {
                float *qh = q + (size_t)t * QD + (size_t)h * HD;
                rmsnorm_head(qh, qn, HD, eps);
                rope_neox(qh, HD, t, rbase);
            }
            for (int h = 0; h < NKV; h++) {
                float *kh = k + (size_t)t * KVD + (size_t)h * HD;
                rmsnorm_head(kh, kn, HD, eps);
                rope_neox(kh, HD, t, rbase);
            }
        }

        /* GQA causal attention (local layers mask to the sliding window) */
        for (int t = 0; t < n_tok; t++)
            for (int h = 0; h < NH; h++)
                kernels_attn_head(q + (size_t)t * QD + (size_t)h * HD, k, vv, t, KVD,
                                  h / group, HD, ascale, win, sc,
                                  ao + (size_t)t * QD + (size_t)h * HD);

        if (matmul(m, ly->attn_output, ao, n_tok, QD, E, ap)) goto done;
        /* sandwich: sa_out = x + post_attn_norm(attn_out) */
        for (int t = 0; t < n_tok; t++) {
            rmsnorm(ap + (size_t)t * E, as_f32(m, ly->post_attn_norm), E, eps, nx + (size_t)t * E);
            float *xt = x + (size_t)t * E; const float *pt = nx + (size_t)t * E;
            for (int i = 0; i < E; i++) xt[i] += pt[i];
        }

        /* ── FFN block (GeGLU) ── */
        for (int t = 0; t < n_tok; t++)
            rmsnorm(x + (size_t)t * E, as_f32(m, ly->ffn_norm), E, eps, nx + (size_t)t * E);
        if (matmul(m, ly->ffn_gate, nx, n_tok, E, FF, g)) goto done;
        if (matmul(m, ly->ffn_up,   nx, n_tok, E, FF, up)) goto done;
        for (size_t i = 0; i < (size_t)n_tok * FF; i++)
            g[i] = gelu_tanh(g[i]) * up[i];
        if (matmul(m, ly->ffn_down, g, n_tok, FF, E, dn)) goto done;
        /* sandwich: out = sa_out + post_ffw_norm(ffn_out) */
        for (int t = 0; t < n_tok; t++) {
            rmsnorm(dn + (size_t)t * E, as_f32(m, ly->post_ffw_norm), E, eps, nx + (size_t)t * E);
            float *xt = x + (size_t)t * E; const float *pt = nx + (size_t)t * E;
            for (int i = 0; i < E; i++) xt[i] += pt[i];
        }
    }

    /* ── final norm + tied LM head ── */
    for (int t = 0; t < n_tok; t++)
        rmsnorm(x + (size_t)t * E, as_f32(m, m->output_norm), E, eps, nx + (size_t)t * E);
    if (matmul(m, m->output, nx, n_tok, E, V, logits)) goto done;

    rc = 0;
done:
    free(x); free(nx); free(q); free(k); free(vv); free(ao); free(ap);
    free(g); free(up); free(dn); free(sc);
    return rc;
}
