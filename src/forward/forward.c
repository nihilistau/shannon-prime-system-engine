/* forward.c — Qwen3 f32 reference forward pass (CPU scalar). E_CPU_2.
 *
 * 13-step transformer prefill over a token-ID sequence, causal:
 *   embed -> per layer { RMSNorm -> Q/K/V proj -> per-head QK-RMSNorm ->
 *     RoPE(NEOX) -> GQA causal attention (fp32 softmax) -> O proj -> residual
 *     -> RMSNorm -> SwiGLU FFN -> residual } -> final RMSNorm -> LM head.
 * Weights are dequantized on demand (each weight row once per matmul).
 */
#define _CRT_SECURE_NO_WARNINGS   /* getenv is fine here (MSVC C4996) */
#include "sp_engine/model.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ggml-faithful precision mode: when SP_ENGINE_F16_ACT=1, matmul rounds each
 * activation to F16 before the dot, mimicking ggml downcasting src1 to F16 for an
 * F16 weight. Read once (see qwen3_forward). Default 0 = pure-f32 reference path,
 * which is what E_CPU_2 validates and what E_CPU_3+ quantize against. (The pure-f32
 * path is actually *closer* to ggml in KL terms; F16-act exists to demonstrate the
 * residual per-logit gap is QK-RMSNorm-amplified F16 precision, not a logic error.) */
static int g_f16_act = 0;

/* bytes occupied by `n` contiguous elements of a ggml weight row. */
static size_t row_bytes(uint32_t type, int n) {
    switch (type) {
        case GGML_T_F32:  return (size_t)n * 4;
        case GGML_T_F16:  return (size_t)n * 2;
        case GGML_T_Q8_0: return (size_t)(n / 32) * 34;
        default:          return 0;
    }
}

/* Y[t,j] = sum_i W[i,j] * X[t,i]; W is the gguf tensor [in, out] (ne0=in). */
static int matmul(const qwen3_model *m, const gguf_tensor *W,
                  const float *X, int n_tok, int in, int out, float *Y) {
    const uint8_t *base = (const uint8_t *)gguf_tensor_data(m->gguf, W);
    size_t rb = row_bytes(W->type, in);
    if (!base || rb == 0) return 1;
    float *wrow = (float *)malloc((size_t)in * sizeof(float));
    if (!wrow) return 1;
    /* When mimicking ggml's F16 src1 downcast, round the activation rows once
     * up front (the same rounded x is reused across all `out` weight rows). */
    float *xr = NULL;
    if (g_f16_act) {
        xr = (float *)malloc((size_t)n_tok * in * sizeof(float));
        if (!xr) { free(wrow); return 1; }
        for (size_t i = 0; i < (size_t)n_tok * in; i++)
            xr[i] = sp_f16_to_f32(sp_f32_to_f16(X[i]));
        X = xr;
    }
    for (int j = 0; j < out; j++) {
        if (sp_dequant_row(base + (size_t)j * rb, W->type, in, wrow)) { free(wrow); free(xr); return 1; }
        for (int t = 0; t < n_tok; t++) {
            const float *x = X + (size_t)t * in;
            float acc = 0.0f;
            for (int i = 0; i < in; i++) acc += wrow[i] * x[i];
            Y[(size_t)t * out + j] = acc;
        }
    }
    free(wrow);
    free(xr);
    return 0;
}

/* RMSNorm over n elements with f32 weight w: out_i = x_i / sqrt(mean(x^2)+eps) * w_i. */
static void rmsnorm(const float *x, const float *w, int n, float eps, float *out) {
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    float scale = 1.0f / sqrtf((float)(ss / n) + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * scale * w[i];
}

/* in-place RMSNorm of a single head vector (length d) with f32 weight. */
static void rmsnorm_head(float *v, const float *w, int d, float eps) {
    double ss = 0.0;
    for (int i = 0; i < d; i++) ss += (double)v[i] * v[i];
    float scale = 1.0f / sqrtf((float)(ss / d) + eps);
    for (int i = 0; i < d; i++) v[i] = v[i] * scale * w[i];
}

/* NEOX RoPE on a head vector (length d) at position p: rotate pairs (i, i+d/2). */
static void rope_neox(float *v, int d, int p, float base) {
    int half = d / 2;
    for (int i = 0; i < half; i++) {
        float freq  = powf(base, -2.0f * (float)i / (float)d);
        float theta = (float)p * freq;
        float c = cosf(theta), s = sinf(theta);
        float a = v[i], b = v[i + half];
        v[i]        = a * c - b * s;
        v[i + half] = a * s + b * c;
    }
}

static const float *as_f32(const qwen3_model *m, const gguf_tensor *t) {
    /* norm weights are F32 1-D: read directly from the mapping. */
    return (const float *)gguf_tensor_data(m->gguf, t);
}

int qwen3_forward(const qwen3_model *m, const int32_t *tokens, int n_tok, float *logits) {
    const qwen3_config *c = &m->cfg;
    const int E = (int)c->n_embd, FF = (int)c->n_ff, HD = (int)c->head_dim;
    const int NH = (int)c->n_head, NKV = (int)c->n_head_kv;
    const int QD = NH * HD;          /* q proj width  (2048) */
    const int KVD = NKV * HD;        /* kv proj width (1024) */
    const int group = NH / NKV;      /* q-heads per kv-head (2) */
    const int V = (int)c->n_vocab;
    const float eps = c->rms_eps, base = c->rope_freq_base;
    const float ascale = 1.0f / sqrtf((float)HD);

    { const char *e = getenv("SP_ENGINE_F16_ACT"); g_f16_act = (e && e[0] == '1'); }

    int rc = 1;
    float *x   = (float *)malloc((size_t)n_tok * E * sizeof(float));   /* residual stream */
    float *nx  = (float *)malloc((size_t)n_tok * E * sizeof(float));   /* normed */
    float *q   = (float *)malloc((size_t)n_tok * QD * sizeof(float));
    float *k   = (float *)malloc((size_t)n_tok * KVD * sizeof(float));
    float *v   = (float *)malloc((size_t)n_tok * KVD * sizeof(float));
    float *ao  = (float *)malloc((size_t)n_tok * QD * sizeof(float));  /* attn out (concat heads) */
    float *ap  = (float *)malloc((size_t)n_tok * E * sizeof(float));   /* attn out proj */
    float *g   = (float *)malloc((size_t)n_tok * FF * sizeof(float));
    float *up  = (float *)malloc((size_t)n_tok * FF * sizeof(float));
    float *dn  = (float *)malloc((size_t)n_tok * E * sizeof(float));
    float *sc  = (float *)malloc((size_t)n_tok * sizeof(float));       /* attn scores */
    if (!x || !nx || !q || !k || !v || !ao || !ap || !g || !up || !dn || !sc) goto done;

    /* ── embedding lookup: token t's embedding is the contiguous E floats at t*E ── */
    {
        const uint8_t *emb = (const uint8_t *)gguf_tensor_data(m->gguf, m->token_embd);
        size_t rb = row_bytes(m->token_embd->type, E);
        for (int t = 0; t < n_tok; t++)
            if (sp_dequant_row(emb + (size_t)tokens[t] * rb, m->token_embd->type, E, x + (size_t)t * E)) goto done;
    }

    for (uint32_t L = 0; L < c->n_layers; L++) {
        const qwen3_layer *ly = &m->layers[L];

        /* ── attention block ── */
        for (int t = 0; t < n_tok; t++)
            rmsnorm(x + (size_t)t * E, as_f32(m, ly->attn_norm), E, eps, nx + (size_t)t * E);

        if (matmul(m, ly->attn_q, nx, n_tok, E, QD, q)) goto done;
        if (matmul(m, ly->attn_k, nx, n_tok, E, KVD, k)) goto done;
        if (matmul(m, ly->attn_v, nx, n_tok, E, KVD, v)) goto done;

        /* per-head QK-RMSNorm (over head_dim) then NEOX RoPE at position t */
        const float *qn = as_f32(m, ly->attn_q_norm);
        const float *kn = as_f32(m, ly->attn_k_norm);
        for (int t = 0; t < n_tok; t++) {
            for (int h = 0; h < NH; h++) {
                float *qh = q + (size_t)t * QD + (size_t)h * HD;
                rmsnorm_head(qh, qn, HD, eps);
                rope_neox(qh, HD, t, base);
            }
            for (int h = 0; h < NKV; h++) {
                float *kh = k + (size_t)t * KVD + (size_t)h * HD;
                rmsnorm_head(kh, kn, HD, eps);
                rope_neox(kh, HD, t, base);
            }
        }

        /* GQA causal attention */
        for (int t = 0; t < n_tok; t++) {
            for (int h = 0; h < NH; h++) {
                int kvh = h / group;
                const float *qh = q + (size_t)t * QD + (size_t)h * HD;
                float maxs = -INFINITY;
                for (int s = 0; s <= t; s++) {
                    const float *kh = k + (size_t)s * KVD + (size_t)kvh * HD;
                    float d = 0.0f;
                    for (int i = 0; i < HD; i++) d += qh[i] * kh[i];
                    d *= ascale;
                    sc[s] = d;
                    if (d > maxs) maxs = d;
                }
                float sum = 0.0f;
                for (int s = 0; s <= t; s++) { sc[s] = expf(sc[s] - maxs); sum += sc[s]; }
                float inv = 1.0f / sum;
                float *out = ao + (size_t)t * QD + (size_t)h * HD;
                for (int i = 0; i < HD; i++) out[i] = 0.0f;
                for (int s = 0; s <= t; s++) {
                    float w = sc[s] * inv;
                    const float *vh = v + (size_t)s * KVD + (size_t)kvh * HD;
                    for (int i = 0; i < HD; i++) out[i] += w * vh[i];
                }
            }
        }

        if (matmul(m, ly->attn_output, ao, n_tok, QD, E, ap)) goto done;
        for (size_t i = 0; i < (size_t)n_tok * E; i++) x[i] += ap[i];   /* residual */

        /* ── FFN block (SwiGLU) ── */
        for (int t = 0; t < n_tok; t++)
            rmsnorm(x + (size_t)t * E, as_f32(m, ly->ffn_norm), E, eps, nx + (size_t)t * E);
        if (matmul(m, ly->ffn_gate, nx, n_tok, E, FF, g)) goto done;
        if (matmul(m, ly->ffn_up,   nx, n_tok, E, FF, up)) goto done;
        for (size_t i = 0; i < (size_t)n_tok * FF; i++) {
            float gv = g[i];
            float silu = gv / (1.0f + expf(-gv));
            g[i] = silu * up[i];
        }
        if (matmul(m, ly->ffn_down, g, n_tok, FF, E, dn)) goto done;
        for (size_t i = 0; i < (size_t)n_tok * E; i++) x[i] += dn[i];    /* residual */
    }

    /* ── final norm + LM head ── */
    for (int t = 0; t < n_tok; t++)
        rmsnorm(x + (size_t)t * E, as_f32(m, m->output_norm), E, eps, nx + (size_t)t * E);
    if (matmul(m, m->output, nx, n_tok, E, V, logits)) goto done;

    rc = 0;
done:
    free(x); free(nx); free(q); free(k); free(v); free(ao); free(ap);
    free(g); free(up); free(dn); free(sc);
    return rc;
}
