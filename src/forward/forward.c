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
#include "sp_engine/arena.h"
#include "sp/frobenius_lift.h"
#include "sp/poly_ring.h"
#include "sp/spinor_block.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

#if defined(SP_ENGINE_AVX2) || defined(SP_ENGINE_AVX512)
#include <immintrin.h>
#endif

/* f32 dot product. AVX2 (8-wide, FMA) when compiled with SP_ENGINE_AVX2 and not
 * forced scalar; falls back to the same sequential scalar reduction the scalar
 * path uses for the tail. E_CPU_4 gates the AVX2 path against the scalar one. */
static int g_scalar = 0;   /* SP_CPU_SCALAR=1 forces the scalar reduction */

/* NTT-attention (E_CPU_5): when SP_ENGINE_NTT_ATTN=1, each attention score <q,k>
 * is computed by the Phase-1C poly-ring kernel — quantize the (post-norm, post-
 * RoPE) head vectors to int32 (scale SP_NTT_ATTN_SCALE), recover <q,k> EXACTLY as
 * coefficient 0 of the negacyclic product (sp_pr_inner), then divide back out the
 * scale. Sieve OFF. Softmax + V-sum stay f32. Gated against the f32-dot baseline. */
static int g_ntt_attn = 0;
#define SP_NTT_ATTN_SCALE 65536.0   /* 2^16: |q_int| ~ 2^21, |<q,k>| ~ 2^49 << M/2 ~ 2^59 */
#define SP_KSTE_KV_SCALE  65536.0   /* fixed int32 quant for KSTE KV signatures (E_CPU_6) */

/* Inline VHT2+Spinor KV-cache compression (E_CPU_8). When SP_KV_SPINOR=1 each
 * post-norm/post-RoPE K and post-proj V head vector (head_dim long) is stored as
 * the frozen 63-byte Spinor block(s) and decoded back (lossy) before attention
 * reads it — the foundational KV codec (§4.5/§4.9), distinct from the KSTE sieve
 * overlay (E_CPU_6). Gate OFF skips it entirely => bit-identical to E_CPU_2. */
static int g_kv_spinor = 0;

/* Round-trip a length-d vector through the Spinor codec in place (encode then
 * decode). A Spinor block carries 55 anchors; head_dim=128 > 55, so split into
 * ceil(d/55) balanced chunks (128 -> 43/43/42). The frozen 63-byte block layout
 * is NOT modified; we just use ceil(d/55) of them per head vector. */
static void kv_spinor_roundtrip(float *vec, int d) {
    int nblk = (d + 54) / 55;
    if (nblk < 1) return;
    int base = d / nblk, extra = d % nblk, off = 0;
    sp_spinor_block_t blk;
    for (int b = 0; b < nblk; b++) {
        int len = base + (b < extra ? 1 : 0);
        sp_spinor_encode(vec + off, len, &blk);
        (void)sp_spinor_decode(&blk, vec + off, len);   /* own freshly-encoded block: CRC valid */
        off += len;
    }
}

static float dot_f32(const float *a, const float *b, int n) {
#if defined(SP_ENGINE_AVX2)
    if (!g_scalar) {
        __m256 acc = _mm256_setzero_ps();
        int i = 0;
        for (; i + 8 <= n; i += 8)
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc);
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        lo = _mm_add_ps(lo, hi);
        lo = _mm_hadd_ps(lo, lo);
        lo = _mm_hadd_ps(lo, lo);
        float s = _mm_cvtss_f32(lo);
        for (; i < n; i++) s += a[i] * b[i];   /* scalar tail */
        return s;
    }
#endif
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

/* ggml-faithful precision mode: when SP_ENGINE_F16_ACT=1, matmul rounds each
 * activation to F16 before the dot, mimicking ggml downcasting src1 to F16 for an
 * F16 weight. Read once (see qwen3_forward). Default 0 = pure-f32 reference path,
 * which is what E_CPU_2 validates and what E_CPU_3+ quantize against. (The pure-f32
 * path is actually *closer* to ggml in KL terms; F16-act exists to demonstrate the
 * residual per-logit gap is QK-RMSNorm-amplified F16 precision, not a logic error.) */
static int g_f16_act = 0;

/* Frobenius weight path. SP_ENGINE_FROB selects how weight matmuls run (see
 * sp/frobenius_lift.h — per-row int8/int4 codes + fp32 scale):
 *   0 = pure f32 (default reference);
 *   1 = Q8 inline lift: accumulate q*x as float, scale once by row_scale/127;
 *   2 = Q8 dequant reference: lift each code back to f32 (q*row_scale/127) then
 *       the plain f32 dot. Modes 1/2 use identical Q8 weights and must agree to
 *       float-associativity (E_CPU_3's "identical to a reference fp32 matmul");
 *   3 = Q4 inline lift (E_CPU_7): symmetric 4-bit codes [-7,7] packed two-per-byte,
 *       scale once by row_scale/7; per-row calibration promotes high-error rows to
 *       Q8 (mixed precision). v1 calibration is weight-only per-row sensitivity —
 *       activation-based calibration is the Phase-4 refinement (roadmap §4.4/§7.5);
 *   4 = Q4 dequant reference: same per-row codes/promotion as mode 3, lifted to
 *       f32 then plain dot (the lift-faithfulness partner of mode 3).
 * The Q4 quant/pack/calibration primitives live in core/frobenius (sp_frob_q4_*)
 * so every backend shares one implementation; the engine supplies only the
 * per-row promotion policy (the SP_Q4_PROMOTE threshold). */
static int g_frob = 0;
static float g_q4_promote = 0.25f;   /* promote a row to Q8 if its Q4 round-trip */
static long  g_q4_promoted = 0;      /* rel-L2 error exceeds this (SP_Q4_PROMOTE) */
static long  g_q4_rows = 0;          /* total rows seen on the Q4 path (for reporting) */

/* bytes occupied by `n` contiguous elements of a ggml weight row. */
static size_t row_bytes(uint32_t type, int n) {
    switch (type) {
        case GGML_T_F32:  return (size_t)n * 4;
        case GGML_T_F16:  return (size_t)n * 2;
        case GGML_T_Q8_0: return (size_t)(n / 32) * 34;
        default:          return 0;
    }
}

/* Arena matmul: Y[t,j] = (sum_i code[j,i]*X[t,i]) * row_scale[j]/qmax, reading the
 * packed Q8/Q4 codes directly (the §4.8 inline-lift production path). Identical
 * arithmetic to the FROB=1 (Q8) / FROB=3 (Q4) GGUF path, so it is byte-identical
 * to them — only the weight bytes' home differs (arena vs re-quantized mapping). */
static int matmul_arena(const sp_arena_tensor *at, const float *X,
                        int n_tok, int in, int out, float *Y) {
    if (at->rows != out || at->cols != in) return 1;
    int8_t *unp = (int8_t *)malloc((size_t)in);   /* Q4 unpack scratch */
    if (!unp) return 1;
    for (int j = 0; j < out; j++) {
        const uint8_t *rc = at->codes + at->row_off[j];
        const int8_t *cp;
        float inv;
        if (at->row_prec[j] == 8) { cp = (const int8_t *)rc; inv = at->row_scale[j] / 127.0f; }
        else { sp_frob_q4_unpack(rc, in, unp); cp = unp; inv = at->row_scale[j] / 7.0f; }
        for (int t = 0; t < n_tok; t++) {
            const float *x = X + (size_t)t * in;
            float acc = 0.0f;
            for (int i = 0; i < in; i++) acc += (float)cp[i] * x[i];
            Y[(size_t)t * out + j] = acc * inv;
        }
    }
    free(unp);
    return 0;
}

/* Y[t,j] = sum_i W[i,j] * X[t,i]; W is the gguf tensor [in, out] (ne0=in). */
static int matmul(const qwen3_model *m, const gguf_tensor *W,
                  const float *X, int n_tok, int in, int out, float *Y) {
    if (m->arena) {                            /* packed-weight arena (§4.8) takes precedence */
        const sp_arena_tensor *at = sp_arena_find(m->arena, W->name);
        if (at) return matmul_arena(at, X, n_tok, in, out, Y);
        /* not arena-ized (e.g. token_embd in 1a): fall through to the GGUF path */
    }
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
    int8_t  *codes = NULL;          /* per-row int8/int4 codes (Q8: [-127,127], Q4: [-7,7]) */
    uint8_t *nib   = NULL;          /* Q4 nibble-packed storage (two codes per byte) */
    if (g_frob) {
        codes = (int8_t *)malloc((size_t)in);
        if (!codes) { free(wrow); free(xr); return 1; }
        if (g_frob >= 3) {
            nib = (uint8_t *)malloc((size_t)(in + 1) / 2);
            if (!nib) { free(wrow); free(xr); free(codes); return 1; }
        }
    }
    const int q4_path = (g_frob == 3 || g_frob == 4);
    for (int j = 0; j < out; j++) {
        if (sp_dequant_row(base + (size_t)j * rb, W->type, in, wrow)) {
            free(wrow); free(xr); free(codes); free(nib); return 1;
        }
        if (g_frob) {
            /* per-row Frobenius lift: quantize the f32 row once. */
            float s = sp_frob_row_scale(wrow, in);
            float inv;
            if (!q4_path) {                                   /* Q8 (modes 1/2) */
                for (int i = 0; i < in; i++) codes[i] = sp_frob_quant1(wrow[i], s);
                inv = s / (float)SP_FROB_QMAX;
            } else {                                          /* Q4 (modes 3/4) + calibration */
                g_q4_rows++;
                /* per-row weight-only sensitivity (core primitive) decides Q4 vs Q8 */
                if (sp_frob_q4_row_relerr(wrow, in) > g_q4_promote) {
                    for (int i = 0; i < in; i++) codes[i] = sp_frob_quant1(wrow[i], s);  /* promote -> Q8 */
                    inv = s / (float)SP_FROB_QMAX;
                    g_q4_promoted++;
                } else {
                    for (int i = 0; i < in; i++) codes[i] = sp_frob_quant1_q4(wrow[i], s);
                    sp_frob_q4_pack(codes, in, nib);          /* round-trip real 4-bit storage */
                    sp_frob_q4_unpack(nib, in, codes);
                    inv = s / 7.0f;
                }
            }
            if (g_frob == 2 || g_frob == 4)                   /* dequant reference: lift to f32 */
                for (int i = 0; i < in; i++) wrow[i] = (float)codes[i] * inv;
            for (int t = 0; t < n_tok; t++) {
                const float *x = X + (size_t)t * in;
                float acc = 0.0f;
                if (g_frob == 2 || g_frob == 4) {             /* plain f32 dot of lifted weights */
                    for (int i = 0; i < in; i++) acc += wrow[i] * x[i];
                    Y[(size_t)t * out + j] = acc;
                } else {                                      /* inline lift: scale once */
                    for (int i = 0; i < in; i++) acc += (float)codes[i] * x[i];
                    Y[(size_t)t * out + j] = acc * inv;
                }
            }
        } else {
            for (int t = 0; t < n_tok; t++)
                Y[(size_t)t * out + j] = dot_f32(wrow, X + (size_t)t * in, in);
        }
    }
    free(wrow);
    free(xr);
    free(codes);
    free(nib);
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

/* Read the runtime gate knobs once per forward/decode entry (all default OFF =
 * the pure-f32 E_CPU_2 reference path). Shared by the prefill and the decode loop
 * so both honor the same gates. */
static void read_env_knobs(void) {
    { const char *e = getenv("SP_ENGINE_F16_ACT");  g_f16_act  = (e && e[0] == '1'); }
    { const char *e = getenv("SP_ENGINE_FROB");     g_frob     = e ? atoi(e) : 0; }
    { const char *e = getenv("SP_Q4_PROMOTE");      if (e) g_q4_promote = (float)atof(e);
      g_q4_promoted = 0; g_q4_rows = 0; }
    { const char *e = getenv("SP_CPU_SCALAR");      g_scalar   = (e && e[0] == '1'); }
    { const char *e = getenv("SP_ENGINE_NTT_ATTN"); g_ntt_attn = (e && e[0] == '1'); }
    { const char *e = getenv("SP_KV_SPINOR");       g_kv_spinor = (e && e[0] == '1'); }
}

int qwen3_forward_ex(const qwen3_model *m, const int32_t *tokens, int n_tok,
                     float *logits, sp_kste_tree_t *kv_trees) {
    const qwen3_config *c = &m->cfg;
    const int E = (int)c->n_embd, FF = (int)c->n_ff, HD = (int)c->head_dim;
    const int NH = (int)c->n_head, NKV = (int)c->n_head_kv;
    const int QD = NH * HD;          /* q proj width  (2048) */
    const int KVD = NKV * HD;        /* kv proj width (1024) */
    const int group = NH / NKV;      /* q-heads per kv-head (2) */
    const int V = (int)c->n_vocab;
    const float eps = c->rms_eps, base = c->rope_freq_base;
    const float ascale = 1.0f / sqrtf((float)HD);

    read_env_knobs();

    int rc = 1;
    sp_pr_ctx *pr = NULL;          /* poly-ring context for NTT-attention (N=head_dim) */
    int32_t *qi = NULL, *ki = NULL;
    int32_t *kq = NULL;            /* int32 scratch for KSTE KV encoding */
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

    if (g_ntt_attn) {
        pr = sp_pr_init((uint32_t)HD);          /* head_dim must be in {128,256,512} */
        qi = (int32_t *)malloc((size_t)HD * sizeof(int32_t));
        ki = (int32_t *)malloc((size_t)HD * sizeof(int32_t));
        if (!pr || !qi || !ki) goto done;
    }
    if (kv_trees) {
        kq = (int32_t *)malloc((size_t)HD * sizeof(int32_t));
        if (!kq) goto done;
    }

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

        /* Inline VHT2+Spinor KV compression (E_CPU_8): store each cached K/V head
         * vector as Spinor block(s) and read back the lossy reconstruction. Applied
         * after QK-norm+RoPE so the cache holds position-finalized K (and post-proj
         * V), exactly what the persistent-KV decode path will store. */
        if (g_kv_spinor) {
            for (int t = 0; t < n_tok; t++)
                for (int h = 0; h < NKV; h++) {
                    kv_spinor_roundtrip(k + (size_t)t * KVD + (size_t)h * HD, HD);
                    kv_spinor_roundtrip(v + (size_t)t * KVD + (size_t)h * HD, HD);
                }
        }

        /* KSTE KV-cache overlay (E_CPU_6): encode each cached K head-vector to its
         * 64-byte signature. Deterministic int32 quantization -> byte-identical. */
        if (kv_trees) {
            for (int t = 0; t < n_tok; t++)
                for (int h = 0; h < NKV; h++) {
                    const float *kh = k + (size_t)t * KVD + (size_t)h * HD;
                    for (int i = 0; i < HD; i++)
                        kq[i] = (int32_t)lrintf(kh[i] * (float)SP_KSTE_KV_SCALE);
                    sp_kste_encode(kq, HD, &kv_trees[((size_t)L * n_tok + t) * NKV + h]);
                }
        }

        /* GQA causal attention */
        for (int t = 0; t < n_tok; t++) {
            for (int h = 0; h < NH; h++) {
                int kvh = h / group;
                const float *qh = q + (size_t)t * QD + (size_t)h * HD;
                if (g_ntt_attn)
                    for (int i = 0; i < HD; i++) qi[i] = (int32_t)lrintf(qh[i] * (float)SP_NTT_ATTN_SCALE);
                float maxs = -INFINITY;
                for (int s = 0; s <= t; s++) {
                    const float *kh = k + (size_t)s * KVD + (size_t)kvh * HD;
                    float d;
                    if (g_ntt_attn) {
                        for (int i = 0; i < HD; i++) ki[i] = (int32_t)lrintf(kh[i] * (float)SP_NTT_ATTN_SCALE);
                        int64_t ip = sp_pr_inner(pr, qi, ki);   /* exact <q_int,k_int> */
                        d = (float)((double)ip / (SP_NTT_ATTN_SCALE * SP_NTT_ATTN_SCALE)) * ascale;
                    } else {
                        float acc = 0.0f;
                        for (int i = 0; i < HD; i++) acc += qh[i] * kh[i];
                        d = acc * ascale;
                    }
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
    free(qi); free(ki); free(kq); sp_pr_free(pr);
    return rc;
}

int qwen3_forward(const qwen3_model *m, const int32_t *tokens, int n_tok, float *logits) {
    return qwen3_forward_ex(m, tokens, n_tok, logits, NULL);
}

/* Q4 calibration stats from the most recent forward on the Q4 path (modes 3/4):
 * how many weight rows were promoted to Q8 out of the total seen. Both 0 when
 * SP_ENGINE_FROB!=3,4. Lets E_CPU_7 report the mixed-precision rate. */
void qwen3_q4_stats(long *promoted, long *rows) {
    if (promoted) *promoted = g_q4_promoted;
    if (rows)     *rows     = g_q4_rows;
}

/* Persistent-KV O(n) greedy decode (GEN_KV). Same result as qwen3_generate but
 * each token is processed once: per-layer K/V are computed for the single new
 * token, stored post-RoPE into a position-indexed cache, and attention reads the
 * cached K/V for all earlier positions. The expensive weight matmuls run on one
 * token per step (O(n) total) instead of re-prefilling the whole prefix (O(n^2)).
 *
 * Honors the same gates as the prefill (SP_ENGINE_FROB, SP_CPU_SCALAR, SP_KV_SPINOR;
 * the cache stores Spinor-compressed K/V when SP_KV_SPINOR=1). The NTT-attention
 * path is prefill-only and not wired here. Greedy argmax must match qwen3_generate
 * up to the float-reassociation floor (different softmax-sum lengths) — GEN_KV
 * gates on argmax/sequence identity, not bit-equal logits. */
int qwen3_generate_kv(const qwen3_model *m, int32_t *seq, int n_prompt, int n_gen,
                      int eos_id) {
    if (!m || !seq || n_prompt <= 0 || n_gen < 0) return -1;
    read_env_knobs();
    const qwen3_config *c = &m->cfg;
    const int E = (int)c->n_embd, FF = (int)c->n_ff, HD = (int)c->head_dim;
    const int NH = (int)c->n_head, NKV = (int)c->n_head_kv;
    const int QD = NH * HD, KVD = NKV * HD, group = NH / NKV, V = (int)c->n_vocab;
    const float eps = c->rms_eps, base = c->rope_freq_base, ascale = 1.0f / sqrtf((float)HD);
    const int P = n_prompt + n_gen;

    int rc = -1, n = n_prompt;
    float *kc   = (float *)malloc((size_t)c->n_layers * P * KVD * sizeof(float)); /* K cache */
    float *vc   = (float *)malloc((size_t)c->n_layers * P * KVD * sizeof(float)); /* V cache */
    float *x    = (float *)malloc((size_t)E * sizeof(float));   /* single-token residual */
    float *nx   = (float *)malloc((size_t)E * sizeof(float));
    float *q    = (float *)malloc((size_t)QD * sizeof(float));
    float *knew = (float *)malloc((size_t)KVD * sizeof(float));
    float *vnew = (float *)malloc((size_t)KVD * sizeof(float));
    float *ao   = (float *)malloc((size_t)QD * sizeof(float));
    float *ap   = (float *)malloc((size_t)E * sizeof(float));
    float *gg   = (float *)malloc((size_t)FF * sizeof(float));
    float *up   = (float *)malloc((size_t)FF * sizeof(float));
    float *dn   = (float *)malloc((size_t)E * sizeof(float));
    float *sc   = (float *)malloc((size_t)P * sizeof(float));
    float *lg   = (float *)malloc((size_t)V * sizeof(float));
    if (!kc || !vc || !x || !nx || !q || !knew || !vnew || !ao || !ap || !gg || !up || !dn || !sc || !lg)
        goto done;

    const uint8_t *emb = (const uint8_t *)gguf_tensor_data(m->gguf, m->token_embd);
    size_t emb_rb = row_bytes(m->token_embd->type, E);
    if (!emb || emb_rb == 0) goto done;

    int produced = 0;
    for (int pos = 0; pos < P; pos++) {
        int tok = seq[pos];
        if (sp_dequant_row(emb + (size_t)tok * emb_rb, m->token_embd->type, E, x)) goto done;

        for (uint32_t L = 0; L < c->n_layers; L++) {
            const qwen3_layer *ly = &m->layers[L];
            rmsnorm(x, as_f32(m, ly->attn_norm), E, eps, nx);
            if (matmul(m, ly->attn_q, nx, 1, E, QD, q))   goto done;
            if (matmul(m, ly->attn_k, nx, 1, E, KVD, knew)) goto done;
            if (matmul(m, ly->attn_v, nx, 1, E, KVD, vnew)) goto done;

            const float *qn = as_f32(m, ly->attn_q_norm), *kn = as_f32(m, ly->attn_k_norm);
            for (int h = 0; h < NH;  h++) { float *qh = q    + (size_t)h * HD; rmsnorm_head(qh, qn, HD, eps); rope_neox(qh, HD, pos, base); }
            for (int h = 0; h < NKV; h++) { float *kh = knew + (size_t)h * HD; rmsnorm_head(kh, kn, HD, eps); rope_neox(kh, HD, pos, base); }
            if (g_kv_spinor)
                for (int h = 0; h < NKV; h++) { kv_spinor_roundtrip(knew + (size_t)h * HD, HD); kv_spinor_roundtrip(vnew + (size_t)h * HD, HD); }

            float *kslot = kc + ((size_t)L * P + pos) * KVD;   /* store position-finalized K/V */
            float *vslot = vc + ((size_t)L * P + pos) * KVD;
            memcpy(kslot, knew, (size_t)KVD * sizeof(float));
            memcpy(vslot, vnew, (size_t)KVD * sizeof(float));

            for (int h = 0; h < NH; h++) {                     /* attention over cached [0,pos] */
                int kvh = h / group;
                const float *qh = q + (size_t)h * HD;
                float maxs = -INFINITY;
                for (int s = 0; s <= pos; s++) {
                    const float *kh = kc + ((size_t)L * P + s) * KVD + (size_t)kvh * HD;
                    float acc = 0.0f;
                    for (int i = 0; i < HD; i++) acc += qh[i] * kh[i];
                    float d = acc * ascale; sc[s] = d; if (d > maxs) maxs = d;
                }
                float sum = 0.0f;
                for (int s = 0; s <= pos; s++) { sc[s] = expf(sc[s] - maxs); sum += sc[s]; }
                float inv = 1.0f / sum;
                float *out = ao + (size_t)h * HD;
                for (int i = 0; i < HD; i++) out[i] = 0.0f;
                for (int s = 0; s <= pos; s++) {
                    float w = sc[s] * inv;
                    const float *vh = vc + ((size_t)L * P + s) * KVD + (size_t)kvh * HD;
                    for (int i = 0; i < HD; i++) out[i] += w * vh[i];
                }
            }
            if (matmul(m, ly->attn_output, ao, 1, QD, E, ap)) goto done;
            for (int i = 0; i < E; i++) x[i] += ap[i];

            rmsnorm(x, as_f32(m, ly->ffn_norm), E, eps, nx);
            if (matmul(m, ly->ffn_gate, nx, 1, E, FF, gg)) goto done;
            if (matmul(m, ly->ffn_up,   nx, 1, E, FF, up)) goto done;
            for (int i = 0; i < FF; i++) { float gv = gg[i]; gg[i] = gv / (1.0f + expf(-gv)) * up[i]; }
            if (matmul(m, ly->ffn_down, gg, 1, FF, E, dn)) goto done;
            for (int i = 0; i < E; i++) x[i] += dn[i];
        }

        if (pos >= n_prompt - 1 && produced < n_gen) {         /* emit next token */
            rmsnorm(x, as_f32(m, m->output_norm), E, eps, nx);
            if (matmul(m, m->output, nx, 1, E, V, lg)) goto done;
            int amax = 0;
            for (int j = 1; j < V; j++) if (lg[j] > lg[amax]) amax = j;
            seq[n_prompt + produced] = amax;
            produced++; n = n_prompt + produced;
            if ((eos_id >= 0 && amax == eos_id) || produced >= n_gen) break;
        }
    }
    rc = n;
done:
    free(kc); free(vc); free(x); free(nx); free(q); free(knew); free(vnew);
    free(ao); free(ap); free(gg); free(up); free(dn); free(sc); free(lg);
    return rc;
}
