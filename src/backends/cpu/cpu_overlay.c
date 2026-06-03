/* kernels.c — backend-agnostic CPU forward-pass kernels (see kernels.h). Moved
 * verbatim from forward.c; the Qwen3 regression guards behavior-preservation. */
#define _CRT_SECURE_NO_WARNINGS
#include "sp_engine/kernels.h"
#include "sp_engine/arena.h"
#include "sp/frobenius_lift.h"
#if defined(SP_ENGINE_AVX512)
#include "sp_engine/avx512.h"   /* sp_avx512_vnni_matvec + g_avx512_caps (VNNI int8 path) */
#endif

#include <stdlib.h>
#include <string.h>
#include <math.h>

#if defined(SP_ENGINE_AVX2) || defined(SP_ENGINE_AVX512)
#include <immintrin.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

/* ── runtime gate knobs honored by these kernels (default OFF = pure-f32) ── */
static int   g_scalar  = 0;   /* SP_CPU_SCALAR=1 forces the scalar reduction */
static int   g_f16_act = 0;   /* SP_ENGINE_F16_ACT=1 rounds matmul activations to F16 (ggml-faithful) */
static int   g_fp16    = 0;   /* SP_ENGINE_FP16=1 fp16 working precision (2-L1.FP16, E_FP16_1):
                               * f16 matmul activations + f16 KV/attention inputs, f32 accumulator,
                               * f32 residual — matches the llama.cpp f16 scheme the oracle uses. */
static int   g_frob    = 0;   /* SP_ENGINE_FROB: 0 f32, 1/2 Q8 inline/dequant, 3/4 Q4 inline/dequant */
static float g_q4_promote = 0.25f;   /* promote a Q4 row to Q8 if its round-trip rel-L2 exceeds this */
static long  g_q4_promoted = 0;
static long  g_q4_rows = 0;
static int   g_vnni    = 0;   /* SP_VNNI=1: AVX-512 VNNI int8×int8 matmul_arena path (dyn act-quant) */
static int   g_avx512dot = 0; /* SP_CPU_AVX512DOT=1: 16-wide AVX-512 int8×f32 dot (vs AVX2's 8-wide) */

/* round one f32 to the nearest IEEE binary16 value and back (fp16 working precision). */
static inline float r16(float v) { return sp_f16_to_f32(sp_f32_to_f16(v)); }

/* Cap OpenMP threads to the physical-core count by default. The decode matmul is
 * memory-bound past ~physical cores; the OMP default of ALL logical threads
 * (2× on HT) oversubscribes and runs ~1.5× SLOWER than physical-core count
 * (measured: 16 logical = 16.5 tok/s, 5-6 threads = 25.8 on an 8-physical box).
 * Override with OMP_NUM_THREADS (wins) or SP_OMP_THREADS. One-time. */
static void sp_set_thread_default(void) {
#ifdef _OPENMP
    static int done = 0;
    if (done) return; done = 1;
    if (getenv("OMP_NUM_THREADS")) return;            /* explicit OMP override wins */
    const char *e = getenv("SP_OMP_THREADS");
    int n = (e && atoi(e) > 0) ? atoi(e)
                               : (omp_get_num_procs() > 2 ? omp_get_num_procs() / 2 : omp_get_num_procs());
    if (n < 1) n = 1;
    omp_set_num_threads(n);
#endif
}

void sp_kernels_read_env(void) {
    sp_set_thread_default();
    { const char *e = getenv("SP_ENGINE_F16_ACT");  g_f16_act  = (e && e[0] == '1'); }
    { const char *e = getenv("SP_ENGINE_FP16");     g_fp16     = (e && e[0] == '1'); }
    { const char *e = getenv("SP_ENGINE_FROB");     g_frob     = e ? atoi(e) : 0; }
    { const char *e = getenv("SP_Q4_PROMOTE");      if (e) g_q4_promote = (float)atof(e);
      g_q4_promoted = 0; g_q4_rows = 0; }
    { const char *e = getenv("SP_CPU_SCALAR");      g_scalar   = (e && e[0] == '1'); }
    { const char *e = getenv("SP_VNNI");            g_vnni     = (e && e[0] == '1'); }
    { const char *e = getenv("SP_CPU_AVX512DOT");   g_avx512dot = (e && e[0] == '1'); }
#if defined(SP_ENGINE_AVX512)
    if (g_vnni) sp_avx512_init();   /* populate g_avx512_caps.has_vnni before dispatch */
#endif
}

void qwen3_q4_stats(long *promoted, long *rows) {
    if (promoted) *promoted = g_q4_promoted;
    if (rows)     *rows     = g_q4_rows;
}

float dot_f32(const float *a, const float *b, int n) {
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

/* int8-weight × f32-activation dot (WIRE-CPU-V2 step: AVX2 widen-int8→f32 + FMA).
 * Numerics-preserved vs the scalar `(float)w[i]*x[i]` up to FMA reassociation (ULP);
 * the scalar branch (SP_CPU_SCALAR=1 or no-AVX2) is the bit-exact oracle. Pure fn,
 * thread-safe inside the OpenMP matmul_arena region. (VNNI int8×int8 is the next
 * step — needs int8 activations + an accuracy gate; see sp_avx512_vnni_matvec.) */
static float dot_i8_f32(const int8_t *w, const float *x, int n) {
#if defined(SP_ENGINE_AVX2)
    if (!g_scalar) {
        __m256 acc = _mm256_setzero_ps();
        int i = 0;
        for (; i + 8 <= n; i += 8) {
            __m128i w8  = _mm_loadl_epi64((const __m128i *)(w + i));  /* 8 int8 */
            __m256  wf  = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w8)); /* sign-ext → 8 f32 */
            acc = _mm256_fmadd_ps(wf, _mm256_loadu_ps(x + i), acc);
        }
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        lo = _mm_add_ps(lo, hi);
        lo = _mm_hadd_ps(lo, lo);
        lo = _mm_hadd_ps(lo, lo);
        float s = _mm_cvtss_f32(lo);
        for (; i < n; i++) s += (float)w[i] * x[i];   /* scalar tail */
        return s;
    }
#endif
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += (float)w[i] * x[i];
    return s;
}

#if defined(SP_ENGINE_AVX512)
/* AVX-512 widen of dot_i8_f32: 16 int8 -> 16 f32 per step (vs AVX2's 8). Same f32
 * accumulate, parity-safe up to FMA reassociation (top-1 gate). Gated by
 * SP_CPU_AVX512DOT — a measurement of whether wider SIMD helps the int8×f32 decode
 * dot (Stage 0 predicts marginal: the loop is bandwidth/convert-bound, not ALU). */
SP_TARGET("avx512f,fma")
static float dot_i8_f32_avx512(const int8_t *w, const float *x, int n) {
    __m512 acc = _mm512_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m128i w16 = _mm_loadu_si128((const __m128i *)(w + i));     /* 16 int8 */
        __m512  wf  = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(w16)); /* sign-ext 16 -> f32 */
        acc = _mm512_fmadd_ps(wf, _mm512_loadu_ps(x + i), acc);
    }
    float s = _mm512_reduce_add_ps(acc);
    for (; i < n; i++) s += (float)w[i] * x[i];   /* scalar tail */
    return s;
}
#endif

/* bytes occupied by `n` contiguous elements of a ggml weight row. */
static size_t row_bytes(uint32_t type, int n) {
    switch (type) {
        case GGML_T_F32:  return (size_t)n * 4;
        case GGML_T_F16:  return (size_t)n * 2;
        case GGML_T_Q8_0: return (size_t)(n / 32) * 34;
        default:          return 0;
    }
}

#if defined(SP_ENGINE_AVX512)
/* Per-tensor VNNI side data, computed once (weights are const across tokens),
 * keyed by the codes pointer: per-row bias = 128*sum(int8 codes) (the dpbusd
 * zero-shift correction) and per-row scale = row_scale/127. `all_q8` caches
 * eligibility (every row Q8 + contiguous stride=in). */
typedef struct { const void *key; int all_q8; int32_t *bias; float *rs; } vnni_cache_t;
#define VNNI_CACHE_N 1024
static vnni_cache_t g_vnni_cache[VNNI_CACHE_N];
static int g_vnni_cache_used = 0;
static uint8_t *g_act_u8 = NULL; static int g_act_cap = 0;   /* act-quant scratch (single-thread call site) */

static const vnni_cache_t *vnni_get(const sp_frob_packed_tensor *pt, int in, int out) {
    for (int i = 0; i < g_vnni_cache_used; i++)
        if (g_vnni_cache[i].key == (const void *)pt->codes) return &g_vnni_cache[i];
    if (g_vnni_cache_used >= VNNI_CACHE_N) return NULL;
    int all_q8 = 1;
    for (int j = 0; j < out; j++)
        if (pt->row_prec[j] != 8 || (size_t)pt->row_off[j] != (size_t)j * (size_t)in) { all_q8 = 0; break; }
    vnni_cache_t *c = &g_vnni_cache[g_vnni_cache_used++];
    c->key = (const void *)pt->codes; c->all_q8 = all_q8; c->bias = NULL; c->rs = NULL;
    if (all_q8) {
        c->bias = (int32_t *)malloc((size_t)out * sizeof(int32_t));
        c->rs   = (float   *)malloc((size_t)out * sizeof(float));
        if (!c->bias || !c->rs) { free(c->bias); free(c->rs); c->bias = NULL; c->rs = NULL; c->all_q8 = 0; }
        else {
            const int8_t *codes = (const int8_t *)pt->codes;
            for (int j = 0; j < out; j++) {
                const int8_t *w = codes + (size_t)j * (size_t)in;
                int s = 0; for (int k = 0; k < in; k++) s += w[k];
                c->bias[j] = 128 * s;
                c->rs[j]   = pt->row_scale[j] / 127.0f;
            }
        }
    }
    return c;
}
#endif

/* Arena matmul: inline-lift the packed Q8/Q4 codes (the §4.8 production path).
 * WIRE-CPU-V2: the per-output-row (j) loop is OpenMP-parallel. Each Y[j] is an
 * independent single-threaded dot, so the result is BIT-IDENTICAL to the serial
 * path for any thread count (no cross-thread reduction). The Q4 unpack scratch is
 * per-thread (allocated inside the parallel region). Without OpenMP the pragmas
 * are ignored and this is the original serial loop. */
static int matmul_arena(const sp_arena_tensor *at, const float *X,
                        int n_tok, int in, int out, float *Y) {
    const sp_frob_packed_tensor *pt = &at->pt;
    if (pt->rows != out || pt->cols != in) return 1;
#if defined(SP_ENGINE_AVX512)
    /* VNNI int8×int8 path (SP_VNNI=1). Dynamic per-vector activation quant → dpbusd.
     * NOT bit-exact to the f32-activation path (lossy act-quant) → gated by a top-1/PPL
     * gate, not a byte-match. Falls through to the AVX2/scalar path if ineligible. */
    if (g_vnni && !g_scalar && g_avx512_caps.has_vnni && (in % 64 == 0)) {
        const vnni_cache_t *vc = vnni_get(pt, in, out);
        if (vc && vc->all_q8) {
            if (g_act_cap < in) { free(g_act_u8); g_act_u8 = (uint8_t *)malloc((size_t)in); g_act_cap = g_act_u8 ? in : 0; }
            if (g_act_u8) {
                const int8_t *codes = (const int8_t *)pt->codes;
                for (int t = 0; t < n_tok; t++) {
                    const float *x = X + (size_t)t * in;
                    float ma = 0.0f; for (int i = 0; i < in; i++) { float a = fabsf(x[i]); if (a > ma) ma = a; }
                    float *yt = Y + (size_t)t * out;
                    if (ma <= 0.0f) { for (int i = 0; i < out; i++) yt[i] = 0.0f; continue; }
                    float as = ma / 127.0f, invas = 127.0f / ma;
                    for (int i = 0; i < in; i++) {
                        int q = (int)lrintf(x[i] * invas);
                        if (q > 127) q = 127; else if (q < -127) q = -127;
                        g_act_u8[i] = (uint8_t)(q + 128);
                    }
                    sp_avx512_vnni_matvec(codes, g_act_u8, vc->rs, vc->bias, out, in, yt);
                    for (int i = 0; i < out; i++) yt[i] *= as;
                }
                return 0;
            }
        }
    }
#endif
    int rc = 0;
    #pragma omp parallel
    {
        int8_t *unp = (int8_t *)malloc((size_t)in);   /* per-thread Q4 unpack scratch */
        if (!unp) { rc = 1; }
        else {
            int j;   /* MSVC OpenMP 2.0: loop var must be declared outside the for-init */
            #pragma omp for
            for (j = 0; j < out; j++) {
                const uint8_t *rcw = pt->codes + pt->row_off[j];
                const int8_t *cp;
                float inv;
                if (pt->row_prec[j] == 8) { cp = (const int8_t *)rcw; inv = pt->row_scale[j] / 127.0f; }
                else { sp_frob_q4_unpack(rcw, in, unp); cp = unp; inv = pt->row_scale[j] / 7.0f; }
                for (int t = 0; t < n_tok; t++) {
                    const float *x = X + (size_t)t * in;
#if defined(SP_ENGINE_AVX512)
                    Y[(size_t)t * out + j] = (g_avx512dot && !g_scalar ? dot_i8_f32_avx512(cp, x, in) : dot_i8_f32(cp, x, in)) * inv;
#else
                    Y[(size_t)t * out + j] = dot_i8_f32(cp, x, in) * inv;
#endif
                }
            }
            free(unp);
        }
    }
    return rc;
}

int matmul(const qwen3_model *m, const gguf_tensor *W,
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
     * up front (the same rounded x is reused across all `out` weight rows).
     * fp16 working precision (SP_ENGINE_FP16) does the same activation downcast. */
    float *xr = NULL;
    if (g_f16_act || g_fp16) {
        xr = (float *)malloc((size_t)n_tok * in * sizeof(float));
        if (!xr) { free(wrow); return 1; }
        for (size_t i = 0; i < (size_t)n_tok * in; i++)
            xr[i] = sp_f16_to_f32(sp_f32_to_f16(X[i]));
        X = xr;
    }
    int8_t  *codes = NULL;          /* per-row int8/int4 codes */
    uint8_t *nib   = NULL;          /* Q4 nibble-packed storage */
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
            float s = sp_frob_row_scale(wrow, in);
            float inv;
            if (!q4_path) {                                   /* Q8 (modes 1/2) */
                for (int i = 0; i < in; i++) codes[i] = sp_frob_quant1(wrow[i], s);
                inv = s / (float)SP_FROB_QMAX;
            } else {                                          /* Q4 (modes 3/4) + calibration */
                g_q4_rows++;
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

void rmsnorm(const float *x, const float *w, int n, float eps, float *out) {
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    float scale = 1.0f / sqrtf((float)(ss / n) + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * scale * w[i];
}

void rmsnorm_head(float *v, const float *w, int d, float eps) {
    double ss = 0.0;
    for (int i = 0; i < d; i++) ss += (double)v[i] * v[i];
    float scale = 1.0f / sqrtf((float)(ss / d) + eps);
    for (int i = 0; i < d; i++) v[i] = v[i] * scale * w[i];
}

void rope_neox(float *v, int d, int p, float base) {
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

void kernels_attn_head(const float *qh, const float *KC, const float *VC,
                       int pos, int KVD, int kvh, int HD, float ascale, int win,
                       float *sc, float *out) {
    int s0 = (win >= 0 && pos - win + 1 > 0) ? pos - win + 1 : 0;
    float maxs = -INFINITY;
    for (int s = s0; s <= pos; s++) {
        const float *kh = KC + (size_t)s * KVD + (size_t)kvh * HD;
        float acc = 0.0f;
        /* fp16 working precision (SP_ENGINE_FP16): fp16 Q/K into the dot, f32 acc. */
        if (g_fp16) for (int i = 0; i < HD; i++) acc += r16(qh[i]) * r16(kh[i]);
        else        for (int i = 0; i < HD; i++) acc += qh[i] * kh[i];   /* scalar acc: matches E_CPU_2 */
        float d = acc * ascale;
        sc[s] = d;
        if (d > maxs) maxs = d;
    }
    float sum = 0.0f;
    for (int s = s0; s <= pos; s++) { sc[s] = expf(sc[s] - maxs); sum += sc[s]; }
    float inv = 1.0f / sum;
    for (int i = 0; i < HD; i++) out[i] = 0.0f;
    for (int s = s0; s <= pos; s++) {
        float w = sc[s] * inv;
        const float *vh = VC + (size_t)s * KVD + (size_t)kvh * HD;
        if (g_fp16) for (int i = 0; i < HD; i++) out[i] += w * r16(vh[i]);   /* fp16 V */
        else        for (int i = 0; i < HD; i++) out[i] += w * vh[i];
    }
}

const float *as_f32(const qwen3_model *m, const gguf_tensor *t) {
    if (m->released) {
        for (int i = 0; i < m->n_norm; i++) if (m->norm_src[i] == t) return m->norm_buf[i];
        return NULL;   /* every norm the forward reads was copied in release */
    }
    return (const float *)gguf_tensor_data(m->gguf, t);
}

int embed_row(const qwen3_model *m, int32_t tok, int E, float *dst) {
    const sp_arena_tensor *at = m->arena ? sp_arena_find(m->arena, m->token_embd->name) : NULL;
    if (at) return sp_arena_dequant_row(at, (int)tok, dst);
    const uint8_t *emb = (const uint8_t *)gguf_tensor_data(m->gguf, m->token_embd);
    size_t rb = row_bytes(m->token_embd->type, E);
    if (!emb || rb == 0) return 1;
    return sp_dequant_row(emb + (size_t)tok * rb, m->token_embd->type, E, dst);
}
