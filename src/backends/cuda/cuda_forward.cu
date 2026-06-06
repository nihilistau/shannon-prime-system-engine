/* cuda_forward.cu — Gemma3 + Qwen3 forward pass on CUDA (Phase 2-CU).
 *   CU.1 f32 + CU.2/CU.4 Q8/Q4 arena (gemma3); CU.5 qwen3_forward_cuda (E_CU_1..4);
 *   CU.6 NTT-attention (E_CU_5); CU.7 KSTE-KV (E_CU_6).
 *
 * Mirrors the CPU forward passes op-for-op (gemma3_forward in src/forward/gemma3.c,
 * qwen3_forward in src/forward/forward.c) so the CUDA-vs-CPU output gate (§8.3,
 * <=1e-3 rel) and T_FRO_4 hold. The two forwards share the device weight cache,
 * the kernels, and the GEMM helper (kernels must stay in one TU). cuBLAS SGEMM for
 * the 8 matmuls (q/k/v/o, gate/up/down, tied head); hand-written kernels for the
 * rmsnorm / per-head QK-norm / NEOX RoPE / GQA windowed softmax / GeGLU / embed
 * scale. Single stream + CUBLAS_DEFAULT_MATH (sm_75 has no TF32 SGEMM, so f32
 * stays true f32) => deterministic, the T_FRO_4 gate-(a) mode.
 *
 * Weight residency: f32 weights are dequantized host-side (GGUF f16/f32/Q8_0 ->
 * f32) and uploaded once; packed-arena weights (SP_ARENA=q8|q4) upload the compact
 * math-core layout (sp_frob_packed_tensor: codes/row_off/row_scale/row_prec) and
 * are decoded ON DEVICE by k_dequant_arena into a reused f32 scratch right before
 * their SGEMM — the §4.8 decode-on-demand path, byte-equivalent to the CPU
 * matmul_arena lift (q*scale/qmax). Cached by model pointer; sp_cuda_model_release
 * frees on qwen3_free.
 *
 * GEMM mapping: ggml weight W is [ne0=in, ne1=out] with y[j]=sum_i W[i+j*in]*x[i],
 * i.e. W is row-major [out x in]. For row-major X [n_tok x in] / Y [n_tok x out],
 * Y = X * W^T. In cuBLAS (column-major): Yc[out x n_tok] = Wc^T * Xc with Wc=
 * [in x out] (lda=in), Xc=[in x n_tok] (ldb=in), Yc (ldc=out):
 *   cublasSgemm(h, OP_T, OP_N, out, n_tok, in, 1, dW, in, dX, in, 0, dY, out).
 */
#include "sp_engine/cuda_backend.h"
#include "sp_engine/kernels.h"   /* as_f32 */
#include "sp_engine/arena.h"     /* sp_arena_find / sp_arena_dequant_row */
#include "sp_engine/gguf.h"
#include "sp/frobenius_lift.h"   /* sp_frob_packed_tensor */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

extern "C" void sp_set_error(const char *msg);

/* ── error plumbing → SP_ECUDA / sp_last_error ── */
static int fail_cuda(cudaError_t e, const char *where) {
    char b[512];
    std::snprintf(b, sizeof(b), "CUDA: %s: %s (%d)", where, cudaGetErrorString(e), (int)e);
    sp_set_error(b);
    return 1;
}
static int fail_cublas(cublasStatus_t s, const char *where) {
    char b[512];
    std::snprintf(b, sizeof(b), "cuBLAS: %s: status %d", where, (int)s);
    sp_set_error(b);
    return 1;
}
#define CB(call, where) do { cublasStatus_t _s = (call); if (_s != CUBLAS_STATUS_SUCCESS) return fail_cublas(_s, where); } while (0)

/* fp16 working precision (2-L1.FP16, SP_ENGINE_FP16): round a device f32 buffer to
 * IEEE binary16 and back — the CUDA mirror of the CPU r16() (cpu_overlay.c). Weights
 * are already f16-valued (f16 GGUF dequant) and cuBLAS SGEMM keeps the f32 accumulator,
 * so rounding the activation + Q/K/V operands reproduces the same f16-weights ×
 * f16-activations → f32 scheme as the CPU path and the llama.cpp f16 oracle. */
__global__ void k_round_f16(float *x, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = __half2float(__float2half_rn(x[i]));
}

/* on-disk bytes of `n` contiguous elements of a ggml weight row (matches CPU). */
static size_t row_bytes(uint32_t type, int n) {
    switch (type) {
        case GGML_T_F32:  return (size_t)n * 4;
        case GGML_T_F16:  return (size_t)n * 2;
        case GGML_T_Q8_0: return (size_t)(n / 32) * 34;
        default:          return 0;
    }
}

/* ════════════════════════ kernels ════════════════════════ */

__global__ void k_embed_scale(const float *embd, const int *toks, int n_tok, int E,
                              float scale, float *x) {
    int t = blockIdx.x;
    int i = blockIdx.y * blockDim.x + threadIdx.x;
    if (t < n_tok && i < E)
        x[(size_t)t * E + i] = embd[(size_t)toks[t] * E + i] * scale;
}

/* out[row] = rmsnorm(x[row]) * w over n elems. One block/row; sum_sq in f64
 * (matches CPU). scale = 1/sqrtf((float)(sum_sq/n) + eps). */
__global__ void k_rmsnorm(const float *x, const float *w, int n, float eps, float *out) {
    int row = blockIdx.x;
    const float *xr = x + (size_t)row * n;
    float *outr = out + (size_t)row * n;
    __shared__ double sh[256];
    double s = 0.0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) { double v = xr[i]; s += v * v; }
    sh[threadIdx.x] = s;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (threadIdx.x < o) sh[threadIdx.x] += sh[threadIdx.x + o];
        __syncthreads();
    }
    float ms = (float)(sh[0] / (double)n);
    float scale = 1.0f / sqrtf(ms + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) outr[i] = xr[i] * scale * w[i];
}

/* in-place per-head RMSNorm over head_dim d; one block/(token,head), blockDim=d. */
__global__ void k_rmsnorm_head(float *base, const float *w, int n_heads, int d,
                               int rowstride, float eps) {
    int b = blockIdx.x, t = b / n_heads, h = b % n_heads;
    float *v = base + (size_t)t * rowstride + (size_t)h * d;
    __shared__ double sh[256];
    double s = 0.0;
    for (int i = threadIdx.x; i < d; i += blockDim.x) { double x = v[i]; s += x * x; }
    sh[threadIdx.x] = s;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (threadIdx.x < o) sh[threadIdx.x] += sh[threadIdx.x + o];
        __syncthreads();
    }
    float ms = (float)(sh[0] / (double)d);
    float scale = 1.0f / sqrtf(ms + eps);
    for (int i = threadIdx.x; i < d; i += blockDim.x) v[i] = v[i] * scale * w[i];
}

/* ETA.1 (Gemma4): WEIGHTLESS per-head RMSNorm — the Gemma4 V-norm (no learned
 * weight, no RoPE; gemma4.c g4_rmsnorm_noweight). Identical to k_rmsnorm_head
 * minus the `* w[i]`; f64 sum-of-squares matches the CPU reference precision. */
__global__ void k_rmsnorm_head_noweight(float *base, int n_heads, int d,
                                        int rowstride, float eps) {
    int b = blockIdx.x, t = b / n_heads, h = b % n_heads;
    float *v = base + (size_t)t * rowstride + (size_t)h * d;
    __shared__ double sh[256];
    double s = 0.0;
    for (int i = threadIdx.x; i < d; i += blockDim.x) { double x = v[i]; s += x * x; }
    sh[threadIdx.x] = s;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (threadIdx.x < o) sh[threadIdx.x] += sh[threadIdx.x + o];
        __syncthreads();
    }
    float ms = (float)(sh[0] / (double)d);
    float scale = 1.0f / sqrtf(ms + eps);
    for (int i = threadIdx.x; i < d; i += blockDim.x) v[i] = v[i] * scale;
}

/* ETA.2 (Gemma4): NEOX RoPE WITH the proportional freq-factor table — mirrors
 * sp_rope_neox_freqs: freq = base^(-2i/d) / ff[i] (DIVIDE by the factor).
 * Gemma4 GLOBAL layers only (SWA layers use plain k_rope, base 1e4, ff=NULL). */
__global__ void k_rope_freqs(float *base, int n_heads, int d, int rowstride,
                             float rbase, const float *ff) {
    int b = blockIdx.x, t = b / n_heads, h = b % n_heads, i = threadIdx.x, half = d / 2;
    if (i < half) {
        float *v = base + (size_t)t * rowstride + (size_t)h * d;
        float freq = powf(rbase, -2.0f * (float)i / (float)d) / ff[i];
        float th = (float)t * freq, c = cosf(th), s = sinf(th);
        float a = v[i], bb = v[i + half];
        v[i] = a * c - bb * s;
        v[i + half] = a * s + bb * c;
    }
}

/* ETA.4 (Gemma4 AltUp): the project_per_layer_inputs PRECOMPUTE fusion — runs
 * ONCE on the post-embed stream, before layer 0; the result persists in VRAM
 * across the whole traversal. Mirrors gemma4.c lines 121-142 exactly:
 *   row(t,L)[i] = ( proj(t,L)[i]·(1/√E) · inv · pn[i] + ple(t,L)[i]·√PL ) · (1/√2)
 * where inv = 1/sqrt( mean( (proj·1/√E)² ) + eps ) over the PL row (f64 sum,
 * the reference precision; ss accumulates AFTER the proj_scale, as the CPU does).
 * proj = per_layer_model_proj · x  [n_tok × NL*PL], ple = the host-gathered
 * per-token rows of per_layer_token_embd [n_tok × NL*PL]. One block per (t,L). */
__global__ void k_altup_ipl(float *proj, const float *ple, const float *pn,
                            int PL, float proj_scale, float ple_scale,
                            float in_scale, float eps) {
    float *row = proj + (size_t)blockIdx.x * PL;          /* (t,L) row */
    const float *pl = ple + (size_t)blockIdx.x * PL;
    __shared__ double sh[256];
    double s = 0.0;
    for (int i = threadIdx.x; i < PL; i += blockDim.x) {
        float v = row[i] * proj_scale;
        row[i] = v;                                        /* keep the scaled value (CPU order) */
        s += (double)v * (double)v;
    }
    sh[threadIdx.x] = s;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (threadIdx.x < o) sh[threadIdx.x] += sh[threadIdx.x + o];
        __syncthreads();
    }
    float inv = 1.0f / sqrtf((float)(sh[0] / (double)PL) + eps);
    for (int i = threadIdx.x; i < PL; i += blockDim.x)
        row[i] = (row[i] * inv * pn[i] + pl[i] * ple_scale) * in_scale;
}

/* ETA.4: the per-layer AltUp gate — pg(t)[i] = gelu(pg(t)[i]) · ipl(t,L)[i].
 * gelu = the tanh approximation (gemma4.c g4_gelu / k_gelu_mul's formula).
 * ipl rows for fixed L are strided NL*PL apart across tokens. */
__global__ void k_altup_gate(float *pg, const float *ipl, int L, int NL, int PL, int n_tok) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_tok * PL) return;
    int t = idx / PL, i = idx - t * PL;
    float v = pg[idx];
    const float k = 0.7978845608028654f;
    float th = tanhf(k * (v + 0.044715f * v * v * v));
    pg[idx] = 0.5f * v * (1.0f + th) * ipl[((size_t)t * NL + L) * PL + i];
}

/* ETA.4: scalar buffer scale by a DEVICE 1-float (the per-layer out_scale) —
 * no host sync, graph/capture-friendly. */
__global__ void k_scale_by_dev(float *x, size_t n, const float *s) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= s[0];
}

/* ETA.5 (Gemma4 decode): NEOX RoPE WITH freq factors at ABSOLUTE position p0
 * (single token, grid = n_heads). The decode twin of k_rope_freqs. */
__global__ void k_rope_freqs_at(float *base, int n_heads, int d, float rbase,
                                const float *ff, int p0) {
    int h = blockIdx.x, i = threadIdx.x, half = d / 2;
    if (i < half) {
        float *v = base + (size_t)h * d;
        float freq = powf(rbase, -2.0f * (float)i / (float)d) / ff[i];
        float th = (float)p0 * freq, c = cosf(th), s = sinf(th);
        float a = v[i], bb = v[i + half];
        v[i] = a * c - bb * s;
        v[i + half] = a * s + bb * c;
    }
}

/* ETA.5 (Gemma4 decode): single-query GQA attention over a cached span with an
 * optional SLIDING WINDOW — the decode twin of the prefill k_attn's windowing
 * (s0 = max(0, pos-win+1), matching sp_attn_head). One block per query head;
 * the cache is the OWNER's jagged buffer [ctx x KVD]. ascale is a parameter
 * (Gemma4 = 1.0). Shared = ctx floats. */
__global__ void k_attn_decode_win(const float *q, const float *Kc, const float *Vc,
                                  int ctx, int KVD, int HD, int group, float ascale,
                                  int win, float *ao) {
    extern __shared__ float sc[];
    int h = blockIdx.x, kvh = h / group;
    int pos = ctx - 1;
    int s0 = (win >= 0 && pos - win + 1 > 0) ? pos - win + 1 : 0;
    const float *qh = q + (size_t)h * HD;
    for (int s = s0 + threadIdx.x; s < ctx; s += blockDim.x) {
        const float *kh = Kc + (size_t)s * KVD + (size_t)kvh * HD;
        float acc = 0.0f;
        for (int i = 0; i < HD; i++) acc += qh[i] * kh[i];
        sc[s] = acc * ascale;
    }
    __syncthreads();
    __shared__ float g_sum;
    if (threadIdx.x == 0) {
        float mx = sc[s0];
        for (int s = s0 + 1; s < ctx; s++) if (sc[s] > mx) mx = sc[s];
        float sum = 0.0f;
        for (int s = s0; s < ctx; s++) { float e = expf(sc[s] - mx); sc[s] = e; sum += e; }
        g_sum = sum;
    }
    __syncthreads();
    float inv = 1.0f / g_sum;
    for (int i = threadIdx.x; i < HD; i += blockDim.x) {
        float acc = 0.0f;
        for (int s = s0; s < ctx; s++)
            acc += sc[s] * Vc[(size_t)s * KVD + (size_t)kvh * HD + i];
        ao[(size_t)h * HD + i] = acc * inv;
    }
}

/* ETA.2 (Gemma4): final-logit softcap, z = tanh(z/cap)*cap (gemma4.c). Applied
 * to the LM-head logits ONLY — gemma4 has NO attention-score cap (that was
 * Gemma2; the oracle runs attention at scale 1.0, uncapped). */
__global__ void k_softcap(float *z, size_t n, float cap) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) z[i] = tanhf(z[i] / cap) * cap;
}

/* NEOX RoPE on each (token,head) vector at position p=t; blockDim=d/2. */
__global__ void k_rope(float *base, int n_heads, int d, int rowstride, float rbase) {
    int b = blockIdx.x, t = b / n_heads, h = b % n_heads, i = threadIdx.x, half = d / 2;
    if (i < half) {
        float *v = base + (size_t)t * rowstride + (size_t)h * d;
        float freq = powf(rbase, -2.0f * (float)i / (float)d);
        float th = (float)t * freq, c = cosf(th), s = sinf(th);
        float a = v[i], bb = v[i + half];
        v[i] = a * c - bb * s;
        v[i + half] = a * s + bb * c;
    }
}

/* GQA causal/windowed softmax. One block per (token t, query head h). Dynamic
 * shared = n_tok floats (scores). blockDim covers max(n_tok, HD). Scalar f32. */
__global__ void k_attn(const float *Q, const float *K, const float *V,
                       int n_tok, int QD, int KVD, int HD, int group,
                       float ascale, int win, float *AO) {
    extern __shared__ float sc[];
    int n_heads = QD / HD;
    int b = blockIdx.x, t = b / n_heads, h = b % n_heads, kvh = h / group;
    const float *qh = Q + (size_t)t * QD + (size_t)h * HD;
    int s0 = (win >= 0 && t - win + 1 > 0) ? t - win + 1 : 0;

    for (int s = s0 + threadIdx.x; s <= t; s += blockDim.x) {
        const float *kh = K + (size_t)s * KVD + (size_t)kvh * HD;
        float acc = 0.0f;
        for (int i = 0; i < HD; i++) acc += qh[i] * kh[i];
        sc[s] = acc * ascale;
    }
    __syncthreads();

    __shared__ float g_sum;
    if (threadIdx.x == 0) {
        float m = sc[s0];
        for (int s = s0 + 1; s <= t; s++) if (sc[s] > m) m = sc[s];
        float sum = 0.0f;
        for (int s = s0; s <= t; s++) { float e = expf(sc[s] - m); sc[s] = e; sum += e; }
        g_sum = sum;
    }
    __syncthreads();

    float inv = 1.0f / g_sum;
    for (int i = threadIdx.x; i < HD; i += blockDim.x) {
        float acc = 0.0f;
        for (int s = s0; s <= t; s++)
            acc += sc[s] * V[(size_t)s * KVD + (size_t)kvh * HD + i];
        AO[(size_t)t * QD + (size_t)h * HD + i] = acc * inv;
    }
}

/* NTT-attention (E_CU_5): the score <q,k> is computed EXACTLY in integers — the
 * head vectors are quantized to int32 (x qscale=2^16) and the inner product is
 * accumulated in int64, which is bit-identical to the CPU poly-ring sp_pr_inner
 * (coeff 0 of the negacyclic product == sum_i q_i k_i; |ip| ~ 2^49 << 2^63). The
 * NTT itself is the CPU's overflow-free mechanism; on GPU int64 holds it directly
 * (see CU.6 note). Full causal (prefill). Softmax + V-sum stay f32, as on CPU. */
__global__ void k_attn_ntt(const float *Q, const float *K, const float *V,
                           int n_tok, int QD, int KVD, int HD, int group,
                           float ascale, float qscale, float *AO) {
    extern __shared__ float sc[];
    int n_heads = QD / HD;
    int b = blockIdx.x, t = b / n_heads, h = b % n_heads, kvh = h / group;
    const float *qh = Q + (size_t)t * QD + (size_t)h * HD;
    double qs2 = (double)qscale * (double)qscale;

    for (int s = threadIdx.x; s <= t; s += blockDim.x) {
        const float *kh = K + (size_t)s * KVD + (size_t)kvh * HD;
        long long ip = 0;
        for (int i = 0; i < HD; i++) {
            int qi = (int)lrintf(qh[i] * qscale);
            int ki = (int)lrintf(kh[i] * qscale);
            ip += (long long)qi * (long long)ki;
        }
        sc[s] = (float)((double)ip / qs2) * ascale;   /* exact integer <q,k>/2^32 */
    }
    __syncthreads();

    __shared__ float g_sum;
    if (threadIdx.x == 0) {
        float m = sc[0];
        for (int s = 1; s <= t; s++) if (sc[s] > m) m = sc[s];
        float sum = 0.0f;
        for (int s = 0; s <= t; s++) { float e = expf(sc[s] - m); sc[s] = e; sum += e; }
        g_sum = sum;
    }
    __syncthreads();

    float inv = 1.0f / g_sum;
    for (int i = threadIdx.x; i < HD; i += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s <= t; s++)
            acc += sc[s] * V[(size_t)s * KVD + (size_t)kvh * HD + i];
        AO[(size_t)t * QD + (size_t)h * HD + i] = acc * inv;
    }
}

/* ════════════════ autoregressive KV-cache decode (Beta) ════════════════
 * The prefill forward above is stateless (recomputes all positions per call).
 * Decode keeps K/V resident in VRAM across steps and processes ONE new token
 * per step, attending its single query over the cached [0..pos]. Position-aware
 * RoPE (the new token sits at absolute `p0`, not batch row 0) + a single-query
 * attention kernel are the only decode-specific pieces; everything else reuses
 * the prefill kernels at n_tok=1. f32 path; gate = argmax sequence == CPU
 * qwen3_generate_kv (knobs off). */

/* Device argmax over `n` logits -> *out_tok (single block). Keeps the winning
 * token in VRAM so the decode loop never round-trips logits to the host (the
 * per-step cudaStreamSynchronize latency killer). out_tok is also appended into
 * the device sequence buffer so the next step's embed reads it directly. */
__global__ void k_argmax(const float *logits, int n, int *out_tok) {
    __shared__ float sval[256];
    __shared__ int   sidx[256];
    float bv = -3.4e38f; int bi = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        if (logits[i] > bv) { bv = logits[i]; bi = i; }
    sval[threadIdx.x] = bv; sidx[threadIdx.x] = bi;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (threadIdx.x < o) {
            if (sval[threadIdx.x + o] > sval[threadIdx.x] ||
                (sval[threadIdx.x + o] == sval[threadIdx.x] && sidx[threadIdx.x + o] < sidx[threadIdx.x])) {
                sval[threadIdx.x] = sval[threadIdx.x + o];
                sidx[threadIdx.x] = sidx[threadIdx.x + o];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) *out_tok = sidx[0];
}

/* NEOX RoPE at ABSOLUTE position p0 (decode: one token, grid=n_heads, t=0). */
__global__ void k_rope_at(float *base, int n_heads, int d, int rowstride,
                          float rbase, int p0) {
    int b = blockIdx.x, t = b / n_heads, h = b % n_heads, i = threadIdx.x, half = d / 2;
    if (i < half) {
        float *v = base + (size_t)t * rowstride + (size_t)h * d;
        float freq = powf(rbase, -2.0f * (float)i / (float)d);
        float th = (float)(t + p0) * freq, c = cosf(th), s = sinf(th);
        float a = v[i], bb = v[i + half];
        v[i] = a * c - bb * s;
        v[i + half] = a * s + bb * c;
    }
}

/* Single-query GQA attention over a cached KV span [0..ctx). One block per query
 * head; layer_off selects the layer slab in the persistent cache. Shared = ctx
 * floats (scores). Mirrors k_attn's f32 math exactly (same order). */
__global__ void k_attn_decode(const float *q, const float *Kc, const float *Vc,
                              int ctx, int KVD, int HD, int group, float ascale,
                              size_t layer_off, float *ao) {
    extern __shared__ float sc[];
    int h = blockIdx.x, kvh = h / group;
    const float *qh = q + (size_t)h * HD;
    for (int s = threadIdx.x; s < ctx; s += blockDim.x) {
        const float *kh = Kc + layer_off + (size_t)s * KVD + (size_t)kvh * HD;
        float acc = 0.0f;
        for (int i = 0; i < HD; i++) acc += qh[i] * kh[i];
        sc[s] = acc * ascale;
    }
    __syncthreads();
    __shared__ float g_sum;
    if (threadIdx.x == 0) {
        float m = sc[0];
        for (int s = 1; s < ctx; s++) if (sc[s] > m) m = sc[s];
        float sum = 0.0f;
        for (int s = 0; s < ctx; s++) { float e = expf(sc[s] - m); sc[s] = e; sum += e; }
        g_sum = sum;
    }
    __syncthreads();
    float inv = 1.0f / g_sum;
    for (int i = threadIdx.x; i < HD; i += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < ctx; s++)
            acc += sc[s] * Vc[layer_off + (size_t)s * KVD + (size_t)kvh * HD + i];
        ao[(size_t)h * HD + i] = acc * inv;
    }
}

__global__ void k_gelu_mul(float *g, const float *up, size_t n) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = g[idx];
        const float k = 0.7978845608028654f;
        float th = tanhf(k * (x + 0.044715f * x * x * x));
        g[idx] = 0.5f * x * (1.0f + th) * up[idx];
    }
}

/* SwiGLU (Qwen3 FFN): g = silu(g) * up, silu(x) = x / (1 + exp(-x)). Matches the
 * CPU forward.c SwiGLU. */
__global__ void k_silu_mul(float *g, const float *up, size_t n) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = g[idx];
        g[idx] = (x / (1.0f + expf(-x))) * up[idx];
    }
}

__global__ void k_add(float *x, const float *y, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

/* Decode a per-row Frobenius packed weight (Q8 or per-row-promoted Q4) to f32,
 * row-major [rows x cols]. Mirrors matmul_arena: w[j][i] = code * row_scale[j]/qmax
 * (qmax 127 for a Q8 row, 7 for a Q4 row). Q4 codes are two-per-byte, low nibble =
 * even index, sign-extended. grid=(rows, ceil(cols/256)): rows on grid.x (limit
 * 2^31, so an untied LM head with V>65535 rows is fine), cols on grid.y. */
__global__ void k_dequant_arena(const unsigned char *codes, const unsigned long long *row_off,
                                const float *row_scale, const unsigned char *row_prec,
                                int rows, int cols, float *out) {
    int j = blockIdx.x;
    int i = blockIdx.y * blockDim.x + threadIdx.x;
    if (j >= rows || i >= cols) return;
    const unsigned char *rc = codes + row_off[j];
    int code; float inv;
    if (row_prec[j] == 8) {
        code = (int)((const signed char *)rc)[i];
        inv = row_scale[j] * (1.0f / 127.0f);
    } else {
        unsigned char byte = rc[i >> 1];
        int nib = (i & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);
        code = (nib ^ 8) - 8;                 /* sign-extend 4-bit */
        inv = row_scale[j] * (1.0f / 7.0f);
    }
    out[(size_t)j * cols + i] = (float)code * inv;
}

/* ETA.2 (Gemma4 oracle parity): write the RAW integer codes as f32 — NO scale.
 * The core's matmul_arena (§4.8 inline-lift) accumulates Σ code·x exactly
 * (codes ≤127 are exact in f32) and applies row_scale/qmax ONCE at the end.
 * Dequantizing per-weight first (k_dequant_arena) injects one extra f32
 * rounding per term — measured 2.8e-3 max-rel divergence vs the oracle at
 * E2B layer 0. This kernel + k_scale_rows reproduce the oracle's arithmetic:
 * exact codes into the SGEMM, ONE lift after. */
__global__ void k_codes_f32(const unsigned char *codes, const unsigned long long *row_off,
                            const unsigned char *row_prec, int rows, int cols, float *out) {
    int j = blockIdx.x;
    int i = blockIdx.y * blockDim.x + threadIdx.x;
    if (j >= rows || i >= cols) return;
    const unsigned char *rc = codes + row_off[j];
    int code;
    if (row_prec[j] == 8) {
        code = (int)((const signed char *)rc)[i];
    } else {
        unsigned char byte = rc[i >> 1];
        int nib = (i & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);
        code = (nib ^ 8) - 8;
    }
    out[(size_t)j * cols + i] = (float)code;
}

/* the single post-GEMM lift: Y[t][j] *= row_scale[j]/qmax(row_prec[j]) */
__global__ void k_scale_rows(float *Y, const float *row_scale, const unsigned char *row_prec,
                             int n_tok, int out) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= out) return;
    float inv = row_scale[j] * (row_prec[j] == 8 ? (1.0f / 127.0f) : (1.0f / 7.0f));
    for (int t = 0; t < n_tok; t++) Y[(size_t)t * out + j] *= inv;
}

/* ═══════════════ BETA.3a: fused INT8 GEMV for decode (dp4a) ═══════════════
 * The decode path is M=1 (single-token GEMV), so the m8n8k16 tensor-core tile
 * (ptx_mma.cuh) can't help — a 1-row query fills 1/8 of the MMA_M tile. The
 * Turing lever for a GEMV is __dp4a (4-wide INT8 dot -> INT32, native sm_75):
 * read the 1-byte Q8 arena codes STRAIGHT from VRAM (no f32 scratch), quantize
 * the activation to int8, accumulate in int32, scale back. The current
 * dequant-then-SGEMM reads the code (1B) then writes (4B) + rereads (4B) an f32
 * scratch = ~9 B/weight; this reads 1 B/weight. That byte ratio IS the win the
 * anchored f32==Q8 result showed was being thrown away. */

/* dynamic per-vector int8 quant of the activation: sx = maxabs/127,
 * qx[i] = round(x[i]/sx) clamped [-127,127]. qx padded to a multiple of 4
 * (zero tail) so the GEMV can int-load 4 codes per dp4a. One block. */
__global__ void k_quant_act_int8(const float *x, int n, int npad,
                                 signed char *qx, float *sx) {
    __shared__ float sm[256];
    float m = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) { float a = fabsf(x[i]); if (a > m) m = a; }
    sm[threadIdx.x] = m; __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (threadIdx.x < o && sm[threadIdx.x + o] > sm[threadIdx.x]) sm[threadIdx.x] = sm[threadIdx.x + o];
        __syncthreads();
    }
    float scale = sm[0] > 0.0f ? sm[0] * (1.0f / 127.0f) : 1.0f;
    if (threadIdx.x == 0) *sx = scale;
    float inv = 1.0f / scale;
    for (int i = threadIdx.x; i < npad; i += blockDim.x) {
        float v = (i < n) ? x[i] * inv : 0.0f;
        int q = __float2int_rn(v); if (q > 127) q = 127; if (q < -127) q = -127;
        qx[i] = (signed char)q;
    }
}

/* fused INT8 GEMV: y[j] = (row_scale[j]/127) * sx * Σ_i code[j][i]·qx[i], the inner
 * INT8·INT8->INT32 dot via __dp4a. One block per output row; `in` must be a
 * multiple of 4 (qwen3 dims are). Q8 rows only (row_prec==8). codes+row_off[j] is
 * 4-aligned (cudaMalloc base + j*in, in%4==0), so the int-cast load is a coalesced
 * 128-byte warp transaction reading 4 weights/thread. */
__global__ void k_gemv_q8_dp4a(const signed char *codes, const unsigned long long *row_off,
                               const float *row_scale, int in, const signed char *qx,
                               const float *sx, float *y, int out) {
    int j = blockIdx.x;
    if (j >= out) return;
    const int *wrow = (const int *)(codes + row_off[j]);   /* in/4 packed-int8 words */
    const int *qxi  = (const int *)qx;
    int n4 = in >> 2, acc = 0;
    for (int k = threadIdx.x; k < n4; k += blockDim.x)
        acc = __dp4a(wrow[k], qxi[k], acc);
    __shared__ int sm[256];
    sm[threadIdx.x] = acc; __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (threadIdx.x < o) sm[threadIdx.x] += sm[threadIdx.x + o];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[j] = (float)sm[0] * (row_scale[j] * (1.0f / 127.0f)) * (*sx);
}

/* TUNED dp4a GEMV: one WARP per output row (32 lanes collaborate), 128-bit int4
 * loads (16 Q8 codes/thread/iter — maximizes the GDDR6 transaction size), and a
 * __shfl_down_sync register-speed reduction (no shared memory). 8 warps/block =
 * 8 rows/block for SM occupancy. `in` must be a multiple of 16 (qwen3 dims are);
 * codes+row_off[j] and qx are 16-byte aligned (cudaMalloc base + j*in, in%16==0).
 * Note: int4 here is the 16-BYTE CUDA vector type, NOT 4-bit precision. */
__global__ void k_gemv_q8_dp4a_v2(const signed char *codes, const unsigned long long *row_off,
                                  const float *row_scale, int in, const signed char *qx,
                                  const float *sx, float *y, int out) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int j = blockIdx.x * (blockDim.x >> 5) + warp;
    if (j >= out) return;
    const int4 *wrow = (const int4 *)(codes + row_off[j]);   /* in/16 16-byte chunks */
    const int4 *qxi  = (const int4 *)qx;
    const int n16 = in >> 4;
    int acc = 0;
    for (int c = lane; c < n16; c += 32) {
        int4 wv = wrow[c], qv = qxi[c];                      /* 128-bit coalesced loads */
        acc = __dp4a(wv.x, qv.x, acc);
        acc = __dp4a(wv.y, qv.y, acc);
        acc = __dp4a(wv.z, qv.z, acc);
        acc = __dp4a(wv.w, qv.w, acc);
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) y[j] = (float)acc * (row_scale[j] * (1.0f / 127.0f)) * (*sx);
}

/* BETA.3a-v4: Q4 dp4a GEMV (0.5 B/weight, ~7x f32 at 12B-scale; bench-proven in
 * tests/bench_gemv_int8.cu, host-ref correctness 1.34e-7). Q4 arena = 2 nibbles/
 * byte (low=even idx, high=odd, sign-ext (n^8)-8, qmax=7). Read int4 (16 B = 32
 * packed weights) straight from VRAM, unpack the nibbles to int8 in the ALU (free
 * under memory-bound), feed dp4a. Activation stays int8 (qmax 127). `in` must be a
 * multiple of 32; codes+row_off[j] 16-aligned (cudaMalloc base + j*in/2, in%32==0
 * => in/2 % 16 == 0). */
__device__ __forceinline__ void sp_unpack8(int w, int &lo4, int &hi4) {
    int b0=w&0xFF, b1=(w>>8)&0xFF, b2=(w>>16)&0xFF, b3=(w>>24)&0xFF;
    #define SPX(byte,hi) (((((((hi)?((byte)>>4):(byte)))&0xF)^0x8)-0x8)&0xFF)
    lo4 = SPX(b0,0) | (SPX(b0,1)<<8) | (SPX(b1,0)<<16) | (SPX(b1,1)<<24);
    hi4 = SPX(b2,0) | (SPX(b2,1)<<8) | (SPX(b3,0)<<16) | (SPX(b3,1)<<24);
    #undef SPX
}
__global__ void k_gemv_q4_dp4a_v2(const unsigned char *codes, const unsigned long long *row_off,
                                  const float *row_scale, int in, const signed char *qx,
                                  const float *sx, float *y, int out) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int j = blockIdx.x * (blockDim.x >> 5) + warp;
    if (j >= out) return;
    const int4 *wrow = (const int4 *)(codes + row_off[j]);   /* 16 B = 32 Q4 weights */
    const int4 *qxi  = (const int4 *)qx;
    const int n32 = in >> 5;
    int acc = 0, a, b;
    for (int c = lane; c < n32; c += 32) {
        int4 wv = wrow[c];
        int4 q0 = qxi[2*c], q1 = qxi[2*c + 1];
        sp_unpack8(wv.x, a, b); acc = __dp4a(a, q0.x, acc); acc = __dp4a(b, q0.y, acc);
        sp_unpack8(wv.y, a, b); acc = __dp4a(a, q0.z, acc); acc = __dp4a(b, q0.w, acc);
        sp_unpack8(wv.z, a, b); acc = __dp4a(a, q1.x, acc); acc = __dp4a(b, q1.y, acc);
        sp_unpack8(wv.w, a, b); acc = __dp4a(a, q1.z, acc); acc = __dp4a(b, q1.w, acc);
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) y[j] = (float)acc * (row_scale[j] * (1.0f / 7.0f)) * (*sx);
}

/* ════════════════════════ device weight cache ════════════════════════ */

/* one weight matrix on device: either plain f32 (f32 != NULL) or packed arena. */
struct DevTensor {
    int in, out;
    int prec;                    /* host-side per-TENSOR precision: 8 (Q8), 4 (Q4),
                                  * 0 = f32 or non-uniform rows (forces dequant path) */
    float *f32;                  /* != NULL => plain f32 [out*in] */
    unsigned char     *codes;    /* packed (f32 == NULL): math-core arena layout */
    unsigned long long *row_off; /* [out] */
    float             *row_scale;/* [out] */
    unsigned char     *row_prec; /* [out] */
};

struct CudaWeights {
    const qwen3_model *key;
    int L;
    cublasHandle_t cublas;
    cudaStream_t   stream;
    float *embd;        /* [V*E] token-embd f32 (embed gather; also gemma3 tied head) */
    float *out_norm;    /* [E] */
    DevTensor head;     /* untied LM head (qwen3 m->output, f32-or-packed); unused when tied */
    size_t scratch_n;   /* max packed weight elem count (0 if no arena) */
    DevTensor *Wq, *Wk, *Wv, *Wo, *Wgate, *Wup, *Wdown;
    /* post_attn/post_ffw are sandwich norms (gemma3 + gemma4); NULL arrays on qwen3. */
    float **attn_norm, **ffn_norm, **q_norm, **k_norm, **post_attn, **post_ffw;

    /* ── ETA.1 Gemma4 (SP_ARCH_GEMMA4) extras; zeroed on other arches. ──
     * Per-layer AltUp: inp_gate [E->PL] + proj [PL->E] matmuls, post_norm [E],
     * out_scale [1] scalar. Model-level: per_layer_model_proj [E -> NL*PL]
     * (the AltUp precompute matmul), per_layer_proj_norm [PL], rope_freqs
     * [hd_global/2] proportional freq-factor table (global layers only).
     * per_layer_token_embd stays HOST-side (row-gathered per token via
     * sp_arena_dequant_row, mirroring the CPU sp_weight_row — uploading it
     * f32 would be ~GBs). Shared-KV: Wk/Wv are built ONLY for owner layers
     * [0, kvfs); sharer entries stay zeroed (the forward reuses the owner's
     * cached K/V and skips its own projection, per gemma4.c). */
    DevTensor *Wplig, *Wplproj;     /* [L] AltUp per-layer matmuls (NULL arrays if PL=0) */
    float **pl_post_norm;           /* [L][E]  per_layer_post_norm */
    float **pl_out_scale;           /* [L][1]  layer_output_scale  */
    DevTensor pl_model_proj;        /* per_layer_model_proj [E -> NL*PL] */
    float *pl_proj_norm;            /* [PL] */
    float *rope_freqs;              /* [hd_global/2] */
    /* ETA.5b: the PLE table [V x NL*PL] uploaded PACKED (codes ~1 B/weight; f32
     * would be ~9 GB) so the decode gathers the per-token row ON DEVICE — this
     * severs the per-token host PLE sync and makes the step graph-capturable.
     * Zeroed when no arena (forward paths keep the host row-gather fallback). */
    DevTensor pl_tok_embd;
    /* ETA.5b: packed token_embd codes (when arena-packed) — the TIED-head dp4a
     * route [V x E]. The f32 embd above stays for the embed gather + parity paths. */
    DevTensor embd_packed;
};
static CudaWeights g_w = {};

/* dequant a GGUF weight tensor [out x in] to device f32 [out*in], row-major. */
static float *upload_weight(const qwen3_model *m, const gguf_tensor *t, int in, int out) {
    const uint8_t *base = (const uint8_t *)gguf_tensor_data(m->gguf, t);
    size_t rb = row_bytes(t->type, in);
    if (!base || rb == 0) { sp_set_error("upload_weight: null/unsupported tensor"); return NULL; }
    size_t n = (size_t)out * in;
    float *host = (float *)malloc(n * sizeof(float));
    if (!host) { sp_set_error("upload_weight: host OOM"); return NULL; }
    for (int j = 0; j < out; j++)
        if (sp_dequant_row(base + (size_t)j * rb, t->type, in, host + (size_t)j * in)) {
            free(host); sp_set_error("upload_weight: dequant failed"); return NULL;
        }
    float *dev = NULL;
    cudaError_t e = cudaMalloc(&dev, n * sizeof(float));
    if (e == cudaSuccess) e = cudaMemcpy(dev, host, n * sizeof(float), cudaMemcpyHostToDevice);
    free(host);
    if (e != cudaSuccess) { fail_cuda(e, "upload_weight cudaMalloc/Memcpy"); if (dev) cudaFree(dev); return NULL; }
    return dev;
}

static float *upload_vec(const qwen3_model *m, const gguf_tensor *t, int n) {
    const float *host = as_f32(m, t);
    if (!host) { sp_set_error("upload_vec: null tensor"); return NULL; }
    float *dev = NULL;
    cudaError_t e = cudaMalloc(&dev, (size_t)n * sizeof(float));
    if (e == cudaSuccess) e = cudaMemcpy(dev, host, (size_t)n * sizeof(float), cudaMemcpyHostToDevice);
    if (e != cudaSuccess) { fail_cuda(e, "upload_vec"); if (dev) cudaFree(dev); return NULL; }
    return dev;
}

/* generic host->device copy of a typed array */
template <typename T>
static int upload_arr(const T *host, size_t count, T **dev) {
    cudaError_t e = cudaMalloc(dev, count * sizeof(T));
    if (e == cudaSuccess) e = cudaMemcpy(*dev, host, count * sizeof(T), cudaMemcpyHostToDevice);
    if (e != cudaSuccess) return fail_cuda(e, "upload_arr");
    return 0;
}

/* upload a packed arena tensor (compact math-core layout) into a DevTensor. */
static int upload_packed(const sp_frob_packed_tensor *pt, DevTensor *d) {
    d->in = pt->cols; d->out = pt->rows; d->f32 = NULL;
    d->codes = NULL; d->row_off = NULL; d->row_scale = NULL; d->row_prec = NULL;
    /* host-side per-tensor precision: the dp4a GEMV needs uniform rows. row_prec[0]
     * is the tensor precision; if ANY row differs, prec=0 forces the dequant path
     * (the on-device k_dequant_arena handles per-row mixed precision; the GEMV does
     * not). This is the fix for mixed-precision arenas (e.g. Q8 head in a Q4 body). */
    d->prec = pt->rows > 0 ? (int)pt->row_prec[0] : 0;
    for (uint32_t r = 1; r < pt->rows; r++)
        if (pt->row_prec[r] != pt->row_prec[0]) { d->prec = 0; break; }
    /* row_off is size_t on the host (8 bytes on x64) -> unsigned long long on device. */
    if (upload_arr<unsigned char>(pt->codes, pt->codes_bytes, &d->codes)) return 1;
    if (upload_arr<unsigned long long>((const unsigned long long *)pt->row_off,
                                       (size_t)pt->rows, &d->row_off)) return 1;
    if (upload_arr<float>(pt->row_scale, (size_t)pt->rows, &d->row_scale)) return 1;
    if (upload_arr<unsigned char>(pt->row_prec, (size_t)pt->rows, &d->row_prec)) return 1;
    return 0;
}

static void free_devtensor(DevTensor *d) {
    if (d->f32) cudaFree(d->f32);
    if (d->codes) cudaFree(d->codes);
    if (d->row_off) cudaFree(d->row_off);
    if (d->row_scale) cudaFree(d->row_scale);
    if (d->row_prec) cudaFree(d->row_prec);
    DevTensor z = {}; *d = z;
}

static void free_weights(CudaWeights *w) {
    if (w->embd) cudaFree(w->embd);
    if (w->out_norm) cudaFree(w->out_norm);
    free_devtensor(&w->head);   /* untied head (no-op when tied: all ptrs NULL) */
    DevTensor **dts[] = { &w->Wq,&w->Wk,&w->Wv,&w->Wo,&w->Wgate,&w->Wup,&w->Wdown,
                          &w->Wplig,&w->Wplproj };
    for (size_t a = 0; a < sizeof(dts)/sizeof(dts[0]); a++) {
        DevTensor *arr = *dts[a];
        if (arr) { for (int L = 0; L < w->L; L++) free_devtensor(&arr[L]); free(arr); }
    }
    float ***ns[] = { &w->attn_norm,&w->ffn_norm,&w->q_norm,&w->k_norm,&w->post_attn,&w->post_ffw,
                      &w->pl_post_norm,&w->pl_out_scale };
    for (size_t a = 0; a < sizeof(ns)/sizeof(ns[0]); a++) {
        float **arr = *ns[a];
        if (arr) { for (int L = 0; L < w->L; L++) if (arr[L]) cudaFree(arr[L]); free(arr); }
    }
    free_devtensor(&w->pl_model_proj);
    free_devtensor(&w->pl_tok_embd);
    free_devtensor(&w->embd_packed);
    if (w->pl_proj_norm) cudaFree(w->pl_proj_norm);
    if (w->rope_freqs) cudaFree(w->rope_freqs);
    if (w->cublas) cublasDestroy(w->cublas);
    if (w->stream) cudaStreamDestroy(w->stream);
    CudaWeights z = {}; *w = z;
}

/* build one matmul weight: packed if it's in the arena, else f32 from GGUF. */
static int build_w(const qwen3_model *m, const gguf_tensor *W, int in, int out,
                   DevTensor *d, size_t *scratch_n) {
    const sp_arena_tensor *at = m->arena ? sp_arena_find(m->arena, W->name) : NULL;
    if (at) {
        if (upload_packed(&at->pt, d)) return 1;
        size_t need = (size_t)d->out * d->in;
        if (need > *scratch_n) *scratch_n = need;
        return 0;
    }
    DevTensor z = {}; *d = z;
    d->f32 = upload_weight(m, W, in, out);
    if (!d->f32) return 1;
    d->in = in; d->out = out;
    return 0;
}

#define ALLOC_DT(field) do { w->field = (DevTensor *)calloc((size_t)L, sizeof(DevTensor)); \
    if (!w->field) { sp_set_error("DevTensor array OOM"); free_weights(w); return 1; } } while (0)
#define ALLOC_NM(field) do { w->field = (float **)calloc((size_t)L, sizeof(float*)); \
    if (!w->field) { sp_set_error("norm array OOM"); free_weights(w); return 1; } } while (0)
#define BUILDW(field, tensor, in, out) do { \
    if (build_w(m, ly->tensor, (in), (out), &w->field[Li], &w->scratch_n)) { free_weights(w); return 1; } } while (0)
#define UPV(field, tensor, n) do { w->field[Li] = upload_vec(m, ly->tensor, (n)); \
    if (!w->field[Li]) { free_weights(w); return 1; } } while (0)

static int build_weights(const qwen3_model *m, CudaWeights *w) {
    const qwen3_config *c = &m->cfg;
    const int E = (int)c->n_embd, FF = (int)c->n_ff, HD = (int)c->head_dim;
    const int NH = (int)c->n_head, NKV = (int)c->n_head_kv, V = (int)c->n_vocab;
    const int QD = NH * HD, KVD = NKV * HD, L = (int)c->n_layers;

    /* ETA.1 Gemma4: per-layer head GEOMETRY (gemma4.c). GLOBAL layers
     * (L % period == period-1) use cfg.head_dim/n_head/n_head_kv (512/4/1);
     * SWA layers use g4_hd_swa/g4_nh_swa/g4_nkv_swa (256/8/2). The Q/K/V
     * projection WIDTHS therefore differ per layer. Shared-KV: only layers
     * [0, kvfs) own K/V. PL = AltUp per-layer-input width (0 = no AltUp). */
    const int is_g4 = (c->arch == SP_ARCH_GEMMA4);
    const int g4_period = is_g4 ? ((int)c->g4_swa_period ? (int)c->g4_swa_period : 6) : 0;
    const int g4_kvfs   = is_g4 ? ((int)c->g4_n_kv_from_start ? (int)c->g4_n_kv_from_start : L) : L;
    const int g4_PL     = is_g4 ? (int)c->g4_n_embd_per_layer : 0;
    const int s_nh = is_g4 ? (int)c->g4_nh_swa  : NH;
    const int s_nkv = is_g4 ? (int)c->g4_nkv_swa : NKV;
    const int s_hd = is_g4 ? (int)c->g4_hd_swa  : HD;

    const int is_gemma = (c->arch == SP_ARCH_GEMMA3) || is_g4;   /* sandwich norms */
    const int tied = (m->output == m->token_embd);

    CudaWeights z = {}; *w = z;
    w->key = m; w->L = L;

    cudaError_t e = cudaStreamCreate(&w->stream);
    if (e != cudaSuccess) return fail_cuda(e, "cudaStreamCreate");
    CB(cublasCreate(&w->cublas), "cublasCreate");
    CB(cublasSetStream(w->cublas, w->stream), "cublasSetStream");
    CB(cublasSetMathMode(w->cublas, CUBLAS_DEFAULT_MATH), "cublasSetMathMode");

    /* token embedding (the embed gather always uses f32; gemma3's tied head reuses
     * it). From the arena if it was packed for release, else GGUF. ETA.5b: when
     * arena-packed, ALSO keep the packed codes resident (embd_packed) — the TIED
     * head is the single largest decode matmul (V x E), and the dp4a route reads
     * 1 B/weight instead of the 4 B/weight f32 GEMM. */
    {
        const sp_arena_tensor *eat = m->arena ? sp_arena_find(m->arena, m->token_embd->name) : NULL;
        if (eat) {
            size_t n = (size_t)V * E;
            /* ETA.5b.4: when the dequanted f32 embedding would blow the VRAM
             * budget (12B: ~4 GB), keep ONLY the packed codes resident — the
             * decode embeds via k_embed_packed_at and serves the tied head via
             * dp4a. The f32 copy is uploaded when it fits (parity/forward paths
             * use it; same dequant arithmetic either way, byte-identical). */
            if (n * sizeof(float) <= (size_t)2u << 30) {
                float *host = (float *)malloc(n * sizeof(float));
                if (!host) { sp_set_error("embd host OOM"); free_weights(w); return 1; }
                for (int r = 0; r < V; r++)
                    if (sp_arena_dequant_row(eat, r, host + (size_t)r * E)) {
                        free(host); sp_set_error("embd arena dequant"); free_weights(w); return 1;
                    }
                int rc = upload_arr<float>(host, n, &w->embd);
                free(host);
                if (rc) { free_weights(w); return 1; }
            }
            if (upload_packed(&eat->pt, &w->embd_packed)) { free_weights(w); return 1; }
        } else {
            w->embd = upload_weight(m, m->token_embd, E, V);
            if (!w->embd) { free_weights(w); return 1; }
        }
    }
    w->out_norm = upload_vec(m, m->output_norm, E);
    if (!w->out_norm) { free_weights(w); return 1; }

    /* untied LM head (qwen3): a separate weight, arena-packed in Q8. Tied (gemma3):
     * the head GEMM reuses w->embd; w->head stays zeroed. */
    if (!tied) {
        if (build_w(m, m->output, E, V, &w->head, &w->scratch_n)) { free_weights(w); return 1; }
    }

    ALLOC_DT(Wq); ALLOC_DT(Wk); ALLOC_DT(Wv); ALLOC_DT(Wo);
    ALLOC_DT(Wgate); ALLOC_DT(Wup); ALLOC_DT(Wdown);
    ALLOC_NM(attn_norm); ALLOC_NM(ffn_norm); ALLOC_NM(q_norm); ALLOC_NM(k_norm);
    if (is_gemma) { ALLOC_NM(post_attn); ALLOC_NM(post_ffw); }   /* sandwich norms */
    if (g4_PL) { ALLOC_DT(Wplig); ALLOC_DT(Wplproj); ALLOC_NM(pl_post_norm); }
    /* out_scale is INDEPENDENT of AltUp: the dense gemma-4 (12B) carries
     * layer_output_scale with PL=0 (the oracle applies it whenever present). */
    if (is_g4 && m->layers[0].out_scale) { ALLOC_NM(pl_out_scale); }

    for (int Li = 0; Li < L; Li++) {
        const qwen3_layer *ly = &m->layers[Li];
        /* per-layer geometry: identical on non-gemma4; global-vs-SWA split on gemma4 */
        const int g4_global = is_g4 && ((Li % g4_period) == g4_period - 1);
        const int nh_L  = is_g4 ? (g4_global ? NH  : s_nh)  : NH;
        const int nkv_L = is_g4 ? (g4_global ? NKV : s_nkv) : NKV;
        const int hd_L  = is_g4 ? (g4_global ? HD  : s_hd)  : HD;
        const int qd_L  = nh_L * hd_L, kvd_L = nkv_L * hd_L;
        /* per-layer ELASTIC FFN width (MatFormer): the layer's ffn_gate out-dim
         * (synth tensors carry dims from the .sp-model entry); fall back to n_ff. */
        const int ff_L  = (is_g4 && ly->ffn_gate && ly->ffn_gate->n_dims >= 2 && ly->ffn_gate->dims[1] > 0)
                          ? (int)ly->ffn_gate->dims[1] : FF;
        const int owns_kv = !is_g4 || (Li < g4_kvfs);   /* shared-KV: sharers skip K/V */

        BUILDW(Wq, attn_q, E, qd_L);
        if (owns_kv) {
            BUILDW(Wk, attn_k, E, kvd_L);
            /* V-less layers (dense gemma-4 globals): attn_v ABSENT — the forward
             * copies the raw K projection (Wv[Li] stays zeroed, detectable). */
            if (ly->attn_v) BUILDW(Wv, attn_v, E, kvd_L);
        }
        BUILDW(Wo, attn_output, qd_L, E);
        BUILDW(Wgate, ffn_gate, E, ff_L); BUILDW(Wup, ffn_up, E, ff_L); BUILDW(Wdown, ffn_down, ff_L, E);
        UPV(attn_norm, attn_norm, E);   UPV(ffn_norm, ffn_norm, E);
        UPV(q_norm, attn_q_norm, hd_L);
        if (owns_kv) UPV(k_norm, attn_k_norm, hd_L);     /* sharers never norm K */
        if (is_gemma) { UPV(post_attn, post_attn_norm, E); UPV(post_ffw, post_ffw_norm, E); }
        if (g4_PL) {
            BUILDW(Wplig, per_layer_inp_gate, E, g4_PL);
            BUILDW(Wplproj, per_layer_proj, g4_PL, E);
            UPV(pl_post_norm, per_layer_post_norm, E);
        }
        if (w->pl_out_scale && ly->out_scale) UPV(pl_out_scale, out_scale, 1);
    }

    /* ETA.1 Gemma4 model-level AltUp tensors. ETA.5b: per_layer_token_embd is
     * now ALSO uploaded — packed, codes-only (see pl_tok_embd below) — for the
     * device-side per-token row gather; the forward/probe paths still host-gather
     * via sp_arena_dequant_row (n_tok rows at once). */
    if (g4_PL) {
        if (build_w(m, m->per_layer_model_proj, E, L * g4_PL, &w->pl_model_proj, &w->scratch_n)) {
            free_weights(w); return 1;
        }
        w->pl_proj_norm = upload_vec(m, m->per_layer_proj_norm, g4_PL);
        if (!w->pl_proj_norm) { free_weights(w); return 1; }
        /* ETA.5b: PLE table packed to VRAM (device per-token row gather). E2B Q8
         * = V*NL*PL ~2.2 GB of codes. Optional: when absent the decode falls back
         * to the ETA.5a host row-gather (and the graph path stays off). */
        const sp_arena_tensor *plt = m->arena ? sp_arena_find(m->arena, "per_layer_token_embd.weight") : NULL;
        if (plt && upload_packed(&plt->pt, &w->pl_tok_embd)) { free_weights(w); return 1; }
    }
    /* global-layer proportional freq factors — INDEPENDENT of AltUp (the dense
     * 12B carries rope_freqs with PL=0). */
    if (is_g4 && m->rope_freqs) {
        w->rope_freqs = upload_vec(m, m->rope_freqs, HD / 2);
        if (!w->rope_freqs) { free_weights(w); return 1; }
    }
    return 0;
}

/* ETA.1 structural probe: build (upload) the full Gemma4 weight set for a
 * core-bridged model (sp_model_load -> sp_model_to_gemma4) and report the
 * per-layer geometry it resolved. Returns 0 on success. The first gate of the
 * Stage-Eta CUDA port: proves the engine CUDA layer can ingest the gemma4
 * arena + owned norms across the core/engine link seam, with per-layer Q/KV
 * widths, shared-KV skips, elastic FFN, and the AltUp tensor set. */
extern "C" int gemma4_cuda_weights_probe(const qwen3_model *m) {
    if (!m || m->cfg.arch != SP_ARCH_GEMMA4) { sp_set_error("gemma4 probe: not a gemma4 model"); return 1; }
    if (g_w.key != m) { free_weights(&g_w); if (build_weights(m, &g_w)) return 1; }
    const qwen3_config *c = &m->cfg;
    const int L = (int)c->n_layers;
    const int period = (int)c->g4_swa_period ? (int)c->g4_swa_period : 6;
    const int kvfs = (int)c->g4_n_kv_from_start ? (int)c->g4_n_kv_from_start : L;
    int n_global = 0, n_owner = 0, ff_min = 1 << 30, ff_max = 0;
    for (int Li = 0; Li < L; Li++) {
        if ((Li % period) == period - 1) n_global++;
        if (Li < kvfs) n_owner++;
        int ff = g_w.Wgate[Li].out; if (ff < ff_min) ff_min = ff; if (ff > ff_max) ff_max = ff;
    }
    fprintf(stderr, "    [g4-cuda-w] L=%d global=%d swa=%d kv-owners=%d sharers=%d "
            "ff=[%d..%d] PL=%d qd(g)=%d qd(s)=%d kvd(g)=%d kvd(s)=%d "
            "plmp=%dx%d prec(Wq0)=%d\n",
            L, n_global, L - n_global, n_owner, L - n_owner, ff_min, ff_max,
            (int)c->g4_n_embd_per_layer,
            (int)c->n_head * (int)c->head_dim, (int)c->g4_nh_swa * (int)c->g4_hd_swa,
            (int)c->n_head_kv * (int)c->head_dim, (int)c->g4_nkv_swa * (int)c->g4_hd_swa,
            g_w.pl_model_proj.in, g_w.pl_model_proj.out, g_w.Wq[0].prec);
    return 0;
}

/* ═══════════ ETA.2: Gemma4 CUDA prefill — TRUNCATABLE probe skeleton ═══════════
 * Mirrors core/forward/gemma4.c (the bit-faithful CPU oracle) onto the CUDA
 * kernel vocabulary, with the bisection boundaries the Stage-Eta plan demands:
 *
 *   n_layers = 0              -> embed + sqrt(E) scale only
 *   n_layers = N, attn_only=1 -> through layer N-1's ATTENTION residual
 *   n_layers = N, attn_only=0 -> through layer N-1's FFN residual
 *
 * and downloads the residual stream x [n_tok x E] at the boundary. Implemented
 * here (gated by the layer-0 parity probe E_G4_CU_L0): per-layer GLOBAL/SWA
 * geometry + per-layer projection widths, attention scale = 1.0 (NO score cap),
 * QK-norm before RoPE, the WEIGHTLESS V-norm, proportional rope_freqs on global
 * layers, shared-KV (owners store K/V on device; sharers reuse owner kvfs-1
 * global / kvfs-2 SWA and skip their own projection), per-layer elastic FFN
 * (GeGLU) + sandwich norms.
 *
 * NOT YET (deliberately, per the gated plan — the FULL-forward gate vs the CPU
 * oracle only goes green once these land):
 *   - ETA.4: AltUp precompute + per-layer injection + per-layer out_scale.
 *   - the final norm + tied head + k_softcap (wired in the full forward entry,
 *     which follows once AltUp closes; k_softcap is already defined above).
 * Probe boundaries therefore stop BEFORE the AltUp injection point. */
static int gemm_w_lift(cublasHandle_t h, cudaStream_t st, const DevTensor *W,
                       const float *dX, float *dY, int n_tok, float *scratch);  /* defined below */
static int gemm(cublasHandle_t h, const float *dW, const float *dX, float *dY,
                int n_tok, int in, int out);                                    /* defined below */
static int gemv_w_packed(cudaStream_t st, const DevTensor *W, const float *dX, float *dY,
                         signed char *dqx, float *dsx);                         /* defined below */

/* attn_only modes: 1 = stop after last layer's attention residual; 2/3/4/5 =
 * intra-block bisection stages; 0 = stop after last layer's FFN residual;
 * ETA.4: -1 = FULL FORWARD — all layers + AltUp injection + per-layer out_scale
 * + final norm + (tied) head + logit softcap; out_x = logits [n_tok x n_vocab]. */
extern "C" int gemma4_cuda_probe(const qwen3_model *m, const int32_t *tokens,
                                 int n_tok, int n_layers, int attn_only, float *out_x) {
    if (!m || m->cfg.arch != SP_ARCH_GEMMA4) { sp_set_error("gemma4_cuda_probe: not gemma4"); return 1; }
    if (n_tok <= 0 || !tokens || !out_x) { sp_set_error("gemma4_cuda_probe: bad args"); return 1; }
    const qwen3_config *c = &m->cfg;
    const int E = (int)c->n_embd, NL = (int)c->n_layers, SW = (int)c->sliding_window;
    const float eps = c->rms_eps, embscale = sqrtf((float)E);
    const int period = (int)c->g4_swa_period ? (int)c->g4_swa_period : 6;
    const int kvfs   = (int)c->g4_n_kv_from_start ? (int)c->g4_n_kv_from_start : NL;
    const int g_nh = (int)c->n_head, g_nkv = (int)c->n_head_kv, g_hd = (int)c->head_dim;
    const int s_nh = (int)c->g4_nh_swa, s_nkv = (int)c->g4_nkv_swa, s_hd = (int)c->g4_hd_swa;
    const float g_base = c->rope_freq_base, s_base = c->g4_rope_base_swa;
    const int QDmax = (g_nh*g_hd > s_nh*s_hd) ? g_nh*g_hd : s_nh*s_hd;
    const int KVDmax = (g_nkv*g_hd > s_nkv*s_hd) ? g_nkv*g_hd : s_nkv*s_hd;
    const int full = (attn_only == -1);                /* ETA.4 full forward */
    const int PL = (int)c->g4_n_embd_per_layer;
    const int V = (int)c->n_vocab;
    const float softcap = c->g4_logit_softcap;
    if (full) n_layers = NL;
    if (n_layers < 0) n_layers = 0;
    if (n_layers > NL) n_layers = NL;

    if (g_w.key != m) { free_weights(&g_w); if (build_weights(m, &g_w)) return 1; }
    cublasHandle_t cb = g_w.cublas;
    cudaStream_t st = g_w.stream;

    /* per-layer max FFN width across the truncated range */
    int FFmax = (int)c->n_ff;
    for (int L = 0; L < n_layers; L++)
        if (g_w.Wgate[L].out > FFmax) FFmax = g_w.Wgate[L].out;

    int *dtoks = NULL;
    float *dx=NULL,*dnx=NULL,*dq=NULL,*dk=NULL,*dv=NULL,*dao=NULL,*dap=NULL,
          *dg=NULL,*dup=NULL,*ddn=NULL,*dscr=NULL;
    float *dipl=NULL,*dple=NULL,*dpg=NULL,*dpp=NULL,*dlog=NULL;   /* ETA.4 AltUp + head */
    float *hple = NULL;
    float **Kst=NULL, **Vst=NULL;     /* per-OWNER device K/V (shared-KV reuse) */
    int rc = 1;
    Kst = (float **)calloc((size_t)NL, sizeof(float *));
    Vst = (float **)calloc((size_t)NL, sizeof(float *));
    if (!Kst || !Vst) { sp_set_error("g4 probe: host OOM"); goto done; }
    #define G4A(p, cnt) do { if (cudaMalloc(&(p), (size_t)(cnt)*sizeof(float)) != cudaSuccess) { \
        sp_set_error("g4 probe OOM"); goto done; } } while (0)
    if (cudaMalloc(&dtoks, (size_t)n_tok*sizeof(int)) != cudaSuccess) { sp_set_error("g4 dtoks OOM"); goto done; }
    G4A(dx, (size_t)n_tok*E);     G4A(dnx, (size_t)n_tok*E);
    G4A(dq, (size_t)n_tok*QDmax); G4A(dk, (size_t)n_tok*KVDmax); G4A(dv, (size_t)n_tok*KVDmax);
    G4A(dao, (size_t)n_tok*QDmax); G4A(dap, (size_t)n_tok*E);
    G4A(dg, (size_t)n_tok*FFmax); G4A(dup, (size_t)n_tok*FFmax); G4A(ddn, (size_t)n_tok*E);
    if (g_w.scratch_n) G4A(dscr, g_w.scratch_n);
    if (full && PL) {                              /* AltUp state: persists ALL layers */
        G4A(dipl, (size_t)n_tok*NL*PL); G4A(dple, (size_t)n_tok*NL*PL);
        G4A(dpg, (size_t)n_tok*PL);     G4A(dpp, (size_t)n_tok*E);
    }
    if (full) G4A(dlog, (size_t)n_tok*V);
    #undef G4A

    if (cudaMemcpyAsync(dtoks, tokens, (size_t)n_tok*sizeof(int), cudaMemcpyHostToDevice, st) != cudaSuccess) {
        sp_set_error("g4 upload tokens"); goto done;
    }
    if (g_w.embd) {
        dim3 grid(n_tok, (E + 255) / 256);   /* embed + sqrt(E) scale (gemma) */
        k_embed_scale<<<grid, 256, 0, st>>>(g_w.embd, dtoks, n_tok, E, embscale, dx);
    } else {
        /* ETA.5b.4: f32 embd not resident (VRAM budget — 12B). Host-gather the
         * n_tok rows (sp_arena_dequant_row, identical arithmetic), scale, upload. */
        const sp_arena_tensor *eat = m->arena ? sp_arena_find(m->arena, m->token_embd->name) : NULL;
        float *hrows = eat ? (float *)malloc((size_t)n_tok * E * sizeof(float)) : NULL;
        if (!hrows) { sp_set_error("g4 probe: no f32 embd and no arena gather"); goto done; }
        for (int t = 0; t < n_tok; t++) {
            if (sp_arena_dequant_row(eat, tokens[t], hrows + (size_t)t * E)) {
                free(hrows); sp_set_error("g4 probe: embd row gather"); goto done; }
            for (int i = 0; i < E; i++) hrows[(size_t)t * E + i] *= embscale;
        }
        int up = (cudaMemcpyAsync(dx, hrows, (size_t)n_tok * E * sizeof(float),
                                  cudaMemcpyHostToDevice, st) != cudaSuccess);
        cudaStreamSynchronize(st);   /* hrows freed next — make the copy land first */
        free(hrows);
        if (up) { sp_set_error("g4 probe: embd rows H2D"); goto done; }
    }

    /* ── ETA.4: AltUp PRECOMPUTE (gemma4.c project_per_layer_inputs) — runs ONCE
     * on the post-embed stream, BEFORE layer 0; dipl persists the whole traversal.
     * PLE rows are host-gathered per token from the arena (the CPU sp_weight_row
     * mirror), uploaded, then fused with per_layer_model_proj·x on device. ── */
    if (full && PL) {
        const sp_arena_tensor *plt = m->arena ? sp_arena_find(m->arena, "per_layer_token_embd.weight") : NULL;
        if (!plt) { sp_set_error("g4: per_layer_token_embd not in arena"); goto done; }
        hple = (float *)malloc((size_t)n_tok * NL * PL * sizeof(float));
        if (!hple) { sp_set_error("g4 hple OOM"); goto done; }
        for (int t = 0; t < n_tok; t++)
            if (sp_arena_dequant_row(plt, tokens[t], hple + (size_t)t * NL * PL)) {
                sp_set_error("g4 ple row dequant"); goto done;
            }
        if (cudaMemcpyAsync(dple, hple, (size_t)n_tok*NL*PL*sizeof(float), cudaMemcpyHostToDevice, st) != cudaSuccess) {
            sp_set_error("g4 ple H2D"); goto done;
        }
        /* proj = per_layer_model_proj · x  [n_tok x NL*PL], into dipl (CPU reuses
         * ipl as the proj scratch — same here), then the fusion in place. */
        if (gemm_w_lift(cb, st, &g_w.pl_model_proj, dx, dipl, n_tok, dscr)) goto done;
        k_altup_ipl<<<n_tok*NL, 256, 0, st>>>(dipl, dple, g_w.pl_proj_norm, PL,
                                              1.0f / sqrtf((float)E), sqrtf((float)PL),
                                              1.0f / sqrtf(2.0f), eps);
    }

    for (int L = 0; L < n_layers; L++) {
        const int global = ((L % period) == period - 1);
        const int nh  = global ? g_nh  : s_nh;
        const int nkv = global ? g_nkv : s_nkv;
        const int hd  = global ? g_hd  : s_hd;
        const int grp = nh / nkv, qd = nh * hd, kvd = nkv * hd;
        const float rbase = global ? g_base : s_base;
        const float *ffac = global ? g_w.rope_freqs : NULL;  /* proportional factors */
        const int win = global ? -1 : SW;
        const float ascale = 1.0f;                            /* Gemma4: scaling = 1.0 */
        const size_t nE = (size_t)n_tok * E;

        /* ── attention ── */
        k_rmsnorm<<<n_tok, 256, 0, st>>>(dx, g_w.attn_norm[L], E, eps, dnx);
        if (attn_only == 2 && L == n_layers - 1) {            /* ── bisect: post-attn_norm nx ── */
            if (cudaMemcpyAsync(out_x, dnx, (size_t)n_tok*E*sizeof(float), cudaMemcpyDeviceToHost, st) != cudaSuccess) {
                sp_set_error("g4 dl nx"); goto done; }
            { cudaError_t e = cudaStreamSynchronize(st); if (e != cudaSuccess) { fail_cuda(e, "g4 sync"); goto done; } }
            rc = 0; goto done;
        }
        if (gemm_w_lift(cb, st, &g_w.Wq[L], dnx, dq, n_tok, dscr)) goto done;
        k_rmsnorm_head<<<n_tok*nh, 256, 0, st>>>(dq, g_w.q_norm[L], nh, hd, qd, eps);
        if (ffac) k_rope_freqs<<<n_tok*nh, hd/2, 0, st>>>(dq, nh, hd, qd, rbase, ffac);
        else      k_rope<<<n_tok*nh, hd/2, 0, st>>>(dq, nh, hd, qd, rbase);
        if (attn_only == 3 && L == n_layers - 1) {            /* ── bisect: q post norm+rope [n_tok*qd] ── */
            if (cudaMemcpyAsync(out_x, dq, (size_t)n_tok*qd*sizeof(float), cudaMemcpyDeviceToHost, st) != cudaSuccess) {
                sp_set_error("g4 dl q"); goto done; }
            { cudaError_t e = cudaStreamSynchronize(st); if (e != cudaSuccess) { fail_cuda(e, "g4 sync"); goto done; } }
            rc = 0; goto done;
        }

        float *Kuse, *Vuse;
        if (L < kvfs) {                                       /* OWNER: project + store */
            if (gemm_w_lift(cb, st, &g_w.Wk[L], dnx, dk, n_tok, dscr)) goto done;
            if (g_w.Wv[L].f32 || g_w.Wv[L].codes) {
                if (gemm_w_lift(cb, st, &g_w.Wv[L], dnx, dv, n_tok, dscr)) goto done;
            } else {  /* V-less layer: V = the RAW K projection (llama.cpp gemma4-iswa) */
                cudaMemcpyAsync(dv, dk, (size_t)n_tok*kvd*sizeof(float), cudaMemcpyDeviceToDevice, st);
            }
            k_rmsnorm_head<<<n_tok*nkv, 256, 0, st>>>(dk, g_w.k_norm[L], nkv, hd, kvd, eps);
            if (ffac) k_rope_freqs<<<n_tok*nkv, hd/2, 0, st>>>(dk, nkv, hd, kvd, rbase, ffac);
            else      k_rope<<<n_tok*nkv, hd/2, 0, st>>>(dk, nkv, hd, kvd, rbase);
            k_rmsnorm_head_noweight<<<n_tok*nkv, 256, 0, st>>>(dv, nkv, hd, kvd, eps);  /* WEIGHTLESS V-norm */
            if (cudaMalloc(&Kst[L], (size_t)n_tok*kvd*sizeof(float)) != cudaSuccess ||
                cudaMalloc(&Vst[L], (size_t)n_tok*kvd*sizeof(float)) != cudaSuccess) {
                sp_set_error("g4 Kst OOM"); goto done;
            }
            cudaMemcpyAsync(Kst[L], dk, (size_t)n_tok*kvd*sizeof(float), cudaMemcpyDeviceToDevice, st);
            cudaMemcpyAsync(Vst[L], dv, (size_t)n_tok*kvd*sizeof(float), cudaMemcpyDeviceToDevice, st);
            Kuse = Kst[L]; Vuse = Vst[L];
        } else {                                              /* SHARER: reuse owner, skip proj */
            const int src = kvfs - (global ? 1 : 2);          /* type matches by construction */
            Kuse = Kst[src]; Vuse = Vst[src];
            if (!Kuse || !Vuse) { sp_set_error("g4 sharer before owner"); goto done; }
        }

        {   int bd = hd > n_tok ? hd : n_tok; if (bd > 1024) bd = 1024;
            k_attn<<<n_tok*nh, bd, (size_t)n_tok*sizeof(float), st>>>(
                dq, Kuse, Vuse, n_tok, qd, kvd, hd, grp, ascale, win, dao); }
        if (attn_only == 4 && L == n_layers - 1) {            /* ── bisect: ao post-attention [n_tok*qd] ── */
            if (cudaMemcpyAsync(out_x, dao, (size_t)n_tok*qd*sizeof(float), cudaMemcpyDeviceToHost, st) != cudaSuccess) {
                sp_set_error("g4 dl ao"); goto done; }
            { cudaError_t e = cudaStreamSynchronize(st); if (e != cudaSuccess) { fail_cuda(e, "g4 sync"); goto done; } }
            rc = 0; goto done;
        }
        if (gemm_w_lift(cb, st, &g_w.Wo[L], dao, dap, n_tok, dscr)) goto done;
        if (attn_only == 5 && L == n_layers - 1) {            /* ── bisect: ap post-Wo PRE-norm [n_tok*E] ── */
            if (cudaMemcpyAsync(out_x, dap, (size_t)n_tok*E*sizeof(float), cudaMemcpyDeviceToHost, st) != cudaSuccess) {
                sp_set_error("g4 dl ap"); goto done; }
            { cudaError_t e = cudaStreamSynchronize(st); if (e != cudaSuccess) { fail_cuda(e, "g4 sync"); goto done; } }
            rc = 0; goto done;
        }
        k_rmsnorm<<<n_tok, 256, 0, st>>>(dap, g_w.post_attn[L], E, eps, dnx);
        k_add<<<(unsigned)((nE+255)/256), 256, 0, st>>>(dx, dnx, nE);

        if (attn_only == 1 && L == n_layers - 1) break;       /* ── probe boundary A ── */

        /* ── FFN (GeGLU, per-layer elastic width) + post_ffw residual ── */
        const int ffL = g_w.Wgate[L].out;
        k_rmsnorm<<<n_tok, 256, 0, st>>>(dx, g_w.ffn_norm[L], E, eps, dnx);
        if (gemm_w_lift(cb, st, &g_w.Wgate[L], dnx, dg, n_tok, dscr)) goto done;
        if (gemm_w_lift(cb, st, &g_w.Wup[L], dnx, dup, n_tok, dscr)) goto done;
        {   size_t nFF = (size_t)n_tok * ffL;
            k_gelu_mul<<<(unsigned)((nFF+255)/256), 256, 0, st>>>(dg, dup, nFF); }
        if (gemm_w_lift(cb, st, &g_w.Wdown[L], dg, ddn, n_tok, dscr)) goto done;
        k_rmsnorm<<<n_tok, 256, 0, st>>>(ddn, g_w.post_ffw[L], E, eps, dnx);
        k_add<<<(unsigned)((nE+255)/256), 256, 0, st>>>(dx, dnx, nE);

        /* ── ETA.4: AltUp per-layer-input INJECTION (gemma4.c 220-234) — its own
         * sandwich block AFTER the FFN residual: inp_gate·x -> gelu·ipl(t,L) ->
         * proj -> per_layer_post_norm -> residual. Then the scalar out_scale. ── */
        if (full && PL) {
            if (gemm_w_lift(cb, st, &g_w.Wplig[L], dx, dpg, n_tok, dscr)) goto done;
            {   int n = n_tok * PL;
                k_altup_gate<<<(unsigned)((n+255)/256), 256, 0, st>>>(dpg, dipl, L, NL, PL, n_tok); }
            if (gemm_w_lift(cb, st, &g_w.Wplproj[L], dpg, dpp, n_tok, dscr)) goto done;
            k_rmsnorm<<<n_tok, 256, 0, st>>>(dpp, g_w.pl_post_norm[L], E, eps, dnx);
            k_add<<<(unsigned)((nE+255)/256), 256, 0, st>>>(dx, dnx, nE);
        }
        if (full && g_w.pl_out_scale && g_w.pl_out_scale[L])
            k_scale_by_dev<<<(unsigned)((nE+255)/256), 256, 0, st>>>(dx, nE, g_w.pl_out_scale[L]);
    }

    /* ── ETA.4: final norm + (tied) head + logit softcap (gemma4.c 243-249) ── */
    if (full) {
        k_rmsnorm<<<n_tok, 256, 0, st>>>(dx, g_w.out_norm, E, eps, dnx);
        if (g_w.head.f32 || g_w.head.codes) {            /* untied head */
            if (gemm_w_lift(cb, st, &g_w.head, dnx, dlog, n_tok, dscr)) goto done;
        } else {                                          /* tied: reuse the f32 embedding */
            if (!g_w.embd) {   /* 12B-class: f32 embd not resident — FULL mode unavailable */
                sp_set_error("g4 probe: FULL head needs the f32 embd (VRAM budget keeps only packed codes; use the decode path)");
                goto done;
            }
            if (gemm(cb, g_w.embd, dnx, dlog, n_tok, E, V)) goto done;
        }
        if (softcap > 0.0f) {
            size_t nl = (size_t)n_tok * V;
            k_softcap<<<(unsigned)((nl+255)/256), 256, 0, st>>>(dlog, nl, softcap);
        }
        if (cudaMemcpyAsync(out_x, dlog, (size_t)n_tok*V*sizeof(float), cudaMemcpyDeviceToHost, st) != cudaSuccess) {
            sp_set_error("g4 download logits"); goto done;
        }
    } else if (cudaMemcpyAsync(out_x, dx, (size_t)n_tok*E*sizeof(float), cudaMemcpyDeviceToHost, st) != cudaSuccess) {
        sp_set_error("g4 download x"); goto done;
    }
    { cudaError_t e = cudaStreamSynchronize(st); if (e != cudaSuccess) { fail_cuda(e, "g4 sync"); goto done; } }
    { cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) { fail_cuda(e, "g4 kernel"); goto done; } }
    rc = 0;
done:
    if (dtoks) cudaFree(dtoks);
    if (dx) cudaFree(dx); if (dnx) cudaFree(dnx); if (dq) cudaFree(dq);
    if (dk) cudaFree(dk); if (dv) cudaFree(dv); if (dao) cudaFree(dao); if (dap) cudaFree(dap);
    if (dg) cudaFree(dg); if (dup) cudaFree(dup); if (ddn) cudaFree(ddn); if (dscr) cudaFree(dscr);
    if (dipl) cudaFree(dipl); if (dple) cudaFree(dple); if (dpg) cudaFree(dpg);
    if (dpp) cudaFree(dpp); if (dlog) cudaFree(dlog); free(hple);
    if (Kst) { for (int L = 0; L < NL; L++) if (Kst[L]) cudaFree(Kst[L]); free(Kst); }
    if (Vst) { for (int L = 0; L < NL; L++) if (Vst[L]) cudaFree(Vst[L]); free(Vst); }
    return rc;
}

/* ETA.4: the official Gemma4 CUDA prefill — the full 35-layer forward (per-layer
 * geometry + shared-KV + proportional RoPE + AltUp + out_scale + tied head +
 * logit softcap). logits = [n_tok x n_vocab]. Gated argmax+KL vs the CPU oracle
 * gemma4_forward (E_G4_CU_FULL). */
extern "C" int gemma4_forward_cuda(const qwen3_model *m, const int32_t *tokens,
                                   int n_tok, float *logits) {
    return gemma4_cuda_probe(m, tokens, n_tok, m ? (int)m->cfg.n_layers : 0, -1, logits);
}

__global__ void k_rope_at(float *base, int n_heads, int d, int rowstride,
                          float rbase, int p0);            /* defined below (BETA.2) */
__global__ void k_argmax(const float *logits, int n, int *out_tok);
/* BETA.2 position-indirect kernels reused by the ETA.5b graph path (defined below). */
__global__ void k_rope_dyn(float *base, int n_heads, int d, int rowstride,
                           float rbase, const int *dpos);
__global__ void k_kv_store(float *Kc, float *Vc, const float *dk, const float *dv,
                           const int *dpos, size_t layer_off, int KVD);
__global__ void k_argmax_at(const float *logits, int n, int *dseq, const int *dpos);
__global__ void k_incr_pos(int *dpos);

/* ═══════════ ETA.5b: device-fed Gemma4 decode kernels ═══════════
 * Token identity and position live in VRAM (dseq[*dpos]); these kernels read
 * them there so a generate step has ZERO host round-trips and is graph-capturable. */

/* embed + sqrt(E) scale of the token at dseq[*dpos] (k_embed_at + gemma scale). */
__global__ void k_embed_scale_at(const float *embd, const int *dseq, const int *dpos,
                                 int E, float scale, float *x) {
    int i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i < E) { int tok = dseq[*dpos]; x[i] = embd[(size_t)tok * E + i] * scale; }
}

/* gather + dequant ONE packed PLE row — the token at dseq[*dpos] — into out[cols]:
 * the device twin of the host sp_arena_dequant_row gather. ARITHMETIC NOTE: the
 * host (sp_frob_packed_dequant_row) computes inv = scale / qmax with a TRUE
 * division — mirror that exactly (NOT scale * (1/qmax), which can differ by an
 * ulp); this feeds the byte-match oracle gate. grid = ceil(cols/256). */
__global__ void k_ple_gather_at(const unsigned char *codes, const unsigned long long *row_off,
                                const float *row_scale, const unsigned char *row_prec,
                                const int *dseq, const int *dpos, int cols, float *out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cols) return;
    int tok = dseq[*dpos];
    const unsigned char *rc = codes + row_off[tok];
    int code; float inv;
    if (row_prec[tok] == 8) {
        code = (int)((const signed char *)rc)[i];
        inv = row_scale[tok] / 127.0f;
    } else {
        unsigned char byte = rc[i >> 1];
        int nib = (i & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);
        code = (nib ^ 8) - 8;
        inv = row_scale[tok] / 7.0f;
    }
    out[i] = (float)code * inv;
}

/* embed gather straight from PACKED codes (the token at dseq[*dpos]) + scale:
 * the f32-embd-free embed path for models whose dequanted embedding would blow
 * the VRAM budget (12B: V x E f32 ~= 4 GB; the packed codes are ~1 GB). Same
 * host-mirror arithmetic as k_ple_gather_at (TRUE division — byte-match). */
__global__ void k_embed_packed_at(const unsigned char *codes, const unsigned long long *row_off,
                                  const float *row_scale, const unsigned char *row_prec,
                                  const int *dseq, const int *dpos, int E, float scale, float *x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= E) return;
    int tok = dseq[*dpos];
    const unsigned char *rc = codes + row_off[tok];
    int code; float inv;
    if (row_prec[tok] == 8) {
        code = (int)((const signed char *)rc)[i];
        inv = row_scale[tok] / 127.0f;
    } else {
        unsigned char byte = rc[i >> 1];
        int nib = (i & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);
        code = (nib ^ 8) - 8;
        inv = row_scale[tok] / 7.0f;
    }
    x[i] = (float)code * inv * scale;
}

/* proportional-freq NEOX RoPE at the DEVICE position (k_rope_freqs_at + *dpos). */
__global__ void k_rope_freqs_dyn(float *base, int n_heads, int d, float rbase,
                                 const float *ff, const int *dpos) {
    int h = blockIdx.x, i = threadIdx.x, half = d / 2;
    if (i < half) {
        float *v = base + (size_t)h * d;
        float freq = powf(rbase, -2.0f * (float)i / (float)d) / ff[i];
        float th = (float)(*dpos) * freq, c = cosf(th), s = sinf(th);
        float a = v[i], bb = v[i + half];
        v[i] = a * c - bb * s;
        v[i + half] = a * s + bb * c;
    }
    (void)n_heads;
}

/* windowed single-query GQA at the DEVICE position (k_attn_decode_win + *dpos).
 * ctx = *dpos+1; s0 = max(0, pos-win+1). Shared mem is FIXED at capture (P floats)
 * — slots below s0 / past ctx are simply untouched. */
__global__ void k_attn_decode_win_dyn(const float *q, const float *Kc, const float *Vc,
                                      const int *dpos, int KVD, int HD, int group,
                                      float ascale, int win, float *ao) {
    extern __shared__ float sc[];
    int ctx = *dpos + 1;
    int h = blockIdx.x, kvh = h / group;
    int pos = ctx - 1;
    int s0 = (win >= 0 && pos - win + 1 > 0) ? pos - win + 1 : 0;
    const float *qh = q + (size_t)h * HD;
    for (int s = s0 + threadIdx.x; s < ctx; s += blockDim.x) {
        const float *kh = Kc + (size_t)s * KVD + (size_t)kvh * HD;
        float acc = 0.0f;
        for (int i = 0; i < HD; i++) acc += qh[i] * kh[i];
        sc[s] = acc * ascale;
    }
    __syncthreads();
    __shared__ float g_sum;
    if (threadIdx.x == 0) {
        float mx = sc[s0];
        for (int s = s0 + 1; s < ctx; s++) if (sc[s] > mx) mx = sc[s];
        float sum = 0.0f;
        for (int s = s0; s < ctx; s++) { float e = expf(sc[s] - mx); sc[s] = e; sum += e; }
        g_sum = sum;
    }
    __syncthreads();
    float inv = 1.0f / g_sum;
    for (int i = threadIdx.x; i < HD; i += blockDim.x) {
        float acc = 0.0f;
        for (int s = s0; s < ctx; s++)
            acc += sc[s] * Vc[(size_t)s * KVD + (size_t)kvh * HD + i];
        ao[(size_t)h * HD + i] = acc * inv;
    }
}

/* ═══════════ ETA.5a/5b: Gemma4 autoregressive CUDA decode ═══════════
 * ETA.5a (the oracle gate, default knobs-off): greedy argmax generation with the
 * FULL Gemma4 stack per step (per-layer geometry, JAGGED shared-KV cache,
 * proportional RoPE at the absolute position, windowed single-query SWA
 * attention, AltUp, out_scale, tied head + softcap), all matmuls on the ORACLE
 * arithmetic (gemm_w_lift). Gated E_G4_CU_DEC: the CPU oracle teacher-forced
 * predicts every generated token.
 *
 * JAGGED KV CACHE: per-OWNER device buffers [P x kvd_L] — global owners write
 * 512-wide rows, SWA owners 256-wide, sharers allocate NOTHING and read their
 * owner's buffer (kvfs-1 global / kvfs-2 SWA). No uniform grid, no padding.
 *
 * ETA.5b (the velocity pass; each lever env-gated, default OFF):
 *   - DEVICE-FED steps: dseq/dpos live in VRAM; k_embed_scale_at /
 *     k_ple_gather_at (the packed PLE table, g_w.pl_tok_embd) / k_argmax_at
 *     read them there — ZERO per-token D2H when eos_id < 0 (one sequence
 *     download at the end). Host PLE row-gather remains the fallback when the
 *     table isn't resident.
 *   - SP_CUDA_DECODE_GRAPH=1: the generate step is captured ONCE into a CUDA
 *     graph and replayed per token (BETA.2 machinery). The jagged per-owner
 *     cache POINTERS are fixed per layer, so the capture is stable; position
 *     enters via *dpos (k_rope_dyn / k_rope_freqs_dyn / k_attn_decode_win_dyn /
 *     k_kv_store). Requires the device PLE gather + attn shm = P floats ≤ 48KB.
 *   - SP_CUDA_DECODE_INT8=1: decode matmuls route through gemv_w_packed (Q8/Q4
 *     dp4a, per-tensor precision dispatch — the ~7x byte diet); non-uniform
 *     tensors fall back to gemm_w_lift. Top-1-lossless, NOT byte-exact (the
 *     activation int8 quant) — gate top-1 vs the knobs-off GPU decode. */
extern "C" int gemma4_decode_cuda(const qwen3_model *m, int32_t *seq,
                                  int n_prompt, int n_gen, int eos_id) {
    if (!m || m->cfg.arch != SP_ARCH_GEMMA4) { sp_set_error("gemma4_decode_cuda: not gemma4"); return -1; }
    if (n_prompt <= 0 || n_gen < 0 || !seq) { sp_set_error("gemma4_decode_cuda: bad args"); return -1; }
    const qwen3_config *c = &m->cfg;
    const int E = (int)c->n_embd, NL = (int)c->n_layers, SW = (int)c->sliding_window;
    const int V = (int)c->n_vocab, P = n_prompt + n_gen;
    const float eps = c->rms_eps, embscale = sqrtf((float)E);
    const float softcap = c->g4_logit_softcap;
    const int period = (int)c->g4_swa_period ? (int)c->g4_swa_period : 6;
    const int kvfs   = (int)c->g4_n_kv_from_start ? (int)c->g4_n_kv_from_start : NL;
    const int PL = (int)c->g4_n_embd_per_layer;
    const int g_nh = (int)c->n_head, g_nkv = (int)c->n_head_kv, g_hd = (int)c->head_dim;
    const int s_nh = (int)c->g4_nh_swa, s_nkv = (int)c->g4_nkv_swa, s_hd = (int)c->g4_hd_swa;
    const float g_base = c->rope_freq_base, s_base = c->g4_rope_base_swa;
    const int QDmax = (g_nh*g_hd > s_nh*s_hd) ? g_nh*g_hd : s_nh*s_hd;
    const int KVDmax = (g_nkv*g_hd > s_nkv*s_hd) ? g_nkv*g_hd : s_nkv*s_hd;

    if (g_w.key != m) { free_weights(&g_w); if (build_weights(m, &g_w)) return -1; }
    cublasHandle_t cb = g_w.cublas;
    cudaStream_t st = g_w.stream;
    int FFmax = (int)c->n_ff;
    for (int L = 0; L < NL; L++) if (g_w.Wgate[L].out > FFmax) FFmax = g_w.Wgate[L].out;
    const sp_arena_tensor *plt = (PL && m->arena) ? sp_arena_find(m->arena, "per_layer_token_embd.weight") : NULL;
    if (PL && !plt) { sp_set_error("g4 decode: per_layer_token_embd not in arena"); return -1; }
    /* ETA.5b: device PLE gather when the packed table is resident (build_weights). */
    const int dev_ple = (PL && g_w.pl_tok_embd.codes != NULL);
    const int NLPL = NL * PL;
    /* ETA.5b: packed dp4a GEMV routing (top-1, not byte-exact — see header). */
    const char *i8e = getenv("SP_CUDA_DECODE_INT8");
    const int use_int8 = (i8e && i8e[0] == '1') && m->arena != NULL;
    /* 12B-class tied head: the f32 embd isn't resident (VRAM budget) — the
     * V x E head matmul must take the dp4a route. Surface it, don't crash. */
    if (!(g_w.head.f32 || g_w.head.codes) && !g_w.embd &&
        !(use_int8 && g_w.embd_packed.codes)) {
        sp_set_error("g4 decode: tied head without resident f32 embd requires SP_CUDA_DECODE_INT8=1");
        return -1;
    }

    int *dseq=NULL;            /* the whole token sequence, RESIDENT in VRAM */
    int *dpos=NULL;            /* device position scalar (shadows the host pos) */
    float *dx=NULL,*dnx=NULL,*dq=NULL,*dk=NULL,*dv=NULL,*dao=NULL,*dap=NULL,
          *dg=NULL,*dup=NULL,*ddn=NULL,*dscr=NULL,*dlog=NULL,
          *dipl=NULL,*dple=NULL,*dpg=NULL,*dpp=NULL,*dsx=NULL;
    signed char *dqx=NULL;     /* int8 activation-quant scratch (max in-dim, padded) */
    float *hple=NULL; float **dKc=NULL, **dVc=NULL;
    cudaGraph_t cgraph=NULL; cudaGraphExec_t cexec=NULL;
    int rc=-1, n=n_prompt;
    dKc = (float **)calloc((size_t)NL, sizeof(float *));
    dVc = (float **)calloc((size_t)NL, sizeof(float *));
    if (!dKc || !dVc) { sp_set_error("g4 decode host OOM"); goto done; }
    #define G4D(p, cnt) do { if (cudaMalloc(&(p), (size_t)(cnt)*sizeof(float)) != cudaSuccess) { \
        sp_set_error("g4 decode OOM"); goto done; } } while (0)
    if (cudaMalloc(&dseq, (size_t)P*sizeof(int)) != cudaSuccess ||
        cudaMalloc(&dpos, sizeof(int)) != cudaSuccess) { sp_set_error("g4 dseq OOM"); goto done; }
    G4D(dx,E); G4D(dnx,E); G4D(dq,QDmax); G4D(dk,KVDmax); G4D(dv,KVDmax);
    G4D(dao,QDmax); G4D(dap,E); G4D(dg,FFmax); G4D(dup,FFmax); G4D(ddn,E); G4D(dlog,V);
    if (g_w.scratch_n) G4D(dscr, g_w.scratch_n);
    if (PL) { G4D(dipl,(size_t)NLPL); G4D(dple,(size_t)NLPL); G4D(dpg,PL); G4D(dpp,E);
              if (!dev_ple) {
                  hple = (float *)malloc((size_t)NLPL*sizeof(float));
                  if (!hple) { sp_set_error("g4 hple OOM"); goto done; } } }
    if (use_int8) {            /* qx sized to the widest matmul input, padded %32 (Q4 chunk) */
        int maxin = E; if (QDmax > maxin) maxin = QDmax; if (FFmax > maxin) maxin = FFmax;
        if (cudaMalloc(&dqx, (size_t)((maxin+31)&~31)) != cudaSuccess) { sp_set_error("g4 dqx OOM"); goto done; }
        G4D(dsx,1);
    }
    /* the JAGGED cache: owners only, per-layer width */
    for (int L = 0; L < kvfs && L < NL; L++) {
        const int global = ((L % period) == period - 1);
        const int kvd = (global ? g_nkv : s_nkv) * (global ? g_hd : s_hd);
        G4D(dKc[L], (size_t)P * kvd);
        G4D(dVc[L], (size_t)P * kvd);
    }
    #undef G4D
    /* prompt into VRAM once; dpos = 0 */
    if (cudaMemcpyAsync(dseq, seq, (size_t)n_prompt*sizeof(int), cudaMemcpyHostToDevice, st) != cudaSuccess) {
        sp_set_error("g4 prompt H2D"); goto done; }
    { int z = 0; if (cudaMemcpyAsync(dpos, &z, sizeof(int), cudaMemcpyHostToDevice, st) != cudaSuccess) {
        sp_set_error("g4 dpos seed"); goto done; } }

    /* MMD: decode matmul — packed dp4a GEMV when routed, else the ORACLE lift. */
    #define MMD(W,X,Y) do { if (!(use_int8 && gemv_w_packed(st,(W),(X),(Y),dqx,dsx))) { \
        if (gemm_w_lift(cb, st, (W), (X), (Y), 1, dscr)) goto done; } } while (0)

    /* ───────── ETA.5b: CUDA-graph generate path ─────────
     * Prompt ingest stays per-step; the GENERATE step is captured once and
     * replayed n_gen times. Needs: device PLE gather (no host work in the
     * step) + fixed attn shm (P floats) within the 48KB sm_75 default. */
    {
        const char *ge = getenv("SP_CUDA_DECODE_GRAPH");
        const int use_graph = (ge && ge[0] == '1');
        const size_t attn_shm = (size_t)P * sizeof(float);
        if (use_graph && attn_shm <= 48u*1024u && n_gen > 0 && (!PL || dev_ple)) {
            cublasSetStream(cb, st);
            /* ── prompt ingest [0, n_prompt-1): fills the jagged cache, no head ── */
            for (int pos = 0; pos < n_prompt - 1; pos++) {
                if (g_w.embd) { dim3 grid(1, (E + 255) / 256);
                  k_embed_scale<<<grid, 256, 0, st>>>(g_w.embd, dseq + pos, 1, E, embscale, dx); }
                else k_embed_packed_at<<<(unsigned)((E+255)/256), 256, 0, st>>>(
                    g_w.embd_packed.codes, g_w.embd_packed.row_off, g_w.embd_packed.row_scale,
                    g_w.embd_packed.row_prec, dseq, dpos, E, embscale, dx);
                if (PL) {
                    k_ple_gather_at<<<(unsigned)((NLPL+255)/256), 256, 0, st>>>(
                        g_w.pl_tok_embd.codes, g_w.pl_tok_embd.row_off,
                        g_w.pl_tok_embd.row_scale, g_w.pl_tok_embd.row_prec,
                        dseq, dpos, NLPL, dple);
                    MMD(&g_w.pl_model_proj, dx, dipl);
                    k_altup_ipl<<<NL, 256, 0, st>>>(dipl, dple, g_w.pl_proj_norm, PL,
                                                    1.0f / sqrtf((float)E), sqrtf((float)PL),
                                                    1.0f / sqrtf(2.0f), eps);
                }
                for (int L = 0; L < NL; L++) {
                    const int global = ((L % period) == period - 1);
                    const int nh = global ? g_nh : s_nh, nkv = global ? g_nkv : s_nkv;
                    const int hd = global ? g_hd : s_hd;
                    const int grp = nh / nkv, qd = nh * hd, kvd = nkv * hd;
                    const float rbase = global ? g_base : s_base;
                    const float *ffac = global ? g_w.rope_freqs : NULL;
                    const int win = global ? -1 : SW;
                    const int ffL = g_w.Wgate[L].out;
                    k_rmsnorm<<<1, 256, 0, st>>>(dx, g_w.attn_norm[L], E, eps, dnx);
                    MMD(&g_w.Wq[L], dnx, dq);
                    k_rmsnorm_head<<<nh, 256, 0, st>>>(dq, g_w.q_norm[L], nh, hd, qd, eps);
                    if (ffac) k_rope_freqs_at<<<nh, hd/2, 0, st>>>(dq, nh, hd, rbase, ffac, pos);
                    else      k_rope_at<<<nh, hd/2, 0, st>>>(dq, nh, hd, qd, rbase, pos);
                    float *Kuse, *Vuse;
                    if (L < kvfs) {
                        MMD(&g_w.Wk[L], dnx, dk);
                        if (g_w.Wv[L].f32 || g_w.Wv[L].codes) { MMD(&g_w.Wv[L], dnx, dv); }
                        else cudaMemcpyAsync(dv, dk, (size_t)kvd*sizeof(float), cudaMemcpyDeviceToDevice, st); /* V-less: raw K proj */
                        k_rmsnorm_head<<<nkv, 256, 0, st>>>(dk, g_w.k_norm[L], nkv, hd, kvd, eps);
                        if (ffac) k_rope_freqs_at<<<nkv, hd/2, 0, st>>>(dk, nkv, hd, rbase, ffac, pos);
                        else      k_rope_at<<<nkv, hd/2, 0, st>>>(dk, nkv, hd, kvd, rbase, pos);
                        k_rmsnorm_head_noweight<<<nkv, 256, 0, st>>>(dv, nkv, hd, kvd, eps);
                        cudaMemcpyAsync(dKc[L] + (size_t)pos*kvd, dk, (size_t)kvd*sizeof(float), cudaMemcpyDeviceToDevice, st);
                        cudaMemcpyAsync(dVc[L] + (size_t)pos*kvd, dv, (size_t)kvd*sizeof(float), cudaMemcpyDeviceToDevice, st);
                        Kuse = dKc[L]; Vuse = dVc[L];
                    } else {
                        const int src = kvfs - (global ? 1 : 2);
                        Kuse = dKc[src]; Vuse = dVc[src];
                        if (!Kuse || !Vuse) { sp_set_error("g4 sharer before owner"); goto done; }
                    }
                    {   const int ctx = pos + 1;
                        int bd = hd > ctx ? hd : ctx; if (bd > 1024) bd = 1024;
                        k_attn_decode_win<<<nh, bd, (size_t)ctx*sizeof(float), st>>>(
                            dq, Kuse, Vuse, ctx, kvd, hd, grp, 1.0f, win, dao); }
                    MMD(&g_w.Wo[L], dao, dap);
                    k_rmsnorm<<<1, 256, 0, st>>>(dap, g_w.post_attn[L], E, eps, dnx);
                    k_add<<<(unsigned)((E+255)/256), 256, 0, st>>>(dx, dnx, (size_t)E);
                    k_rmsnorm<<<1, 256, 0, st>>>(dx, g_w.ffn_norm[L], E, eps, dnx);
                    MMD(&g_w.Wgate[L], dnx, dg);
                    MMD(&g_w.Wup[L], dnx, dup);
                    k_gelu_mul<<<(unsigned)(((size_t)ffL+255)/256), 256, 0, st>>>(dg, dup, (size_t)ffL);
                    MMD(&g_w.Wdown[L], dg, ddn);
                    k_rmsnorm<<<1, 256, 0, st>>>(ddn, g_w.post_ffw[L], E, eps, dnx);
                    k_add<<<(unsigned)((E+255)/256), 256, 0, st>>>(dx, dnx, (size_t)E);
                    if (PL) {
                        MMD(&g_w.Wplig[L], dx, dpg);
                        k_altup_gate<<<(unsigned)((PL+255)/256), 256, 0, st>>>(dpg, dipl, L, NL, PL, 1);
                        MMD(&g_w.Wplproj[L], dpg, dpp);
                        k_rmsnorm<<<1, 256, 0, st>>>(dpp, g_w.pl_post_norm[L], E, eps, dnx);
                        k_add<<<(unsigned)((E+255)/256), 256, 0, st>>>(dx, dnx, (size_t)E);
                    }
                    if (g_w.pl_out_scale && g_w.pl_out_scale[L])
                        k_scale_by_dev<<<(unsigned)((E+255)/256), 256, 0, st>>>(dx, (size_t)E, g_w.pl_out_scale[L]);
                }
                k_incr_pos<<<1, 1, 0, st>>>(dpos);   /* keep dpos == pos+1 */
            }
            /* ── capture ONE generate step (dpos = n_prompt-1 after ingest) ── */
            if (cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
                sp_set_error("g4 begin capture"); goto done; }
            if (g_w.embd) { dim3 grid(1, (E + 255) / 256);
              k_embed_scale_at<<<grid, 256, 0, st>>>(g_w.embd, dseq, dpos, E, embscale, dx); }
            else k_embed_packed_at<<<(unsigned)((E+255)/256), 256, 0, st>>>(
                g_w.embd_packed.codes, g_w.embd_packed.row_off, g_w.embd_packed.row_scale,
                g_w.embd_packed.row_prec, dseq, dpos, E, embscale, dx);
            if (PL) {
                k_ple_gather_at<<<(unsigned)((NLPL+255)/256), 256, 0, st>>>(
                    g_w.pl_tok_embd.codes, g_w.pl_tok_embd.row_off,
                    g_w.pl_tok_embd.row_scale, g_w.pl_tok_embd.row_prec,
                    dseq, dpos, NLPL, dple);
                MMD(&g_w.pl_model_proj, dx, dipl);
                k_altup_ipl<<<NL, 256, 0, st>>>(dipl, dple, g_w.pl_proj_norm, PL,
                                                1.0f / sqrtf((float)E), sqrtf((float)PL),
                                                1.0f / sqrtf(2.0f), eps);
            }
            for (int L = 0; L < NL; L++) {
                const int global = ((L % period) == period - 1);
                const int nh = global ? g_nh : s_nh, nkv = global ? g_nkv : s_nkv;
                const int hd = global ? g_hd : s_hd;
                const int grp = nh / nkv, qd = nh * hd, kvd = nkv * hd;
                const float rbase = global ? g_base : s_base;
                const float *ffac = global ? g_w.rope_freqs : NULL;
                const int win = global ? -1 : SW;
                const int ffL = g_w.Wgate[L].out;
                k_rmsnorm<<<1, 256, 0, st>>>(dx, g_w.attn_norm[L], E, eps, dnx);
                MMD(&g_w.Wq[L], dnx, dq);
                k_rmsnorm_head<<<nh, 256, 0, st>>>(dq, g_w.q_norm[L], nh, hd, qd, eps);
                if (ffac) k_rope_freqs_dyn<<<nh, hd/2, 0, st>>>(dq, nh, hd, rbase, ffac, dpos);
                else      k_rope_dyn<<<nh, hd/2, 0, st>>>(dq, nh, hd, qd, rbase, dpos);
                float *Kuse, *Vuse;
                if (L < kvfs) {
                    MMD(&g_w.Wk[L], dnx, dk);
                    if (g_w.Wv[L].f32 || g_w.Wv[L].codes) { MMD(&g_w.Wv[L], dnx, dv); }
                    else cudaMemcpyAsync(dv, dk, (size_t)kvd*sizeof(float), cudaMemcpyDeviceToDevice, st); /* V-less: raw K proj */
                    k_rmsnorm_head<<<nkv, 256, 0, st>>>(dk, g_w.k_norm[L], nkv, hd, kvd, eps);
                    if (ffac) k_rope_freqs_dyn<<<nkv, hd/2, 0, st>>>(dk, nkv, hd, rbase, ffac, dpos);
                    else      k_rope_dyn<<<nkv, hd/2, 0, st>>>(dk, nkv, hd, kvd, rbase, dpos);
                    k_rmsnorm_head_noweight<<<nkv, 256, 0, st>>>(dv, nkv, hd, kvd, eps);
                    k_kv_store<<<(unsigned)((kvd+255)/256), 256, 0, st>>>(dKc[L], dVc[L], dk, dv, dpos, 0, kvd);
                    Kuse = dKc[L]; Vuse = dVc[L];
                } else {
                    const int src = kvfs - (global ? 1 : 2);
                    Kuse = dKc[src]; Vuse = dVc[src];
                    if (!Kuse || !Vuse) { sp_set_error("g4 sharer before owner"); goto done; }
                }
                {   int bd = hd > 256 ? hd : 256; if (bd > 1024) bd = 1024;
                    k_attn_decode_win_dyn<<<nh, bd, attn_shm, st>>>(
                        dq, Kuse, Vuse, dpos, kvd, hd, grp, 1.0f, win, dao); }
                MMD(&g_w.Wo[L], dao, dap);
                k_rmsnorm<<<1, 256, 0, st>>>(dap, g_w.post_attn[L], E, eps, dnx);
                k_add<<<(unsigned)((E+255)/256), 256, 0, st>>>(dx, dnx, (size_t)E);
                k_rmsnorm<<<1, 256, 0, st>>>(dx, g_w.ffn_norm[L], E, eps, dnx);
                MMD(&g_w.Wgate[L], dnx, dg);
                MMD(&g_w.Wup[L], dnx, dup);
                k_gelu_mul<<<(unsigned)(((size_t)ffL+255)/256), 256, 0, st>>>(dg, dup, (size_t)ffL);
                MMD(&g_w.Wdown[L], dg, ddn);
                k_rmsnorm<<<1, 256, 0, st>>>(ddn, g_w.post_ffw[L], E, eps, dnx);
                k_add<<<(unsigned)((E+255)/256), 256, 0, st>>>(dx, dnx, (size_t)E);
                if (PL) {
                    MMD(&g_w.Wplig[L], dx, dpg);
                    k_altup_gate<<<(unsigned)((PL+255)/256), 256, 0, st>>>(dpg, dipl, L, NL, PL, 1);
                    MMD(&g_w.Wplproj[L], dpg, dpp);
                    k_rmsnorm<<<1, 256, 0, st>>>(dpp, g_w.pl_post_norm[L], E, eps, dnx);
                    k_add<<<(unsigned)((E+255)/256), 256, 0, st>>>(dx, dnx, (size_t)E);
                }
                if (g_w.pl_out_scale && g_w.pl_out_scale[L])
                    k_scale_by_dev<<<(unsigned)((E+255)/256), 256, 0, st>>>(dx, (size_t)E, g_w.pl_out_scale[L]);
            }
            k_rmsnorm<<<1, 256, 0, st>>>(dx, g_w.out_norm, E, eps, dnx);
            if (g_w.head.f32 || g_w.head.codes) { MMD(&g_w.head, dnx, dlog); }
            else if (use_int8 && g_w.embd_packed.codes &&
                     gemv_w_packed(st, &g_w.embd_packed, dnx, dlog, dqx, dsx)) { /* tied head, dp4a */ }
            else { if (gemm(cb, g_w.embd, dnx, dlog, 1, E, V)) goto done; }
            if (softcap > 0.0f)
                k_softcap<<<(unsigned)(((size_t)V+255)/256), 256, 0, st>>>(dlog, (size_t)V, softcap);
            k_argmax_at<<<1, 256, 0, st>>>(dlog, V, dseq, dpos);   /* writes dseq[*dpos+1] */
            k_incr_pos<<<1, 1, 0, st>>>(dpos);
            if (cudaStreamEndCapture(st, &cgraph) != cudaSuccess) { sp_set_error("g4 end capture"); goto done; }
            if (cudaGraphInstantiate(&cexec, cgraph, NULL, NULL, 0) != cudaSuccess) { sp_set_error("g4 graph instantiate"); goto done; }
            /* ── replay per token ── */
            for (int g = 0; g < n_gen; g++) {
                if (cudaGraphLaunch(cexec, st) != cudaSuccess) { sp_set_error("g4 graph launch"); goto done; }
                n = n_prompt + g + 1;
                if (eos_id >= 0) {
                    int tok = -1;
                    if (cudaMemcpyAsync(&tok, dseq + n - 1, sizeof(int), cudaMemcpyDeviceToHost, st) != cudaSuccess) {
                        sp_set_error("g4 eos D2H"); goto done; }
                    { cudaError_t e = cudaStreamSynchronize(st); if (e != cudaSuccess) { fail_cuda(e, "g4 eos sync"); goto done; } }
                    if (tok == eos_id) break;
                }
            }
            goto download;
        }
    }

    /* ───────── per-step path (the ETA.5a oracle stack, device-fed) ───────── */
    for (int pos = 0; pos < P - 1; pos++) {
        const int gen_here = (pos >= n_prompt - 1);
        if (g_w.embd) { dim3 grid(1, (E + 255) / 256);
          k_embed_scale<<<grid, 256, 0, st>>>(g_w.embd, dseq + pos, 1, E, embscale, dx); }
        else k_embed_packed_at<<<(unsigned)((E+255)/256), 256, 0, st>>>(
            g_w.embd_packed.codes, g_w.embd_packed.row_off, g_w.embd_packed.row_scale,
            g_w.embd_packed.row_prec, dseq, dpos, E, embscale, dx);

        /* per-step AltUp precompute (1 x NL*PL): the PLE row for THIS token —
         * gathered ON DEVICE from the packed table (ETA.5b), or host-gathered
         * via sp_arena_dequant_row when the table isn't resident (ETA.5a). */
        if (PL) {
            if (dev_ple) {
                k_ple_gather_at<<<(unsigned)((NLPL+255)/256), 256, 0, st>>>(
                    g_w.pl_tok_embd.codes, g_w.pl_tok_embd.row_off,
                    g_w.pl_tok_embd.row_scale, g_w.pl_tok_embd.row_prec,
                    dseq, dpos, NLPL, dple);
            } else {
                int32_t tok;
                if (pos < n_prompt) tok = seq[pos];
                else {   /* generated token: fetch it back (host-gather fallback only) */
                    if (cudaMemcpyAsync(&tok, dseq + pos, sizeof(int), cudaMemcpyDeviceToHost, st) != cudaSuccess) {
                        sp_set_error("g4 tok D2H"); goto done; }
                    cudaError_t e = cudaStreamSynchronize(st);
                    if (e != cudaSuccess) { fail_cuda(e, "g4 tok sync"); goto done; }
                }
                if (sp_arena_dequant_row(plt, tok, hple)) { sp_set_error("g4 ple row"); goto done; }
                if (cudaMemcpyAsync(dple, hple, (size_t)NLPL*sizeof(float), cudaMemcpyHostToDevice, st) != cudaSuccess) {
                    sp_set_error("g4 ple H2D"); goto done; }
            }
            MMD(&g_w.pl_model_proj, dx, dipl);
            k_altup_ipl<<<NL, 256, 0, st>>>(dipl, dple, g_w.pl_proj_norm, PL,
                                            1.0f / sqrtf((float)E), sqrtf((float)PL),
                                            1.0f / sqrtf(2.0f), eps);
        }

        for (int L = 0; L < NL; L++) {
            const int global = ((L % period) == period - 1);
            const int nh = global ? g_nh : s_nh, nkv = global ? g_nkv : s_nkv;
            const int hd = global ? g_hd : s_hd;
            const int grp = nh / nkv, qd = nh * hd, kvd = nkv * hd;
            const float rbase = global ? g_base : s_base;
            const float *ffac = global ? g_w.rope_freqs : NULL;
            const int win = global ? -1 : SW;
            const int ffL = g_w.Wgate[L].out;

            k_rmsnorm<<<1, 256, 0, st>>>(dx, g_w.attn_norm[L], E, eps, dnx);
            MMD(&g_w.Wq[L], dnx, dq);
            k_rmsnorm_head<<<nh, 256, 0, st>>>(dq, g_w.q_norm[L], nh, hd, qd, eps);
            if (ffac) k_rope_freqs_at<<<nh, hd/2, 0, st>>>(dq, nh, hd, rbase, ffac, pos);
            else      k_rope_at<<<nh, hd/2, 0, st>>>(dq, nh, hd, qd, rbase, pos);

            float *Kuse, *Vuse;
            if (L < kvfs) {
                MMD(&g_w.Wk[L], dnx, dk);
                if (g_w.Wv[L].f32 || g_w.Wv[L].codes) { MMD(&g_w.Wv[L], dnx, dv); }
                else cudaMemcpyAsync(dv, dk, (size_t)kvd*sizeof(float), cudaMemcpyDeviceToDevice, st); /* V-less: raw K proj */
                k_rmsnorm_head<<<nkv, 256, 0, st>>>(dk, g_w.k_norm[L], nkv, hd, kvd, eps);
                if (ffac) k_rope_freqs_at<<<nkv, hd/2, 0, st>>>(dk, nkv, hd, rbase, ffac, pos);
                else      k_rope_at<<<nkv, hd/2, 0, st>>>(dk, nkv, hd, kvd, rbase, pos);
                k_rmsnorm_head_noweight<<<nkv, 256, 0, st>>>(dv, nkv, hd, kvd, eps);
                cudaMemcpyAsync(dKc[L] + (size_t)pos*kvd, dk, (size_t)kvd*sizeof(float), cudaMemcpyDeviceToDevice, st);
                cudaMemcpyAsync(dVc[L] + (size_t)pos*kvd, dv, (size_t)kvd*sizeof(float), cudaMemcpyDeviceToDevice, st);
                Kuse = dKc[L]; Vuse = dVc[L];
            } else {
                const int src = kvfs - (global ? 1 : 2);
                Kuse = dKc[src]; Vuse = dVc[src];
                if (!Kuse || !Vuse) { sp_set_error("g4 sharer before owner"); goto done; }
            }
            {   const int ctx = pos + 1;
                int bd = hd > ctx ? hd : ctx; if (bd > 1024) bd = 1024;
                k_attn_decode_win<<<nh, bd, (size_t)ctx*sizeof(float), st>>>(
                    dq, Kuse, Vuse, ctx, kvd, hd, grp, 1.0f, win, dao); }
            MMD(&g_w.Wo[L], dao, dap);
            k_rmsnorm<<<1, 256, 0, st>>>(dap, g_w.post_attn[L], E, eps, dnx);
            k_add<<<(unsigned)((E+255)/256), 256, 0, st>>>(dx, dnx, (size_t)E);

            k_rmsnorm<<<1, 256, 0, st>>>(dx, g_w.ffn_norm[L], E, eps, dnx);
            MMD(&g_w.Wgate[L], dnx, dg);
            MMD(&g_w.Wup[L], dnx, dup);
            {   size_t nFF = (size_t)ffL;
                k_gelu_mul<<<(unsigned)((nFF+255)/256), 256, 0, st>>>(dg, dup, nFF); }
            MMD(&g_w.Wdown[L], dg, ddn);
            k_rmsnorm<<<1, 256, 0, st>>>(ddn, g_w.post_ffw[L], E, eps, dnx);
            k_add<<<(unsigned)((E+255)/256), 256, 0, st>>>(dx, dnx, (size_t)E);

            if (PL) {
                MMD(&g_w.Wplig[L], dx, dpg);
                k_altup_gate<<<(unsigned)((PL+255)/256), 256, 0, st>>>(dpg, dipl, L, NL, PL, 1);
                MMD(&g_w.Wplproj[L], dpg, dpp);
                k_rmsnorm<<<1, 256, 0, st>>>(dpp, g_w.pl_post_norm[L], E, eps, dnx);
                k_add<<<(unsigned)((E+255)/256), 256, 0, st>>>(dx, dnx, (size_t)E);
            }
            if (g_w.pl_out_scale && g_w.pl_out_scale[L])
                k_scale_by_dev<<<(unsigned)((E+255)/256), 256, 0, st>>>(dx, (size_t)E, g_w.pl_out_scale[L]);
        }

        if (gen_here) {
            k_rmsnorm<<<1, 256, 0, st>>>(dx, g_w.out_norm, E, eps, dnx);
            if (g_w.head.f32 || g_w.head.codes) { MMD(&g_w.head, dnx, dlog); }
            else if (use_int8 && g_w.embd_packed.codes &&
                     gemv_w_packed(st, &g_w.embd_packed, dnx, dlog, dqx, dsx)) { /* tied head, dp4a */ }
            else { if (gemm(cb, g_w.embd, dnx, dlog, 1, E, V)) goto done; }
            if (softcap > 0.0f)
                k_softcap<<<(unsigned)(((size_t)V+255)/256), 256, 0, st>>>(dlog, (size_t)V, softcap);
            k_argmax<<<1, 256, 0, st>>>(dlog, V, dseq + pos + 1);   /* GPU feeds itself */
            n = pos + 2;
            if (eos_id >= 0) {                     /* eos needs the token host-side */
                int tok = -1;
                if (cudaMemcpyAsync(&tok, dseq + pos + 1, sizeof(int), cudaMemcpyDeviceToHost, st) != cudaSuccess) {
                    sp_set_error("g4 eos D2H"); goto done; }
                cudaError_t e = cudaStreamSynchronize(st);
                if (e != cudaSuccess) { fail_cuda(e, "g4 eos sync"); goto done; }
                if (tok == eos_id) { k_incr_pos<<<1, 1, 0, st>>>(dpos); break; }
            }
        }
        k_incr_pos<<<1, 1, 0, st>>>(dpos);         /* keep dpos == pos+1 */
    }
download:
    /* single sequence download at the end */
    if (cudaMemcpyAsync(seq, dseq, (size_t)n*sizeof(int), cudaMemcpyDeviceToHost, st) != cudaSuccess) {
        sp_set_error("g4 seq D2H"); goto done; }
    { cudaError_t e = cudaStreamSynchronize(st); if (e != cudaSuccess) { fail_cuda(e, "g4 final sync"); goto done; } }
    { cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) { fail_cuda(e, "g4 decode kernel"); goto done; } }
    rc = n;
done:
    if (dseq) cudaFree(dseq); if (dpos) cudaFree(dpos);
    if (dx) cudaFree(dx); if (dnx) cudaFree(dnx); if (dq) cudaFree(dq); if (dk) cudaFree(dk);
    if (dv) cudaFree(dv); if (dao) cudaFree(dao); if (dap) cudaFree(dap); if (dg) cudaFree(dg);
    if (dup) cudaFree(dup); if (ddn) cudaFree(ddn); if (dscr) cudaFree(dscr); if (dlog) cudaFree(dlog);
    if (dipl) cudaFree(dipl); if (dple) cudaFree(dple); if (dpg) cudaFree(dpg); if (dpp) cudaFree(dpp);
    if (dqx) cudaFree(dqx); if (dsx) cudaFree(dsx);
    if (cexec) cudaGraphExecDestroy(cexec); if (cgraph) cudaGraphDestroy(cgraph);
    free(hple);
    if (dKc) { for (int L = 0; L < NL; L++) if (dKc[L]) cudaFree(dKc[L]); free(dKc); }
    if (dVc) { for (int L = 0; L < NL; L++) if (dVc[L]) cudaFree(dVc[L]); free(dVc); }
    #undef MMD
    return rc;
}

/* row-major GEMM Y[n_tok x out] = X[n_tok x in] * W^T (see file header). */
static int gemm(cublasHandle_t h, const float *dW, const float *dX, float *dY,
                int n_tok, int in, int out) {
    const float a = 1.0f, b = 0.0f;
    CB(cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, out, n_tok, in,
                   &a, dW, in, dX, in, &b, dY, out), "cublasSgemm");
    return 0;
}

/* matmul through a DevTensor: f32 weights go straight to SGEMM; packed weights
 * are decoded to `scratch` first (decode-on-demand). */
static int gemm_w(cublasHandle_t h, cudaStream_t st, const DevTensor *W,
                  const float *dX, float *dY, int n_tok, float *scratch) {
    if (W->f32) return gemm(h, W->f32, dX, dY, n_tok, W->in, W->out);
    dim3 grid(W->out, (W->in + 255) / 256);   /* rows on grid.x (V can exceed 65535) */
    k_dequant_arena<<<grid, 256, 0, st>>>(W->codes, W->row_off, W->row_scale, W->row_prec,
                                          W->out, W->in, scratch);
    cudaError_t le = cudaGetLastError();
    if (le != cudaSuccess) return fail_cuda(le, "k_dequant_arena launch");
    return gemm(h, scratch, dX, dY, n_tok, W->in, W->out);
}

/* ETA.2: oracle-parity matmul — the core matmul_arena arithmetic on cuBLAS.
 * Packed weights: RAW codes (exact f32) -> SGEMM -> ONE per-row lift on Y.
 * f32 weights: plain SGEMM (no lift to apply). See k_codes_f32 for why. */
static int gemm_w_lift(cublasHandle_t h, cudaStream_t st, const DevTensor *W,
                       const float *dX, float *dY, int n_tok, float *scratch) {
    if (W->f32) return gemm(h, W->f32, dX, dY, n_tok, W->in, W->out);
    dim3 grid(W->out, (W->in + 255) / 256);
    k_codes_f32<<<grid, 256, 0, st>>>(W->codes, W->row_off, W->row_prec, W->out, W->in, scratch);
    cudaError_t le = cudaGetLastError();
    if (le != cudaSuccess) return fail_cuda(le, "k_codes_f32 launch");
    if (gemm(h, scratch, dX, dY, n_tok, W->in, W->out)) return 1;
    k_scale_rows<<<(unsigned)((W->out + 255) / 256), 256, 0, st>>>(dY, W->row_scale, W->row_prec, n_tok, W->out);
    le = cudaGetLastError();
    if (le != cudaSuccess) return fail_cuda(le, "k_scale_rows launch");
    return 0;
}

/* BETA.3a/v4: single-token packed dp4a GEMV (decode). Quantize x->int8 (dqx/dsx
 * scratch), dp4a against the packed Q8 (1 B/weight) or Q4 (0.5 B/weight, nibbles
 * unpacked in-ALU) codes — no f32 scratch materialization. `prec` is the arena
 * precision (8 or 4) the caller resolved via sp_arena_precision; uniform per arena
 * for dense qwen3. Returns 1 if it TOOK the packed path, 0 if it declined (caller
 * falls back to gemm_w dequant). */
static int gemv_w_packed(cudaStream_t st, const DevTensor *W, const float *dX, float *dY,
                         signed char *dqx, float *dsx) {
    if (W->f32) return 0;                                 /* not packed */
    const int prec = W->prec;                             /* per-TENSOR precision (uniform rows) */
    unsigned blocks = ((unsigned)W->out + 7u) / 8u;       /* 8 warps (rows) per block */
    if (prec == 4) {                                      /* Q4: int4 load = 32 weights */
        if ((W->in & 31) != 0) return 0;
        int npad = (W->in + 31) & ~31;
        k_quant_act_int8<<<1, 256, 0, st>>>(dX, W->in, npad, dqx, dsx);
        k_gemv_q4_dp4a_v2<<<blocks, 256, 0, st>>>(
            W->codes, W->row_off, W->row_scale, W->in, dqx, dsx, dY, W->out);
        return 1;
    }
    /* prec == 8 (Q8) */
    if ((W->in & 15) == 0) {                              /* tuned: warp/row + int4 loads */
        int npad = (W->in + 15) & ~15;
        k_quant_act_int8<<<1, 256, 0, st>>>(dX, W->in, npad, dqx, dsx);
        k_gemv_q8_dp4a_v2<<<blocks, 256, 0, st>>>(
            (const signed char *)W->codes, W->row_off, W->row_scale, W->in, dqx, dsx, dY, W->out);
        return 1;
    }
    if ((W->in & 3) == 0) {                               /* fallback: naive block/row */
        int npad = (W->in + 3) & ~3;
        k_quant_act_int8<<<1, 256, 0, st>>>(dX, W->in, npad, dqx, dsx);
        k_gemv_q8_dp4a<<<(unsigned)W->out, 256, 0, st>>>(
            (const signed char *)W->codes, W->row_off, W->row_scale, W->in, dqx, dsx, dY, W->out);
        return 1;
    }
    return 0;
}

/* ════════════════════════ forward ════════════════════════ */

extern "C" int gemma3_forward_cuda(const qwen3_model *m, const int32_t *tokens,
                                   int n_tok, float *logits) {
    if (!m || m->cfg.arch != SP_ARCH_GEMMA3) { sp_set_error("gemma3_forward_cuda: not a gemma3 model"); return 1; }

    const qwen3_config *c = &m->cfg;
    const int E = (int)c->n_embd, FF = (int)c->n_ff, HD = (int)c->head_dim;
    const int NH = (int)c->n_head, NKV = (int)c->n_head_kv, V = (int)c->n_vocab;
    const int QD = NH * HD, KVD = NKV * HD, SW = (int)c->sliding_window;
    const int group = NH / NKV;
    const float eps = c->rms_eps;
    const float gbase = c->rope_freq_base, lbase = 10000.0f;
    const float ascale = 1.0f / sqrtf((float)HD);
    const float embscale = sqrtf((float)E);

    if (g_w.key != m) { free_weights(&g_w); if (build_weights(m, &g_w)) return 1; }
    cublasHandle_t cb = g_w.cublas;
    cudaStream_t st = g_w.stream;

    int *dtoks = NULL;
    float *dx=NULL,*dnx=NULL,*dq=NULL,*dk=NULL,*dv=NULL,*dao=NULL,*dap=NULL,*dg=NULL,*dup=NULL,*ddn=NULL,*dlog=NULL,*dscr=NULL;
    int rc = 1;
    #define DALLOC(p, count) do { cudaError_t _e = cudaMalloc(&(p), (size_t)(count)*sizeof(float)); \
        if (_e != cudaSuccess) { fail_cuda(_e, "scratch cudaMalloc"); goto done; } } while (0)
    if (cudaMalloc(&dtoks, (size_t)n_tok*sizeof(int)) != cudaSuccess) { sp_set_error("dtoks OOM"); goto done; }
    DALLOC(dx, (size_t)n_tok*E);   DALLOC(dnx, (size_t)n_tok*E);
    DALLOC(dq, (size_t)n_tok*QD);  DALLOC(dk, (size_t)n_tok*KVD); DALLOC(dv, (size_t)n_tok*KVD);
    DALLOC(dao, (size_t)n_tok*QD); DALLOC(dap, (size_t)n_tok*E);
    DALLOC(dg, (size_t)n_tok*FF);  DALLOC(dup, (size_t)n_tok*FF); DALLOC(ddn, (size_t)n_tok*E);
    DALLOC(dlog, (size_t)n_tok*V);
    if (g_w.scratch_n) DALLOC(dscr, g_w.scratch_n);   /* arena decode scratch */

    if (cudaMemcpyAsync(dtoks, tokens, (size_t)n_tok*sizeof(int), cudaMemcpyHostToDevice, st) != cudaSuccess) {
        sp_set_error("upload tokens"); goto done;
    }
    {   dim3 grid(n_tok, (E + 255) / 256);
        k_embed_scale<<<grid, 256, 0, st>>>(g_w.embd, dtoks, n_tok, E, embscale, dx); }

    for (int L = 0; L < (int)c->n_layers; L++) {
        const int global = ((L % 6) == 5);
        const float rbase = global ? gbase : lbase;
        const int win = global ? -1 : SW;
        const size_t nE = (size_t)n_tok * E;

        k_rmsnorm<<<n_tok, 256, 0, st>>>(dx, g_w.attn_norm[L], E, eps, dnx);
        if (gemm_w(cb, st, &g_w.Wq[L], dnx, dq, n_tok, dscr)) goto done;
        if (gemm_w(cb, st, &g_w.Wk[L], dnx, dk, n_tok, dscr)) goto done;
        if (gemm_w(cb, st, &g_w.Wv[L], dnx, dv, n_tok, dscr)) goto done;
        k_rmsnorm_head<<<n_tok*NH, HD, 0, st>>>(dq, g_w.q_norm[L], NH, HD, QD, eps);
        k_rmsnorm_head<<<n_tok*NKV, HD, 0, st>>>(dk, g_w.k_norm[L], NKV, HD, KVD, eps);
        k_rope<<<n_tok*NH, HD/2, 0, st>>>(dq, NH, HD, QD, rbase);
        k_rope<<<n_tok*NKV, HD/2, 0, st>>>(dk, NKV, HD, KVD, rbase);
        {
            int bd = HD > n_tok ? HD : n_tok; if (bd > 1024) bd = 1024;
            k_attn<<<n_tok*NH, bd, (size_t)n_tok*sizeof(float), st>>>(
                dq, dk, dv, n_tok, QD, KVD, HD, group, ascale, win, dao);
        }
        if (gemm_w(cb, st, &g_w.Wo[L], dao, dap, n_tok, dscr)) goto done;
        k_rmsnorm<<<n_tok, 256, 0, st>>>(dap, g_w.post_attn[L], E, eps, dnx);
        k_add<<<(unsigned)((nE+255)/256), 256, 0, st>>>(dx, dnx, nE);

        k_rmsnorm<<<n_tok, 256, 0, st>>>(dx, g_w.ffn_norm[L], E, eps, dnx);
        if (gemm_w(cb, st, &g_w.Wgate[L], dnx, dg, n_tok, dscr)) goto done;
        if (gemm_w(cb, st, &g_w.Wup[L], dnx, dup, n_tok, dscr)) goto done;
        {   size_t nFF = (size_t)n_tok * FF;
            k_gelu_mul<<<(unsigned)((nFF+255)/256), 256, 0, st>>>(dg, dup, nFF); }
        if (gemm_w(cb, st, &g_w.Wdown[L], dg, ddn, n_tok, dscr)) goto done;
        k_rmsnorm<<<n_tok, 256, 0, st>>>(ddn, g_w.post_ffw[L], E, eps, dnx);
        k_add<<<(unsigned)((nE+255)/256), 256, 0, st>>>(dx, dnx, nE);
    }

    k_rmsnorm<<<n_tok, 256, 0, st>>>(dx, g_w.out_norm, E, eps, dnx);
    if (gemm(cb, g_w.embd, dnx, dlog, n_tok, E, V)) goto done;   /* tied head, f32 */

    if (cudaMemcpyAsync(logits, dlog, (size_t)n_tok*V*sizeof(float), cudaMemcpyDeviceToHost, st) != cudaSuccess) {
        sp_set_error("download logits"); goto done;
    }
    { cudaError_t e = cudaStreamSynchronize(st); if (e != cudaSuccess) { fail_cuda(e, "stream sync"); goto done; } }
    { cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) { fail_cuda(e, "kernel launch"); goto done; } }
    rc = 0;

done:
    if (dtoks) cudaFree(dtoks);
    if (dx) cudaFree(dx); if (dnx) cudaFree(dnx);
    if (dq) cudaFree(dq); if (dk) cudaFree(dk); if (dv) cudaFree(dv);
    if (dao) cudaFree(dao); if (dap) cudaFree(dap);
    if (dg) cudaFree(dg); if (dup) cudaFree(dup); if (ddn) cudaFree(ddn);
    if (dlog) cudaFree(dlog); if (dscr) cudaFree(dscr);
    return rc;
}

/* Qwen3 forward on CUDA (CU.5). Deltas vs gemma3: no embedding scale, plain
 * residuals (no sandwich post-norms), SwiGLU (silu) not GeGLU, single RoPE base
 * + full causal (no sliding window), untied LM head (g_w.head, arena-packed in
 * Q8). Mirrors the CPU qwen3_forward (src/forward/forward.c).
 *
 * _ex: if kv_trees != NULL, KSTE-encode each cached K head-vector (E_CU_6). The
 * post-norm/post-RoPE K is copied D->H per layer and run through the existing
 * host sp_kste_encode (byte-identical to the CPU E_CPU_6 path by construction;
 * the on-device KSTE kernel is deferred — see SESSION-STATE). */
extern "C" int qwen3_forward_cuda_ex(const qwen3_model *m, const int32_t *tokens,
                                     int n_tok, float *logits, sp_kste_tree_t *kv_trees) {
    if (!m || m->cfg.arch != SP_ARCH_QWEN3) { sp_set_error("qwen3_forward_cuda: not a qwen3 model"); return 1; }

    const qwen3_config *c = &m->cfg;
    const int E = (int)c->n_embd, FF = (int)c->n_ff, HD = (int)c->head_dim;
    const int NH = (int)c->n_head, NKV = (int)c->n_head_kv, V = (int)c->n_vocab;
    const int QD = NH * HD, KVD = NKV * HD;
    const int group = NH / NKV;
    const float eps = c->rms_eps, base = c->rope_freq_base;
    const float ascale = 1.0f / sqrtf((float)HD);
    /* E_CU_5 NTT-attention: exact integer <q,k> via int64 (== CPU sp_pr_inner). */
    const char *ntt_e = getenv("SP_ENGINE_NTT_ATTN");
    const int ntt = (ntt_e && ntt_e[0] == '1');
    const float ntt_qscale = 65536.0f;   /* SP_NTT_ATTN_SCALE */
    const float kste_scale = 65536.0f;   /* SP_KSTE_KV_SCALE (E_CU_6) */

    if (g_w.key != m) { free_weights(&g_w); if (build_weights(m, &g_w)) return 1; }
    cublasHandle_t cb = g_w.cublas;
    cudaStream_t st = g_w.stream;

    /* fp16 working precision (E_FP16_1/E_FP16_2). Weights stay f32 (f16-valued) +
     * SGEMM f32 accumulate; round the activation + Q/K/V operands to f16 at the same
     * points the CPU SP_ENGINE_FP16 path does, so CUDA-fp16 == CPU-fp16 cross-backend. */
    const char *fp16_e = getenv("SP_ENGINE_FP16");
    const int fp16 = (fp16_e && fp16_e[0] == '1');
    #define R16(buf, cnt) do { if (fp16) k_round_f16<<<(unsigned)(((size_t)(cnt)+255)/256), 256, 0, st>>>((buf), (size_t)(cnt)); } while (0)

    int *dtoks = NULL;
    float *dx=NULL,*dnx=NULL,*dq=NULL,*dk=NULL,*dv=NULL,*dao=NULL,*dap=NULL,*dg=NULL,*dup=NULL,*ddn=NULL,*dlog=NULL,*dscr=NULL;
    float *host_k = NULL; int32_t *kq = NULL;   /* E_CU_6 KSTE: D->H K + int32 quant scratch */
    int rc = 1;
    if (cudaMalloc(&dtoks, (size_t)n_tok*sizeof(int)) != cudaSuccess) { sp_set_error("dtoks OOM"); goto done; }
    DALLOC(dx, (size_t)n_tok*E);   DALLOC(dnx, (size_t)n_tok*E);
    DALLOC(dq, (size_t)n_tok*QD);  DALLOC(dk, (size_t)n_tok*KVD); DALLOC(dv, (size_t)n_tok*KVD);
    DALLOC(dao, (size_t)n_tok*QD); DALLOC(dap, (size_t)n_tok*E);
    DALLOC(dg, (size_t)n_tok*FF);  DALLOC(dup, (size_t)n_tok*FF); DALLOC(ddn, (size_t)n_tok*E);
    DALLOC(dlog, (size_t)n_tok*V);
    if (g_w.scratch_n) DALLOC(dscr, g_w.scratch_n);
    if (kv_trees) {
        host_k = (float *)malloc((size_t)n_tok * KVD * sizeof(float));
        kq = (int32_t *)malloc((size_t)HD * sizeof(int32_t));
        if (!host_k || !kq) { sp_set_error("kste host OOM"); goto done; }
    }

    if (cudaMemcpyAsync(dtoks, tokens, (size_t)n_tok*sizeof(int), cudaMemcpyHostToDevice, st) != cudaSuccess) {
        sp_set_error("upload tokens"); goto done;
    }
    {   dim3 grid(n_tok, (E + 255) / 256);   /* embed lookup, no scale (embscale=1) */
        k_embed_scale<<<grid, 256, 0, st>>>(g_w.embd, dtoks, n_tok, E, 1.0f, dx); }

    for (int L = 0; L < (int)c->n_layers; L++) {
        const size_t nE = (size_t)n_tok * E;

        k_rmsnorm<<<n_tok, 256, 0, st>>>(dx, g_w.attn_norm[L], E, eps, dnx);
        R16(dnx, (size_t)n_tok*E);                        /* fp16 q/k/v matmul activations */
        if (gemm_w(cb, st, &g_w.Wq[L], dnx, dq, n_tok, dscr)) goto done;
        if (gemm_w(cb, st, &g_w.Wk[L], dnx, dk, n_tok, dscr)) goto done;
        if (gemm_w(cb, st, &g_w.Wv[L], dnx, dv, n_tok, dscr)) goto done;
        k_rmsnorm_head<<<n_tok*NH, HD, 0, st>>>(dq, g_w.q_norm[L], NH, HD, QD, eps);
        k_rmsnorm_head<<<n_tok*NKV, HD, 0, st>>>(dk, g_w.k_norm[L], NKV, HD, KVD, eps);
        k_rope<<<n_tok*NH, HD/2, 0, st>>>(dq, NH, HD, QD, base);
        k_rope<<<n_tok*NKV, HD/2, 0, st>>>(dk, NKV, HD, KVD, base);

        /* E_CU_6 KSTE-KV: encode each post-norm/post-RoPE K head-vector to its
         * 64-byte signature via the host sp_kste_encode (byte-identical to the CPU
         * E_CPU_6 path). dk must be finalized first, so sync + copy D->H. */
        if (kv_trees) {
            cudaError_t e = cudaStreamSynchronize(st);
            if (e != cudaSuccess) { fail_cuda(e, "kste stream sync"); goto done; }
            if (cudaMemcpy(host_k, dk, (size_t)n_tok*KVD*sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
                sp_set_error("kste K D->H"); goto done;
            }
            for (int t = 0; t < n_tok; t++)
                for (int h = 0; h < NKV; h++) {
                    const float *kh = host_k + (size_t)t * KVD + (size_t)h * HD;
                    for (int i = 0; i < HD; i++) kq[i] = (int32_t)lrintf(kh[i] * kste_scale);
                    sp_kste_encode(kq, HD, &kv_trees[((size_t)L * n_tok + t) * NKV + h]);
                }
        }

        R16(dq, (size_t)n_tok*QD); R16(dk, (size_t)n_tok*KVD); R16(dv, (size_t)n_tok*KVD);  /* fp16 Q/K/V into attention */
        {   int bd = HD > n_tok ? HD : n_tok; if (bd > 1024) bd = 1024;
            size_t shm = (size_t)n_tok * sizeof(float);
            if (ntt)   /* E_CU_5: exact integer score path (full causal) */
                k_attn_ntt<<<n_tok*NH, bd, shm, st>>>(
                    dq, dk, dv, n_tok, QD, KVD, HD, group, ascale, ntt_qscale, dao);
            else       /* plain f32 dot, full causal (win=-1) */
                k_attn<<<n_tok*NH, bd, shm, st>>>(
                    dq, dk, dv, n_tok, QD, KVD, HD, group, ascale, -1, dao); }
        R16(dao, (size_t)n_tok*QD);                       /* fp16 o-proj activations */
        if (gemm_w(cb, st, &g_w.Wo[L], dao, dap, n_tok, dscr)) goto done;
        k_add<<<(unsigned)((nE+255)/256), 256, 0, st>>>(dx, dap, nE);    /* plain residual */

        k_rmsnorm<<<n_tok, 256, 0, st>>>(dx, g_w.ffn_norm[L], E, eps, dnx);
        R16(dnx, (size_t)n_tok*E);                        /* fp16 ffn matmul activations */
        if (gemm_w(cb, st, &g_w.Wgate[L], dnx, dg, n_tok, dscr)) goto done;
        if (gemm_w(cb, st, &g_w.Wup[L], dnx, dup, n_tok, dscr)) goto done;
        {   size_t nFF = (size_t)n_tok * FF;
            k_silu_mul<<<(unsigned)((nFF+255)/256), 256, 0, st>>>(dg, dup, nFF); }
        R16(dg, (size_t)n_tok*FF);                        /* fp16 down-proj activations */
        if (gemm_w(cb, st, &g_w.Wdown[L], dg, ddn, n_tok, dscr)) goto done;
        k_add<<<(unsigned)((nE+255)/256), 256, 0, st>>>(dx, ddn, nE);    /* plain residual */
    }

    k_rmsnorm<<<n_tok, 256, 0, st>>>(dx, g_w.out_norm, E, eps, dnx);
    R16(dnx, (size_t)n_tok*E);                            /* fp16 head activations */
    if (gemm_w(cb, st, &g_w.head, dnx, dlog, n_tok, dscr)) goto done;   /* untied head */

    if (cudaMemcpyAsync(logits, dlog, (size_t)n_tok*V*sizeof(float), cudaMemcpyDeviceToHost, st) != cudaSuccess) {
        sp_set_error("download logits"); goto done;
    }
    { cudaError_t e = cudaStreamSynchronize(st); if (e != cudaSuccess) { fail_cuda(e, "stream sync"); goto done; } }
    { cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) { fail_cuda(e, "kernel launch"); goto done; } }
    rc = 0;

done:
    if (dtoks) cudaFree(dtoks);
    if (dx) cudaFree(dx); if (dnx) cudaFree(dnx);
    if (dq) cudaFree(dq); if (dk) cudaFree(dk); if (dv) cudaFree(dv);
    if (dao) cudaFree(dao); if (dap) cudaFree(dap);
    if (dg) cudaFree(dg); if (dup) cudaFree(dup); if (ddn) cudaFree(ddn);
    if (dlog) cudaFree(dlog); if (dscr) cudaFree(dscr);
    free(host_k); free(kq);
    #undef R16
    return rc;
}

extern "C" int qwen3_forward_cuda(const qwen3_model *m, const int32_t *tokens,
                                  int n_tok, float *logits) {
    return qwen3_forward_cuda_ex(m, tokens, n_tok, logits, NULL);
}

/* ═══════════ BETA.2: position-indirect decode kernels (CUDA graphs) ═══════════
 * The per-step decode loop launches ~13 kernels/layer * NL + head/argmax per
 * token — all tiny, so wall-clock is dominated by launch overhead, not compute.
 * CUDA graphs collapse that to ONE graph launch/token, but a captured graph
 * freezes every node's arguments + launch config. The per-step loop changes
 * those each step (embed reads dseq+pos, KV-store offset = pos*KVD, attn ctx +
 * shared-mem grow with pos, argmax writes dseq+pos+1).
 *
 * Fix: hold `pos` in a device scalar `int *dpos`. These kernels DEREFERENCE it
 * instead of taking it as a host launch arg, so the graph topology AND all node
 * params are constant across replays — capture once, replay per token. The math
 * is byte-identical to the non-dyn kernels above (k_attn_decode_dyn only reads
 * ctx from *dpos+1; same accumulation order), so the decode==prefill gate holds. */

__global__ void k_embed_at(const float *embd, const int *dseq, const int *dpos,
                           int E, float *x) {
    int i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i < E) { int tok = dseq[*dpos]; x[i] = embd[(size_t)tok * E + i]; }
}

/* NEOX RoPE at the device position *dpos for a SINGLE token (t=0, grid=n_heads). */
__global__ void k_rope_dyn(float *base, int n_heads, int d, int rowstride,
                           float rbase, const int *dpos) {
    int h = blockIdx.x, i = threadIdx.x, half = d / 2;
    if (i < half) {
        float *v = base + (size_t)h * d;          /* t=0 → rowstride term vanishes */
        float freq = powf(rbase, -2.0f * (float)i / (float)d);
        float th = (float)(*dpos) * freq, c = cosf(th), s = sinf(th);
        float a = v[i], bb = v[i + half];
        v[i] = a * c - bb * s;
        v[i + half] = a * s + bb * c;
    }
    (void)n_heads; (void)rowstride;
}

/* Store the finalized single-token K/V into the persistent cache at (layer,*dpos). */
__global__ void k_kv_store(float *Kc, float *Vc, const float *dk, const float *dv,
                           const int *dpos, size_t layer_off, int KVD) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < KVD) {
        size_t off = layer_off + (size_t)(*dpos) * KVD + i;
        Kc[off] = dk[i]; Vc[off] = dv[i];
    }
}

/* Single-query GQA over [0..*dpos+1). Identical math to k_attn_decode; ctx comes
 * from the device scalar. Shared mem is FIXED at capture (P floats) — over-
 * allocation past ctx is harmless. */
__global__ void k_attn_decode_dyn(const float *q, const float *Kc, const float *Vc,
                                  const int *dpos, int KVD, int HD, int group,
                                  float ascale, size_t layer_off, float *ao) {
    extern __shared__ float sc[];
    int ctx = *dpos + 1;
    int h = blockIdx.x, kvh = h / group;
    const float *qh = q + (size_t)h * HD;
    for (int s = threadIdx.x; s < ctx; s += blockDim.x) {
        const float *kh = Kc + layer_off + (size_t)s * KVD + (size_t)kvh * HD;
        float acc = 0.0f;
        for (int i = 0; i < HD; i++) acc += qh[i] * kh[i];
        sc[s] = acc * ascale;
    }
    __syncthreads();
    __shared__ float g_sum;
    if (threadIdx.x == 0) {
        float mx = sc[0];
        for (int s = 1; s < ctx; s++) if (sc[s] > mx) mx = sc[s];
        float sum = 0.0f;
        for (int s = 0; s < ctx; s++) { float e = expf(sc[s] - mx); sc[s] = e; sum += e; }
        g_sum = sum;
    }
    __syncthreads();
    float inv = 1.0f / g_sum;
    for (int i = threadIdx.x; i < HD; i += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < ctx; s++)
            acc += sc[s] * Vc[layer_off + (size_t)s * KVD + (size_t)kvh * HD + i];
        ao[(size_t)h * HD + i] = acc * inv;
    }
}

/* Argmax over logits → write the winner straight into dseq[*dpos+1]. */
__global__ void k_argmax_at(const float *logits, int n, int *dseq, const int *dpos) {
    __shared__ float sval[256];
    __shared__ int   sidx[256];
    float bv = -3.4e38f; int bi = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        if (logits[i] > bv) { bv = logits[i]; bi = i; }
    sval[threadIdx.x] = bv; sidx[threadIdx.x] = bi;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (threadIdx.x < o) {
            if (sval[threadIdx.x + o] > sval[threadIdx.x] ||
                (sval[threadIdx.x + o] == sval[threadIdx.x] && sidx[threadIdx.x + o] < sidx[threadIdx.x])) {
                sval[threadIdx.x] = sval[threadIdx.x + o];
                sidx[threadIdx.x] = sidx[threadIdx.x + o];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) dseq[*dpos + 1] = sidx[0];
}

/* Advance the device position by one (end of each graph replay). */
__global__ void k_incr_pos(int *dpos) { if (threadIdx.x == 0) (*dpos)++; }

/* Autoregressive KV-cache decode on the GPU. seq[0..n_prompt) is the prompt;
 * writes greedy-argmax continuations into seq[n_prompt .. n_prompt+n_gen).
 * Returns the final length (n_prompt+produced), or -1 on error. Mirrors the CPU
 * qwen3_generate_kv (knobs off) so argmax sequences match.
 *
 * SP_CUDA_DECODE_GRAPH=1 captures the generate step into a CUDA graph and replays
 * it per token (BETA.2) — see the graph branch below. */
extern "C" int qwen3_decode_cuda(const qwen3_model *m, int32_t *seq,
                                 int n_prompt, int n_gen, int eos_id) {
    if (!m || m->cfg.arch != SP_ARCH_QWEN3) { sp_set_error("qwen3_decode_cuda: not a qwen3 model"); return -1; }
    if (n_prompt <= 0 || n_gen < 0) { sp_set_error("qwen3_decode_cuda: bad lengths"); return -1; }
    const qwen3_config *c = &m->cfg;
    const int E=(int)c->n_embd, FF=(int)c->n_ff, HD=(int)c->head_dim;
    const int NH=(int)c->n_head, NKV=(int)c->n_head_kv, V=(int)c->n_vocab;
    const int QD=NH*HD, KVD=NKV*HD, group=NH/NKV, NL=(int)c->n_layers;
    const float eps=c->rms_eps, base=c->rope_freq_base, ascale=1.0f/sqrtf((float)HD);
    const int P = n_prompt + n_gen;

    if (g_w.key != m) { free_weights(&g_w); if (build_weights(m, &g_w)) return -1; }
    cublasHandle_t cb = g_w.cublas;
    cudaStream_t st = g_w.stream;

    /* BETA.3a/v4: fused packed dp4a GEMV for the decode matmuls when the env is set
     * and the model carries a packed arena. The arena PRECISION (8 or 4) selects the
     * kernel (Q8 1 B/weight, or Q4 0.5 B/weight with in-ALU nibble unpack) — passed
     * to gemv_w_packed so Q4 nibbles are never misread as int8. */
    const char *i8e = getenv("SP_CUDA_DECODE_INT8");
    const int use_int8 = (i8e && i8e[0]=='1') && m->arena;   /* per-TENSOR W->prec selects the kernel */

    float *dKc=NULL,*dVc=NULL,*dx=NULL,*dnx=NULL,*dq=NULL,*dk=NULL,*dv=NULL,*dao=NULL,*dap=NULL,
          *dg=NULL,*dup=NULL,*ddn=NULL,*dlog=NULL,*dscr=NULL,*dsx=NULL;
    signed char *dqx=NULL;     /* int8 activation-quant scratch (max in-dim, padded) */
    int *dseq=NULL;            /* the whole token sequence, RESIDENT in VRAM */
    int *dpos=NULL;            /* BETA.2: device position scalar for graph replay */
    cudaGraph_t cgraph=NULL; cudaGraphExec_t cexec=NULL;
    int rc=-1, n=n_prompt;
    const size_t kvn=(size_t)NL*P*KVD;
    #define DA(p,cnt) do{ if(cudaMalloc(&(p),(size_t)(cnt)*sizeof(float))!=cudaSuccess){sp_set_error("decode OOM");goto done;} }while(0)
    DA(dKc,kvn); DA(dVc,kvn);
    DA(dx,E); DA(dnx,E); DA(dq,QD); DA(dk,KVD); DA(dv,KVD); DA(dao,QD); DA(dap,E);
    DA(dg,FF); DA(dup,FF); DA(ddn,E); DA(dlog,V);
    if (g_w.scratch_n) DA(dscr,g_w.scratch_n);
    if (use_int8) {            /* qx scratch sized to the widest matmul input (FF), padded to %32 (Q4 chunk) */
        int maxin = E; if (QD>maxin) maxin=QD; if (FF>maxin) maxin=FF;
        if (cudaMalloc(&dqx,(size_t)((maxin+31)&~31))!=cudaSuccess){sp_set_error("dqx OOM");goto done;}
        DA(dsx,1);
    }
    /* MM(W,X,Y): the decode matmul — packed dp4a GEMV (Q8 or Q4) when enabled, else
     * the cuBLAS dequant path. Both capture cleanly into the CUDA graph. */
    #define MM(W,X,Y) do{ if(!(use_int8 && gemv_w_packed(st,(W),(X),(Y),dqx,dsx))){ if(gemm_w(cb,st,(W),(X),(Y),1,dscr)) goto done; } }while(0)
    if (cudaMalloc(&dseq,(size_t)P*sizeof(int))!=cudaSuccess){sp_set_error("dseq OOM");goto done;}
    if (cudaMemcpyAsync(dseq,seq,(size_t)n_prompt*sizeof(int),cudaMemcpyHostToDevice,st)!=cudaSuccess){sp_set_error("prompt H2D");goto done;}

    /* ───────────── BETA.2: CUDA-graph generate path ─────────────
     * Prompt ingest stays per-step (one-time, fills the KV cache for [0,n_prompt-1)).
     * Then the GENERATE step is captured once into a graph and replayed n_gen times,
     * collapsing ~250 launches/token into one graph launch/token. Position lives in
     * dpos (device), so the captured node params never change. Requires the fixed
     * attn shared-mem (P floats) to fit the 48KB/block sm_75 default. */
    {
        const char *ge = getenv("SP_CUDA_DECODE_GRAPH");
        const int use_graph = (ge && ge[0]=='1');
        const size_t attn_shm = (size_t)P * sizeof(float);
        if (use_graph && attn_shm <= 48u*1024u && n_gen > 0) {
            if (cudaMalloc(&dpos,sizeof(int))!=cudaSuccess){sp_set_error("dpos OOM");goto done;}
            /* ingest the prompt: positions [0,n_prompt-1) fill the cache, no head/argmax */
            for (int pos=0; pos<n_prompt-1; pos++) {
                { dim3 grid(1,(E+255)/256); k_embed_scale<<<grid,256,0,st>>>(g_w.embd,dseq+pos,1,E,1.0f,dx); }
                for (int L=0; L<NL; L++) {
                    k_rmsnorm<<<1,256,0,st>>>(dx, g_w.attn_norm[L], E, eps, dnx);
                    MM(&g_w.Wq[L],dnx,dq);
                    MM(&g_w.Wk[L],dnx,dk);
                    MM(&g_w.Wv[L],dnx,dv);
                    k_rmsnorm_head<<<NH,HD,0,st>>>(dq, g_w.q_norm[L], NH, HD, QD, eps);
                    k_rmsnorm_head<<<NKV,HD,0,st>>>(dk, g_w.k_norm[L], NKV, HD, KVD, eps);
                    k_rope_at<<<NH,HD/2,0,st>>>(dq, NH, HD, QD, base, pos);
                    k_rope_at<<<NKV,HD/2,0,st>>>(dk, NKV, HD, KVD, base, pos);
                    const size_t loff=(size_t)L*P*KVD, koff=loff+(size_t)pos*KVD;
                    cudaMemcpyAsync(dKc+koff, dk, (size_t)KVD*sizeof(float), cudaMemcpyDeviceToDevice, st);
                    cudaMemcpyAsync(dVc+koff, dv, (size_t)KVD*sizeof(float), cudaMemcpyDeviceToDevice, st);
                    const int ctx=pos+1; int bd=HD>ctx?HD:ctx; if(bd>1024)bd=1024;
                    k_attn_decode<<<NH,bd,(size_t)ctx*sizeof(float),st>>>(dq,dKc,dVc,ctx,KVD,HD,group,ascale,loff,dao);
                    MM(&g_w.Wo[L],dao,dap);
                    k_add<<<(unsigned)((E+255)/256),256,0,st>>>(dx,dap,(size_t)E);
                    k_rmsnorm<<<1,256,0,st>>>(dx, g_w.ffn_norm[L], E, eps, dnx);
                    MM(&g_w.Wgate[L],dnx,dg);
                    MM(&g_w.Wup[L],dnx,dup);
                    k_silu_mul<<<(unsigned)(((size_t)FF+255)/256),256,0,st>>>(dg,dup,(size_t)FF);
                    MM(&g_w.Wdown[L],dg,ddn);
                    k_add<<<(unsigned)((E+255)/256),256,0,st>>>(dx,ddn,(size_t)E);
                }
            }
            /* seed dpos = n_prompt-1 (the last prompt token generates the first new one) */
            { int p0=n_prompt-1; if(cudaMemcpyAsync(dpos,&p0,sizeof(int),cudaMemcpyHostToDevice,st)!=cudaSuccess){sp_set_error("dpos seed");goto done;} }
            cublasSetStream(cb, st);
            /* ── capture ONE generate step ── */
            if (cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal)!=cudaSuccess){sp_set_error("begin capture");goto done;}
            { dim3 grid(1,(E+255)/256); k_embed_at<<<grid,256,0,st>>>(g_w.embd,dseq,dpos,E,dx); }
            for (int L=0; L<NL; L++) {
                const size_t loff=(size_t)L*P*KVD;
                k_rmsnorm<<<1,256,0,st>>>(dx, g_w.attn_norm[L], E, eps, dnx);
                MM(&g_w.Wq[L],dnx,dq);
                MM(&g_w.Wk[L],dnx,dk);
                MM(&g_w.Wv[L],dnx,dv);
                k_rmsnorm_head<<<NH,HD,0,st>>>(dq, g_w.q_norm[L], NH, HD, QD, eps);
                k_rmsnorm_head<<<NKV,HD,0,st>>>(dk, g_w.k_norm[L], NKV, HD, KVD, eps);
                k_rope_dyn<<<NH,HD/2,0,st>>>(dq, NH, HD, QD, base, dpos);
                k_rope_dyn<<<NKV,HD/2,0,st>>>(dk, NKV, HD, KVD, base, dpos);
                k_kv_store<<<(unsigned)((KVD+255)/256),256,0,st>>>(dKc,dVc,dk,dv,dpos,loff,KVD);
                int bd=HD>256?HD:256; if(bd>1024)bd=1024;
                k_attn_decode_dyn<<<NH,bd,attn_shm,st>>>(dq,dKc,dVc,dpos,KVD,HD,group,ascale,loff,dao);
                MM(&g_w.Wo[L],dao,dap);
                k_add<<<(unsigned)((E+255)/256),256,0,st>>>(dx,dap,(size_t)E);
                k_rmsnorm<<<1,256,0,st>>>(dx, g_w.ffn_norm[L], E, eps, dnx);
                MM(&g_w.Wgate[L],dnx,dg);
                MM(&g_w.Wup[L],dnx,dup);
                k_silu_mul<<<(unsigned)(((size_t)FF+255)/256),256,0,st>>>(dg,dup,(size_t)FF);
                MM(&g_w.Wdown[L],dg,ddn);
                k_add<<<(unsigned)((E+255)/256),256,0,st>>>(dx,ddn,(size_t)E);
            }
            k_rmsnorm<<<1,256,0,st>>>(dx, g_w.out_norm, E, eps, dnx);
            MM(&g_w.head,dnx,dlog);
            k_argmax_at<<<1,256,0,st>>>(dlog, V, dseq, dpos);   /* writes dseq[*dpos+1] */
            k_incr_pos<<<1,1,0,st>>>(dpos);
            if (cudaStreamEndCapture(st,&cgraph)!=cudaSuccess){sp_set_error("end capture");goto done;}
            if (cudaGraphInstantiate(&cexec,cgraph,NULL,NULL,0)!=cudaSuccess){sp_set_error("graph instantiate");goto done;}
            /* ── replay per token ── */
            for (int g=0; g<n_gen; g++) {
                if (cudaGraphLaunch(cexec,st)!=cudaSuccess){sp_set_error("graph launch");goto done;}
                n = n_prompt + g + 1;
                if (eos_id>=0) {
                    int tok=-1;
                    if (cudaMemcpyAsync(&tok,dseq+n-1,sizeof(int),cudaMemcpyDeviceToHost,st)!=cudaSuccess){sp_set_error("eos D2H");goto done;}
                    { cudaError_t e=cudaStreamSynchronize(st); if(e!=cudaSuccess){fail_cuda(e,"eos sync");goto done;} }
                    if (tok==eos_id) break;
                }
            }
            goto download;
        }
    }

    for (int pos=0; pos<P-1; pos++) {
        /* embed reads dseq[pos] in place — the previous step's argmax already
         * wrote it on-device; no host round-trip. */
        { dim3 grid(1,(E+255)/256); k_embed_scale<<<grid,256,0,st>>>(g_w.embd,dseq+pos,1,E,1.0f,dx); }
        for (int L=0; L<NL; L++) {
            k_rmsnorm<<<1,256,0,st>>>(dx, g_w.attn_norm[L], E, eps, dnx);
            MM(&g_w.Wq[L],dnx,dq);
            MM(&g_w.Wk[L],dnx,dk);
            MM(&g_w.Wv[L],dnx,dv);
            k_rmsnorm_head<<<NH,HD,0,st>>>(dq, g_w.q_norm[L], NH, HD, QD, eps);
            k_rmsnorm_head<<<NKV,HD,0,st>>>(dk, g_w.k_norm[L], NKV, HD, KVD, eps);
            k_rope_at<<<NH,HD/2,0,st>>>(dq, NH, HD, QD, base, pos);
            k_rope_at<<<NKV,HD/2,0,st>>>(dk, NKV, HD, KVD, base, pos);
            /* write the finalized K/V into the persistent cache at (L,pos) */
            const size_t loff=(size_t)L*P*KVD, koff=loff+(size_t)pos*KVD;
            cudaMemcpyAsync(dKc+koff, dk, (size_t)KVD*sizeof(float), cudaMemcpyDeviceToDevice, st);
            cudaMemcpyAsync(dVc+koff, dv, (size_t)KVD*sizeof(float), cudaMemcpyDeviceToDevice, st);
            const int ctx=pos+1;
            int bd=HD>ctx?HD:ctx; if(bd>1024)bd=1024;
            k_attn_decode<<<NH,bd,(size_t)ctx*sizeof(float),st>>>(dq,dKc,dVc,ctx,KVD,HD,group,ascale,loff,dao);
            MM(&g_w.Wo[L],dao,dap);
            k_add<<<(unsigned)((E+255)/256),256,0,st>>>(dx,dap,(size_t)E);
            k_rmsnorm<<<1,256,0,st>>>(dx, g_w.ffn_norm[L], E, eps, dnx);
            MM(&g_w.Wgate[L],dnx,dg);
            MM(&g_w.Wup[L],dnx,dup);
            k_silu_mul<<<(unsigned)(((size_t)FF+255)/256),256,0,st>>>(dg,dup,(size_t)FF);
            MM(&g_w.Wdown[L],dg,ddn);
            k_add<<<(unsigned)((E+255)/256),256,0,st>>>(dx,ddn,(size_t)E);
        }
        if (pos < n_prompt-1) continue;            /* still ingesting the prompt */
        k_rmsnorm<<<1,256,0,st>>>(dx, g_w.out_norm, E, eps, dnx);
        MM(&g_w.head,dnx,dlog);
        /* DEVICE argmax → write the next token straight into dseq[pos+1]. The
         * GPU feeds itself: no logits D2H, no per-step stream sync (eos=-1). */
        k_argmax<<<1,256,0,st>>>(dlog, V, dseq+pos+1);
        n=pos+2;
        if (eos_id>=0) {                           /* eos needs the token host-side */
            int tok=-1;
            if (cudaMemcpyAsync(&tok,dseq+pos+1,sizeof(int),cudaMemcpyDeviceToHost,st)!=cudaSuccess){sp_set_error("eos D2H");goto done;}
            { cudaError_t e=cudaStreamSynchronize(st); if(e!=cudaSuccess){fail_cuda(e,"eos sync");goto done;} }
            if (tok==eos_id) break;
        }
    }
download:
    /* single sequence download at the end */
    if (cudaMemcpyAsync(seq,dseq,(size_t)n*sizeof(int),cudaMemcpyDeviceToHost,st)!=cudaSuccess){sp_set_error("seq D2H");goto done;}
    { cudaError_t e=cudaStreamSynchronize(st); if(e!=cudaSuccess){fail_cuda(e,"decode sync");goto done;} }
    { cudaError_t e=cudaGetLastError(); if(e!=cudaSuccess){fail_cuda(e,"decode kernel");goto done;} }
    rc=n;
done:
    if(dKc)cudaFree(dKc); if(dVc)cudaFree(dVc);
    if(dx)cudaFree(dx); if(dnx)cudaFree(dnx); if(dq)cudaFree(dq); if(dk)cudaFree(dk); if(dv)cudaFree(dv);
    if(dao)cudaFree(dao); if(dap)cudaFree(dap); if(dg)cudaFree(dg); if(dup)cudaFree(dup); if(ddn)cudaFree(ddn);
    if(dlog)cudaFree(dlog); if(dscr)cudaFree(dscr); if(dseq)cudaFree(dseq);
    if(dpos)cudaFree(dpos); if(dqx)cudaFree(dqx); if(dsx)cudaFree(dsx);
    if(cexec)cudaGraphExecDestroy(cexec); if(cgraph)cudaGraphDestroy(cgraph);
    #undef DA
    #undef MM
    return rc;
}

extern "C" void sp_cuda_model_release(const qwen3_model *m) {
    if (g_w.key == m) free_weights(&g_w);
}
