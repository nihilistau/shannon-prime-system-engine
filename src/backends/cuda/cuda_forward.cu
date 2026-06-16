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
#include "sp/xbar_episode.h"     /* XBAR P3.1b: episode manifest serialize/deserialize */

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

/* XBAR P3.2-b-2a: windowed decode attention over a RING cache of `Wring` slots.
 * The cache holds the most-recent positions: absolute position p lives at slot
 * p % Wring. We iterate the in-window positions [s0, ctx) in POSITION ORDER —
 * logical index j -> slot (s0+j) % Wring — so the per-thread work and the single-
 * thread softmax reduction are byte-for-byte identical to k_attn_decode_win over
 * a full cache (same scores, same max, same exp-sum order, same V-weighted sum).
 * Bit-exactness is the gate (G-P3-R2.b-2a). Shared = wl floats (the window len). */
__global__ void k_attn_decode_ring(const float *q, const float *Kc, const float *Vc,
                                   int ctx, int KVD, int HD, int group, float ascale,
                                   int win, int Wring, float *ao) {
    extern __shared__ float sc[];
    int h = blockIdx.x, kvh = h / group;
    int pos = ctx - 1;
    int s0 = (win >= 0 && pos - win + 1 > 0) ? pos - win + 1 : 0;
    int wl = ctx - s0;                                  /* in-window length, <= Wring */
    const float *qh = q + (size_t)h * HD;
    for (int j = threadIdx.x; j < wl; j += blockDim.x) {
        int slot = (s0 + j) % Wring;                    /* position s0+j -> ring slot */
        const float *kh = Kc + (size_t)slot * KVD + (size_t)kvh * HD;
        float acc = 0.0f;
        for (int i = 0; i < HD; i++) acc += qh[i] * kh[i];
        sc[j] = acc * ascale;
    }
    __syncthreads();
    __shared__ float g_sum;
    if (threadIdx.x == 0) {
        float mx = sc[0];
        for (int j = 1; j < wl; j++) if (sc[j] > mx) mx = sc[j];
        float sum = 0.0f;
        for (int j = 0; j < wl; j++) { float e = expf(sc[j] - mx); sc[j] = e; sum += e; }
        g_sum = sum;
    }
    __syncthreads();
    float inv = 1.0f / g_sum;
    for (int i = threadIdx.x; i < HD; i += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < wl; j++) {
            int slot = (s0 + j) % Wring;
            acc += sc[j] * Vc[(size_t)slot * KVD + (size_t)kvh * HD + i];
        }
        ao[(size_t)h * HD + i] = acc * inv;
    }
}

/* XBAR P3.2-b-2b Phase 2: gated GLOBAL attention over a sparse RECALL SET.
 * Block h attends only the mh[h] positions listed in ri[h*ristr + j] (host-selected
 * per query head by sp_arm_select_geom, H2D'd) — reading Kc/Vc at those absolute
 * positions of the live cache. Same softmax structure as k_attn_decode_win; the key
 * SET is the recall set, not [0,ctx). LOSSY by design (drops keys) — the bounded gate
 * G-P3-R2.b-2b (PPL deflection < 2.0%) replaces diffs=0 here. shared = max_m floats. */
__global__ void k_attn_decode_gather(const float *q, const float *Kc, const float *Vc,
                                     const int *ri, const int *mh, int ristr,
                                     int KVD, int HD, int group, float ascale, float *ao) {
    extern __shared__ float sc[];
    int h = blockIdx.x, kvh = h / group;
    int m = mh[h];
    const int *rih = ri + (size_t)h * ristr;
    const float *qh = q + (size_t)h * HD;
    for (int j = threadIdx.x; j < m; j += blockDim.x) {
        int s = rih[j];
        const float *kh = Kc + (size_t)s * KVD + (size_t)kvh * HD;
        float acc = 0.0f;
        for (int i = 0; i < HD; i++) acc += qh[i] * kh[i];
        sc[j] = acc * ascale;
    }
    __syncthreads();
    __shared__ float g_sum;
    if (threadIdx.x == 0) {
        float mx = sc[0];
        for (int j = 1; j < m; j++) if (sc[j] > mx) mx = sc[j];
        float sum = 0.0f;
        for (int j = 0; j < m; j++) { float e = expf(sc[j] - mx); sc[j] = e; sum += e; }
        g_sum = sum;
    }
    __syncthreads();
    float inv = 1.0f / g_sum;
    for (int i = threadIdx.x; i < HD; i += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < m; j++) {
            int s = rih[j];
            acc += sc[j] * Vc[(size_t)s * KVD + (size_t)kvh * HD + i];
        }
        ao[(size_t)h * HD + i] = acc * inv;
    }
}

/* §3q oracle: exact q.K scores per head over [0,ctx) — same dot/indexing as
 * k_attn_decode_gather (kvh = h/group, scale 1.0). scores[h*Pstr + p]. The host
 * then picks the exact top-B (the selection ceiling). One block per query head. */
__global__ void k_qk_scores(const float *q, const float *Kc, int ctx, int Pstr,
                            int KVD, int HD, int group, float *scores) {
    int h = blockIdx.x, kvh = h / group;
    const float *qh = q + (size_t)h * HD;
    for (int p = threadIdx.x; p < ctx; p += blockDim.x) {
        const float *kh = Kc + (size_t)p * KVD + (size_t)kvh * HD;
        float acc = 0.0f;
        for (int i = 0; i < HD; i++) acc += qh[i] * kh[i];
        scores[(size_t)h * Pstr + p] = acc;
    }
}

/* exact top-B indices of sc[0,ctx) via a size-B min-heap (O(ctx log B), no std). */
static int sp_oracle_top_b(const float *sc, int ctx, int B, float *hs, int *hi, int *out) {
    int m = (B < ctx) ? B : ctx, n = 0;
    for (int p = 0; p < ctx; p++) {
        float v = sc[p];
        if (n < m) {
            int i = n++; hs[i] = v; hi[i] = p;
            while (i > 0) { int par = (i - 1) / 2; if (hs[par] <= hs[i]) break;
                float t = hs[par]; hs[par] = hs[i]; hs[i] = t;
                int ti = hi[par]; hi[par] = hi[i]; hi[i] = ti; i = par; }
        } else if (v > hs[0]) {
            hs[0] = v; hi[0] = p; int i = 0;
            for (;;) { int l = 2*i+1, r = 2*i+2, sm = i;
                if (l < n && hs[l] < hs[sm]) sm = l;
                if (r < n && hs[r] < hs[sm]) sm = r;
                if (sm == i) break;
                float t = hs[sm]; hs[sm] = hs[i]; hs[i] = t;
                int ti = hi[sm]; hi[sm] = hi[i]; hi[i] = ti; i = sm; }
        }
    }
    for (int j = 0; j < n; j++) out[j] = hi[j];
    return n;
}

/* §3q Learned-LSH: transform query per head q'[h] = M·q[h] (M=R·Rᵀ, [HD,HD] row-major).
 * Then (Mq)·K == (Rq)·(RK) is the learned-projection score, fed to k_qk_scores. nh blocks. */
__global__ void k_apply_M(const float *q, const float *M, int nh, int HD, float *qp) {
    int h = blockIdx.x;
    const float *qh = q + (size_t)h * HD;
    for (int i = threadIdx.x; i < HD; i += blockDim.x) {
        const float *Mi = M + (size_t)i * HD;
        float acc = 0.0f;
        for (int j = 0; j < HD; j++) acc += Mi[j] * qh[j];
        qp[(size_t)h * HD + i] = acc;
    }
}

/* §3q C-a device-side top-B: one block per head, thread 0 runs the same size-B
 * min-heap as sp_oracle_top_b over scores[h*Pstr,..ctx) — keeps the select on the
 * decode stream, no D2H/host-heap/H2D, no sync stall. ri[h*Pstr+j], m_out[h].
 * shared = B*(float+int). Identical algorithm+input ⇒ identical top-B set to host. */
__global__ void k_topb_dev(const float *scores, int ctx, int Pstr, int B, int *ri, int *m_out) {
    if (threadIdx.x != 0) return;
    int h = blockIdx.x;
    const float *sc = scores + (size_t)h * Pstr;
    extern __shared__ char smem[];
    float *hs = (float *)smem; int *hi = (int *)(hs + B);
    int mm = (B < ctx) ? B : ctx, n = 0;
    for (int p = 0; p < ctx; p++) {
        float v = sc[p];
        if (n < mm) {
            int i = n++; hs[i] = v; hi[i] = p;
            while (i > 0) { int par = (i-1)/2; if (hs[par] <= hs[i]) break;
                float t = hs[par]; hs[par] = hs[i]; hs[i] = t;
                int ti = hi[par]; hi[par] = hi[i]; hi[i] = ti; i = par; }
        } else if (v > hs[0]) {
            hs[0] = v; hi[0] = p; int i = 0;
            for (;;) { int l = 2*i+1, r = 2*i+2, s = i;
                if (l < n && hs[l] < hs[s]) s = l;
                if (r < n && hs[r] < hs[s]) s = r;
                if (s == i) break;
                float t = hs[s]; hs[s] = hs[i]; hs[i] = t;
                int ti = hi[s]; hi[s] = hi[i]; hi[i] = ti; i = s; }
        }
    }
    int *rih = ri + (size_t)h * Pstr;
    for (int j = 0; j < n; j++) rih[j] = hi[j];
    m_out[h] = n;
}

/* §3q C-b.1 Learned-LSH sidecar: project nb vectors in[b,0..hd) by Rᵀ (R is [hd,r]
 * row-major) → out[b,0..r). Used to mint the r-dim projected query (Rq) and the
 * per-position projected-key sidecar (RK). (Rq)·(RK) ≡ (Mq)·K — output-invariant,
 * but needs only the r-dim sidecar resident, not full K (enables C-b.2 alloc-shrink). */
__global__ void k_proj_RT(const float *in, const float *R, int nb, int hd, int r, float *out) {
    int b = blockIdx.x;
    const float *ib = in + (size_t)b * hd;
    for (int j = threadIdx.x; j < r; j += blockDim.x) {
        float acc = 0.0f;
        for (int i = 0; i < hd; i++) acc += R[(size_t)i * r + j] * ib[i];
        out[(size_t)b * r + j] = acc;
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

/* OK_Q4B prefill dequant: w[j][i] = code * f16(bscale[j][i/32]). EXACT in f32
 * (4-bit code x f16 scale products are representable), so dequant->SGEMM carries
 * no extra rounding for Q4B and needs NO post-GEMM lift — both gemm_w and
 * gemm_w_lift route here when bscale is set. */
__global__ void k_dequant_arena_q4b(const unsigned char *codes, const unsigned long long *row_off,
                                    const unsigned short *bscale, int bs_nblk,
                                    int rows, int cols, float *out) {
    int j = blockIdx.x;
    int i = blockIdx.y * blockDim.x + threadIdx.x;
    if (j >= rows || i >= cols) return;
    const unsigned char *rc = codes + row_off[j];
    unsigned char byte = rc[i >> 1];
    int nib  = (i & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);
    int code = (nib ^ 8) - 8;
    float s = __half2float(__ushort_as_half(bscale[(size_t)j * bs_nblk + (i >> 5)]));
    out[(size_t)j * cols + i] = (float)code * s;
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

/* dynamic PER-BLOCK int8 quant of the activation: 16-code blocks, one scale
 * each (sxb[b] = maxabs(block b)/127), qx padded + zero-tailed.
 *
 * WHY per-block, not per-vector (ETA.5b.4, measured on gemma-4-12B): a single
 * per-vector maxabs scale COLLAPSES on outlier-heavy activations — at the
 * 12B's layer 11 (trained out_scale 0.005: the model itself flags that
 * layer's magnitudes) one outlier dominated the scale and stripped the
 * mantissa from the other 3839 dims; logits landed at oracle-rank 205596.
 * Decode-vs-probe layer bisect in LIFT arithmetic pinned the structure clean
 * (1.5e-4 floors) — the damage was 100% activation-quant. Per-block scales
 * are the llama.cpp activation-quant pattern and the GPU twin of the CPU
 * engine's block-Q8 (WIRE-CPU stage 1b). Blocks of 16 align EXACTLY with the
 * GEMV's 128-bit int4 loads — one extra f32 mul per 16 codes, no extra bus. */
__global__ void k_quant_act_int8(const float *x, int n, int npad,
                                 signed char *qx, float *sxb) {
    const int nblk = npad >> 4;
    for (int b = threadIdx.x; b < nblk; b += blockDim.x) {
        const int base = b << 4;
        float m = 0.0f;
        for (int i = 0; i < 16; i++) {
            const int idx = base + i;
            const float a = (idx < n) ? fabsf(x[idx]) : 0.0f;
            if (a > m) m = a;
        }
        const float scale = m > 0.0f ? m * (1.0f / 127.0f) : 1.0f;
        sxb[b] = scale;
        const float inv = 1.0f / scale;
        for (int i = 0; i < 16; i++) {
            const int idx = base + i;
            float v = (idx < n) ? x[idx] * inv : 0.0f;
            int q = __float2int_rn(v); if (q > 127) q = 127; if (q < -127) q = -127;
            qx[idx] = (signed char)q;
        }
    }
}

/* fused INT8 GEMV: y[j] = (row_scale[j]/127) * sx * Σ_i code[j][i]·qx[i], the inner
 * INT8·INT8->INT32 dot via __dp4a. One block per output row; `in` must be a
 * multiple of 4 (qwen3 dims are). Q8 rows only (row_prec==8). codes+row_off[j] is
 * 4-aligned (cudaMalloc base + j*in, in%4==0), so the int-cast load is a coalesced
 * 128-byte warp transaction reading 4 weights/thread. */
__global__ void k_gemv_q8_dp4a(const signed char *codes, const unsigned long long *row_off,
                               const float *row_scale, int in, const signed char *qx,
                               const float *sxb, float *y, int out) {
    int j = blockIdx.x;
    if (j >= out) return;
    const int *wrow = (const int *)(codes + row_off[j]);   /* in/4 packed-int8 words */
    const int *qxi  = (const int *)qx;
    int n4 = in >> 2;
    float facc = 0.0f;
    for (int k = threadIdx.x; k < n4; k += blockDim.x)
        facc += (float)__dp4a(wrow[k], qxi[k], 0) * sxb[k >> 2];   /* 4 ints = one 16-block */
    __shared__ float sm[256];
    sm[threadIdx.x] = facc; __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (threadIdx.x < o) sm[threadIdx.x] += sm[threadIdx.x + o];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[j] = sm[0] * (row_scale[j] * (1.0f / 127.0f));
}

/* TUNED dp4a GEMV: one WARP per output row (32 lanes collaborate), 128-bit int4
 * loads (16 Q8 codes/thread/iter — maximizes the GDDR6 transaction size), and a
 * __shfl_down_sync register-speed reduction (no shared memory). 8 warps/block =
 * 8 rows/block for SM occupancy. `in` must be a multiple of 16 (qwen3 dims are);
 * codes+row_off[j] and qx are 16-byte aligned (cudaMalloc base + j*in, in%16==0).
 * Note: int4 here is the 16-BYTE CUDA vector type, NOT 4-bit precision. */
__global__ void k_gemv_q8_dp4a_v2(const signed char *codes, const unsigned long long *row_off,
                                  const float *row_scale, int in, const signed char *qx,
                                  const float *sxb, float *y, int out) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int j = blockIdx.x * (blockDim.x >> 5) + warp;
    if (j >= out) return;
    const int4 *wrow = (const int4 *)(codes + row_off[j]);   /* in/16 16-byte chunks */
    const int4 *qxi  = (const int4 *)qx;
    const int n16 = in >> 4;
    float facc = 0.0f;
    for (int c = lane; c < n16; c += 32) {
        int4 wv = wrow[c], qv = qxi[c];                      /* 128-bit coalesced loads */
        int acc = 0;                                         /* one int4 chunk == one 16-block */
        acc = __dp4a(wv.x, qv.x, acc);
        acc = __dp4a(wv.y, qv.y, acc);
        acc = __dp4a(wv.z, qv.z, acc);
        acc = __dp4a(wv.w, qv.w, acc);
        facc += (float)acc * sxb[c];
    }
    for (int o = 16; o > 0; o >>= 1) facc += __shfl_down_sync(0xffffffffu, facc, o);
    if (lane == 0) y[j] = facc * (row_scale[j] * (1.0f / 127.0f));
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
                                  const float *sxb, float *y, int out) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int j = blockIdx.x * (blockDim.x >> 5) + warp;
    if (j >= out) return;
    const int4 *wrow = (const int4 *)(codes + row_off[j]);   /* 16 B = 32 Q4 weights */
    const int4 *qxi  = (const int4 *)qx;
    const int n32 = in >> 5;
    float facc = 0.0f;
    int a, b;
    for (int c = lane; c < n32; c += 32) {
        int4 wv = wrow[c];
        int4 q0 = qxi[2*c], q1 = qxi[2*c + 1];
        int a0 = 0, a1 = 0;          /* the 32-code chunk splits into TWO 16-blocks */
        sp_unpack8(wv.x, a, b); a0 = __dp4a(a, q0.x, a0); a0 = __dp4a(b, q0.y, a0);
        sp_unpack8(wv.y, a, b); a0 = __dp4a(a, q0.z, a0); a0 = __dp4a(b, q0.w, a0);
        sp_unpack8(wv.z, a, b); a1 = __dp4a(a, q1.x, a1); a1 = __dp4a(b, q1.y, a1);
        sp_unpack8(wv.w, a, b); a1 = __dp4a(a, q1.z, a1); a1 = __dp4a(b, q1.w, a1);
        facc += (float)a0 * sxb[2*c] + (float)a1 * sxb[2*c + 1];
    }
    for (int o = 16; o > 0; o >>= 1) facc += __shfl_down_sync(0xffffffffu, facc, o);
    if (lane == 0) y[j] = facc * (row_scale[j] * (1.0f / 7.0f));
}

/* SPEC OK_Q4B (the B1 recipe's kernel): identical code layout + loads + unpack +
 * dp4a sequence as k_gemv_q4_dp4a_v2 — the ONLY change is the scale application.
 * One 32-code chunk == one Q4B weight block, so each chunk applies its own f16
 * block scale wbsc[c] instead of a trailing per-row scale; codes are quantized
 * against the STORED f16 scale (store-then-derive), so w = code * wbsc exactly.
 * Two dp4a halves per chunk keep the per-16 activation block scales exact:
 * facc += wbsc * (sxb[2c]*acc0 + sxb[2c+1]*acc1). Zero extra code-bus traffic;
 * the bscale stream is 1/16 of code bytes, sequential, __ldg-cached. */
__global__ void k_gemv_q4b_dp4a_v2(const unsigned char *codes, const unsigned long long *row_off,
                                   const unsigned short *bscale, int bs_nblk, int in,
                                   const signed char *qx, const float *sxb, float *y, int out) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int j = blockIdx.x * (blockDim.x >> 5) + warp;
    if (j >= out) return;
    const int4 *wrow = (const int4 *)(codes + row_off[j]);   /* 16 B = 32 Q4B weights */
    const int4 *qxi  = (const int4 *)qx;
    const unsigned short *bs = bscale + (size_t)j * (size_t)bs_nblk;
    const int n32 = in >> 5;
    float facc = 0.0f;
    int a, b;
    for (int c = lane; c < n32; c += 32) {
        int4 wv = wrow[c];
        int4 q0 = qxi[2*c], q1 = qxi[2*c + 1];
        int a0 = 0, a1 = 0;
        sp_unpack8(wv.x, a, b); a0 = __dp4a(a, q0.x, a0); a0 = __dp4a(b, q0.y, a0);
        sp_unpack8(wv.y, a, b); a0 = __dp4a(a, q0.z, a0); a0 = __dp4a(b, q0.w, a0);
        sp_unpack8(wv.z, a, b); a1 = __dp4a(a, q1.x, a1); a1 = __dp4a(b, q1.y, a1);
        sp_unpack8(wv.w, a, b); a1 = __dp4a(a, q1.z, a1); a1 = __dp4a(b, q1.w, a1);
        float wbsc = __half2float(__ushort_as_half(__ldg(&bs[c])));
        facc += wbsc * ((float)a0 * sxb[2*c] + (float)a1 * sxb[2*c + 1]);
    }
    for (int o = 16; o > 0; o >>= 1) facc += __shfl_down_sync(0xffffffffu, facc, o);
    if (lane == 0) y[j] = facc;
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
    float             *row_scale;/* [out]; NULL for OK_Q4B (bscale governs) */
    unsigned char     *row_prec; /* [out] */
    unsigned short    *bscale;   /* OK_Q4B (arena v2): [out * bs_nblk] per-32-block f16
                                  * scales; NULL = per-row semantics */
    int                bs_nblk;  /* blocks per row = in/32 (0 when bscale == NULL) */
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
    if (pt->row_scale &&
        upload_arr<float>(pt->row_scale, (size_t)pt->rows, &d->row_scale)) return 1;
    if (upload_arr<unsigned char>(pt->row_prec, (size_t)pt->rows, &d->row_prec)) return 1;
    d->bscale = NULL; d->bs_nblk = 0;
    if (pt->bscale) {                                /* OK_Q4B: per-32-block f16 scales */
        d->bs_nblk = pt->bs_nblk;
        if (upload_arr<unsigned short>(pt->bscale,
                (size_t)pt->rows * (size_t)pt->bs_nblk, &d->bscale)) return 1;
    } else if (!pt->row_scale) return 1;             /* neither scale stream: invalid */
    return 0;
}

static void free_devtensor(DevTensor *d) {
    if (d->f32) cudaFree(d->f32);
    if (d->codes) cudaFree(d->codes);
    if (d->row_off) cudaFree(d->row_off);
    if (d->row_scale) cudaFree(d->row_scale);
    if (d->row_prec) cudaFree(d->row_prec);
    if (d->bscale) cudaFree(d->bscale);
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
__global__ void k_nll(const float *lg, int n, const int *dtarget, double *nll);  /* ETA.5b PPL */

/* SP_G4_SCORE result handoff (eval lane; single-threaded use). */
static double g_g4_score_nll = 0.0;
static long   g_g4_score_cnt = 0;
extern "C" void gemma4_score_result(double *nll, long *cnt) {
    if (nll) *nll = g_g4_score_nll;
    if (cnt) *cnt = g_g4_score_cnt;
}

/* XBAR P3.2-b-2b Phase-1 shadow-router result handoff (G-P3-GEOM.a oracle parity).
 * mism = # of projk floats where the incrementally-minted global sidecar (built
 * from the seam's post-RoPE K) differs from a fresh reprojection of the FINAL
 * global K cache — must be 0 (the K fed to the router IS the K attention reads).
 * sel = total host selections run (must be > 0 to prove the router executed). */
static long g_arm_shadow_mism = -1;
static long g_arm_shadow_sel  = 0;
extern "C" void xbar_arm_shadow_result(long *mism, long *sel) {
    if (mism) *mism = g_arm_shadow_mism;
    if (sel)  *sel  = g_arm_shadow_sel;
}

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

/* ═══════════ XBAR-P1: Inception Probe payload I/O ═══════════
 * CONTRACT-XBAR-P1 (lattice papers): capture/splice the per-owner jagged KV
 * cache rows at ONE absolute position. Payload = XBP1 file: header + per
 * owner layer {L, kvd, K[kvd], V[kvd]}. Host-side, called with the stream
 * SYNCED. Diagnostic lane — per-step path only (the graph path declines). */
typedef struct {
    int32_t magic, version, n_layers, row, period, kvfs, rsv0, rsv1;
} xbar_hdr;
#define XBAR_MAGIC 0x31504258  /* "XBP1" little-endian */

static int xbar_kvd_of(int L, int period, int g_nkv, int g_hd, int s_nkv, int s_hd) {
    const int global = ((L % period) == period - 1);
    return (global ? g_nkv : s_nkv) * (global ? g_hd : s_hd);
}

static int xbar_capture(const char *path, float **dKc, float **dVc, int kvfs, int period,
                        int g_nkv, int g_hd, int s_nkv, int s_hd, int row, int n_rows) {
    FILE *f = fopen(path, "wb");
    if (!f) { sp_set_error("xbar capture: cannot open payload for write"); return -1; }
    xbar_hdr h; h.magic = XBAR_MAGIC; h.version = 1; h.n_layers = kvfs; h.row = row;
    h.period = period; h.kvfs = kvfs; h.rsv0 = n_rows; h.rsv1 = 0;  /* rsv0 = n_rows (0 => 1, v1 compat) */
    fwrite(&h, sizeof h, 1, f);
    int kvd_max = 0;
    for (int L = 0; L < kvfs; L++) {
        int k = xbar_kvd_of(L, period, g_nkv, g_hd, s_nkv, s_hd);
        if (k > kvd_max) kvd_max = k;
    }
    float *tmp = (float *)malloc((size_t)kvd_max * (size_t)n_rows * sizeof(float));
    if (!tmp) { fclose(f); sp_set_error("xbar capture OOM"); return -1; }
    for (int L = 0; L < kvfs; L++) {
        const int kvd = xbar_kvd_of(L, period, g_nkv, g_hd, s_nkv, s_hd);
        const size_t cnt = (size_t)kvd * (size_t)n_rows;   /* rows are CONTIGUOUS in the cache */
        fwrite(&L, sizeof(int32_t), 1, f); fwrite(&kvd, sizeof(int32_t), 1, f);
        if (cudaMemcpy(tmp, dKc[L] + (size_t)row * kvd, cnt * sizeof(float),
                       cudaMemcpyDeviceToHost) != cudaSuccess ||
            fwrite(tmp, sizeof(float), cnt, f) != cnt) {
            free(tmp); fclose(f); sp_set_error("xbar capture: K rows D2H/write"); return -1; }
        if (cudaMemcpy(tmp, dVc[L] + (size_t)row * kvd, cnt * sizeof(float),
                       cudaMemcpyDeviceToHost) != cudaSuccess ||
            fwrite(tmp, sizeof(float), cnt, f) != cnt) {
            free(tmp); fclose(f); sp_set_error("xbar capture: V rows D2H/write"); return -1; }
    }
    free(tmp); fclose(f);
    fprintf(stderr, "    [xbar] CAPTURED rows %d..%d (%d owner layers) -> %s\n",
            row, row + n_rows - 1, kvfs, path);
    return 0;
}

/* mask: 0=all owner layers, 1=GLOBAL only, 2=SWA only. Position discipline:
 * payload row must equal the target row unless posfree (CONTRACT §1 — RoPE
 * phase is minted at the absolute position; a mismatched transplant is Arm-B
 * territory and must be EXPLICIT, never accidental). */
static int xbar_splice(const char *path, float **dKc, float **dVc, int kvfs, int period,
                       int g_nkv, int g_hd, int s_nkv, int s_hd, int row, int n_rows,
                       int mask, int posfree) {
    FILE *f = fopen(path, "rb");
    if (!f) { sp_set_error("xbar splice: cannot open payload"); return -1; }
    xbar_hdr h;
    if (fread(&h, sizeof h, 1, f) != 1 || h.magic != XBAR_MAGIC || h.version != 1) {
        fclose(f); sp_set_error("xbar splice: bad payload header"); return -1; }
    if (h.n_layers != kvfs || h.period != period) {
        fclose(f); sp_set_error("xbar splice: payload geometry mismatch (kvfs/period)"); return -1; }
    const int f_rows = h.rsv0 ? h.rsv0 : 1;       /* v1 compat: rsv0==0 => single row */
    if (f_rows != n_rows) {
        fclose(f); sp_set_error("xbar splice: payload n_rows != SP_XBAR_NROWS"); return -1; }
    if (h.row != row && !posfree) {
        fclose(f); sp_set_error("xbar splice: payload row != target row (set SP_XBAR_POSFREE=1 only for a deliberate phase-mismatch arm)"); return -1; }
    int kvd_max = 0;
    for (int L = 0; L < kvfs; L++) {
        int k = xbar_kvd_of(L, period, g_nkv, g_hd, s_nkv, s_hd);
        if (k > kvd_max) kvd_max = k;
    }
    float *tmp = (float *)malloc((size_t)kvd_max * (size_t)n_rows * sizeof(float));
    if (!tmp) { fclose(f); sp_set_error("xbar splice OOM"); return -1; }
    int spliced = 0;
    for (int L = 0; L < kvfs; L++) {
        int32_t fL = -1, fkvd = -1;
        const int kvd = xbar_kvd_of(L, period, g_nkv, g_hd, s_nkv, s_hd);
        const size_t cnt = (size_t)kvd * (size_t)n_rows;
        const int global = ((L % period) == period - 1);
        const int take = (mask == 0) || (mask == 1 && global) || (mask == 2 && !global);
        if (fread(&fL, sizeof(int32_t), 1, f) != 1 || fread(&fkvd, sizeof(int32_t), 1, f) != 1 ||
            fL != L || fkvd != kvd) {
            free(tmp); fclose(f); sp_set_error("xbar splice: per-layer geometry mismatch"); return -1; }
        if (fread(tmp, sizeof(float), cnt, f) != cnt) {
            free(tmp); fclose(f); sp_set_error("xbar splice: K rows read"); return -1; }
        if (take && cudaMemcpy(dKc[L] + (size_t)row * kvd, tmp, cnt * sizeof(float),
                               cudaMemcpyHostToDevice) != cudaSuccess) {
            free(tmp); fclose(f); sp_set_error("xbar splice: K rows H2D"); return -1; }
        if (fread(tmp, sizeof(float), cnt, f) != cnt) {
            free(tmp); fclose(f); sp_set_error("xbar splice: V rows read"); return -1; }
        if (take && cudaMemcpy(dVc[L] + (size_t)row * kvd, tmp, cnt * sizeof(float),
                               cudaMemcpyHostToDevice) != cudaSuccess) {
            free(tmp); fclose(f); sp_set_error("xbar splice: V rows H2D"); return -1; }
        if (take) spliced++;
    }
    free(tmp); fclose(f);
    fprintf(stderr, "    [xbar] SPLICED rows %d..%d (%d/%d owner layers, mask=%s, payload row %d) <- %s\n",
            row, row + n_rows - 1, spliced, kvfs, mask == 1 ? "global" : mask == 2 ? "swa" : "all", h.row, path);
    return 0;
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
    /* XBAR P3.2-b-2a SWA ring shrink: SWA owners attend only the sliding window,
     * so the live cache only needs W slots (ring) — write pos -> slot pos%Wring,
     * evicting the position that just left the window. Bit-exact to the full cache
     * because (a) the attended position SET is identical and (b) the ring kernel
     * iterates in POSITION order (logical j -> slot (s0+j)%Wring), preserving the
     * fp reduction order. NO disk paging (the window is always live — that's the
     * whole point of SWA; the two-source-with-staging kernel is the GLOBALS' job,
     * b-2b). SP_XBAR_SWA_RING=1 enables the ring; SP_XBAR_SWA_W=<w> overrides the
     * window (applies to BOTH the full-cache baseline and the ring, so the gate can
     * exercise wrap+eviction cheaply at small P). Both off = current full cache. */
    const char *swa_w_e = getenv("SP_XBAR_SWA_W");
    const int   swa_w_ovr = swa_w_e ? atoi(swa_w_e) : 0;        /* 0 = use model SW */
    const int   ws  = (swa_w_ovr > 0) ? swa_w_ovr : SW;         /* effective SWA window */
    const int   swa_ring = (getenv("SP_XBAR_SWA_RING") != NULL);
    const int   Wring = (ws < P) ? ws : P;                      /* ring slots (degenerates to P when no eviction) */
    const int   swa_active = (swa_ring || swa_w_ovr > 0);       /* either knob forces the per-step path */
    if (swa_ring || swa_w_ovr > 0)
        fprintf(stderr, "    [xbar-swa] ring=%d ws=%d Wring=%d P=%d swa_active=%d (SWA owners alloc %s slots)\n",
                swa_ring, ws, Wring, (int)P, swa_active, swa_ring ? "Wring" : "P");
    /* XBAR P3.2-b-2b Phase 0/1: SHADOW global recall router (the policy-domain entry).
     * SP_ARM_SHADOW=1 enables it; per global step we D2H the post-RoPE K + q, mint the
     * geom projk sidecar host-side, and run the FROZEN ±1 projection router
     * `sp_arm_select_geom` per global query head — but the recall set is LOGGED ONLY,
     * attention is UNCHANGED (full cache), so the live decode stays bit-exact (Phase 0/1).
     * B=0 = identity (null floor); B>0 exercises the sparse selection. Gate G-P3-GEOM.a =
     * the incrementally-minted projk == a fresh reprojection of the FINAL global K cache
     * (the K fed to the router IS the K attention reads) + decode byte-identical to no-flag.
     * Wiring the recall set INTO attention (the lossy step) is Phase 2 — NOT here. */
    const int   shadow_on = (getenv("SP_ARM_SHADOW") != NULL);
    const int   arm_B    = (getenv("SP_ARM_B")    ? atoi(getenv("SP_ARM_B"))    : 0);
    const int   arm_W    = (getenv("SP_ARM_W")    ? atoi(getenv("SP_ARM_W"))    : 0);
    const int   arm_sink = (getenv("SP_ARM_SINK") ? atoi(getenv("SP_ARM_SINK")) : 0);
    const int   arm_r    = (getenv("SP_ARM_R")    ? atoi(getenv("SP_ARM_R"))    : 32);
    int         arm_hd_max = 0;                                /* set in setup (needs g_hd/s_hd, declared below) */
    signed char    *armR = NULL; sp_arm_geom *armG = NULL; float *arm_projk = NULL;
    sp_arm_sidx    *arm_cand = NULL; int *arm_ri = NULL; float *arm_kh = NULL, *arm_qh = NULL;
    size_t          arm_projk_n = 0; long arm_sel = 0, arm_msum = 0;
    /* P3.2-b-2b Phase 2: SP_ARM_GATHER wires the recall set INTO attention (lossy) —
     * globals attend only the per-head selected set via k_attn_decode_gather. Implies
     * shadow_on (the select must run). diffs=0 dies here; the gate is G2 PPL < 2.0%. */
    const int   arm_gather = (getenv("SP_ARM_GATHER") != NULL) && shadow_on;
    int        *arm_ri_host = NULL, *arm_m_host = NULL, *arm_ri_dev = NULL, *arm_m_dev = NULL;
    /* P3.2-b-2b G1: SP_ARM_PAGE=dir serves the recalled GLOBAL set off Ring-2 instead
     * of the live cache. Per step: spill pos's global K/V → Ring-2, NaN-POISON the live
     * global cache [0,pos] (0xFF bytes = NaN float), then page back ONLY the recalled
     * union from disk into their slots; the gather then reads the live cache (recalled
     * slots = disk bytes, the rest = NaN, never read). Bit-exact vs gather-from-live
     * proves every recalled byte came off disk (a poisoned slot that wasn't paged would
     * NaN-corrupt the output). Globals-only, own contiguous store. Implies arm_gather. */
    const char *arm_page_dir = getenv("SP_ARM_PAGE");
    const int   arm_page = (arm_page_dir != NULL) && arm_gather;
    sp_arm_ring2_backend arm_be; arm_be.handle=NULL; arm_be.close=NULL; arm_be.write_block=NULL; arm_be.read_block=NULL;
    size_t     *arm_off = NULL; float *arm_phK = NULL, *arm_phV = NULL;
    unsigned char *arm_seen = NULL; int *arm_union = NULL;
    /* §3q Phase A: SP_ARM_DUMP=dir dumps the post-RoPE per-position (K,q) on the GLOBAL owners —
     * the training corpus for the per-position addresser (the b-2b shortlister the 8× failure needs;
     * Fork-6 is a span-level addresser, wrong modality). Pure offline capture: reuses the shadow_on
     * D2H of arm_kh/arm_qh, writes one self-contained file per decode call. Implies shadow_on. */
    const char *arm_dump_dir = getenv("SP_ARM_DUMP");
    FILE       *arm_dump_f = NULL;
    /* §3q oracle ceiling: SP_ARM_ORACLE selects the EXACT top-B by q.K (the best any
     * selector can do) instead of the ±1 projection — measures whether even perfect
     * selection holds < 2.0% PPL at 8×, i.e. is 8× learnable or information-bounded.
     * Scores on the GPU (q.K GEMV, microseconds), top-B on the host (min-heap). Implies gather. */
    const int   arm_oracle = (getenv("SP_ARM_ORACLE") != NULL) && arm_gather;
    float      *arm_dscores = NULL, *arm_scores_h = NULL, *arm_hs = NULL; int *arm_hi = NULL;
    /* §3q Phase B Learned-LSH: SP_ARM_LSH=M.bin (raw f32 [hd*hd] = R·Rᵀ). Selection is
     * top-B by (Rq)·(RK) = qᵀ(RRᵀ)K = (Mq)·K — reuse k_qk_scores on the transformed query
     * q'=Mq, identical select to the deployed r-dim projection, zero new hot-path kernels. */
    const char *arm_lsh_path = getenv("SP_ARM_LSH");
    const int   arm_lsh = (arm_lsh_path != NULL) && arm_gather;
    const int   arm_gpusel = arm_oracle || arm_lsh;   /* both score on GPU + top-B on host */
    float      *arm_dM = NULL, *arm_dqp = NULL;
    /* §3q C-a: SP_ARM_DEVSEL moves the top-B min-heap onto the GPU (k_topb_dev) — no
     * per-step score-D2H / host-heap / index-H2D / sync stall. GPU-score modes only
     * (oracle/LSH), and not the G1 page lane (that needs host indices for the union). */
    const int   arm_devsel = (getenv("SP_ARM_DEVSEL") != NULL) && arm_gpusel && !arm_page;
    /* §3q C-b.1: SP_ARM_LSH_R=R.bin (raw f32 [hd*r]) switches the select to score the
     * r-dim projected-key sidecar (RK) instead of the M-transform over full K — the
     * resident router state needed before the C-b.2 compact-slab shrink. output-invariant. */
    const char *arm_lshr_path = getenv("SP_ARM_LSH_R");
    const int   arm_lshr = (arm_lshr_path != NULL) && arm_lsh;
    float      *arm_dR = NULL, *arm_pk = NULL, *arm_Rq = NULL; int arm_r_dim = 0;
    /* §3q C-b.2 COMPACT GLOBAL SLAB: SP_ARM_SLAB caps the 8 global dKc/dVc at B+sink+margin
     * slots (not P). The full global history lives in a host-RAM store (arm_ramK/V); the slab
     * is an ephemeral per-step scratchpad — each step the recalled union {sinks∪pos∪top-B} is
     * paged from RAM into compact slots [0,m) and the gather reads compact-slot indices.
     * Requires the sidecar (arm_lshr) for selection; forces host top-B (union built on host). */
    const int   arm_slab = (getenv("SP_ARM_SLAB") != NULL) && arm_lshr;
    int         arm_Bslab = 0;                       /* compact slab depth = B + sink + margin */
    float     **arm_ramK = NULL, **arm_ramV = NULL;  /* host RAM store, per global layer [P*kvd] */
    float      *arm_stageK = NULL, *arm_stageV = NULL; /* host staging for the batched union page-in */
    int        *arm_slotmap = NULL, *arm_ri_host2 = NULL; /* abs→compact-slot map + remapped per-head ri */
    long        arm_umax = 0, arm_uclip = 0;          /* C-b.2 telemetry: max per-step union size + clip events */
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
     * V x E head matmul must take the dp4a route. Surface it, don't crash.
     * Exception: a PROBED diagnostic run (SP_G4_DEC_PROBE) may proceed headless
     * (the per-layer x dumps are the product; the produced token is discarded). */
    const char *pbe_g = getenv("SP_G4_DEC_PROBE");
    const int headless_probe = (pbe_g != NULL) && !(g_w.head.f32 || g_w.head.codes) &&
                               !g_w.embd && !(use_int8 && g_w.embd_packed.codes);
    if (!(g_w.head.f32 || g_w.head.codes) && !g_w.embd &&
        !(use_int8 && g_w.embd_packed.codes) && !headless_probe) {
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
    /* XBAR P3.1 (§P3.1 decode-wiring): recall self-test. When SP_XBAR_RECALL_SELFTEST=1,
     * the per-layer K/V is ALSO mirrored into a separately-laid-out episode store at the
     * owner-resolved byte offset off[L] (the §3 prefix-sum law, replicated inline here;
     * formal sp_xbar_block_off link = P3.1b), and attention READS from the store instead
     * of the live cache. Diff vs a legacy run = bit-exact gate on the off[L] addressing. */
    char *d_xbar_K=NULL, *d_xbar_V=NULL; size_t *xbar_off=NULL; size_t xbar_store_bytes=0; sp_xbar_manifest xmf{};
    /* XBAR P3.1b-2 recall-as-history: pre-load an episode's K/V into the FRONT of the live cache
     * [0,H) and decode the prompt at absolute [H,..) — pos is already absolute, so NO offset
     * threading; just skip the forward for the pre-loaded episode positions. */
    int xbar_hist_on = 0, xbar_hist_H = 0;
    /* XBAR P3.2-a shadow spill (§P3.2 write-path, the INVERSE of the read-path):
     * behind SP_XBAR_SPILL=dir, every step — AFTER the layer loop has written
     * dKc/dVc[L][pos] for all owners — D2H each owner's just-minted row into ONE
     * pinned staging buffer, ONE stream sync, then spill via the Ring-2 stdio
     * backend (sp_arm_ring2_stdio_open, the CPU decode.c twin) at the owner-
     * resolved byte off[L]+pos*kvd*4 (the §3 law; reused from the recall block).
     * Owners only — sharers spill nothing. Gate G-P3-R2.a = read every spilled
     * block back and memcmp vs a final D2H of dKc/dVc[own[L]] (must be 0 diffs,
     * no sharer block in store). Identity is structural: we copy the authoritative
     * cache bytes post-sync (sync-vs-async is irrelevant to identity). Byte-inert
     * when SP_XBAR_SPILL unset. */
    sp_arm_ring2_backend spill_be; spill_be.handle=NULL; spill_be.close=NULL; spill_be.write_block=NULL; spill_be.read_block=NULL;
    int spill_on=0; size_t *spill_off=NULL; size_t spill_store_bytes=0; float *spill_hK=NULL, *spill_hV=NULL;
    /* XBAR P3.2-b-1 paged-read (recall the spilled history off Ring-2 during LIVE
     * generation): SP_XBAR_PAGE=dir implies spill, and additionally — per step,
     * BEFORE attention — restores `[0,pos)` for every owner from the Ring-2 store
     * into the live cache, then AFTER the step POISONS `[0,pos]` (zeros it). The
     * poison makes the gate non-hollow: the live cache provably cannot be the
     * attention source, so a bit-identical output proves the bytes came off disk
     * through `off[L]`. Closed loop = write-path (spill) ∘ read-path (off[L] read).
     * NOTE: this does NOT shrink allocation (global layers attend all positions —
     * true shrink needs the sparse-recall router, P3.2-b-2). Byte-inert off. */
    int page_on=0; float *page_hK=NULL, *page_hV=NULL; size_t page_owner_bytes=0;
    cudaGraph_t cgraph=NULL; cudaGraphExec_t cexec=NULL;
    int rc=-1, n=n_prompt;
    /* SP_G4_DEC_PROBE=<pos>: ETA.5b bug-hunt — intercept the step at <pos>:
     * sync, pull the device token + the post-embed x, diff vs the HOST arena
     * dequant of the same row (the oracle's own gather arithmetic), print.
     * SP_G4_DEC_DUMP=<file>: additionally dump x after EVERY layer's FFN
     * residual (the attn_only=0 probe boundary) at the probed step — the
     * layer-bisect reference for diffing against gemma4_cuda_probe.
     * Diagnostic-only; declared up here so no goto crosses an initializer. */
    const char *pbe = getenv("SP_G4_DEC_PROBE");
    const int probe_pos = pbe ? atoi(pbe) : -1;
    const char *pdump = getenv("SP_G4_DEC_DUMP");
    float *probe_xs = (probe_pos >= 0 && pdump)
                      ? (float *)malloc((size_t)NL * E * sizeof(float)) : NULL;
    /* SP_G4_NO_OSCALE=1 (diagnostic): skip per-layer out_scale so the dump is
     * comparable to the TRUNCATED probe (which skips PL/out_scale by design).
     * out_scale is position-independent — the correct pos-(probe-1) step
     * already exonerates it — this only aligns the two references. */
    const char *noso = getenv("SP_G4_NO_OSCALE");
    const int skip_oscale = (noso && noso[0] == '1');
    /* SP_G4_SCORE=<first> (ETA.5b PPL gate): TEACHER-FORCED scoring mode — no
     * argmax write (dseq keeps the given tokens); at every pos with pos+1 >=
     * first, run the head and accumulate -log softmax(seq[pos+1]) in f64 on
     * device. Results via gemma4_score_result(). Call with n_gen=0. Forces the
     * per-step path. */
    const char *sce = getenv("SP_G4_SCORE");
    const int score_first = sce ? atoi(sce) : -1;
    /* ═══ XBAR-P1 knobs (CONTRACT-XBAR-P1-inception-probe.md) — diagnostic lane.
     * SP_XBAR_AT=<pos>       step at whose START the cache action fires
     * SP_XBAR_ROW=<row>      cache row acted on (default AT-1)
     * SP_XBAR_CAPTURE=<f>    dump owner K/V rows at ROW -> XBP1 payload
     * SP_XBAR_SPLICE=<f>     overwrite owner K/V rows at ROW <- XBP1 payload
     * SP_XBAR_MASK=all|global|swa   layer-class subset for the splice
     * SP_XBAR_POSFREE=1      allow payload-row != ROW (deliberate phase-mismatch arm)
     * SP_XBAR_RESID=<f>      dump residual x (E f32) at the END of step ROW
     * SP_XBAR_RANKS=<f>      per gen step: append tracked-token logit ranks
     * SP_XBAR_TOKENS=a,b,..  tracked token ids (<=32) for RANKS
     * Per-step path only — the graph path declines when AT >= 0. Banner below
     * echoes every knob via getenv (feedback: banners must echo getenv). */
    const char *xb_at_e   = getenv("SP_XBAR_AT");
    const int   xbar_at   = xb_at_e ? atoi(xb_at_e) : -1;
    const char *xb_row_e  = getenv("SP_XBAR_ROW");
    const int   xbar_row  = xb_row_e ? atoi(xb_row_e) : (xbar_at > 0 ? xbar_at - 1 : -1);
    const char *xb_nr_e   = getenv("SP_XBAR_NROWS");          /* P1.b: contiguous row span */
    const int   xbar_nr   = (xb_nr_e && atoi(xb_nr_e) > 0) ? atoi(xb_nr_e) : 1;
    const char *xbar_cap  = getenv("SP_XBAR_CAPTURE");
    const char *xbar_spl  = getenv("SP_XBAR_SPLICE");
    const char *xb_mask_e = getenv("SP_XBAR_MASK");
    const int   xbar_mask = (!xb_mask_e || !strcmp(xb_mask_e, "all")) ? 0
                          : !strcmp(xb_mask_e, "global") ? 1
                          : !strcmp(xb_mask_e, "swa") ? 2 : 0;
    const char *xb_pf_e   = getenv("SP_XBAR_POSFREE");
    const int   xbar_pf   = (xb_pf_e && xb_pf_e[0] == '1');
    const char *xbar_res  = getenv("SP_XBAR_RESID");
    /* XBAR-P2.a (CONTRACT-XBAR-P2): residual-ENTRY pseudo-token lane.
     * SP_XBAR_EMB_CAPTURE=<f>  dump post-embed-scale x at steps ROW..ROW+NROWS-1 (XBE1)
     * SP_XBAR_EMB=<f>          overwrite x at those steps (payload validated E/rows)
     * KV is then minted NATIVELY by the forward — geometry exact by construction.
     * NOTE: PLE/AltUp lane still gathers by token id (main-stream-only injection). */
    const char *xbar_embc = getenv("SP_XBAR_EMB_CAPTURE");
    const char *xbar_embi = getenv("SP_XBAR_EMB");
    FILE *xbar_embc_f = NULL;
    float *xbar_emb_rows = NULL;     /* loaded payload rows (NROWS x E) */
    if (xbar_embi) {
        FILE *ef = fopen(xbar_embi, "rb");
        int32_t eh[8];
        if (!ef || fread(eh, sizeof(int32_t), 8, ef) != 8 || eh[0] != 0x31454258 /* XBE1 */ ||
            eh[1] != 1 || eh[2] != xbar_nr || eh[3] != xbar_row || eh[4] != E) {
            if (ef) fclose(ef);
            sp_set_error("xbar emb: payload missing or header mismatch (XBE1/nrows/row/E)"); return -1; }
        xbar_emb_rows = (float *)malloc((size_t)xbar_nr * E * sizeof(float));
        if (!xbar_emb_rows || fread(xbar_emb_rows, sizeof(float), (size_t)xbar_nr * E, ef)
                              != (size_t)xbar_nr * E) {
            fclose(ef); free(xbar_emb_rows);
            sp_set_error("xbar emb: payload body read"); return -1; }
        fclose(ef);
    }
    const char *xbar_rkf  = getenv("SP_XBAR_RANKS");
    const char *xb_tok_e  = getenv("SP_XBAR_TOKENS");
    int xbar_ids[32]; int xbar_nids = 0;
    if (xb_tok_e) {
        const char *q = xb_tok_e;
        while (*q && xbar_nids < 32) {
            xbar_ids[xbar_nids++] = atoi(q);
            while (*q && *q != ',') q++;
            if (*q == ',') q++;
        }
    }
    const int xbar_on = (xbar_at >= 0 || xbar_res != NULL || xbar_rkf != NULL ||
                         xbar_embc != NULL || xbar_emb_rows != NULL);
    if (xbar_on) {
        static const char *xbar_envs[] = { "SP_XBAR_AT", "SP_XBAR_ROW", "SP_XBAR_NROWS", "SP_XBAR_CAPTURE",
            "SP_XBAR_SPLICE", "SP_XBAR_MASK", "SP_XBAR_POSFREE", "SP_XBAR_RESID",
            "SP_XBAR_EMB_CAPTURE", "SP_XBAR_EMB",
            "SP_XBAR_RANKS", "SP_XBAR_TOKENS", "SP_CUDA_DECODE_INT8",
            "SP_CUDA_DECODE_GRAPH", "SP_G4_SCORE", NULL };
        fprintf(stderr, "    [xbar] ── banner (getenv echo) ──\n");
        for (int i = 0; xbar_envs[i]; i++) {
            const char *v = getenv(xbar_envs[i]);
            fprintf(stderr, "    [xbar]   %s=%s\n", xbar_envs[i], v ? v : "(unset)");
        }
        if ((xbar_cap || xbar_spl) && (xbar_at < 0 || xbar_row < 0)) {
            sp_set_error("xbar: CAPTURE/SPLICE need SP_XBAR_AT (and a valid ROW)"); return -1; }
    }
    double *dnll = NULL;
    if (score_first >= 0) {
        if (cudaMalloc(&dnll, sizeof(double)) != cudaSuccess) { sp_set_error("g4 dnll OOM"); goto done; }
        cudaMemsetAsync(dnll, 0, sizeof(double), st);
        g_g4_score_nll = 0.0; g_g4_score_cnt = 0;
    }
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
    if (use_int8) {            /* qx sized to the widest matmul input, padded %32 (Q4 chunk);
                                * dsx = PER-16-BLOCK activation scales (npad/16 floats) */
        int maxin = E; if (QDmax > maxin) maxin = QDmax; if (FFmax > maxin) maxin = FFmax;
        const size_t npad = (size_t)((maxin+31)&~31);
        if (cudaMalloc(&dqx, npad) != cudaSuccess) { sp_set_error("g4 dqx OOM"); goto done; }
        G4D(dsx, npad >> 4);
    }
    /* the JAGGED cache: owners only, per-layer width. P3.2-b-2a: SWA owners shrink
     * to a Wring-slot ring (globals keep the full P — they attend all positions). */
    if (arm_slab) {   /* compact depth: SP_ARM_BSLAB or default full-P (safe: validates mechanics + measures the
                       * per-step UNION size before committing a shrink, since ∪_heads(top-B) can exceed B). */
        arm_Bslab = getenv("SP_ARM_BSLAB") ? atoi(getenv("SP_ARM_BSLAB")) : P;
        if (arm_Bslab < 1 || arm_Bslab > P) arm_Bslab = P;
    }
    for (int L = 0; L < kvfs && L < NL; L++) {
        const int global = ((L % period) == period - 1);
        const int kvd = (global ? g_nkv : s_nkv) * (global ? g_hd : s_hd);
        const size_t slots = (arm_slab && global) ? (size_t)arm_Bslab        /* §3q C-b.2: compact slab */
                           : (swa_ring && !global) ? (size_t)Wring : (size_t)P;
        G4D(dKc[L], slots * kvd);
        G4D(dVc[L], slots * kvd);
    }
    #undef G4D
    /* XBAR P3.1 / P3.1b (§P3.1): recall episode store. SELFTEST mirrors the live KV into a
     * separately-laid-out store at off[L] and reads from it; WRITE additionally serializes the
     * manifest + dumps the store to disk; LOAD deserializes + mounts a store from disk (read-only).
     * off[L] = the §3 owner-resolved prefix-sum byte law, populated straight into the formal manifest. */
    const char *xbar_rst_e = getenv("SP_XBAR_RECALL_SELFTEST");
    const char *xbar_wr_e  = getenv("SP_XBAR_RECALL_WRITE");
    const char *xbar_ld_e  = getenv("SP_XBAR_RECALL_LOAD");
    const int xbar_recall_on = (xbar_rst_e && xbar_rst_e[0] == '1') || (xbar_wr_e != NULL) || (xbar_ld_e != NULL);
    const int xbar_mirror = (xbar_ld_e == NULL);   /* LOAD reads the pre-loaded store; others mirror */
    if (xbar_recall_on) {
        char xpath[1024]; FILE *xf;
        if (xbar_ld_e) {                              /* ── LOAD: deserialize + mount from disk ── */
            snprintf(xpath, sizeof(xpath), "%s/ep.mf", xbar_ld_e); xf = fopen(xpath, "rb");
            if (!xf) { sp_set_error("xbar load: ep.mf open"); goto done; }
            fseek(xf, 0, SEEK_END); long mlen = ftell(xf); fseek(xf, 0, SEEK_SET);
            uint8_t *mbuf = (uint8_t *)malloc((size_t)mlen);
            if (!mbuf || fread(mbuf, 1, (size_t)mlen, xf) != (size_t)mlen) { free(mbuf); fclose(xf); sp_set_error("xbar load: ep.mf read"); goto done; }
            fclose(xf);
            if (sp_xbar_manifest_deserialize(&xmf, mbuf, (size_t)mlen)) { free(mbuf); sp_set_error("xbar load: deserialize"); goto done; }
            free(mbuf);
            xbar_off = (size_t *)malloc((size_t)NL * sizeof(size_t));
            if (!xbar_off) { sp_set_error("xbar load off OOM"); goto done; }
            for (int L = 0; L < NL; L++) xbar_off[L] = (size_t)xmf.layers[L].off;
            xbar_store_bytes = (size_t)xmf.store_bytes;
            if (cudaMalloc((void **)&d_xbar_K, xbar_store_bytes) != cudaSuccess ||
                cudaMalloc((void **)&d_xbar_V, xbar_store_bytes) != cudaSuccess) { sp_set_error("xbar load store OOM"); goto done; }
            uint8_t *hs = (uint8_t *)malloc(xbar_store_bytes);
            if (!hs) { sp_set_error("xbar load store host OOM"); goto done; }
            snprintf(xpath, sizeof(xpath), "%s/ep.k", xbar_ld_e); xf = fopen(xpath, "rb");
            if (!xf || fread(hs, 1, xbar_store_bytes, xf) != xbar_store_bytes) { free(hs); if (xf) fclose(xf); sp_set_error("xbar load: ep.k"); goto done; }
            fclose(xf); cudaMemcpy(d_xbar_K, hs, xbar_store_bytes, cudaMemcpyHostToDevice);
            snprintf(xpath, sizeof(xpath), "%s/ep.v", xbar_ld_e); xf = fopen(xpath, "rb");
            if (!xf || fread(hs, 1, xbar_store_bytes, xf) != xbar_store_bytes) { free(hs); if (xf) fclose(xf); sp_set_error("xbar load: ep.v"); goto done; }
            fclose(xf); cudaMemcpy(d_xbar_V, hs, xbar_store_bytes, cudaMemcpyHostToDevice); free(hs);
            fprintf(stderr, "    [xbar-p3.1b] LOADED episode store %.1f MiB from %s (%d layers, read-only)\n",
                    (double)xbar_store_bytes / 1048576.0, xbar_ld_e, xmf.NL);
        } else {                                      /* ── SELFTEST / WRITE: build store in-memory ── */
            xbar_off = (size_t *)malloc((size_t)NL * sizeof(size_t));
            if (!xbar_off) { sp_set_error("xbar recall off OOM"); goto done; }
            xmf.version = SP_XBAR_EP_VERSION; xmf.NL = NL; xmf.P = P; xmf.period = period;
            xmf.kvfs = kvfs; xmf.r = 0; xmf.proj_seed = (uint64_t)SP_ARM_PROJ_SEED;
            xmf.layers = (sp_xbar_layer *)calloc((size_t)NL, sizeof(sp_xbar_layer));
            if (!xmf.layers) { sp_set_error("xbar manifest OOM"); goto done; }
            size_t acc = 0;
            for (int L = 0; L < NL; L++) {
                const int global = ((L % period) == period - 1);
                const int kvd = (global ? g_nkv : s_nkv) * (global ? g_hd : s_hd);
                sp_xbar_layer *ly = &xmf.layers[L];
                ly->cls = (uint8_t)(global ? SP_XBAR_CLASS_GLOBAL : SP_XBAR_CLASS_SWA);
                ly->nh = global ? g_nh : s_nh; ly->nkv = global ? g_nkv : s_nkv;
                ly->hd = global ? g_hd : s_hd; ly->kvd = kvd;
                ly->window = global ? -1 : SW; ly->rope_base = global ? g_base : s_base;
                ly->has_freq_factors = (uint8_t)((global && g_w.rope_freqs) ? 1 : 0);
                if (L < kvfs) {                       /* owner: prefix-sum */
                    ly->owns_kv = 1; ly->own = L; ly->off = (uint64_t)acc;
                    ly->vless = (uint8_t)((global && !(g_w.Wv[L].f32 || g_w.Wv[L].codes)) ? 1 : 0);
                    xbar_off[L] = acc; acc += (size_t)P * kvd * 4;
                } else {                              /* sharer: off[own[L]] */
                    const int src = kvfs - (global ? 1 : 2);
                    if (src < 0 || src >= kvfs) { sp_set_error("xbar recall bad owner"); goto done; }
                    ly->owns_kv = 0; ly->own = src; ly->off = xmf.layers[src].off;
                    xbar_off[L] = (size_t)xmf.layers[src].off;
                }
            }
            xmf.store_bytes = (uint64_t)acc; xbar_store_bytes = acc;
            if (cudaMalloc((void **)&d_xbar_K, xbar_store_bytes) != cudaSuccess ||
                cudaMalloc((void **)&d_xbar_V, xbar_store_bytes) != cudaSuccess) { sp_set_error("xbar recall store OOM"); goto done; }
            fprintf(stderr, "    [xbar-p3.1] recall %s: store %.1f MiB, off[L] owner-resolved over %d layers\n",
                    xbar_wr_e ? "WRITE" : "self-test", (double)xbar_store_bytes / 1048576.0, NL);
        }
    }
    /* XBAR P3.1b-2: recall-as-history — pre-load an episode's K/V into the live cache front [0,H). */
    const char *xbar_hist_e = getenv("SP_XBAR_RECALL_HISTORY");
    if (xbar_hist_e) {
        char hp[1024]; FILE *hf; sp_xbar_manifest hmf; memset(&hmf, 0, sizeof(hmf));
        snprintf(hp, sizeof(hp), "%s/ep.mf", xbar_hist_e); hf = fopen(hp, "rb");
        if (!hf) { sp_set_error("xbar hist: ep.mf open"); goto done; }
        fseek(hf, 0, SEEK_END); long hl = ftell(hf); fseek(hf, 0, SEEK_SET);
        uint8_t *hbm = (uint8_t *)malloc((size_t)hl);
        if (!hbm || fread(hbm, 1, (size_t)hl, hf) != (size_t)hl) { free(hbm); fclose(hf); sp_set_error("xbar hist: ep.mf read"); goto done; }
        fclose(hf);
        if (sp_xbar_manifest_deserialize(&hmf, hbm, (size_t)hl)) { free(hbm); sp_set_error("xbar hist: deserialize"); goto done; }
        free(hbm);
        { const char *he = getenv("SP_XBAR_HIST_H"); xbar_hist_H = he ? atoi(he) : 0; }  /* episode prefix length to mount */
        if (xbar_hist_H <= 0 || xbar_hist_H > hmf.P || xbar_hist_H >= P) { sp_xbar_manifest_free(&hmf); sp_set_error("xbar hist: bad H (need 0<H<=store.P and H<cache.P)"); goto done; }
        size_t hsb = (size_t)hmf.store_bytes;
        uint8_t *hs = (uint8_t *)malloc(hsb);
        if (!hs) { sp_xbar_manifest_free(&hmf); sp_set_error("xbar hist host OOM"); goto done; }
        for (int two = 0; two < 2; two++) {                  /* 0 = K stream, 1 = V stream */
            snprintf(hp, sizeof(hp), "%s/ep.%c", xbar_hist_e, two ? 'v' : 'k'); hf = fopen(hp, "rb");
            if (!hf || fread(hs, 1, hsb, hf) != hsb) { free(hs); if (hf) fclose(hf); sp_xbar_manifest_free(&hmf); sp_set_error("xbar hist: ep store read"); goto done; }
            fclose(hf);
            for (int L = 0; L < kvfs && L < NL; L++) {        /* owners only; sharers reuse the owner block */
                const int global = ((L % period) == period - 1);
                const int kvd = (global ? g_nkv : s_nkv) * (global ? g_hd : s_hd);
                float *dst = two ? dVc[L] : dKc[L];
                cudaMemcpy(dst, hs + (size_t)hmf.layers[L].off, (size_t)xbar_hist_H * kvd * 4, cudaMemcpyHostToDevice);
            }
        }
        free(hs); sp_xbar_manifest_free(&hmf);
        xbar_hist_on = 1;
        fprintf(stderr, "    [xbar-p3.1b2] recall-as-history: pre-loaded episode H=%d positions into the live-cache front\n", xbar_hist_H);
    }
    /* ── XBAR P3.2-a shadow spill: open the Ring-2 stdio store + the off[L] law ──
     * Same §3 owner prefix-sum used by the recall block; sharers resolve to the
     * owner block but are never spilled. Per-pos staging = one position across all
     * owners (= store_bytes/P) so the per-step path is one batched D2H + one sync. */
    const char *xbar_spill_e = getenv("SP_XBAR_SPILL");
    const char *xbar_page_e  = getenv("SP_XBAR_PAGE");   /* P3.2-b-1: page implies spill */
    const char *xbar_store_dir = xbar_page_e ? xbar_page_e : xbar_spill_e;
    size_t spill_pos_bytes = 0;
    if (xbar_store_dir) {
        spill_off = (size_t *)malloc((size_t)NL * sizeof(size_t));
        if (!spill_off) { sp_set_error("xbar spill off OOM"); goto done; }
        size_t acc = 0, pacc = 0, maxkvd = 0;
        for (int L = 0; L < NL; L++) {
            const int global = ((L % period) == period - 1);
            const size_t kvd = (size_t)(global ? g_nkv : s_nkv) * (global ? g_hd : s_hd);
            if (L < kvfs) { spill_off[L] = acc; acc += (size_t)P * kvd * 4; pacc += kvd * 4; if (kvd > maxkvd) maxkvd = kvd; }
            else          { const int src = kvfs - (global ? 1 : 2);
                            if (src < 0 || src >= kvfs) { sp_set_error("xbar spill bad owner"); goto done; }
                            spill_off[L] = spill_off[src]; }
        }
        spill_store_bytes = acc; spill_pos_bytes = pacc;
        if (sp_arm_ring2_stdio_open(xbar_store_dir, &spill_be)) { sp_set_error("xbar spill: ring2 stdio open"); goto done; }
        if (cudaHostAlloc((void **)&spill_hK, spill_pos_bytes, cudaHostAllocDefault) != cudaSuccess ||
            cudaHostAlloc((void **)&spill_hV, spill_pos_bytes, cudaHostAllocDefault) != cudaSuccess) { sp_set_error("xbar spill staging OOM"); goto done; }
        spill_on = 1;
        if (xbar_page_e) {   /* P3.2-b-1: also alloc the per-owner page-in staging (largest owner region) */
            page_owner_bytes = (size_t)P * maxkvd * 4;
            if (cudaHostAlloc((void **)&page_hK, page_owner_bytes, cudaHostAllocDefault) != cudaSuccess ||
                cudaHostAlloc((void **)&page_hV, page_owner_bytes, cudaHostAllocDefault) != cudaSuccess) { sp_set_error("xbar page staging OOM"); goto done; }
            page_on = 1;
        }
        fprintf(stderr, "    [xbar-p3.2%s] %s ON: store %.1f MiB over %d owners -> %s%s\n",
                page_on ? "b1" : "a", page_on ? "paged-read (spill+poison+page-in)" : "shadow spill",
                (double)spill_store_bytes / 1048576.0, kvfs, xbar_store_dir,
                page_on ? " (per-step recall off disk, live source POISONED)" : " (per-step owner K/V, Ring-2 stdio)");
    }

    /* ── XBAR P3.2-b-2b Phase 0/1: shadow-router setup (frozen ±1 projection, geom API) ──
     * Build ONE Rademacher R at hd_max=512 (globals project full-width; SWA would use the
     * first 256 cols, but the shadow runs globals-only). Lay out the per-class projk sidecar
     * (sp_arm_geom_layout, kind=0 f32). Host scratch for the per-step K/q D2H + select. */
    if (shadow_on) {
        arm_hd_max = (g_hd > s_hd) ? g_hd : s_hd;             /* 512 — R built at max class head_dim */
        armR = (signed char *)malloc((size_t)arm_r * arm_hd_max);
        armG = (sp_arm_geom *)malloc((size_t)NL * sizeof(sp_arm_geom));
        if (!armR || !armG) { sp_set_error("xbar shadow R/G OOM"); goto done; }
        sp_arm_build_R(armR, arm_r, arm_hd_max);
        for (int L = 0; L < NL; L++) {
            const int gl = ((L % period) == period - 1);
            armG[L].nkv = gl ? g_nkv : s_nkv; armG[L].hd = gl ? g_hd : s_hd;
        }
        arm_projk_n = sp_arm_geom_layout(armG, NL, P, arm_r, 0);
        arm_projk = (float *)calloc(arm_projk_n ? arm_projk_n : 1, sizeof(float));
        arm_cand  = (sp_arm_sidx *)malloc((size_t)P * sizeof(sp_arm_sidx));
        arm_ri    = (int *)malloc((size_t)P * sizeof(int));
        arm_kh    = (float *)malloc((size_t)arm_hd_max * sizeof(float));
        arm_qh    = (float *)malloc((size_t)g_nh * g_hd * sizeof(float));
        if (!arm_projk || !arm_cand || !arm_ri || !arm_kh || !arm_qh) { sp_set_error("xbar shadow scratch OOM"); goto done; }
        if (arm_gather) {   /* per-head recall-set buffers (host fill -> H2D -> gather kernel) */
            arm_ri_host = (int *)malloc((size_t)g_nh * P * sizeof(int));
            arm_m_host  = (int *)malloc((size_t)g_nh * sizeof(int));
            if (!arm_ri_host || !arm_m_host) { sp_set_error("xbar gather host OOM"); goto done; }
            if (cudaMalloc((void **)&arm_ri_dev, (size_t)g_nh * P * sizeof(int)) != cudaSuccess ||
                cudaMalloc((void **)&arm_m_dev,  (size_t)g_nh * sizeof(int))     != cudaSuccess) { sp_set_error("xbar gather dev OOM"); goto done; }
        }
        if (arm_page) {   /* G1: open the globals-only Ring-2 store + the contiguous off[L] */
            arm_off = (size_t *)malloc((size_t)NL * sizeof(size_t));
            if (!arm_off) { sp_set_error("xbar arm_off OOM"); goto done; }
            size_t gacc = 0;
            for (int L = 0; L < NL; L++) {
                if ((L % period) == period - 1) { arm_off[L] = gacc; gacc += (size_t)P * g_nkv * g_hd * 4; }
                else arm_off[L] = 0;   /* SWA layers unused on the ARM page lane */
            }
            if (sp_arm_ring2_stdio_open(arm_page_dir, &arm_be)) { sp_set_error("xbar arm_page: ring2 open"); goto done; }
            arm_phK  = (float *)malloc((size_t)g_nkv * g_hd * sizeof(float));
            arm_phV  = (float *)malloc((size_t)g_nkv * g_hd * sizeof(float));
            arm_seen = (unsigned char *)malloc((size_t)P);
            arm_union= (int *)malloc((size_t)P * sizeof(int));
            if (!arm_phK || !arm_phV || !arm_seen || !arm_union) { sp_set_error("xbar arm_page scratch OOM"); goto done; }
        }
        if (arm_gpusel) {   /* §3q oracle/LSH: GPU scores [g_nh x P] + host min-heap scratch */
            if (cudaMalloc((void **)&arm_dscores, (size_t)g_nh * P * sizeof(float)) != cudaSuccess) { sp_set_error("xbar gpusel dscores OOM"); goto done; }
            arm_scores_h = (float *)malloc((size_t)g_nh * P * sizeof(float));
            arm_hs = (float *)malloc((size_t)(arm_B > 0 ? arm_B : 1) * sizeof(float));
            arm_hi = (int   *)malloc((size_t)(arm_B > 0 ? arm_B : 1) * sizeof(int));
            if (!arm_scores_h || !arm_hs || !arm_hi) { sp_set_error("xbar gpusel host OOM"); goto done; }
            if (arm_oracle) fprintf(stderr, "    [xbar-oracle] EXACT top-B by q.K selection (the selection ceiling)\n");
        }
        if (arm_lsh) {   /* §3q Phase B: load M=R·Rᵀ [g_hd x g_hd] f32, alloc transformed-query buffer */
            FILE *mf = fopen(arm_lsh_path, "rb");
            if (!mf) { sp_set_error("xbar LSH: open M.bin"); goto done; }
            size_t mn = (size_t)g_hd * g_hd;
            float *Mh = (float *)malloc(mn * sizeof(float));
            if (!Mh || fread(Mh, sizeof(float), mn, mf) != mn) { fclose(mf); free(Mh); sp_set_error("xbar LSH: read M"); goto done; }
            fclose(mf);
            if (cudaMalloc((void **)&arm_dM, mn * sizeof(float)) != cudaSuccess ||
                cudaMalloc((void **)&arm_dqp, (size_t)g_nh * g_hd * sizeof(float)) != cudaSuccess) { free(Mh); sp_set_error("xbar LSH dev OOM"); goto done; }
            cudaMemcpy(arm_dM, Mh, mn * sizeof(float), cudaMemcpyHostToDevice); free(Mh);
            fprintf(stderr, "    [xbar-lsh] Learned-LSH M=R·Rᵀ loaded (%s); select = top-B by (Mq)·K\n", arm_lsh_path);
        }
        if (arm_lshr) {   /* §3q C-b.1: load R [g_hd x r], infer r from file size; alloc r-dim sidecar + proj-query */
            FILE *rf = fopen(arm_lshr_path, "rb");
            if (!rf) { sp_set_error("xbar LSH_R: open R.bin"); goto done; }
            fseek(rf, 0, SEEK_END); long rb = ftell(rf); fseek(rf, 0, SEEK_SET);
            arm_r_dim = (int)(rb / ((long)g_hd * 4));
            if (arm_r_dim <= 0 || (long)arm_r_dim * g_hd * 4 != rb) { fclose(rf); sp_set_error("xbar LSH_R: bad size"); goto done; }
            size_t rn = (size_t)g_hd * arm_r_dim;
            float *Rh = (float *)malloc(rn * sizeof(float));
            if (!Rh || fread(Rh, sizeof(float), rn, rf) != rn) { fclose(rf); free(Rh); sp_set_error("xbar LSH_R: read"); goto done; }
            fclose(rf);
            if (cudaMalloc((void **)&arm_dR, rn * sizeof(float)) != cudaSuccess ||
                cudaMalloc((void **)&arm_pk, (size_t)NL * P * arm_r_dim * sizeof(float)) != cudaSuccess ||
                cudaMalloc((void **)&arm_Rq, (size_t)g_nh * arm_r_dim * sizeof(float)) != cudaSuccess) { free(Rh); sp_set_error("xbar LSH_R dev OOM"); goto done; }
            cudaMemcpy(arm_dR, Rh, rn * sizeof(float), cudaMemcpyHostToDevice); free(Rh);
            fprintf(stderr, "    [xbar-lsh-r] C-b.1 projected-key sidecar ON: r=%d, select scores RᵀK (resident r-dim, not full K)\n", arm_r_dim);
        }
        if (arm_slab) {   /* §3q C-b.2: host-RAM store for the full global K/V + union staging/remap scratch */
            const size_t kvd_g = (size_t)g_nkv * g_hd;
            arm_ramK = (float **)calloc((size_t)NL, sizeof(float *));
            arm_ramV = (float **)calloc((size_t)NL, sizeof(float *));
            if (!arm_ramK || !arm_ramV) { sp_set_error("xbar slab ramptr OOM"); goto done; }
            for (int L = 0; L < NL; L++) if ((L % period) == period - 1) {
                arm_ramK[L] = (float *)malloc((size_t)P * kvd_g * sizeof(float));
                arm_ramV[L] = (float *)malloc((size_t)P * kvd_g * sizeof(float));
                if (!arm_ramK[L] || !arm_ramV[L]) { sp_set_error("xbar slab RAM store OOM"); goto done; }
            }
            arm_stageK   = (float *)malloc((size_t)arm_Bslab * kvd_g * sizeof(float));
            arm_stageV   = (float *)malloc((size_t)arm_Bslab * kvd_g * sizeof(float));
            arm_slotmap  = (int *)malloc((size_t)P * sizeof(int));
            arm_ri_host2 = (int *)malloc((size_t)g_nh * arm_Bslab * sizeof(int));
            if (!arm_seen)  arm_seen  = (unsigned char *)malloc((size_t)P);
            if (!arm_union) arm_union = (int *)malloc((size_t)P * sizeof(int));
            if (!arm_stageK || !arm_stageV || !arm_slotmap || !arm_ri_host2 || !arm_seen || !arm_union) { sp_set_error("xbar slab scratch OOM"); goto done; }
            fprintf(stderr, "    [xbar-slab] C-b.2 COMPACT GLOBAL SLAB: globals capped at %d slots (was P=%d); full K/V in host RAM, union paged per step\n", arm_Bslab, P);
        }
        fprintf(stderr, "    [xbar-p3.2b2b] %s router ON: B=%d W=%d sink=%d r=%d, GLOBALS ONLY%s\n",
                arm_gather ? "GATHER (Phase 2, LOSSY)" : "SHADOW (Phase 0/1, bit-exact)",
                arm_B, arm_W, arm_sink, arm_r,
                arm_gather ? " — recall set wired into attention, output diverges (gate = G2 PPL<2%)" : " — attention UNCHANGED");
        if (arm_dump_dir) {   /* §3q Phase A: one self-contained K/q file per decode call (chunk) */
            static int arm_dump_seq = 0;
            char dpath[1024];
            snprintf(dpath, sizeof dpath, "%s/kq_call%d.bin", arm_dump_dir, arm_dump_seq++);
            arm_dump_f = fopen(dpath, "wb");
            if (!arm_dump_f) { sp_set_error("xbar arm_dump: open"); goto done; }
            int32_t fhdr[8] = { 0x4651444B /*KDQF file hdr*/, 1, NL, period, g_nh, g_nkv, g_hd, n_prompt };
            fwrite(fhdr, sizeof(int32_t), 8, arm_dump_f);
            fprintf(stderr, "    [xbar-dumpA] K/q dump -> %s (globals; kvd=%d qd=%d, per-rec hdr {magic,L,pos,nkv,nh,hd})\n",
                    dpath, g_nkv*g_hd, g_nh*g_hd);
        }
    }

    /* prompt into VRAM once; dpos = 0 */
    if (cudaMemcpyAsync(dseq, seq, (size_t)n_prompt*sizeof(int), cudaMemcpyHostToDevice, st) != cudaSuccess) {
        sp_set_error("g4 prompt H2D"); goto done; }
    { int z = xbar_hist_H;   /* P3.1b-2: seed dpos = H so the device position counter (k_incr_pos skipped
                              * for the pre-loaded episode positions) stays == absolute pos for the prompt. */
      if (cudaMemcpyAsync(dpos, &z, sizeof(int), cudaMemcpyHostToDevice, st) != cudaSuccess) {
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
        const int use_graph = (ge && ge[0] == '1') && !xbar_on && !xbar_recall_on && !xbar_hist_on && !spill_on && !swa_active && !shadow_on;  /* XBAR / recall / history / spill / SWA-ring / shadow-router = per-step only */
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
        if (xbar_hist_on && pos < xbar_hist_H) continue;   /* P3.1b-2: episode KV pre-loaded @ [0,H); skip its forward */
        /* ── XBAR P3.2-b-1 PAGE-IN: restore [0,pos) for every owner from the Ring-2
         * store (those positions were spilled + then POISONED on prior steps), so
         * this step's attention reads its own history back off disk. read_block →
         * pinned host → H2D, synced per owner (page_hK/hV reused). [0,pos) of owner L
         * is contiguous at off[L]. Sharers ride the owner block (transitive). ── */
        if (page_on && pos > 0) {
            for (int L = 0; L < kvfs && L < NL; L++) {
                const int global = ((L % period) == period - 1);
                const size_t kvd = (size_t)(global ? g_nkv : s_nkv) * (global ? g_hd : s_hd);
                const size_t nb = (size_t)pos * kvd * 4;
                if (spill_be.read_block(spill_be.handle, 0, (uint64_t)spill_off[L], page_hK, nb) ||
                    spill_be.read_block(spill_be.handle, 1, (uint64_t)spill_off[L], page_hV, nb)) {
                    sp_set_error("xbar page: read_block"); goto done; }
                cudaMemcpyAsync(dKc[L], page_hK, nb, cudaMemcpyHostToDevice, st);
                cudaMemcpyAsync(dVc[L], page_hV, nb, cudaMemcpyHostToDevice, st);
                cudaError_t e = cudaStreamSynchronize(st);   /* page_hK/hV reused next owner */
                if (e != cudaSuccess) { fail_cuda(e, "xbar page H2D sync"); goto done; }
            }
        }
        const int gen_here = (pos >= n_prompt - 1);

        /* XBAR-P1: at the START of step AT, act on cache row ROW. Order is
         * CAPTURE then SPLICE (a same-file capture+splice is the G0 identity).
         * Row ROW was minted during step ROW; every step >= AT attends to the
         * (possibly foreign) row. Stream synced first — the row's writes are
         * in-flight on st. */
        if (pos == xbar_at && (xbar_cap || xbar_spl)) {
            cudaError_t xe = cudaStreamSynchronize(st);
            if (xe != cudaSuccess) { fail_cuda(xe, "xbar sync"); goto done; }
            if (xbar_row + xbar_nr - 1 >= pos) { sp_set_error("xbar: ROW+NROWS-1 must be < AT (rows not minted yet)"); goto done; }
            if (xbar_cap && xbar_capture(xbar_cap, dKc, dVc, kvfs, period,
                                         g_nkv, g_hd, s_nkv, s_hd, xbar_row, xbar_nr)) goto done;
            if (xbar_spl && xbar_splice(xbar_spl, dKc, dVc, kvfs, period,
                                        g_nkv, g_hd, s_nkv, s_hd, xbar_row, xbar_nr,
                                        xbar_mask, xbar_pf)) goto done;
        }
        if (g_w.embd) { dim3 grid(1, (E + 255) / 256);
          k_embed_scale<<<grid, 256, 0, st>>>(g_w.embd, dseq + pos, 1, E, embscale, dx); }
        else k_embed_packed_at<<<(unsigned)((E+255)/256), 256, 0, st>>>(
            g_w.embd_packed.codes, g_w.embd_packed.row_off, g_w.embd_packed.row_scale,
            g_w.embd_packed.row_prec, dseq, dpos, E, embscale, dx);

        /* XBAR-P2.a residual-entry lane: act on x right after the embed kernel,
         * BEFORE PLE/AltUp and the layer stack — the pseudo-token entry point. */
        if ((xbar_embc || xbar_emb_rows) && pos >= xbar_row && pos < xbar_row + xbar_nr) {
            cudaError_t xe = cudaStreamSynchronize(st);
            if (xe != cudaSuccess) { fail_cuda(xe, "xbar emb sync"); goto done; }
            if (xbar_embc) {
                if (!xbar_embc_f) {
                    xbar_embc_f = fopen(xbar_embc, "wb");
                    if (!xbar_embc_f) { sp_set_error("xbar emb capture: open"); goto done; }
                    int32_t eh[8] = { 0x31454258, 1, xbar_nr, xbar_row, E, 0, 0, 0 };
                    fwrite(eh, sizeof(int32_t), 8, xbar_embc_f);
                }
                float *hx = (float *)malloc((size_t)E * sizeof(float));
                if (!hx) { sp_set_error("xbar emb capture OOM"); goto done; }
                cudaMemcpy(hx, dx, (size_t)E * sizeof(float), cudaMemcpyDeviceToHost);
                fwrite(hx, sizeof(float), (size_t)E, xbar_embc_f);
                free(hx);
                if (pos == xbar_row + xbar_nr - 1)
                    fprintf(stderr, "    [xbar] EMB captured steps %d..%d (E=%d) -> %s\n",
                            xbar_row, pos, E, xbar_embc);
            }
            if (xbar_emb_rows) {
                if (cudaMemcpy(dx, xbar_emb_rows + (size_t)(pos - xbar_row) * E,
                               (size_t)E * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
                    sp_set_error("xbar emb inject H2D"); goto done; }
                if (pos == xbar_row + xbar_nr - 1)
                    fprintf(stderr, "    [xbar] EMB injected steps %d..%d (E=%d)\n",
                            xbar_row, pos, E);
            }
        }

        if (pos == probe_pos) {
            cudaStreamSynchronize(st);
            int tok_d = -1, pos_d = -1;
            cudaMemcpy(&tok_d, dseq + pos, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&pos_d, dpos, sizeof(int), cudaMemcpyDeviceToHost);
            float *hx = (float *)malloc((size_t)E * sizeof(float));
            float *hr = (float *)malloc((size_t)E * sizeof(float));
            const sp_arena_tensor *eat = m->arena ? sp_arena_find(m->arena, m->token_embd->name) : NULL;
            if (hx && hr && eat) {
                cudaMemcpy(hx, dx, (size_t)E * sizeof(float), cudaMemcpyDeviceToHost);
                if (!sp_arena_dequant_row(eat, tok_d, hr)) {
                    double ma = 0.0; int bi = -1; double sg = 0.0, sr = 0.0;
                    for (int i = 0; i < E; i++) {
                        float ref = hr[i] * embscale;
                        double d = fabs((double)hx[i] - (double)ref);
                        if (d > ma) { ma = d; bi = i; }
                        sg += (double)hx[i] * hx[i]; sr += (double)ref * ref;
                    }
                    fprintf(stderr, "    [g4-dec-probe] pos %d (*dpos=%d) tok=%d  embed max|diff| %.3e at i=%d "
                                    "(gpu %.6e vs host %.6e)  |x|_gpu %.4e |x|_ref %.4e\n",
                            pos, pos_d, tok_d, ma, bi,
                            bi >= 0 ? (double)hx[bi] : 0.0,
                            bi >= 0 ? (double)hr[bi] * embscale : 0.0,
                            sqrt(sg), sqrt(sr));
                } else fprintf(stderr, "    [g4-dec-probe] pos %d tok=%d: host dequant FAILED\n", pos, tok_d);
            }
            free(hx); free(hr);
        }

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
            const int win = global ? -1 : ws;   /* P3.2-b-2a: ws = SW, or the SP_XBAR_SWA_W override */
            const int ffL = g_w.Wgate[L].out;
            int arm_max_m = 0;                   /* P3.2-b-2b: widest per-head recall set this global step (gather shared-mem size) */

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
                /* P3.2-b-2a: SWA owners write into the ring at slot pos%Wring (evicting
                 * the position that just left the window); globals write at pos. */
                const size_t wslot = (swa_ring && !global) ? (size_t)(pos % Wring) : (size_t)pos;
                if (arm_slab && global) {   /* §3q C-b.2: global K/V lives in the host-RAM store, not the compact slab */
                    cudaMemcpyAsync(arm_ramK[L] + (size_t)pos*kvd, dk, (size_t)kvd*sizeof(float), cudaMemcpyDeviceToHost, st);
                    cudaMemcpyAsync(arm_ramV[L] + (size_t)pos*kvd, dv, (size_t)kvd*sizeof(float), cudaMemcpyDeviceToHost, st);
                } else {
                    cudaMemcpyAsync(dKc[L] + wslot*kvd, dk, (size_t)kvd*sizeof(float), cudaMemcpyDeviceToDevice, st);
                    cudaMemcpyAsync(dVc[L] + wslot*kvd, dv, (size_t)kvd*sizeof(float), cudaMemcpyDeviceToDevice, st);
                }
                if (arm_lshr && global)   /* §3q C-b.1: mint the r-dim projected key RᵀK[pos] into the sidecar */
                    k_proj_RT<<<1, arm_r_dim, 0, st>>>(dk, arm_dR, 1, hd, arm_r_dim, arm_pk + ((size_t)L*P + pos)*arm_r_dim);
                Kuse = dKc[L]; Vuse = dVc[L];
                /* ── XBAR P3.2-b-2b Phase 0/1: host shadow-select on the GLOBAL owners.
                 * D2H the just-finalized post-RoPE K (dk == the cache row) + q, mint the geom
                 * projk sidecar incrementally, run the frozen sp_arm_select_geom per query head.
                 * LOG ONLY — Kuse stays the full cache, so attention is byte-identical. ── */
                if (shadow_on && global) {
                    if (!arm_devsel || arm_dump_f) {   /* host needs K/q only for geom-select, host-select, or the dump */
                        cudaError_t se = cudaStreamSynchronize(st);
                        if (se != cudaSuccess) { fail_cuda(se, "xbar shadow sync"); goto done; }
                        cudaMemcpy(arm_kh, dk, (size_t)kvd*sizeof(float), cudaMemcpyDeviceToHost);
                        cudaMemcpy(arm_qh, dq, (size_t)qd*sizeof(float),  cudaMemcpyDeviceToHost);
                        if (arm_dump_f) {   /* §3q Phase A corpus: post-RoPE K (kvd) + q (qd), this (L,pos) */
                            int32_t rh[6] = { 0x504B5251 /*QRKP rec*/, L, pos, nkv, nh, hd };
                            fwrite(rh, sizeof(int32_t), 6, arm_dump_f);
                            fwrite(arm_kh, sizeof(float), (size_t)kvd, arm_dump_f);
                            fwrite(arm_qh, sizeof(float), (size_t)qd,  arm_dump_f);
                        }
                    }
                    if (arm_gpusel) {   /* §3q oracle/LSH: GPU scores */
                        if (arm_lshr) {   /* §3q C-b.1: score Rq · sidecar(RᵀK) — r-dim resident, no full K */
                            k_proj_RT<<<nh, arm_r_dim, 0, st>>>(dq, arm_dR, nh, hd, arm_r_dim, arm_Rq);
                            k_qk_scores<<<nh, 256, 0, st>>>(arm_Rq, arm_pk + (size_t)L*P*arm_r_dim,
                                                            pos + 1, P, arm_r_dim, arm_r_dim, nh, arm_dscores);
                        } else {
                            const float *qsel = dq;
                            if (arm_lsh) {  /* transform q' = M·q per head, then score on q' */
                                k_apply_M<<<nh, 256, 0, st>>>(dq, arm_dM, nh, hd, arm_dqp);
                                qsel = arm_dqp;
                            }
                            k_qk_scores<<<nh, 256, 0, st>>>(qsel, dKc[L], pos + 1, P, kvd, hd, grp, arm_dscores);
                        }
                        if (arm_devsel && !arm_slab) {   /* §3q C-a: top-B on DEVICE (C-b.2 slab needs host indices for the union) */
                            k_topb_dev<<<nh, 32, (size_t)arm_B*(sizeof(float)+sizeof(int)), st>>>(
                                arm_dscores, pos + 1, P, arm_B, arm_ri_dev, arm_m_dev);
                        } else {
                            cudaError_t oe = cudaStreamSynchronize(st);
                            if (oe != cudaSuccess) { fail_cuda(oe, "xbar gpusel scores sync"); goto done; }
                            cudaMemcpy(arm_scores_h, arm_dscores, (size_t)nh * P * sizeof(float), cudaMemcpyDeviceToHost);
                        }
                    } else {
                        sp_arm_project_geom(armR, arm_r, arm_hd_max, armG[L].hd, arm_kh,
                                            arm_projk + armG[L].off + (size_t)pos*armG[L].nkv*arm_r);
                    }
                    if (!arm_devsel || arm_slab) for (int h = 0; h < nh; h++) {   /* host top-B path (devsel did it on-device; slab needs host indices) */
                        int m = arm_gpusel
                            ? sp_oracle_top_b(arm_scores_h + (size_t)h*P, pos + 1, arm_B, arm_hs, arm_hi, arm_ri)
                            : sp_arm_select_geom(armR, arm_r, arm_hd_max, &armG[L], arm_qh + (size_t)h*hd,
                                                 arm_projk, P, 0, arm_B, arm_W, arm_sink, pos, arm_cand, arm_ri);
                        arm_sel++; arm_msum += m;
                        if (arm_gather) {   /* stash this head's recall set for the gather kernel */
                            arm_m_host[h] = m;
                            memcpy(arm_ri_host + (size_t)h * P, arm_ri, (size_t)m * sizeof(int));
                            if (m > arm_max_m) arm_max_m = m;
                        }
                    }
                }
                if (xbar_recall_on) {   /* P3.1: read attention from the episode store @ off[L] */
                    if (xbar_mirror) {  /* SELFTEST/WRITE: mirror the live KV into the store first; LOAD reads disk as-is */
                        cudaMemcpyAsync(d_xbar_K + xbar_off[L] + (size_t)pos*kvd*4, dk, (size_t)kvd*sizeof(float), cudaMemcpyDeviceToDevice, st);
                        cudaMemcpyAsync(d_xbar_V + xbar_off[L] + (size_t)pos*kvd*4, dv, (size_t)kvd*sizeof(float), cudaMemcpyDeviceToDevice, st);
                    }
                    Kuse = (float *)(d_xbar_K + xbar_off[L]); Vuse = (float *)(d_xbar_V + xbar_off[L]);
                }
            } else {
                const int src = kvfs - (global ? 1 : 2);
                Kuse = dKc[src]; Vuse = dVc[src];
                if (!Kuse || !Vuse) { sp_set_error("g4 sharer before owner"); goto done; }
                if (xbar_recall_on)     /* P3.1: sharer reads the owner's mirrored block @ off[L] (= off[own[L]]) */
                    { Kuse = (float *)(d_xbar_K + xbar_off[L]); Vuse = (float *)(d_xbar_V + xbar_off[L]); }
            }
            {   const int ctx = pos + 1;
                if (arm_gather && global && !xbar_recall_on) {  /* P3.2-b-2b Phase 2: global SPARSE recall (LOSSY) */
                    if (arm_page) {   /* G1: serve the recalled set off Ring-2 (NaN-poison rigor) */
                        cudaError_t pe = cudaStreamSynchronize(st);
                        if (pe != cudaSuccess) { fail_cuda(pe, "xbar arm_page sync"); goto done; }
                        cudaMemcpy(arm_phK, dKc[L] + (size_t)pos*kvd, (size_t)kvd*4, cudaMemcpyDeviceToHost);  /* spill pos */
                        cudaMemcpy(arm_phV, dVc[L] + (size_t)pos*kvd, (size_t)kvd*4, cudaMemcpyDeviceToHost);
                        arm_be.write_block(arm_be.handle, 0, (uint64_t)arm_off[L] + (uint64_t)pos*kvd*4, arm_phK, (size_t)kvd*4);
                        arm_be.write_block(arm_be.handle, 1, (uint64_t)arm_off[L] + (uint64_t)pos*kvd*4, arm_phV, (size_t)kvd*4);
                        cudaMemset(dKc[L], 0xFF, (size_t)(pos+1)*kvd*4);  /* NaN-poison live globals [0,pos] (0xFF=NaN) */
                        cudaMemset(dVc[L], 0xFF, (size_t)(pos+1)*kvd*4);
                        memset(arm_seen, 0, (size_t)P);                  /* page back ONLY the recalled union off disk */
                        int nun = 0;
                        for (int hh = 0; hh < nh; hh++) for (int j = 0; j < arm_m_host[hh]; j++) {
                            int s = arm_ri_host[(size_t)hh*P + j];
                            if (s >= 0 && s <= pos && !arm_seen[s]) { arm_seen[s] = 1; arm_union[nun++] = s; }
                        }
                        for (int u = 0; u < nun; u++) {
                            int s = arm_union[u];
                            if (arm_be.read_block(arm_be.handle, 0, (uint64_t)arm_off[L] + (uint64_t)s*kvd*4, arm_phK, (size_t)kvd*4) ||
                                arm_be.read_block(arm_be.handle, 1, (uint64_t)arm_off[L] + (uint64_t)s*kvd*4, arm_phV, (size_t)kvd*4)) {
                                sp_set_error("xbar arm_page: read_block"); goto done; }
                            cudaMemcpy(dKc[L] + (size_t)s*kvd, arm_phK, (size_t)kvd*4, cudaMemcpyHostToDevice);
                            cudaMemcpy(dVc[L] + (size_t)s*kvd, arm_phV, (size_t)kvd*4, cudaMemcpyHostToDevice);
                        }
                    }
                    if (arm_slab) {   /* §3q C-b.2: build union {sinks∪pos∪top-B}, batch-page from RAM into compact
                                       * slots [0,m), remap per-head ri abs→compact-slot, gather over the compact slab. */
                        int mu = 0; memset(arm_seen, 0, (size_t)P);
                        #define ARM_ADD(u) do { int _u=(u); if (_u>=0 && _u<=pos && !arm_seen[_u]) { \
                            if (mu < arm_Bslab) { arm_seen[_u]=1; arm_slotmap[_u]=mu; arm_union[mu++]=_u; } else { arm_uclip++; } } } while(0)
                        for (int s = 0; s < arm_sink; s++) ARM_ADD(s);
                        ARM_ADD(pos);
                        for (int hh = 0; hh < nh; hh++) for (int j = 0; j < arm_m_host[hh]; j++) ARM_ADD(arm_ri_host[(size_t)hh*P + j]);
                        #undef ARM_ADD
                        if (mu > arm_umax) arm_umax = mu;
                        for (int s = 0; s < mu; s++) {   /* stage the union K/V contiguously, then one H2D each */
                            int u = arm_union[s];
                            memcpy(arm_stageK + (size_t)s*kvd, arm_ramK[L] + (size_t)u*kvd, (size_t)kvd*sizeof(float));
                            memcpy(arm_stageV + (size_t)s*kvd, arm_ramV[L] + (size_t)u*kvd, (size_t)kvd*sizeof(float));
                        }
                        cudaMemcpyAsync(dKc[L], arm_stageK, (size_t)mu*kvd*sizeof(float), cudaMemcpyHostToDevice, st);
                        cudaMemcpyAsync(dVc[L], arm_stageV, (size_t)mu*kvd*sizeof(float), cudaMemcpyHostToDevice, st);
                        for (int hh = 0; hh < nh; hh++) for (int j = 0; j < arm_m_host[hh]; j++) {
                            int u = arm_ri_host[(size_t)hh*P + j];
                            arm_ri_host2[(size_t)hh*arm_Bslab + j] = (u >= 0 && u <= pos && arm_seen[u]) ? arm_slotmap[u] : 0;  /* clipped→sink slot 0 */
                        }
                        cudaMemcpyAsync(arm_ri_dev, arm_ri_host2, (size_t)nh*arm_Bslab*sizeof(int), cudaMemcpyHostToDevice, st);
                        cudaMemcpyAsync(arm_m_dev,  arm_m_host,    (size_t)nh*sizeof(int),          cudaMemcpyHostToDevice, st);
                        int mm = arm_B; int bd = hd > mm ? hd : mm; if (bd > 1024) bd = 1024;
                        k_attn_decode_gather<<<nh, bd, (size_t)mm*sizeof(float), st>>>(
                            dq, dKc[L], dVc[L], arm_ri_dev, arm_m_dev, arm_Bslab, kvd, hd, grp, 1.0f, dao);
                    } else {
                        if (!arm_devsel) {   /* host-select uploads indices; devsel filled arm_ri_dev/arm_m_dev on-device */
                            cudaMemcpyAsync(arm_ri_dev, arm_ri_host, (size_t)nh*P*sizeof(int), cudaMemcpyHostToDevice, st);
                            cudaMemcpyAsync(arm_m_dev,  arm_m_host,  (size_t)nh*sizeof(int),   cudaMemcpyHostToDevice, st);
                        }
                        int mm = arm_devsel ? arm_B : (arm_max_m > 0 ? arm_max_m : 1);
                        int bd = hd > mm ? hd : mm; if (bd > 1024) bd = 1024;
                        k_attn_decode_gather<<<nh, bd, (size_t)mm*sizeof(float), st>>>(
                            dq, Kuse, Vuse, arm_ri_dev, arm_m_dev, P, kvd, hd, grp, 1.0f, dao);
                    }
                } else if (swa_ring && !global && !xbar_recall_on) {   /* P3.2-b-2a: ring read, window-only */
                    const int wl = (ctx < ws) ? ctx : ws;       /* in-window length present in the ring */
                    int bd = hd > wl ? hd : wl; if (bd > 1024) bd = 1024;
                    k_attn_decode_ring<<<nh, bd, (size_t)wl*sizeof(float), st>>>(
                        dq, Kuse, Vuse, ctx, kvd, hd, grp, 1.0f, win, Wring, dao);
                } else {
                    int bd = hd > ctx ? hd : ctx; if (bd > 1024) bd = 1024;
                    k_attn_decode_win<<<nh, bd, (size_t)ctx*sizeof(float), st>>>(
                        dq, Kuse, Vuse, ctx, kvd, hd, grp, 1.0f, win, dao);
                } }
            { static int _attn_diag = 0; cudaError_t _ae = cudaPeekAtLastError();   /* XBAR diag: name the failing attention launch (print-once) */
              if (_ae != cudaSuccess && !_attn_diag) { _attn_diag = 1;
                fprintf(stderr, "    [attn-fail] L=%d global=%d pos=%d hd=%d kvd=%d swa_ring=%d arm_slab=%d shadow=%d shared_win=%zuB: %s\n",
                        L, global, pos, hd, kvd, swa_ring, arm_slab, shadow_on, (size_t)(pos+1)*sizeof(float), cudaGetErrorString(_ae)); } }
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

            /* layer-bisect dump (FFN-residual boundary == probe attn_only=0) */
            if (probe_xs && pos == probe_pos) {
                cudaStreamSynchronize(st);
                cudaMemcpy(probe_xs + (size_t)L * E, dx, (size_t)E * sizeof(float), cudaMemcpyDeviceToHost);
            }

            if (PL) {
                MMD(&g_w.Wplig[L], dx, dpg);
                k_altup_gate<<<(unsigned)((PL+255)/256), 256, 0, st>>>(dpg, dipl, L, NL, PL, 1);
                MMD(&g_w.Wplproj[L], dpg, dpp);
                k_rmsnorm<<<1, 256, 0, st>>>(dpp, g_w.pl_post_norm[L], E, eps, dnx);
                k_add<<<(unsigned)((E+255)/256), 256, 0, st>>>(dx, dnx, (size_t)E);
            }
            if (!skip_oscale && g_w.pl_out_scale && g_w.pl_out_scale[L])
                k_scale_by_dev<<<(unsigned)((E+255)/256), 256, 0, st>>>(dx, (size_t)E, g_w.pl_out_scale[L]);

            /* SP_G4_DEC_PROBE layer telemetry: per-layer residual-stream norms at
             * the probed step AND the step before it (the good/bad pair) — a
             * catastrophic layer shows as a sudden |x| anomaly in the bad column. */
            if (probe_pos >= 0 && (pos == probe_pos || pos == probe_pos - 1)) {
                cudaStreamSynchronize(st);
                float *hx = (float *)malloc((size_t)E * sizeof(float));
                if (hx) {
                    cudaMemcpy(hx, dx, (size_t)E * sizeof(float), cudaMemcpyDeviceToHost);
                    double ss = 0.0, mx = 0.0;
                    for (int i = 0; i < E; i++) {
                        double v = hx[i]; ss += v * v; if (fabs(v) > mx) mx = fabs(v);
                    }
                    fprintf(stderr, "    [g4-dec-x] pos %d L%-2d |x| %.6e max|x| %.6e\n",
                            pos, L, sqrt(ss), mx);
                    free(hx);
                }
            }
        }

        /* ── XBAR P3.2-a shadow spill: now that the layer loop has written
         * dKc/dVc[L][pos] for every owner this step, stage all owner rows for
         * THIS pos into one pinned buffer (batched async D2H), ONE sync, then
         * write_block owners-only at off[L]+pos*kvd*4. We copy the authoritative
         * cache bytes, so the spilled store == the live cache by construction.
         * Owners only — sharers reuse the owner block and spill nothing. ── */
        if (spill_on) {
            for (int L = 0; L < kvfs && L < NL; L++) {
                const int global = ((L % period) == period - 1);
                const size_t kvd = (size_t)(global ? g_nkv : s_nkv) * (global ? g_hd : s_hd);
                const size_t poff = spill_off[L] / (size_t)P;   /* per-pos staging byte offset (exact: off = P*Σkvd*4) */
                cudaMemcpyAsync((char *)spill_hK + poff, dKc[L] + (size_t)pos*kvd, kvd*4, cudaMemcpyDeviceToHost, st);
                cudaMemcpyAsync((char *)spill_hV + poff, dVc[L] + (size_t)pos*kvd, kvd*4, cudaMemcpyDeviceToHost, st);
            }
            { cudaError_t e = cudaStreamSynchronize(st); if (e != cudaSuccess) { fail_cuda(e, "xbar spill D2H sync"); goto done; } }
            for (int L = 0; L < kvfs && L < NL; L++) {
                const int global = ((L % period) == period - 1);
                const size_t kvd = (size_t)(global ? g_nkv : s_nkv) * (global ? g_hd : s_hd);
                const size_t poff = spill_off[L] / (size_t)P;
                const uint64_t boff = (uint64_t)spill_off[L] + (uint64_t)pos*kvd*4;
                if (spill_be.write_block(spill_be.handle, 0, boff, (char *)spill_hK + poff, kvd*4) ||
                    spill_be.write_block(spill_be.handle, 1, boff, (char *)spill_hV + poff, kvd*4)) {
                    sp_set_error("xbar spill: write_block"); goto done; }
            }
        }

        /* ── XBAR P3.2-b-1 POISON: this step's positions [0,pos] are now on disk
         * (spilled above); zero them in the live cache so the NEXT step is forced
         * to page them back from Ring-2. Makes the gate non-hollow — a bit-exact
         * output then proves attention read off disk, not a stale live copy. ── */
        if (page_on) {
            for (int L = 0; L < kvfs && L < NL; L++) {
                const int global = ((L % period) == period - 1);
                const size_t kvd = (size_t)(global ? g_nkv : s_nkv) * (global ? g_hd : s_hd);
                const size_t nb = (size_t)(pos + 1) * kvd * 4;
                cudaMemsetAsync(dKc[L], 0, nb, st);
                cudaMemsetAsync(dVc[L], 0, nb, st);
            }
        }

        /* XBAR-P1 Arm-B ammunition: dump the final-layer residual x (post all
         * layers + oscale, PRE out_norm) at step ROW — the donor concept
         * token's residual, E raw f32. */
        if (xbar_res && pos == xbar_row) {
            cudaError_t xe = cudaStreamSynchronize(st);
            if (xe != cudaSuccess) { fail_cuda(xe, "xbar resid sync"); goto done; }
            float *hx = (float *)malloc((size_t)E * sizeof(float));
            FILE *rf = hx ? fopen(xbar_res, "wb") : NULL;
            if (!hx || !rf) { free(hx); if (rf) fclose(rf);
                sp_set_error("xbar resid: alloc/open"); goto done; }
            cudaMemcpy(hx, dx, (size_t)E * sizeof(float), cudaMemcpyDeviceToHost);
            fwrite(hx, sizeof(float), (size_t)E, rf);
            fclose(rf); free(hx);
            fprintf(stderr, "    [xbar] RESID dumped (E=%d f32) at step %d -> %s\n", E, pos, xbar_res);
        }

        if (score_first >= 0) {
            /* TEACHER-FORCED scoring: head + NLL at scored positions; dseq is
             * never overwritten (the given tokens are the targets). */
            if (pos + 1 >= score_first) {
                k_rmsnorm<<<1, 256, 0, st>>>(dx, g_w.out_norm, E, eps, dnx);
                if (g_w.head.f32 || g_w.head.codes) { MMD(&g_w.head, dnx, dlog); }
                else if (use_int8 && g_w.embd_packed.codes &&
                         gemv_w_packed(st, &g_w.embd_packed, dnx, dlog, dqx, dsx)) { /* dp4a head */ }
                else { if (gemm(cb, g_w.embd, dnx, dlog, 1, E, V)) goto done; }
                { static int _hd=0; cudaError_t _e=cudaPeekAtLastError(); if(_e!=cudaSuccess&&!_hd){_hd=1;
                    fprintf(stderr,"    [score-fail] phase=HEAD pos=%d V=%d E=%d: %s\n", pos, V, E, cudaGetErrorString(_e)); } }

                /* SP_G4_SCORE_DBG=<n>: THE LOGIT INTERCEPT (PPL-gate bug hunt).
                 * D2H the RAW pre-softcap row for the first n scored positions;
                 * print max/target pre-cap (post-cap is the analytic tanh) and
                 * the plateau width (#logits within 1% of the cap). */
                {
                    static int dbg_left = -2;
                    if (dbg_left == -2) {
                        const char *de = getenv("SP_G4_SCORE_DBG");
                        dbg_left = de ? atoi(de) : 0;
                    }
                    if (dbg_left > 0) {
                        dbg_left--;
                        cudaStreamSynchronize(st);
                        float *hl = (float *)malloc((size_t)V * sizeof(float));
                        if (hl) {
                            cudaMemcpy(hl, dlog, (size_t)V * sizeof(float), cudaMemcpyDeviceToHost);
                            const int tgt = seq[pos + 1];
                            float mx = hl[0]; int mi = 0; long plateau = 0;
                            for (int i = 1; i < V; i++) if (hl[i] > mx) { mx = hl[i]; mi = i; }
                            const float thr = (softcap > 0.0f) ? 3.0f * softcap : mx;  /* tanh(3)=0.995 */
                            for (int i = 0; i < V; i++) if (hl[i] >= thr) plateau++;
                            fprintf(stderr, "    [g4-score-dbg] pos %d PRE-cap: max %.4f (id %d) target[%d] %.4f "
                                            "| #pre>=3*cap %ld | POST-cap(analytic): max %.4f target %.4f\n",
                                    pos, (double)mx, mi, tgt, (double)hl[tgt], plateau,
                                    softcap > 0 ? (double)(tanhf(mx / softcap) * softcap) : (double)mx,
                                    softcap > 0 ? (double)(tanhf(hl[tgt] / softcap) * softcap) : (double)hl[tgt]);
                            free(hl);
                        }
                    }
                }

                if (softcap > 0.0f)
                    k_softcap<<<(unsigned)(((size_t)V+255)/256), 256, 0, st>>>(dlog, (size_t)V, softcap);
                k_nll<<<1, 256, 0, st>>>(dlog, V, dseq + pos + 1, dnll);
                { static int _sc=0; cudaError_t _e=cudaPeekAtLastError(); if(_e!=cudaSuccess&&!_sc){_sc=1;
                    fprintf(stderr,"    [score-fail] phase=SOFTCAP/NLL pos=%d V=%d softcap=%.3f score_first=%d: %s\n",
                            pos, V, (double)softcap, score_first, cudaGetErrorString(_e)); } }
                g_g4_score_cnt++;
            }
        } else if (gen_here && headless_probe) {
            /* diagnostic-only: no head available knobs-off on 12B-class — emit
             * token 0; the per-layer dumps above are the run's product. */
            cudaMemsetAsync(dseq + pos + 1, 0, sizeof(int), st);
            n = pos + 2;
        } else if (gen_here) {
            k_rmsnorm<<<1, 256, 0, st>>>(dx, g_w.out_norm, E, eps, dnx);
            if (g_w.head.f32 || g_w.head.codes) { MMD(&g_w.head, dnx, dlog); }
            else if (use_int8 && g_w.embd_packed.codes &&
                     gemv_w_packed(st, &g_w.embd_packed, dnx, dlog, dqx, dsx)) { /* tied head, dp4a */ }
            else { if (gemm(cb, g_w.embd, dnx, dlog, 1, E, V)) goto done; }
            if (softcap > 0.0f)
                k_softcap<<<(unsigned)(((size_t)V+255)/256), 256, 0, st>>>(dlog, (size_t)V, softcap);

            /* XBAR-P1 resonance telemetry: rank + logit of every tracked token
             * at every generated step (oracle-rank discipline — print the rank,
             * never just a boolean). Softcap is monotonic, rank is cap-invariant.
             * Host sync per step — diagnostic lane, N is small. */
            if (xbar_rkf && xbar_nids > 0) {
                cudaError_t xe = cudaStreamSynchronize(st);
                if (xe != cudaSuccess) { fail_cuda(xe, "xbar rank sync"); goto done; }
                float *hl = (float *)malloc((size_t)V * sizeof(float));
                FILE *kf = hl ? fopen(xbar_rkf, "ab") : NULL;
                if (!hl || !kf) { free(hl); if (kf) fclose(kf);
                    sp_set_error("xbar ranks: alloc/open"); goto done; }
                cudaMemcpy(hl, dlog, (size_t)V * sizeof(float), cudaMemcpyDeviceToHost);
                float bmx = hl[0]; int bmi = 0;
                for (int i = 1; i < V; i++) if (hl[i] > bmx) { bmx = hl[i]; bmi = i; }
                for (int t = 0; t < xbar_nids; t++) {
                    const int id = xbar_ids[t];
                    long rank = 1;
                    if (id >= 0 && id < V) {
                        const float lv = hl[id];
                        for (int i = 0; i < V; i++) if (hl[i] > lv) rank++;
                        fprintf(kf, "pos=%d gen=%d tok=%d rank=%ld logit=%.6f top1=%d top1_logit=%.6f\n",
                                pos, pos - (n_prompt - 1), id, rank, (double)lv, bmi, (double)bmx);
                    }
                }
                fclose(kf); free(hl);
            }

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
    if (shadow_on && armR && !arm_slab) {   /* ── P3.2-b-2b Phase 1 gate: G-P3-GEOM.a oracle parity ──
        * The K fed to the router must BE the K attention reads. Re-project the FINAL global
        * K cache fresh and compare to the incrementally-minted projk: mismatch ⇒ a wrong-K
        * capture or a sidecar-indexing fault (the bug class Phase 1 isolates from router logic).
        * SKIPPED under arm_slab (C-b.2): the compact slab holds only arm_Bslab<P slots, so the
        * p2∈[0,P) re-projection would read past dKc[L] (cudaErrorInvalidValue). The parity was
        * validated in Phase 1 without the slab; the slab cannot host the full-P re-projection. */
        cudaStreamSynchronize(st);
        long mism = 0; float oproj[SP_ARM_R_MAX];
        for (int L = 0; L < kvfs && L < NL; L++) {
            if (((L % period) == period - 1) == 0) continue;   /* globals only (L%6==5) */
            const int kvd2 = armG[L].nkv * armG[L].hd;
            for (int p2 = 0; p2 < P - 1; p2++) {               /* the loop minted [0,P-1) */
                cudaMemcpy(arm_kh, dKc[L] + (size_t)p2*kvd2, (size_t)kvd2*sizeof(float), cudaMemcpyDeviceToHost);
                sp_arm_project_geom(armR, arm_r, arm_hd_max, armG[L].hd, arm_kh, oproj);
                const float *inc = arm_projk + armG[L].off + (size_t)p2*armG[L].nkv*arm_r;
                for (int p = 0; p < arm_r; p++) if (oproj[p] != inc[p]) mism++;
            }
        }
        g_arm_shadow_mism = mism; g_arm_shadow_sel = arm_sel;
        fprintf(stderr, "    [xbar-p3.2b2b] G-P3-GEOM.a oracle parity: global projk fresh-vs-incremental mismatches=%ld; selections=%ld avg|recall|=%.1f => %s\n",
                mism, arm_sel, arm_sel ? (double)arm_msum/arm_sel : 0.0, mism == 0 ? "PARITY OK" : "MISMATCH");
    }
    if (probe_xs) {   /* flush the layer-bisect dump (diagnostic only) */
        FILE *pf = fopen(pdump, "wb");
        if (pf) { fwrite(probe_xs, sizeof(float), (size_t)NL * E, pf); fclose(pf); }
        fprintf(stderr, "    [g4-dec-probe] dumped %d x-vectors (E=%d) to %s\n", NL, E, pdump);
    }
    if (xbar_wr_e && d_xbar_K && xmf.layers) {   /* P3.1b: serialize manifest + dump filled store to disk (write-once) */
        static int xbar_written = 0;
        if (!xbar_written) {
            xbar_written = 1; cudaStreamSynchronize(st);
            char wp[1024]; FILE *wf;
            size_t msz = sp_xbar_manifest_serial_size(&xmf);
            uint8_t *mb = (uint8_t *)malloc(msz);
            if (mb && sp_xbar_manifest_serialize(&xmf, mb, msz) == msz) {
                snprintf(wp, sizeof(wp), "%s/ep.mf", xbar_wr_e); wf = fopen(wp, "wb");
                if (wf) { fwrite(mb, 1, msz, wf); fclose(wf); }
            }
            free(mb);
            uint8_t *hs = (uint8_t *)malloc(xbar_store_bytes);
            if (hs) {
                cudaMemcpy(hs, d_xbar_K, xbar_store_bytes, cudaMemcpyDeviceToHost);
                snprintf(wp, sizeof(wp), "%s/ep.k", xbar_wr_e); wf = fopen(wp, "wb"); if (wf) { fwrite(hs, 1, xbar_store_bytes, wf); fclose(wf); }
                cudaMemcpy(hs, d_xbar_V, xbar_store_bytes, cudaMemcpyDeviceToHost);
                snprintf(wp, sizeof(wp), "%s/ep.v", xbar_wr_e); wf = fopen(wp, "wb"); if (wf) { fwrite(hs, 1, xbar_store_bytes, wf); fclose(wf); }
                free(hs);
            }
            fprintf(stderr, "    [xbar-p3.1b] WROTE episode (manifest %zu B + K/V store %.1f MiB) -> %s\n",
                    msz, (double)xbar_store_bytes / 1048576.0, xbar_wr_e);
        }
    }
    if (spill_on && spill_be.read_block) {   /* ── G-P3-R2.a byte-identity gate ──
        * Read every spilled owner block back from the Ring-2 store and memcmp vs a
        * fresh D2H of the FINAL live cache dKc/dVc[L][0..P-1). The store was filled
        * per-step from the same cache bytes, and nothing evicts in P3.2-a, so the
        * two must be byte-identical. Owners only — no sharer block exists in the
        * store (sharers spilled nothing). diffs==0 == PASS. */
        cudaStreamSynchronize(st);
        const size_t Pm1 = (size_t)(P - 1);   /* the loop minted KV for pos in [0,P-1) */
        size_t maxbytes = 0;
        for (int L = 0; L < kvfs && L < NL; L++) {
            const int global = ((L % period) == period - 1);
            const size_t kvd = (size_t)(global ? g_nkv : s_nkv) * (global ? g_hd : s_hd);
            if (kvd * Pm1 * 4 > maxbytes) maxbytes = kvd * Pm1 * 4;
        }
        float *cref = (float *)malloc(maxbytes ? maxbytes : 4);
        uint8_t *sbk = (uint8_t *)malloc(maxbytes ? maxbytes : 4);
        size_t diffs = 0; int owners = 0; int io_err = 0;
        if (cref && sbk) {
            for (int two = 0; two < 2 && !io_err; two++) {
                for (int L = 0; L < kvfs && L < NL; L++) {
                    const int global = ((L % period) == period - 1);
                    const size_t kvd = (size_t)(global ? g_nkv : s_nkv) * (global ? g_hd : s_hd);
                    const size_t nb = kvd * Pm1 * 4;
                    cudaMemcpy(cref, two ? dVc[L] : dKc[L], nb, cudaMemcpyDeviceToHost);
                    if (spill_be.read_block(spill_be.handle, two, (uint64_t)spill_off[L], sbk, nb)) { io_err = 1; break; }
                    const uint8_t *a = (const uint8_t *)cref;
                    for (size_t i = 0; i < nb; i++) if (a[i] != sbk[i]) diffs++;
                    if (two == 0) owners++;
                }
            }
        }
        free(cref); free(sbk);
        fprintf(stderr, "    [xbar-p3.2a] G-P3-R2.a byte-identity: owners=%d positions=%zu store=%.1f MiB diffs=%zu%s => %s\n",
                owners, Pm1, (double)spill_store_bytes / 1048576.0, diffs,
                io_err ? " (READ-BLOCK IO ERR)" : "", (diffs == 0 && !io_err) ? "PASS" : "FAIL");
    }
    if (dnll) {       /* SP_G4_SCORE: pull the accumulated NLL */
        cudaStreamSynchronize(st);
        cudaMemcpy(&g_g4_score_nll, dnll, sizeof(double), cudaMemcpyDeviceToHost);
    }
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
    if (d_xbar_K) cudaFree(d_xbar_K); if (d_xbar_V) cudaFree(d_xbar_V); if (xbar_off) free(xbar_off); sp_xbar_manifest_free(&xmf);  /* P3.1/b recall store */
    if (spill_on && spill_be.close) spill_be.close(spill_be.handle);  /* P3.2-a Ring-2 spill store */
    if (spill_off) free(spill_off); if (spill_hK) cudaFreeHost(spill_hK); if (spill_hV) cudaFreeHost(spill_hV);
    if (page_hK) cudaFreeHost(page_hK); if (page_hV) cudaFreeHost(page_hV);  /* P3.2-b-1 page-in staging */
    if (armR) free(armR); if (armG) free(armG); if (arm_projk) free(arm_projk);  /* P3.2-b-2b shadow router */
    if (arm_cand) free(arm_cand); if (arm_ri) free(arm_ri); if (arm_kh) free(arm_kh); if (arm_qh) free(arm_qh);
    if (arm_dump_f) fclose(arm_dump_f);  /* §3q Phase A K/q dump */
    if (arm_dscores) cudaFree(arm_dscores); if (arm_scores_h) free(arm_scores_h);  /* §3q oracle ceiling */
    if (arm_hs) free(arm_hs); if (arm_hi) free(arm_hi);
    if (arm_dM) cudaFree(arm_dM); if (arm_dqp) cudaFree(arm_dqp);  /* §3q Learned-LSH */
    if (arm_dR) cudaFree(arm_dR); if (arm_pk) cudaFree(arm_pk); if (arm_Rq) cudaFree(arm_Rq);  /* §3q C-b.1 sidecar */
    if (arm_slab) {   /* §3q C-b.2: report the measured union size, free the host RAM store + scratch */
        fprintf(stderr, "    [xbar-slab] C-b.2 union telemetry: max per-step union = %ld (slab depth %d, B=%d); clip events = %ld\n",
                arm_umax, arm_Bslab, arm_B, arm_uclip);
        if (arm_ramK) { for (int L = 0; L < NL; L++) if (arm_ramK[L]) free(arm_ramK[L]); free(arm_ramK); }
        if (arm_ramV) { for (int L = 0; L < NL; L++) if (arm_ramV[L]) free(arm_ramV[L]); free(arm_ramV); }
        if (arm_stageK) free(arm_stageK); if (arm_stageV) free(arm_stageV);
        if (arm_slotmap) free(arm_slotmap); if (arm_ri_host2) free(arm_ri_host2);
    }
    if (arm_ri_host) free(arm_ri_host); if (arm_m_host) free(arm_m_host);  /* P3.2-b-2b gather buffers */
    if (arm_ri_dev) cudaFree(arm_ri_dev); if (arm_m_dev) cudaFree(arm_m_dev);
    if (arm_page && arm_be.close) arm_be.close(arm_be.handle);  /* P3.2-b-2b G1 Ring-2 store */
    if (arm_off) free(arm_off); if (arm_phK) free(arm_phK); if (arm_phV) free(arm_phV);
    if (arm_seen) free(arm_seen); if (arm_union) free(arm_union);
    if (dVc) { for (int L = 0; L < NL; L++) if (dVc[L]) cudaFree(dVc[L]); free(dVc); }
    if (dnll) cudaFree(dnll);
    free(probe_xs);
    if (xbar_embc_f) fclose(xbar_embc_f);
    free(xbar_emb_rows);
    #undef MMD
    return rc;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * KAI-1b METAL EVICTION — persistent-KV gemma4 decode + O(1) rewind.
 *
 * gemma4_decode_cuda (above) is ONE-SHOT (alloc cache → prefill+gen → free) and
 * is left BYTE-FOR-BYTE UNTOUCHED (the null floor: every citable gate — 06-R10,
 * X-R2, NIAH, KAIROS-01 — keeps running it unchanged). This is the resident
 * twin: a persistent cache the KAIROS heartbeat appends to and shears.
 *
 *   gemma4_kv_open   → alloc resident KV (full cache, P=Pmax) + scratch, dpos=0
 *   gemma4_kv_prefill→ forward n tokens, store K/V at [dpos,dpos+n), dpos+=n
 *   gemma4_kv_decode → greedy-argmax n_gen steps, append to cache + dseq
 *   gemma4_kv_rewind → O(1): dpos-=Δ (full cache slot=pos, so writes to [a,..)
 *                       NEVER touch [0,a) ⇒ rewind is a perfect inverse, T8.1)
 *   gemma4_kv_pos / gemma4_kv_close
 *
 * Per-position forward = the EXACT per-step oracle path (kernels + order copied
 * from the generate step), using DEVICE dpos so cache writes land at the live
 * position. Bit-exact to the one-shot split across calls because cache state at
 * position p depends only on tokens [0,p], not on call chunking.
 * Scope: FULL cache only (windowed attention via `win` handles SWA correctly);
 * the SWA-ring/slab variants are a follow-on (they need wrap-aware rewind). ══ */
struct sp_g4_kv {
    const qwen3_model *m;
    int   Pmax, dpos_host, use_int8, dev_ple;
    int  *dseq, *dpos;
    float **dKc, **dVc;
    float *dx,*dnx,*dq,*dk,*dv,*dao,*dap,*dg,*dup,*ddn,*dscr,*dlog,*dipl,*dple,*dpg,*dpp;
    signed char *dqx; float *dsx; float *hple;
    /* cached geometry/scales */
    int   E,NL,V,SW,period,kvfs,PL,NLPL,FFmax,QDmax,KVDmax;
    int   g_nh,g_nkv,g_hd,s_nh,s_nkv,s_hd;
    float g_base,s_base,eps,embscale,softcap;
    /* KAI-1c wrap-aware ring: SWA owners use a Wring-slot ring (slot=pos%Wring);
     * the per-tick undo-journal saves each clobbered slot BEFORE overwrite so
     * rewind restores it (the wrap-crossing perfect inverse). ring_W=0 ⇒ full
     * cache (KAI-1b, every owner Pmax slots, no journal). commit_pos = the
     * baseline a rewind cannot pass (cleared by gemma4_kv_commit on ACTION);
     * jcur = journal depth = dpos_host - commit_pos. jK/jV[L] = [Jmax*kvd] per
     * SWA owner; globals stay full-cache (no window ⇒ no alias ⇒ no journal). */
    int   ring_W, Jmax, commit_pos, jcur;
    float **jK, **jV;
    /* KAI-2 latent interrupt: residual-entry capture/inject at the post-embed/pre-layer
     * point (the SP_XBAR_EMB seam, persistent-ABI edition). Off by default = null floor.
     * cap_active ⇒ next step D2H its post-embed dx into cap_host; inj_active ⇒ next step
     * overrides dx with dinj (E floats). One-shot flags, cleared after the step consumes them. */
    float *dinj; float *cap_host; int inj_active, cap_active;
};

extern "C" void gemma4_kv_close(sp_g4_kv *s);   /* fwd: gemma4_kv_open frees on OOM */

/* one resident-cache forward step at the live dpos. do_head: run out_norm+head+
 * argmax (writes dseq[dpos+1]) for a decode step; skip for a prefill ingest.
 * Always: embed dseq[dpos] → PLE/AltUp → 48-layer stack (store K/V at dpos,
 * windowed attention over [0,dpos]) → optional head → k_incr_pos. */
static int g4_kv_step(sp_g4_kv *s, int do_head) {
    const qwen3_model *m = s->m; const qwen3_config *c = &m->cfg;
    cublasHandle_t cb = g_w.cublas; cudaStream_t st = g_w.stream;
    const int E=s->E, NL=s->NL, V=s->V, SW=s->SW, period=s->period, kvfs=s->kvfs;
    const int PL=s->PL, NLPL=s->NLPL;
    const int g_nh=s->g_nh,g_nkv=s->g_nkv,g_hd=s->g_hd,s_nh=s->s_nh,s_nkv=s->s_nkv,s_hd=s->s_hd;
    const float eps=s->eps, embscale=s->embscale, softcap=s->softcap;
    const float g_base=s->g_base, s_base=s->s_base;
    const int use_int8=s->use_int8;
    int *dpos=s->dpos, *dseq=s->dseq;
    float **dKc=s->dKc, **dVc=s->dVc;
    const size_t attn_shm = (size_t)s->Pmax * sizeof(float);
    #define KMMD(W,X,Y) do { if (!(use_int8 && gemv_w_packed(st,(W),(X),(Y),s->dqx,s->dsx))) { \
        if (gemm_w_lift(cb, st, (W), (X), (Y), 1, s->dscr)) return -1; } } while (0)
    /* embed dseq[dpos] (device position) */
    if (g_w.embd) { dim3 grid(1, (E + 255) / 256);
        k_embed_scale_at<<<grid, 256, 0, st>>>(g_w.embd, dseq, dpos, E, embscale, s->dx); }
    else k_embed_packed_at<<<(unsigned)((E+255)/256), 256, 0, st>>>(
            g_w.embd_packed.codes, g_w.embd_packed.row_off, g_w.embd_packed.row_scale,
            g_w.embd_packed.row_prec, dseq, dpos, E, embscale, s->dx);
    /* KAI-2: residual-entry capture/inject (post-embed, pre-layer — the SP_XBAR_EMB point).
     * Off by default (both flags 0) ⇒ identical to a stock step = the null floor. */
    if (s->cap_active && s->cap_host) {
        cudaStreamSynchronize(st);
        cudaMemcpy(s->cap_host, s->dx, (size_t)E*sizeof(float), cudaMemcpyDeviceToHost);
        s->cap_active = 0;
    }
    int injecting = 0;
    if (s->inj_active && s->dinj) {
        cudaMemcpyAsync(s->dx, s->dinj, (size_t)E*sizeof(float), cudaMemcpyDeviceToDevice, st);
        s->inj_active = 0; injecting = 1;
    }
    if (PL) {
        k_ple_gather_at<<<(unsigned)((NLPL+255)/256), 256, 0, st>>>(
            g_w.pl_tok_embd.codes, g_w.pl_tok_embd.row_off, g_w.pl_tok_embd.row_scale,
            g_w.pl_tok_embd.row_prec, dseq, dpos, NLPL, s->dple);
        /* KAI-2 PLE-suppression probe (SP_KAI2_INJ_NOPLE): gemma4's AltUp folds per-layer embeddings
         * gathered from the PLACEHOLDER token id (dseq[dpos]) into the residual at EVERY layer. The
         * inject overrides only the main embedding stream (dx), so the placeholder token leaks back in
         * via PLE — and the codec was distilled against an inputs_embeds forward that had no token-PLE.
         * Zeroing dple at the injected position removes that leak so serving matches training. */
        static int nople = -1;
        if (nople < 0) { const char *e = getenv("SP_KAI2_INJ_NOPLE"); nople = (e && e[0]=='1') ? 1 : 0; }
        if (injecting && nople) cudaMemsetAsync(s->dple, 0, (size_t)NLPL*sizeof(float), st);
        KMMD(&g_w.pl_model_proj, s->dx, s->dipl);
        k_altup_ipl<<<NL, 256, 0, st>>>(s->dipl, s->dple, g_w.pl_proj_norm, PL,
                                        1.0f / sqrtf((float)E), sqrtf((float)PL),
                                        1.0f / sqrtf(2.0f), eps);
    }
    for (int L = 0; L < NL; L++) {
        const int global = ((L % period) == period - 1);
        const int nh = global ? g_nh : s_nh, nkv = global ? g_nkv : s_nkv;
        const int hd = global ? g_hd : s_hd;
        const int grp = nh / nkv, kvd = nkv * hd;
        const float rbase = global ? g_base : s_base;
        const float *ffac = global ? g_w.rope_freqs : NULL;
        const int win = global ? -1 : SW;
        const int ffL = g_w.Wgate[L].out;
        k_rmsnorm<<<1, 256, 0, st>>>(s->dx, g_w.attn_norm[L], E, eps, s->dnx);
        KMMD(&g_w.Wq[L], s->dnx, s->dq);
        k_rmsnorm_head<<<nh, 256, 0, st>>>(s->dq, g_w.q_norm[L], nh, hd, nh*hd, eps);
        if (ffac) k_rope_freqs_dyn<<<nh, hd/2, 0, st>>>(s->dq, nh, hd, rbase, ffac, dpos);
        else      k_rope_dyn<<<nh, hd/2, 0, st>>>(s->dq, nh, hd, nh*hd, rbase, dpos);
        float *Kuse, *Vuse;
        if (L < kvfs) {
            KMMD(&g_w.Wk[L], s->dnx, s->dk);
            if (g_w.Wv[L].f32 || g_w.Wv[L].codes) { KMMD(&g_w.Wv[L], s->dnx, s->dv); }
            else cudaMemcpyAsync(s->dv, s->dk, (size_t)kvd*sizeof(float), cudaMemcpyDeviceToDevice, st);
            k_rmsnorm_head<<<nkv, 256, 0, st>>>(s->dk, g_w.k_norm[L], nkv, hd, kvd, eps);
            if (ffac) k_rope_freqs_dyn<<<nkv, hd/2, 0, st>>>(s->dk, nkv, hd, rbase, ffac, dpos);
            else      k_rope_dyn<<<nkv, hd/2, 0, st>>>(s->dk, nkv, hd, kvd, rbase, dpos);
            k_rmsnorm_head_noweight<<<nkv, 256, 0, st>>>(s->dv, nkv, hd, kvd, eps);
            if (s->ring_W > 0 && !global) {            /* KAI-1c: SWA-owner ring write + undo-journal */
                const size_t ws = (size_t)(s->dpos_host % s->ring_W);
                const size_t j  = (size_t)(s->dpos_host - s->commit_pos);   /* journal index for this step */
                if (j >= (size_t)s->Jmax) { sp_set_error("g4_kv ring: uncommitted tick span exceeds Jmax"); return -1; }
                /* save the slot's pre-write K/V (the clobbered live-window position) BEFORE overwrite */
                cudaMemcpyAsync(s->jK[L] + j*kvd, dKc[L] + ws*kvd, (size_t)kvd*sizeof(float), cudaMemcpyDeviceToDevice, st);
                cudaMemcpyAsync(s->jV[L] + j*kvd, dVc[L] + ws*kvd, (size_t)kvd*sizeof(float), cudaMemcpyDeviceToDevice, st);
                cudaMemcpyAsync(dKc[L] + ws*kvd, s->dk, (size_t)kvd*sizeof(float), cudaMemcpyDeviceToDevice, st);
                cudaMemcpyAsync(dVc[L] + ws*kvd, s->dv, (size_t)kvd*sizeof(float), cudaMemcpyDeviceToDevice, st);
            } else {
                k_kv_store<<<(unsigned)((kvd+255)/256), 256, 0, st>>>(dKc[L], dVc[L], s->dk, s->dv, dpos, 0, kvd);
            }
            Kuse = dKc[L]; Vuse = dVc[L];
        } else {
            const int src = kvfs - (global ? 1 : 2);
            Kuse = dKc[src]; Vuse = dVc[src];
            if (!Kuse || !Vuse) { sp_set_error("g4_kv: sharer before owner"); return -1; }
        }
        {   int bd = hd > 256 ? hd : 256; if (bd > 1024) bd = 1024;
            if (s->ring_W > 0 && !global) {           /* KAI-1c: ring attention (slot=(s0+j)%Wring) */
                const int ctx = s->dpos_host + 1;
                const int s0 = (win >= 0 && ctx - win > 0) ? ctx - win : 0;
                const int wl = ctx - s0;
                k_attn_decode_ring<<<nh, bd, (size_t)wl*sizeof(float), st>>>(
                    s->dq, Kuse, Vuse, ctx, kvd, hd, grp, 1.0f, win, s->ring_W, s->dao);
            } else {
                k_attn_decode_win_dyn<<<nh, bd, attn_shm, st>>>(
                    s->dq, Kuse, Vuse, dpos, kvd, hd, grp, 1.0f, win, s->dao);
            } }
        KMMD(&g_w.Wo[L], s->dao, s->dap);
        k_rmsnorm<<<1, 256, 0, st>>>(s->dap, g_w.post_attn[L], E, eps, s->dnx);
        k_add<<<(unsigned)((E+255)/256), 256, 0, st>>>(s->dx, s->dnx, (size_t)E);
        k_rmsnorm<<<1, 256, 0, st>>>(s->dx, g_w.ffn_norm[L], E, eps, s->dnx);
        KMMD(&g_w.Wgate[L], s->dnx, s->dg);
        KMMD(&g_w.Wup[L], s->dnx, s->dup);
        k_gelu_mul<<<(unsigned)(((size_t)ffL+255)/256), 256, 0, st>>>(s->dg, s->dup, (size_t)ffL);
        KMMD(&g_w.Wdown[L], s->dg, s->ddn);
        k_rmsnorm<<<1, 256, 0, st>>>(s->ddn, g_w.post_ffw[L], E, eps, s->dnx);
        k_add<<<(unsigned)((E+255)/256), 256, 0, st>>>(s->dx, s->dnx, (size_t)E);
        if (PL) {
            KMMD(&g_w.Wplig[L], s->dx, s->dpg);
            k_altup_gate<<<(unsigned)((PL+255)/256), 256, 0, st>>>(s->dpg, s->dipl, L, NL, PL, 1);
            KMMD(&g_w.Wplproj[L], s->dpg, s->dpp);
            k_rmsnorm<<<1, 256, 0, st>>>(s->dpp, g_w.pl_post_norm[L], E, eps, s->dnx);
            k_add<<<(unsigned)((E+255)/256), 256, 0, st>>>(s->dx, s->dnx, (size_t)E);
        }
        if (g_w.pl_out_scale && g_w.pl_out_scale[L])
            k_scale_by_dev<<<(unsigned)((E+255)/256), 256, 0, st>>>(s->dx, (size_t)E, g_w.pl_out_scale[L]);
    }
    if (do_head) {
        k_rmsnorm<<<1, 256, 0, st>>>(s->dx, g_w.out_norm, E, eps, s->dnx);
        if (g_w.head.f32 || g_w.head.codes) { KMMD(&g_w.head, s->dnx, s->dlog); }
        else if (use_int8 && g_w.embd_packed.codes &&
                 gemv_w_packed(st, &g_w.embd_packed, s->dnx, s->dlog, s->dqx, s->dsx)) { }
        else { if (gemm(cb, g_w.embd, s->dnx, s->dlog, 1, E, V)) return -1; }
        if (softcap > 0.0f)
            k_softcap<<<(unsigned)(((size_t)V+255)/256), 256, 0, st>>>(s->dlog, (size_t)V, softcap);
        k_argmax_at<<<1, 256, 0, st>>>(s->dlog, V, dseq, dpos);   /* writes dseq[*dpos+1] */
    }
    k_incr_pos<<<1, 1, 0, st>>>(dpos);
    (void)c;
    #undef KMMD
    return 0;
}

extern "C" sp_g4_kv *gemma4_kv_open(const qwen3_model *m, int Pmax) {
    if (!m || m->cfg.arch != SP_ARCH_GEMMA4 || Pmax <= 0) { sp_set_error("gemma4_kv_open: bad args"); return NULL; }
    if (g_w.key != m) { free_weights(&g_w); if (build_weights(m, &g_w)) return NULL; }
    const qwen3_config *c = &m->cfg;
    sp_g4_kv *s = (sp_g4_kv *)calloc(1, sizeof(sp_g4_kv));
    if (!s) { sp_set_error("gemma4_kv_open: host OOM"); return NULL; }
    s->m = m; s->Pmax = Pmax; s->dpos_host = 0;
    s->E=(int)c->n_embd; s->NL=(int)c->n_layers; s->V=(int)c->n_vocab; s->SW=(int)c->sliding_window;
    s->period=(int)c->g4_swa_period?(int)c->g4_swa_period:6;
    s->kvfs=(int)c->g4_n_kv_from_start?(int)c->g4_n_kv_from_start:s->NL;
    s->PL=(int)c->g4_n_embd_per_layer; s->NLPL=s->NL*s->PL;
    s->g_nh=(int)c->n_head; s->g_nkv=(int)c->n_head_kv; s->g_hd=(int)c->head_dim;
    s->s_nh=(int)c->g4_nh_swa; s->s_nkv=(int)c->g4_nkv_swa; s->s_hd=(int)c->g4_hd_swa;
    s->g_base=c->rope_freq_base; s->s_base=c->g4_rope_base_swa;
    s->eps=c->rms_eps; s->embscale=sqrtf((float)s->E); s->softcap=c->g4_logit_softcap;
    s->QDmax=(s->g_nh*s->g_hd>s->s_nh*s->s_hd)?s->g_nh*s->g_hd:s->s_nh*s->s_hd;
    s->KVDmax=(s->g_nkv*s->g_hd>s->s_nkv*s->s_hd)?s->g_nkv*s->g_hd:s->s_nkv*s->s_hd;
    s->FFmax=(int)c->n_ff; for (int L=0; L<s->NL; L++) if (g_w.Wgate[L].out>s->FFmax) s->FFmax=g_w.Wgate[L].out;
    const char *i8e = getenv("SP_CUDA_DECODE_INT8");
    s->use_int8 = (i8e && i8e[0]=='1') && m->arena!=NULL;
    s->dev_ple = (s->PL && g_w.pl_tok_embd.codes!=NULL);
    if (!(g_w.head.f32 || g_w.head.codes) && !g_w.embd && !(s->use_int8 && g_w.embd_packed.codes)) {
        sp_set_error("gemma4_kv_open: tied head needs SP_CUDA_DECODE_INT8=1"); free(s); return NULL; }
    int ok = 1;
    #define KA(p,cnt) do { if (cudaMalloc(&(p),(size_t)(cnt)*sizeof(float))!=cudaSuccess) ok=0; } while(0)
    if (cudaMalloc(&s->dseq,(size_t)Pmax*sizeof(int))!=cudaSuccess) ok=0;
    if (cudaMalloc(&s->dpos,sizeof(int))!=cudaSuccess) ok=0;
    cudaMemset(s->dpos, 0, sizeof(int));
    KA(s->dx,s->E);KA(s->dnx,s->E);KA(s->dq,s->QDmax);KA(s->dk,s->KVDmax);KA(s->dv,s->KVDmax);
    KA(s->dao,s->QDmax);KA(s->dap,s->E);KA(s->dg,s->FFmax);KA(s->dup,s->FFmax);KA(s->ddn,s->E);KA(s->dlog,s->V);
    if (g_w.scratch_n) KA(s->dscr,g_w.scratch_n);
    if (s->PL){ KA(s->dipl,s->NLPL);KA(s->dple,s->NLPL);KA(s->dpg,s->PL);KA(s->dpp,s->E); }
    if (s->use_int8){ int mx=s->E; if(s->QDmax>mx)mx=s->QDmax; if(s->FFmax>mx)mx=s->FFmax;
        size_t npad=(size_t)((mx+31)&~31);
        if (cudaMalloc(&s->dqx,npad)!=cudaSuccess) ok=0; KA(s->dsx,npad>>4); }
    /* KAI-1c ring config (env-gated; unset = full cache = KAI-1b base). SWA owners
     * shrink to a ring_W-slot ring (the X-R2 space win) + a Jmax-deep undo-journal;
     * globals stay full-cache (no window ⇒ no alias). */
    s->ring_W = getenv("SP_G4_KV_RING_W") ? atoi(getenv("SP_G4_KV_RING_W")) : 0;
    s->Jmax   = getenv("SP_G4_KV_JMAX")   ? atoi(getenv("SP_G4_KV_JMAX"))   : 64;
    if (s->ring_W < 0 || s->ring_W > Pmax) s->ring_W = 0;
    if (s->Jmax < 1) s->Jmax = 1;
    s->commit_pos = 0; s->jcur = 0; s->jK = NULL; s->jV = NULL;
    /* the JAGGED cache — owners only, per-layer width. */
    s->dKc=(float**)calloc((size_t)s->NL,sizeof(float*));
    s->dVc=(float**)calloc((size_t)s->NL,sizeof(float*));
    if (!s->dKc||!s->dVc) ok=0;
    if (s->ring_W>0){ s->jK=(float**)calloc((size_t)s->NL,sizeof(float*)); s->jV=(float**)calloc((size_t)s->NL,sizeof(float*)); if(!s->jK||!s->jV) ok=0; }
    for (int L=0; ok && L<s->kvfs && L<s->NL; L++){
        const int global=((L%s->period)==s->period-1);
        const int kvd=(global?s->g_nkv:s->s_nkv)*(global?s->g_hd:s->s_hd);
        const size_t slots=(s->ring_W>0 && !global)?(size_t)s->ring_W:(size_t)Pmax;
        KA(s->dKc[L],slots*kvd); KA(s->dVc[L],slots*kvd);
        if (s->ring_W>0 && !global){ KA(s->jK[L],(size_t)s->Jmax*kvd); KA(s->jV[L],(size_t)s->Jmax*kvd); }
    }
    if (s->ring_W>0) fprintf(stderr,"    [g4-kv] RING mode: Wring=%d Jmax=%d (SWA owners ring+journal; globals full-cache)\n",s->ring_W,s->Jmax);
    #undef KA
    if (!ok) { sp_set_error("gemma4_kv_open: device OOM"); gemma4_kv_close(s); return NULL; }
    return s;
}

extern "C" int gemma4_kv_prefill(sp_g4_kv *s, const int32_t *toks, int n) {
    if (!s || !toks || n <= 0) { sp_set_error("gemma4_kv_prefill: bad args"); return -1; }
    if (s->dpos_host + n > s->Pmax) { sp_set_error("gemma4_kv_prefill: exceeds Pmax"); return -1; }
    cudaStream_t st = g_w.stream;
    if (cudaMemcpyAsync(s->dseq + s->dpos_host, toks, (size_t)n*sizeof(int),
                        cudaMemcpyHostToDevice, st) != cudaSuccess) { sp_set_error("g4_kv prefill H2D"); return -1; }
    for (int i = 0; i < n; i++) {                 /* every position: forward + store K/V; last runs the head */
        if (g4_kv_step(s, /*do_head=*/(i == n - 1))) return -1;
        s->dpos_host++;
    }
    cudaError_t e = cudaStreamSynchronize(st);
    if (e != cudaSuccess) return fail_cuda(e, "g4_kv prefill sync");
    return 0;
}

extern "C" int gemma4_kv_decode(sp_g4_kv *s, int n_gen, int32_t *out) {
    if (!s || n_gen < 0) { sp_set_error("gemma4_kv_decode: bad args"); return -1; }
    if (s->dpos_host + n_gen >= s->Pmax) { sp_set_error("gemma4_kv_decode: exceeds Pmax"); return -1; }
    cudaStream_t st = g_w.stream;
    for (int g = 0; g < n_gen; g++) {             /* process the just-predicted token, predict next */
        if (g4_kv_step(s, /*do_head=*/1)) return -1;
        s->dpos_host++;
        if (out && cudaMemcpyAsync(&out[g], s->dseq + s->dpos_host, sizeof(int),
                                   cudaMemcpyDeviceToHost, st) != cudaSuccess) { sp_set_error("g4_kv decode D2H"); return -1; }
    }
    cudaError_t e = cudaStreamSynchronize(st);
    if (e != cudaSuccess) return fail_cuda(e, "g4_kv decode sync");
    return 0;
}

/* O(1) cold-evict: shear the logical decode position back by delta. Full cache
 * slot==pos ⇒ the sheared slots [dpos-delta, dpos) are never read (attention
 * bounds at dpos) and are overwritten on the next prefill ⇒ rewind is a perfect
 * inverse (the G-1b-REWIND-NULL invariant; T8.1 analog on the GPU). */
extern "C" int gemma4_kv_rewind(sp_g4_kv *s, int delta) {
    if (!s || delta < 0 || delta > s->dpos_host) { sp_set_error("gemma4_kv_rewind: bad delta"); return -1; }
    if (s->ring_W > 0) {
        /* KAI-1c wrap-aware: restore each clobbered SWA ring slot from the journal.
         * REVERSE order (highest position first) so on intra-tick slot reuse the
         * EARLIEST snapshot (the true pre-tick value) wins the final write. */
        if (delta > s->dpos_host - s->commit_pos) { sp_set_error("gemma4_kv_rewind: delta crosses a commit (journal cleared)"); return -1; }
        cudaStream_t st = g_w.stream;
        for (int d = 0; d < delta; d++) {
            const int p  = s->dpos_host - 1 - d;          /* a position written this tick */
            const size_t ws = (size_t)(p % s->ring_W);
            const size_t j  = (size_t)(p - s->commit_pos);/* its journal slot */
            for (int L = 0; L < s->kvfs && L < s->NL; L++) {
                if (((L % s->period) == s->period - 1)) continue;   /* globals: full-cache, no journal */
                const int kvd = s->s_nkv * s->s_hd;
                cudaMemcpyAsync(s->dKc[L] + ws*kvd, s->jK[L] + j*kvd, (size_t)kvd*sizeof(float), cudaMemcpyDeviceToDevice, st);
                cudaMemcpyAsync(s->dVc[L] + ws*kvd, s->jV[L] + j*kvd, (size_t)kvd*sizeof(float), cudaMemcpyDeviceToDevice, st);
            }
        }
        cudaStreamSynchronize(st);
    }
    s->dpos_host -= delta;
    if (cudaMemcpy(s->dpos, &s->dpos_host, sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
        sp_set_error("g4_kv rewind dpos H2D"); return -1; }
    return 0;
}

/* KAI-1c: an ACTION is retained — the current cache becomes permanent and the
 * undo-journal window resets (so the next idle tick's journal starts fresh and a
 * later rewind cannot pass this baseline). For the full cache this is a no-op. */
extern "C" int gemma4_kv_commit(sp_g4_kv *s) {
    if (!s) return -1;
    s->commit_pos = s->dpos_host;
    s->jcur = 0;
    return 0;
}

extern "C" int gemma4_kv_pos(const sp_g4_kv *s) { return s ? s->dpos_host : -1; }

/* In-place re-anchor: reset the session to empty WITHOUT free/realloc (the soak leak fix —
 * per-loop close/reopen fragmented VRAM ⇒ kv_open OOM at ~loop 209). Buffers stay allocated;
 * only the position counters reset. Next prefill overwrites slots from pos=0; stale slots
 * beyond the new window are never read (attention reads [s0,ctx) only). */
extern "C" int gemma4_kv_reset(sp_g4_kv *s) {
    if (!s) return -1;
    s->dpos_host = 0; s->commit_pos = 0; s->jcur = 0;
    int zero = 0;
    if (cudaMemcpy(s->dpos, &zero, sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
        sp_set_error("gemma4_kv_reset: dpos H2D"); return -1;
    }
    return 0;
}

/* KAI-2: arm a capture — the NEXT step D2H-copies its post-embed residual (E floats) into
 * emb_out. Used by the self-null gate to grab the model's own residual and feed it back. */
extern "C" int gemma4_kv_capture(sp_g4_kv *s, float *emb_out) {
    if (!s || !emb_out) return -1;
    s->cap_host = emb_out; s->cap_active = 1;
    return 0;
}

/* KAI-2: stage a latent event packet (one E-float residual) to OVERRIDE the next step's
 * post-embed residual (residual-entry injection). The forward then mints K/V natively at the
 * live dpos ⇒ RoPE phase correct by construction. One-shot (consumed by the next step). */
extern "C" int gemma4_kv_inject(sp_g4_kv *s, const float *emb) {
    if (!s || !emb) return -1;
    if (!s->dinj) { if (cudaMalloc(&s->dinj, (size_t)s->E*sizeof(float)) != cudaSuccess) { sp_set_error("gemma4_kv_inject: dinj OOM"); return -1; } }
    /* KAI-2 inject-scale control (G-KAIROS-2 diagnostic): the codec packet is the RAW residual the
     * teacher saw at the inputs_embeds seam. If gemma-4's text forward applies the sqrt(hidden)
     * embedding-normalizer to inputs_embeds during distillation but the engine seam injects the raw
     * value (post-embed, NO embscale — see the k_embed_scale_at at the decode head), the packet lands
     * ~sqrt(E)≈62x too weak ⇒ inert. SP_KAI2_INJSCALE lets us sweep {1, sqrt(E)} on metal without a
     * retrain to localize scale-mismatch vs structural. Default 1.0 = byte-identical to the null floor. */
    static float inj_scale = -1.0f;
    if (inj_scale < 0.0f) { const char *e = getenv("SP_KAI2_INJSCALE"); inj_scale = (e && *e) ? (float)atof(e) : 1.0f; }
    const float *src = emb; float *tmp = NULL;
    if (inj_scale != 1.0f) {
        tmp = (float*)malloc((size_t)s->E * sizeof(float));
        if (!tmp) { sp_set_error("gemma4_kv_inject: scale tmp OOM"); return -1; }
        for (int i = 0; i < s->E; i++) tmp[i] = emb[i] * inj_scale;
        src = tmp;
    }
    int rc = (cudaMemcpy(s->dinj, src, (size_t)s->E*sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess) ? 0 : -1;
    if (tmp) free(tmp);
    if (rc) { sp_set_error("gemma4_kv_inject: H2D"); return -1; }
    s->inj_active = 1;
    return 0;
}

/* KAI-3 (§7.1): inject a SEQUENCE of n_frames raw E-float residual vectors at n_frames consecutive
 * positions, each minted at a placeholder token (ph_token = audio_token_id 258881 for the audio port).
 * STRICT LOOP over the verified gemma4_kv_inject + gemma4_kv_prefill primitives — NO new tensor routing.
 * This is the exact per-position pattern the Phase-1 EMB control ran 2/2 on metal
 * (test_gemma4_cuda.c run_kai2_packet_gate, L981-986), moved into the engine. embs = row-major
 * [n_frames][E], raw (caller applies any scale). Advances dpos by n_frames. Null floor preserved:
 * dead code unless called. Returns 0 on success, -1 on any inject/prefill failure. */
extern "C" int gemma4_kv_inject_seq(sp_g4_kv *s, const float *embs, int n_frames, int ph_token) {
    if (!s || !embs || n_frames <= 0) { sp_set_error("gemma4_kv_inject_seq: bad args"); return -1; }
    int32_t ph = (int32_t)ph_token;
    for (int i = 0; i < n_frames; i++) {
        if (gemma4_kv_inject(s, embs + (size_t)i * s->E)) return -1;   /* sets error */
        if (gemma4_kv_prefill(s, &ph, 1)) return -1;                   /* consumes inj_active at live dpos */
    }
    return 0;
}

/* Free device VRAM in MiB (cudaMemGetInfo) — fragmentation-aware, unlike nvidia-smi's coarse
 * 'used'. Lets the soak tripwire watch the allocator's real headroom. */
extern "C" long gemma4_kv_devfree_mib(void) {
    size_t freeb = 0, totalb = 0;
    if (cudaMemGetInfo(&freeb, &totalb) != cudaSuccess) return -1;
    return (long)(freeb / (1024u * 1024u));
}

/* D2H the live K/V cache for all owner layers into host buffers (the gate's
 * byte-comparator). Caller passes pre-sized host arrays [NL][Pmax*kvd]; only
 * owner layers (L<kvfs) are filled. Returns total bytes copied. */
extern "C" long gemma4_kv_snapshot(const sp_g4_kv *s, float **hK, float **hV) {
    if (!s || !hK || !hV) return -1;
    cudaStreamSynchronize(g_w.stream);
    long bytes = 0;
    for (int L = 0; L < s->kvfs && L < s->NL; L++) {
        const int global = ((L % s->period) == s->period - 1);
        const int kvd = (global ? s->g_nkv : s->s_nkv) * (global ? s->g_hd : s->s_hd);
        const size_t slots = (s->ring_W > 0 && !global) ? (size_t)s->ring_W : (size_t)s->Pmax;  /* KAI-1c: ring buffers are Wring */
        size_t nb = slots * kvd * sizeof(float);
        if (hK[L] && cudaMemcpy(hK[L], s->dKc[L], nb, cudaMemcpyDeviceToHost) != cudaSuccess) return -1;
        if (hV[L] && cudaMemcpy(hV[L], s->dVc[L], nb, cudaMemcpyDeviceToHost) != cudaSuccess) return -1;
        bytes += 2 * (long)nb;
    }
    return bytes;
}

extern "C" void gemma4_kv_close(sp_g4_kv *s) {
    if (!s) return;
    if (s->dseq) cudaFree(s->dseq); if (s->dpos) cudaFree(s->dpos);
    float *ptrs[] = {s->dx,s->dnx,s->dq,s->dk,s->dv,s->dao,s->dap,s->dg,s->dup,s->ddn,s->dscr,
                     s->dlog,s->dipl,s->dple,s->dpg,s->dpp,s->dsx};
    for (size_t i = 0; i < sizeof(ptrs)/sizeof(ptrs[0]); i++) if (ptrs[i]) cudaFree(ptrs[i]);
    if (s->dqx) cudaFree(s->dqx);
    if (s->dKc) { for (int L=0;L<s->NL;L++) if (s->dKc[L]) cudaFree(s->dKc[L]); free(s->dKc); }
    if (s->dVc) { for (int L=0;L<s->NL;L++) if (s->dVc[L]) cudaFree(s->dVc[L]); free(s->dVc); }
    if (s->jK) { for (int L=0;L<s->NL;L++) if (s->jK[L]) cudaFree(s->jK[L]); free(s->jK); }   /* KAI-1c undo-journal */
    if (s->jV) { for (int L=0;L<s->NL;L++) if (s->jV[L]) cudaFree(s->jV[L]); free(s->jV); }
    if (s->dinj) cudaFree(s->dinj);   /* KAI-2 inject staging */
    free(s);
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
    if (W->bscale) {                          /* OK_Q4B: exact dequant, no post-lift */
        k_dequant_arena_q4b<<<grid, 256, 0, st>>>(W->codes, W->row_off, W->bscale,
                                                  W->bs_nblk, W->out, W->in, scratch);
        cudaError_t lq = cudaGetLastError();
        if (lq != cudaSuccess) return fail_cuda(lq, "k_dequant_arena_q4b launch");
        return gemm(h, scratch, dX, dY, n_tok, W->in, W->out);
    }
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
    if (W->bscale)                            /* Q4B dequant is already exact — same path */
        return gemm_w(h, st, W, dX, dY, n_tok, scratch);
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

/* BETA.3a/v4 + ETA.5b.4: single-token packed dp4a GEMV (decode). Quantize x->int8
 * PER-16-BLOCK (dqx codes + dsx block scales — the outlier-robust activation
 * quant; see k_quant_act_int8), dp4a against the packed Q8 (1 B/weight) or Q4
 * (0.5 B/weight, nibbles unpacked in-ALU) codes — no f32 scratch
 * materialization. Returns 1 if it TOOK the packed path, 0 if it declined
 * (caller falls back to the dequant/lift path). */
static int gemv_w_packed(cudaStream_t st, const DevTensor *W, const float *dX, float *dY,
                         signed char *dqx, float *dsx) {
    if (W->f32) return 0;                                 /* not packed */
    const int prec = W->prec;                             /* per-TENSOR precision (uniform rows) */
    unsigned blocks = ((unsigned)W->out + 7u) / 8u;       /* 8 warps (rows) per block */
    if (prec == 4) {                                      /* Q4: int4 load = 32 weights */
        if ((W->in & 31) != 0) return 0;
        int npad = (W->in + 31) & ~31;
        k_quant_act_int8<<<1, 256, 0, st>>>(dX, W->in, npad, dqx, dsx);
        if (W->bscale)                                    /* OK_Q4B (B1 recipe) */
            k_gemv_q4b_dp4a_v2<<<blocks, 256, 0, st>>>(
                W->codes, W->row_off, W->bscale, W->bs_nblk, W->in, dqx, dsx, dY, W->out);
        else
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

/* ETA.5b PPL gate: -log softmax(logits)[*dtarget] accumulated into *nll (f64,
 * single block — calls are stream-sequential, no atomics needed). Two-pass
 * numerically-stable log-softmax over the full vocab, f64 accumulation. */
__global__ void k_nll(const float *lg, int n, const int *dtarget, double *nll) {
    __shared__ float  smax[256];
    __shared__ double ssum[256];
    float m = -3.4e38f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) if (lg[i] > m) m = lg[i];
    smax[threadIdx.x] = m; __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (threadIdx.x < o && smax[threadIdx.x + o] > smax[threadIdx.x]) smax[threadIdx.x] = smax[threadIdx.x + o];
        __syncthreads();
    }
    const float mx = smax[0];
    double s = 0.0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += exp((double)lg[i] - (double)mx);
    ssum[threadIdx.x] = s; __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (threadIdx.x < o) ssum[threadIdx.x] += ssum[threadIdx.x + o];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        double logp = (double)lg[*dtarget] - (double)mx - log(ssum[0]);
        *nll += -logp;
    }
}

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
    if (use_int8) {            /* qx scratch sized to the widest matmul input (FF), padded to %32 (Q4 chunk);
                                * dsx = PER-16-BLOCK activation scales (npad/16 floats) */
        int maxin = E; if (QD>maxin) maxin=QD; if (FF>maxin) maxin=FF;
        const size_t npad = (size_t)((maxin+31)&~31);
        if (cudaMalloc(&dqx,npad)!=cudaSuccess){sp_set_error("dqx OOM");goto done;}
        DA(dsx, npad >> 4);
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
