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

/* ════════════════════════ device weight cache ════════════════════════ */

/* one weight matrix on device: either plain f32 (f32 != NULL) or packed arena. */
struct DevTensor {
    int in, out;
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
    /* post_attn/post_ffw are gemma3-only (sandwich norms); NULL arrays on qwen3. */
    float **attn_norm, **ffn_norm, **q_norm, **k_norm, **post_attn, **post_ffw;
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
    DevTensor **dts[] = { &w->Wq,&w->Wk,&w->Wv,&w->Wo,&w->Wgate,&w->Wup,&w->Wdown };
    for (size_t a = 0; a < sizeof(dts)/sizeof(dts[0]); a++) {
        DevTensor *arr = *dts[a];
        if (arr) { for (int L = 0; L < w->L; L++) free_devtensor(&arr[L]); free(arr); }
    }
    float ***ns[] = { &w->attn_norm,&w->ffn_norm,&w->q_norm,&w->k_norm,&w->post_attn,&w->post_ffw };
    for (size_t a = 0; a < sizeof(ns)/sizeof(ns[0]); a++) {
        float **arr = *ns[a];
        if (arr) { for (int L = 0; L < w->L; L++) if (arr[L]) cudaFree(arr[L]); free(arr); }
    }
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

    const int is_gemma = (c->arch == SP_ARCH_GEMMA3);
    const int tied = (m->output == m->token_embd);

    CudaWeights z = {}; *w = z;
    w->key = m; w->L = L;

    cudaError_t e = cudaStreamCreate(&w->stream);
    if (e != cudaSuccess) return fail_cuda(e, "cudaStreamCreate");
    CB(cublasCreate(&w->cublas), "cublasCreate");
    CB(cublasSetStream(w->cublas, w->stream), "cublasSetStream");
    CB(cublasSetMathMode(w->cublas, CUBLAS_DEFAULT_MATH), "cublasSetMathMode");

    /* token embedding (the embed gather always uses f32; gemma3's tied head reuses
     * it). From the arena if it was packed for release, else GGUF. */
    {
        const sp_arena_tensor *eat = m->arena ? sp_arena_find(m->arena, m->token_embd->name) : NULL;
        if (eat) {
            size_t n = (size_t)V * E;
            float *host = (float *)malloc(n * sizeof(float));
            if (!host) { sp_set_error("embd host OOM"); free_weights(w); return 1; }
            for (int r = 0; r < V; r++)
                if (sp_arena_dequant_row(eat, r, host + (size_t)r * E)) {
                    free(host); sp_set_error("embd arena dequant"); free_weights(w); return 1;
                }
            int rc = upload_arr<float>(host, n, &w->embd);
            free(host);
            if (rc) { free_weights(w); return 1; }
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

    for (int Li = 0; Li < L; Li++) {
        const qwen3_layer *ly = &m->layers[Li];
        BUILDW(Wq, attn_q, E, QD);   BUILDW(Wk, attn_k, E, KVD);  BUILDW(Wv, attn_v, E, KVD);
        BUILDW(Wo, attn_output, QD, E);
        BUILDW(Wgate, ffn_gate, E, FF); BUILDW(Wup, ffn_up, E, FF); BUILDW(Wdown, ffn_down, FF, E);
        UPV(attn_norm, attn_norm, E);   UPV(ffn_norm, ffn_norm, E);
        UPV(q_norm, attn_q_norm, HD);   UPV(k_norm, attn_k_norm, HD);
        if (is_gemma) { UPV(post_attn, post_attn_norm, E); UPV(post_ffw, post_ffw_norm, E); }
    }
    return 0;
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

/* BETA.3a: single-token INT8 GEMV (decode). Quantize x->int8 (dqx/dsx scratch),
 * dp4a against the packed Q8 codes. Returns 1 if it TOOK the int8 path, 0 if it
 * declined (caller falls back to gemm_w). Only for packed Q8 weights with in%4==0;
 * the caller must already have verified the arena precision is 8. */
static int gemv_w_int8(cudaStream_t st, const DevTensor *W, const float *dX, float *dY,
                       signed char *dqx, float *dsx) {
    if (W->f32) return 0;                                 /* not packed */
    if ((W->in & 15) == 0) {                              /* tuned: warp/row + int4 loads */
        int npad = (W->in + 15) & ~15;
        k_quant_act_int8<<<1, 256, 0, st>>>(dX, W->in, npad, dqx, dsx);
        unsigned blocks = ((unsigned)W->out + 7u) / 8u;   /* 8 warps (rows) per block */
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

    /* BETA.3a: fused INT8 dp4a GEMV for the decode matmuls, ONLY when the arena is
     * Q8 (sp_arena_precision==8) and the env is set. Q4 rows are nibble-packed and
     * would misread as int8, so the precision gate is load-bearing, not cosmetic. */
    const char *i8e = getenv("SP_CUDA_DECODE_INT8");
    const int use_int8 = (i8e && i8e[0]=='1') && m->arena && sp_arena_precision(m->arena) == 8;

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
    if (use_int8) {            /* qx scratch sized to the widest matmul input (FF), padded to %16 */
        int maxin = E; if (QD>maxin) maxin=QD; if (FF>maxin) maxin=FF;
        if (cudaMalloc(&dqx,(size_t)((maxin+15)&~15))!=cudaSuccess){sp_set_error("dqx OOM");goto done;}
        DA(dsx,1);
    }
    /* MM(W,X,Y): the decode matmul — int8 dp4a GEMV when enabled+packed-Q8, else
     * the cuBLAS dequant path. Both capture cleanly into the CUDA graph. */
    #define MM(W,X,Y) do{ if(!(use_int8 && gemv_w_int8(st,(W),(X),(Y),dqx,dsx))){ if(gemm_w(cb,st,(W),(X),(Y),1,dscr)) goto done; } }while(0)
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
