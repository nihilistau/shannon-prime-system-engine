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
#include <malloc.h>   /* N6 diag: _heapchk() host-heap-corruption checkpoint */

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

/* CONTRACT-CHAT-FULLSTACK B5 — gather ONE token's post-embed residual (embd[tok]*scale)
 * for the f32 embedding table into a target buffer `out[E]`. The single-position twin
 * of k_embed_scale_at, but with a HOST-supplied literal `tok` (not dseq[*dpos]) and a
 * caller buffer — so gemma4_kv_inject_tokens can stage embd[id]*sqrt(E) into s->dinj
 * (the inject seam) with arithmetic BIT-IDENTICAL to the embed-at kernel the stock
 * prefill step runs. grid = ceil(E/256). */
__global__ void k_embed_scale_one(const float *embd, int tok, int E, float scale, float *out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < E) out[i] = embd[(size_t)tok * E + i] * scale;
}

/* CONTRACT-CHAT-FULLSTACK B5 — packed (OK_Q4B/Q8) twin of k_embed_scale_one. Mirrors
 * k_embed_packed_at exactly (same dequant: inv = row_scale/qmax, true division) but with
 * a HOST-supplied literal `tok` and a caller buffer. Bit-identical to the packed
 * embed-at the stock prefill step runs for token `tok`. grid = ceil(E/256). */
__global__ void k_embed_packed_one(const unsigned char *codes, const unsigned long long *row_off,
                                   const float *row_scale, const unsigned char *row_prec,
                                   int tok, int E, float scale, float *out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= E) return;
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
    out[i] = (float)code * inv * scale;
}

/* ===================== BYTE-EXACT substrate (SP_BYTEEXACT) =====================
 * Dual-prime modular constants + the exact-integer device helpers shared by the
 * byte-exact attention (k_attn_decode_win_bx, below) AND the nonlinear islands
 * (RMSNorm/GELU/RoPE/softcap branches in the k_rmsnorm, k_gelu_mul, k_rope, k_softcap
 * kernels). Hoisted above the first island kernel so every consumer sees the defs.
 * Frozen Proth primes q1/q2 + Garner CRT (no __int128; products < q^2 ~ 2^60 fit
 * int64); FB30 fixed-point for exp/tanh; CORDIC for cos/sin; integer isqrt for 1/sqrt.
 * Gated by d_bx_flag (set once at gemma4 decode entry from SP_BYTEEXACT, BEFORE graph
 * capture); d_bx_flag==0 => the float path runs unchanged (byte-identical null floor).
 * Validated offline on REAL 12B activations: RMS 3.84e-5 / GELU 8.18e-7 / RoPE 9.62e-6
 * (G-BYTEEXACT-ISLANDS-CUDA) + attention G-BYTEEXACT-ATTN-NTT/-FULL. */
#define BX_Q1     1073738753LL
#define BX_Q2     1073732609LL
#define BX_INVQ1  894602413LL          /* Q1^{-1} mod Q2 */
#define BX_FB     30
#define BX_ONE    (1LL << BX_FB)
#define BX_LOG2E  1549082005LL         /* round(log2(e) * 2^30) */
#define BX_DSH    16                   /* Delta = 2^16 */
#define BX_ZB     14                   /* softmax logit grid Z = 2^14 */
__device__ __constant__ long long BX_EXPC[7] =
    {1073741824LL,744261118LL,257941248LL,59597083LL,10327387LL,1431680LL,165394LL};

__device__ __forceinline__ long long bx_garner(long long r1, long long r2) {
    const long long M = BX_Q1 * BX_Q2;                 /* ~2^60, < 2^63 */
    long long t = ((r2 - r1) % BX_Q2 + BX_Q2) % BX_Q2;
    t = (t * BX_INVQ1) % BX_Q2;                        /* t<Q2; product<Q2^2~2^60 */
    long long x = r1 + BX_Q1 * t;                      /* < M */
    return (x > M / 2) ? x - M : x;                    /* centered exact integer */
}
__device__ __forceinline__ long long bx_exp2_frac(long long r) {
    long long acc = BX_EXPC[6];
    for (int k = 5; k >= 0; k--) acc = ((acc * r) >> BX_FB) + BX_EXPC[k];  /* <2^61 */
    return acc;
}
/* e^d, d<=0 (FB30 fixed) -> FB30 fixed. The d*LOG2E product can exceed int64 for
 * far-from-max keys, so use __umul64hi for the 128-bit product (no __int128). */
__device__ __forceinline__ long long bx_exp_fixed(long long d) {
    if (d >= 0) return BX_ONE;
    unsigned long long ad = (unsigned long long)(-d);
    unsigned long long lo = ad * (unsigned long long)BX_LOG2E;
    unsigned long long hi = __umul64hi(ad, (unsigned long long)BX_LOG2E);
    long long g = (long long)((hi << (64 - BX_FB)) | (lo >> BX_FB));   /* g = -d*log2e >= 0 */
    long long n = g >> BX_FB;
    if (n >= 32) return 0;                              /* underflow -> 0 */
    long long r = g - (n << BX_FB);
    return r ? (bx_exp2_frac(BX_ONE - r) >> (n + 1)) : (BX_ONE >> n);
}

/* ===================== BYTE-EXACT nonlinear islands (SP_BYTEEXACT) =====================
 * Exact-integer device ports of the float forward's RMSNorm / GELU-tanh / RoPE / softcap,
 * matching the crate scalar references (tools/sp_dsp_smoke/src/sp_islands_q_ref.rs) bit-for-bit
 * in their integer arithmetic. Every reduction is exact integer (long long) => reduction-order-
 * immune (cross-machine reproducible). Every transcendental is a deterministic integer function:
 *   1/sqrt via integer isqrt; cos/sin via rotation-mode CORDIC; exp/tanh via the FB30 2^x poly.
 * NVCC has NO __int128: wide products use __umul64hi; the RMS isqrt numerator n<<72 uses the
 * validated 64-bit split (SH=50: num=(n<<50)/sumsq; val=num<<22; inv=isqrt_u64(val); ~1e-5).
 * Gated by d_bx_flag (set once at gemma4 decode entry from SP_BYTEEXACT, BEFORE graph capture);
 * d_bx_flag==0 => the float path runs unchanged (byte-identical null floor). Validated offline on
 * REAL 12B activations: RMS 3.84e-5 / GELU 8.18e-7 / RoPE 9.62e-6 (G-BYTEEXACT-ISLANDS-CUDA). */
__device__ __constant__ int d_bx_flag = 0;
/* CONTRACT-CUDA-KV-FOUNDATION: per-session KV codec flags (bit0 = SP_KV_SPINOR).
 * 0 = float null floor; consumed by the KV alloc/read once the O_K Spinor carrier lands. */
__device__ __constant__ unsigned int d_kv_flags = 0u;

/* RMSNorm fixed-point layout (frozen, == the ref): Q=16 / IB=20 / Qw=16. */
#define BX_RMS_Q   16
#define BX_RMS_IB  20
#define BX_RMS_QW  16
#define BX_RMS_SH  50          /* the 64-bit-split shift (num=(n<<SH)/sumsq); SH=50 -> val=num<<22 ~ n<<72 */

/* floor(sqrt(v)) over u64 — exact integer isqrt (the 64-bit twin of the ref isqrt_u128). */
__device__ __forceinline__ unsigned long long bx_isqrt_u64(unsigned long long v) {
    if (v == 0ULL) return 0ULL;
    unsigned long long x = 0ULL;
    unsigned long long b = 1ULL << 62;
    while (b > v) b >>= 2;
    while (b != 0ULL) {
        if (v >= x + b) { v -= x + b; x = (x >> 1) + b; }
        else            { x >>= 1; }
        b >>= 2;
    }
    return x;
}

/* Island 1 — exact-integer RMSNorm scale. Given the integer sum-of-squares `sumsq`
 * (= Sum (round(x_i*2^Q))^2, accumulated exactly => order-immune) and the row length n,
 * returns inv = isqrt((n<<72)/sumsq) (the FB-shifted reciprocal sqrt) via the 64-bit split.
 * out[i] = (round(x_i*2^Q) * inv * w_q) / 2^52, where w_q = round(w_i*2^Qw) or 2^Qw. */
__device__ __forceinline__ long long bx_rms_inv(long long sumsq, int n) {
    if (sumsq <= 0) return 0;
    unsigned long long num = ((unsigned long long)n << BX_RMS_SH) / (unsigned long long)sumsq;
    unsigned long long val = num << 22;                       /* ~ (n<<72)/sumsq */
    return (long long)bx_isqrt_u64(val);
}
/* encode a float as round(v * 2^shift) (the ref `enc`). */
__device__ __forceinline__ long long bx_enc(float v, int shift) {
    return (long long)llrintf(v * (float)(1ULL << shift));
}

/* tanh(t) FB30-fixed via the shared exp primitive (the ref tanh_fixed). */
__device__ __forceinline__ long long bx_tanh_fixed(long long t) {
    long long s = (t >= 0) ? 1 : -1;
    long long a = (t >= 0) ? t : -t;
    long long e2 = bx_exp_fixed(-(2 * a));
    long long num = (2 * e2) << BX_FB;                        /* 2*e2 <= 2^31, <<30 -> <2^62 */
    return s * (BX_ONE - num / (BX_ONE + e2));
}

/* Island 2 — exact-integer GELU-tanh: 0.5 x (1 + tanh(sqrt(2/pi)(x + 0.044715 x^3))).
 * FB30 cubic+tanh; X*X exceeds int64 so use __umul64hi for the 128-bit product (ref uses i128). */
#define BX_GK  856722024LL        /* round(sqrt(2/pi) * 2^30) */
#define BX_GA  48012366LL         /* round(0.044715  * 2^30) */
#define BX_GELU_Z 16              /* ref ZB */
/* signed (a*b)>>FB for |a|,|b| up to ~2^34 (the GELU cubic intermediates) using __umul64hi. */
__device__ __forceinline__ long long bx_mulshift_fb(long long a, long long b) {
    long long sa = (a < 0) ? -1 : 1, sb = (b < 0) ? -1 : 1;
    unsigned long long ua = (unsigned long long)(a < 0 ? -a : a);
    unsigned long long ub = (unsigned long long)(b < 0 ? -b : b);
    unsigned long long lo = ua * ub;
    unsigned long long hi = __umul64hi(ua, ub);
    unsigned long long r  = (hi << (64 - BX_FB)) | (lo >> BX_FB);
    return (sa * sb) * (long long)r;
}
__device__ __forceinline__ float bx_gelu(float xv) {
    long long xq    = bx_enc(xv, BX_GELU_Z);
    long long big_x = (xq << BX_FB) >> BX_GELU_Z;             /* i64; ~2^34 */
    long long x2 = bx_mulshift_fb(big_x, big_x);
    long long x3 = bx_mulshift_fb(x2, big_x);
    long long inner = bx_mulshift_fb(BX_GK, big_x + bx_mulshift_fb(BX_GA, x3));
    long long t = bx_tanh_fixed(inner);
    long long g = bx_mulshift_fb(big_x >> 1, BX_ONE + t);
    return (float)((double)g / (double)BX_ONE);
}

/* Island 4 — RoPE via deterministic rotation-mode CORDIC (the ref cordic_cossin). */
#define BX_CORDIC_N 30
__device__ __constant__ long long BX_ATAN_FB30[BX_CORDIC_N] = {
    843314857LL,497837829LL,263043837LL,133525159LL,67021687LL,33543516LL,16775851LL,8388437LL,
    4194283LL,2097149LL,1048576LL,524288LL,262144LL,131072LL,65536LL,32768LL,16384LL,8192LL,4096LL,
    2048LL,1024LL,512LL,256LL,128LL,64LL,32LL,16LL,8LL,4LL,2LL};
#define BX_CORDIC_K 652032874LL   /* round( prod 1/sqrt(1+2^-2k) * 2^30 ) */
#define BX_PI_FB    3373259426LL
#define BX_HALFPI_FB 1686629713LL
#define BX_TWOPI_FB 6746518852LL
/* (cos,sin) in FB30-fixed for FB30-fixed radians theta (reduce to [-pi/2,pi/2] then CORDIC). */
__device__ __forceinline__ void bx_cordic_cossin(long long theta, long long *co, long long *si) {
    long long z = theta % BX_TWOPI_FB;
    if (z > BX_PI_FB) z -= BX_TWOPI_FB; else if (z < -BX_PI_FB) z += BX_TWOPI_FB;
    int neg = 0;
    if (z > BX_HALFPI_FB) { z -= BX_PI_FB; neg = 1; }
    else if (z < -BX_HALFPI_FB) { z += BX_PI_FB; neg = 1; }
    long long x = BX_CORDIC_K, y = 0;
    for (int k = 0; k < BX_CORDIC_N; k++) {
        long long xs = x >> k, ys = y >> k;
        if (z >= 0) { x -= ys; y += xs; z -= BX_ATAN_FB30[k]; }
        else        { x += ys; y -= xs; z += BX_ATAN_FB30[k]; }
    }
    *co = neg ? -x : x;
    *si = neg ? -y : y;
}
/* one RoPE pair rotated integer-exact (ref rope_q_ref body, Q=16). `theta` is the FB30-fixed
 * angle pos*freq already reduced mod 2pi; a,b the pair (v[i], v[i+half]). */
#define BX_ROPE_Q 16
__device__ __forceinline__ void bx_rope_pair(float *va, float *vb, long long theta) {
    long long c, s; bx_cordic_cossin(theta, &c, &s);
    long long a = (long long)llrint((double)(*va) * (double)(1u << BX_ROPE_Q));
    long long b = (long long)llrint((double)(*vb) * (double)(1u << BX_ROPE_Q));
    long long oa = (a * c - b * s) >> BX_FB;
    long long ob = (a * s + b * c) >> BX_FB;
    *va = (float)((double)oa / (double)(1u << BX_ROPE_Q));
    *vb = (float)((double)ob / (double)(1u << BX_ROPE_Q));
}
/* the FB30-fixed angle pos*freq reduced into [0,2pi): freq baked from the float freq the float
 * kernel would use (powf(rbase,-2i/d)[/ff]); pos*freq computed in fixed-point then reduced. The
 * ref takes a baked freq_fix table; here freq is recomputed from the same float inputs the float
 * kernel uses (deterministic: the encode round() is the only nonexact step, ~1e-5 fidelity). */
__device__ __forceinline__ long long bx_rope_theta(float freq, long long pos) {
    long long freq_fix = (long long)llrint((double)freq * (double)BX_ONE);  /* FB30, <= 2^30 */
    /* pos < 2^20 (context) and freq_fix <= 2^30 => product < 2^50, fits int64 (no __int128). */
    long long th = (pos * freq_fix) % BX_TWOPI_FB;
    return th;
}

/* out[row] = rmsnorm(x[row]) * w over n elems. One block/row; sum_sq in f64
 * (matches CPU). scale = 1/sqrtf((float)(sum_sq/n) + eps). */
/* device byte-exact RMSNorm core (shared by all three k_rmsnorm* kernels). Integer
 * sum-of-squares (order-immune) + bx_rms_inv; out[i] = (enc(x_i,Q)*inv*w_q)/2^52. */
__device__ __forceinline__ void bx_rmsnorm_core(const float *xr, float *outr,
                                                const float *w_or_null, int n) {
    __shared__ long long shi[256];
    long long s = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        long long xi = bx_enc(xr[i], BX_RMS_Q); s += xi * xi;
    }
    shi[threadIdx.x] = s;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (threadIdx.x < o) shi[threadIdx.x] += shi[threadIdx.x + o];
        __syncthreads();
    }
    long long inv = bx_rms_inv(shi[0], n);
    const double denom = (double)(1ULL << (BX_RMS_Q + BX_RMS_IB + BX_RMS_QW));  /* 2^52 */
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        long long xi = bx_enc(xr[i], BX_RMS_Q);
        long long wq = w_or_null ? bx_enc(w_or_null[i], BX_RMS_QW) : (1LL << BX_RMS_QW);
        /* xi*inv*wq is the ref's i128 product / 2^52 -> f32; the dropped low bits are far
         * below f32 precision, so the double accumulation rounds to the identical f32. */
        double yv = (double)xi * (double)inv * (double)wq;
        outr[i] = (float)(yv / denom);
    }
}

__global__ void k_rmsnorm(const float *x, const float *w, int n, float eps, float *out) {
    int row = blockIdx.x;
    const float *xr = x + (size_t)row * n;
    float *outr = out + (size_t)row * n;
    if (d_bx_flag) { bx_rmsnorm_core(xr, outr, w, n); return; }
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
    if (d_bx_flag) { bx_rmsnorm_core(v, v, w, d); return; }
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
    if (d_bx_flag) { bx_rmsnorm_core(v, v, NULL, d); return; }
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
        if (d_bx_flag) { bx_rope_pair(&v[i], &v[i + half], bx_rope_theta(freq, (long long)t)); return; }
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
    if (d_bx_flag) { pg[idx] = bx_gelu(v) * ipl[((size_t)t * NL + L) * PL + i]; return; }
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

/* B3-JUDGE/G-INT-2-FIX: scale a K row by a HOST scalar alpha in-place (mirrors the
 * replay path's kf[i]*=replay_alpha, but on-device for the natively-minted inject K). */
__global__ void k_scale_by_const(float *x, size_t n, float a) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= a;
}

/* ETA.5 (Gemma4 decode): NEOX RoPE WITH freq factors at ABSOLUTE position p0
 * (single token, grid = n_heads). The decode twin of k_rope_freqs. */
__global__ void k_rope_freqs_at(float *base, int n_heads, int d, float rbase,
                                const float *ff, int p0) {
    int h = blockIdx.x, i = threadIdx.x, half = d / 2;
    if (i < half) {
        float *v = base + (size_t)h * d;
        float freq = powf(rbase, -2.0f * (float)i / (float)d) / ff[i];
        if (d_bx_flag) { bx_rope_pair(&v[i], &v[i + half], bx_rope_theta(freq, (long long)p0)); return; }
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

/* ===================== BYTE-EXACT attention (SP_BYTEEXACT) =====================
 * Exact-integer decode attention via dual-prime modular arithmetic -- the PPT negacyclic
 * substitution: <q,k> = the plain dot Sum q_i k_i computed mod the frozen Proth primes +
 * Garner CRT (the BX_* constants / bx_garner / bx_exp_fixed helpers are hoisted above, in
 * the BYTE-EXACT substrate block). Q.K and p.V are exact integer dots; softmax is the FB30
 * 2^x integer island. Cross-machine reproducible by construction (reduction-order-immune).
 * Validated offline: G-BYTEEXACT-ATTN-NTT + G-BYTEEXACT-ATTN-FULL. Delta = 2^16 CKKS scale
 * (relerr ~2e-5; the p.V accumulator stays ~2^46 << 2^60 at every context). */

/* Drop-in for k_attn_decode_win (ascale folded; gemma4 scaling=1.0). Shared = ctx*int64
 * (scores, reused as softmax weights). One block per query head. */
__global__ void k_attn_decode_win_bx(const float *q, const float *Kc, const float *Vc,
                                     int ctx, int KVD, int HD, int group, float ascale,
                                     int win, float *ao) {
    extern __shared__ long long shl[];                 /* [ctx]: D_s then e_s */
    int h = blockIdx.x, kvh = h / group;
    int pos = ctx - 1;
    int s0 = (win >= 0 && pos - win + 1 > 0) ? pos - win + 1 : 0;
    const float *qh = q + (size_t)h * HD;
    const float DSC = (float)(1 << BX_DSH);
    for (int s = s0 + threadIdx.x; s < ctx; s += blockDim.x) {
        const float *kh = Kc + (size_t)s * KVD + (size_t)kvh * HD;
        long long a1 = 0, a2 = 0;
        for (int i = 0; i < HD; i++) {
            long long ea = (long long)llrintf(qh[i] * DSC);
            long long eb = (long long)llrintf(kh[i] * DSC);
            long long p = ea * eb;                     /* ~2^36 < 2^63 */
            a1 = ((a1 + p) % BX_Q1 + BX_Q1) % BX_Q1;
            a2 = ((a2 + p) % BX_Q2 + BX_Q2) % BX_Q2;
        }
        shl[s] = bx_garner(a1, a2);                    /* exact Delta^2 * <q,k_s> */
    }
    __syncthreads();
    __shared__ long long g_S;
    if (threadIdx.x == 0) {
        long long mxz = ((shl[s0] << BX_ZB) >> (2 * BX_DSH));   /* score on Z grid */
        for (int s = s0 + 1; s < ctx; s++) {
            long long z = (shl[s] << BX_ZB) >> (2 * BX_DSH);
            if (z > mxz) mxz = z;
        }
        long long S = 0;
        for (int s = s0; s < ctx; s++) {
            long long z = (shl[s] << BX_ZB) >> (2 * BX_DSH);
            long long e = bx_exp_fixed((z - mxz) << (BX_FB - BX_ZB));  /* exp((z-mx)/Z) */
            shl[s] = e; S += e;
        }
        g_S = S;
    }
    __syncthreads();
    long long S = g_S;
    for (int i = threadIdx.x; i < HD; i += blockDim.x) {
        long long a1 = 0, a2 = 0;
        for (int s = s0; s < ctx; s++) {
            long long ev = (long long)llrintf(Vc[(size_t)s * KVD + (size_t)kvh * HD + i] * DSC);
            long long p = shl[s] * ev;                 /* ~2^48 < 2^63 */
            a1 = ((a1 + p) % BX_Q1 + BX_Q1) % BX_Q1;
            a2 = ((a2 + p) % BX_Q2 + BX_Q2) % BX_Q2;
        }
        long long num = bx_garner(a1, a2);             /* exact Sum e_s * enc(v_s,i) */
        ao[(size_t)h * HD + i] = (float)((double)num / ((double)S * (double)(1 << BX_DSH)));
    }
}
/* SP_BYTEEXACT: route decode attention through the exact-integer kernel (default off =
 * byte-identical null floor). Cached once. */
static int sp_byteexact_attn(void) {
    static int v = -1;
    if (v < 0) v = (getenv("SP_BYTEEXACT") != NULL) ? 1 : 0;
    return v;
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

/* CONTRACT-CHAT-FULLSTACK B2 RING-FIX: BYTE-EXACT windowed decode attention over a
 * RING cache of `Wring` slots. This is k_attn_decode_win_bx (exact-integer dual-prime
 * dot + FB30 softmax — build-independent, reduction-order-immune) with the ring slot
 * remap (s0+j)%Wring from k_attn_decode_ring. It lets the 40 SWA layers stay on the
 * byte-exact substrate when the ring is armed AND byte-exact ("auditable") is on — so
 * the served chat keeps S1's build-determinism (the FP-reorder fragility that flips
 * coherent<->garbage across rebuilds is what byte-exact removes; the float ring kernel
 * would re-introduce it on 40/48 layers and break coherence). Shared = wl*int64
 * (scores -> softmax weights), wl = in-window length. One block per query head. */
__global__ void k_attn_decode_ring_bx(const float *q, const float *Kc, const float *Vc,
                                      int ctx, int KVD, int HD, int group, float ascale,
                                      int win, int Wring, float *ao) {
    (void)ascale;                                          /* gemma4 scaling=1.0 (folded) */
    extern __shared__ long long shlr[];                    /* [wl]: D_j then e_j */
    int h = blockIdx.x, kvh = h / group;
    int pos = ctx - 1;
    int s0 = (win >= 0 && pos - win + 1 > 0) ? pos - win + 1 : 0;
    int wl = ctx - s0;                                     /* in-window length, <= Wring */
    const float *qh = q + (size_t)h * HD;
    const float DSC = (float)(1 << BX_DSH);
    for (int j = threadIdx.x; j < wl; j += blockDim.x) {
        int slot = (s0 + j) % Wring;                       /* position s0+j -> ring slot */
        const float *kh = Kc + (size_t)slot * KVD + (size_t)kvh * HD;
        long long a1 = 0, a2 = 0;
        for (int i = 0; i < HD; i++) {
            long long ea = (long long)llrintf(qh[i] * DSC);
            long long eb = (long long)llrintf(kh[i] * DSC);
            long long p = ea * eb;
            a1 = ((a1 + p) % BX_Q1 + BX_Q1) % BX_Q1;
            a2 = ((a2 + p) % BX_Q2 + BX_Q2) % BX_Q2;
        }
        shlr[j] = bx_garner(a1, a2);                       /* exact Delta^2 * <q,k_{s0+j}> */
    }
    __syncthreads();
    __shared__ long long g_Sr;
    if (threadIdx.x == 0) {
        long long mxz = ((shlr[0] << BX_ZB) >> (2 * BX_DSH));
        for (int j = 1; j < wl; j++) {
            long long z = (shlr[j] << BX_ZB) >> (2 * BX_DSH);
            if (z > mxz) mxz = z;
        }
        long long S = 0;
        for (int j = 0; j < wl; j++) {
            long long z = (shlr[j] << BX_ZB) >> (2 * BX_DSH);
            long long e = bx_exp_fixed((z - mxz) << (BX_FB - BX_ZB));
            shlr[j] = e; S += e;
        }
        g_Sr = S;
    }
    __syncthreads();
    long long S = g_Sr;
    for (int i = threadIdx.x; i < HD; i += blockDim.x) {
        long long a1 = 0, a2 = 0;
        for (int j = 0; j < wl; j++) {
            int slot = (s0 + j) % Wring;
            long long ev = (long long)llrintf(Vc[(size_t)slot * KVD + (size_t)kvh * HD + i] * DSC);
            long long p = shlr[j] * ev;
            a1 = ((a1 + p) % BX_Q1 + BX_Q1) % BX_Q1;
            a2 = ((a2 + p) % BX_Q2 + BX_Q2) % BX_Q2;
        }
        long long num = bx_garner(a1, a2);
        ao[(size_t)h * HD + i] = (float)((double)num / ((double)S * (double)(1 << BX_DSH)));
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
    if (i >= n) return;
    if (d_bx_flag) {
        /* exact-integer tanh via the FB30 shared primitive (the softcap of the LM-head logits).
         * encode z/cap to FB30, bx_tanh_fixed, decode * cap. */
        long long t = (long long)llrint((double)(z[i] / cap) * (double)BX_ONE);
        long long th = bx_tanh_fixed(t);
        z[i] = (float)((double)th / (double)BX_ONE) * cap;
        return;
    }
    z[i] = tanhf(z[i] / cap) * cap;
}

/* NEOX RoPE on each (token,head) vector at position p=t; blockDim=d/2. */
__global__ void k_rope(float *base, int n_heads, int d, int rowstride, float rbase) {
    int b = blockIdx.x, t = b / n_heads, h = b % n_heads, i = threadIdx.x, half = d / 2;
    if (i < half) {
        float *v = base + (size_t)t * rowstride + (size_t)h * d;
        float freq = powf(rbase, -2.0f * (float)i / (float)d);
        if (d_bx_flag) { bx_rope_pair(&v[i], &v[i + half], bx_rope_theta(freq, (long long)t)); return; }
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
        if (d_bx_flag) { bx_rope_pair(&v[i], &v[i + half], bx_rope_theta(freq, (long long)(t + p0))); return; }
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
        if (d_bx_flag) { g[idx] = bx_gelu(x) * up[idx]; return; }
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
static int gemv_t(cublasHandle_t h, const float *dW, const float *dX, float *dY,
                  int in, int out);                                            /* defined below (n_tok=1 fast path) */
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
    /* G-BYTEEXACT-ISLANDS-CUDA dump seam (additive, default-off = byte-inert):
     * SP_BYTEEXACT_DUMP=path writes the INPUT+OUTPUT of the float islands
     * (RMSNorm / GELU-tanh / RoPE) for ONE chosen layer (SP_BYTEEXACT_LAYER,
     * default = last layer) on the prefill forward, so the host comparator can
     * re-run the SAME inputs through the crate's exact-integer *_q_ref and gate
     * the per-island fidelity (contract CONTRACT-BYTEEXACT-forward §5.1). The
     * file is self-describing: a magic header then per-island {tag,len} records.
     * Pure observation — no kernel behaviour changes; default-off path untouched. */
    const char *bx_dump_path = getenv("SP_BYTEEXACT_DUMP");
    FILE *bx_dump_f = NULL;
    int   bx_layer = -1; { const char *e = getenv("SP_BYTEEXACT_LAYER"); if (e) bx_layer = atoi(e); }
    float *bx_h = NULL;               /* host staging for D2H of the island buffers */
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

    /* G-BYTEEXACT-ISLANDS-CUDA: open the dump file + host staging once n_layers is
     * finalized (so the default layer = last layer is valid). bx_layer<0 => last. */
    if (bx_dump_path) {
        /* default to a MID layer (not the last): under attn_only=1 the loop breaks at
         * L==n_layers-1 right after the attention residual, BEFORE the FFN RMSNorm/GELU
         * captures — so the chosen layer must be < n_layers-1 to dump all three islands. */
        if (bx_layer < 0 || bx_layer >= n_layers - 1) bx_layer = (n_layers > 1) ? n_layers / 2 : 0;
        bx_dump_f = fopen(bx_dump_path, "wb");
        if (!bx_dump_f) { sp_set_error("bx_dump: open"); goto done; }
        size_t bx_hmax = (size_t)n_tok * (QDmax > E ? QDmax : E);
        if ((size_t)n_tok * FFmax > bx_hmax) bx_hmax = (size_t)n_tok * FFmax;
        bx_h = (float *)malloc(bx_hmax * sizeof(float));
        if (!bx_h) { sp_set_error("bx_dump: host OOM"); goto done; }
        /* file header: magic 'BXI1', version, n_tok, E, chosen layer, period */
        int32_t fh[8] = { 0x31495842 /*'BXI1'*/, 1, n_tok, E, bx_layer, period,
                          (int)c->g4_swa_period, 0 };
        fwrite(fh, sizeof(int32_t), 8, bx_dump_f);
        fprintf(stderr, "    [bx-dump] island dump -> %s (layer %d, n_tok %d)\n",
                bx_dump_path, bx_layer, n_tok);
    }

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

    /* G-BYTEEXACT-ISLANDS-CUDA: dump one device buffer as a self-describing record.
     * tag: 4-byte ASCII island/role ('RMSi'/'RMSw'/'RMSo'/'GELi'/'GELu'/'GELo'/
     * 'ROPi'/'ROPo'); rows = n_tok (or n_head_rows), width = per-row floats. Syncs
     * the stream first (capture is post-kernel). Only invoked under bx_dump_f. */
    #define BX_REC(tagstr, dptr, rows, width) do { \
        cudaError_t _e = cudaStreamSynchronize(st); \
        if (_e != cudaSuccess) { fail_cuda(_e, "bx_dump sync"); goto done; } \
        size_t _n = (size_t)(rows) * (size_t)(width); \
        cudaMemcpy(bx_h, (dptr), _n * sizeof(float), cudaMemcpyDeviceToHost); \
        int32_t _rh[4]; memcpy(_rh, (tagstr), 4); _rh[1] = (int)(rows); _rh[2] = (int)(width); _rh[3] = 0; \
        fwrite(_rh, sizeof(int32_t), 4, bx_dump_f); \
        fwrite(bx_h, sizeof(float), _n, bx_dump_f); \
    } while (0)

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
        /* G-BYTEEXACT-ISLANDS-CUDA: RoPE input = post-q-norm q [n_tok*nh rows x hd];
         * width carries hd, rope base + freq-factor mode recorded in the rope-out tag. */
        if (bx_dump_f && L == bx_layer) {
            BX_REC("ROPi", dq, n_tok*nh, hd);
            /* RoPE metadata: rbase as a float record [1x1], then the freq-factor table
             * (ffac, length hd/2) if proportional, else a single -1.0 sentinel. NEOX
             * layout; the comparator builds freq_fix[i] = base^(-2i/d)/ff[i]. */
            { float rb = rbase;
              int32_t _rh[4]; memcpy(_rh, "ROPb", 4); _rh[1]=1; _rh[2]=1; _rh[3]=0;
              fwrite(_rh, sizeof(int32_t), 4, bx_dump_f); fwrite(&rb, sizeof(float), 1, bx_dump_f); }
            if (ffac) { BX_REC("ROPf", ffac, 1, hd/2); }
            else { int32_t _rh[4]; memcpy(_rh, "ROPf", 4); _rh[1]=1; _rh[2]=1; _rh[3]=0;
                   float none = -1.0f; fwrite(_rh, sizeof(int32_t), 4, bx_dump_f);
                   fwrite(&none, sizeof(float), 1, bx_dump_f); }
        }
        if (ffac) k_rope_freqs<<<n_tok*nh, hd/2, 0, st>>>(dq, nh, hd, qd, rbase, ffac);
        else      k_rope<<<n_tok*nh, hd/2, 0, st>>>(dq, nh, hd, qd, rbase);
        if (bx_dump_f && L == bx_layer) BX_REC("ROPo", dq, n_tok*nh, hd);
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
        /* G-BYTEEXACT-ISLANDS-CUDA: RMSNorm island = ffn_norm (clean E-wide weighted).
         * Capture input x, the weight vector, and the output nx (post k_rmsnorm). */
        if (bx_dump_f && L == bx_layer) BX_REC("RMSi", dx, n_tok, E);
        if (bx_dump_f && L == bx_layer) BX_REC("RMSw", g_w.ffn_norm[L], 1, E);
        k_rmsnorm<<<n_tok, 256, 0, st>>>(dx, g_w.ffn_norm[L], E, eps, dnx);
        if (bx_dump_f && L == bx_layer) BX_REC("RMSo", dnx, n_tok, E);
        if (gemm_w_lift(cb, st, &g_w.Wgate[L], dnx, dg, n_tok, dscr)) goto done;
        if (gemm_w_lift(cb, st, &g_w.Wup[L], dnx, dup, n_tok, dscr)) goto done;
        /* GELU-tanh island: capture pre-gelu gate g, the up tensor, and the post-gelu
         * product (g <- 0.5 g (1+tanh(k(g+0.044715 g^3))) * up). */
        if (bx_dump_f && L == bx_layer) BX_REC("GELi", dg, n_tok, ffL);
        if (bx_dump_f && L == bx_layer) BX_REC("GELu", dup, n_tok, ffL);
        {   size_t nFF = (size_t)n_tok * ffL;
            k_gelu_mul<<<(unsigned)((nFF+255)/256), 256, 0, st>>>(dg, dup, nFF); }
        if (bx_dump_f && L == bx_layer) BX_REC("GELo", dg, n_tok, ffL);
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
    if (bx_dump_f) fclose(bx_dump_f);   /* G-BYTEEXACT-ISLANDS-CUDA dump */
    if (bx_h) free(bx_h);
    #undef BX_REC
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
        if (d_bx_flag) { bx_rope_pair(&v[i], &v[i + half], bx_rope_theta(freq, (long long)(*dpos))); (void)n_heads; return; }
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
    /* SP_BYTEEXACT: route the nonlinear islands (RMSNorm/GELU/RoPE/softcap) through the
     * exact-integer device branches (d_bx_flag). Set ONCE here, BEFORE any kernel launch
     * or graph capture, so the captured graph bakes the chosen path. Default off = the
     * byte-identical null floor. Cached to dodge the getenv per call. */
    { static int bx = -1; if (bx < 0) bx = (getenv("SP_BYTEEXACT") != NULL) ? 1 : 0;
      cudaMemcpyToSymbolAsync(d_bx_flag, &bx, sizeof(int), 0, cudaMemcpyHostToDevice, st);
      cudaStreamSynchronize(st); }
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
    /* XBAR P3.3 (SP_REPLAY): episode injected over the freshly-minted prefill owner rows [0,NPOS). */
    char *d_replay_K=NULL, *d_replay_V=NULL; size_t *replay_off=NULL; int replay_on=0, replay_npos=0, replay_zero=0;
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
    /* ── XBAR P3.3: SP_REPLAY — inject a stored episode's owner K/V over the freshly-minted
     *    prefill rows for pos<NPOS, BEFORE the cache store lands (the decode.c:457-468 seam ported
     *    to the CUDA store boundary). Attention stays on the LIVE cache (no recall-read reroute) —
     *    the injected rows ARE the live cache front. Owner layers only; sharers ride by construction
     *    (Kuse=dKc[src]). Intact replay == baseline (f32 store is lossless); zeroed -> diverge; unset
     *    -> bit-exact floor. Episode loaded read-only from ep.{mf,k,v} (the recall serialization). */
    const char *replay_e = getenv("SP_REPLAY");
    if (replay_e) {
        { const char *np = getenv("SP_REPLAY_NPOS"); replay_npos = np ? atoi(np) : 0; }
        { const char *z  = getenv("SP_REPLAY_ZERO"); replay_zero = (z && z[0] == '1'); }
        if (replay_npos <= 0 || replay_npos > P) { sp_set_error("SP_REPLAY_NPOS out of range (need 0<NPOS<=P)"); goto done; }
        char rp[1024]; FILE *rf; sp_xbar_manifest rmf; memset(&rmf, 0, sizeof(rmf));
        snprintf(rp, sizeof(rp), "%s/ep.mf", replay_e); rf = fopen(rp, "rb");
        if (!rf) { sp_set_error("SP_REPLAY: ep.mf open"); goto done; }
        fseek(rf, 0, SEEK_END); long rl = ftell(rf); fseek(rf, 0, SEEK_SET);
        uint8_t *rbm = (uint8_t *)malloc((size_t)rl);
        if (!rbm || fread(rbm, 1, (size_t)rl, rf) != (size_t)rl) { free(rbm); fclose(rf); sp_set_error("SP_REPLAY: ep.mf read"); goto done; }
        fclose(rf);
        if (sp_xbar_manifest_deserialize(&rmf, rbm, (size_t)rl)) { free(rbm); sp_set_error("SP_REPLAY: deserialize"); goto done; }
        free(rbm);
        if (rmf.P < replay_npos) { sp_xbar_manifest_free(&rmf); sp_set_error("SP_REPLAY: episode shorter than NPOS"); goto done; }
        replay_off = (size_t *)malloc((size_t)NL * sizeof(size_t));
        if (!replay_off) { sp_xbar_manifest_free(&rmf); sp_set_error("SP_REPLAY off OOM"); goto done; }
        for (int L = 0; L < NL; L++) replay_off[L] = (size_t)rmf.layers[L].off;
        size_t rsb = (size_t)rmf.store_bytes;
        if (cudaMalloc((void **)&d_replay_K, rsb) != cudaSuccess ||
            cudaMalloc((void **)&d_replay_V, rsb) != cudaSuccess) { sp_xbar_manifest_free(&rmf); sp_set_error("SP_REPLAY store OOM"); goto done; }
        uint8_t *rs = (uint8_t *)malloc(rsb);
        if (!rs) { sp_xbar_manifest_free(&rmf); sp_set_error("SP_REPLAY host OOM"); goto done; }
        snprintf(rp, sizeof(rp), "%s/ep.k", replay_e); rf = fopen(rp, "rb");
        if (!rf || fread(rs, 1, rsb, rf) != rsb) { free(rs); if (rf) fclose(rf); sp_xbar_manifest_free(&rmf); sp_set_error("SP_REPLAY: ep.k read"); goto done; }
        fclose(rf); cudaMemcpy(d_replay_K, rs, rsb, cudaMemcpyHostToDevice);
        snprintf(rp, sizeof(rp), "%s/ep.v", replay_e); rf = fopen(rp, "rb");
        if (!rf || fread(rs, 1, rsb, rf) != rsb) { free(rs); if (rf) fclose(rf); sp_xbar_manifest_free(&rmf); sp_set_error("SP_REPLAY: ep.v read"); goto done; }
        fclose(rf); cudaMemcpy(d_replay_V, rs, rsb, cudaMemcpyHostToDevice); free(rs);
        sp_xbar_manifest_free(&rmf);
        replay_on = 1;
        fprintf(stderr, "    [xbar-p3.3] SP_REPLAY ON: inject owner K/V over prefill rows [0,%d)%s\n",
                replay_npos, replay_zero ? " ZEROED (G-P3-SHARED divergence control)" : " (intact)");
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
                        if (replay_on && pos < replay_npos) {   /* P3.3: episode block over the just-minted owner row, before attention reads it */
                            if (replay_zero) {
                                cudaMemsetAsync(dKc[L] + (size_t)pos*kvd, 0, (size_t)kvd*sizeof(float), st);
                                cudaMemsetAsync(dVc[L] + (size_t)pos*kvd, 0, (size_t)kvd*sizeof(float), st);
                            } else {
                                cudaMemcpyAsync(dKc[L] + (size_t)pos*kvd, (const float *)(d_replay_K + replay_off[L]) + (size_t)pos*kvd,
                                                (size_t)kvd*sizeof(float), cudaMemcpyDeviceToDevice, st);
                                cudaMemcpyAsync(dVc[L] + (size_t)pos*kvd, (const float *)(d_replay_V + replay_off[L]) + (size_t)pos*kvd,
                                                (size_t)kvd*sizeof(float), cudaMemcpyDeviceToDevice, st);
                            }
                        }
                        Kuse = dKc[L]; Vuse = dVc[L];
                    } else {
                        const int src = kvfs - (global ? 1 : 2);
                        Kuse = dKc[src]; Vuse = dVc[src];
                        if (!Kuse || !Vuse) { sp_set_error("g4 sharer before owner"); goto done; }
                    }
                    {   const int ctx = pos + 1;
                        int bd = hd > ctx ? hd : ctx; if (bd > 1024) bd = 1024;
                        if (sp_byteexact_attn())
                            k_attn_decode_win_bx<<<nh, bd, (size_t)ctx*sizeof(long long), st>>>(
                                dq, Kuse, Vuse, ctx, kvd, hd, grp, 1.0f, win, dao);
                        else
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
                    /* P3.3 SP_REPLAY (VELOCITY path — the one taken when use_graph is off, i.e. under the gate):
                     * inject the episode owner row over the just-minted [0,NPOS) prefill row before attention. */
                    if (replay_on && pos < replay_npos) {
                        if (replay_zero) {
                            cudaMemsetAsync(dKc[L] + wslot*kvd, 0, (size_t)kvd*sizeof(float), st);
                            cudaMemsetAsync(dVc[L] + wslot*kvd, 0, (size_t)kvd*sizeof(float), st);
                        } else {
                            cudaMemcpyAsync(dKc[L] + wslot*kvd, (const float *)(d_replay_K + replay_off[L]) + (size_t)pos*kvd,
                                            (size_t)kvd*sizeof(float), cudaMemcpyDeviceToDevice, st);
                            cudaMemcpyAsync(dVc[L] + wslot*kvd, (const float *)(d_replay_V + replay_off[L]) + (size_t)pos*kvd,
                                            (size_t)kvd*sizeof(float), cudaMemcpyDeviceToDevice, st);
                        }
                    }
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
                    if (sp_byteexact_attn())
                        k_attn_decode_win_bx<<<nh, bd, (size_t)ctx*sizeof(long long), st>>>(
                            dq, Kuse, Vuse, ctx, kvd, hd, grp, 1.0f, win, dao);
                    else
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
    if (xbar_wr_e && d_xbar_K && xmf.layers) {   /* P3.1b: serialize manifest + dump filled store to disk.
        * B4-NIGHTSHIFT write-once FIX: the `static int xbar_written` guard serialized the
        * episode dump to the FIRST gemma4_decode_cuda call in the process. The curator (PPL
        * harness) is a fresh process per episode so it never tripped, but the resident daemon
        * is ONE long-lived process: with the guard, ep_live_000 wrote and every later live
        * capture (ep_live_001+) silently skipped -> load_episode_global_k -> None. The WRITE
        * path keys on the SP_XBAR_RECALL_WRITE out_dir which is freshly set+unset per capture,
        * so it is correct to write EVERY call. SELFTEST/LOAD paths are elsewhere and untouched. */
        {
            cudaStreamSynchronize(st);
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
    if (d_replay_K) cudaFree(d_replay_K); if (d_replay_V) cudaFree(d_replay_V); if (replay_off) free(replay_off);  /* P3.3 SP_REPLAY */
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
    /* EAGLE/MTP feature tap (serving): when feat_active, the next decode step D2H-copies its
     * post-output_norm hidden s->dnx (E floats = the feature the LM head consumes) into
     * feat_host. This is the seed inp_h for the gemma4-assistant draft. One-shot. */
    float *feat_host; int feat_active;
    /* CONTRACT-CHAT-FULLSTACK B3-v2 — AUTONOMOUS RECALL by q·K attention relevance.
     * When qcap_active, g4_kv_step D2H-copies, per GLOBAL owner layer (period-6,
     * ascending), this step's last-token post-RoPE query dq (g_nh*g_hd floats) into
     * qcap_host, packed [n_global][g_nh*g_hd] row-major. One-shot flag, cleared by the
     * step. This is the SAME resident decode path the daemon then scores against the
     * episode registry's stored global-K (ep.k) — the model's native attention
     * relevance ("does this query attend to this memory?"), replacing the centroid-
     * Hamming selector that failed B3-v1. Off by default = byte-untouched null floor. */
    float *qcap_host; int qcap_active, qcap_gi;
    /* CONTRACT-CHAT-FULLSTACK A1 — CUDA-graph fast decode. The decode GENERATE
     * step is topology-static (every position-dependent kernel reads the DEVICE
     * dpos pointer, exactly as the one-shot graph decode in gemma4_decode_cuda),
     * so it can be captured ONCE and replayed per token — collapsing ~960 host
     * kernel launches/token to a single cudaGraphLaunch. Off-by-default null
     * floor: only armed when SP_G4_KV_GRAPH=1 AND the step is graph-safe (do_head,
     * no inject/capture seam, no SWA-ring, attn_shm<=48KB, PLE device-ready).
     * graph_mode: -1 = undecided, 0 = disabled (stay per-step), 1 = armed. */
    int graph_mode, graph_ready;
    cudaGraph_t gcap; cudaGraphExec_t gexec;
    /* CONTRACT-CHAT-FULLSTACK B1 — per-session byte-exact "auditable mode".
     * bx_on mirrors the device __constant__ d_bx_flag on the host so g4_kv_step
     * can route the resident decode ATTENTION through the exact-integer kernel
     * (k_attn_decode_win_bx) — the islands (RMSNorm/RoPE/GELU) already branch on
     * d_bx_flag internally, but the decode attention used here
     * (k_attn_decode_win_dyn) does not. Set per-request via gemma4_kv_byteexact_set
     * under the resident-cache Mutex; default 0 = byte-identical null floor.
     * When on, the graph path is declined (the bx attention takes a host ctx +
     * ctx-sized i64 shared mem, not graph-capturable) — chat runs per-step. */
    int bx_on;
};

extern "C" void gemma4_kv_close(sp_g4_kv *s);   /* fwd: gemma4_kv_open frees on OOM */
static int g4_kv_step(sp_g4_kv *s, int do_head);  /* A1: fwd for g4_kv_step_graph fallback */

/* CONTRACT-CHAT-FULLSTACK A1 — the GRAPH-SAFE per-step kernel body (full cache,
 * no inject/capture seam, no SWA-ring). This is the EXACT kernel sequence
 * g4_kv_step launches on its full-cache, no-overlay path, factored out so it can
 * be either (a) launched directly per token, or (b) recorded into a CUDA graph
 * once and replayed. It contains NO host synchronization and NO host-conditional
 * D2H — a hard requirement for stream capture. Every position-dependent kernel
 * reads the DEVICE dpos pointer, so a graph captured at one position replays
 * bit-identically at any later position (the same invariant gemma4_decode_cuda's
 * one-shot graph relies on). do_head: run out_norm+tied head+softcap+argmax. */
static int g4_kv_launch_full(sp_g4_kv *s, int do_head) {
    const qwen3_model *m = s->m;
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
    if (g_w.embd) { dim3 grid(1, (E + 255) / 256);
        k_embed_scale_at<<<grid, 256, 0, st>>>(g_w.embd, dseq, dpos, E, embscale, s->dx); }
    else k_embed_packed_at<<<(unsigned)((E+255)/256), 256, 0, st>>>(
            g_w.embd_packed.codes, g_w.embd_packed.row_off, g_w.embd_packed.row_scale,
            g_w.embd_packed.row_prec, dseq, dpos, E, embscale, s->dx);
    if (PL) {
        k_ple_gather_at<<<(unsigned)((NLPL+255)/256), 256, 0, st>>>(
            g_w.pl_tok_embd.codes, g_w.pl_tok_embd.row_off, g_w.pl_tok_embd.row_scale,
            g_w.pl_tok_embd.row_prec, dseq, dpos, NLPL, s->dple);
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
            k_kv_store<<<(unsigned)((kvd+255)/256), 256, 0, st>>>(dKc[L], dVc[L], s->dk, s->dv, dpos, 0, kvd);
            Kuse = dKc[L]; Vuse = dVc[L];
        } else {
            const int src = kvfs - (global ? 1 : 2);
            Kuse = dKc[src]; Vuse = dVc[src];
            if (!Kuse || !Vuse) { sp_set_error("g4_kv: sharer before owner"); return -1; }
        }
        {   int bd = hd > 256 ? hd : 256; if (bd > 1024) bd = 1024;
            k_attn_decode_win_dyn<<<nh, bd, attn_shm, st>>>(
                s->dq, Kuse, Vuse, dpos, kvd, hd, grp, 1.0f, win, s->dao); }
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
        k_argmax_at<<<1, 256, 0, st>>>(s->dlog, V, dseq, dpos);
    }
    k_incr_pos<<<1, 1, 0, st>>>(dpos);
    (void)m;
    #undef KMMD
    return 0;
}

/* A1 — graph-safe decode step. Decides ONCE whether this cache can run on a
 * captured graph (SP_G4_KV_GRAPH=1, full-cache, no inject/capture seam,
 * attn_shm<=48KB, PLE device-ready when PL). If armed: capture g4_kv_launch_full
 * the first time, then replay the exec graph per call. If not armed, the caller
 * stays on the per-step g4_kv_step. Returns 0/-1. do_head must be 1 (decode). */
static int g4_kv_step_graph(sp_g4_kv *s) {
    cudaStream_t st = g_w.stream;
    if (s->graph_mode < 0) {
        static int want = -1;
        if (want < 0) { const char *e = getenv("SP_G4_KV_GRAPH"); want = (e && e[0]=='1') ? 1 : 0; }
        const size_t attn_shm = (size_t)s->Pmax * sizeof(float);
        /* B1: byte-exact decode uses the host-ctx bx attention (k_attn_decode_win_bx,
         * ctx-sized i64 shared mem) which is NOT graph-capturable; force per-step. */
        const int safe = want && s->ring_W == 0 && !s->inj_active && !s->cap_active
                         && !s->feat_active && !s->bx_on
                         && attn_shm <= 48u*1024u && (!s->PL || s->dev_ple);
        s->graph_mode = safe ? 1 : 0;
        if (safe) fprintf(stderr, "    [g4-kv] GRAPH decode armed (SP_G4_KV_GRAPH=1; capturing one decode step)\n");
    }
    if (s->graph_mode == 0) return g4_kv_step(s, /*do_head=*/1);

    /* Inject/capture cannot coexist with a replayed graph (they mutate dx via a
     * host-sync seam). If a seam fires mid-session, drop to per-step for it.
     * B1: byte-exact mode likewise routes the per-step bx attention (host ctx,
     * not graph-bakeable), so a session that armed graph then turned bx on this
     * request decodes per-step for the duration. */
    if (s->inj_active || s->cap_active || s->feat_active || s->bx_on) return g4_kv_step(s, /*do_head=*/1);

    if (!s->graph_ready) {
        if (cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
            sp_set_error("g4_kv graph begin capture"); return -1; }
        if (g4_kv_launch_full(s, /*do_head=*/1)) {
            cudaStreamEndCapture(st, &s->gcap); return -1; }
        if (cudaStreamEndCapture(st, &s->gcap) != cudaSuccess) {
            sp_set_error("g4_kv graph end capture"); return -1; }
        if (cudaGraphInstantiate(&s->gexec, s->gcap, NULL, NULL, 0) != cudaSuccess) {
            sp_set_error("g4_kv graph instantiate"); return -1; }
        s->graph_ready = 1;
        /* The capture itself executed one real step (capture mode records but
         * does NOT run the kernels), so dpos has NOT advanced — launch once to
         * make the first decode actually happen. */
        if (cudaGraphLaunch(s->gexec, st) != cudaSuccess) {
            sp_set_error("g4_kv graph first launch"); return -1; }
        return 0;
    }
    if (cudaGraphLaunch(s->gexec, st) != cudaSuccess) {
        sp_set_error("g4_kv graph launch"); return -1; }
    return 0;
}

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
    /* CONTRACT-CHAT-FULLSTACK B2 RING-FIX: SWA-owner undo-journal overflow.
     * The journal index for this step is j = dpos_host - commit_pos; the journal
     * is a flat [Jmax] buffer, so it can only hold Jmax uncommitted ticks. Chat
     * NEVER calls gemma4_kv_commit (forward decode never rewinds mid-turn), so
     * commit_pos would stay 0 and g4_kv_step hard-failed (return -1) once a turn
     * passed ~Jmax positions -> decode silently died past ~64 tokens (the
     * diagnosed B2 ring bug). Fix: when the journal would overflow, auto-advance
     * commit_pos so the journal slides to cover only the most-recent (Jmax-1)
     * ticks. Dropping rewind history older than that is SAFE for forward chat
     * (it never rewinds past the live turn); the curator/#222 rewind callers DO
     * call gemma4_kv_commit, so their j stays bounded and this never trips for
     * them. Done ONCE per step (before the layer loop) so every owner layer in
     * this step writes the SAME journal slot j. */
    if (s->ring_W > 0 && s->jK && s->Jmax > 0) {
        if ((size_t)(s->dpos_host - s->commit_pos) >= (size_t)s->Jmax) {
            s->commit_pos = s->dpos_host - (s->Jmax - 1);
            if (s->jcur > s->Jmax - 1) s->jcur = s->Jmax - 1;
        }
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
        /* B3-v2: capture the last-token post-RoPE query on each GLOBAL owner layer
         * for the daemon's q·K attention-relevance recall (one-shot, read-only D2H). */
        if (s->qcap_active && global && s->qcap_host) {
            float *dst = s->qcap_host + (size_t)s->qcap_gi * (g_nh * g_hd);
            cudaMemcpyAsync(dst, s->dq, (size_t)(g_nh * g_hd) * sizeof(float),
                            cudaMemcpyDeviceToHost, st);
            s->qcap_gi++;
        }
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
                if (s->bx_on) {                       /* B2 RING-FIX: byte-exact ring (build-deterministic;
                    * keeps S1's FP-reorder immunity on the 40 SWA layers when the ring is armed). Shared
                    * = wl*int64; widen the block to cover the window like the non-ring bx path. */
                    int bdb = hd > wl ? hd : wl; if (bdb > 1024) bdb = 1024;
                    k_attn_decode_ring_bx<<<nh, bdb, (size_t)wl*sizeof(long long), st>>>(
                        s->dq, Kuse, Vuse, ctx, kvd, hd, grp, 1.0f, win, s->ring_W, s->dao);
                } else {
                    k_attn_decode_ring<<<nh, bd, (size_t)wl*sizeof(float), st>>>(
                        s->dq, Kuse, Vuse, ctx, kvd, hd, grp, 1.0f, win, s->ring_W, s->dao);
                }
            } else if (s->bx_on) {                    /* B1: byte-exact (auditable) decode attention.
                * The exact-integer kernel takes a HOST ctx (= dpos_host+1) — this is the
                * per-step (non-graph) path, so dpos_host is the live position. Shared mem
                * is ctx*int64 (the dual-prime score lane). Windowing handled inside via win. */
                const int ctx = s->dpos_host + 1;
                int bdb = hd > ctx ? hd : ctx; if (bdb > 1024) bdb = 1024;
                k_attn_decode_win_bx<<<nh, bdb, (size_t)ctx*sizeof(long long), st>>>(
                    s->dq, Kuse, Vuse, ctx, kvd, hd, grp, 1.0f, win, s->dao);
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
        if (s->feat_active && s->feat_host) {   /* EAGLE/MTP feature tap: post-output_norm hidden (seed inp_h) */
            cudaMemcpy(s->feat_host, s->dnx, (size_t)E * sizeof(float), cudaMemcpyDeviceToHost);
            s->feat_active = 0;
        }
        if (g_w.head.f32 || g_w.head.codes) { KMMD(&g_w.head, s->dnx, s->dlog); }
        else if (use_int8 && g_w.embd_packed.codes &&
                 gemv_w_packed(st, &g_w.embd_packed, s->dnx, s->dlog, s->dqx, s->dsx)) { }
        else { if (gemm(cb, g_w.embd, s->dnx, s->dlog, 1, E, V)) return -1; }
        if (softcap > 0.0f)
            k_softcap<<<(unsigned)(((size_t)V+255)/256), 256, 0, st>>>(s->dlog, (size_t)V, softcap);
        k_argmax_at<<<1, 256, 0, st>>>(s->dlog, V, dseq, dpos);   /* writes dseq[*dpos+1] */
    }
    k_incr_pos<<<1, 1, 0, st>>>(dpos);
    s->qcap_active = 0;   /* B3-v2: one-shot query capture consumed this step */
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
    /* A1: CUDA-graph decode state (lazily armed in g4_kv_step_graph). */
    s->graph_mode = -1; s->graph_ready = 0; s->gcap = NULL; s->gexec = NULL;
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

/* WIRE-CUDA-DECODE-GEMMA4 §3.1.A — logits-returning persistent-KV decode step.
 *
 * The additive sibling of gemma4_kv_decode for the universal-daemon L1 kvdecode
 * verb (sp_session_register_kvdecode_backend → sp_decode_step → this). Forwards
 * ONE token at the live dpos through the SAME g4_kv_step path, but instead of the
 * internal greedy argmax it D2H-copies the post-head full-vocab logits row
 * [n_vocab] so L2 owns sampling (greedy / temperature / top-p / spec verify).
 *
 * `token` is written into dseq[dpos] first (the input the step embeds), so the
 * caller — not a prior step's argmax — controls the decode input. g4_kv_step
 * runs out_norm + tied head + softcap into s->dlog and ALSO writes its own argmax
 * into dseq[dpos+1] (harmless: L2 picks its own next token); dpos then advances.
 *
 * BIT-EXACT to gemma4_kv_decode when L2 argmaxes the returned row: identical
 * dseq[dpos] input ⇒ identical g4_kv_step arithmetic ⇒ identical s->dlog ⇒
 * identical argmax. The null floor gemma4_kv_decode / gemma4_decode_cuda are
 * BYTE-UNTOUCHED; this is a new extern "C" symbol that merely reuses the step.
 *
 * `logits` is caller-allocated [n_vocab] f32 (host). Returns 0 on success. */
extern "C" int gemma4_kv_decode_logits(sp_g4_kv *s, int32_t token, float *logits) {
    if (!s || !logits) { sp_set_error("gemma4_kv_decode_logits: bad args"); return -1; }
    if (s->dpos_host + 1 >= s->Pmax) { sp_set_error("gemma4_kv_decode_logits: exceeds Pmax"); return -1; }
    cudaStream_t st = g_w.stream;
    /* set the input token at the live device position (overrides any argmax-
     * predicted dseq[dpos]); the step embeds dseq[*dpos]. */
    if (cudaMemcpyAsync(s->dseq + s->dpos_host, &token, sizeof(int),
                        cudaMemcpyHostToDevice, st) != cudaSuccess) {
        sp_set_error("gemma4_kv_decode_logits: token H2D"); return -1; }
    /* A1: route through the CUDA-graph fast step when SP_G4_KV_GRAPH=1 (and the
     * cache is graph-safe). Off-by-default = the per-step g4_kv_step null floor.
     * Bit-identical either way: the graph records the SAME kernel sequence with
     * the SAME device-dpos reads. */
    if (g4_kv_step_graph(s)) return -1;
    /* D2H the post-softcap full-vocab logits row for the NEXT position. */
    if (cudaMemcpyAsync(logits, s->dlog, (size_t)s->V * sizeof(float),
                        cudaMemcpyDeviceToHost, st) != cudaSuccess) {
        sp_set_error("gemma4_kv_decode_logits: logits D2H"); return -1; }
    s->dpos_host++;
    cudaError_t e = cudaStreamSynchronize(st);
    if (e != cudaSuccess) return fail_cuda(e, "gemma4_kv_decode_logits sync");
    return 0;
}

/* SPEC-DECODE batched verify: forward B candidate tokens at [dpos,dpos+B) with the head ON at
 * every position, returning per-position logits (B×V) and optional per-position features (B×E),
 * with ONE host sync for the whole batch (vs one sync + one full-vocab D2H PER token in the
 * sequential gemma4_kv_decode_logits loop). Each step sets dseq[dpos]=toks[i] (overriding the
 * prior step's argmax-write), runs the SAME g4_kv_step, async-D2Hs its logits row. Bit-identical
 * arithmetic to B sequential decode_logits; the win is collapsing the per-token host overhead the
 * PROFILE showed dominates (35ms forward vs 477ms decode). feat_out=NULL skips feature capture
 * (the feature D2H is per-step synchronous; the drive arms it only where the recurrence needs it). */
extern "C" int gemma4_kv_decode_batch(sp_g4_kv *s, const int32_t *toks, int B,
                                      float *logits_out, float *feat_out) {
    if (!s || !toks || B <= 0 || !logits_out) { sp_set_error("gemma4_kv_decode_batch: bad args"); return -1; }
    if (s->dpos_host + B >= s->Pmax) { sp_set_error("gemma4_kv_decode_batch: exceeds Pmax"); return -1; }
    cudaStream_t st = g_w.stream;
    for (int i = 0; i < B; i++) {
        if (cudaMemcpyAsync(s->dseq + s->dpos_host, &toks[i], sizeof(int),
                            cudaMemcpyHostToDevice, st) != cudaSuccess) {
            sp_set_error("gemma4_kv_decode_batch: tok H2D"); return -1; }
        if (feat_out) { s->feat_host = feat_out + (size_t)i * s->E; s->feat_active = 1; }
        if (g4_kv_step(s, /*do_head=*/1)) return -1;
        if (cudaMemcpyAsync(logits_out + (size_t)i * s->V, s->dlog, (size_t)s->V * sizeof(float),
                            cudaMemcpyDeviceToHost, st) != cudaSuccess) {
            sp_set_error("gemma4_kv_decode_batch: logits D2H"); return -1; }
        s->dpos_host++;
    }
    cudaError_t e = cudaStreamSynchronize(st);
    if (e != cudaSuccess) return fail_cuda(e, "gemma4_kv_decode_batch sync");
    return 0;
}

/* CONTRACT-CHAT-FULLSTACK B1 — per-session byte-exact "auditable mode" toggle.
 *
 * Sets the device __constant__ d_bx_flag (the same flag the one-shot/forward path
 * sets ONCE from SP_BYTEEXACT) so the resident-decode nonlinear islands
 * (RMSNorm/RoPE/GELU/AltUp-gate, which all branch on d_bx_flag) run exact-integer,
 * AND mirrors it on the host (s->bx_on) so g4_kv_step routes the decode ATTENTION
 * through the exact-integer k_attn_decode_win_bx (the decode attention kernel
 * k_attn_decode_win_dyn has no internal d_bx_flag branch; the islands do).
 *
 * Per-request callable: the chat path serializes on the resident-cache Mutex, so
 * the daemon sets on=1 at request start and on=0 at request end. Default (never
 * called / on=0) = byte-identical null floor — the Stage-A float path is untouched.
 *
 * The memcpy-to-symbol is synchronous (the cache's stream is idle between requests
 * under the Mutex), so the flag is in place before the next decode launches.
 * Returns 0 on success, -1 on a bad handle / CUDA error. */
extern "C" int gemma4_kv_byteexact_set(sp_g4_kv *s, int on) {
    if (!s) { sp_set_error("gemma4_kv_byteexact_set: NULL handle"); return -1; }
    int bx = on ? 1 : 0;
    cudaError_t e = cudaMemcpyToSymbol(d_bx_flag, &bx, sizeof(int), 0, cudaMemcpyHostToDevice);
    if (e != cudaSuccess) return fail_cuda(e, "gemma4_kv_byteexact_set: d_bx_flag H2D");
    s->bx_on = bx;
    return 0;
}

/* CONTRACT-CUDA-KV-FOUNDATION — set the per-session KV codec flags (bit0 = SP_KV_SPINOR)
 * on the resident decode cache. flags==0 = float null floor (default). Mirrors
 * gemma4_kv_byteexact_set; consumed by the KV alloc/read when the O_K Spinor carrier lands. */
extern "C" int gemma4_kv_set_kv_flags(sp_g4_kv *s, unsigned int flags) {
    if (!s) { sp_set_error("gemma4_kv_set_kv_flags: NULL handle"); return -1; }
    cudaError_t e = cudaMemcpyToSymbol(d_kv_flags, &flags, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
    if (e != cudaSuccess) return fail_cuda(e, "gemma4_kv_set_kv_flags: d_kv_flags H2D");
    return 0;
}

/* WIRE-CUDA-DECODE-GEMMA4 gate helper — read-only D2H peek of the resident
 * dseq window [from, from+n). Used by the G-WIRE-CUDA-DECODE-GEMMA4 oracle to
 * recover the EXACT token sequence the null-floor gemma4_kv_decode CONSUMED
 * (dseq[from] is the prefill-head prediction; the rest are its greedy outputs),
 * so the daemon's kvdecode path can be teacher-forced with identical inputs and
 * the two argmax streams compared bit-for-bit. Read-only/additive; touches no
 * decode arithmetic. 0 on success. */
extern "C" int gemma4_kv_seq_peek(const sp_g4_kv *s, int32_t *out, int from, int n) {
    if (!s || !out || from < 0 || n <= 0 || from + n > s->Pmax) {
        sp_set_error("gemma4_kv_seq_peek: bad args"); return -1; }
    cudaStreamSynchronize(g_w.stream);
    if (cudaMemcpy(out, s->dseq + from, (size_t)n * sizeof(int),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        sp_set_error("gemma4_kv_seq_peek: dseq D2H"); return -1; }
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

/* G-INT-2-FIX: COLD reset. gemma4_kv_reset only rewinds the position counters
 * (dpos/commit_pos/jcur) — correct for the in-order full-cache decode (slot==pos,
 * attention bounds at dpos, stale slots overwritten next prefill). But the B3-JUDGE
 * branch runs a NESTED forward that advances dpos PAST the prompt anchor, writing
 * judge K/V at global slots [n-1, jhead+J). A plain reset()+prefill(head) leaves
 * those judge entries resident at slots >= n-1; on a NULL turn the final decode_step
 * sits at pos=n-1 (judge residue is beyond dpos => never read => clean), but on a
 * PICK turn the injected memory advances dpos forward, so the synthesis window can
 * sweep over the stale judge slots => prompt-echo degeneration. This verb ADDITIONALLY
 * zeroes every owner K/V cache (and the SWA undo-journal) so the reconstruction truly
 * starts cold: no residue of ANY prior pass can be attended. Byte-identical to the
 * normal null-floor path (a plain reset()+prefill(head) at request start produces the
 * SAME attended state, since prefill overwrites [0,dpos) and attention bounds at dpos;
 * zeroing the rest of the cache changes nothing that is read). */
extern "C" int gemma4_kv_reset_cold(sp_g4_kv *s) {
    if (!s) return -1;
    cudaStream_t st = g_w.stream;
    for (int L = 0; L < s->kvfs && L < s->NL; L++) {
        const int global = ((L % s->period) == s->period - 1);
        const int kvd = (global ? s->g_nkv : s->s_nkv) * (global ? s->g_hd : s->s_hd);
        const size_t slots = (s->ring_W > 0 && !global) ? (size_t)s->ring_W : (size_t)s->Pmax;
        if (s->dKc && s->dKc[L]) cudaMemsetAsync(s->dKc[L], 0, slots * (size_t)kvd * sizeof(float), st);
        if (s->dVc && s->dVc[L]) cudaMemsetAsync(s->dVc[L], 0, slots * (size_t)kvd * sizeof(float), st);
        if (s->ring_W > 0 && !global) {
            if (s->jK && s->jK[L]) cudaMemsetAsync(s->jK[L], 0, (size_t)s->Jmax * (size_t)kvd * sizeof(float), st);
            if (s->jV && s->jV[L]) cudaMemsetAsync(s->jV[L], 0, (size_t)s->Jmax * (size_t)kvd * sizeof(float), st);
        }
    }
    cudaError_t e = cudaStreamSynchronize(st);
    if (e != cudaSuccess) return fail_cuda(e, "gemma4_kv_reset_cold memset sync");
    s->dpos_host = 0; s->commit_pos = 0; s->jcur = 0;
    int zero = 0;
    if (cudaMemcpy(s->dpos, &zero, sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
        sp_set_error("gemma4_kv_reset_cold: dpos H2D"); return -1;
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

/* EAGLE/MTP feature tap (serving): arm a capture so the NEXT decode step D2H-copies its
 * post-output_norm hidden (E floats = the feature the LM head consumes) into feat_out — the
 * seed inp_h for the gemma4-assistant draft. One-shot; forces the per-step path that turn. */
extern "C" int gemma4_kv_capture_feat(sp_g4_kv *s, float *feat_out) {
    if (!s || !feat_out) return -1;
    s->feat_host = feat_out; s->feat_active = 1;
    return 0;
}

/* ===================== EAGLE/MTP draft (gemma4-assistant) — WEIGHT LOADER =====================
 * Loads the draft GGUF weights onto the GPU so the served decode can run the speculative draft
 * against the resident sp_g4_kv KV ring (dKc[NL-1] full / dKc[NL-2] SWA, via k_attn_decode_ring)
 * + the gemma4_kv_capture_feat feature. The draft FORWARD MATH is already proven:
 * G-EAGLE-DRAFT-FWD-CUDA GREEN (tools/eagle/sp_eagle_fwd_cuda.cu). This is the loader half (no
 * open details); the draft_step + K-step drive + batched verify wire is the live-daemon session
 * (resolves full-layer rope_freqs, the 12B embd packing for the input-embed gather, and the
 * accept-length/tok-s gate which only exists with the running 12B). */
#define DRAFT_NL 4
static const int g_draft_swa[DRAFT_NL] = {1, 1, 1, 0};   /* layers 0-2 SWA (read target NL-2), 3 full (NL-1) */
struct DraftWeights {
    int loaded; gguf_ctx *g;
    float *pre, *post, *out_norm, *head;      /* head = draft token_embd [Vd x 1024] (tied) */
    float *rope_freqs;                        /* [hd_full/2] proportional RoPE freq-factors (full layer) */
    int Vd, BBt;                              /* draft vocab; target hidden (pre in / post out) */
    float *attn_norm[DRAFT_NL], *wq[DRAFT_NL], *qn[DRAFT_NL], *wo[DRAFT_NL], *post_attn[DRAFT_NL],
          *ffn_norm[DRAFT_NL], *wg[DRAFT_NL], *wu[DRAFT_NL], *wd[DRAFT_NL], *post_ffw[DRAFT_NL], *osc[DRAFT_NL];
    int qd[DRAFT_NL], hd[DRAFT_NL], ff[DRAFT_NL];
};
static DraftWeights g_draft = {};

/* y[r] = sum_c W[r*cols+c]*x[c] (double accum) — the draft's f32 matmul (used by draft_step). */
__global__ void k_matmul_draft(const float *W, const float *x, float *y, int rows, int cols) {
    int r = blockIdx.x * blockDim.x + threadIdx.x; if (r >= rows) return;
    const float *w = W + (size_t)r * cols; double a = 0;
    for (int c = 0; c < cols; c++) a += (double)w[c] * x[c]; y[r] = (float)a;
}

/* dequant a draft GGUF tensor [out x in] -> device f32; fills *in_o/*out_o (out=1 for a vector). */
static float *draft_up(gguf_ctx *g, const char *name, int *in_o, int *out_o) {
    const gguf_tensor *t = gguf_find_tensor(g, name);
    if (!t) { sp_set_error("draft_up: missing draft tensor"); return NULL; }
    int in = (int)t->dims[0], out = (t->n_dims >= 2) ? (int)t->dims[1] : 1;
    const uint8_t *base = (const uint8_t *)gguf_tensor_data(g, t);
    size_t rb = row_bytes(t->type, in);
    if (!base || rb == 0) { sp_set_error("draft_up: null/unsupported tensor"); return NULL; }
    size_t n = (size_t)out * in; float *host = (float *)malloc(n * sizeof(float));
    if (!host) { sp_set_error("draft_up: host OOM"); return NULL; }
    for (int j = 0; j < out; j++)
        if (sp_dequant_row(base + (size_t)j * rb, t->type, in, host + (size_t)j * in)) {
            free(host); sp_set_error("draft_up: dequant failed"); return NULL; }
    float *dev = NULL; cudaError_t e = cudaMalloc(&dev, n * sizeof(float));
    if (e == cudaSuccess) e = cudaMemcpy(dev, host, n * sizeof(float), cudaMemcpyHostToDevice);
    free(host);
    if (e != cudaSuccess) { fail_cuda(e, "draft_up memcpy"); if (dev) cudaFree(dev); return NULL; }
    if (in_o) *in_o = in; if (out_o) *out_o = out;
    return dev;
}

extern "C" void gemma4_draft_close(void);

/* Load the gemma4-assistant draft GGUF weights onto the GPU. Returns 0 / -1. */
extern "C" int gemma4_draft_open(const char *gguf_path) {
    if (g_draft.loaded) return 0;
    gguf_ctx *g = gguf_open(gguf_path);
    if (!g) { sp_set_error("gemma4_draft_open: cannot open draft gguf"); return -1; }
    memset(&g_draft, 0, sizeof g_draft); g_draft.g = g;
    int in = 0, out = 0;
    #define DUP(dst, nm) do { (dst) = draft_up(g, nm, &in, &out); if (!(dst)) { gemma4_draft_close(); return -1; } } while (0)
    DUP(g_draft.pre,  "nextn.pre_projection.weight");                 /* [7680 x 1024]: out=1024,in=7680 */
    DUP(g_draft.post, "nextn.post_projection.weight"); g_draft.BBt = out;  /* [1024 x 3840]: out=3840 = target hidden */
    DUP(g_draft.out_norm, "output_norm.weight");
    DUP(g_draft.head, "token_embd.weight"); g_draft.Vd = out;          /* [1024 x Vd]: out=Vd */
    for (int il = 0; il < DRAFT_NL; il++) {
        char nm[80];
        #define LDUP(dst, suf) do { snprintf(nm, sizeof nm, "blk.%d." suf, il); DUP(dst, nm); } while (0)
        LDUP(g_draft.attn_norm[il], "attn_norm.weight");
        LDUP(g_draft.wq[il], "attn_q.weight"); g_draft.qd[il] = out; g_draft.hd[il] = out / 16;
        LDUP(g_draft.qn[il], "attn_q_norm.weight");
        LDUP(g_draft.wo[il], "attn_output.weight");
        LDUP(g_draft.post_attn[il], "post_attention_norm.weight");
        LDUP(g_draft.ffn_norm[il], "ffn_norm.weight");
        LDUP(g_draft.wg[il], "ffn_gate.weight"); g_draft.ff[il] = out;
        LDUP(g_draft.wu[il], "ffn_up.weight");
        LDUP(g_draft.wd[il], "ffn_down.weight");
        LDUP(g_draft.post_ffw[il], "post_ffw_norm.weight");
        LDUP(g_draft.osc[il], "layer_output_scale.weight");
        #undef LDUP
    }
    #undef DUP
    /* full-layer proportional RoPE freq-factors (gemma4 globals use them; SWA layers don't).
     * Optional: older drafts may lack it -> NULL falls back to plain base-rope. */
    if (gguf_find_tensor(g, "rope_freqs.weight")) {
        int ri = 0, ro = 0;
        g_draft.rope_freqs = draft_up(g, "rope_freqs.weight", &ri, &ro);
    }
    g_draft.loaded = 1;
    fprintf(stderr, "    [draft] gemma4-assistant loaded: %d layers hd=%d/%d/%d/%d Vd=%d BBt=%d\n",
            DRAFT_NL, g_draft.hd[0], g_draft.hd[1], g_draft.hd[2], g_draft.hd[3], g_draft.Vd, g_draft.BBt);
    return 0;
}

extern "C" void gemma4_draft_close(void) {
    float *glob[] = { g_draft.pre, g_draft.post, g_draft.out_norm, g_draft.head };
    for (float *p : glob) if (p) cudaFree(p);
    for (int il = 0; il < DRAFT_NL; il++) {
        float *ls[] = { g_draft.attn_norm[il],g_draft.wq[il],g_draft.qn[il],g_draft.wo[il],g_draft.post_attn[il],
                        g_draft.ffn_norm[il],g_draft.wg[il],g_draft.wu[il],g_draft.wd[il],g_draft.post_ffw[il],g_draft.osc[il] };
        for (float *x : ls) if (x) cudaFree(x);
    }
    if (g_draft.g) gguf_close(g_draft.g);
    memset(&g_draft, 0, sizeof g_draft);
}

/* SPEC-DECODE: device-side suppress — ONE kernel sets logits[ids[i]]=-inf over the soft/control
 * token set, replacing the per-id cudaMemcpy loop in gemma4_draft_step (n_suppress=6248 synchronous
 * H2D copies PER step = the measured draft bottleneck, ~62ms of the 89ms). The id list is uploaded
 * once and cached on-device by the caller. */
__global__ void k_suppress_ids(float *logits, const int *ids, int n, int V) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { int id = ids[i]; if (id >= 0 && id < V) logits[id] = -3.4e38f; }
}

/* One EAGLE/MTP draft step on the LIVE target ring. Given the target's feature h (BBt floats,
 * host; from gemma4_kv_capture_feat) and the last token, runs the proven draft forward reading
 * s->dKc[NL-1 full]/dKc[NL-2 SWA] (via k_attn_decode_ring) and writes the draft's argmax to
 * *out_token (+ optional h_next, BBt host). The draft owns no KV — recurrence is via h, which the
 * caller threads (h_{k+1} = out_hnext). Attention scale is env-tunable (SP_DRAFT_ASCALE=one|rsqrt;
 * default rsqrt = the proven-standalone choice) — the live accept rate dials it. rope = plain neox
 * base (full-layer rope_freqs is the next knob if accept is low). Math == G-EAGLE-DRAFT-FWD-CUDA. */
extern "C" int gemma4_draft_step(sp_g4_kv *s, const float *feat_host, int token,
                                 const int *suppress, int n_suppress,
                                 int *out_token, float *out_hnext) {
    if (!g_draft.loaded || !s || !out_token) { sp_set_error("gemma4_draft_step: bad args/not loaded"); return -1; }
    if (!g_w.embd && !g_w.embd_packed.codes) { sp_set_error("gemma4_draft_step: target has no embd (f32 or packed)"); return -1; }
    const int HID = 1024, BBt = g_draft.BBt, Vd = g_draft.Vd, FFm = 8192;
    static float *sx=0,*sfeat=0,*sxh=0,*scur=0,*snorm=0,*satt=0,*sao=0,*sq=0,*sff=0,*sff2=0,*slog=0,*shn=0; static int *dtok=0;
    if (!sxh) {
        cudaMalloc(&sx,BBt*4); cudaMalloc(&sfeat,BBt*4); cudaMalloc(&sxh,2*BBt*4); cudaMalloc(&scur,HID*4);
        cudaMalloc(&snorm,HID*4); cudaMalloc(&satt,HID*4); cudaMalloc(&sao,16*512*4); cudaMalloc(&sq,16*512*4);
        cudaMalloc(&sff,FFm*4); cudaMalloc(&sff2,FFm*4); cudaMalloc(&slog,(size_t)Vd*4); cudaMalloc(&shn,BBt*4); cudaMalloc(&dtok,4);
    }
    const float eps = s->eps; const int ctx = s->dpos_host;            /* positions 0..ctx-1 are written */
    const char *am = getenv("SP_DRAFT_ASCALE");
    cublasSetStream(g_w.cublas, 0);   /* draft matmuls via cublas on the default stream (ordered with the k_* kernels) */
    /* x = TARGET tok_embd[token] * sqrt(BBt) ; xh = [x ; feat] ; cur = pre_proj @ xh.
     * The served 12B (OK_Q4B recipe) keeps the embd PACKED (g_w.embd_packed), not f32 — gather
     * via the packed twin; fall back to f32 when present. (Resolved at the first live run.) */
    if (g_w.embd)
        k_embed_scale_one<<<(unsigned)((BBt+255)/256),256>>>(g_w.embd, token, BBt, sqrtf((float)BBt), sx);
    else
        k_embed_packed_one<<<(unsigned)((BBt+255)/256),256>>>(g_w.embd_packed.codes, g_w.embd_packed.row_off,
            g_w.embd_packed.row_scale, g_w.embd_packed.row_prec, token, BBt, sqrtf((float)BBt), sx);
    cudaMemcpy(sfeat, feat_host, (size_t)BBt*4, cudaMemcpyHostToDevice);
    cudaMemcpy(sxh, sx, (size_t)BBt*4, cudaMemcpyDeviceToDevice);
    cudaMemcpy(sxh+BBt, sfeat, (size_t)BBt*4, cudaMemcpyDeviceToDevice);
    if (gemm(g_w.cublas, g_draft.pre, sxh, scur, 1, 2*BBt, HID)) return -1;   /* pre_proj */
    for (int il = 0; il < DRAFT_NL; il++) {
        const int qd = g_draft.qd[il], hd = g_draft.hd[il], swa = g_draft_swa[il];
        const int il_src = swa ? (s->NL - 2) : (s->NL - 1);
        const int HD   = swa ? s->s_hd  : s->g_hd;
        const int nkv  = swa ? s->s_nkv : s->g_nkv;
        const int KVD  = nkv * HD, group = (nkv > 0 ? 16 / nkv : 16);
        const int win  = swa ? s->SW : -1;
        const int Wring= swa ? (s->ring_W > 0 ? s->ring_W : s->Pmax) : s->Pmax;
        const float rb = swa ? s->s_base : s->g_base;
        const float asc = (am && !strcmp(am,"one")) ? 1.0f : 1.0f/sqrtf((float)hd);
        k_rmsnorm<<<1,256>>>(scur, g_draft.attn_norm[il], HID, eps, snorm);
        if (gemm(g_w.cublas, g_draft.wq[il], snorm, sq, 1, HID, qd)) return -1;
        k_rmsnorm_head<<<16,256>>>(sq, g_draft.qn[il], 16, hd, qd, eps);
        if (!swa && g_draft.rope_freqs)   /* full layer: proportional freq-factors (gemma4 global RoPE) */
            k_rope_freqs_at<<<16, hd/2>>>(sq, 16, hd, rb, g_draft.rope_freqs, ctx);
        else
            k_rope_at<<<16, hd/2>>>(sq, 16, hd, qd, rb, ctx);          /* SWA: plain base-rope; query pos = ctx */
        k_attn_decode_ring<<<16,256,(size_t)ctx*sizeof(float)>>>(sq, s->dKc[il_src], s->dVc[il_src],
                                                                 ctx, KVD, HD, group, asc, win, Wring, sao);
        if (gemm(g_w.cublas, g_draft.wo[il], sao, satt, 1, qd, HID)) return -1;
        k_rmsnorm<<<1,256>>>(satt, g_draft.post_attn[il], HID, eps, snorm);
        cudaMemcpy(satt, snorm, (size_t)HID*4, cudaMemcpyDeviceToDevice);
        k_add<<<(unsigned)((HID+255)/256),256>>>(satt, scur, (size_t)HID);   /* satt = rms(attn) + cur = attn_out */
        k_rmsnorm<<<1,256>>>(satt, g_draft.ffn_norm[il], HID, eps, snorm);
        if (gemm(g_w.cublas, g_draft.wg[il], snorm, sff, 1, HID, FFm)) return -1;
        if (gemm(g_w.cublas, g_draft.wu[il], snorm, sff2, 1, HID, FFm)) return -1;
        k_gelu_mul<<<(unsigned)((FFm+255)/256),256>>>(sff, sff2, (size_t)FFm);
        if (gemm(g_w.cublas, g_draft.wd[il], sff, snorm, 1, FFm, HID)) return -1;
        k_rmsnorm<<<1,256>>>(snorm, g_draft.post_ffw[il], HID, eps, scur);   /* scur = rms(ffn) */
        k_add<<<(unsigned)((HID+255)/256),256>>>(scur, satt, (size_t)HID);   /* cur = rms(ffn) + attn_out */
        k_scale_by_dev<<<(unsigned)((HID+255)/256),256>>>(scur, (size_t)HID, g_draft.osc[il]); /* cur *= layer_output_scale */
    }
    k_rmsnorm<<<1,256>>>(scur, g_draft.out_norm, HID, eps, snorm);           /* final hidden */
    if (out_hnext) { if (gemm(g_w.cublas, g_draft.post, snorm, shn, 1, HID, BBt)) return -1;
                     cudaMemcpy(out_hnext, shn, (size_t)BBt*4, cudaMemcpyDeviceToHost); }
    if (gemm(g_w.cublas, g_draft.head, snorm, slog, 1, HID, Vd)) return -1;   /* draft tied head (cublas) */
    if (suppress && n_suppress > 0) {   /* token-mgmt contract: mask soft/control tokens before argmax (parity w/ the target sampler) */
        static int *d_sup = 0; static int d_sup_n = 0;   /* cache the id list on-device (was 6248 cudaMemcpy/step) */
        if (d_sup_n != n_suppress) {
            if (d_sup) cudaFree(d_sup);
            if (cudaMalloc(&d_sup, (size_t)n_suppress * sizeof(int)) != cudaSuccess) { sp_set_error("draft suppress malloc"); return -1; }
            cudaMemcpy(d_sup, suppress, (size_t)n_suppress * sizeof(int), cudaMemcpyHostToDevice);
            d_sup_n = n_suppress;
        }
        k_suppress_ids<<<(unsigned)((n_suppress + 255) / 256), 256>>>(slog, d_sup, n_suppress, Vd);
    }
    k_argmax<<<1,256>>>(slog, Vd, dtok);
    cudaError_t e = cudaMemcpy(out_token, dtok, 4, cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) return fail_cuda(e, "gemma4_draft_step out_token D2H");
    return 0;
}

/* LATENT INTERCEPTOR body — gemma4_draft_step WITHOUT the 262k vocab head. Runs pre_proj + the 4
 * draft layers + out_norm and returns ONLY the 1024-d latent manifold (out_latent[1024], host). This
 * is the shared substrate the action/memory/tool heads tap (each a tiny HID->small projection). No
 * vocab gemm, no argmax, no suppress -> the per-call cost is the 4-layer body alone (CPU/Hexagon-able
 * once off-GPU). CONTRACT-LATENT-INTERCEPTOR.md. Math identical to gemma4_draft_step's body. */
extern "C" int gemma4_draft_body(sp_g4_kv *s, const float *feat_host, int token, float *out_latent) {
    if (!g_draft.loaded || !s || !out_latent) { sp_set_error("gemma4_draft_body: bad args/not loaded"); return -1; }
    if (!g_w.embd && !g_w.embd_packed.codes) { sp_set_error("gemma4_draft_body: target has no embd"); return -1; }
    const int HID = 1024, BBt = g_draft.BBt, FFm = 8192;
    static float *bx=0,*bfeat=0,*bxh=0,*bcur=0,*bnorm=0,*batt=0,*bao=0,*bq=0,*bff=0,*bff2=0;
    if (!bxh) {
        cudaMalloc(&bx,BBt*4); cudaMalloc(&bfeat,BBt*4); cudaMalloc(&bxh,2*BBt*4); cudaMalloc(&bcur,HID*4);
        cudaMalloc(&bnorm,HID*4); cudaMalloc(&batt,HID*4); cudaMalloc(&bao,16*512*4); cudaMalloc(&bq,16*512*4);
        cudaMalloc(&bff,FFm*4); cudaMalloc(&bff2,FFm*4);
    }
    const float eps = s->eps; const int ctx = s->dpos_host;
    const char *am = getenv("SP_DRAFT_ASCALE");
    cublasSetStream(g_w.cublas, 0);
    static int bprof = -1; if (bprof < 0) bprof = getenv("SP_DRAFT_BODY_PROFILE") ? 1 : 0;
    cudaEvent_t be0, be1; if (bprof) { cudaEventCreate(&be0); cudaEventCreate(&be1); cudaEventRecord(be0, 0); }
    if (g_w.embd)
        k_embed_scale_one<<<(unsigned)((BBt+255)/256),256>>>(g_w.embd, token, BBt, sqrtf((float)BBt), bx);
    else
        k_embed_packed_one<<<(unsigned)((BBt+255)/256),256>>>(g_w.embd_packed.codes, g_w.embd_packed.row_off,
            g_w.embd_packed.row_scale, g_w.embd_packed.row_prec, token, BBt, sqrtf((float)BBt), bx);
    /* internal copies ASYNC on stream 0 (the cublas + k_* stream) so the body pipelines instead of
     * draining the stream ~8x/call (the measured ~91ms gate was these host syncs, not compute). */
    cudaMemcpyAsync(bfeat, feat_host, (size_t)BBt*4, cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(bxh, bx, (size_t)BBt*4, cudaMemcpyDeviceToDevice, 0);
    cudaMemcpyAsync(bxh+BBt, bfeat, (size_t)BBt*4, cudaMemcpyDeviceToDevice, 0);
    if (gemv_t(g_w.cublas, g_draft.pre, bxh, bcur, 2*BBt, HID)) return -1;        /* pre_proj */
    for (int il = 0; il < DRAFT_NL; il++) {
        const int qd = g_draft.qd[il], hd = g_draft.hd[il], swa = g_draft_swa[il];
        const int il_src = swa ? (s->NL - 2) : (s->NL - 1);
        const int HD   = swa ? s->s_hd  : s->g_hd;
        const int nkv  = swa ? s->s_nkv : s->g_nkv;
        const int KVD  = nkv * HD, group = (nkv > 0 ? 16 / nkv : 16);
        const int win  = swa ? s->SW : -1;
        const int Wring= swa ? (s->ring_W > 0 ? s->ring_W : s->Pmax) : s->Pmax;
        const float rb = swa ? s->s_base : s->g_base;
        const float asc = (am && !strcmp(am,"one")) ? 1.0f : 1.0f/sqrtf((float)hd);
        k_rmsnorm<<<1,256>>>(bcur, g_draft.attn_norm[il], HID, eps, bnorm);
        if (gemv_t(g_w.cublas, g_draft.wq[il], bnorm, bq, HID, qd)) return -1;
        k_rmsnorm_head<<<16,256>>>(bq, g_draft.qn[il], 16, hd, qd, eps);
        if (!swa && g_draft.rope_freqs)
            k_rope_freqs_at<<<16, hd/2>>>(bq, 16, hd, rb, g_draft.rope_freqs, ctx);
        else
            k_rope_at<<<16, hd/2>>>(bq, 16, hd, qd, rb, ctx);
        k_attn_decode_ring<<<16,256,(size_t)ctx*sizeof(float)>>>(bq, s->dKc[il_src], s->dVc[il_src],
                                                                 ctx, KVD, HD, group, asc, win, Wring, bao);
        if (gemv_t(g_w.cublas, g_draft.wo[il], bao, batt, qd, HID)) return -1;
        k_rmsnorm<<<1,256>>>(batt, g_draft.post_attn[il], HID, eps, bnorm);
        cudaMemcpyAsync(batt, bnorm, (size_t)HID*4, cudaMemcpyDeviceToDevice, 0);
        k_add<<<(unsigned)((HID+255)/256),256>>>(batt, bcur, (size_t)HID);
        k_rmsnorm<<<1,256>>>(batt, g_draft.ffn_norm[il], HID, eps, bnorm);
        if (gemv_t(g_w.cublas, g_draft.wg[il], bnorm, bff, HID, FFm)) return -1;
        if (gemv_t(g_w.cublas, g_draft.wu[il], bnorm, bff2, HID, FFm)) return -1;
        k_gelu_mul<<<(unsigned)((FFm+255)/256),256>>>(bff, bff2, (size_t)FFm);
        if (gemv_t(g_w.cublas, g_draft.wd[il], bff, bnorm, FFm, HID)) return -1;
        k_rmsnorm<<<1,256>>>(bnorm, g_draft.post_ffw[il], HID, eps, bcur);
        k_add<<<(unsigned)((HID+255)/256),256>>>(bcur, batt, (size_t)HID);
        k_scale_by_dev<<<(unsigned)((HID+255)/256),256>>>(bcur, (size_t)HID, g_draft.osc[il]);
    }
    k_rmsnorm<<<1,256>>>(bcur, g_draft.out_norm, HID, eps, bnorm);               /* the 1024-d latent */
    if (bprof) { cudaEventRecord(be1, 0); cudaEventSynchronize(be1); float ms = 0; cudaEventElapsedTime(&ms, be0, be1);
                 fprintf(stderr, "[draft_body] GPU-kernels %.2fms (ctx=%d)\n", ms, ctx); cudaEventDestroy(be0); cudaEventDestroy(be1); }
    cudaError_t e = cudaMemcpy(out_latent, bnorm, (size_t)HID*4, cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) return fail_cuda(e, "gemma4_draft_body latent D2H");
    return 0;
}

/* EAGLE flywheel capture: D2H the TARGET KV context the draft attends — the global owner
 * (kvfs-1, kvd=g_nkv*g_hd) and SWA owner (kvfs-2, kvd=s_nkv*s_hd), positions [0,dpos).
 * Caller allocates kg/vg (Pmax*g_nkv*g_hd) and ks/vs (Pmax*s_nkv*s_hd); pass NULL to skip.
 * Fills kvd_g/kvd_s/npos. This is the (per-sequence) KV half of the training record. */
extern "C" int gemma4_kv_ctx_dump(sp_g4_kv *s, float *kg, float *vg, float *ks, float *vs,
                                  int *kvd_g, int *kvd_s, int *npos) {
    if (!s) { sp_set_error("gemma4_kv_ctx_dump: null s"); return -1; }
    const int gl = s->kvfs - 1, sl = s->kvfs - 2;
    if (gl < 0 || sl < 0) { sp_set_error("gemma4_kv_ctx_dump: kvfs<2"); return -1; }
    const int kg_d = s->g_nkv * s->g_hd, ks_d = s->s_nkv * s->s_hd, P = s->dpos_host;
    cudaStreamSynchronize(g_w.stream);
    if (kg && s->dKc[gl]) cudaMemcpy(kg, s->dKc[gl], (size_t)P * kg_d * sizeof(float), cudaMemcpyDeviceToHost);
    if (vg && s->dVc[gl]) cudaMemcpy(vg, s->dVc[gl], (size_t)P * kg_d * sizeof(float), cudaMemcpyDeviceToHost);
    if (ks && s->dKc[sl]) cudaMemcpy(ks, s->dKc[sl], (size_t)P * ks_d * sizeof(float), cudaMemcpyDeviceToHost);
    if (vs && s->dVc[sl]) cudaMemcpy(vs, s->dVc[sl], (size_t)P * ks_d * sizeof(float), cudaMemcpyDeviceToHost);
    if (kvd_g) *kvd_g = kg_d; if (kvd_s) *kvd_s = ks_d; if (npos) *npos = P;
    return 0;
}

/* EAGLE flywheel: gather the TARGET token embedding row * sqrt(E) (the draft's pre_proj input x)
 * for `token` into out[E] (host). Mirrors gemma4_draft_step's x computation so the captured x is
 * byte-exact to what the live draft feeds. f32 or packed embd. */
extern "C" int gemma4_embd_row(int token, int E, float *out) {
    static float *d = NULL; static int cap = 0;
    if (cap < E) { if (d) cudaFree(d); if (cudaMalloc(&d, (size_t)E * sizeof(float)) != cudaSuccess) return -1; cap = E; }
    const float sc = sqrtf((float)E);
    if (g_w.embd)
        k_embed_scale_one<<<(unsigned)((E+255)/256),256>>>(g_w.embd, token, E, sc, d);
    else if (g_w.embd_packed.codes)
        k_embed_packed_one<<<(unsigned)((E+255)/256),256>>>(g_w.embd_packed.codes, g_w.embd_packed.row_off,
            g_w.embd_packed.row_scale, g_w.embd_packed.row_prec, token, E, sc, d);
    else return -1;
    return cudaMemcpy(out, d, (size_t)E * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess ? 0 : -1;
}

/* Geometry the trainer needs (draft attention dims + the global/SWA layer indices + rope bases).
 * Lets the capture/trainer size buffers + replicate the draft attention without re-deriving. */
extern "C" int gemma4_kv_ctx_geom(sp_g4_kv *s, int *g_nkv, int *g_hd, int *s_nkv, int *s_hd,
                                  int *period, int *kvfs, float *g_base, float *s_base) {
    if (!s) return -1;
    if (g_nkv) *g_nkv = s->g_nkv; if (g_hd) *g_hd = s->g_hd;
    if (s_nkv) *s_nkv = s->s_nkv; if (s_hd) *s_hd = s->s_hd;
    if (period) *period = s->period; if (kvfs) *kvfs = s->kvfs;
    if (g_base) *g_base = s->g_base; if (s_base) *s_base = s->s_base;
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

/* CONTRACT-CHAT-FULLSTACK B5 — TEXT through the single latent entry seam.
 *
 * The operator's single-entry-point architecture (CONTRACT §6): every modality
 * enters the model through the ONE residual seam (gemma4_kv_inject* → the model
 * mints K/V natively). This is the TEXT source of that seam: per token id, stage
 * embd[id]*sqrt(E) DEVICE-SIDE into s->dinj (the same arithmetic k_embed_scale_at
 * runs in a stock step — f32 table or packed OK_Q4B), arm the inject override, and
 * step the SAME token id as the "placeholder" so the per-layer-embedding (PLE)
 * stream gathers from the REAL id (not an audio placeholder).
 *
 * PARITY (the B5 proof): because (a) dinj holds embd[id]*sqrt(E) computed by the
 * SAME kernel the stock embed-at step uses, so the dx override equals the value the
 * step would have produced, and (b) the step token == id so PLE/dseq match, the
 * residual entering layer 0 is BIT-IDENTICAL to gemma4_kv_prefill(&id,1). The whole
 * decode downstream is therefore bit-identical — text-via-inject == text-via-prefill,
 * by construction. The override path (inj_active → dx <- dinj in g4_kv_step) is
 * genuinely exercised: this is the audio/memory entry, fed by the text projector.
 *
 * NULL floor: dead code unless called; gemma4_kv_prefill / decode stay byte-untouched.
 * Returns 0 on success, -1 on any failure. */
/* atten=0 ⇒ full-strength inject (byte-identical to the prefill seam; the prompt-head
 * + B5 frame channel path). atten=1 ⇒ G-INT-2-FIX: attenuate the minted memory K by
 * the constant-budget alpha (the LIVE recall path — a recalled episode must BIND not
 * HIJACK). The thin gemma4_kv_inject_tokens / _atten wrappers below pin the flag. */
static int gemma4_kv_inject_tokens_impl(sp_g4_kv *s, const int32_t *toks, int n, int atten) {
    if (!s || !toks || n <= 0) { sp_set_error("gemma4_kv_inject_tokens: bad args"); return -1; }
    if (s->dpos_host + n > s->Pmax) { sp_set_error("gemma4_kv_inject_tokens: exceeds Pmax"); return -1; }
    if (!s->dinj) { if (cudaMalloc(&s->dinj, (size_t)s->E*sizeof(float)) != cudaSuccess) {
        sp_set_error("gemma4_kv_inject_tokens: dinj OOM"); return -1; } }
    cudaStream_t st = g_w.stream;
    const int E = s->E;
    const float embscale = s->embscale;   /* sqrt(hidden_size), == prefill's embed_scale */
    const int base = s->dpos_host;        /* the injected episode occupies [base, base+n) */
    for (int i = 0; i < n; i++) {
        const int32_t id = toks[i];
        /* Stage embd[id]*sqrt(E) into dinj using the resident embedding source
         * (f32 table or packed), arithmetic-identical to the stock embed-at step. */
        if (g_w.embd) {
            k_embed_scale_one<<<(unsigned)((E+255)/256), 256, 0, st>>>(g_w.embd, id, E, embscale, s->dinj);
        } else {
            k_embed_packed_one<<<(unsigned)((E+255)/256), 256, 0, st>>>(
                g_w.embd_packed.codes, g_w.embd_packed.row_off, g_w.embd_packed.row_scale,
                g_w.embd_packed.row_prec, id, E, embscale, s->dinj);
        }
        s->inj_active = 1;            /* g4_kv_step overrides dx <- dinj this position */
        /* Step the REAL id (not a placeholder) so dseq[dpos]=id ⇒ PLE gathers from id,
         * matching prefill exactly. gemma4_kv_prefill stores the H2D token + steps. */
        if (gemma4_kv_prefill(s, &id, 1)) return -1;
    }
    /* G-INT-2-FIX (Phase-4 sealer): port the curated replay path's CONSTANT-BUDGET
     * attention attenuation to the LIVE inject path. gemma4_kv_replay scales the stored
     * (post-RoPE) K by alpha=clamp(M_target/npos,0,1) so a recalled memory BINDS
     * (matched Q·K still clears softmax) instead of HIJACKS (full-mass injection right
     * before the question's last token loops generation). Here the K is MINTED natively
     * by the forward above into s->dKc[L] at the injected positions, so we scale those
     * rows IN-PLACE on-device (identical math to replay's kf[i]*=alpha, identical env
     * logic). V is left unscaled (full value at reduced attention). SP_REPLAY_MTARGET
     * (default 42 here — the proven B3-v9b operating point for the live judge path)
     * takes precedence over a fixed SP_REPLAY_ALPHA. alpha==1 ⇒ K untouched ⇒
     * byte-identical to the un-attenuated inject (null floor when M_target/n >= 1).
     * Gated by atten=1 so the prompt-head / B5-frame inject seam stays full-strength. */
    if (atten) {
        const char *mt_e = getenv("SP_REPLAY_MTARGET");
        const char *alpha_e = getenv("SP_REPLAY_ALPHA");
        float inj_alpha;
        if (mt_e) {
            const float M = (float)atof(mt_e);
            inj_alpha = (n > 0) ? (M / (float)n) : 1.0f;
        } else if (alpha_e) {
            inj_alpha = (float)atof(alpha_e);
        } else {
            inj_alpha = 42.0f / (float)n;   /* live judge default M_target=42 */
        }
        if (inj_alpha > 1.0f) inj_alpha = 1.0f;   /* never amplify a memory */
        if (inj_alpha < 0.0f) inj_alpha = 0.0f;
        if (inj_alpha != 1.0f) {
            for (int L = 0; L < s->kvfs && L < s->NL; L++) {
                const int global = ((L % s->period) == s->period - 1);
                const int kvd = (global ? s->g_nkv : s->s_nkv) * (global ? s->g_hd : s->s_hd);
                const int ring = (s->ring_W > 0 && !global);
                for (int p = 0; p < n; p++) {
                    const int pos = base + p;
                    const size_t slot = ring ? (size_t)(pos % s->ring_W) : (size_t)pos;
                    k_scale_by_const<<<(unsigned)((kvd+255)/256), 256, 0, st>>>(
                        s->dKc[L] + slot * kvd, (size_t)kvd, inj_alpha);
                }
            }
            cudaStreamSynchronize(st);
        }
        fprintf(stderr, "    [xbar-#222] gemma4_kv_inject_tokens: attenuated %d injected tok K into cache [%d,%d) alpha=%.3f eff_mass(a*npos)=%.1f\n",
                n, base, base + n, inj_alpha, inj_alpha * (float)n);
    }
    return 0;
}

/* B5 (§6e) text/frame seam — FULL-strength inject (byte-identical to the prefill seam). */
extern "C" int gemma4_kv_inject_tokens(sp_g4_kv *s, const int32_t *toks, int n) {
    return gemma4_kv_inject_tokens_impl(s, toks, n, /*atten=*/0);
}

/* G-INT-2-FIX live recall seam — constant-budget ATTENUATED inject (memory binds, not hijacks). */
extern "C" int gemma4_kv_inject_tokens_atten(sp_g4_kv *s, const int32_t *toks, int n) {
    return gemma4_kv_inject_tokens_impl(s, toks, n, /*atten=*/1);
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
/* C2 #222: replay-inject a stored episode's owner K/V DIRECTLY into the resident cache at
 * [dpos, dpos+npos) (full-cache slot==pos), then advance dpos. The persistent-ABI twin of the
 * one-shot SP_REPLAY: the curator SPECULATES a recalled memory here, scores it, and on reject
 * UNDOES it bit-exactly in O(1) via gemma4_kv_rewind(npos) (the KAI-1b slot==pos inverse — the
 * sheared [dpos,..) slots are never read again). zero=1 injects a ZEROED (corrupted) episode =
 * the reject control. Owner layers only (sharers ride owners by off-indirection). Ring-aware (KAI-1c):
 * in SWA-ring mode each clobbered ring slot is journaled before overwrite so rewind reconstructs the
 * window; globals stay full-cache (slot==pos, no journal). Returns 0 on success. */
extern "C" int gemma4_kv_replay(sp_g4_kv *s, const char *epdir, int npos, int zero) {
    if (!s || !epdir || npos <= 0) { sp_set_error("gemma4_kv_replay: bad args"); return -1; }
    if (s->dpos_host + npos > s->Pmax) { sp_set_error("gemma4_kv_replay: exceeds Pmax"); return -1; }
    /* WEIGHTED REFERENCE INJECTION (B3-v9): SP_REPLAY_ALPHA in [0,1] attenuates the injected
     * memory's ATTENTION WEIGHT by scaling its stored (post-RoPE) K by alpha ⇒ logit = Q·(alpha K)
     * = alpha·(Q·K) for exactly the memory slots. A MATCHED memory (huge Q·K) still clears the
     * softmax even at low alpha (conditions recall); a MISMATCHED memory (small Q·K) is crushed
     * (prompt keeps primacy ⇒ no hijack). V is left unscaled (full value at reduced attention).
     * alpha=1 = today's full-strength replay (B6). Decouples inform-when-relevant from
     * dominate-when-not — the length→weight fix for the mass-dominance hijack. */
    /* B3-v9b CONSTANT-BUDGET refinement: SP_REPLAY_MTARGET normalizes the memory's
     * thermodynamic budget by episode length — alpha = clamp(M_target / npos, 0, 1) — so
     * every episode (10 tokens or 1000) carries the SAME total sub-dominant attention
     * budget regardless of how long the source spoke (the alpha*npos invariant; the same
     * bounded-budget principle as the O(1) SWA ring). M_target is an EMPIRICAL knob:
     * alpha*npos is only a first-order proxy for the true softmax mass Sum_p exp(alpha*s_p)
     * (nonlinear in alpha, dependent on the Q.K values), so M_target is swept on the metal,
     * not closed-form. SP_REPLAY_MTARGET takes precedence over a fixed SP_REPLAY_ALPHA;
     * both unset => alpha=1.0 => stored K untouched => byte-identical null floor. */
    const char *mt_e = getenv("SP_REPLAY_MTARGET");
    const char *alpha_e = getenv("SP_REPLAY_ALPHA");
    float replay_alpha;
    if (mt_e) {
        const float M = (float)atof(mt_e);
        replay_alpha = (npos > 0) ? (M / (float)npos) : 1.0f;
        if (replay_alpha > 1.0f) replay_alpha = 1.0f;   /* never amplify a memory */
        if (replay_alpha < 0.0f) replay_alpha = 0.0f;
    } else {
        replay_alpha = alpha_e ? (float)atof(alpha_e) : 1.0f;
    }
    /* KAI-1c: in ring mode the replay overwrites SWA ring slots that may still hold live earlier
     * positions, so each clobbered slot is journaled BEFORE overwrite (exactly as g4_kv_step does)
     * ⇒ gemma4_kv_rewind reconstructs the pre-injection window. Needs journal headroom past commit. */
    if (s->ring_W > 0 && (s->dpos_host - s->commit_pos) + npos > s->Jmax) {
        sp_set_error("gemma4_kv_replay: ring journal depth exceeded (commit before replay)"); return -1; }
    char rp[1024]; FILE *rf; sp_xbar_manifest mf; memset(&mf, 0, sizeof(mf));
    snprintf(rp, sizeof rp, "%s/ep.mf", epdir); rf = fopen(rp, "rb");
    if (!rf) { sp_set_error("kv_replay: ep.mf open"); return -1; }
    fseek(rf, 0, SEEK_END); long ml = ftell(rf); fseek(rf, 0, SEEK_SET);
    uint8_t *mb = (uint8_t *)malloc((size_t)ml);
    if (!mb || fread(mb, 1, (size_t)ml, rf) != (size_t)ml) { free(mb); fclose(rf); sp_set_error("kv_replay: ep.mf read"); return -1; }
    fclose(rf);
    if (sp_xbar_manifest_deserialize(&mf, mb, (size_t)ml)) { free(mb); sp_set_error("kv_replay: deserialize"); return -1; }
    free(mb);
    if (mf.P < npos) { sp_xbar_manifest_free(&mf); sp_set_error("kv_replay: episode shorter than npos"); return -1; }
    size_t sb = (size_t)mf.store_bytes; char *dK = NULL, *dV = NULL;
    if (cudaMalloc((void **)&dK, sb) != cudaSuccess || cudaMalloc((void **)&dV, sb) != cudaSuccess) {
        if (dK) cudaFree(dK); sp_xbar_manifest_free(&mf); sp_set_error("kv_replay: dev OOM"); return -1; }
    uint8_t *hs = (uint8_t *)malloc(sb);
    if (!hs) { cudaFree(dK); cudaFree(dV); sp_xbar_manifest_free(&mf); sp_set_error("kv_replay: host OOM"); return -1; }
    snprintf(rp, sizeof rp, "%s/ep.k", epdir); rf = fopen(rp, "rb");
    if (!rf || fread(hs, 1, sb, rf) != sb) { if (rf) fclose(rf); free(hs); cudaFree(dK); cudaFree(dV); sp_xbar_manifest_free(&mf); sp_set_error("kv_replay: ep.k read"); return -1; }
    fclose(rf);
    if (replay_alpha != 1.0f) {   /* attenuate the memory's attention weight: scale stored K by alpha */
        float *kf = (float *)hs; size_t nf = sb / sizeof(float);
        for (size_t i = 0; i < nf; i++) kf[i] *= replay_alpha;
    }
    cudaMemcpy(dK, hs, sb, cudaMemcpyHostToDevice);
    snprintf(rp, sizeof rp, "%s/ep.v", epdir); rf = fopen(rp, "rb");
    if (!rf || fread(hs, 1, sb, rf) != sb) { if (rf) fclose(rf); free(hs); cudaFree(dK); cudaFree(dV); sp_xbar_manifest_free(&mf); sp_set_error("kv_replay: ep.v read"); return -1; }
    fclose(rf); cudaMemcpy(dV, hs, sb, cudaMemcpyHostToDevice); free(hs);
    cudaStream_t st = g_w.stream; int base = s->dpos_host;
    for (int L = 0; L < s->kvfs && L < s->NL; L++) {
        const int global = ((L % s->period) == s->period - 1);
        const int kvd = (global ? s->g_nkv : s->s_nkv) * (global ? s->g_hd : s->s_hd);
        const size_t off = (size_t)mf.layers[L].off;   /* owner prefix-sum byte offset */
        const int ring = (s->ring_W > 0 && !global);   /* SWA owner under a sliding window */
        for (int p = 0; p < npos; p++) {
            const int pos = base + p;
            const size_t slot = ring ? (size_t)(pos % s->ring_W) : (size_t)pos;   /* ring wraps; globals/full-cache slot==pos */
            if (ring) {                                  /* KAI-1c: checkpoint the slot we are about to clobber */
                const size_t j = (size_t)(pos - s->commit_pos);
                cudaMemcpyAsync(s->jK[L] + j*kvd, s->dKc[L] + slot*kvd, (size_t)kvd*sizeof(float), cudaMemcpyDeviceToDevice, st);
                cudaMemcpyAsync(s->jV[L] + j*kvd, s->dVc[L] + slot*kvd, (size_t)kvd*sizeof(float), cudaMemcpyDeviceToDevice, st);
            }
            float *dstK = s->dKc[L] + slot * kvd;
            float *dstV = s->dVc[L] + slot * kvd;
            if (zero) {
                cudaMemsetAsync(dstK, 0, (size_t)kvd * sizeof(float), st);
                cudaMemsetAsync(dstV, 0, (size_t)kvd * sizeof(float), st);
            } else {
                cudaMemcpyAsync(dstK, (const float *)(dK + off) + (size_t)p * kvd, (size_t)kvd * sizeof(float), cudaMemcpyDeviceToDevice, st);
                cudaMemcpyAsync(dstV, (const float *)(dV + off) + (size_t)p * kvd, (size_t)kvd * sizeof(float), cudaMemcpyDeviceToDevice, st);
            }
        }
    }
    cudaStreamSynchronize(st);
    cudaFree(dK); cudaFree(dV); sp_xbar_manifest_free(&mf);
    s->dpos_host += npos;
    if (cudaMemcpy(s->dpos, &s->dpos_host, sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) { sp_set_error("kv_replay: dpos H2D"); return -1; }
    fprintf(stderr, "    [xbar-#222] gemma4_kv_replay: %s episode into resident cache [%d,%d) alpha=%.3f eff_mass(a*npos)=%.1f%s\n",
            zero ? "ZEROED" : "intact", base, base + npos, replay_alpha, replay_alpha * (float)npos, zero ? " (reject control)" : "");
    return 0;
}

/* B3-v10 ABLATION GATE (the Thermodynamic Knockout): content-blind k specific positions of an
 * injected episode by memset-zeroing their K/V rows (the replay 'zero' mechanism, scoped to k
 * rows at base+pos[i]). Used to measure CAUSAL episodic dependency — re-score the payload with
 * the source tokens knocked out; a true match collapses, a parametric/empty match is unmoved.
 * No journaling: gemma4_kv_replay already journaled the pre-replay slot values, so the eventual
 * gemma4_kv_rewind(npos) restores these rows bit-exactly (ablation is transient/non-destructive). */
extern "C" int gemma4_kv_ablate_rows(sp_g4_kv *s, int base, const int *pos, int k) {
    if (!s || !pos || k <= 0) return 0;   /* empty mask (e.g. no payload-token match) = no-op */
    cudaStream_t st = g_w.stream;
    for (int L = 0; L < s->kvfs && L < s->NL; L++) {
        const int global = ((L % s->period) == s->period - 1);
        const int kvd = (global ? s->g_nkv : s->s_nkv) * (global ? s->g_hd : s->s_hd);
        const int ring = (s->ring_W > 0 && !global);
        for (int i = 0; i < k; i++) {
            const int p = base + pos[i];
            if (p < 0 || p >= s->Pmax) continue;
            const size_t slot = ring ? (size_t)(p % s->ring_W) : (size_t)p;
            cudaMemsetAsync(s->dKc[L] + slot * kvd, 0, (size_t)kvd * sizeof(float), st);
            cudaMemsetAsync(s->dVc[L] + slot * kvd, 0, (size_t)kvd * sizeof(float), st);
        }
    }
    cudaStreamSynchronize(st);
    return 0;
}

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

/* CONTRACT-CHAT-FULLSTACK B3 (AUTONOMOUS RECALL) — read the GLOBAL-owner K rows
 * out of the resident cache for the daemon's C2 query-signature computation.
 *
 * The C2 256-bit LSH signature (tools/curator/discrete_resolve.py) is the
 * sign of the ±1 projection (SP_ARM_PROJ_SEED) of the per-position GLOBAL-owner
 * K, meaned over the global layers {period-1, 2*period-1, ...} (period-6 ⇒
 * {5,11,...,47}) and the prefilled positions. The episode registry sigs were
 * built off the on-disk ep.k dumps; to recall AUTONOMOUSLY the daemon must
 * compute the SAME signature for the live query, which means reading the live
 * cache's global-layer K back to host. gemma4_kv_snapshot reads ALL layers into
 * caller-sized per-layer buffers; this is the focused B3 read — just the global
 * owners, just [0,npos), packed contiguously as [n_global][npos][g_kvd] row-major
 * in GLOBAL-LAYER ASCENDING order (the same order the curator iterates: L = 5,
 * 11, ... < NL). Globals are full-cache (slot==pos, no ring) so the copy is a
 * straight per-position memcpy. Returns the number of global layers written
 * (>0) on success, -1 on bad args / OOB / CUDA error. Null floor: dead code
 * unless called; the cache is byte-untouched (read-only D2H). */
extern "C" int gemma4_kv_read_global_k(const sp_g4_kv *s, float *out, int npos) {
    if (!s || !out || npos <= 0) { sp_set_error("gemma4_kv_read_global_k: bad args"); return -1; }
    if (npos > s->dpos_host) { sp_set_error("gemma4_kv_read_global_k: npos exceeds dpos"); return -1; }
    cudaStreamSynchronize(g_w.stream);
    const int g_kvd = s->g_nkv * s->g_hd;
    int gi = 0;
    for (int L = 0; L < s->kvfs && L < s->NL; L++) {
        const int global = ((L % s->period) == s->period - 1);
        if (!global) continue;
        /* globals are full-cache: slot==pos, contiguous [pos*g_kvd]. Copy [0,npos). */
        float *dst = out + (size_t)gi * npos * g_kvd;
        if (cudaMemcpy(dst, s->dKc[L], (size_t)npos * g_kvd * sizeof(float),
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
            sp_set_error("gemma4_kv_read_global_k: D2H"); return -1;
        }
        gi++;
    }
    return gi;
}

/* CONTRACT-CHAT-FULLSTACK B3-v2 — read the live query's last-token GLOBAL-layer Q.
 *
 * The B3-v1 selector (a position-meaned global-K centroid → 256-bit LSH → Hamming)
 * did not separate a short QUESTION from its answer PASSAGE (right episode argmax
 * only 1/5; episodes mutually agree ~200/256 = structural common-mode). v2 selects
 * by the model's NATIVE attention relevance instead: score each registry episode by
 * q·K, where q is THIS query's last-token query on the global owners and K is the
 * episode's stored global-K (ep.k). That is literally what attention computes —
 * "does this query attend to this memory?" — and the project already proved q·K
 * retrieves the right content (§3q LSH router, C-c NIAH).
 *
 * This verb runs ONE non-committing forward of `token` at the live dpos through the
 * SAME resident g4_kv_step the chat decode uses (so the query's representation is on
 * the identical path — the parity rule the v1 writeup established), capturing the
 * post-RoPE last-token query on each global owner layer into `out`, packed
 * [n_global][g_nh*g_hd] row-major (global layers ascending). dpos is then ROLLED
 * BACK so the cache state is unchanged for the caller's subsequent replay + real
 * decode_step(last): the K/V this step wrote at slot dpos for `token` is byte-
 * identical to what decode_step(last) rewrites at the same position. Returns the
 * number of global layers written (>0) on success, -1 on error.
 *
 * RING SAFETY: the verb reads only the GLOBAL owners' Q (globals are full-cache,
 * slot==pos). The peek step DOES write the SWA owners' ring slot pos%Wring (and a
 * journal pre-save entry) at the live dpos; after rollback the real decode_step at
 * the SAME dpos overwrites the SAME ring slot + journal slot identically. The journal
 * is REWIND-ONLY (forward chat never commits/rewinds mid-turn — the B2 ring-fix), so
 * the transient double-save is never read. Off the recall path = dead code (the cache
 * is byte-untouched; null floor). */
extern "C" int gemma4_kv_read_global_q(sp_g4_kv *s, int32_t token, float *out) {
    if (!s || !out) { sp_set_error("gemma4_kv_read_global_q: bad args"); return -1; }
    if (s->dpos_host + 1 >= s->Pmax) { sp_set_error("gemma4_kv_read_global_q: exceeds Pmax"); return -1; }
    cudaStream_t st = g_w.stream;
    const int n_global = s->NL / s->period;       /* period-6 over 48 ⇒ 8 globals */
    const int qd_g = s->g_nh * s->g_hd;
    if (!s->qcap_host) {
        if (cudaMallocHost(&s->qcap_host, (size_t)n_global * qd_g * sizeof(float)) != cudaSuccess) {
            s->qcap_host = NULL; sp_set_error("gemma4_kv_read_global_q: pinned host alloc"); return -1;
        }
    }
    /* write the input token at the live device position (the step embeds dseq[*dpos]). */
    if (cudaMemcpyAsync(s->dseq + s->dpos_host, &token, sizeof(int),
                        cudaMemcpyHostToDevice, st) != cudaSuccess) {
        sp_set_error("gemma4_kv_read_global_q: token H2D"); return -1; }
    /* save the ring-journal counters so the peek's (rewind-only) journal bookkeeping
     * is fully reverted — the real decode_step then runs as if the peek never happened. */
    const int save_commit = s->commit_pos, save_jcur = s->jcur;
    /* arm the one-shot global-Q capture and run a no-head step on the per-step path. */
    s->qcap_active = 1; s->qcap_gi = 0;
    if (g4_kv_step(s, /*do_head=*/0)) { s->qcap_active = 0; return -1; }
    s->commit_pos = save_commit; s->jcur = save_jcur;
    /* roll the device position back to dpos_host (the host counter was NOT advanced),
     * so the next replay/decode sees the cache exactly as before this peek. */
    if (cudaMemcpy(s->dpos, &s->dpos_host, sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
        sp_set_error("gemma4_kv_read_global_q: dpos restore H2D"); return -1; }
    cudaError_t e = cudaStreamSynchronize(st);
    if (e != cudaSuccess) return fail_cuda(e, "gemma4_kv_read_global_q sync");
    const int gi = s->qcap_gi;
    memcpy(out, s->qcap_host, (size_t)gi * qd_g * sizeof(float));
    return gi;
}

extern "C" void gemma4_kv_close(sp_g4_kv *s) {
    if (!s) return;
    if (s->qcap_host) cudaFreeHost(s->qcap_host);   /* B3-v2 query-capture pinned buf */
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
    if (s->gexec) cudaGraphExecDestroy(s->gexec);   /* A1: graph decode teardown */
    if (s->gcap)  cudaGraphDestroy(s->gcap);
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

/* n_tok=1 fast path: cublasSgemv (GEMV-optimized) instead of Sgemm-with-m=1 (cublas picks a slow
 * generic kernel for the m=1 case — ~2.8ms vs ~0.2ms each; the Latent Interceptor draft body had
 * ~33 of these = the measured 93ms). dY[out] = dW[in x out, col-major lda=in]^T @ dX[in]. */
static int gemv_t(cublasHandle_t h, const float *dW, const float *dX, float *dY, int in, int out) {
    const float a = 1.0f, b = 0.0f;
    CB(cublasSgemv(h, CUBLAS_OP_T, in, out, &a, dW, in, dX, 1, &b, dY, 1), "cublasSgemv");
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

/* ════════════════════ N5a: DiffusionGemma MoE FFN (arch_id 9) ════════════════════
 *
 * Additive, self-contained CUDA MoE forward for the DiffusionGemma backbone
 * (SP_ARCH_DIFFUSION_GEMMA). Mirrors the qwen36.c moe_ffn arithmetic oracle EXACTLY
 * (router prep -> f32 logit GEMV -> full softmax -> top-NU by prob -> renorm ->
 * per-expert GeGLU GELU-tanh -> weighted accumulate) so the CPU<->CUDA parity gate
 * (G_N5A_MOE_PARITY) holds bit-tight on a pure-f32 fixture.
 *
 * This entry takes RAW f32 host pointers (no DevTensor / .sp-model dependence) so the
 * standalone parity test can drive it with deterministic synthetic weights free of
 * OK_Q4B quant noise. It touches NO existing forward path (gemma3/gemma4/qwen3 stay
 * byte-identical) — purely a new function + new kernels below.
 *
 * Spec (verified vs _diffgemma_reference/gemma4-common.h gemma4_build_ffn_moe):
 *   router input prep: tmp = rms_norm(hidden,eps); tmp *= 1/sqrt(E); tmp[i] *= scale[i]
 *   gate_up_exps[e] : [FF*2, E] row-major; gate = rows[0:FF], up = rows[FF:2FF]
 *   h[i] = gelu_tanh(gate[i]) * up[i]
 *   down_exps[e]    : [E, FF] row-major
 *   y[i] += wt[k] * (down_exps[e] @ h)[i]
 */

/* device GELU-tanh (the exact k_gelu_mul / g4_gelu approximation, fp32). */
__device__ __forceinline__ float dg_gelu_tanh(float x) {
    const float k = 0.7978845608028654f;
    float th = tanhf(k * (x + 0.044715f * x * x * x));
    return 0.5f * x * (1.0f + th);
}

/* one-block f32 RMSNorm of a single vector: out[i] = x[i]/sqrt(mean(x^2)+eps).
 * (No weight gain — the router prep in moe_ffn norms the raw hidden; the gain is
 * folded into the model's attn/ffn_norm upstream, not part of THIS standalone op.) */
__global__ void dg_k_rmsnorm_vec(const float *x, int E, float eps, float *out) {
    extern __shared__ float sh[];
    float acc = 0.0f;
    for (int i = threadIdx.x; i < E; i += blockDim.x) acc += x[i] * x[i];
    sh[threadIdx.x] = acc; __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    float inv = rsqrtf(sh[0] / (float)E + eps);
    for (int i = threadIdx.x; i < E; i += blockDim.x) out[i] = x[i] * inv;
}

/* tmp[i] *= router_scale * scale_vec[i]   (router_scale = 1/sqrt(E)). */
__global__ void dg_k_router_scale(float *tmp, const float *scale_vec, float rscale, int E) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < E) tmp[i] *= rscale * scale_vec[i];
}

/* row-major f32 GEMV: y[o] = dot(W_row_o[0:in], x)  for o in [0,out).
 * one warp per output row (mirrors the dp4a kernel's row mapping, f32 arithmetic). */
__global__ void dg_k_gemv_f32(const float *W, const float *x, float *y, int in, int out) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int o = blockIdx.x * (blockDim.x >> 5) + warp;
    if (o >= out) return;
    const float *wr = W + (size_t)o * in;
    float acc = 0.0f;
    for (int i = lane; i < in; i += 32) acc += wr[i] * x[i];
    for (int s = 16; s > 0; s >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, s);
    if (lane == 0) y[o] = acc;
}

/* GeGLU: h[i] = gelu_tanh(gate[i]) * up[i], gate = gu[0:FF], up = gu[FF:2FF]. */
__global__ void dg_k_geglu(const float *gu, float *h, int FF) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < FF) h[i] = dg_gelu_tanh(gu[i]) * gu[FF + i];
}

/* batched GeGLU: gu is [cnt columns x gu_rows] row-major (gu_rows = FF*2); within column
 * col, gate = gu[col*gu_rows + i], up = gu[col*gu_rows + FF + i]; h is [cnt x FF] row-major.
 * idx over [0, FF*cnt): col = idx/FF, i = idx%FF. (N1b MoE expert batching.) */
__global__ void dg_k_geglu_batched(const float *gu, float *h, int FF, int gu_rows, int cnt) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)FF * cnt) return;
    int col = (int)(idx / FF), i = (int)(idx % FF);
    const float *g = gu + (size_t)col * gu_rows;
    h[(size_t)col * FF + i] = dg_gelu_tanh(g[i]) * g[FF + i];
}

/* yo[i] += w * de[i]  (weighted expert accumulate into the output vector). */
__global__ void dg_k_axpy(float *yo, const float *de, float w, int E) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < E) yo[i] += w * de[i];
}

/* N5a entry: single-token MoE FFN on f32 host weights. Writes out[E].
 *   gate_inp      : [NE, E] f32 router weights (row o = expert o's logit row)
 *   gate_inp_scale: [E] f32 elementwise router-input scale sidecar
 *   gate_up_exps  : [NE, FF*2, E] f32 (expert e at offset e*FF*2*E)
 *   down_exps     : [NE, E, FF]  f32 (expert e at offset e*E*FF)
 *   hidden        : [E] f32 input activation (the SAME hidden the router norms)
 *   sel_out       : optional [NU] int, receives the selected expert indices (or NULL)
 * Returns 0 on success. */
extern "C" int gemma4_moe_ffn_cuda(int E, int NE, int NU, int FF, float eps,
                                   const float *gate_inp, const float *gate_inp_scale,
                                   const float *gate_up_exps, const float *down_exps,
                                   const float *hidden, float *out, int *sel_out) {
    int rc = 1;
    cudaStream_t st = 0;
    float *d_hidden=NULL, *d_tmp=NULL, *d_scale=NULL, *d_gi=NULL, *d_logits=NULL;
    float *d_gu=NULL, *d_dn=NULL, *d_x=NULL, *d_guout=NULL, *d_h=NULL, *d_de=NULL, *d_yo=NULL;
    float *h_logits = (float *)malloc((size_t)NE * sizeof(float));
    if (!h_logits) { sp_set_error("moe: host OOM"); return 1; }

    #define DGM(p, n) do { if (cudaMalloc(&(p), (size_t)(n)*sizeof(float)) != cudaSuccess) { \
        sp_set_error("moe: cudaMalloc"); goto done; } } while (0)
    DGM(d_hidden, E); DGM(d_tmp, E); DGM(d_scale, E); DGM(d_gi, (size_t)NE*E);
    DGM(d_logits, NE);
    DGM(d_gu, (size_t)FF*2*E); DGM(d_dn, (size_t)E*FF);
    DGM(d_x, E); DGM(d_guout, FF*2); DGM(d_h, FF); DGM(d_de, E); DGM(d_yo, E);

    if (cudaMemcpy(d_hidden, hidden, (size_t)E*sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_scale, gate_inp_scale, (size_t)E*sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_gi, gate_inp, (size_t)NE*E*sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        sp_set_error("moe: H2D copy"); goto done;
    }

    /* 1. expert input x = rms_norm(hidden) (the ffn-branch normed hidden the experts
     *    consume); router input tmp = x then *= 1/sqrt(E) then *= scale[] (router-only). */
    {
        int bd = 256; while (bd > E && bd > 32) bd >>= 1;
        dg_k_rmsnorm_vec<<<1, bd, (size_t)bd*sizeof(float), st>>>(d_hidden, E, eps, d_x);
        cudaMemcpyAsync(d_tmp, d_x, (size_t)E*sizeof(float), cudaMemcpyDeviceToDevice, st);
        dg_k_router_scale<<<(unsigned)((E+255)/256), 256, 0, st>>>(d_tmp, d_scale, 1.0f/sqrtf((float)E), E);
    }
    /* 2. router GEMV: logits[o] = dot(gate_inp_row[o], tmp) */
    {
        unsigned blocks = ((unsigned)NE + 7u) / 8u;   /* 8 warps/block */
        dg_k_gemv_f32<<<blocks, 256, 0, st>>>(d_gi, d_tmp, d_logits, E, NE);
    }
    if (cudaMemcpy(h_logits, d_logits, (size_t)NE*sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        sp_set_error("moe: logits D2H"); goto done;
    }
    /* 3. softmax over all NE, then top-NU by prob, then renorm (qwen36.c moe_ffn) */
    {
        float mx = h_logits[0];
        for (int i = 1; i < NE; i++) if (h_logits[i] > mx) mx = h_logits[i];
        double se = 0.0;
        for (int i = 0; i < NE; i++) { h_logits[i] = expf(h_logits[i] - mx); se += h_logits[i]; }
        for (int i = 0; i < NE; i++) h_logits[i] = (float)(h_logits[i] / se);
        char *used = (char *)calloc((size_t)NE, 1);
        int *idx = (int *)malloc((size_t)NU*sizeof(int));
        float *wt = (float *)malloc((size_t)NU*sizeof(float));
        if (!used || !idx || !wt) { free(used); free(idx); free(wt); sp_set_error("moe: top-k OOM"); goto done; }
        float wsum = 0.0f;
        for (int k = 0; k < NU; k++) {
            int best = -1; float bv = -1.0f;
            for (int i = 0; i < NE; i++) if (!used[i] && h_logits[i] > bv) { bv = h_logits[i]; best = i; }
            used[best] = 1; idx[k] = best; wt[k] = bv; wsum += bv;
        }
        for (int k = 0; k < NU; k++) wt[k] = (wt[k] / wsum) * 1.0f;   /* expert_weights_scale = 1.0 */
        if (sel_out) for (int k = 0; k < NU; k++) sel_out[k] = idx[k];

        /* 4. expert dispatch + weighted accumulate */
        cudaMemset(d_yo, 0, (size_t)E*sizeof(float));
        unsigned gu_blocks = ((unsigned)(FF*2) + 7u) / 8u;
        unsigned dn_blocks = ((unsigned)E + 7u) / 8u;
        for (int k = 0; k < NU; k++) {
            int e = idx[k];
            const float *gu_e = gate_up_exps + (size_t)e * (size_t)(FF*2) * E;
            const float *dn_e = down_exps    + (size_t)e * (size_t)E * FF;
            if (cudaMemcpy(d_gu, gu_e, (size_t)(FF*2)*E*sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
                cudaMemcpy(d_dn, dn_e, (size_t)E*FF*sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
                free(used); free(idx); free(wt); sp_set_error("moe: expert H2D"); goto done;
            }
            /* gate_up = gate_up_exps[e] @ x, x = rms_norm(hidden) (router scaling is
             * router-only; the experts consume the unscaled normed hidden). */
            dg_k_gemv_f32<<<gu_blocks, 256, 0, st>>>(d_gu, d_x, d_guout, E, FF*2);
            dg_k_geglu<<<(unsigned)((FF+255)/256), 256, 0, st>>>(d_guout, d_h, FF);
            dg_k_gemv_f32<<<dn_blocks, 256, 0, st>>>(d_dn, d_h, d_de, FF, E);
            dg_k_axpy<<<(unsigned)((E+255)/256), 256, 0, st>>>(d_yo, d_de, wt[k], E);
        }
        free(used); free(idx); free(wt);
    }
    if (cudaMemcpy(out, d_yo, (size_t)E*sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        sp_set_error("moe: out D2H"); goto done;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) { sp_set_error("moe: sync"); goto done; }
    rc = 0;
done:
    free(h_logits);
    cudaFree(d_hidden); cudaFree(d_tmp); cudaFree(d_scale); cudaFree(d_gi); cudaFree(d_logits);
    cudaFree(d_gu); cudaFree(d_dn); cudaFree(d_x); cudaFree(d_guout); cudaFree(d_h);
    cudaFree(d_de); cudaFree(d_yo);
    #undef DGM
    return rc;
}

/* ════════════════ N5a-packed: DiffusionGemma MoE FFN on REAL OK_Q4B experts ════════════════
 *
 * Same MoE algorithm as gemma4_moe_ffn_cuda (router prep -> f32 logit GEMV -> full
 * softmax -> top-NU by prob -> renorm -> per-expert GeGLU -> weighted accumulate),
 * but the THREE expert GEMVs (gate, up — fused as gate_up [FF*2,E] — and down
 * [E,FF]) run through the existing OK_Q4B dp4a path (k_quant_act_int8 +
 * k_gemv_q4b_dp4a_v2) instead of f32 GEMV. The router stays f32 (matches the CPU
 * oracle, which never quantizes the router). Additive — a NEW function; the f32
 * gemma4_moe_ffn_cuda and every dense forward stay byte-identical.
 *
 * The experts are passed as the REAL packed OK_Q4B arena layout (the loader's
 * build_packed_q4b format): per-expert nibble-packed codes (2 codes/byte, low
 * nibble even idx, sign-ext (n^8)-8) + per-32-block f16 bscale [rows*bs_nblk].
 * Expert e of a fused rank-3 tensor occupies rows [e*rows_per_expert, ...); the
 * caller passes the FULL codes/bscale arrays + the per-expert row stride and the
 * code routes into expert e's slice (codes offset = e*rows_per_expert*nib_cols,
 * bscale offset = e*rows_per_expert*bs_nblk).
 *
 * Parity is ~1e-3 not bit-exact: the CPU oracle dequants the SAME OK_Q4B codes to
 * f32 and does an f32 dot, while this path quantizes the activation to int8
 * (per-16-block, qmax 127 — k_quant_act_int8) and accumulates via integer dp4a.
 * The weight codes + per-32-block scale are IDENTICAL on both sides (so the weight
 * arithmetic is exact); the ~1e-3 deflection is solely the int8 activation-quant +
 * integer reduction order, exactly as the dense dp4a decode path carries vs its f32
 * twin.
 *
 *   gu_codes  : [NE*FF*2 rows] nibble-packed Q4B codes for gate_up (in=E)
 *   gu_bscale : [NE*FF*2 * gu_bsnblk] f16 per-32-block scales (gu_bsnblk = E/32)
 *   dn_codes  : [NE*E rows]    nibble-packed Q4B codes for down    (in=FF)
 *   dn_bscale : [NE*E * dn_bsnblk] f16 (dn_bsnblk = FF/32)
 * Returns 0 on success.  E and FF must be multiples of 32 (Q4B dp4a precondition). */
extern "C" int gemma4_moe_ffn_q4b_cuda(int E, int NE, int NU, int FF, float eps,
                                       const float *gate_inp, const float *gate_inp_scale,
                                       const unsigned char *gu_codes, const unsigned short *gu_bscale,
                                       const unsigned char *dn_codes, const unsigned short *dn_bscale,
                                       const float *hidden, float *out, int *sel_out) {
    int rc = 1;
    cudaStream_t st = 0;
    if ((E & 31) || (FF & 31)) { sp_set_error("moe_q4b: E,FF must be %%32"); return 1; }
    const int gu_rows = FF * 2;            /* rows per expert in the fused gate_up tensor */
    const int dn_rows = E;                 /* rows per expert in the down tensor */
    const int gu_bsnblk = E  >> 5;         /* per-32 blocks per gate_up row (in=E)  */
    const int dn_bsnblk = FF >> 5;         /* per-32 blocks per down    row (in=FF) */
    const size_t gu_nibcols = (size_t)((E  + 1) / 2);   /* bytes per gate_up row */
    const size_t dn_nibcols = (size_t)((FF + 1) / 2);   /* bytes per down    row */

    float *h_logits = (float *)malloc((size_t)NE * sizeof(float));
    /* device buffers: f32 router lane (same as the f32 entry) + the packed expert
     * lane (whole gate_up/down codes+bscale resident, per-expert row_off, act-quant
     * scratch, GEMV outputs). */
    float *d_hidden=NULL, *d_tmp=NULL, *d_scale=NULL, *d_gi=NULL, *d_logits=NULL;
    float *d_x=NULL, *d_guout=NULL, *d_h=NULL, *d_de=NULL, *d_yo=NULL;
    unsigned char *d_gu_codes=NULL, *d_dn_codes=NULL;
    unsigned short *d_gu_bscale=NULL, *d_dn_bscale=NULL;
    unsigned long long *d_gu_roff=NULL, *d_dn_roff=NULL;
    signed char *d_qx=NULL; float *d_sx=NULL;
    if (!h_logits) { sp_set_error("moe_q4b: host OOM"); return 1; }

    #define QGM(p, n) do { if (cudaMalloc(&(p), (size_t)(n)*sizeof(float)) != cudaSuccess) { \
        sp_set_error("moe_q4b: cudaMalloc"); goto qdone; } } while (0)
    QGM(d_hidden, E); QGM(d_tmp, E); QGM(d_scale, E); QGM(d_gi, (size_t)NE*E);
    QGM(d_logits, NE);
    QGM(d_x, E); QGM(d_guout, FF*2); QGM(d_h, FF); QGM(d_de, E); QGM(d_yo, E);

    /* the full packed expert arrays, resident (single-token gate — sizes are small) */
    {
        size_t gu_codes_bytes = (size_t)NE * gu_rows * gu_nibcols;
        size_t dn_codes_bytes = (size_t)NE * dn_rows * dn_nibcols;
        size_t gu_bs_n = (size_t)NE * gu_rows * gu_bsnblk;
        size_t dn_bs_n = (size_t)NE * dn_rows * dn_bsnblk;
        if (cudaMalloc(&d_gu_codes, gu_codes_bytes) != cudaSuccess ||
            cudaMalloc(&d_dn_codes, dn_codes_bytes) != cudaSuccess ||
            cudaMalloc(&d_gu_bscale, gu_bs_n*sizeof(unsigned short)) != cudaSuccess ||
            cudaMalloc(&d_dn_bscale, dn_bs_n*sizeof(unsigned short)) != cudaSuccess ||
            cudaMalloc(&d_gu_roff, (size_t)gu_rows*sizeof(unsigned long long)) != cudaSuccess ||
            cudaMalloc(&d_dn_roff, (size_t)dn_rows*sizeof(unsigned long long)) != cudaSuccess) {
            sp_set_error("moe_q4b: packed cudaMalloc"); goto qdone;
        }
        /* act-quant scratch: int8 codes + per-16-block scales, padded to max in */
        int max_in = E > FF ? E : FF;
        int max_npad = (max_in + 31) & ~31;
        if (cudaMalloc(&d_qx, (size_t)max_npad) != cudaSuccess ||
            cudaMalloc(&d_sx, (size_t)(max_npad>>4)*sizeof(float)) != cudaSuccess) {
            sp_set_error("moe_q4b: actq cudaMalloc"); goto qdone;
        }
        if (cudaMemcpy(d_gu_codes, gu_codes, gu_codes_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(d_dn_codes, dn_codes, dn_codes_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(d_gu_bscale, gu_bscale, gu_bs_n*sizeof(unsigned short), cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(d_dn_bscale, dn_bscale, dn_bs_n*sizeof(unsigned short), cudaMemcpyHostToDevice) != cudaSuccess) {
            sp_set_error("moe_q4b: packed H2D"); goto qdone;
        }
        /* per-expert row_off: byte offset of row r WITHIN the expert slice (the codes
         * base pointer is advanced to the expert start, so row_off restarts at 0). */
        unsigned long long *h_gu_roff = (unsigned long long *)malloc((size_t)gu_rows*sizeof(unsigned long long));
        unsigned long long *h_dn_roff = (unsigned long long *)malloc((size_t)dn_rows*sizeof(unsigned long long));
        if (!h_gu_roff || !h_dn_roff) { free(h_gu_roff); free(h_dn_roff); sp_set_error("moe_q4b: roff OOM"); goto qdone; }
        for (int r = 0; r < gu_rows; r++) h_gu_roff[r] = (unsigned long long)((size_t)r * gu_nibcols);
        for (int r = 0; r < dn_rows; r++) h_dn_roff[r] = (unsigned long long)((size_t)r * dn_nibcols);
        cudaMemcpy(d_gu_roff, h_gu_roff, (size_t)gu_rows*sizeof(unsigned long long), cudaMemcpyHostToDevice);
        cudaMemcpy(d_dn_roff, h_dn_roff, (size_t)dn_rows*sizeof(unsigned long long), cudaMemcpyHostToDevice);
        free(h_gu_roff); free(h_dn_roff);
    }

    if (cudaMemcpy(d_hidden, hidden, (size_t)E*sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_scale, gate_inp_scale, (size_t)E*sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_gi, gate_inp, (size_t)NE*E*sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        sp_set_error("moe_q4b: H2D copy"); goto qdone;
    }

    /* 1. expert input x = rms_norm(hidden); router input tmp = x * 1/sqrt(E) * scale[] */
    {
        int bd = 256; while (bd > E && bd > 32) bd >>= 1;
        dg_k_rmsnorm_vec<<<1, bd, (size_t)bd*sizeof(float), st>>>(d_hidden, E, eps, d_x);
        cudaMemcpyAsync(d_tmp, d_x, (size_t)E*sizeof(float), cudaMemcpyDeviceToDevice, st);
        dg_k_router_scale<<<(unsigned)((E+255)/256), 256, 0, st>>>(d_tmp, d_scale, 1.0f/sqrtf((float)E), E);
    }
    /* 2. router GEMV (f32, exactly the f32 entry) */
    {
        unsigned blocks = ((unsigned)NE + 7u) / 8u;
        dg_k_gemv_f32<<<blocks, 256, 0, st>>>(d_gi, d_tmp, d_logits, E, NE);
    }
    if (cudaMemcpy(h_logits, d_logits, (size_t)NE*sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        sp_set_error("moe_q4b: logits D2H"); goto qdone;
    }
    /* 3. softmax + top-NU + renorm (identical to the f32 entry / CPU oracle) */
    {
        float mx = h_logits[0];
        for (int i = 1; i < NE; i++) if (h_logits[i] > mx) mx = h_logits[i];
        double se = 0.0;
        for (int i = 0; i < NE; i++) { h_logits[i] = expf(h_logits[i] - mx); se += h_logits[i]; }
        for (int i = 0; i < NE; i++) h_logits[i] = (float)(h_logits[i] / se);
        char *used = (char *)calloc((size_t)NE, 1);
        int *idx = (int *)malloc((size_t)NU*sizeof(int));
        float *wt = (float *)malloc((size_t)NU*sizeof(float));
        if (!used || !idx || !wt) { free(used); free(idx); free(wt); sp_set_error("moe_q4b: top-k OOM"); goto qdone; }
        float wsum = 0.0f;
        for (int k = 0; k < NU; k++) {
            int best = -1; float bv = -1.0f;
            for (int i = 0; i < NE; i++) if (!used[i] && h_logits[i] > bv) { bv = h_logits[i]; best = i; }
            used[best] = 1; idx[k] = best; wt[k] = bv; wsum += bv;
        }
        for (int k = 0; k < NU; k++) wt[k] = (wt[k] / wsum) * 1.0f;
        if (sel_out) for (int k = 0; k < NU; k++) sel_out[k] = idx[k];

        /* 4. per-expert dp4a dispatch (gate_up [FF*2,E] then down [E,FF]) + accumulate */
        cudaMemset(d_yo, 0, (size_t)E*sizeof(float));
        unsigned gu_blocks = ((unsigned)(FF*2) + 7u) / 8u;   /* 8 warps(rows)/block */
        unsigned dn_blocks = ((unsigned)E + 7u) / 8u;
        for (int k = 0; k < NU; k++) {
            int e = idx[k];
            /* expert e's codes/bscale base inside the fused rank-3 tensors */
            const unsigned char  *gu_c  = d_gu_codes  + (size_t)e * gu_rows * gu_nibcols;
            const unsigned short *gu_bs = d_gu_bscale + (size_t)e * gu_rows * gu_bsnblk;
            const unsigned char  *dn_c  = d_dn_codes  + (size_t)e * dn_rows * dn_nibcols;
            const unsigned short *dn_bs = d_dn_bscale + (size_t)e * dn_rows * dn_bsnblk;
            /* gate_up = Q4B(gate_up_exps[e]) @ x  (in=E, out=FF*2), dp4a + act int8 */
            {
                int npad = (E + 31) & ~31;
                k_quant_act_int8<<<1, 256, 0, st>>>(d_x, E, npad, d_qx, d_sx);
                k_gemv_q4b_dp4a_v2<<<gu_blocks, 256, 0, st>>>(
                    gu_c, d_gu_roff, gu_bs, gu_bsnblk, E, d_qx, d_sx, d_guout, FF*2);
            }
            dg_k_geglu<<<(unsigned)((FF+255)/256), 256, 0, st>>>(d_guout, d_h, FF);
            /* de = Q4B(down_exps[e]) @ h  (in=FF, out=E), dp4a + act int8 */
            {
                int npad = (FF + 31) & ~31;
                k_quant_act_int8<<<1, 256, 0, st>>>(d_h, FF, npad, d_qx, d_sx);
                k_gemv_q4b_dp4a_v2<<<dn_blocks, 256, 0, st>>>(
                    dn_c, d_dn_roff, dn_bs, dn_bsnblk, FF, d_qx, d_sx, d_de, E);
            }
            dg_k_axpy<<<(unsigned)((E+255)/256), 256, 0, st>>>(d_yo, d_de, wt[k], E);
        }
        free(used); free(idx); free(wt);
    }
    if (cudaMemcpy(out, d_yo, (size_t)E*sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        sp_set_error("moe_q4b: out D2H"); goto qdone;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) { sp_set_error("moe_q4b: sync"); goto qdone; }
    rc = 0;
qdone:
    free(h_logits);
    cudaFree(d_hidden); cudaFree(d_tmp); cudaFree(d_scale); cudaFree(d_gi); cudaFree(d_logits);
    cudaFree(d_x); cudaFree(d_guout); cudaFree(d_h); cudaFree(d_de); cudaFree(d_yo);
    cudaFree(d_gu_codes); cudaFree(d_dn_codes); cudaFree(d_gu_bscale); cudaFree(d_dn_bscale);
    cudaFree(d_gu_roff); cudaFree(d_dn_roff); cudaFree(d_qx); cudaFree(d_sx);
    #undef QGM
    return rc;
}

/* ════════════════════ N1b: DiffusionGemma region-aware FORWARD (arch_id 9) ════════════════════
 *
 * diffusion_gemma_forward_cuda — a single no-cache bidirectional pass over the
 * concatenated sequence [prompt | canvas] -> logits, exactly the UNIFIED graph of
 * the PR-24423 reference (_diffgemma_reference/diffusion-gemma.cpp, the !is_prefill
 * && !is_decode branch). Split point P = n_tok - canvas_length; canvas = the last
 * canvas_length positions. THREE things are region-aware (everything else is the
 * gemma4 backbone verbatim):
 *   1. embeddings : prompt rows = embed*sqrt(E); canvas rows = rmsnorm_noscale(embed*sqrt(E))
 *   2. attn mask  : prompt query -> causal over prompt only (SWA-clipped on sliding layers);
 *                   canvas query -> bidirectional over ALL prompt + ALL canvas (sliding layers:
 *                   last (n_swa-1) prompt positions + all canvas)
 *   3. layer scalar: prompt rows * enc_layer_output_scale[L]; canvas rows * layer_output_scale[L]
 * Self-conditioning is OFF (single step-0 forward), so the SC subgraph is skipped.
 *
 * Weights stream per-layer from the arena (dequant-to-f32 on the host, upload, cublas
 * GEMM, free) so VRAM stays O(one layer) regardless of the 13 GB model — the 26B MoE
 * does not fit resident on a 12 GB GPU. The MoE experts are gathered PER ROUTED EXPERT
 * (only the n_expert_used selected per token are dequanted), bounding both VRAM and
 * host work for the one-shot gate. f32 dequant matches the CPU oracle's "dequant then
 * f32 dot" path (no int8 act-quant deflection) — best for the parity gate.
 *
 * ADDITIVE: a new function + new kernels. NO existing forward (gemma3/gemma4/qwen3) is
 * touched; the dense path stays byte-identical. Dispatched on arch == SP_ARCH_DIFFUSION_GEMMA. */

/* Region-aware attention. One block per (query token t, query head h). Mirrors k_attn's
 * f32 math (same reduction order over the allowed key set) but the allowed key set is the
 * diffusion region mask, not a causal window:
 *   q_is_canvas = (t >= P)
 *     canvas query: global layer -> all keys; sliding -> keys k where (k>=P) || (k>=P-n_swa+1)
 *     prompt query: keys k<=t with k<P (causal over prompt) AND (sliding -> k > t-n_swa)  */
__global__ void k_attn_diffusion(const float *Q, const float *K, const float *V,
                                 int n_tok, int QD, int KVD, int HD, int group,
                                 float ascale, int P, int n_swa, int is_swa, float *AO) {
    extern __shared__ float sc[];
    int n_heads = QD / HD;
    int b = blockIdx.x, t = b / n_heads, h = b % n_heads, kvh = h / group;
    const float *qh = Q + (size_t)t * QD + (size_t)h * HD;
    const int q_is_canvas = (t >= P);
    const long long canvas_prompt_lo = (long long)P - (long long)n_swa + 1;

    /* scores over the allowed key set (-inf elsewhere). Predicate inlined (no extended
     * lambda); mirrors the reference fill(): see diffusion-gemma.cpp set_input. */
    for (int s = threadIdx.x; s < n_tok; s += blockDim.x) {
        const int k_is_canvas = (s >= P);
        bool allow;
        if (q_is_canvas) {
            allow = is_swa ? (k_is_canvas || ((long long)s >= canvas_prompt_lo)) : true;
        } else {
            /* prompt query: causal over earlier prompt, never canvas (+ SWA clip) */
            allow = (!k_is_canvas) && (s <= t);
            if (allow && is_swa && (long long)s <= (long long)t - (long long)n_swa) allow = false;
        }
        if (!allow) { sc[s] = -3.4e38f; continue; }
        const float *kh = K + (size_t)s * KVD + (size_t)kvh * HD;
        float acc = 0.0f;
        for (int i = 0; i < HD; i++) acc += qh[i] * kh[i];
        sc[s] = acc * ascale;
    }
    __syncthreads();

    __shared__ float g_sum;
    if (threadIdx.x == 0) {
        float m = -3.4e38f;
        for (int s = 0; s < n_tok; s++) if (sc[s] > m) m = sc[s];
        float sum = 0.0f;
        for (int s = 0; s < n_tok; s++) {
            if (sc[s] <= -3.0e38f) { sc[s] = 0.0f; continue; }
            float e = expf(sc[s] - m); sc[s] = e; sum += e;
        }
        g_sum = sum;
    }
    __syncthreads();

    float inv = 1.0f / g_sum;
    for (int i = threadIdx.x; i < HD; i += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < n_tok; s++)
            acc += sc[s] * V[(size_t)s * KVD + (size_t)kvh * HD + i];
        AO[(size_t)t * QD + (size_t)h * HD + i] = acc * inv;
    }
}

/* ── N6 PREFIX-KV Increment 1: canvas-only position-coupled kernels (compiled, not yet wired) ──
 * The fast path (steps 2..N of a denoise query) recomputes ONLY canvas rows [P,n_tok). These three
 * kernels are the originals with a position base so canvas rows land at their ABSOLUTE positions:
 *   k_rope_off / k_rope_freqs_off : t = pos0 + blockIdx.x/n_heads  (launch grid = C*n_heads, pos0=P)
 *   k_attn_diffusion_canvas       : t = q_base + blockIdx.x/n_heads (q_base=P); K/V = full Kst/Vst
 * Every other line is identical to k_rope/k_rope_freqs/k_attn_diffusion -> byte-identical on canvas
 * rows. base/Q/AO are the FULL [n_tok x stride] buffers; canvas data lives at rows [P,n_tok). */
__global__ void k_rope_off(float *base, int n_heads, int d, int rowstride, float rbase, int pos0) {
    int b = blockIdx.x, t = pos0 + b / n_heads, h = b % n_heads, i = threadIdx.x, half = d / 2;
    if (i < half) {
        float *v = base + (size_t)t * rowstride + (size_t)h * d;
        float freq = powf(rbase, -2.0f * (float)i / (float)d);
        if (d_bx_flag) { bx_rope_pair(&v[i], &v[i + half], bx_rope_theta(freq, (long long)t)); return; }
        float th = (float)t * freq, c = cosf(th), s = sinf(th);
        float a = v[i], bb = v[i + half];
        v[i] = a * c - bb * s;
        v[i + half] = a * s + bb * c;
    }
}
__global__ void k_rope_freqs_off(float *base, int n_heads, int d, int rowstride,
                                 float rbase, const float *ff, int pos0) {
    int b = blockIdx.x, t = pos0 + b / n_heads, h = b % n_heads, i = threadIdx.x, half = d / 2;
    if (i < half) {
        float *v = base + (size_t)t * rowstride + (size_t)h * d;
        float freq = powf(rbase, -2.0f * (float)i / (float)d) / ff[i];
        if (d_bx_flag) { bx_rope_pair(&v[i], &v[i + half], bx_rope_theta(freq, (long long)t)); return; }
        float th = (float)t * freq, c = cosf(th), s = sinf(th);
        float a = v[i], bb = v[i + half];
        v[i] = a * c - bb * s;
        v[i + half] = a * s + bb * c;
    }
}
/* k_attn_diffusion restricted to canvas queries at absolute positions [q_base, q_base+C). */
__global__ void k_attn_diffusion_canvas(const float *Q, const float *K, const float *V,
                                 int n_tok, int QD, int KVD, int HD, int group,
                                 float ascale, int P, int n_swa, int is_swa, int q_base, float *AO) {
    extern __shared__ float sc[];
    int n_heads = QD / HD;
    int b = blockIdx.x, t = q_base + b / n_heads, h = b % n_heads, kvh = h / group;
    const float *qh = Q + (size_t)t * QD + (size_t)h * HD;
    const int q_is_canvas = (t >= P);
    const long long canvas_prompt_lo = (long long)P - (long long)n_swa + 1;
    for (int s = threadIdx.x; s < n_tok; s += blockDim.x) {
        const int k_is_canvas = (s >= P);
        bool allow;
        if (q_is_canvas) {
            allow = is_swa ? (k_is_canvas || ((long long)s >= canvas_prompt_lo)) : true;
        } else {
            allow = (!k_is_canvas) && (s <= t);
            if (allow && is_swa && (long long)s <= (long long)t - (long long)n_swa) allow = false;
        }
        if (!allow) { sc[s] = -3.4e38f; continue; }
        const float *kh = K + (size_t)s * KVD + (size_t)kvh * HD;
        float acc = 0.0f;
        for (int i = 0; i < HD; i++) acc += qh[i] * kh[i];
        sc[s] = acc * ascale;
    }
    __syncthreads();
    __shared__ float g_sum;
    if (threadIdx.x == 0) {
        float m = -3.4e38f;
        for (int s = 0; s < n_tok; s++) if (sc[s] > m) m = sc[s];
        float sum = 0.0f;
        for (int s = 0; s < n_tok; s++) {
            if (sc[s] <= -3.0e38f) { sc[s] = 0.0f; continue; }
            float e = expf(sc[s] - m); sc[s] = e; sum += e;
        }
        g_sum = sum;
    }
    __syncthreads();
    float inv = 1.0f / g_sum;
    for (int i = threadIdx.x; i < HD; i += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < n_tok; s++)
            acc += sc[s] * V[(size_t)s * KVD + (size_t)kvh * HD + i];
        AO[(size_t)t * QD + (size_t)h * HD + i] = acc * inv;
    }
}

/* full-E weightless RMSNorm of canvas rows, in place: x[r] = x[r]/sqrt(mean(x[r]^2)+eps),
 * for r in [P, n_tok). One block per canvas row (grid = C, blockDim 256). */
__global__ void k_rmsnorm_noscale_rows(float *x, int E, float eps, int P) {
    int row = P + blockIdx.x;
    float *xr = x + (size_t)row * E;
    __shared__ double sh[256];
    double s = 0.0;
    for (int i = threadIdx.x; i < E; i += blockDim.x) { double v = xr[i]; s += v * v; }
    sh[threadIdx.x] = s; __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (threadIdx.x < o) sh[threadIdx.x] += sh[threadIdx.x + o];
        __syncthreads();
    }
    float scale = 1.0f / sqrtf((float)(sh[0] / (double)E) + eps);
    for (int i = threadIdx.x; i < E; i += blockDim.x) xr[i] = xr[i] * scale;
}

/* region scalar split: rows [0,P) *= encS[0], rows [P,n_tok) *= decS[0]. encS/decS are
 * device 1-float per-layer scalars. n = n_tok*E. */
__global__ void k_region_scale(float *x, int E, int P, int n_tok,
                               const float *encS, const float *decS) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)n_tok * E) return;
    int row = (int)(idx / E);
    x[idx] *= (row < P) ? encS[0] : decS[0];
}

/* ── N4a: the entropy-bound sample kernel (dg_sample_kernel) ──────────────────────
 * Port of the PR-24423 device sampler `diffusion_dense_sample_kernel`
 * (_diffgemma_reference/diffusion-sampling.cu) into our backend. PURE vocab-space
 * float math on the logits the forward already produced — NO weights, NO quant.
 *
 * One CUDA block per canvas position (row). Per row:
 *   1. parallel max -> argmax over v of (logit_v * inv_temp)
 *   2. parallel Z = sum_v exp(d_v), T = sum_v d_v*exp(d_v), with d_v = logit_v*inv_temp - max
 *      -> entropy = log(Z) - T/Z
 *   3. multinomial draw: first v (vocab order) with cumulative exp(d_v) >= r = u[row]*Z,
 *      via a 256-slice exclusive-scan so the serial walk is one slice not the whole vocab.
 * Argmax is exact; Z/entropy match a host reference to ~1e-4 (FP reduction order).
 * blockDim is fixed at 256 (the slice scheme assumes blockDim == 256 buffers). */
__global__ void dg_sample_kernel(const float * __restrict__ logits,
                                 const float * __restrict__ u,
                                 int   * __restrict__ argmax,
                                 float * __restrict__ entropy,
                                 int   * __restrict__ sampled,
                                 const int   n_vocab,
                                 const float inv_temp) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    __shared__ float s_val[256];
    __shared__ float s_sum[256];
    __shared__ int   s_idx[256];

    const float *row_logits = logits + (size_t)row * n_vocab;

    /* ── parallel max -> argmax ── */
    float local_max = -3.4e38f;
    int   local_idx = 0;
    for (int v = tid; v < n_vocab; v += blockDim.x) {
        const float x = row_logits[v] * inv_temp;
        if (x > local_max) { local_max = x; local_idx = v; }
    }
    s_val[tid] = local_max;
    s_idx[tid] = local_idx;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride && s_val[tid + stride] > s_val[tid]) {
            s_val[tid] = s_val[tid + stride];
            s_idx[tid] = s_idx[tid + stride];
        }
        __syncthreads();
    }
    const float max_l = s_val[0];
    const int   amax  = s_idx[0];
    __syncthreads();

    /* ── parallel Z and T (entropy = logZ - T/Z) ── */
    float local_sum = 0.0f;
    float local_t   = 0.0f;
    for (int v = tid; v < n_vocab; v += blockDim.x) {
        const float d = row_logits[v] * inv_temp - max_l;
        const float e = expf(d);
        local_sum += e;
        local_t   += d * e;
    }
    s_sum[tid] = local_sum;
    s_val[tid] = local_t;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_val[tid] += s_val[tid + stride];
        }
        __syncthreads();
    }
    const float z = s_sum[0];
    const float t = s_val[0];
    if (tid == 0) {
        argmax[row]  = amax;
        entropy[row] = logf(z) - t / z;
    }
    __syncthreads();

    /* ── multinomial draw: first v with cumulative exp(d) >= r, in vocab order.
     * Split the vocab into blockDim contiguous slices; each thread sums its slice;
     * exclusive-scan the slice sums; only the crossing thread walks its slice. ── */
    const float r = u[row] * z;
    const int   chunk = (n_vocab + blockDim.x - 1) / blockDim.x;
    const int   beg   = tid * chunk;
    const int   end   = min(beg + chunk, n_vocab);

    float slice_sum = 0.0f;
    for (int v = beg; v < end; ++v) {
        slice_sum += expf(row_logits[v] * inv_temp - max_l);
    }
    s_sum[tid] = slice_sum;
    __syncthreads();

    __shared__ int s_tok;
    if (tid == 0) {
        s_tok    = n_vocab - 1;     /* default if cum never reaches r (FP guard) */
        s_idx[0] = -1;              /* no crossing slice -> default stands */
        float pref = 0.0f;
        for (int i = 0; i < blockDim.x; ++i) {
            const float next = pref + s_sum[i];
            if (next >= r) { s_idx[0] = i; s_val[0] = pref; break; }
            pref = next;
        }
    }
    __syncthreads();

    if (tid == s_idx[0]) {          /* only the crossing thread walks its slice */
        float cum = s_val[0];
        for (int v = beg; v < end; ++v) {
            cum += expf(row_logits[v] * inv_temp - max_l);
            if (cum >= r) { s_tok = v; break; }
        }
    }
    __syncthreads();
    if (tid == 0) { sampled[row] = s_tok; }
}

/* N4a host wrapper: dg_sample_logits — run dg_sample_kernel over `n_pos` rows of a
 * DEVICE logits buffer `d_logits` ([n_pos x n_vocab], row-major), with per-position
 * seeded uniforms u_host[n_pos]. Writes argmax_host/entropy_host/sampled_host (all
 * caller-allocated, length n_pos). Returns 0 on success. Self-contained scratch
 * (alloc/free per call). Used by the N4-full host renoise loop and the N4a gate. */
extern "C" int dg_sample_logits(const float *d_logits, int n_pos, int n_vocab,
                                const float *u_host, float inv_temp,
                                int *argmax_host, float *entropy_host, int *sampled_host) {
    if (!d_logits || n_pos <= 0 || n_vocab <= 0 || !u_host ||
        !argmax_host || !entropy_host || !sampled_host) {
        sp_set_error("dg_sample_logits: bad args"); return 1; }
    float *d_u = NULL, *d_ent = NULL;
    int   *d_am = NULL, *d_sm = NULL;
    int rc = 1;
    cudaError_t e;
    e = cudaMalloc(&d_u,   (size_t)n_pos * sizeof(float));  if (e != cudaSuccess) { sp_set_error("dg_sample: u OOM"); goto out; }
    e = cudaMalloc(&d_am,  (size_t)n_pos * sizeof(int));    if (e != cudaSuccess) { sp_set_error("dg_sample: am OOM"); goto out; }
    e = cudaMalloc(&d_ent, (size_t)n_pos * sizeof(float));  if (e != cudaSuccess) { sp_set_error("dg_sample: ent OOM"); goto out; }
    e = cudaMalloc(&d_sm,  (size_t)n_pos * sizeof(int));    if (e != cudaSuccess) { sp_set_error("dg_sample: sm OOM"); goto out; }
    e = cudaMemcpy(d_u, u_host, (size_t)n_pos * sizeof(float), cudaMemcpyHostToDevice);
    if (e != cudaSuccess) { sp_set_error("dg_sample: u H2D"); goto out; }
    dg_sample_kernel<<<n_pos, 256>>>(d_logits, d_u, d_am, d_ent, d_sm, n_vocab, inv_temp);
    e = cudaGetLastError();          if (e != cudaSuccess) { fail_cuda(e, "dg_sample kernel"); goto out; }
    e = cudaDeviceSynchronize();     if (e != cudaSuccess) { fail_cuda(e, "dg_sample sync"); goto out; }
    e = cudaMemcpy(argmax_host,  d_am,  (size_t)n_pos * sizeof(int),   cudaMemcpyDeviceToHost); if (e != cudaSuccess) { sp_set_error("dg_sample: am D2H"); goto out; }
    e = cudaMemcpy(entropy_host, d_ent, (size_t)n_pos * sizeof(float), cudaMemcpyDeviceToHost); if (e != cudaSuccess) { sp_set_error("dg_sample: ent D2H"); goto out; }
    e = cudaMemcpy(sampled_host, d_sm,  (size_t)n_pos * sizeof(int),   cudaMemcpyDeviceToHost); if (e != cudaSuccess) { sp_set_error("dg_sample: sm D2H"); goto out; }
    rc = 0;
out:
    if (d_u)   cudaFree(d_u);
    if (d_am)  cudaFree(d_am);
    if (d_ent) cudaFree(d_ent);
    if (d_sm)  cudaFree(d_sm);
    return rc;
}

/* N4a gate entry: dg_sample_logits_host — same as dg_sample_logits but takes a HOST
 * logits buffer (uploads it). Lets the gate test feed a fixed CPU fixture without
 * owning device memory. Returns 0 on success. */
extern "C" int dg_sample_logits_host(const float *h_logits, int n_pos, int n_vocab,
                                     const float *u_host, float inv_temp,
                                     int *argmax_host, float *entropy_host, int *sampled_host) {
    if (!h_logits || n_pos <= 0 || n_vocab <= 0) { sp_set_error("dg_sample_host: bad args"); return 1; }
    float *d_logits = NULL;
    cudaError_t e = cudaMalloc(&d_logits, (size_t)n_pos * n_vocab * sizeof(float));
    if (e != cudaSuccess) { sp_set_error("dg_sample_host: logits OOM"); return 1; }
    e = cudaMemcpy(d_logits, h_logits, (size_t)n_pos * n_vocab * sizeof(float), cudaMemcpyHostToDevice);
    if (e != cudaSuccess) { sp_set_error("dg_sample_host: logits H2D"); cudaFree(d_logits); return 1; }
    int rc = dg_sample_logits(d_logits, n_pos, n_vocab, u_host, inv_temp,
                              argmax_host, entropy_host, sampled_host);
    cudaFree(d_logits);
    return rc;
}

/* ── N3 self-conditioning support ──────────────────────────────────────────────
 * Per-row softmax over a [n_row x n_vocab] device buffer, scaled by sc_temp_inv before
 * the softmax: out[r][v] = softmax_v(logits[r][v] * sc_temp_inv). One block per row
 * (blockDim 256). Mirrors the reference soft_max(scale(sc_logits, sc_temp_inv)). */
__global__ void dg_k_softmax_rows(const float *logits, float *out, int n_vocab, float sc_temp_inv) {
    int r = blockIdx.x; int tid = threadIdx.x;
    const float *row = logits + (size_t)r * n_vocab;
    float *o = out + (size_t)r * n_vocab;
    __shared__ float red[256];
    /* max */
    float m = -3.4e38f;
    for (int v = tid; v < n_vocab; v += blockDim.x) { float x = row[v] * sc_temp_inv; if (x > m) m = x; }
    red[tid] = m; __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) { if (tid < s && red[tid+s] > red[tid]) red[tid] = red[tid+s]; __syncthreads(); }
    float mx = red[0]; __syncthreads();
    /* sum exp */
    float se = 0.0f;
    for (int v = tid; v < n_vocab; v += blockDim.x) se += expf(row[v] * sc_temp_inv - mx);
    red[tid] = se; __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid+s]; __syncthreads(); }
    float inv = 1.0f / red[0];
    for (int v = tid; v < n_vocab; v += blockDim.x) o[v] = expf(row[v] * sc_temp_inv - mx) * inv;
}

/* MoE router input prep (per token): tmp[t] = rms_norm_noscale(x[t]) * (1/sqrt(E)) * gis[i].
 * One block per token (grid = n_tok, blockDim 256). Mirrors gemma4_build_ffn_moe's router:
 * tmp = rms_norm(attn_out, eps); tmp *= 1/sqrt(n_embd); tmp *= ffn_gate_inp_s. */
__global__ void k_dg_router_prep(const float *x, float *tmp, const float *gis,
                                 int E, float eps, float rscale) {
    int t = blockIdx.x;
    const float *xr = x + (size_t)t * E;
    float *tr = tmp + (size_t)t * E;
    __shared__ double sh[256];
    double s = 0.0;
    for (int i = threadIdx.x; i < E; i += blockDim.x) { double v = xr[i]; s += v * v; }
    sh[threadIdx.x] = s; __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (threadIdx.x < o) sh[threadIdx.x] += sh[threadIdx.x + o];
        __syncthreads();
    }
    float inv = 1.0f / sqrtf((float)(sh[0] / (double)E) + eps);
    for (int i = threadIdx.x; i < E; i += blockDim.x) tr[i] = xr[i] * inv * rscale * gis[i];
}

/* ── DiffusionGemma streaming-residency fix (G-DG-STREAMFIX) ──────────────────────
 * The 26B-A4B OK_Q4B model is 14 GB; its arena packed-weight buffers (codes/row_scale/
 * row_prec/bscale in sp_frob_packed_tensor) ALIAS the read-only MapViewOfFile view of
 * the .sp-model (sp_model_load.c: "arena tensors alias mmap"). On a memory-constrained
 * box the per-layer expert-weight dequant reads those aliased pointers DURING compute,
 * triggering a demand page-fault on the mmap view under commit pressure — which
 * deadlocks (CPU ~0% / GPU 0% / disk idle, WorkingSet pinned), nondeterministically at
 * L5/L12/L14 (a page-cache boundary). Diagnosis: tests/fixtures/chat_fullstack/
 * G-DIFFJUDGE-NATIVE.log (a full sequential pre-warm read completed in 6.7s but the OS
 * did not retain the pages → the hang is the demand-fault path, not cold-read latency).
 *
 * FIX (diffusion-path / arch_id 9 ONLY, additive): before dequant, COPY the exact
 * aliased source byte-ranges for the rows we are about to read into OWNED, committed
 * heap buffers and dequant from THOSE. The memcpy forces every source page resident
 * SYNCHRONOUSLY (no fault during the dequant loop), and the owned buffer is reclaimable
 * heap — not a reclaimable file-cache page — so it cannot be trimmed mid-read. Resident
 * host memory is bounded to ~one tensor slice (freed per call). The dense gemma4/
 * gemma3/qwen3 forwards and the 12B path never call this and stay byte-identical.
 *
 * dg_dequant_resident_rows: dequant `rows` rows starting at r0 of arena tensor `at`
 * into `dst` [rows*at->cols], reading from an owned snapshot of the source slices. The
 * dequant arithmetic is sp_frob_packed_dequant_row's, byte-for-byte (we build a local
 * sp_frob_packed_tensor over the copies and call sp_arena_dequant_row on it). Returns 0
 * on success; nonzero (and frees temporaries) on a bad arg / OOM. */
/* ── N5b: resident-weight reservoir (G-DG-N5b) ───────────────────────────────────
 * The diffusion 26B's arena packed buffers ALIAS the .sp-model mmap; the OS evicts the
 * 14GB between denoise steps -> every step re-faults experts from disk (~3 min/step).
 * Fix: on first touch of an arena tensor, clone its packed buffers into OWNED, committed
 * heap ONCE (the memcpy faults the mmap in a single time); every later step reads RAM.
 * Same bytes -> byte-identical output. Gated by SP_DG_RESERVOIR (default-off = null floor:
 * dg_dequant_resident_rows reads the mmap-aliased &at->pt exactly as before). Freed in
 * sp_cuda_model_release. Diffusion/arch-9 path only; the dense 12B never calls this. */
typedef struct { const sp_arena_tensor *key; sp_frob_packed_tensor own; } dg_res_ent;
static dg_res_ent *dg_res_tab = NULL;
static int dg_res_n = 0, dg_res_cap = 0;
static int dg_reservoir_enabled(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("SP_DG_RESERVOIR"); v = (e && *e && *e != '0') ? 1 : 0; }
    return v;
}
/* Resident owned-heap clone of at->pt, built+cached on first touch. On any failure returns
 * &at->pt (the mmap-aliased original) = safe, byte-identical fallback. */
static const sp_frob_packed_tensor *dg_resident_pt(const sp_arena_tensor *at) {
    const sp_frob_packed_tensor *s = &at->pt;
    if (!dg_reservoir_enabled()) return s;
    for (int i = 0; i < dg_res_n; i++) if (dg_res_tab[i].key == at) return &dg_res_tab[i].own;
    int rows = s->rows, cols = s->cols;
    if (rows <= 0) return s;
    size_t lastlen = (s->row_prec[rows - 1] == 8) ? (size_t)cols : (size_t)((cols + 1) / 2);
    size_t code_bytes = s->row_off[rows - 1] + lastlen;        /* row_off[0] == 0 */
    sp_frob_packed_tensor o; memset(&o, 0, sizeof o);
    o.rows = rows; o.cols = cols; o.alias_mask = 0; o.bs_nblk = s->bs_nblk; o.codes_bytes = code_bytes;
    size_t bsnb = (s->bscale && s->bs_nblk) ? (size_t)s->bs_nblk : 0;
    uint8_t  *codes = (uint8_t *)malloc(code_bytes ? code_bytes : 1);
    size_t   *roff  = (size_t  *)malloc((size_t)rows * sizeof(size_t));
    uint8_t  *rprec = (uint8_t *)malloc((size_t)rows);
    float    *rscl  = s->row_scale ? (float *)malloc((size_t)rows * sizeof(float)) : NULL;
    uint16_t *bscl  = bsnb ? (uint16_t *)malloc((size_t)rows * bsnb * sizeof(uint16_t)) : NULL;
    if (!codes || !roff || !rprec || (s->row_scale && !rscl) || (bsnb && !bscl)) {
        free(codes); free(roff); free(rprec); free(rscl); free(bscl); return s; }
    memcpy(codes, s->codes,    code_bytes);                    /* faults all mmap pages in ONCE */
    memcpy(roff,  s->row_off,  (size_t)rows * sizeof(size_t));
    memcpy(rprec, s->row_prec, (size_t)rows);
    if (rscl) memcpy(rscl, s->row_scale, (size_t)rows * sizeof(float));
    if (bscl) memcpy(bscl, s->bscale,    (size_t)rows * bsnb * sizeof(uint16_t));
    o.codes = codes; o.row_off = roff; o.row_prec = rprec; o.row_scale = rscl; o.bscale = bscl;
    if (dg_res_n == dg_res_cap) {
        int nc = dg_res_cap ? dg_res_cap * 2 : 64;
        dg_res_ent *nt = (dg_res_ent *)realloc(dg_res_tab, (size_t)nc * sizeof(dg_res_ent));
        if (!nt) { free(codes); free(roff); free(rprec); free(rscl); free(bscl); return s; }
        dg_res_tab = nt; dg_res_cap = nc; }
    fprintf(stderr, "[N5b] reservoir clone #%d rows=%d code_bytes=%zu\n", dg_res_n + 1, rows, (size_t)code_bytes);
    dg_res_tab[dg_res_n].key = at; dg_res_tab[dg_res_n].own = o; dg_res_n++;
    return &dg_res_tab[dg_res_n - 1].own;
}
static void dg_reservoir_free(void) {
    for (int i = 0; i < dg_res_n; i++) { sp_frob_packed_tensor *o = &dg_res_tab[i].own;
        free((void *)o->codes); free((void *)o->row_off); free((void *)o->row_prec); free((void *)o->row_scale); free((void *)o->bscale); }
    free(dg_res_tab); dg_res_tab = NULL; dg_res_n = dg_res_cap = 0;
}
static int dg_dequant_resident_rows(const sp_arena_tensor *at, int r0, int rows, float *dst) {
    if (!at || r0 < 0 || rows <= 0 || r0 + rows > at->pt.rows || !dst) {
        sp_set_error("dg_dequant_resident: bad row range"); return 1; }
    const sp_frob_packed_tensor *s = dg_resident_pt(at);
    const int cols = s->cols;
    /* code byte span for rows [r0, r0+rows): from row_off[r0] to the end of the last row.
     * The last row's code length: Q8 row = cols bytes; Q4/Q4B row = ceil(cols/2) bytes.
     * row_off is monotonic, so [off0, last_off + last_len) covers the whole slice. */
    size_t off0   = s->row_off[r0];
    int    lastr  = r0 + rows - 1;
    size_t lastoff = s->row_off[lastr];
    size_t lastlen = (s->row_prec[lastr] == 8) ? (size_t)cols : (size_t)((cols + 1) / 2);
    size_t span    = (lastoff + lastlen) - off0;            /* code bytes to snapshot */

    /* owned snapshots (committed heap) — the memcpy faults the aliased mmap pages in */
    uint8_t *codes_c = (uint8_t *)malloc(span ? span : 1);
    uint8_t *prec_c  = (uint8_t *)malloc((size_t)rows);
    size_t  *off_c   = (size_t  *)malloc((size_t)rows * sizeof(size_t));
    float   *scale_c = s->row_scale ? (float *)malloc((size_t)rows * sizeof(float)) : NULL;
    uint16_t *bscl_c = NULL;
    size_t   bs_nblk = (size_t)(s->bscale ? s->bs_nblk : 0);
    if (s->bscale && bs_nblk) bscl_c = (uint16_t *)malloc((size_t)rows * bs_nblk * sizeof(uint16_t));
    if (!codes_c || !prec_c || !off_c || (s->row_scale && !scale_c) || (s->bscale && bs_nblk && !bscl_c)) {
        free(codes_c); free(prec_c); free(off_c); free(scale_c); free(bscl_c);
        sp_set_error("dg_dequant_resident: snapshot OOM"); return 1; }

    memcpy(codes_c, s->codes + off0, span);                 /* forces codes pages resident */
    for (int r = 0; r < rows; r++) {
        prec_c[r] = s->row_prec[r0 + r];                    /* forces row_prec resident */
        off_c[r]  = s->row_off[r0 + r] - off0;              /* rebase into the snapshot */
        if (scale_c) scale_c[r] = s->row_scale[r0 + r];     /* forces row_scale resident */
        if (bscl_c)  memcpy(bscl_c + (size_t)r * bs_nblk,
                            s->bscale + (size_t)(r0 + r) * bs_nblk,
                            bs_nblk * sizeof(uint16_t));     /* forces bscale resident */
    }

    /* local packed tensor over the OWNED snapshot — identical dequant, no mmap deref */
    sp_arena_tensor loc; memset(&loc, 0, sizeof loc);
    loc.pt.rows = rows; loc.pt.cols = cols;
    loc.pt.row_prec = prec_c; loc.pt.row_scale = scale_c; loc.pt.row_off = off_c;
    loc.pt.codes = codes_c; loc.pt.codes_bytes = span;
    loc.pt.alias_mask = 0; loc.pt.bscale = bscl_c; loc.pt.bs_nblk = (int)(bscl_c ? bs_nblk : 0);

    int rc = 0;
    for (int r = 0; r < rows && rc == 0; r++)
        rc = sp_arena_dequant_row(&loc, r, dst + (size_t)r * cols);

    free(codes_c); free(prec_c); free(off_c); free(scale_c); free(bscl_c);
    if (rc) sp_set_error("dg_dequant_resident: dequant row");
    return rc;
}

/* ── host helpers: dequant a packed arena tensor [out x in] -> device f32 [out*in].
 * Mirrors upload_weight's dequant-then-upload but reads the arena (packed OK_Q4B/Q8)
 * via the RESIDENT-snapshot path (dg_dequant_resident_rows) so the demand-faulting mmap
 * view is never dereferenced during compute (G-DG-STREAMFIX), so the f32 GEMM matches
 * the CPU oracle's dequant path. ── */
/* ── N5c-v3: resident DEVICE DevTensor cache (the weight-residence lever) ──
 * Keyed by tensor name. On hit, dg_gemm_packed reuses the device-resident DevTensor and SKIPS
 * upload_packed -> kills the per-forward packed-weight PCIe H2D on steps 2..N (the diffusion judge
 * re-runs the full forward ~48x over a static prompt). BYTE-IDENTICAL (same DevTensor -> same
 * gemm_w). FIRST CUT: dense backbone only (dg_gemm_packed); experts (dg_gemm_packed_rows) unchanged.
 * NO eviction + a cudaMemGetInfo budget-stop guard so it can NEVER OOM (stops caching near the
 * limit; uncached tensors upload+free as before). Gated SP_DG_WCACHE (default-off = null floor:
 * lookup misses + insert refuses => dg_gemm_packed uploads+frees exactly as v2). Freed at release. */
typedef struct { char name[160]; DevTensor dt; } dg_wc_ent;
static dg_wc_ent *dg_wc_tab = NULL;
static int dg_wc_n = 0, dg_wc_cap = 0;
static int dg_wcache_enabled(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("SP_DG_WCACHE"); v = (e && *e && *e != '0') ? 1 : 0; }
    return v;
}
static size_t dg_wcache_margin(void) {
    static size_t m = 0;
    if (!m) { const char *e = getenv("SP_DG_WCACHE_MARGIN_MB"); long mb = (e && *e) ? atol(e) : 3072; if (mb < 256) mb = 256; m = (size_t)mb << 20; }
    return m;
}
static DevTensor *dg_wcache_lookup(const char *name) {
    if (!dg_wcache_enabled()) return NULL;
    for (int i = 0; i < dg_wc_n; i++) if (!strcmp(dg_wc_tab[i].name, name)) return &dg_wc_tab[i].dt;
    return NULL;
}
/* Copy the DevTensor handle into the table (cache OWNS the device buffers). Returns 1 if cached
 * (caller must NOT free), 0 if not (caller frees): disabled / name too long / grow fail / budget. */
static int dg_wcache_insert(const char *name, const DevTensor *dt) {
    if (!dg_wcache_enabled() || strlen(name) >= 160) return 0;
    size_t freeb = 0, totb = 0;
    if (cudaMemGetInfo(&freeb, &totb) != cudaSuccess || freeb < dg_wcache_margin()) return 0;  /* budget-stop (no eviction) */
    if (dg_wc_n == dg_wc_cap) {
        int nc = dg_wc_cap ? dg_wc_cap * 2 : 128;
        dg_wc_ent *nt = (dg_wc_ent *)realloc(dg_wc_tab, (size_t)nc * sizeof(dg_wc_ent));
        if (!nt) return 0;
        dg_wc_tab = nt; dg_wc_cap = nc;
    }
    snprintf(dg_wc_tab[dg_wc_n].name, 160, "%s", name);
    dg_wc_tab[dg_wc_n].dt = *dt; dg_wc_n++;
    if (getenv("SP_DG_TRACE")) fprintf(stderr, "[N5c-wc] resident #%d %s (free=%zuMB)\n", dg_wc_n, name, (size_t)(freeb >> 20));
    return 1;
}
static void dg_wcache_free(void) {
    for (int i = 0; i < dg_wc_n; i++) free_devtensor(&dg_wc_tab[i].dt);
    free(dg_wc_tab); dg_wc_tab = NULL; dg_wc_n = dg_wc_cap = 0;
}
static int dg_packed_enabled(void){ static int v=-1; if(v<0){ const char*e=getenv("SP_DG_PACKED"); v=(e&&*e&&*e!='0')?1:0; } return v; }
/* N6.2: reusable device dequant scratch -- hoists the synchronizing per-expert cudaMalloc/cudaFree
 * out of the MoE expert loop. cudaFree syncs the WHOLE device, serializing the stream and starving
 * the SMs (T2 profile: SM~55% with gaps, PCIe 28%, VRAM 12% = a serialization wall, not bw/compute).
 * One scratch reused across all experts, grown on demand, freed at teardown. Gated SP_DG_SCRATCHREUSE
 * (default-off = per-call malloc/free = BYTE-IDENTICAL null floor; dequant overwrites the full region). */
static float *dg_dqs = NULL; static size_t dg_dqs_floats = 0;
static int dg_scratch_reuse(void){ static int v=-1; if(v<0){ const char*e=getenv("SP_DG_SCRATCHREUSE"); v=(e&&*e&&*e=='0')?0:1; } return v; } /* N6.2b: default-ON (~1.45x diffjudge, byte-identical); SP_DG_SCRATCHREUSE=0 = opt-out null floor */
static float *dg_scratch_get(size_t need_floats){
    if (dg_dqs_floats < need_floats) {
        if (dg_dqs) cudaFree(dg_dqs);
        if (cudaMalloc(&dg_dqs, need_floats*sizeof(float)) != cudaSuccess) { dg_dqs=NULL; dg_dqs_floats=0; return NULL; }
        dg_dqs_floats = need_floats;
    }
    return dg_dqs;
}
static void dg_scratch_free(void){ if(dg_dqs) cudaFree(dg_dqs); dg_dqs=NULL; dg_dqs_floats=0; }
/* N5c: run an arena weight's GEMM through the EXISTING packed dp4a path (upload_packed +
 * per-token gemv_w_packed) instead of dg_upload_arena_w's f32 CPU-dequant + 28GB H2D. The
 * packed OK_Q4B H2D is HALF the f32 payload and the dequant runs ON-GPU (dp4a). Returns 0
 * if the packed path ran, 1 on hard error, 2 if unsupported (non-Q4B / in%32) -> caller
 * falls back to f32. Parity ~1e-3 (int8 act-quant), same as the dense dp4a decode path.
 * Gated by SP_DG_PACKED (default-off = the f32 path byte-for-byte = null floor). */
static int dg_gemm_packed(const qwen3_model *m, const char *name, int in, int out,
                          const float *dX, float *dY, int n_tok, cublasHandle_t cb, cudaStream_t st) {
    /* N5c-v3: resident DevTensor cache hit -> reuse device weights, SKIP upload_packed. */
    DevTensor *hit = dg_wcache_lookup(name);
    DevTensor devW; int cached = 0;
    if (hit) { devW = *hit; cached = 1; }
    else {
        const sp_arena_tensor *at = m->arena ? sp_arena_find(m->arena, name) : NULL;
        if (!at) { sp_set_error("dg_gemm_packed: tensor not in arena"); return 1; }
        memset(&devW, 0, sizeof devW);
        if (upload_packed(&at->pt, &devW)) { free_devtensor(&devW); return 1; }
    }
    /* N5c-v2: packed DevTensor -> k_dequant_arena_q4b (device dequant -> f32 scratch) -> cuBLAS
     * SGEMM via gemm_w. BYTE-IDENTICAL f32 (Q4B dequant->SGEMM has no extra rounding). */
    float *scratch = NULL; int own_scratch = 0;
    if (dg_scratch_reuse()) { scratch = dg_scratch_get((size_t)out * in);
        if (!scratch) { if (!cached) free_devtensor(&devW); sp_set_error("dg_gemm_packed: reuse scratch OOM"); return 1; } }
    else if (cudaMalloc(&scratch, (size_t)out * in * sizeof(float)) != cudaSuccess) {
        if (!cached) free_devtensor(&devW); sp_set_error("dg_gemm_packed: dequant scratch OOM"); return 1; }
    else own_scratch = 1;
    int rc = gemm_w(cb, st, &devW, dX, dY, n_tok, scratch);
    if (own_scratch) cudaFree(scratch);
    if (!cached) { if (!dg_wcache_insert(name, &devW)) free_devtensor(&devW); }   /* cache owns on insert; else free */
    return rc ? 1 : 0;
}
/* N5c MoE: one expert's GEMM via the packed dp4a path on a ROW-SLICE of a fused arena tensor.
 * Builds a slice descriptor over rows [r0,r0+rows) of `name` (cols=in), uploads it PACKED
 * (upload_packed) and runs cnt per-token gemv_w_packed -> dY[cnt x rows]. Mirrors
 * dg_dequant_resident_rows' slice extraction but uploads PACKED (half the f32 H2D, GPU dequant)
 * instead of dequanting to f32. Returns 0 ok / 1 error / 2 unsupported (non-Q4B / in%32 -> f32).
 * Reuses gemv_w_packed + k_gemv_q4b_dp4a_v2; NO new kernel. Gated by the caller (SP_DG_PACKED). */
/* ---- N6.3 async spillover double-buffer (gated SP_DG_ASYNC, default-off=byte-identical) ----
 * Prefetch the next expert's packed OK_Q4B weights on a dedicated upload stream (pinned
 * staging -> cudaMemcpyAsync) so the H2D overlaps the current expert's dequant+GEMM.
 * 4 fixed-size double-buffered slots (0,1=gate_up; 2,3=down); Q4B fixed row-width => constant
 * slot size, alloc once + reuse. up_ev: compute stream waits upload. cons_ev: upload stream
 * waits prior consume before reusing a slot buffer. Forward is deterministic (MOECHK) so this
 * is gated BYTE-EXACT. Default-off => dg_pf_consume misses => exact serial path. */
static int dg_async_enabled(void){ static int v=-1; if(v<0){ const char*e=getenv("SP_DG_ASYNC"); v=(e&&*e&&*e!='0')?1:0; } return v; }
static cudaStream_t dg_ustream = NULL;
typedef struct { char key[176]; int valid; int inited; DevTensor dt; unsigned char *pin; size_t pin_cap; cudaEvent_t up_ev; cudaEvent_t cons_ev; int cons_pending; int up_rec; } dg_pf_slot_t;
static dg_pf_slot_t dg_pf[4];
static int dg_pf_rr[2] = {0,0};
static void dg_async_ensure(void){
    if (dg_ustream) return;
    if (cudaStreamCreate(&dg_ustream) != cudaSuccess) { dg_ustream = NULL; return; }
    for (int i=0;i<4;i++){ memset(&dg_pf[i],0,sizeof(dg_pf[i])); cudaEventCreate(&dg_pf[i].up_ev); cudaEventCreate(&dg_pf[i].cons_ev); }
}
static void dg_prefetch_rows(const qwen3_model *m, const char *name, int r0, int rows, int in, int kind){
    if (!dg_async_enabled()) return;
    dg_async_ensure(); if (!dg_ustream) return;
    const sp_arena_tensor *at = m->arena ? sp_arena_find(m->arena, name) : NULL; if (!at) return;
    const sp_frob_packed_tensor *s = &at->pt;
    if (r0 < 0 || rows <= 0 || (uint32_t)(r0+rows) > s->rows) return;
    if (in & 31) return;
    for (int r=r0;r<r0+rows;r++) if (s->row_prec[r] != 4) return;
    size_t off0 = s->row_off[r0];
    size_t lastlen = (size_t)((in+1)/2);
    size_t span = (s->row_off[r0+rows-1] + lastlen) - off0;
    if (off0 + span > s->codes_bytes) return;
    int base = (kind==0)?0:2;
    int slot = base + (dg_pf_rr[kind] & 1); dg_pf_rr[kind]++;
    dg_pf_slot_t *S = &dg_pf[slot];
    if (S->valid) return;
    size_t bsc = s->bscale ? (size_t)rows * (size_t)s->bs_nblk * sizeof(unsigned short) : 0;
    if (!bsc) return;
    size_t need_pin = span + (size_t)rows*sizeof(unsigned long long) + bsc;
    if (!S->inited){
        if (cudaMalloc((void**)&S->dt.codes, span) != cudaSuccess) return;
        if (cudaMalloc((void**)&S->dt.row_off, (size_t)rows*sizeof(unsigned long long)) != cudaSuccess) return;
        if (cudaMalloc((void**)&S->dt.bscale, bsc) != cudaSuccess) return;
        if (cudaHostAlloc((void**)&S->pin, need_pin, cudaHostAllocDefault) != cudaSuccess) return;
        S->pin_cap = need_pin; S->inited = 1;
    }
    if (need_pin > S->pin_cap) return;
    if (S->up_rec) cudaEventSynchronize(S->up_ev);  /* HOST W-A-R: wait prior DMA done reading S->pin before CPU overwrites it (the n_tok-small race) */
    unsigned char *p = S->pin;
    memcpy(p, s->codes + off0, span);
    unsigned long long *po = (unsigned long long*)(p + span);
    for (int r=0;r<rows;r++) po[r] = (unsigned long long)(s->row_off[r0+r] - off0);
    unsigned char *pb = (unsigned char*)(po + rows);
    memcpy(pb, (const unsigned char*)(s->bscale + (size_t)r0*(size_t)s->bs_nblk), bsc);
    if (S->cons_pending) cudaStreamWaitEvent(dg_ustream, S->cons_ev, 0);
    cudaMemcpyAsync(S->dt.codes, p, span, cudaMemcpyHostToDevice, dg_ustream);
    cudaMemcpyAsync(S->dt.row_off, po, (size_t)rows*sizeof(unsigned long long), cudaMemcpyHostToDevice, dg_ustream);
    cudaMemcpyAsync(S->dt.bscale, pb, bsc, cudaMemcpyHostToDevice, dg_ustream);
    cudaEventRecord(S->up_ev, dg_ustream); S->up_rec = 1;
    S->dt.in = in; S->dt.out = rows; S->dt.prec = 4; S->dt.f32 = NULL; S->dt.row_scale = NULL; S->dt.row_prec = NULL; S->dt.bs_nblk = s->bs_nblk;
    snprintf(S->key, sizeof S->key, "%s#%d", name, r0);
    S->valid = 1;
}
static int dg_pf_consume(const char *key, DevTensor *out, cudaStream_t st){
    for (int i=0;i<4;i++){ if (dg_pf[i].valid && !strcmp(dg_pf[i].key, key)){ cudaStreamWaitEvent(st, dg_pf[i].up_ev, 0); *out = dg_pf[i].dt; dg_pf[i].valid = 0; return i; } }
    return -1;
}
static void dg_pf_mark_consumed(int slot, cudaStream_t st){ if (slot<0||slot>=4) return; cudaEventRecord(dg_pf[slot].cons_ev, st); dg_pf[slot].cons_pending = 1; }
static void dg_async_free(void){
    for (int i=0;i<4;i++){
        if (dg_pf[i].inited){ if(dg_pf[i].dt.codes)cudaFree(dg_pf[i].dt.codes); if(dg_pf[i].dt.row_off)cudaFree(dg_pf[i].dt.row_off); if(dg_pf[i].dt.bscale)cudaFree(dg_pf[i].dt.bscale); if(dg_pf[i].pin)cudaFreeHost(dg_pf[i].pin); }
        if (dg_pf[i].up_ev) cudaEventDestroy(dg_pf[i].up_ev);
        if (dg_pf[i].cons_ev) cudaEventDestroy(dg_pf[i].cons_ev);
        memset(&dg_pf[i],0,sizeof(dg_pf[i]));
    }
    if (dg_ustream){ cudaStreamDestroy(dg_ustream); dg_ustream=NULL; }
}
static int dg_gemm_packed_rows(const qwen3_model *m, const char *name, int r0, int rows, int in,
                               const float *dX, float *dY, int cnt, cublasHandle_t cb, cudaStream_t st) {
    char key[176]; snprintf(key, sizeof key, "%s#%d", name, r0);
    DevTensor *hit = dg_wcache_lookup(key);
    DevTensor devW; int cached = 0; int pf_slot = -1;
    if (hit) { devW = *hit; cached = 1; }
    else if (dg_async_enabled() && (pf_slot = dg_pf_consume(key, &devW, st)) >= 0) { cached = 2; }
    else {
        const sp_arena_tensor *at = m->arena ? sp_arena_find(m->arena, name) : NULL;
        if (!at) { sp_set_error("dg_gemm_packed_rows: tensor not in arena"); return 1; }
        const sp_frob_packed_tensor *s = &at->pt;
        if (r0 < 0 || rows <= 0 || (uint32_t)(r0 + rows) > s->rows) { sp_set_error("dg_gemm_packed_rows: bad slice"); return 1; }
        if (in & 31) return 2;
        for (int r = r0; r < r0 + rows; r++) if (s->row_prec[r] != 4) return 2;
        size_t off0 = s->row_off[r0];
        int    lastr = r0 + rows - 1;
        size_t lastlen = (size_t)((in + 1) / 2);
        size_t span = (s->row_off[lastr] + lastlen) - off0;
        if (off0 + span > s->codes_bytes) {   /* N6 fix+diag: slice would read past the mmap'd codes region -> host AV */
            fprintf(stderr, "[N6-SPANCLAMP] %s r0=%d rows=%d in=%d: off0=%zu span=%zu end=%zu > codes_bytes=%zu\n",
                    name, r0, rows, in, off0, span, off0+span, (size_t)s->codes_bytes); fflush(stderr);
            span = (off0 < s->codes_bytes) ? (s->codes_bytes - off0) : 0;
        }
        size_t *off_c = (size_t *)malloc((size_t)rows * sizeof(size_t));
        if (!off_c) { sp_set_error("dg_gemm_packed_rows: off OOM"); return 1; }
        for (int r = 0; r < rows; r++) off_c[r] = s->row_off[r0 + r] - off0;
        sp_frob_packed_tensor sl; memset(&sl, 0, sizeof sl);
        sl.rows = rows; sl.cols = in; sl.alias_mask = 0; sl.bs_nblk = s->bs_nblk;
        sl.codes = s->codes + off0; sl.codes_bytes = span;
        sl.row_off = off_c;
        sl.row_prec = s->row_prec + r0;
        sl.row_scale = s->row_scale ? s->row_scale + r0 : NULL;
        sl.bscale = s->bscale ? s->bscale + (size_t)r0 * s->bs_nblk : NULL;
        memset(&devW, 0, sizeof devW);
        if (getenv("SP_DG_PKVCKPT")) { cudaDeviceSynchronize(); fprintf(stderr, "[dgrows r0=%d rows=%d in=%d cnt=%d] slice built, upload_packed next (span=%zu off0=%zu codes_bytes=%zu)\n", r0, rows, in, cnt, span, off0, (size_t)sl.codes_bytes); fflush(stderr); }
        int urc = upload_packed(&sl, &devW);
        if (getenv("SP_DG_PKVCKPT")) { cudaError_t _e=cudaDeviceSynchronize(); int _h=_heapchk(); fprintf(stderr, "[dgrows r0=%d] upload_packed done urc=%d heap=%s sync=%s\n", r0, urc, _h==_HEAPOK?"OK":"CORRUPT", cudaGetErrorString(_e)); fflush(stderr); }
        free(off_c);
        if (urc) { free_devtensor(&devW); sp_set_error("dg_gemm_packed_rows: upload_packed"); return 1; }
        if (devW.prec != 4) { free_devtensor(&devW); return 2; }
    }
    float *scratch = NULL; int own_scratch = 0;
    if (dg_scratch_reuse()) { scratch = dg_scratch_get((size_t)rows * in);
        if (!scratch) { if (!cached) free_devtensor(&devW); sp_set_error("dg_gemm_packed_rows: reuse scratch OOM"); return 1; } }
    else if (cudaMalloc(&scratch, (size_t)rows * in * sizeof(float)) != cudaSuccess) {
        if (!cached) free_devtensor(&devW); sp_set_error("dg_gemm_packed_rows: dequant scratch OOM"); return 1; }
    else own_scratch = 1;
    int rc = gemm_w(cb, st, &devW, dX, dY, cnt, scratch);
    if (own_scratch) cudaFree(scratch);
    if (pf_slot >= 0) dg_pf_mark_consumed(pf_slot, st);
    if (getenv("SP_DG_PKVCKPT")) { cudaError_t _e=cudaDeviceSynchronize(); int _h=_heapchk(); fprintf(stderr, "[dgrows r0=%d] gemm_w+scratchfree done rc=%d heap=%s sync=%s (free_devtensor next, cached=%d)\n", r0, rc, _h==_HEAPOK?"OK":"CORRUPT", cudaGetErrorString(_e), cached); fflush(stderr); }
    if (!cached) { if (!dg_wcache_insert(key, &devW)) free_devtensor(&devW); }
    if (getenv("SP_DG_PKVCKPT")) { int _h=_heapchk(); fprintf(stderr, "[dgrows r0=%d] free_devtensor done, returning rc=%d heap=%s\n", r0, rc, _h==_HEAPOK?"OK":"CORRUPT"); fflush(stderr); }
    return rc ? 1 : 0;
}
/* expose for the MoE dispatch (wired next session): silence unused-static until the 2-site flip. */
static int (* const dg_gemm_packed_rows_ref)(const qwen3_model*,const char*,int,int,int,const float*,float*,int,cublasHandle_t,cudaStream_t) = dg_gemm_packed_rows;
static float *dg_upload_arena_w(const qwen3_model *m, const char *name, int in, int out) {
    const sp_arena_tensor *at = m->arena ? sp_arena_find(m->arena, name) : NULL;
    if (!at) { sp_set_error("dg_upload_arena_w: tensor not in arena"); return NULL; }
    size_t n = (size_t)out * in;
    float *host = (float *)malloc(n * sizeof(float));
    if (!host) { sp_set_error("dg_upload_arena_w: host OOM"); return NULL; }
    if (dg_dequant_resident_rows(at, 0, out, host)) { free(host); return NULL; }
    float *dev = NULL;
    cudaError_t e = cudaMalloc(&dev, n * sizeof(float));
    if (e == cudaSuccess) e = cudaMemcpy(dev, host, n * sizeof(float), cudaMemcpyHostToDevice);
    free(host);
    if (e != cudaSuccess) { fail_cuda(e, "dg_upload_arena_w upload"); if (dev) cudaFree(dev); return NULL; }
    return dev;
}

/* dequant a slice of `rows` consecutive rows of an arena tensor starting at row r0 into a
 * fresh device f32 [rows*in] (used for the fused rank-3 expert tensors: expert e occupies
 * rows [e*rows_per_expert, ...)). Reads via the resident-snapshot path (G-DG-STREAMFIX). */
static float *dg_upload_arena_rows(const qwen3_model *m, const char *name,
                                   int r0, int rows, int in) {
    const sp_arena_tensor *at = m->arena ? sp_arena_find(m->arena, name) : NULL;
    if (!at) { sp_set_error("dg_upload_arena_rows: tensor not in arena"); return NULL; }
    size_t n = (size_t)rows * in;
    float *host = (float *)malloc(n * sizeof(float));
    if (!host) { sp_set_error("dg_upload_arena_rows: host OOM"); return NULL; }
    if (dg_dequant_resident_rows(at, r0, rows, host)) { free(host); return NULL; }
    float *dev = NULL;
    cudaError_t e = cudaMalloc(&dev, n * sizeof(float));
    if (e == cudaSuccess) e = cudaMemcpy(dev, host, n * sizeof(float), cudaMemcpyHostToDevice);
    free(host);
    if (e != cudaSuccess) { fail_cuda(e, "dg_upload_arena_rows upload"); if (dev) cudaFree(dev); return NULL; }
    return dev;
}

/* device vector (per-layer norm weight, already f32 host-side in qm->norm_buf). The bridge
 * holds the norm f32 host buffers; find by tensor name through the arena? No — norms are NOT
 * in the arena. The bridge stores them in qm->norm_buf[ni] paired with qm->norm_src[ni]->name.
 * dg_norm_host returns the host f32 pointer for a norm tensor name (NULL if absent). */
static const float *dg_norm_host(const qwen3_model *m, const char *name) {
    for (int i = 0; i < m->n_norm; i++)
        if (m->norm_src[i] && strcmp(m->norm_src[i]->name, name) == 0) return m->norm_buf[i];
    return NULL;
}
static float *dg_upload_norm(const qwen3_model *m, const char *name, int n) {
    const float *h = dg_norm_host(m, name);
    if (!h) { sp_set_error("dg_upload_norm: norm not found"); return NULL; }
    float *dev = NULL;
    cudaError_t e = cudaMalloc(&dev, (size_t)n * sizeof(float));
    if (e == cudaSuccess) e = cudaMemcpy(dev, h, (size_t)n * sizeof(float), cudaMemcpyHostToDevice);
    if (e != cudaSuccess) { fail_cuda(e, "dg_upload_norm"); if (dev) cudaFree(dev); return NULL; }
    return dev;
}

/* ── N3 self-conditioning: add the previous step's canvas-logit feedback into the
 * canvas embedding (dx rows [P, n_tok)) BEFORE the canvas rmsnorm_noscale.
 * Mirrors the reference dg_canvas_embed SC subgraph (diffusion-gemma.cpp 384-426):
 *   probs   = softmax(prev_logits[c] * sc_temp_inv)              [C x V]
 *   soft    = (probs @ embed) * sqrt(E)                          [C x E]   (soft-embedding)
 *   normed  = rms_norm(soft) * sc_pre_norm                       [C x E]   (plain weighted RMS)
 *   sc_sig  = sc_down( gelu_tanh(sc_gate(normed)) * sc_up(normed) )        [C x E]
 *   dx[P+c] += sc_sig[c]
 * prev_logits is a DEVICE [C x V] buffer (canvas rows of the prior step's raw logits).
 * embedT = token_embd dequanted to device f32 [V x E] (native row-major). The probs@embed
 * GEMM runs in column-major directly (no host transpose): with embedT laid out [v*E+e] it
 * is col-major [E x V]; probs [c*V+v] is col-major [V x C]; soft [c*E+e] col-major [E x C];
 * Scm = Ecm * Pcm = [E x V][V x C] = [E x C]. Returns 0 on success. SC FF width = FF (dense).
 * Streams the V x E embedding to f32 (~3 GB transient on a 12 GB card); freed before return. */
/* the dequanted token-embedding [V x E] f32 device buffer, cached across SC steps/queries
 * (the embedding is a constant; re-dequanting + re-uploading ~3 GB every step dominated the
 * SC cost). Keyed by the model pointer; rebuilt only if the model changes. */
static const qwen3_model *g_dg_sc_embed_key = NULL;
static float *g_dg_sc_embed = NULL;

static int dg_self_cond(const qwen3_model *m, cublasHandle_t cb, cudaStream_t st,
                        float *dx, const float *prev_logits, int P, int C,
                        int E, int V, int FF, float eps, float sc_temp_inv) {
    int rc = 1;
    float *d_embed = NULL, *d_probs = NULL, *d_soft = NULL, *d_norm = NULL;
    float *d_g = NULL, *d_up = NULL, *d_sig = NULL;
    float *d_preW = NULL, *dWg = NULL, *dWu = NULL, *dWd = NULL;
    /* dequant the full token-embedding [V x E] to device f32 ONCE (cached) */
    if (g_dg_sc_embed && g_dg_sc_embed_key == m) {
        d_embed = g_dg_sc_embed;          /* cache hit: reuse */
    } else {
        if (g_dg_sc_embed) { cudaFree(g_dg_sc_embed); g_dg_sc_embed = NULL; g_dg_sc_embed_key = NULL; }
        const sp_arena_tensor *eat = m->arena ? sp_arena_find(m->arena, m->token_embd->name) : NULL;
        if (!eat) { sp_set_error("dg_self_cond: token_embd not in arena"); return 1; }
        float *h_embed = (float *)malloc((size_t)V * E * sizeof(float));
        if (!h_embed) { sp_set_error("dg_self_cond: embed host OOM"); return 1; }
        if (dg_dequant_resident_rows(eat, 0, V, h_embed)) { free(h_embed); return 1; }
        cudaError_t e = cudaMalloc(&d_embed, (size_t)V * E * sizeof(float));
        if (e == cudaSuccess) e = cudaMemcpy(d_embed, h_embed, (size_t)V * E * sizeof(float), cudaMemcpyHostToDevice);
        free(h_embed);
        if (e != cudaSuccess) { fail_cuda(e, "dg_self_cond: embed upload"); if (d_embed) cudaFree(d_embed); d_embed = NULL; goto out; }
        g_dg_sc_embed = d_embed; g_dg_sc_embed_key = m;   /* cache for subsequent steps */
    }
    if (cudaMalloc(&d_probs, (size_t)C * V * sizeof(float)) != cudaSuccess) { sp_set_error("dg_self_cond probs OOM"); goto out; }
    if (cudaMalloc(&d_soft,  (size_t)C * E * sizeof(float)) != cudaSuccess) { sp_set_error("dg_self_cond soft OOM"); goto out; }
    if (cudaMalloc(&d_norm,  (size_t)C * E * sizeof(float)) != cudaSuccess) { sp_set_error("dg_self_cond norm OOM"); goto out; }
    if (cudaMalloc(&d_g,     (size_t)C * FF * sizeof(float)) != cudaSuccess) { sp_set_error("dg_self_cond g OOM"); goto out; }
    if (cudaMalloc(&d_up,    (size_t)C * FF * sizeof(float)) != cudaSuccess) { sp_set_error("dg_self_cond up OOM"); goto out; }
    if (cudaMalloc(&d_sig,   (size_t)C * E * sizeof(float)) != cudaSuccess) { sp_set_error("dg_self_cond sig OOM"); goto out; }

    /* probs = softmax(prev_logits * sc_temp_inv) over V, per canvas row */
    dg_k_softmax_rows<<<C, 256, 0, st>>>(prev_logits, d_probs, V, sc_temp_inv);

    /* soft = probs @ embed  (col-major: Scm[E x C] = Ecm[E x V] . Pcm[V x C]) */
    {
        const float a = 1.0f, b = 0.0f;
        if (cublasSgemm(cb, CUBLAS_OP_N, CUBLAS_OP_N, E, C, V,
                        &a, d_embed, E, d_probs, V, &b, d_soft, E) != CUBLAS_STATUS_SUCCESS) {
            sp_set_error("dg_self_cond: probs@embed sgemm"); goto out; }
    }
    { size_t nCE = (size_t)C * E;
      k_scale_by_const<<<(unsigned)((nCE+255)/256), 256, 0, st>>>(d_soft, nCE, sqrtf((float)E)); }

    /* normed = rms_norm(soft) * sc_pre_norm  (plain weighted RMS, per row) */
    d_preW = dg_upload_norm(m, "self_cond_pre_norm.weight", E);
    if (!d_preW) goto out;
    k_rmsnorm<<<C, 256, 0, st>>>(d_soft, d_preW, E, eps, d_norm);

    /* SC gated MLP: g = sc_gate@normed [C x FF]; up = sc_up@normed; h = gelu(g)*up; sig = sc_down@h */
    dWg = dg_upload_arena_w(m, "self_cond_gate.weight", E, FF); if (!dWg) goto out;
    if (gemm(cb, dWg, d_norm, d_g, C, E, FF)) goto out;
    dWu = dg_upload_arena_w(m, "self_cond_up.weight", E, FF);   if (!dWu) goto out;
    if (gemm(cb, dWu, d_norm, d_up, C, E, FF)) goto out;
    { size_t nCF = (size_t)C * FF;
      k_gelu_mul<<<(unsigned)((nCF+255)/256), 256, 0, st>>>(d_g, d_up, nCF); }   /* d_g = gelu(g)*up */
    dWd = dg_upload_arena_w(m, "self_cond_down.weight", FF, E); if (!dWd) goto out;
    if (gemm(cb, dWd, d_g, d_sig, C, FF, E)) goto out;

    /* dx[P + c] += sc_sig[c]  (add into the canvas embedding rows in place) */
    { size_t nCE = (size_t)C * E;
      k_add<<<(unsigned)((nCE+255)/256), 256, 0, st>>>(dx + (size_t)P * E, d_sig, nCE); }
    cudaStreamSynchronize(st);
    rc = 0;
out:
    /* d_embed is CACHED (g_dg_sc_embed) — do NOT free here */
    if (d_probs) cudaFree(d_probs); if (d_soft) cudaFree(d_soft);
    if (d_norm) cudaFree(d_norm); if (d_g) cudaFree(d_g); if (d_up) cudaFree(d_up); if (d_sig) cudaFree(d_sig);
    if (d_preW) cudaFree(d_preW); if (dWg) cudaFree(dWg); if (dWu) cudaFree(dWu); if (dWd) cudaFree(dWd);
    return rc;
}

/* free the cached SC embed buffer (call on model unload to avoid a leak). */
extern "C" void dg_sc_embed_release(void) {
    if (g_dg_sc_embed) { cudaFree(g_dg_sc_embed); g_dg_sc_embed = NULL; g_dg_sc_embed_key = NULL; }
}

/* ── N6 PREFIX-KV invariance PROOF (S1a, observational, default-off SP_DG_PREFIXKV_PROOF) ──
 * The diffusion judge re-runs the full [prompt|canvas] forward every denoise step, but the mask
 * (k_attn_diffusion) makes prompt queries CAUSAL OVER PROMPT ONLY (k<P) -> each owner layer's
 * prompt K/V is a PURE FUNCTION OF THE PROMPT, INVARIANT as the canvas denoises. Before building
 * the canvas-only fast path on that, prove it BYTE-EXACT on the metal: cross-call cache the prompt
 * rows [0,P) of each owner Kst/Vst keyed by (model,P,prompt-hash); on a later step of the SAME
 * query (same key) compare fresh-vs-cached and report max|delta|. Touches NO logit (pure observe);
 * default-off = zero work = byte-identical null floor. */
static int dg_pkv_proof_enabled(void){ static int v=-1; if(v<0){ const char*e=getenv("SP_DG_PREFIXKV_PROOF"); v=(e&&*e&&*e!='0')?1:0; } return v; }
static int dg_prefixkv_enabled(void){ static int v=-1; if(v<0){ const char*e=getenv("SP_DG_PREFIXKV"); v=(e&&*e&&*e!='0')?1:0; } return v; }
static const qwen3_model *g_pkv_m = NULL;
static int       g_pkv_P = -1, g_pkv_NL = 0;
static unsigned long long g_pkv_h = 0ULL;
static float   **g_pkv_K = NULL, **g_pkv_V = NULL;
static float    *g_pkv_diff = NULL;     /* device scalar: running max|delta| this forward */
static long      dg_nan_routes = 0;     /* N6 diag: count degenerate (NaN/Inf-logit) router tokens */
static unsigned long long dg_pkv_hash(const int32_t *t, int n){
    unsigned long long h = 1469598103934665603ULL;
    for (int i=0;i<n;i++){ h ^= (unsigned int)t[i]; h *= 1099511628211ULL; }
    return h;
}
static void dg_pkv_free(void){
    /* N6: a prior fast-hit forward enqueues async D2D splices that READ g_pkv_K/V[L]; freeing the
     * device buffers before those complete is a use-after-free. Drain the device first. */
    if (g_pkv_K || g_pkv_V) cudaDeviceSynchronize();
    if (g_pkv_K){ for(int i=0;i<g_pkv_NL;i++) if(g_pkv_K[i]) cudaFree(g_pkv_K[i]); free(g_pkv_K); g_pkv_K=NULL; }
    if (g_pkv_V){ for(int i=0;i<g_pkv_NL;i++) if(g_pkv_V[i]) cudaFree(g_pkv_V[i]); free(g_pkv_V); g_pkv_V=NULL; }
    g_pkv_m=NULL; g_pkv_P=-1; g_pkv_NL=0; g_pkv_h=0ULL;
}
extern "C" void dg_prefixkv_release(void){ dg_pkv_free(); if(g_pkv_diff){ cudaFree(g_pkv_diff); g_pkv_diff=NULL; } }
/* max|a-b| over n floats, atomic-max into out (preset 0). abs diff>=0 so IEEE uint bits order. */
__global__ void dg_k_maxabsdiff(const float *a, const float *b, size_t n, float *out){
    size_t i = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (i>=n) return;
    float d = fabsf(a[i]-b[i]);
    atomicMax((unsigned int*)out, __float_as_uint(d));
}

/* DiffusionGemma UNIFIED forward. tokens = [prompt | canvas] (length n_tok, canvas =
 * last cfg.dg_canvas_length). out_logits = [n_tok x n_vocab] f32 (caller-allocated).
 * Returns 0 on success. Self-contained: builds its own cublas handle + stream, streams
 * weights per layer from the arena. */
/* internal: the forward, with optional N3 self-conditioning. prev_logits = a DEVICE
 * [C x V] buffer (canvas rows of the prior step's raw logits) or NULL (no SC = the
 * step-0 / single-forward path, byte-identical to the original). sc_temp_inv = 1/temp
 * of the PRIOR step (the temperature the prev logits were produced at). */
static int dg_forward_impl(const qwen3_model *m, const int32_t *tokens,
                           int n_tok, float *out_logits,
                           const float *prev_logits, float sc_temp_inv) {
    if (!m || m->cfg.arch != SP_ARCH_DIFFUSION_GEMMA) {
        sp_set_error("diffusion_gemma_forward_cuda: not a DiffusionGemma model"); return 1; }
    if (n_tok <= 0 || !tokens || !out_logits) {
        sp_set_error("diffusion_gemma_forward_cuda: bad args"); return 1; }
    const qwen3_config *c = &m->cfg;
    const int E = (int)c->n_embd, NL = (int)c->n_layers, V = (int)c->n_vocab;
    const int FF = (int)c->n_ff;                          /* dense shared MLP width */
    const int NE = (int)c->q36_n_expert, NU = (int)c->q36_n_expert_used;
    const int FFx = (int)c->q36_n_ff_exp;                 /* per-expert FFN width */
    const int CL = (int)c->dg_canvas_length;
    const float eps = c->rms_eps, embscale = sqrtf((float)E);
    const int period = (int)c->g4_swa_period ? (int)c->g4_swa_period : 6;
    const int kvfs   = (int)c->g4_n_kv_from_start ? (int)c->g4_n_kv_from_start : NL;
    const int g_nh = (int)c->n_head, g_nkv = (int)c->n_head_kv, g_hd = (int)c->head_dim;
    const int s_nh = (int)c->g4_nh_swa, s_nkv = (int)c->g4_nkv_swa, s_hd = (int)c->g4_hd_swa;
    const float g_base = c->rope_freq_base, s_base = c->g4_rope_base_swa;
    const int n_swa = (int)c->sliding_window;
    const float softcap = c->g4_logit_softcap;
    const int QDmax = (g_nh*g_hd > s_nh*s_hd) ? g_nh*g_hd : s_nh*s_hd;
    const int KVDmax = (g_nkv*g_hd > s_nkv*s_hd) ? g_nkv*g_hd : s_nkv*s_hd;

    /* region split P|C (UNIFIED): prompt = first (n_tok - canvas_length), canvas = last */
    const int P = (CL > 0 && n_tok > CL) ? (n_tok - CL) : 0;
    const int C = n_tok - P;

    int rc = 1;
    cublasHandle_t cb = NULL; cudaStream_t st = 0;
    int *dtoks = NULL;
    float *dx=NULL,*dnx=NULL,*dtmp=NULL,*dq=NULL,*dk=NULL,*dv=NULL,*dao=NULL,*dap=NULL;
    float *dg=NULL,*dup=NULL,*ddn=NULL,*dmlp=NULL,*dmoe=NULL,*dlog=NULL;
    float **Kst=NULL,**Vst=NULL;
    float *hrows=NULL,*h_logits=NULL,*h_router=NULL;
    char nm[96];

    #define DGDBG(...) do { if (getenv("SP_DG_TRACE")) { if (getenv("SP_DG_PKVCKPT")) { cudaError_t _ck = cudaDeviceSynchronize(); int _hc = _heapchk(); fprintf(stderr, "[ckpt sync=%s heap=%s] ", cudaGetErrorString(_ck), _hc==_HEAPOK?"OK":"CORRUPT"); } fprintf(stderr, "[dgtrace] " __VA_ARGS__); fprintf(stderr, "\n"); fflush(stderr); } } while (0)
    DGDBG("enter n_tok=%d E=%d NL=%d V=%d P=%d C=%d CL=%d", n_tok, E, NL, V, P, C, CL);
    if (cublasCreate(&cb) != CUBLAS_STATUS_SUCCESS) { sp_set_error("dg: cublasCreate"); return 1; }
    DGDBG("cublasCreate ok");
    Kst = (float **)calloc((size_t)NL, sizeof(float *));
    Vst = (float **)calloc((size_t)NL, sizeof(float *));
    if (!Kst || !Vst) { sp_set_error("dg: host OOM"); goto done; }

    #define DGA(p, cnt) do { if (cudaMalloc(&(p), (size_t)(cnt)*sizeof(float)) != cudaSuccess) { \
        sp_set_error("dg: cudaMalloc"); goto done; } } while (0)
    if (cudaMalloc(&dtoks, (size_t)n_tok*sizeof(int)) != cudaSuccess) { sp_set_error("dg dtoks OOM"); goto done; }
    DGA(dx, (size_t)n_tok*E);   DGA(dnx, (size_t)n_tok*E);   DGA(dtmp, (size_t)n_tok*E);
    DGA(dq, (size_t)n_tok*QDmax); DGA(dk, (size_t)n_tok*KVDmax); DGA(dv, (size_t)n_tok*KVDmax);
    DGA(dao, (size_t)n_tok*QDmax); DGA(dap, (size_t)n_tok*E);
    DGA(dg, (size_t)n_tok*((FF>FFx)?FF:FFx)); DGA(dup, (size_t)n_tok*((FF>FFx)?FF:FFx));
    DGA(ddn, (size_t)n_tok*E);  DGA(dmlp, (size_t)n_tok*E); DGA(dmoe, (size_t)n_tok*E);
    DGA(dlog, (size_t)n_tok*V);

    DGDBG("device allocs done");
    h_logits = (float *)malloc((size_t)NE * sizeof(float));
    h_router = (float *)malloc((size_t)n_tok * E * sizeof(float));
    if (!h_logits || !h_router) { sp_set_error("dg: host scratch OOM"); goto done; }
    DGDBG("host scratch done; gathering embeddings");

    /* ── embeddings: prompt = embed*sqrt(E); canvas = rmsnorm_noscale(embed*sqrt(E)).
     * Host-gather the token rows from the arena (the 13 GB model keeps no f32 embd
     * resident), scale, upload; then rmsnorm the canvas rows in place. ── */
    if (cudaMemcpyAsync(dtoks, tokens, (size_t)n_tok*sizeof(int), cudaMemcpyHostToDevice, st) != cudaSuccess) {
        sp_set_error("dg: upload tokens"); goto done; }
    {
        const sp_arena_tensor *eat = m->arena ? sp_arena_find(m->arena, m->token_embd->name) : NULL;
        if (!eat) { sp_set_error("dg: token_embd not in arena"); goto done; }
        hrows = (float *)malloc((size_t)n_tok * E * sizeof(float));
        if (!hrows) { sp_set_error("dg: embd gather OOM"); goto done; }
        for (int t = 0; t < n_tok; t++) {
            /* G-DG-STREAMFIX: read each embedding row through the resident-snapshot path
             * (one row at a time, token-scattered) so the mmap view is not demand-faulted
             * during compute. */
            if (dg_dequant_resident_rows(eat, tokens[t], 1, hrows + (size_t)t * E)) {
                sp_set_error("dg: embd row gather"); goto done; }
            for (int i = 0; i < E; i++) hrows[(size_t)t * E + i] *= embscale;
        }
        if (cudaMemcpyAsync(dx, hrows, (size_t)n_tok*E*sizeof(float), cudaMemcpyHostToDevice, st) != cudaSuccess) {
            sp_set_error("dg: embd H2D"); goto done; }
        cudaStreamSynchronize(st); free(hrows); hrows = NULL;
    }
    /* ── N3 self-conditioning: add prev-step canvas-logit feedback into the canvas
     * embedding BEFORE the weightless RMSNorm (reference dg_canvas_embed: canvas +=
     * sc_sig; then rms_norm_noscale). prev_logits==NULL -> no SC = step-0 path. ── */
    if (C > 0 && prev_logits) {
        DGDBG("self-conditioning: prev_logits present, sc_temp_inv=%.4f", sc_temp_inv);
        if (dg_self_cond(m, cb, st, dx, prev_logits, P, C, E, V, FF, eps, sc_temp_inv)) goto done;
    }
    if (C > 0) {                                          /* canvas rows: weightless RMSNorm */
        k_rmsnorm_noscale_rows<<<C, 256, 0, st>>>(dx, E, eps, P);
    }
    DGDBG("embeddings done; entering layer loop");

    /* N6 prefix-KV setup: cache prompt K/V (step-1 save) + on a repeat step of the same query run
     * CANVAS-ONLY (SP_DG_PREFIXKV) and/or byte-compare (SP_DG_PREFIXKV_PROOF). pkv_on drives the cache. */
    int pkv_fast_on = dg_prefixkv_enabled() && P > 0;
    int pkv_proof   = dg_pkv_proof_enabled() && P > 0;
    int pkv_on = pkv_fast_on || pkv_proof;
    int pkv_hit = 0;
    if (pkv_on) {
        unsigned long long h = dg_pkv_hash(tokens, P);
        pkv_hit = (g_pkv_K && g_pkv_V && g_pkv_m==m && g_pkv_P==P && g_pkv_NL==NL && g_pkv_h==h);
        if (!pkv_hit) {                       /* first step of a (new) query -> rebuild cache */
            dg_pkv_free();
            g_pkv_K = (float**)calloc((size_t)NL, sizeof(float*));
            g_pkv_V = (float**)calloc((size_t)NL, sizeof(float*));
            if (g_pkv_K && g_pkv_V) { g_pkv_m=m; g_pkv_P=P; g_pkv_NL=NL; g_pkv_h=h; }
            else { dg_pkv_free(); pkv_on = 0; pkv_proof = 0; pkv_fast_on = 0; }  /* OOM -> disable */
        }
        if (pkv_proof) {
            if (!g_pkv_diff) cudaMalloc(&g_pkv_diff, sizeof(float));
            if (g_pkv_diff) cudaMemsetAsync(g_pkv_diff, 0, sizeof(float), st);
        }
    }
    DGDBG("pkv setup done: pkv_on=%d pkv_hit=%d pkv_fast_on=%d P=%d (cache %s)", pkv_on, pkv_hit, pkv_fast_on, P, pkv_hit?"HIT/reuse":"rebuilt");
    const int pkv_fast = pkv_fast_on && pkv_hit;          /* canvas-only compute this forward */
    const size_t rb = (size_t)(pkv_fast ? P : 0);         /* compute-row base */
    const int    rn = pkv_fast ? C : n_tok;               /* compute-row count */
    dg_nan_routes = 0;                                    /* N6 diag: per-forward degenerate-router counter */

    /* ── layer loop ── */
    for (int L = 0; L < NL; L++) {
        DGDBG("layer %d/%d start", L, NL);
        const int global = ((L % period) == period - 1);
        const int nh  = global ? g_nh  : s_nh;
        const int nkv = global ? g_nkv : s_nkv;
        const int hd  = global ? g_hd  : s_hd;
        const int grp = nh / nkv, qd = nh * hd, kvd = nkv * hd;
        const float rbase = global ? g_base : s_base;
        const int is_swa = global ? 0 : 1;
        const float ascale = 1.0f;
        const size_t nE = (size_t)n_tok * E;
        const int has_v = (m->layers[L].attn_v != NULL);
        float *dW=NULL, *dWk=NULL, *dWv=NULL, *dWo=NULL, *dNw=NULL;

        /* rope-freqs table for global layers (proportional rope); host-side norm buf */
        float *d_ffac = NULL;
        if (global && m->rope_freqs) {
            d_ffac = dg_upload_norm(m, "rope_freqs.weight", g_hd / 2);
            if (!d_ffac) goto layerfail;
        }

        /* attn_norm -> nx */
        snprintf(nm, sizeof nm, "blk.%d.attn_norm.weight", L);
        dNw = dg_upload_norm(m, nm, E); if (!dNw) goto layerfail;
        k_rmsnorm<<<n_tok, 256, 0, st>>>(dx, dNw, E, eps, dnx);
        cudaFree(dNw); dNw = NULL;

        /* Q proj + q-norm + rope */
        snprintf(nm, sizeof nm, "blk.%d.attn_q.weight", L);
        { int dgp = dg_packed_enabled() ? dg_gemm_packed(m, nm, E, qd, dnx+rb*E, dq+rb*qd, rn, cb, st) : 2;
          if (dgp == 1) goto layerfail;
          if (dgp != 0) { dW = dg_upload_arena_w(m, nm, E, qd); if (!dW) goto layerfail;
            if (gemm(cb, dW, dnx+rb*E, dq+rb*qd, rn, E, qd)) { cudaFree(dW); goto layerfail; }
            cudaFree(dW); dW = NULL; } }
        { snprintf(nm, sizeof nm, "blk.%d.attn_q_norm.weight", L);
          float *dqn = dg_upload_norm(m, nm, hd); if (!dqn) goto layerfail;
          k_rmsnorm_head<<<n_tok*nh, 256, 0, st>>>(dq, dqn, nh, hd, qd, eps);
          cudaFree(dqn); }
        if (d_ffac) k_rope_freqs<<<n_tok*nh, hd/2, 0, st>>>(dq, nh, hd, qd, rbase, d_ffac);
        else        k_rope<<<n_tok*nh, hd/2, 0, st>>>(dq, nh, hd, qd, rbase);

        /* K/V — shared-KV: owners [0,kvfs) project + store; sharers reuse owner */
        float *Kuse, *Vuse;
        if (L < kvfs) {
            snprintf(nm, sizeof nm, "blk.%d.attn_k.weight", L);
            { int dgp = dg_packed_enabled() ? dg_gemm_packed(m, nm, E, kvd, dnx+rb*E, dk+rb*kvd, rn, cb, st) : 2;
              if (dgp == 1) goto layerfail;
              if (dgp != 0) { dWk = dg_upload_arena_w(m, nm, E, kvd); if (!dWk) goto layerfail;
                if (gemm(cb, dWk, dnx+rb*E, dk+rb*kvd, rn, E, kvd)) goto layerfail; } }
            if (has_v) {
                snprintf(nm, sizeof nm, "blk.%d.attn_v.weight", L);
                { int dgp = dg_packed_enabled() ? dg_gemm_packed(m, nm, E, kvd, dnx+rb*E, dv+rb*kvd, rn, cb, st) : 2;
                  if (dgp == 1) goto layerfail;
                  if (dgp != 0) { dWv = dg_upload_arena_w(m, nm, E, kvd); if (!dWv) goto layerfail;
                    if (gemm(cb, dWv, dnx+rb*E, dv+rb*kvd, rn, E, kvd)) goto layerfail; } }
            } else {  /* V-less (global) layer: V = raw K projection */
                cudaMemcpyAsync(dv, dk, (size_t)n_tok*kvd*sizeof(float), cudaMemcpyDeviceToDevice, st);
            }
            { snprintf(nm, sizeof nm, "blk.%d.attn_k_norm.weight", L);
              float *dkn = dg_upload_norm(m, nm, hd); if (!dkn) goto layerfail;
              k_rmsnorm_head<<<n_tok*nkv, 256, 0, st>>>(dk, dkn, nkv, hd, kvd, eps);
              cudaFree(dkn); }
            if (d_ffac) k_rope_freqs<<<n_tok*nkv, hd/2, 0, st>>>(dk, nkv, hd, kvd, rbase, d_ffac);
            else        k_rope<<<n_tok*nkv, hd/2, 0, st>>>(dk, nkv, hd, kvd, rbase);
            k_rmsnorm_head_noweight<<<n_tok*nkv, 256, 0, st>>>(dv, nkv, hd, kvd, eps);  /* weightless V-norm */
            if (cudaMalloc(&Kst[L], (size_t)n_tok*kvd*sizeof(float)) != cudaSuccess ||
                cudaMalloc(&Vst[L], (size_t)n_tok*kvd*sizeof(float)) != cudaSuccess) {
                sp_set_error("dg Kst OOM"); goto layerfail; }
            if (pkv_fast) {   /* N6 fast: prompt K/V [0,P) from cache (canvas-invariant), canvas [P,n_tok) fresh */
                cudaMemcpyAsync(Kst[L], g_pkv_K[L], (size_t)P*kvd*sizeof(float), cudaMemcpyDeviceToDevice, st);
                cudaMemcpyAsync(Vst[L], g_pkv_V[L], (size_t)P*kvd*sizeof(float), cudaMemcpyDeviceToDevice, st);
                cudaMemcpyAsync(Kst[L]+(size_t)P*kvd, dk+(size_t)P*kvd, (size_t)C*kvd*sizeof(float), cudaMemcpyDeviceToDevice, st);
                cudaMemcpyAsync(Vst[L]+(size_t)P*kvd, dv+(size_t)P*kvd, (size_t)C*kvd*sizeof(float), cudaMemcpyDeviceToDevice, st);
            } else {
                cudaMemcpyAsync(Kst[L], dk, (size_t)n_tok*kvd*sizeof(float), cudaMemcpyDeviceToDevice, st);
                cudaMemcpyAsync(Vst[L], dv, (size_t)n_tok*kvd*sizeof(float), cudaMemcpyDeviceToDevice, st);
            }
            Kuse = Kst[L]; Vuse = Vst[L];
            DGDBG("layer %d: KV store done (pkv_fast=%d pkv_hit=%d) -> %s", L, pkv_fast, pkv_hit, pkv_fast?"spliced from cache":"full");
            if (pkv_on && !pkv_hit) {   /* N6 step-1 SAVE: cache the canvas-invariant prompt K/V [0,P) */
                size_t prow = (size_t)P * kvd;
                if (cudaMalloc(&g_pkv_K[L], prow*sizeof(float))==cudaSuccess)
                    cudaMemcpyAsync(g_pkv_K[L], Kst[L], prow*sizeof(float), cudaMemcpyDeviceToDevice, st);
                if (cudaMalloc(&g_pkv_V[L], prow*sizeof(float))==cudaSuccess)
                    cudaMemcpyAsync(g_pkv_V[L], Vst[L], prow*sizeof(float), cudaMemcpyDeviceToDevice, st);
            } else if (pkv_proof && pkv_hit && !pkv_fast) {   /* N6 PROOF: compare fresh vs cached prompt K/V */
                size_t prow = (size_t)P * kvd;
                unsigned int gx = (unsigned int)((prow + 255) / 256);
                if (g_pkv_K[L]) dg_k_maxabsdiff<<<gx,256,0,st>>>(Kst[L], g_pkv_K[L], prow, g_pkv_diff);
                if (g_pkv_V[L]) dg_k_maxabsdiff<<<gx,256,0,st>>>(Vst[L], g_pkv_V[L], prow, g_pkv_diff);
            }
            if (dWk) { cudaFree(dWk); dWk = NULL; }
            if (dWv) { cudaFree(dWv); dWv = NULL; }
        } else {
            const int src = kvfs - (global ? 1 : 2);
            Kuse = Kst[src]; Vuse = Vst[src];
            if (!Kuse || !Vuse) { sp_set_error("dg sharer before owner"); goto layerfail; }
        }

        /* region-aware attention */
        DGDBG("layer %d: attn launch grid=%d bd=%d shmem=%zu", L, n_tok*nh,
              (n_tok<hd?hd:(n_tok>1024?1024:n_tok)), (size_t)n_tok*sizeof(float));
        {   int bd = n_tok; if (bd < hd) bd = hd; if (bd > 1024) bd = 1024;
            /* N6: +1 float of dynamic shared — the kernels also use a static __shared__ g_sum which
             * shares the shared window with the extern sc[]; without the pad, sc[n_tok-1] reads one
             * float past the requested dynamic region (compute-sanitizer: Invalid __shared__ read @5508). */
            const size_t attn_shmem = (size_t)(n_tok + 1) * sizeof(float);
            if (pkv_fast)
                k_attn_diffusion_canvas<<<C*nh, bd, attn_shmem, st>>>(
                    dq, Kuse, Vuse, n_tok, qd, kvd, hd, grp, ascale, P, n_swa, is_swa, P, dao);
            else
                k_attn_diffusion<<<n_tok*nh, bd, attn_shmem, st>>>(
                    dq, Kuse, Vuse, n_tok, qd, kvd, hd, grp, ascale, P, n_swa, is_swa, dao); }
        if (getenv("SP_DG_TRACE")) { cudaStreamSynchronize(st); DGDBG("layer %d: attn done (synced)", L); }

        /* O proj -> ap, post_attn_norm, residual into dx */
        snprintf(nm, sizeof nm, "blk.%d.attn_output.weight", L);
        { int dgp = dg_packed_enabled() ? dg_gemm_packed(m, nm, qd, E, dao+rb*qd, dap+rb*E, rn, cb, st) : 2;
          if (dgp == 1) goto layerfail;
          if (dgp != 0) { dWo = dg_upload_arena_w(m, nm, qd, E); if (!dWo) goto layerfail;
            if (gemm(cb, dWo, dao+rb*qd, dap+rb*E, rn, qd, E)) goto layerfail; cudaFree(dWo); dWo = NULL; } }
        { snprintf(nm, sizeof nm, "blk.%d.post_attention_norm.weight", L);
          float *dpn = dg_upload_norm(m, nm, E); if (!dpn) goto layerfail;
          k_rmsnorm<<<n_tok, 256, 0, st>>>(dap, dpn, E, eps, dnx);
          cudaFree(dpn); }
        k_add<<<(unsigned)((nE+255)/256), 256, 0, st>>>(dx, dnx, nE);
        /* dx now holds attn_out (the post-attention residual) — the FFN input */

        /* ── FFN: dense shared MLP + 128-expert MoE, combined, post_ffw_norm, residual ── */
        if (getenv("SP_DG_TRACE")) { cudaStreamSynchronize(st); DGDBG("layer %d: FFN start (dense MLP)", L); }
        /* (1) dense MLP: cur_mlp = post_ffw_norm_1( down( gelu(gate(ffn_norm(attn_out))) * up(...) ) ) */
        { snprintf(nm, sizeof nm, "blk.%d.ffn_norm.weight", L);
          float *dfn = dg_upload_norm(m, nm, E); if (!dfn) goto layerfail;
          k_rmsnorm<<<n_tok, 256, 0, st>>>(dx, dfn, E, eps, dnx);
          cudaFree(dfn); }
        snprintf(nm, sizeof nm, "blk.%d.ffn_gate.weight", L);
        { int dgp = dg_packed_enabled() ? dg_gemm_packed(m, nm, E, FF, dnx+rb*E, dg+rb*FF, rn, cb, st) : 2;
          if (dgp == 1) goto layerfail;
          if (dgp != 0) { dW = dg_upload_arena_w(m, nm, E, FF); if (!dW) goto layerfail;
            if (gemm(cb, dW, dnx+rb*E, dg+rb*FF, rn, E, FF)) goto layerfail; cudaFree(dW); dW = NULL; } }
        snprintf(nm, sizeof nm, "blk.%d.ffn_up.weight", L);
        { int dgp = dg_packed_enabled() ? dg_gemm_packed(m, nm, E, FF, dnx+rb*E, dup+rb*FF, rn, cb, st) : 2;
          if (dgp == 1) goto layerfail;
          if (dgp != 0) { dW = dg_upload_arena_w(m, nm, E, FF); if (!dW) goto layerfail;
            if (gemm(cb, dW, dnx+rb*E, dup+rb*FF, rn, E, FF)) goto layerfail; cudaFree(dW); dW = NULL; } }
        { size_t nFF = (size_t)n_tok * FF;
          k_gelu_mul<<<(unsigned)((nFF+255)/256), 256, 0, st>>>(dg, dup, nFF); }
        snprintf(nm, sizeof nm, "blk.%d.ffn_down.weight", L);
        { int dgp = dg_packed_enabled() ? dg_gemm_packed(m, nm, FF, E, dg+rb*FF, dmlp+rb*E, rn, cb, st) : 2;
          if (dgp == 1) goto layerfail;
          if (dgp != 0) { dW = dg_upload_arena_w(m, nm, FF, E); if (!dW) goto layerfail;
            if (gemm(cb, dW, dg+rb*FF, dmlp+rb*E, rn, FF, E)) goto layerfail; cudaFree(dW); dW = NULL; } }
        { snprintf(nm, sizeof nm, "blk.%d.post_ffw_norm_1.weight", L);
          float *dn1 = dg_upload_norm(m, nm, E); if (!dn1) goto layerfail;
          /* normalize dmlp in place: reuse dnx as scratch then copy back via k_rmsnorm out */
          k_rmsnorm<<<n_tok, 256, 0, st>>>(dmlp, dn1, E, eps, dnx);
          cudaMemcpyAsync(dmlp, dnx, nE*sizeof(float), cudaMemcpyDeviceToDevice, st);
          cudaFree(dn1); }

        if (getenv("SP_DG_TRACE")) { cudaStreamSynchronize(st); DGDBG("layer %d: dense MLP done, MoE start", L); }
        /* (2) MoE. router input tmp = rms_norm_noscale(attn_out)*(1/sqrt(E))*gate_inp_s;
         *     expert input cur_moe = pre_ffw_norm_2(attn_out). Both off the SAME attn_out (dx). */
        /* expert input -> dnx (pre_ffw_norm_2 . dx) */
        { snprintf(nm, sizeof nm, "blk.%d.pre_ffw_norm_2.weight", L);
          float *dp2 = dg_upload_norm(m, nm, E); if (!dp2) goto layerfail;
          k_rmsnorm<<<n_tok, 256, 0, st>>>(dx, dp2, E, eps, dnx);   /* dnx = expert hidden, all tokens */
          cudaFree(dp2); }
        /* router tmp -> dtmp = rms_norm_noscale(dx) * 1/sqrt(E) * gate_inp_s (per element) */
        snprintf(nm, sizeof nm, "blk.%d.ffn_gate_inp.scale", L);
        { const float *gis_h = dg_norm_host(m, nm);
          if (!gis_h) { sp_set_error("dg: ffn_gate_inp.scale missing"); goto layerfail; }
          float *dgis = NULL;
          if (cudaMalloc(&dgis, (size_t)E*sizeof(float)) != cudaSuccess) { sp_set_error("dg gis OOM"); goto layerfail; }
          cudaMemcpyAsync(dgis, gis_h, (size_t)E*sizeof(float), cudaMemcpyHostToDevice, st);
          k_dg_router_prep<<<n_tok, 256, 0, st>>>(dx, dtmp, dgis, E, eps, 1.0f/sqrtf((float)E));
          cudaFree(dgis); }
        /* router logits per token: gate_inp [NE x E] @ dtmp[t] -> h_logits; upload gate_inp once */
        {
            /* gate_inp is a NORM-class (f32) tensor in the bridge -> host pointer */
            snprintf(nm, sizeof nm, "blk.%d.ffn_gate_inp.weight", L);
            const float *gi_h = dg_norm_host(m, nm);
            if (!gi_h) { sp_set_error("dg: ffn_gate_inp.weight missing"); goto layerfail; }
            float *d_router = NULL;
            if (cudaMalloc(&d_router, (size_t)NE*E*sizeof(float)) != cudaSuccess) { sp_set_error("dg router OOM"); goto layerfail; }
            cudaMemcpyAsync(d_router, gi_h, (size_t)NE*E*sizeof(float), cudaMemcpyHostToDevice, st);
            float *d_rlog = NULL;
            if (cudaMalloc(&d_rlog, (size_t)n_tok*NE*sizeof(float)) != cudaSuccess) { cudaFree(d_router); sp_set_error("dg rlog OOM"); goto layerfail; }
            if (gemm(cb, d_router, dtmp, d_rlog, n_tok, E, NE)) { cudaFree(d_router); cudaFree(d_rlog); goto layerfail; }
            cudaFree(d_router);
            /* download router logits, do softmax+top-NU per token on host, gather experts */
            float *h_rlog = (float *)malloc((size_t)n_tok*NE*sizeof(float));
            if (!h_rlog) { cudaFree(d_rlog); sp_set_error("dg h_rlog OOM"); goto layerfail; }
            { cudaError_t _rl = cudaMemcpy(h_rlog, d_rlog, (size_t)n_tok*NE*sizeof(float), cudaMemcpyDeviceToHost);
              if (_rl != cudaSuccess) { fprintf(stderr, "[dgtrace] L%d h_rlog D2H FAULT: %s\n", L, cudaGetErrorString(_rl)); }
            }
            cudaFree(d_rlog);
            DGDBG("layer %d: router downloaded, dispatching experts (NE=%d NU=%d)", L, NE, NU);

            /* zero MoE accumulator */
            cudaMemsetAsync(dmoe, 0, nE*sizeof(float), st);

            /* EXPERT-MAJOR dispatch (perf): route every token first (softmax/top-NU/renorm),
             * build the per-expert token-list, then dequant each HIT expert ONCE per layer and
             * batch all its tokens through one gate_up GEMM (n routing tokens) + GeGLU + down
             * GEMM + per-token weighted accumulate. This collapses up to n_tok*NU per-(token,slot)
             * dequants down to <= NE unique-expert dequants per layer (the binding cost is the
             * Q4B host dequant). Arithmetic is identical to the per-token path (same softmax,
             * same renorm, same expert math) — only the dequant/GEMM batching changes. */
            const int gu_rows = FFx * 2;
            /* per-token routing tables (NU entries each) */
            /* N6: calloc (not malloc) so the prompt region [0,P) the fast path never fills
             * holds a deterministic 0, not stale heap bytes that could be read as an index. */
            int   *rt_idx = (int   *)calloc((size_t)n_tok * NU, sizeof(int));
            float *rt_wt  = (float *)calloc((size_t)n_tok * NU, sizeof(float));
            /* per-expert token membership: et_cnt[e], et_tok[e][*], et_w[e][*] (flattened) */
            int   *et_cnt = (int   *)calloc((size_t)NE, sizeof(int));
            int   *et_tok = (int   *)calloc((size_t)n_tok * NU, sizeof(int));   /* worst case */
            float *et_w   = (float *)calloc((size_t)n_tok * NU, sizeof(float));
            int   *et_off = (int   *)calloc((size_t)(NE + 1), sizeof(int));
            if (!rt_idx || !rt_wt || !et_cnt || !et_tok || !et_w || !et_off) {
                free(rt_idx); free(rt_wt); free(et_cnt); free(et_tok); free(et_w); free(et_off);
                free(h_rlog); sp_set_error("dg routing tables OOM"); goto layerfail; }
            for (int t = (pkv_fast ? P : 0); t < n_tok; t++) {   /* N6: canvas tokens only */
                const float *lt = h_rlog + (size_t)t * NE;
                float mx = lt[0]; for (int i = 1; i < NE; i++) if (lt[i] > mx) mx = lt[i];
                double se = 0.0; for (int i = 0; i < NE; i++) { h_logits[i] = expf(lt[i]-mx); se += h_logits[i]; }
                for (int i = 0; i < NE; i++) h_logits[i] = (float)(h_logits[i]/se);
                char used[256] = {0};
                float wsum = 0.0f;
                int *ti = rt_idx + (size_t)t * NU; float *tw = rt_wt + (size_t)t * NU;
                for (int k = 0; k < NU; k++) {
                    int best = -1; float bv = -1.0f;
                    for (int i = 0; i < NE; i++) if (!used[i] && h_logits[i] > bv) { bv = h_logits[i]; best = i; }
                    if (best < 0) {            /* degenerate (NaN/Inf logits): pick first unused, weight 0 -- prevents used[-1]/et_cnt[-1] host OOB */
                        for (int i = 0; i < NE; i++) if (!used[i]) { best = i; break; }
                        if (best < 0) best = 0; bv = 0.0f; dg_nan_routes++;
                    }
                    used[best] = 1; ti[k] = best; tw[k] = bv; wsum += bv;
                }
                if (wsum > 0.0f) { for (int k = 0; k < NU; k++) { tw[k] /= wsum; et_cnt[ti[k]]++; } }
                else { for (int k = 0; k < NU; k++) { tw[k] = 1.0f/(float)NU; et_cnt[ti[k]]++; } }
            }
            /* prefix-sum offsets, then fill et_tok/et_w */
            for (int e = 0; e < NE; e++) et_off[e+1] = et_off[e] + et_cnt[e];
            { int *cur = (int *)calloc((size_t)NE, sizeof(int));
              if (!cur) { free(rt_idx); free(rt_wt); free(et_cnt); free(et_tok); free(et_w); free(et_off); free(h_rlog); sp_set_error("dg cur OOM"); goto layerfail; }
              for (int t = (pkv_fast ? P : 0); t < n_tok; t++) {   /* N6: canvas tokens only */
                  const int *ti = rt_idx + (size_t)t * NU; const float *tw = rt_wt + (size_t)t * NU;
                  for (int k = 0; k < NU; k++) { int e = ti[k]; int pos = et_off[e] + cur[e]++; et_tok[pos] = t; et_w[pos] = tw[k]; }
              }
              free(cur);
            }

            cudaMemsetAsync(dmoe, 0, nE*sizeof(float), st);

            /* batched per-expert scratch: hold up to n_tok columns (rare; usually << n_tok) */
            float *d_guw=NULL, *d_dnw=NULL, *d_xb=NULL, *d_gub=NULL, *d_hb=NULL, *d_deb=NULL;
            if (cudaMalloc(&d_xb,  (size_t)E   * n_tok * sizeof(float)) != cudaSuccess ||
                cudaMalloc(&d_gub, (size_t)gu_rows * n_tok * sizeof(float)) != cudaSuccess ||
                cudaMalloc(&d_hb,  (size_t)FFx * n_tok * sizeof(float)) != cudaSuccess ||
                cudaMalloc(&d_deb, (size_t)E   * n_tok * sizeof(float)) != cudaSuccess) {
                free(rt_idx); free(rt_wt); free(et_cnt); free(et_tok); free(et_w); free(et_off);
                free(h_rlog); if(d_xb)cudaFree(d_xb); if(d_gub)cudaFree(d_gub); if(d_hb)cudaFree(d_hb);
                sp_set_error("dg batched expert scratch OOM"); goto layerfail; }
            int dbg_ecount = 0;
            DGDBG("layer %d: MoE routing+scratch done, entering expert loop (n_tok=%d pkv_fast=%d)", L, n_tok, pkv_fast);
                if (dg_async_enabled() && !getenv("SP_DG_ASYNC_NOGU")){ int _ef=-1; for(int _e=0;_e<NE;_e++){ if(et_cnt[_e]>0){_ef=_e;break;} } if(_ef>=0){ snprintf(nm,sizeof nm,"blk.%d.ffn_gate_up_exps.weight",L); dg_prefetch_rows(m,nm,_ef*gu_rows,gu_rows,E,0); } }
            for (int e = 0; e < NE; e++) {
                int cnt = et_cnt[e];
                if (cnt == 0) continue;
                if (getenv("SP_DG_TRACE") && dbg_ecount < 4) { DGDBG("layer %d: expert %d cnt=%d dequant+gemm", L, e, cnt); dbg_ecount++; }
                /* gather this expert's token hidden columns into d_xb [E x cnt] (row-major n_tok) */
                for (int j = 0; j < cnt; j++) {
                    int t = et_tok[et_off[e] + j];
                    cudaMemcpyAsync(d_xb + (size_t)j*E, dnx + (size_t)t*E, (size_t)E*sizeof(float), cudaMemcpyDeviceToDevice, st);
                }
                DGDBG("L%d e%d: gather done cnt=%d (gate_up next)", L, e, cnt);
                /* gate_up: packed dp4a slice GEMV (SP_DG_PACKED) or f32 upload+gemm  [gu_rows x cnt] */
                snprintf(nm, sizeof nm, "blk.%d.ffn_gate_up_exps.weight", L);
                { int pr = dg_packed_enabled() ? dg_gemm_packed_rows(m, nm, e*gu_rows, gu_rows, E, d_xb, d_gub, cnt, cb, st) : 2;
                  if (pr == 1) { free(rt_idx); free(rt_wt); free(et_cnt); free(et_tok); free(et_w); free(et_off); free(h_rlog); cudaFree(d_xb); cudaFree(d_gub); cudaFree(d_hb); cudaFree(d_deb); goto layerfail; }
                  if (pr != 0) { d_guw = dg_upload_arena_rows(m, nm, e*gu_rows, gu_rows, E);
                    if (!d_guw) { free(rt_idx); free(rt_wt); free(et_cnt); free(et_tok); free(et_w); free(et_off); free(h_rlog); cudaFree(d_xb); cudaFree(d_gub); cudaFree(d_hb); cudaFree(d_deb); goto layerfail; }
                    if (gemm(cb, d_guw, d_xb, d_gub, cnt, E, gu_rows)) { cudaFree(d_guw); free(rt_idx); free(rt_wt); free(et_cnt); free(et_tok); free(et_w); free(et_off); free(h_rlog); cudaFree(d_xb); cudaFree(d_gub); cudaFree(d_hb); cudaFree(d_deb); goto layerfail; }
                    cudaFree(d_guw); d_guw = NULL; } }
                DGDBG("L%d e%d: gate_up gemm done (geglu+down next)", L, e);
                if (dg_async_enabled()){ snprintf(nm,sizeof nm,"blk.%d.ffn_down_exps.weight",L); dg_prefetch_rows(m,nm,e*E,E,FFx,1); }
                /* GeGLU per column: h[col*FFx + i] = gelu(gu[col*gu_rows+i]) * gu[col*gu_rows+FFx+i] */
                { size_t total = (size_t)FFx * cnt;
                  dg_k_geglu_batched<<<(unsigned)((total+255)/256), 256, 0, st>>>(d_gub, d_hb, FFx, gu_rows, cnt); }
                /* down: packed dp4a slice GEMV or f32 upload+gemm  (h -> de [E x cnt]) */
                snprintf(nm, sizeof nm, "blk.%d.ffn_down_exps.weight", L);
                { int pr = dg_packed_enabled() ? dg_gemm_packed_rows(m, nm, e*E, E, FFx, d_hb, d_deb, cnt, cb, st) : 2;
                  if (pr == 1) { free(rt_idx); free(rt_wt); free(et_cnt); free(et_tok); free(et_w); free(et_off); free(h_rlog); cudaFree(d_xb); cudaFree(d_gub); cudaFree(d_hb); cudaFree(d_deb); goto layerfail; }
                  if (pr != 0) { d_dnw = dg_upload_arena_rows(m, nm, e*E, E, FFx);
                    if (!d_dnw) { free(rt_idx); free(rt_wt); free(et_cnt); free(et_tok); free(et_w); free(et_off); free(h_rlog); cudaFree(d_xb); cudaFree(d_gub); cudaFree(d_hb); cudaFree(d_deb); goto layerfail; }
                    if (gemm(cb, d_dnw, d_hb, d_deb, cnt, FFx, E)) { cudaFree(d_dnw); free(rt_idx); free(rt_wt); free(et_cnt); free(et_tok); free(et_w); free(et_off); free(h_rlog); cudaFree(d_xb); cudaFree(d_gub); cudaFree(d_hb); cudaFree(d_deb); goto layerfail; }
                    cudaFree(d_dnw); d_dnw = NULL; } }
                DGDBG("L%d e%d: down gemm done (axpy next)", L, e);
                if (dg_async_enabled() && !getenv("SP_DG_ASYNC_NOGU")){ int _en=-1; for(int _e2=e+1;_e2<NE;_e2++){ if(et_cnt[_e2]>0){_en=_e2;break;} } if(_en>=0){ snprintf(nm,sizeof nm,"blk.%d.ffn_gate_up_exps.weight",L); dg_prefetch_rows(m,nm,_en*gu_rows,gu_rows,E,0); } }
                /* weighted accumulate each column back into its token's dmoe row */
                for (int j = 0; j < cnt; j++) {
                    int t = et_tok[et_off[e] + j]; float w = et_w[et_off[e] + j];
                    dg_k_axpy<<<(unsigned)((E+255)/256), 256, 0, st>>>(dmoe + (size_t)t*E, d_deb + (size_t)j*E, w, E);
                }
                cudaFree(d_guw); d_guw = NULL; cudaFree(d_dnw); d_dnw = NULL;
            }
            if (getenv("SP_DG_MOECHK")) {  /* deterministic MoE-output oracle for async parity (default-off) */
                cudaStreamSynchronize(st);
                float *hck=(float*)malloc((size_t)nE*sizeof(float));
                if (hck && cudaMemcpy(hck,dmoe,(size_t)nE*sizeof(float),cudaMemcpyDeviceToHost)==cudaSuccess){
                    double s=0.0; unsigned int h=2166136261u;
                    for(size_t i=0;i<(size_t)nE;i++){ s+=(double)hck[i]; unsigned int b; memcpy(&b,&hck[i],4); h=(h^b)*16777619u; }
                    fprintf(stderr,"[MOECHK] L=%d nE=%d sum=%.6f hash=%08x\n",L,nE,s,h); fflush(stderr);
                }
                if(hck) free(hck);
            }
            cudaFree(d_xb); cudaFree(d_gub); cudaFree(d_hb); cudaFree(d_deb);
            free(rt_idx); free(rt_wt); free(et_cnt); free(et_tok); free(et_w); free(et_off);
            free(h_rlog);
        }
        /* post_ffw_norm_2(cur_moe) -> dmoe */
        { snprintf(nm, sizeof nm, "blk.%d.post_ffw_norm_2.weight", L);
          float *dn2 = dg_upload_norm(m, nm, E); if (!dn2) goto layerfail;
          k_rmsnorm<<<n_tok, 256, 0, st>>>(dmoe, dn2, E, eps, dnx);
          cudaMemcpyAsync(dmoe, dnx, nE*sizeof(float), cudaMemcpyDeviceToDevice, st);
          cudaFree(dn2); }
        /* combined = dmlp + dmoe -> dnx ; post_ffw_norm(combined) ; residual + attn_out(dx) */
        cudaMemcpyAsync(dnx, dmlp, nE*sizeof(float), cudaMemcpyDeviceToDevice, st);
        k_add<<<(unsigned)((nE+255)/256), 256, 0, st>>>(dnx, dmoe, nE);   /* dnx = cur_mlp + cur_moe */
        { snprintf(nm, sizeof nm, "blk.%d.post_ffw_norm.weight", L);
          float *dpf = dg_upload_norm(m, nm, E); if (!dpf) goto layerfail;
          k_rmsnorm<<<n_tok, 256, 0, st>>>(dnx, dpf, E, eps, dao);   /* dao reused as scratch [n_tok*E<=n_tok*QDmax] */
          cudaFree(dpf); }
        /* dao now = post_ffw_norm(combined); add the attn_out residual (dx) -> ffn block output */
        k_add<<<(unsigned)((nE+255)/256), 256, 0, st>>>(dao, dx, nE);
        cudaMemcpyAsync(dx, dao, nE*sizeof(float), cudaMemcpyDeviceToDevice, st);   /* dx = layer FFN output */

        /* ── region-aware per-layer scalar: prompt * enc_out_scale, canvas * out_scale ── */
        {
            snprintf(nm, sizeof nm, "blk.%d.enc_layer_output_scale.weight", L);
            const float *encH = dg_norm_host(m, nm);
            snprintf(nm, sizeof nm, "blk.%d.layer_output_scale.weight", L);
            const float *decH = dg_norm_host(m, nm);
            if (!encH || !decH) { sp_set_error("dg: layer scalar missing"); goto layerfail; }
            float *dEnc=NULL, *dDec=NULL;
            if (cudaMalloc(&dEnc, sizeof(float)) != cudaSuccess ||
                cudaMalloc(&dDec, sizeof(float)) != cudaSuccess) { sp_set_error("dg scalar OOM"); if(dEnc)cudaFree(dEnc); goto layerfail; }
            cudaMemcpyAsync(dEnc, encH, sizeof(float), cudaMemcpyHostToDevice, st);
            cudaMemcpyAsync(dDec, decH, sizeof(float), cudaMemcpyHostToDevice, st);
            k_region_scale<<<(unsigned)((nE+255)/256), 256, 0, st>>>(dx, E, P, n_tok, dEnc, dDec);
            cudaStreamSynchronize(st);
            cudaFree(dEnc); cudaFree(dDec);
        }
        if (d_ffac) { cudaFree(d_ffac); d_ffac = NULL; }
        DGDBG("layer %d done", L);
        continue;
    layerfail:
        if (dW) cudaFree(dW); if (dWk) cudaFree(dWk); if (dWv) cudaFree(dWv);
        if (dWo) cudaFree(dWo); if (dNw) cudaFree(dNw); if (d_ffac) cudaFree(d_ffac);
        goto done;
    }

    if (getenv("SP_DG_TRACE")) { fprintf(stderr, "[dgtrace] forward done: n_tok=%d P=%d pkv_fast=%d dg_nan_routes=%ld\n", n_tok, P, pkv_fast, dg_nan_routes); fflush(stderr); }

    /* N6 prefix-KV proof report: on a repeat step, prompt K/V should be byte-identical (max|d|==0). */
    if (pkv_proof && pkv_hit && g_pkv_diff) {
        float md = -1.0f; cudaStreamSynchronize(st);
        cudaMemcpy(&md, g_pkv_diff, sizeof(float), cudaMemcpyDeviceToHost);
        fprintf(stderr, "[pkv-proof] HIT P=%d NL=%d  max|fresh-cached prompt K/V| = %.3e  (%s)\n",
                P, NL, md, md==0.0f ? "BYTE-IDENTICAL -> prompt tower INVARIANT" : "NONZERO -> coupling exists");
        fflush(stderr);
    } else if (pkv_proof) {
        fprintf(stderr, "[pkv-proof] MISS (cached prompt K/V for new query P=%d NL=%d)\n", P, NL); fflush(stderr);
    }

    /* ── final norm + tied head + softcap -> logits ── */
    {
        float *don = dg_upload_norm(m, "output_norm.weight", E); if (!don) goto done;
        k_rmsnorm<<<n_tok, 256, 0, st>>>(dx, don, E, eps, dnx);
        cudaFree(don);
        /* head: tied to token_embd (no output.weight) or untied output.weight; stream f32 */
        const char *head_name = c->tied_embedding ? m->token_embd->name : m->output->name;
        { int dgp = dg_packed_enabled() ? dg_gemm_packed(m, head_name, E, V, dnx, dlog, n_tok, cb, st) : 2;
          if (dgp == 1) goto done;
          if (dgp != 0) { float *dHead = dg_upload_arena_w(m, head_name, E, V); if (!dHead) goto done;
            if (gemm(cb, dHead, dnx, dlog, n_tok, E, V)) { cudaFree(dHead); goto done; }
            cudaFree(dHead); } }
        if (softcap > 0.0f) {
            size_t nl = (size_t)n_tok * V;
            k_softcap<<<(unsigned)((nl+255)/256), 256, 0, st>>>(dlog, nl, softcap);
        }
        if (cudaMemcpyAsync(out_logits, dlog, (size_t)n_tok*V*sizeof(float), cudaMemcpyDeviceToHost, st) != cudaSuccess) {
            sp_set_error("dg: logits D2H"); goto done; }
    }
    { cudaError_t e = cudaStreamSynchronize(st); if (e != cudaSuccess) { fail_cuda(e, "dg sync"); goto done; } }
    { cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) { fail_cuda(e, "dg kernel"); goto done; } }
    rc = 0;
done:
    if (hrows) free(hrows);
    if (h_logits) free(h_logits);
    if (h_router) free(h_router);
    if (dtoks) cudaFree(dtoks);
    if (dx) cudaFree(dx); if (dnx) cudaFree(dnx); if (dtmp) cudaFree(dtmp);
    if (dq) cudaFree(dq); if (dk) cudaFree(dk); if (dv) cudaFree(dv);
    if (dao) cudaFree(dao); if (dap) cudaFree(dap);
    if (dg) cudaFree(dg); if (dup) cudaFree(dup); if (ddn) cudaFree(ddn);
    if (dmlp) cudaFree(dmlp); if (dmoe) cudaFree(dmoe); if (dlog) cudaFree(dlog);
    if (Kst) { for (int L = 0; L < NL; L++) if (Kst[L]) cudaFree(Kst[L]); free(Kst); }
    if (Vst) { for (int L = 0; L < NL; L++) if (Vst[L]) cudaFree(Vst[L]); free(Vst); }
    #undef DGA
    if (cb) cublasDestroy(cb);
    return rc;
}

/* Public N1b entry (no self-conditioning) — byte-identical to the original single-
 * forward path (prev_logits=NULL). */
extern "C" int diffusion_gemma_forward_cuda(const qwen3_model *m, const int32_t *tokens,
                                            int n_tok, float *out_logits) {
    return dg_forward_impl(m, tokens, n_tok, out_logits, NULL, 1.0f);
}

/* Public N3 entry — self-conditioning forward. prev_logits = DEVICE [C x V] buffer
 * (canvas rows of the prior denoise step's RAW logits); NULL => no SC (== the N1b
 * path, byte-identical). sc_temp_inv = 1/(prior step temperature). The N4-full host
 * renoise loop passes the persistent device prev-logits buffer here each step. */
extern "C" int diffusion_gemma_forward_cuda_sc(const qwen3_model *m, const int32_t *tokens,
                                               int n_tok, float *out_logits,
                                               const float *prev_logits_dev, float sc_temp_inv) {
    return dg_forward_impl(m, tokens, n_tok, out_logits, prev_logits_dev, sc_temp_inv);
}

/* N4-full host-loop device-buffer helpers (opaque to the .c test harness). */
extern "C" void *dg_dev_alloc_f32(long n) {
    if (n <= 0) return NULL;
    void *p = NULL;
    if (cudaMalloc(&p, (size_t)n * sizeof(float)) != cudaSuccess) return NULL;
    cudaMemset(p, 0, (size_t)n * sizeof(float));  /* zero-init: prev_logits over-read region deterministic (SP_DG_ASYNC parity) */
    return p;
}
extern "C" int dg_dev_upload(void *dev, const float *host, long n) {
    if (!dev || !host || n <= 0) return 1;
    return (cudaMemcpy(dev, host, (size_t)n * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess) ? 0 : 1;
}
extern "C" void dg_dev_free(void *dev) { if (dev) cudaFree(dev); }


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
        if (d_bx_flag) { bx_rope_pair(&v[i], &v[i + half], bx_rope_theta(freq, (long long)(*dpos))); return; }
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
    dg_reservoir_free();
    dg_wcache_free(); dg_scratch_free(); dg_async_free();
}

/* ===================== NORTHSTAR GPU-1: qwen36 hybrid GEMV service =====================
 * CPU-orchestrated qwen36_step keeps GDN recurrence + MoE routing on host; the BIG
 * dense matmuls hit RESIDENT packed weights on device via the proven dp4a GEMV
 * (upload_packed + gemv_w_packed — the exact 43.9x diffusion-campaign machinery).
 * Name-keyed registry; per call: H2D x (in*4B), GEMV, D2H y (out*4B). Registered
 * into the math-core through sp_matmul_register_ext (unregistered = null floor). */
typedef struct { char name[160]; DevTensor dt; } q36g_ent;
typedef struct { char tag[96]; DevTensor g, u, d; int NE, FF, E; } q36moe_ent_fwd;
typedef struct q36gpu {
    cudaStream_t st;
    q36g_ent *e; int n, cap;
    float *dX, *dY; int xcap, ycap;
    signed char *dqx; float *dsx; int qcap;
    /* GPU-2 resident-expert registry + scratch */
    q36moe_ent_fwd *moes; int mn, mcap;
    float *dmoe; int moe_ff, moe_e;
    /* GPU-3 streaming slab (rump layers: ship the SELECTED experts per token) */
    unsigned char *slab; size_t slab_cap;
    unsigned long long *hoff; int hoff_cap;   /* host scratch: adjusted row_off */
} q36gpu;

extern "C" void *sp_q36gpu_new(void) {
    q36gpu *h = (q36gpu *)calloc(1, sizeof(q36gpu));
    if (!h) return NULL;
    if (cudaStreamCreate(&h->st) != cudaSuccess) { free(h); return NULL; }
    return h;
}

extern "C" void sp_q36gpu_free(void *vh) {
    q36gpu *h = (q36gpu *)vh;
    if (!h) return;
    for (int i = 0; i < h->n; i++) free_devtensor(&h->e[i].dt);
    free(h->e);
    for (int i = 0; i < h->mn; i++) {
        free_devtensor(&h->moes[i].g);
        free_devtensor(&h->moes[i].u);
        free_devtensor(&h->moes[i].d);
    }
    free(h->moes);
    if (h->slab) cudaFree(h->slab);
    free(h->hoff);
    if (h->dmoe) cudaFree(h->dmoe);
    if (h->dX) cudaFree(h->dX);
    if (h->dY) cudaFree(h->dY);
    if (h->dqx) cudaFree(h->dqx);
    if (h->dsx) cudaFree(h->dsx);
    cudaStreamDestroy(h->st);
    free(h);
}

extern "C" int sp_q36gpu_upload(void *vh, const char *name, const sp_frob_packed_tensor *pt) {
    q36gpu *h = (q36gpu *)vh;
    if (!h || !name || !pt) return 1;
    if (h->n == h->cap) {
        int nc = h->cap ? h->cap * 2 : 64;
        q36g_ent *ne = (q36g_ent *)realloc(h->e, (size_t)nc * sizeof(q36g_ent));
        if (!ne) return 1;
        h->e = ne; h->cap = nc;
    }
    q36g_ent *E = &h->e[h->n];
    memset(E, 0, sizeof(*E));
    strncpy(E->name, name, sizeof(E->name) - 1);
    if (upload_packed(pt, &E->dt)) { free_devtensor(&E->dt); return 1; }
    /* grow the shared activation/result/act-quant buffers */
    int in = E->dt.in, out = E->dt.out;
    if (in > h->xcap) {
        if (h->dX) cudaFree(h->dX);
        if (cudaMalloc(&h->dX, (size_t)in * sizeof(float)) != cudaSuccess) return 1;
        h->xcap = in;
    }
    if (out > h->ycap) {
        if (h->dY) cudaFree(h->dY);
        if (cudaMalloc(&h->dY, (size_t)out * sizeof(float)) != cudaSuccess) return 1;
        h->ycap = out;
    }
    if (in + 64 > h->qcap) {
        if (h->dqx) cudaFree(h->dqx);
        if (h->dsx) cudaFree(h->dsx);
        if (cudaMalloc(&h->dqx, (size_t)(in + 64)) != cudaSuccess) return 1;
        if (cudaMalloc(&h->dsx, ((size_t)in / 32 + 8) * sizeof(float)) != cudaSuccess) return 1;
        h->qcap = in + 64;
    }
    h->n++;
    return 0;
}

/* ── GPU-2: resident-EXPERT MoE service ──────────────────────────────────────────
 * Per layer, the three rank-3 expert tensors (gate/up/down, 256 experts each) are
 * uploaded whole; an expert's matmul is a VIEW DevTensor over the parent — row_off
 * is absolute into codes, so the view just offsets the row_off/row_scale/row_prec/
 * bscale pointers and sets out to the slice height. One device flow per MoE call:
 * H2D x once -> per expert: gate GEMV + up GEMV + k_silu_mul + down GEMV +
 * dg_k_axpy(weighted accumulate) all ASYNC on the stream -> ONE sync -> D2H y. */
typedef q36moe_ent_fwd q36moe_ent;

static DevTensor q36_expert_view(const DevTensor *p, int e, int rows) {
    DevTensor v = *p;
    v.out       = rows;
    v.row_off   = p->row_off  + (size_t)e * rows;
    v.row_prec  = p->row_prec + (size_t)e * rows;
    if (p->row_scale) v.row_scale = p->row_scale + (size_t)e * rows;
    if (p->bscale)    v.bscale    = p->bscale + (size_t)e * rows * (size_t)p->bs_nblk;
    return v;
}

extern "C" int sp_q36gpu_upload_experts(void *vh, const char *tag,
                                        const sp_frob_packed_tensor *ptg,
                                        const sp_frob_packed_tensor *ptu,
                                        const sp_frob_packed_tensor *ptd,
                                        int NE, int FF, int E) {
    q36gpu *h = (q36gpu *)vh;
    if (!h || !tag || !ptg || !ptu || !ptd) return 1;
    if (h->mn == h->mcap) {
        int nc = h->mcap ? h->mcap * 2 : 64;
        q36moe_ent *ne = (q36moe_ent *)realloc(h->moes, (size_t)nc * sizeof(q36moe_ent));
        if (!ne) return 1;
        h->moes = ne; h->mcap = nc;
    }
    q36moe_ent *M = &h->moes[h->mn];
    memset(M, 0, sizeof(*M));
    strncpy(M->tag, tag, sizeof(M->tag) - 1);
    M->NE = NE; M->FF = FF; M->E = E;
    if (upload_packed(ptg, &M->g) || upload_packed(ptu, &M->u) || upload_packed(ptd, &M->d)) {
        free_devtensor(&M->g); free_devtensor(&M->u); free_devtensor(&M->d);
        return 1;
    }
    h->mn++;
    /* ensure MoE scratch (dX reused for x[E]; add dG/dU/dH [FF], dDe/dYo [E]) */
    if (!h->dmoe) {
        if (cudaMalloc(&h->dmoe, ((size_t)FF * 3 + (size_t)E * 2) * sizeof(float)) != cudaSuccess)
            return 1;
        h->moe_ff = FF; h->moe_e = E;
    }
    if (E > h->xcap) {
        if (h->dX) cudaFree(h->dX);
        if (cudaMalloc(&h->dX, (size_t)E * sizeof(float)) != cudaSuccess) return 1;
        h->xcap = E;
    }
    return 0;
}

/* MoE ext contract: 0 = handled (y filled), nonzero = not-mine (layer not resident). */
extern "C" int sp_q36gpu_moe(void *vh, const char *tag, const int *idx, const float *wt,
                             int NU, const float *x, int E, int FF, float *y) {
    q36gpu *h = (q36gpu *)vh;
    if (!h || !h->moes) return -1;
    q36moe_ent *M = NULL;
    for (int i = 0; i < h->mn; i++)
        if (strcmp(h->moes[i].tag, tag) == 0) { M = &h->moes[i]; break; }
    if (!M || M->FF != FF || M->E != E) return -1;
    float *dG  = h->dmoe;               /* [FF] gate   */
    float *dU  = dG + FF;               /* [FF] up     */
    float *dH  = dU + FF;               /* [FF] silu*u (in-place on dG actually) */
    float *dDe = dH + FF;               /* [E]  down   */
    float *dYo = dDe + E;               /* [E]  accum  */
    (void)dH;
    if (cudaMemcpyAsync(h->dX, x, (size_t)E * sizeof(float), cudaMemcpyHostToDevice, h->st)
        != cudaSuccess) return -1;
    if (cudaMemsetAsync(dYo, 0, (size_t)E * sizeof(float), h->st) != cudaSuccess) return -1;
    for (int k = 0; k < NU; k++) {
        DevTensor vg = q36_expert_view(&M->g, idx[k], FF);
        DevTensor vu = q36_expert_view(&M->u, idx[k], FF);
        DevTensor vd = q36_expert_view(&M->d, idx[k], E);
        if (!gemv_w_packed(h->st, &vg, h->dX, dG, h->dqx, h->dsx)) return -1;
        if (!gemv_w_packed(h->st, &vu, h->dX, dU, h->dqx, h->dsx)) return -1;
        k_silu_mul<<<(unsigned)((FF + 255) / 256), 256, 0, h->st>>>(dG, dU, (size_t)FF);
        if (!gemv_w_packed(h->st, &vd, dG, dDe, h->dqx, h->dsx)) return -1;
        dg_k_axpy<<<(unsigned)((E + 255) / 256), 256, 0, h->st>>>(dYo, dDe, wt[k], E);
    }
    if (cudaMemcpyAsync(y, dYo, (size_t)E * sizeof(float), cudaMemcpyDeviceToHost, h->st)
        != cudaSuccess) return -1;
    return cudaStreamSynchronize(h->st) == cudaSuccess ? 0 : -1;
}

/* ── GPU-3: STREAMED experts for non-resident layers ─────────────────────────────
 * Ship ONLY the selected NU experts' packed slices (codes + adjusted row_off +
 * row_prec + bscale/row_scale) into a preallocated device slab, then run the same
 * per-expert dp4a flow. Slices are CONTIGUOUS byte ranges (row_off is ascending
 * absolute), so each tensor slice is ONE H2D copy plus small metadata copies —
 * ~10MB/layer/token total, all async on the handle stream, one sync per layer. */
static int q36_slab_ensure(q36gpu *h, size_t need) {
    if (need <= h->slab_cap) return 0;
    if (h->slab) cudaFree(h->slab);
    if (cudaMalloc(&h->slab, need) != cudaSuccess) { h->slab = NULL; h->slab_cap = 0; return 1; }
    h->slab_cap = need;
    return 0;
}

static int q36_stream_slice(q36gpu *h, const sp_frob_packed_tensor *pt, int r0, int rows,
                            unsigned char **bump, DevTensor *dt) {
    memset(dt, 0, sizeof(*dt));
    dt->in = pt->cols; dt->out = rows;
    size_t cbase = pt->row_off[r0];
    size_t cend  = ((uint32_t)(r0 + rows) < pt->rows) ? pt->row_off[r0 + rows] : pt->codes_bytes;
    size_t cbytes = cend - cbase;
    /* adjusted row_off (slab-relative) via host scratch */
    if (rows > h->hoff_cap) {
        free(h->hoff);
        h->hoff = (unsigned long long *)malloc((size_t)rows * sizeof(unsigned long long));
        if (!h->hoff) { h->hoff_cap = 0; return 1; }
        h->hoff_cap = rows;
    }
    for (int i = 0; i < rows; i++)
        h->hoff[i] = (unsigned long long)(pt->row_off[r0 + i] - cbase);
    unsigned char *p = *bump;
    #define BUMP(nbytes) (p += ((nbytes) + 255) & ~(size_t)255)
    dt->codes = p;
    if (cudaMemcpyAsync(p, pt->codes + cbase, cbytes, cudaMemcpyHostToDevice, h->st)
        != cudaSuccess) return 1;
    BUMP(cbytes);
    dt->row_off = (unsigned long long *)p;
    if (cudaMemcpyAsync(p, h->hoff, (size_t)rows * 8, cudaMemcpyHostToDevice, h->st)
        != cudaSuccess) return 1;
    BUMP((size_t)rows * 8);
    dt->row_prec = p;
    if (cudaMemcpyAsync(p, pt->row_prec + r0, (size_t)rows, cudaMemcpyHostToDevice, h->st)
        != cudaSuccess) return 1;
    BUMP((size_t)rows);
    if (pt->bscale) {
        dt->bs_nblk = pt->bs_nblk;
        dt->bscale = (unsigned short *)p;
        size_t bs = (size_t)rows * pt->bs_nblk * 2;
        if (cudaMemcpyAsync(p, pt->bscale + (size_t)r0 * pt->bs_nblk, bs,
                            cudaMemcpyHostToDevice, h->st) != cudaSuccess) return 1;
        BUMP(bs);
    } else if (pt->row_scale) {
        dt->row_scale = (float *)p;
        if (cudaMemcpyAsync(p, pt->row_scale + r0, (size_t)rows * 4,
                            cudaMemcpyHostToDevice, h->st) != cudaSuccess) return 1;
        BUMP((size_t)rows * 4);
    } else return 1;
    #undef BUMP
    /* per-tensor precision for the dp4a GEMV: uniform-row check on the slice */
    dt->prec = (int)pt->row_prec[r0];
    for (int i = 1; i < rows; i++)
        if (pt->row_prec[r0 + i] != pt->row_prec[r0]) { dt->prec = 0; break; }
    *bump = p;
    return 0;
}

extern "C" int sp_q36gpu_moe_stream(void *vh,
                                    const sp_frob_packed_tensor *ptg,
                                    const sp_frob_packed_tensor *ptu,
                                    const sp_frob_packed_tensor *ptd,
                                    const int *idx, const float *wt, int NU,
                                    const float *x, int E, int FF, float *y) {
    q36gpu *h = (q36gpu *)vh;
    if (!h || !ptg || !ptu || !ptd) return -1;
    /* scratch (shared with the resident path; ensure when no layer was resident) */
    if (!h->dmoe) {
        if (cudaMalloc(&h->dmoe, ((size_t)FF * 3 + (size_t)E * 2) * sizeof(float)) != cudaSuccess)
            return -1;
        h->moe_ff = FF; h->moe_e = E;
    }
    if (E > h->xcap) {
        if (h->dX) cudaFree(h->dX);
        if (cudaMalloc(&h->dX, (size_t)E * sizeof(float)) != cudaSuccess) return -1;
        h->xcap = E;
    }
    /* slab: worst case per expert = 3 slices' codes (<= rows*cols bytes @Q8) + meta */
    size_t per_e = ((size_t)FF * E + (size_t)FF * E + (size_t)E * FF) /* codes worst */
                 + 3 * ((size_t)(E > FF ? E : FF) * 8 + 4096 + (size_t)(E > FF ? E : FF) * 4 + 8192);
    if (q36_slab_ensure(h, (size_t)NU * per_e + 65536)) return -1;
    float *dG  = h->dmoe;
    float *dU  = dG + FF;
    float *dDe = dU + FF * 2;      /* skip dH slot */
    float *dYo = dDe + E;
    if (cudaMemcpyAsync(h->dX, x, (size_t)E * sizeof(float), cudaMemcpyHostToDevice, h->st)
        != cudaSuccess) return -1;
    if (cudaMemsetAsync(dYo, 0, (size_t)E * sizeof(float), h->st) != cudaSuccess) return -1;
    unsigned char *bump = h->slab;
    for (int k = 0; k < NU; k++) {
        DevTensor vg, vu, vd;
        if (q36_stream_slice(h, ptg, idx[k] * FF, FF, &bump, &vg)) return -1;
        if (q36_stream_slice(h, ptu, idx[k] * FF, FF, &bump, &vu)) return -1;
        if (q36_stream_slice(h, ptd, idx[k] * E,  E,  &bump, &vd)) return -1;
        if (!gemv_w_packed(h->st, &vg, h->dX, dG, h->dqx, h->dsx)) return -1;
        if (!gemv_w_packed(h->st, &vu, h->dX, dU, h->dqx, h->dsx)) return -1;
        k_silu_mul<<<(unsigned)((FF + 255) / 256), 256, 0, h->st>>>(dG, dU, (size_t)FF);
        if (!gemv_w_packed(h->st, &vd, dG, dDe, h->dqx, h->dsx)) return -1;
        dg_k_axpy<<<(unsigned)((E + 255) / 256), 256, 0, h->st>>>(dYo, dDe, wt[k], E);
    }
    if (cudaMemcpyAsync(y, dYo, (size_t)E * sizeof(float), cudaMemcpyDeviceToHost, h->st)
        != cudaSuccess) return -1;
    return cudaStreamSynchronize(h->st) == cudaSuccess ? 0 : -1;
}

/* sp_matmul_ext contract: 0 = handled (Y filled), nonzero = not-mine -> caller falls
 * through to the CPU path. n_tok must be 1 (the decode step). */
extern "C" int sp_q36gpu_matmul(void *vh, const char *name, const float *X,
                                int n_tok, int in, int out, float *Y) {
    q36gpu *h = (q36gpu *)vh;
    if (!h || n_tok != 1) return -1;
    q36g_ent *E = NULL;
    for (int i = 0; i < h->n; i++)
        if (strcmp(h->e[i].name, name) == 0) { E = &h->e[i]; break; }
    if (!E || E->dt.in != in || E->dt.out != out) return -1;
    if (cudaMemcpyAsync(h->dX, X, (size_t)in * sizeof(float),
                        cudaMemcpyHostToDevice, h->st) != cudaSuccess) return -1;
    if (!gemv_w_packed(h->st, &E->dt, h->dX, h->dY, h->dqx, h->dsx)) return -1;
    if (cudaMemcpyAsync(Y, h->dY, (size_t)out * sizeof(float),
                        cudaMemcpyDeviceToHost, h->st) != cudaSuccess) return -1;
    return cudaStreamSynchronize(h->st) == cudaSuccess ? 0 : -1;
}
