/* cuda_gemma3.cu — Gemma3 forward pass on CUDA (Phase 2-CU, CU.1 f32 + CU.2 Q8/Q4).
 *
 * Mirrors the CPU gemma3_forward (src/forward/gemma3.c) op-for-op so the
 * CUDA-vs-CPU output gate (§8.3, <=1e-3 rel) and T_FRO_4 hold. cuBLAS SGEMM for
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

__global__ void k_gelu_mul(float *g, const float *up, size_t n) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = g[idx];
        const float k = 0.7978845608028654f;
        float th = tanhf(k * (x + 0.044715f * x * x * x));
        g[idx] = 0.5f * x * (1.0f + th) * up[idx];
    }
}

__global__ void k_add(float *x, const float *y, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

/* Decode a per-row Frobenius packed weight (Q8 or per-row-promoted Q4) to f32,
 * row-major [rows x cols]. Mirrors matmul_arena: w[j][i] = code * row_scale[j]/qmax
 * (qmax 127 for a Q8 row, 7 for a Q4 row). Q4 codes are two-per-byte, low nibble =
 * even index, sign-extended. grid=(ceil(cols/256), rows). */
__global__ void k_dequant_arena(const unsigned char *codes, const unsigned long long *row_off,
                                const float *row_scale, const unsigned char *row_prec,
                                int rows, int cols, float *out) {
    int j = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
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
    float *embd;        /* [V*E] tied token-embd / LM head, f32 */
    float *out_norm;    /* [E] */
    size_t scratch_n;   /* max packed weight elem count (0 if no arena) */
    DevTensor *Wq, *Wk, *Wv, *Wo, *Wgate, *Wup, *Wdown;
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

    CudaWeights z = {}; *w = z;
    w->key = m; w->L = L;

    cudaError_t e = cudaStreamCreate(&w->stream);
    if (e != cudaSuccess) return fail_cuda(e, "cudaStreamCreate");
    CB(cublasCreate(&w->cublas), "cublasCreate");
    CB(cublasSetStream(w->cublas, w->stream), "cublasSetStream");
    CB(cublasSetMathMode(w->cublas, CUBLAS_DEFAULT_MATH), "cublasSetMathMode");

    /* tied embedding / head: f32 (from the arena if it was packed for release;
     * else GGUF). Used by both the embed gather and the head SGEMM. */
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

    ALLOC_DT(Wq); ALLOC_DT(Wk); ALLOC_DT(Wv); ALLOC_DT(Wo);
    ALLOC_DT(Wgate); ALLOC_DT(Wup); ALLOC_DT(Wdown);
    ALLOC_NM(attn_norm); ALLOC_NM(ffn_norm); ALLOC_NM(q_norm); ALLOC_NM(k_norm);
    ALLOC_NM(post_attn); ALLOC_NM(post_ffw);

    for (int Li = 0; Li < L; Li++) {
        const qwen3_layer *ly = &m->layers[Li];
        BUILDW(Wq, attn_q, E, QD);   BUILDW(Wk, attn_k, E, KVD);  BUILDW(Wv, attn_v, E, KVD);
        BUILDW(Wo, attn_output, QD, E);
        BUILDW(Wgate, ffn_gate, E, FF); BUILDW(Wup, ffn_up, E, FF); BUILDW(Wdown, ffn_down, FF, E);
        UPV(attn_norm, attn_norm, E);   UPV(ffn_norm, ffn_norm, E);
        UPV(q_norm, attn_q_norm, HD);   UPV(k_norm, attn_k_norm, HD);
        UPV(post_attn, post_attn_norm, E); UPV(post_ffw, post_ffw_norm, E);
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
    dim3 grid((W->in + 255) / 256, W->out);
    k_dequant_arena<<<grid, 256, 0, st>>>(W->codes, W->row_off, W->row_scale, W->row_prec,
                                          W->out, W->in, scratch);
    return gemm(h, scratch, dX, dY, n_tok, W->in, W->out);
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

extern "C" void sp_cuda_model_release(const qwen3_model *m) {
    if (g_w.key == m) free_weights(&g_w);
}
