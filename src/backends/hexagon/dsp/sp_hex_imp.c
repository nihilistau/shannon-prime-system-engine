/* sp_hex_imp.c — Phase 2-HX cDSP-side implementation (Hexagon V69 HTP).
 *
 * Recreated fresh. The S22U reference + SDK examples are structural reference
 * only — no code copied; the forward-pass logic comes from the engine
 * (gemma3.c / forward.c / cuda_forward.cu), recreated for HVX in HX.3.
 *
 * HX.2: open/close + ping (FastRPC wiring smoke). The skel (sp_hex_skel.c) is
 * generated from ../inc/sp_hex.idl by qaic and dispatches to these.
 *
 * V69 HVX rules for HX.3 (do NOT rediscover — see SESSION-STATE-lat-2-HX):
 *   - V69 has NO sf-result float multiply/add: the float multiply/add ALWAYS emit
 *     32-bit qfloat (qf32), per the V69 HVX Programmer's Reference (Multiply/Add
 *     single precision vector by vector, p.150/246). The sf family is NOT broken --
 *     sf inputs feed Q6_Vqf32_vmpy_VsfVsf directly. The mandated matmul shape is:
 *     sf inputs -> qf32 products -> qf32 accumulate (Q6_Vqf32_vadd_*) -> a single
 *     Q6_Vsf_equals_Vqf32 convert at the end. The retired "BROKEN, off 4-20 absolute"
 *     note was a qf32 result emitted without that final convert -- a missing
 *     conversion mislabeled a silicon defect. -mhvx-ieee-fp (dsp/CMakeLists.txt) is
 *     the codegen prerequisite the prior fixed-point HVX build never carried.
 *   - qurt_hvx_lock(QURT_HVX_MODE_128B) is thread-local; FastRPC runs the method
 *     on a worker thread — lock ONCE at the top of the forward method (whole-
 *     forward-on-DSP = one method), not in open.
 *   - 128-byte-align stack arrays fed to HVX; DSP malloc unreliable on unsigned
 *     PD — use stack / rpcmem.
 */
#include <stdlib.h>
#include <math.h>
#include "HAP_farf.h"
#include "sp_hex.h"
#include "../sp_hex_layout.h"   /* host<->DSP weight-blob contract */

#ifdef __HVX__   /* HX.3b f32 HVX matmul primitive (-mhvx-ieee-fp enables the float family) */
#include <hexagon_types.h>     /* HVX_Vector */
#include <hexagon_protos.h>    /* Q6_* HVX intrinsics */
#include "qurt_hvx.h"          /* qurt_hvx_lock/unlock + QURT_HVX_MODE_128B (gotcha #4) */
/* f32 dot on V69 HVX, the hardware-mandated float shape (see the header note):
 * sf inputs -> Q6_Vqf32_vmpy_VsfVsf products (V69 float multiply emits qf32) ->
 * qf32 lane accumulate -> a 5-step vror/qf32-add tree reduces the 32 lanes to lane
 * 0 -> one Q6_Vsf_equals_Vqf32 convert. The vector body runs whole 32-float
 * (128-byte) blocks; a scalar epilogue takes the n%32 tail. The aligned vector loads
 * assume 128-byte-aligned a,b -- true for page-aligned rpcmem tensor rows whose cols
 * is a multiple of 32 (the Gemma3/Qwen3 projection dims are). The lane-reduction
 * sums in a different order than the sequential scalar path, so it tracks the scalar
 * reference only to the 8.6.1 precision floor, never bit-for-bit. */
static float hx_dot_hvx(const float *a, const float *b, int n) {
    int nb = n & ~31;                 /* whole 32-float (128-byte) blocks */
    float sum = 0.0f;
    if (nb > 0) {
        /* first block initialises the qf32 accumulator (avoids relying on a
         * canonical "zero qf32" bit pattern). */
        HVX_Vector acc = Q6_Vqf32_vmpy_VsfVsf(*(const HVX_Vector *)a,
                                              *(const HVX_Vector *)b);
        for (int i = 32; i < nb; i += 32)
            acc = Q6_Vqf32_vadd_Vqf32Vqf32(
                      acc, Q6_Vqf32_vmpy_VsfVsf(*(const HVX_Vector *)(a + i),
                                                *(const HVX_Vector *)(b + i)));
        /* horizontal reduce qf32 lanes: rotate right by 64/32/16/8/4 bytes + add. */
        acc = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_V_vror_VR(acc, 64));
        acc = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_V_vror_VR(acc, 32));
        acc = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_V_vror_VR(acc, 16));
        acc = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_V_vror_VR(acc, 8));
        acc = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_V_vror_VR(acc, 4));
        float lanes[32] __attribute__((aligned(128)));
        *(HVX_Vector *)lanes = Q6_Vsf_equals_Vqf32(acc);   /* qf32 -> IEEE single */
        sum = lanes[0];
    }
    for (int i = nb; i < n; i++) sum += a[i] * b[i];       /* scalar tail */
    return sum;
}
/* Q8 dot: int8 codes (row) against f32 activations (x), same qf32 shape as
 * hx_dot_hvx with a tiny scalar int8->sf widen per 32-chunk into a 128-byte-aligned
 * tile (V69 exposes no in-vector int8->sf convert here; the widen is cheap next to
 * the HVX multiply-accumulate it feeds — the in-vector-widen / integer-vrmpy fast
 * path is the deferred acceleration). Returns the raw dot; caller applies the row
 * scale. The HVX unit is held by the enclosing method (sp_hex_forward), not here. */
static float hx_dot_q8_hvx(const signed char *row, const float *x, int n) {
    int nb = n & ~31;
    float sum = 0.0f;
    if (nb > 0) {
        float wf[32] __attribute__((aligned(128)));
        HVX_Vector acc; int first = 1;
        for (int i = 0; i < nb; i += 32) {
            for (int c = 0; c < 32; c++) wf[c] = (float)row[i + c];   /* scalar widen */
            HVX_Vector p = Q6_Vqf32_vmpy_VsfVsf(*(const HVX_Vector *)wf,
                                                *(const HVX_Vector *)(x + i));
            acc = first ? p : Q6_Vqf32_vadd_Vqf32Vqf32(acc, p); first = 0;
        }
        acc = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_V_vror_VR(acc, 64));
        acc = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_V_vror_VR(acc, 32));
        acc = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_V_vror_VR(acc, 16));
        acc = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_V_vror_VR(acc, 8));
        acc = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_V_vror_VR(acc, 4));
        float lanes[32] __attribute__((aligned(128)));
        *(HVX_Vector *)lanes = Q6_Vsf_equals_Vqf32(acc);
        sum = lanes[0];
    }
    for (int i = nb; i < n; i++) sum += (float)row[i] * x[i];   /* scalar tail */
    return sum;
}
#endif /* __HVX__ */

int sp_hex_open(const char *uri, remote_handle64 *h) {
    (void)uri;
    /* Opaque handle; the rpc layer does not inspect it. HX.3 hangs the
     * uploaded-weight table + scratch off this. */
    void *ctx = malloc(1);
    *h = (remote_handle64)ctx;
    return ctx ? 0 : -1;
}

int sp_hex_close(remote_handle64 h) {
    if (h) free((void *)h);
    return 0;
}

int sp_hex_ping(remote_handle64 h, int x, int *y) {
    (void)h;
    *y = x + 1;
    FARF(RUNTIME_HIGH, "sp_hex: ping x=%d -> y=%d", x, *y);
    return 0;
}

/* Standard reflected CRC-32 (IEEE / zlib polynomial 0xEDB88320), table-less.
 * Identical algorithm host-side (sp_hex_rt.c) — equal CRC iff equal bytes. */
static unsigned int sp_hex_crc32(const unsigned char *p, int n) {
    unsigned int crc = 0xFFFFFFFFu;
    for (int i = 0; i < n; i++) {
        crc ^= (unsigned int)p[i];
        for (int k = 0; k < 8; k++)
            crc = (crc >> 1) ^ (0xEDB88320u & (unsigned int)(-(int)(crc & 1u)));
    }
    return crc ^ 0xFFFFFFFFu;
}

/* HX.3a upload byte-exactness proof: CRC-32 the bytes the DSP received over
 * FastRPC. If the host's rpcmem registration size != the IDL length, the bridge
 * silently zero-fills and this CRC will not match the host's — the exact-alloc
 * trap, caught here before it can poison a forward pass. */
int sp_hex_upload_crc(remote_handle64 h, const unsigned char *data, int dataLen, int *crc) {
    (void)h;
    unsigned int c = sp_hex_crc32(data, dataLen);
    *crc = (int)c;
    FARF(RUNTIME_HIGH, "sp_hex: upload_crc len=%d crc=0x%08x", dataLen, c);
    return 0;
}

/* HX.3a: scalar f32 matmul on the cDSP — the core forward kernel (ggml weight
 * layout: y[j] = sum_i w[j*cols + i] * x[i]). Under __HVX__ each row runs the qf32
 * HVX dot (hx_dot_hvx, HX.3b); otherwise the scalar f32 path -- which the host A/B
 * still computes as the reference. The HVX lane-reduction reorders the sum, so the
 * cDSP-HVX result tracks the scalar host reference only to the 8.6.1 precision
 * floor, not bit-for-bit (the prior scalar-vs-host exact-zero match no longer holds
 * once HVX is on -- this is the expected, gated behaviour). */
int sp_hex_matmul_f32(remote_handle64 h, const float *w, int wLen, int rows, int cols,
                      const float *x, int xLen, float *y, int yLen) {
    (void)h; (void)wLen; (void)xLen;
#ifdef __HVX__
    /* Reserve the 128B HVX unit on this FastRPC worker thread (gotcha #4: the lock is
     * thread-local and FastRPC dispatches the method on a worker thread). Lock once
     * around the whole row loop, not per row. */
    qurt_hvx_lock(QURT_HVX_MODE_128B);
    for (int j = 0; j < rows && j < yLen; j++)
        y[j] = hx_dot_hvx(w + (long)j * cols, x, cols);
    qurt_hvx_unlock();
#else
    for (int j = 0; j < rows && j < yLen; j++) {
        const float *wr = w + (long)j * cols;
        float acc = 0.0f;
        for (int i = 0; i < cols; i++) acc += wr[i] * x[i];
        y[j] = acc;
    }
#endif
    FARF(RUNTIME_HIGH, "sp_hex: matmul_f32 rows=%d cols=%d", rows, cols);
    return 0;
}

/* ── HX.3a: the gemma3 transformer layers + final RMSNorm, scalar f32 on the cDSP.
 * Mirrors src/forward/gemma3.c op-for-op (recreated fresh); reads the Q8/f32
 * weight blob via sp_hex_layout.h. Embedding + tied head run host-side. ── */

static void hx_rmsnorm(const float *x, const float *w, int n, float eps, float *out) {
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    float scale = 1.0f / sqrtf((float)(ss / n) + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * scale * w[i];
}
static void hx_rmsnorm_head(float *v, const float *w, int d, float eps) {
    double ss = 0.0;
    for (int i = 0; i < d; i++) ss += (double)v[i] * v[i];
    float scale = 1.0f / sqrtf((float)(ss / d) + eps);
    for (int i = 0; i < d; i++) v[i] = v[i] * scale * w[i];
}
static void hx_rope_neox(float *v, int d, int p, float base) {
    int half = d / 2;
    for (int i = 0; i < half; i++) {
        float freq = powf(base, -2.0f * (float)i / (float)d);
        float th = (float)p * freq, c = cosf(th), s = sinf(th);
        float a = v[i], b = v[i + half];
        v[i] = a * c - b * s;
        v[i + half] = a * s + b * c;
    }
}
static float hx_gelu_tanh(float x) {
    const float k = 0.7978845608028654f;
    return 0.5f * x * (1.0f + tanhf(k * (x + 0.044715f * x * x * x)));
}
/* per-row Q8 matmul (ggml [out,in]); blob = int8 codes (padded) then f32 scales.
 * Y[t,j] = (sum_i code[j,i]*X[t,i]) * scale[j]/127 — the matmul_arena inline lift. */
static void hx_matmul_q8(const unsigned char *blk, int out, int in,
                         const float *X, int n_tok, float *Y) {
    const signed char *codes = (const signed char *)blk;
    const float *scales = (const float *)(blk + sp_hex_align((size_t)out * in));
    for (int j = 0; j < out; j++) {
        const signed char *row = codes + (size_t)j * in;
        float inv = scales[j] / 127.0f;
        for (int t = 0; t < n_tok; t++) {
            const float *x = X + (size_t)t * in;
#ifdef __HVX__
            Y[(size_t)t * out + j] = hx_dot_q8_hvx(row, x, in) * inv;
#else
            float acc = 0.0f;
            for (int i = 0; i < in; i++) acc += (float)row[i] * x[i];
            Y[(size_t)t * out + j] = acc * inv;
#endif
        }
    }
}
/* GQA causal/windowed softmax for one query head (matches kernels_attn_head). */
static void hx_attn_head(const float *qh, const float *KC, const float *VC,
                         int pos, int KVD, int kvh, int HD, float ascale, int win,
                         float *sc, float *out) {
    int s0 = (win >= 0 && pos - win + 1 > 0) ? pos - win + 1 : 0;
    float maxs = -INFINITY;
    for (int s = s0; s <= pos; s++) {
        const float *kh = KC + (size_t)s * KVD + (size_t)kvh * HD;
        float acc = 0.0f;
        for (int i = 0; i < HD; i++) acc += qh[i] * kh[i];
        float d = acc * ascale; sc[s] = d; if (d > maxs) maxs = d;
    }
    float sum = 0.0f;
    for (int s = s0; s <= pos; s++) { sc[s] = expf(sc[s] - maxs); sum += sc[s]; }
    float inv = 1.0f / sum;
    for (int i = 0; i < HD; i++) out[i] = 0.0f;
    for (int s = s0; s <= pos; s++) {
        float w = sc[s] * inv;
        const float *vh = VC + (size_t)s * KVD + (size_t)kvh * HD;
        for (int i = 0; i < HD; i++) out[i] += w * vh[i];
    }
}

int sp_hex_forward(remote_handle64 hdl, int n_layers, int n_embd, int n_ff, int head_dim,
                   int n_head, int n_head_kv, int sliding_window,
                   float eps, float rope_global, float rope_local,
                   int n_tok, const float *x, int xLen,
                   const unsigned char *weights, int weightsLen,
                   const float *scratch, int scratchLen, float *hidden, int hiddenLen) {
    (void)hdl; (void)xLen; (void)weightsLen; (void)scratchLen; (void)hiddenLen;
#ifdef __HVX__
    /* Reserve the 128B HVX unit once for the whole forward (gotcha #4: thread-local,
     * one FastRPC method = one worker thread). All hx_matmul_q8 calls below run HVX
     * dots under this lock; the scalar norm/rope/attn kernels are unaffected. */
    qurt_hvx_lock(QURT_HVX_MODE_128B);
#endif
    float *scr = (float *)scratch;   /* `in` buffer is writable host rpcmem; carve work area */
    sp_hex_cfg cfg = { n_layers, n_embd, n_ff, head_dim, n_head, n_head_kv,
                       sliding_window, eps, rope_global, rope_local };
    const int E = n_embd, FF = n_ff, HD = head_dim;
    const int NH = n_head, NKV = n_head_kv, QD = NH * HD, KVD = NKV * HD;
    const int group = NH / NKV;
    const float ascale = 1.0f / sqrtf((float)HD);

    /* carve scratch (order matches sp_hex_scratch_elems) */
    float *resid = scr;
    float *nx = resid + (size_t)n_tok * E;
    float *q  = nx + (size_t)n_tok * E;
    float *k  = q  + (size_t)n_tok * QD;
    float *v  = k  + (size_t)n_tok * KVD;
    float *ao = v  + (size_t)n_tok * KVD;
    float *ap = ao + (size_t)n_tok * QD;
    float *g  = ap + (size_t)n_tok * E;
    float *up = g  + (size_t)n_tok * FF;
    float *dn = up + (size_t)n_tok * FF;
    float *sc = dn + (size_t)n_tok * E;

    for (size_t i = 0; i < (size_t)n_tok * E; i++) resid[i] = x[i];

    for (int L = 0; L < n_layers; L++) {
        const int global = ((L % 6) == 5);
        const float rbase = global ? rope_global : rope_local;
        const int win = global ? -1 : sliding_window;
        const unsigned char *base = weights;
        #define WPTR(kind) (base + sp_hex_weight_off(&cfg, L, (kind)))
        const float *attn_norm = (const float *)WPTR(SP_HEX_ATTN_NORM);
        const float *qn = (const float *)WPTR(SP_HEX_Q_NORM);
        const float *kn = (const float *)WPTR(SP_HEX_K_NORM);

        for (int t = 0; t < n_tok; t++)
            hx_rmsnorm(resid + (size_t)t * E, attn_norm, E, eps, nx + (size_t)t * E);
        hx_matmul_q8(WPTR(SP_HEX_WQ), QD,  E, nx, n_tok, q);
        hx_matmul_q8(WPTR(SP_HEX_WK), KVD, E, nx, n_tok, k);
        hx_matmul_q8(WPTR(SP_HEX_WV), KVD, E, nx, n_tok, v);
        for (int t = 0; t < n_tok; t++) {
            for (int h = 0; h < NH; h++) {
                float *qh = q + (size_t)t * QD + (size_t)h * HD;
                hx_rmsnorm_head(qh, qn, HD, eps); hx_rope_neox(qh, HD, t, rbase);
            }
            for (int h = 0; h < NKV; h++) {
                float *kh = k + (size_t)t * KVD + (size_t)h * HD;
                hx_rmsnorm_head(kh, kn, HD, eps); hx_rope_neox(kh, HD, t, rbase);
            }
        }
        for (int t = 0; t < n_tok; t++)
            for (int h = 0; h < NH; h++)
                hx_attn_head(q + (size_t)t * QD + (size_t)h * HD, k, v, t, KVD,
                             h / group, HD, ascale, win, sc, ao + (size_t)t * QD + (size_t)h * HD);
        hx_matmul_q8(WPTR(SP_HEX_WO), E, QD, ao, n_tok, ap);
        { const float *pn = (const float *)WPTR(SP_HEX_POST_ATTN);
          for (int t = 0; t < n_tok; t++) {
              hx_rmsnorm(ap + (size_t)t * E, pn, E, eps, nx + (size_t)t * E);
              float *xt = resid + (size_t)t * E; const float *p = nx + (size_t)t * E;
              for (int i = 0; i < E; i++) xt[i] += p[i];
          } }
        { const float *fn = (const float *)WPTR(SP_HEX_FFN_NORM);
          for (int t = 0; t < n_tok; t++)
              hx_rmsnorm(resid + (size_t)t * E, fn, E, eps, nx + (size_t)t * E); }
        hx_matmul_q8(WPTR(SP_HEX_WGATE), FF, E, nx, n_tok, g);
        hx_matmul_q8(WPTR(SP_HEX_WUP),   FF, E, nx, n_tok, up);
        for (size_t i = 0; i < (size_t)n_tok * FF; i++) g[i] = hx_gelu_tanh(g[i]) * up[i];
        hx_matmul_q8(WPTR(SP_HEX_WDOWN), E, FF, g, n_tok, dn);
        { const float *pf = (const float *)WPTR(SP_HEX_POST_FFW);
          for (int t = 0; t < n_tok; t++) {
              hx_rmsnorm(dn + (size_t)t * E, pf, E, eps, nx + (size_t)t * E);
              float *xt = resid + (size_t)t * E; const float *p = nx + (size_t)t * E;
              for (int i = 0; i < E; i++) xt[i] += p[i];
          } }
        #undef WPTR
    }

    { const float *on = (const float *)(weights + sp_hex_weight_off(&cfg, n_layers, 0));
      for (int t = 0; t < n_tok; t++)
          hx_rmsnorm(resid + (size_t)t * E, on, E, eps, hidden + (size_t)t * E); }

#ifdef __HVX__
    qurt_hvx_unlock();
#endif
    FARF(RUNTIME_HIGH, "sp_hex: forward n_tok=%d n_layers=%d done", n_tok, n_layers);
    return 0;
}
