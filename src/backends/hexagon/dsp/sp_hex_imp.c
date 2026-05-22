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
 *   - Q6_Vsf_* IEEE single-float family is BROKEN on V69 (off 4-20 absolute):
 *     do intermediate math in qf32 (Q6_Vqf32_*), sf->qf32 at input,
 *     Q6_Vsf_equals_Vqf32 only at final store.
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
 * layout: y[j] = sum_i w[j*cols + i] * x[i]). Pure scalar f32, no HVX yet (HX.3b
 * swaps this to qf32 HVX). Same op order as the engine's dot_f32 scalar path, so
 * the result matches the host bit-for-bit. */
int sp_hex_matmul_f32(remote_handle64 h, const float *w, int wLen, int rows, int cols,
                      const float *x, int xLen, float *y, int yLen) {
    (void)h; (void)wLen; (void)xLen;
    for (int j = 0; j < rows && j < yLen; j++) {
        const float *wr = w + (long)j * cols;
        float acc = 0.0f;
        for (int i = 0; i < cols; i++) acc += wr[i] * x[i];
        y[j] = acc;
    }
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
            float acc = 0.0f;
            for (int i = 0; i < in; i++) acc += (float)row[i] * x[i];
            Y[(size_t)t * out + j] = acc * inv;
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

    FARF(RUNTIME_HIGH, "sp_hex: forward n_tok=%d n_layers=%d done", n_tok, n_layers);
    return 0;
}
