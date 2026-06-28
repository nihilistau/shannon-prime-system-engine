/* sp_eagle_fwd.c — engine C reference of the gemma4-assistant (EAGLE/MTP) draft forward.
 *
 * Step 2c: ports the numpy oracle (tools/eagle/sp_eagle_ref.py) into engine C, gated
 * NUMERICALLY against it via the dumped fixture (identical inputs => match, no RNG skew).
 * Reads weights from the F16 MTP GGUF via the engine gguf API (same values as the sp-Q4).
 * Forward grounded VERBATIM in llama.cpp PR #23398; conventions match core/forward/gemma4.c
 * (g4_gelu tanh GeGLU, RMSNorm x*inv*g, sp_rope_neox, layer_output_scale x*=s).
 *
 *   xh  = concat([x(=target_emb*sqrt3840), inp_h], 0)  -> pre_proj -> 1024
 *   4x sandwich block (Q-only attn over the fixture's target K/V) -> output_norm
 *   -> logits = token_embd @ cur (DRAFT tied head) ; h_next = post_proj @ cur
 *
 * Usage: sp_eagle_fwd <f16_mtp.gguf> <fixture_dir>
 * Gate G-EAGLE-DRAFT-FWD-C: argmax(logits)==oracle && max|dlogit|<0.1 && relL2<2e-3 (+h_next).
 */
#define _CRT_SECURE_NO_WARNINGS
#include "sp_engine/gguf.h"
#include "sp_engine/model.h"      /* sp_dequant_row */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#define NH 16
#define HID 1024
#define BB 3840
#define VOCAB 262144
#define FF 8192
#define EPSF 1e-6f
#define P 12
#define POS 7
static const int SWAp[4] = {1, 1, 1, 0};                 /* [8,8,8,1] kv: 3 SWA, 1 full */

/* dequant a whole GGUF tensor to a freshly-malloc'd f32 [rows*cols]; rows=dims[1], cols=dims[0] */
static float *loadw(const gguf_ctx *g, const char *name, int *rows, int *cols) {
    const gguf_tensor *W = gguf_find_tensor(g, name);
    if (!W) { fprintf(stderr, "FATAL missing tensor %s\n", name); exit(2); }
    int c = (int)W->dims[0];
    int r = (W->n_dims >= 2) ? (int)W->dims[1] : 1;
    const uint8_t *base = (const uint8_t *)gguf_tensor_data(g, W);
    size_t rb = (W->type == GGML_T_F16) ? (size_t)c * 2 : (size_t)c * 4;   /* draft = F16 matmul / F32 norm */
    float *out = (float *)malloc((size_t)r * c * sizeof(float));
    if (!out) { fprintf(stderr, "FATAL oom %s\n", name); exit(2); }
    for (int i = 0; i < r; i++) sp_dequant_row(base + (size_t)i * rb, W->type, c, out + (size_t)i * c);
    *rows = r; *cols = c; return out;
}
static float *loadf(const char *dir, const char *nm, int n) {
    char p[1024]; snprintf(p, sizeof p, "%s/%s", dir, nm);
    FILE *f = fopen(p, "rb"); if (!f) { fprintf(stderr, "FATAL no fixture %s\n", p); exit(2); }
    float *b = (float *)malloc((size_t)n * 4);
    if (fread(b, 4, (size_t)n, f) != (size_t)n) { fprintf(stderr, "FATAL short %s\n", p); exit(2); }
    fclose(f); return b;
}
static void rms(const float *x, const float *g, float *o, int d) {
    double ss = 0; for (int i = 0; i < d; i++) ss += (double)x[i] * x[i];
    float inv = 1.0f / sqrtf((float)(ss / d) + EPSF);
    for (int i = 0; i < d; i++) o[i] = x[i] * inv * g[i];
}
static void matmul(const float *W, const float *x, float *y, int rows, int cols) {  /* y[rows]=W[rows,cols]@x[cols] */
    for (int r = 0; r < rows; r++) {
        const float *w = W + (size_t)r * cols; double a = 0;
        for (int c = 0; c < cols; c++) a += (double)w[c] * x[c];
        y[r] = (float)a;
    }
}
static float geluf(float x) { return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x))); }
static void ropef(float *v, int hd, int pos, double base) {
    int half = hd / 2;
    for (int i = 0; i < half; i++) {
        double inv = pow(base, -(2.0 * i) / hd), c = cos(pos * inv), s = sin(pos * inv);
        float a = v[i], b = v[i + half];
        v[i] = (float)(a * c - b * s); v[i + half] = (float)(b * c + a * s);
    }
}
static int argmaxf(const float *v, int n) { int m = 0; for (int i = 1; i < n; i++) if (v[i] > v[m]) m = i; return m; }
static void cmp(const char *tag, const float *a, const float *b, int n, double *maxabs, double *rel) {
    double mx = 0, ne = 0, de = 0;
    for (int i = 0; i < n; i++) { double d = (double)a[i] - b[i]; if (fabs(d) > mx) mx = fabs(d); ne += d * d; de += (double)b[i] * b[i]; }
    *maxabs = mx; *rel = sqrt(ne / (de + 1e-12));
    (void)tag;
}

int main(int argc, char **argv) {
    const char *gguf = argc > 1 ? argv[1] : "D:/Files/Models/Gemma4/gemma-4-it-mtp/gemma-4-12b-it-F16-MTP.gguf";
    const char *fix  = argc > 2 ? argv[2] : "fixture";
    gguf_ctx *g = gguf_open(gguf);
    if (!g) { fprintf(stderr, "FATAL cannot open %s\n", gguf); return 2; }
    printf("[load] %s | fixture %s\n", gguf, fix);

    float *x = loadf(fix, "x.f32", BB), *h = loadf(fix, "h.f32", BB);
    float *exp_lg = loadf(fix, "logits.f32", VOCAB), *exp_hn = loadf(fix, "hnext.f32", BB);

    int rr, cc;
    float *pre = loadw(g, "nextn.pre_projection.weight", &rr, &cc);     /* [1024,7680] */
    float xh[2 * BB]; memcpy(xh, x, BB * 4); memcpy(xh + BB, h, BB * 4);
    float cur[HID]; matmul(pre, xh, cur, rr, cc); free(pre);
    printf("[chain] pre_proj=%d", HID);

    static float q[NH * 512], ctx[NH * 512], nbuf[HID], abuf[HID], fbuf[FF], gbuf[FF], attn_out[HID];
    for (int il = 0; il < 4; il++) {
        char nm[80];
        #define LW(suf, R, C) (snprintf(nm, sizeof nm, "blk.%d." suf, il), loadw(g, nm, (R), (C)))
        int r, c;
        float *attn_norm = LW("attn_norm.weight", &r, &c);
        float *wq        = LW("attn_q.weight", &r, &c); int qd = r; int hd = qd / NH;
        float *qn        = LW("attn_q_norm.weight", &r, &c);
        float *wo        = LW("attn_output.weight", &r, &c);
        float *pan       = LW("post_attention_norm.weight", &r, &c);
        float *ffn_norm  = LW("ffn_norm.weight", &r, &c);
        float *wg        = LW("ffn_gate.weight", &r, &c);
        float *wu        = LW("ffn_up.weight", &r, &c);
        float *wd        = LW("ffn_down.weight", &r, &c);
        float *pfn       = LW("post_ffw_norm.weight", &r, &c);
        float *osc       = LW("layer_output_scale.weight", &r, &c);
        char kn[80], vn[80]; snprintf(kn, sizeof kn, "k%d.f32", il); snprintf(vn, sizeof vn, "v%d.f32", il);
        float *K = loadf(fix, kn, P * hd), *V = loadf(fix, vn, P * hd);

        rms(cur, attn_norm, nbuf, HID);
        matmul(wq, nbuf, q, qd, HID);                                   /* [NH*hd] */
        double base = SWAp[il] ? 1e4 : 1e6;
        for (int hh = 0; hh < NH; hh++) {
            float *qh = q + (size_t)hh * hd, tmp[512];
            rms(qh, qn, tmp, hd); memcpy(qh, tmp, hd * 4);
            ropef(qh, hd, POS, base);
        }
        float asc = 1.0f / sqrtf((float)hd);
        for (int hh = 0; hh < NH; hh++) {
            const float *qh = q + (size_t)hh * hd; float sc[P], mx = -1e30f;
            for (int t = 0; t < P; t++) { double s = 0; const float *kt = K + (size_t)t * hd; for (int d = 0; d < hd; d++) s += (double)kt[d] * qh[d]; sc[t] = (float)s * asc; if (sc[t] > mx) mx = sc[t]; }
            float sum = 0; for (int t = 0; t < P; t++) { sc[t] = expf(sc[t] - mx); sum += sc[t]; }
            float *ch = ctx + (size_t)hh * hd;
            for (int d = 0; d < hd; d++) { double a = 0; for (int t = 0; t < P; t++) a += (double)sc[t] * V[(size_t)t * hd + d]; ch[d] = (float)(a / sum); }
        }
        matmul(wo, ctx, abuf, HID, qd);                                 /* [HID] */
        rms(abuf, pan, nbuf, HID);
        for (int i = 0; i < HID; i++) attn_out[i] = nbuf[i] + cur[i];
        rms(attn_out, ffn_norm, nbuf, HID);
        matmul(wg, nbuf, gbuf, FF, HID); matmul(wu, nbuf, fbuf, FF, HID);
        for (int i = 0; i < FF; i++) gbuf[i] = geluf(gbuf[i]) * fbuf[i];
        matmul(wd, gbuf, nbuf, HID, FF);
        rms(nbuf, pfn, abuf, HID);
        float s = osc[0];
        for (int i = 0; i < HID; i++) cur[i] = (abuf[i] + attn_out[i]) * s;
        printf(" -> blk.%d(hd=%d)=%d", il, hd, HID);
        free(attn_norm); free(wq); free(qn); free(wo); free(pan); free(ffn_norm);
        free(wg); free(wu); free(wd); free(pfn); free(osc); free(K); free(V);
        #undef LW
    }

    rms(cur, (loadw(g, "output_norm.weight", &rr, &cc)), nbuf, HID);    /* small leak ok (one-shot) */
    float *emb = loadw(g, "token_embd.weight", &rr, &cc);               /* [VOCAB,1024] */
    float *logits = (float *)malloc((size_t)VOCAB * 4); matmul(emb, nbuf, logits, VOCAB, HID); free(emb);
    float *post = loadw(g, "nextn.post_projection.weight", &rr, &cc);   /* [3840,1024] */
    float hnext[BB]; matmul(post, nbuf, hnext, BB, HID); free(post);
    printf(" -> output_norm=%d -> logits=%d -> h_next=%d\n", HID, VOCAB, BB);

    int am = argmaxf(logits, VOCAB), ae = argmaxf(exp_lg, VOCAB);
    double lg_max, lg_rel, hn_max, hn_rel;
    cmp("logits", logits, exp_lg, VOCAB, &lg_max, &lg_rel);
    cmp("hnext", hnext, exp_hn, BB, &hn_max, &hn_rel);
    int ok = (am == ae) && lg_max < 0.1 && lg_rel < 2e-3 && hn_max < 0.05 && hn_rel < 2e-3;
    printf("[gate] argmax C=%d oracle=%d match=%d | logits max|d|=%.4g relL2=%.3g | h_next max|d|=%.4g relL2=%.3g\n",
           am, ae, am == ae, lg_max, lg_rel, hn_max, hn_rel);
    printf("G-EAGLE-DRAFT-FWD-C: %s\n", ok ? "GREEN" : "RED");
    return ok ? 0 : 1;
}
