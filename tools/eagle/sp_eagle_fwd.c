/* sp_eagle_fwd.c — engine C reference of the gemma4-assistant (EAGLE/MTP) draft forward + drive.
 *
 * Steps 2c/2d-algorithm: ports the numpy oracle (sp_eagle_ref.py) into engine C, gated
 * NUMERICALLY against it via the dumped fixture (identical inputs => match, no RNG skew).
 * Reads weights from the F16 MTP GGUF via the engine gguf API (same values as the sp-Q4).
 * Forward grounded VERBATIM in llama.cpp PR #23398; conventions match core/forward/gemma4.c.
 *
 *   draft_step: xh=concat([x(=target_emb*sqrt3840), h],0) -> pre_proj -> 4 sandwich blocks
 *   (Q-only attn over the fixture's target K/V; per-head q_norm; GeGLU tanh; rope swa1e4/full1e6;
 *   layer_output_scale) -> output_norm -> logits=token_embd@cur (DRAFT tied head) + h_next=post_proj@cur
 *
 *   drive (K steps, EAGLE recurrence): token_k=argmax(logits); h_{k+1}=h_next carries the
 *   recurrence (draft owns no KV -- its memory of its own speculated tokens lives in h). pos=POS+k.
 *   This is the loop spec.rs's independent-draft assumption gets wrong.
 *
 * Usage: sp_eagle_fwd <f16_mtp.gguf> <fixture_dir> [--drive]
 * Gates: G-EAGLE-DRAFT-FWD-C (single: argmax==oracle, max|d|<0.1, relL2<2e-3)
 *        G-EAGLE-DRIVE-C     (drive: speculative token sequence == oracle)
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
#define MAXK 16
static const int SWAp[4] = {1, 1, 1, 0};

typedef struct { float *attn_norm,*wq,*qn,*wo,*pan,*ffn_norm,*wg,*wu,*wd,*pfn,*osc,*K,*V; int hd,qd; } Layer;
typedef struct { float *pre,*post,*emb,*onorm; int pre_r,pre_c,post_r,post_c; Layer L[4]; } Weights;

static float *loadw(const gguf_ctx *g, const char *name, int *rows, int *cols) {
    const gguf_tensor *W = gguf_find_tensor(g, name);
    if (!W) { fprintf(stderr, "FATAL missing tensor %s\n", name); exit(2); }
    int c = (int)W->dims[0], r = (W->n_dims >= 2) ? (int)W->dims[1] : 1;
    const uint8_t *base = (const uint8_t *)gguf_tensor_data(g, W);
    size_t rb = (W->type == GGML_T_F16) ? (size_t)c * 2 : (size_t)c * 4;
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
static void matmul(const float *W, const float *x, float *y, int rows, int cols) {
    for (int r = 0; r < rows; r++) { const float *w = W + (size_t)r * cols; double a = 0;
        for (int c = 0; c < cols; c++) a += (double)w[c] * x[c]; y[r] = (float)a; }
}
static float geluf(float x) { return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x))); }
static void ropef(float *v, int hd, int pos, double base) {
    int half = hd / 2;
    for (int i = 0; i < half; i++) { double inv = pow(base, -(2.0 * i) / hd), c = cos(pos * inv), s = sin(pos * inv);
        float a = v[i], b = v[i + half]; v[i] = (float)(a * c - b * s); v[i + half] = (float)(b * c + a * s); }
}
static int argmaxf(const float *v, int n) { int m = 0; for (int i = 1; i < n; i++) if (v[i] > v[m]) m = i; return m; }
static void cmp(const float *a, const float *b, int n, double *maxabs, double *rel) {
    double mx = 0, ne = 0, de = 0;
    for (int i = 0; i < n; i++) { double d = (double)a[i] - b[i]; if (fabs(d) > mx) mx = fabs(d); ne += d * d; de += (double)b[i] * b[i]; }
    *maxabs = mx; *rel = sqrt(ne / (de + 1e-12));
}

static void load_weights(const gguf_ctx *g, const char *fix, Weights *W) {
    int r, c;
    W->pre = loadw(g, "nextn.pre_projection.weight", &W->pre_r, &W->pre_c);
    W->post = loadw(g, "nextn.post_projection.weight", &W->post_r, &W->post_c);
    W->emb = loadw(g, "token_embd.weight", &r, &c);
    W->onorm = loadw(g, "output_norm.weight", &r, &c);
    for (int il = 0; il < 4; il++) {
        char nm[80]; Layer *L = &W->L[il];
        #define LW(suf) (snprintf(nm, sizeof nm, "blk.%d." suf, il), loadw(g, nm, &r, &c))
        L->attn_norm = LW("attn_norm.weight");
        L->wq = LW("attn_q.weight"); L->qd = r; L->hd = L->qd / NH;
        L->qn = LW("attn_q_norm.weight");
        L->wo = LW("attn_output.weight");
        L->pan = LW("post_attention_norm.weight");
        L->ffn_norm = LW("ffn_norm.weight");
        L->wg = LW("ffn_gate.weight"); L->wu = LW("ffn_up.weight"); L->wd = LW("ffn_down.weight");
        L->pfn = LW("post_ffw_norm.weight"); L->osc = LW("layer_output_scale.weight");
        #undef LW
        char kn[80], vn[80]; snprintf(kn, sizeof kn, "k%d.f32", il); snprintf(vn, sizeof vn, "v%d.f32", il);
        L->K = loadf(fix, kn, P * L->hd); L->V = loadf(fix, vn, P * L->hd);
    }
}

static void draft_step(const Weights *W, const float *x, const float *h, int pos, float *logits, float *hnext) {
    static float xh[2 * BB], cur[HID], nbuf[HID], abuf[HID], attn_out[HID], q[NH * 512], ctx[NH * 512], gbuf[FF], fbuf[FF];
    memcpy(xh, x, BB * 4); memcpy(xh + BB, h, BB * 4);
    matmul(W->pre, xh, cur, W->pre_r, W->pre_c);
    for (int il = 0; il < 4; il++) {
        const Layer *L = &W->L[il]; int hd = L->hd, qd = L->qd;
        rms(cur, L->attn_norm, nbuf, HID);
        matmul(L->wq, nbuf, q, qd, HID);
        double base = SWAp[il] ? 1e4 : 1e6;
        for (int hh = 0; hh < NH; hh++) { float *qh = q + (size_t)hh * hd, t[512]; rms(qh, L->qn, t, hd); memcpy(qh, t, hd * 4); ropef(qh, hd, pos, base); }
        float asc = 1.0f / sqrtf((float)hd);
        for (int hh = 0; hh < NH; hh++) {
            const float *qh = q + (size_t)hh * hd; float sc[P], mx = -1e30f;
            for (int t = 0; t < P; t++) { double s = 0; const float *kt = L->K + (size_t)t * hd; for (int d = 0; d < hd; d++) s += (double)kt[d] * qh[d]; sc[t] = (float)s * asc; if (sc[t] > mx) mx = sc[t]; }
            float sum = 0; for (int t = 0; t < P; t++) { sc[t] = expf(sc[t] - mx); sum += sc[t]; }
            float *ch = ctx + (size_t)hh * hd;
            for (int d = 0; d < hd; d++) { double a = 0; for (int t = 0; t < P; t++) a += (double)sc[t] * L->V[(size_t)t * hd + d]; ch[d] = (float)(a / sum); }
        }
        matmul(L->wo, ctx, abuf, HID, qd);
        rms(abuf, L->pan, nbuf, HID);
        for (int i = 0; i < HID; i++) attn_out[i] = nbuf[i] + cur[i];
        rms(attn_out, L->ffn_norm, nbuf, HID);
        matmul(L->wg, nbuf, gbuf, FF, HID); matmul(L->wu, nbuf, fbuf, FF, HID);
        for (int i = 0; i < FF; i++) gbuf[i] = geluf(gbuf[i]) * fbuf[i];
        matmul(L->wd, gbuf, nbuf, HID, FF);
        rms(nbuf, L->pfn, abuf, HID);
        float s = L->osc[0];
        for (int i = 0; i < HID; i++) cur[i] = (abuf[i] + attn_out[i]) * s;
    }
    rms(cur, W->onorm, nbuf, HID);
    matmul(W->emb, nbuf, logits, VOCAB, HID);
    matmul(W->post, nbuf, hnext, BB, HID);
}

int main(int argc, char **argv) {
    const char *gguf = argc > 1 ? argv[1] : "D:/Files/Models/Gemma4/gemma-4-it-mtp/gemma-4-12b-it-F16-MTP.gguf";
    const char *fix  = argc > 2 ? argv[2] : "fixture";
    int do_drive = 0; for (int i = 3; i < argc; i++) if (!strcmp(argv[i], "--drive")) do_drive = 1;
    gguf_ctx *g = gguf_open(gguf);
    if (!g) { fprintf(stderr, "FATAL cannot open %s\n", gguf); return 2; }
    printf("[load] %s | fixture %s\n", gguf, fix);
    Weights W; load_weights(g, fix, &W);

    float *logits = (float *)malloc((size_t)VOCAB * 4); float hnext[BB];
    int rc = 0;

    /* ── single-step gate vs oracle ── */
    {
        float *x = loadf(fix, "x.f32", BB), *h = loadf(fix, "h.f32", BB);
        float *exp_lg = loadf(fix, "logits.f32", VOCAB), *exp_hn = loadf(fix, "hnext.f32", BB);
        draft_step(&W, x, h, POS, logits, hnext);
        int am = argmaxf(logits, VOCAB), ae = argmaxf(exp_lg, VOCAB);
        double lm, lr, hm, hr; cmp(logits, exp_lg, VOCAB, &lm, &lr); cmp(hnext, exp_hn, BB, &hm, &hr);
        int ok = (am == ae) && lm < 0.1 && lr < 2e-3 && hm < 0.05 && hr < 2e-3;
        printf("[single] argmax C=%d oracle=%d match=%d | logits max|d|=%.4g relL2=%.3g | h_next max|d|=%.4g relL2=%.3g\n",
               am, ae, am == ae, lm, lr, hm, hr);
        printf("G-EAGLE-DRAFT-FWD-C: %s\n", ok ? "GREEN" : "RED");
        rc |= !ok; free(x); free(h); free(exp_lg); free(exp_hn);
    }

    /* ── K-step EAGLE drive gate vs oracle (token sequence) ── */
    if (do_drive) {
        char p[1024]; snprintf(p, sizeof p, "%s/drive_tokens.txt", fix);
        FILE *f = fopen(p, "r"); if (!f) { fprintf(stderr, "FATAL no %s\n", p); return 2; }
        int K = 0, exp_tok[MAXK]; if (fscanf(f, "%d", &K) != 1 || K < 1 || K > MAXK) { fprintf(stderr, "bad K\n"); return 2; }
        for (int k = 0; k < K; k++) if (fscanf(f, "%d", &exp_tok[k]) != 1) { fprintf(stderr, "bad tok\n"); return 2; }
        fclose(f);
        float *h = loadf(fix, "h.f32", BB), hcur[BB]; memcpy(hcur, h, BB * 4);
        int got[MAXK], match = 1;
        for (int k = 0; k < K; k++) {
            char dx[80]; snprintf(dx, sizeof dx, "dx%d.f32", k);
            float *xk = loadf(fix, dx, BB);
            draft_step(&W, xk, hcur, POS + k, logits, hnext);
            got[k] = argmaxf(logits, VOCAB);
            memcpy(hcur, hnext, BB * 4);                 /* EAGLE recurrence: h carries forward */
            if (got[k] != exp_tok[k]) match = 0;
            free(xk);
        }
        printf("[drive] K=%d C=[", K); for (int k = 0; k < K; k++) printf("%d%s", got[k], k + 1 < K ? ", " : "");
        printf("] oracle=["); for (int k = 0; k < K; k++) printf("%d%s", exp_tok[k], k + 1 < K ? ", " : "");
        printf("] match=%d\n", match);
        printf("G-EAGLE-DRIVE-C: %s\n", match ? "GREEN" : "RED");
        rc |= !match; free(h);
    }
    return rc ? 1 : 0;
}
