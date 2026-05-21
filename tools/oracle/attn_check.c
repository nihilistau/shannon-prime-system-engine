/* attn_check.c — isolate the attention core. Reads ggml's post-RoPE Qcur/Kcur,
 * Vcur and its attention output kqv (captured by dump_layers) for one layer, runs
 * the engine's exact GQA causal softmax + V-weighted sum on the SAME inputs, and
 * diffs against kqv. Decides whether the E_CPU_2 pos>0 gap lives in the attention
 * core (scale/softmax/V-sum/GQA mapping) or upstream in Q/K/V computation.
 *
 *   attn_check.exe <ckpt_dir> <layer>
 *
 * Layouts (ggml contiguous, ne0 fastest):
 *   Qcur [head_dim, n_head,    n_tok]   q(i,h,t) = t*n_head*hd + h*hd + i
 *   Kcur [head_dim, n_head_kv, n_tok]   k(i,h,t) = t*nkv*hd  + h*hd + i
 *   Vcur [head_dim, n_head_kv, n_tok]   same as K
 *   kqv  [head_dim, n_tok, n_head]      o(i,t,h) = h*n_tok*hd + t*hd + i
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static float *load(const char *dir, const char *name, int ne[4]) {
    char path[512];
    snprintf(path, sizeof path, "%s/%s.bin", dir, name);
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "open %s\n", path); exit(1); }
    if (fread(ne, sizeof(int), 4, f) != 4) { fprintf(stderr, "hdr %s\n", path); exit(1); }
    size_t n = (size_t)ne[0] * ne[1] * ne[2] * ne[3];
    float *d = (float *)malloc(n * sizeof(float));
    if (fread(d, sizeof(float), n, f) != n) { fprintf(stderr, "data %s\n", path); exit(1); }
    fclose(f);
    return d;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <ckpt_dir> <layer>\n", argv[0]); return 2; }
    const char *dir = argv[1];
    int L = atoi(argv[2]);
    char nm[64];
    int qne[4], kne[4], vne[4], one[4];
    snprintf(nm, sizeof nm, "Qcur-%d", L); float *Q = load(dir, nm, qne);
    snprintf(nm, sizeof nm, "Kcur-%d", L); float *K = load(dir, nm, kne);
    snprintf(nm, sizeof nm, "Vcur-%d", L); float *V = load(dir, nm, vne);
    snprintf(nm, sizeof nm, "kqv-%d",  L); float *O = load(dir, nm, one);

    const int HD = qne[0], NH = qne[1], NT = qne[2];
    const int NKV = kne[1], group = NH / NKV;
    const float scale = 1.0f / sqrtf((float)HD);
    fprintf(stderr, "layer %d: HD=%d NH=%d NKV=%d NT=%d group=%d\n", L, HD, NH, NKV, NT, group);

    float *sc = (float *)malloc((size_t)NT * sizeof(float));
    double worst = 0.0, worst_p0 = 0.0; int wh = -1, wt = -1;
    for (int h = 0; h < NH; h++) {
        int kvh = h / group;
        for (int t = 0; t < NT; t++) {
            const float *qh = Q + (size_t)t * NH * HD + (size_t)h * HD;
            float maxs = -INFINITY;
            for (int s = 0; s <= t; s++) {
                const float *kh = K + (size_t)s * NKV * HD + (size_t)kvh * HD;
                float d = 0.0f;
                for (int i = 0; i < HD; i++) d += qh[i] * kh[i];
                d *= scale; sc[s] = d;
                if (d > maxs) maxs = d;
            }
            float sum = 0.0f;
            for (int s = 0; s <= t; s++) { sc[s] = expf(sc[s] - maxs); sum += sc[s]; }
            float inv = 1.0f / sum;
            const float *oref = O + (size_t)h * NT * HD + (size_t)t * HD;
            for (int i = 0; i < HD; i++) {
                float acc = 0.0f;
                for (int s = 0; s <= t; s++)
                    acc += (sc[s] * inv) * V[(size_t)s * NKV * HD + (size_t)kvh * HD + i];
                double ad = fabs((double)acc - oref[i]);
                if (ad > worst) { worst = ad; wh = h; wt = t; }
                if (t == 0 && ad > worst_p0) worst_p0 = ad;
            }
        }
    }
    fprintf(stderr, "worst_abs vs ggml kqv = %.3e (head %d pos %d) | worst@pos0 = %.3e\n",
            worst, wh, wt, worst_p0);
    free(Q); free(K); free(V); free(O); free(sc);
    return 0;
}
