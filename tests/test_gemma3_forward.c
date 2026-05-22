/* test_gemma3_forward.c — M_GEMMA3_CPU: the engine's Gemma3 f32 forward pass
 * distributionally matches the stock llama.cpp oracle on identical token IDs.
 * Mirrors E_CPU_2 (test_forward.c): reads the oracle dump (token IDs + per-position
 * logits), runs gemma3_forward, and gates on argmax + top-5 cross + mean KL. */
#define _CRT_SECURE_NO_WARNINGS
#include "sp/sp_test.h"
#include "sp_engine/model.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef SP_GEMMA3_GGUF
#define SP_GEMMA3_GGUF "gemma-3-1b-it-f16.gguf"
#endif
#ifndef SP_GEMMA3_REF
#define SP_GEMMA3_REF "gemma3_ref.bin"
#endif

static void top5(const float *x, uint32_t n, int out5[5]) {
    for (int r = 0; r < 5; r++) {
        int best = -1;
        for (uint32_t j = 0; j < n; j++) {
            int taken = 0;
            for (int p = 0; p < r; p++) if (out5[p] == (int)j) { taken = 1; break; }
            if (taken) continue;
            if (best < 0 || x[j] > x[best]) best = (int)j;
        }
        out5[r] = best;
    }
}
static int in_set(int v, const int *set, int n) {
    for (int i = 0; i < n; i++) if (set[i] == v) return 1;
    return 0;
}
/* KL(P||Q) in nats where P=softmax(p), Q=softmax(q), over the full vocab. */
static double kl_div(const float *p, const float *q, uint32_t n) {
    float pmax = p[0], qmax = q[0];
    for (uint32_t j = 1; j < n; j++) { if (p[j] > pmax) pmax = p[j]; if (q[j] > qmax) qmax = q[j]; }
    double zp = 0.0, zq = 0.0;
    for (uint32_t j = 0; j < n; j++) { zp += exp((double)p[j] - pmax); zq += exp((double)q[j] - qmax); }
    double logzp = log(zp) + pmax, logzq = log(zq) + qmax;
    double kl = 0.0;
    for (uint32_t j = 0; j < n; j++) {
        double logP = (double)p[j] - logzp;
        double Pj = exp(logP);
        if (Pj > 0.0) kl += Pj * (logP - ((double)q[j] - logzq));
    }
    return kl;
}

static void M_GEMMA3_CPU(void) {
    FILE *f = fopen(SP_GEMMA3_REF, "rb");
    SP_CHECK(f != NULL, "open oracle gemma3_ref.bin");
    if (!f) { fprintf(stderr, "    (no ref: %s — regenerate with tools/oracle/dump_logits)\n", SP_GEMMA3_REF); return; }
    uint32_t magic = 0, nt = 0, nv = 0;
    if (fread(&magic, 4, 1, f) != 1 || fread(&nt, 4, 1, f) != 1 || fread(&nv, 4, 1, f) != 1) {
        SP_CHECK(0, "read ref header"); fclose(f); return;
    }
    SP_CHECK(magic == 0x47474C53u, "ref magic");
    SP_CHECK(nt > 0 && nv > 0, "ref dims");
    int32_t *toks = (int32_t *)malloc((size_t)nt * sizeof(int32_t));
    float   *ref  = (float *)malloc((size_t)nt * nv * sizeof(float));
    int read_ok = toks && ref &&
                  fread(toks, sizeof(int32_t), nt, f) == nt &&
                  fread(ref, sizeof(float), (size_t)nt * nv, f) == (size_t)nt * nv;
    fclose(f);
    SP_CHECK(read_ok, "read ref tokens + logits");
    if (!read_ok) { free(toks); free(ref); return; }

    qwen3_model *m = qwen3_load(SP_GEMMA3_GGUF);
    SP_CHECK(m != NULL, "gemma3 load");
    if (!m) { free(toks); free(ref); return; }
    SP_CHECK(m->cfg.arch == SP_ARCH_GEMMA3, "arch gemma3");
    SP_CHECK_EQ_I64(m->cfg.n_vocab, nv, "vocab matches ref");

    float *got = (float *)malloc((size_t)nt * nv * sizeof(float));
    int rc = got ? gemma3_forward(m, toks, (int)nt, got) : 1;
    SP_CHECK(rc == 0, "gemma3_forward");

    if (rc == 0) {
        /* Distributional gate, same rationale as E_CPU_2 (§8.6.1): a scalar f32
         * forward cannot bit-match ggml's SIMD+F16 arithmetic (QK-RMSNorm amplifies
         * the matmul-precision floor), so we gate on argmax + top-5 cross + mean KL. */
        double worst_abs = 0.0, worst_rel = 0.0, kl_sum = 0.0, kl_max = 0.0;
        long argmax_ok = 0, top5_ok = 0;
        for (uint32_t t = 0; t < nt; t++) {
            const float *gr = got + (size_t)t * nv;   /* engine */
            const float *rr = ref + (size_t)t * nv;   /* ggml reference */
            int gmax = 0, rmax = 0;
            for (uint32_t j = 0; j < nv; j++) {
                float ad = fabsf(gr[j] - rr[j]);
                float scale = fabsf(gr[j]) > fabsf(rr[j]) ? fabsf(gr[j]) : fabsf(rr[j]);
                if (ad > worst_abs) worst_abs = ad;
                if (scale > 1.0f && ad / scale > worst_rel) worst_rel = ad / scale;
                if (gr[j] > gr[gmax]) gmax = (int)j;
                if (rr[j] > rr[rmax]) rmax = (int)j;
            }
            if (gmax == rmax) argmax_ok++;
            int g5[5], r5[5];
            top5(gr, nv, g5);
            top5(rr, nv, r5);
            int cross = in_set(rmax, g5, 5) && in_set(gmax, r5, 5);
            if (cross) top5_ok++;
            double kl = kl_div(rr, gr, nv);
            kl_sum += kl; if (kl > kl_max) kl_max = kl;
            fprintf(stderr, "      pos %2u: argmax %s (eng %d / ggml %d) | top5_cross %s | KL=%.3e\n",
                    t, gmax == rmax ? "ok" : "MISS", gmax, rmax, cross ? "ok" : "MISS", kl);
        }
        double kl_mean = kl_sum / nt;
        const char *kl_env = getenv("SP_KL_MAX");
        double kl_gate = kl_env ? atof(kl_env) : 1.0e-5;   /* nats */
        fprintf(stderr, "    %u pos x %u vocab | worst_abs=%.3e worst_rel=%.3e "
                "| argmax=%ld/%u top5_cross=%ld/%u | KL mean=%.3e max=%.3e (gate %.1e)\n",
                nt, nv, worst_abs, worst_rel, argmax_ok, nt, top5_ok, nt, kl_mean, kl_max, kl_gate);
        SP_CHECK_EQ_I64(argmax_ok, nt, "engine argmax matches ggml at every position");
        SP_CHECK_EQ_I64(top5_ok, nt, "engine/ggml top-1 each inside the other's top-5");
        SP_CHECK(kl_mean < kl_gate, "mean KL(ggml||engine) below gate");
    }

    free(got); free(toks); free(ref);
    qwen3_free(m);
}

int main(void) {
    SP_RUN(M_GEMMA3_CPU);
    return SP_DONE();
}
