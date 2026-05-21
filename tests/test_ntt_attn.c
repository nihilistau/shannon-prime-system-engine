/* test_ntt_attn.c — E_CPU_5: NTT-attention end-to-end vs the softmax (f32-dot)
 * baseline, sieve OFF.
 *
 * The poly-ring kernel recovers each attention score <q,k> EXACTLY as an integer
 * (coefficient 0 of the negacyclic product, sp_pr_inner) after quantizing the head
 * vectors to int32. So the only deviation from the f32-dot baseline is that int32
 * quantization, applied at every layer. Phase 1C's T_PR_2 bounds a single
 * attention at KL <= 1e-7; this test bounds the *end-to-end* logit distribution
 * (which compounds the per-layer quantization over 28 layers, then runs through
 * QK-RMSNorm — cf. §8.6.1). Gate: argmax agreement + mean KL below SP_NTT_KL_MAX.
 */
#define _CRT_SECURE_NO_WARNINGS
#include "sp/sp_test.h"
#include "sp_engine/model.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef SP_QWEN3_GGUF
#define SP_QWEN3_GGUF "Qwen3-0.6B-f16.gguf"
#endif
#ifndef SP_QWEN3_REF
#define SP_QWEN3_REF "qwen3_ref.bin"
#endif

static void set_ntt(int on) {
    const char *v = on ? "1" : "0";
#ifdef _WIN32
    _putenv_s("SP_ENGINE_NTT_ATTN", v);
#else
    setenv("SP_ENGINE_NTT_ATTN", v, 1);
#endif
}
static int argmax(const float *x, uint32_t n) {
    int a = 0; for (uint32_t j = 1; j < n; j++) if (x[j] > x[a]) a = (int)j; return a;
}
static double kl_div(const float *p, const float *q, uint32_t n) {
    float pmax = p[0], qmax = q[0];
    for (uint32_t j = 1; j < n; j++) { if (p[j] > pmax) pmax = p[j]; if (q[j] > qmax) qmax = q[j]; }
    double zp = 0.0, zq = 0.0;
    for (uint32_t j = 0; j < n; j++) { zp += exp((double)p[j] - pmax); zq += exp((double)q[j] - qmax); }
    double logzp = log(zp) + pmax, logzq = log(zq) + qmax, kl = 0.0;
    for (uint32_t j = 0; j < n; j++) {
        double logP = (double)p[j] - logzp, Pj = exp(logP);
        if (Pj > 0.0) kl += Pj * (logP - ((double)q[j] - logzq));
    }
    return kl;
}

static void E_CPU_5(void) {
    FILE *f = fopen(SP_QWEN3_REF, "rb");
    SP_CHECK(f != NULL, "open oracle ref.bin (token IDs)");
    if (!f) { fprintf(stderr, "    (no ref: %s)\n", SP_QWEN3_REF); return; }
    uint32_t magic = 0, nt = 0, nv = 0;
    if (fread(&magic, 4, 1, f) != 1 || fread(&nt, 4, 1, f) != 1 || fread(&nv, 4, 1, f) != 1) {
        SP_CHECK(0, "read ref header"); fclose(f); return;
    }
    int32_t *toks = (int32_t *)malloc((size_t)nt * sizeof(int32_t));
    int toks_ok = toks && fread(toks, sizeof(int32_t), nt, f) == nt;
    fclose(f);
    SP_CHECK(toks_ok, "read ref token IDs");
    if (!toks_ok) { free(toks); return; }

    qwen3_model *m = qwen3_load(SP_QWEN3_GGUF);
    SP_CHECK(m != NULL, "qwen3_load");
    if (!m) { free(toks); return; }
    nv = m->cfg.n_vocab;

    size_t nlog = (size_t)nt * nv;
    float *base = (float *)malloc(nlog * sizeof(float));   /* f32-dot softmax baseline */
    float *ntt  = (float *)malloc(nlog * sizeof(float));   /* NTT-attention */
    int ok = base && ntt;
    if (ok) { set_ntt(0); ok = qwen3_forward(m, toks, (int)nt, base) == 0; }
    if (ok) { set_ntt(1); ok = qwen3_forward(m, toks, (int)nt, ntt)  == 0; }
    set_ntt(0);
    SP_CHECK(ok, "forward pass in f32-dot and NTT-attention modes");

    if (ok) {
        double worst_abs = 0.0, kl_sum = 0.0, kl_max = 0.0; long argmax_ok = 0;
        for (uint32_t t = 0; t < nt; t++) {
            const float *a = ntt + (size_t)t * nv, *b = base + (size_t)t * nv;
            if (argmax(a, nv) == argmax(b, nv)) argmax_ok++;
            for (uint32_t j = 0; j < nv; j++) {
                double ad = fabs((double)a[j] - b[j]);
                if (ad > worst_abs) worst_abs = ad;
            }
            double kl = kl_div(b, a, nv);   /* KL(softmax-baseline || ntt) */
            kl_sum += kl; if (kl > kl_max) kl_max = kl;
        }
        double kl_mean = kl_sum / nt;
        const char *kg = getenv("SP_NTT_KL_MAX");
        double kl_gate = kg ? atof(kg) : 1.0e-7;   /* literal T_PR_2; measured ~2.7e-10 */
        fprintf(stderr, "    %u pos x %u vocab | NTT-attn vs f32-dot: worst_abs=%.3e "
                "argmax=%ld/%u | KL mean=%.3e max=%.3e (gate %.1e)\n",
                nt, nv, worst_abs, argmax_ok, nt, kl_mean, kl_max, kl_gate);
        SP_CHECK_EQ_I64(argmax_ok, nt, "NTT-attention preserves argmax vs softmax baseline");
        SP_CHECK(kl_mean < kl_gate, "NTT-attention end-to-end KL below T_PR_2-style gate");
    }

    free(base); free(ntt); free(toks);
    qwen3_free(m);
}

int main(void) {
    SP_RUN(E_CPU_5);
    return SP_DONE();
}
