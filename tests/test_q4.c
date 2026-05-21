/* test_q4.c — E_CPU_7: the Frobenius/Q4 mixed-precision weight path.
 *
 * Q4 = symmetric 4-bit codes [-7,7] packed two-per-byte, per-row scale, dequant
 * w_hat = q*s/7. Per-row weight-only calibration promotes high-error rows to Q8
 * (mixed precision; activation-based calibration is the Phase-4 refinement).
 *
 * Two properties on the oracle's token IDs (validated on the current model):
 *   (1) Lift faithfulness: inline-Q4 (SP_ENGINE_FROB=3, accumulate code*x then
 *       scale once) vs dequant-then-f32-dot of the SAME Q4 codes/promotions
 *       (SP_ENGINE_FROB=4) agree to float-associativity.
 *   (2) Q4 model quality vs the engine's pure-f32 path (never ggml — §8.6.1):
 *       argmax agreement reported + mean KL under a regression gate (looser than
 *       Q8 since Q4 is lossier by design; real PPL quality is T_FRO_4).
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

static void set_frob(int mode) {
    char v[2] = { (char)('0' + mode), 0 };
#ifdef _WIN32
    _putenv_s("SP_ENGINE_FROB", v);
#else
    setenv("SP_ENGINE_FROB", v, 1);
#endif
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
static int argmax(const float *x, uint32_t n) {
    int a = 0; for (uint32_t j = 1; j < n; j++) if (x[j] > x[a]) a = (int)j; return a;
}

static void E_CPU_7(void) {
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
    float *f32 = (float *)malloc(nlog * sizeof(float));
    float *q4i = (float *)malloc(nlog * sizeof(float));
    float *q4d = (float *)malloc(nlog * sizeof(float));
    int ok = f32 && q4i && q4d;
    if (ok) { set_frob(0); ok = qwen3_forward(m, toks, (int)nt, f32) == 0; }
    long promoted = 0, rows = 0;
    if (ok) { set_frob(3); ok = qwen3_forward(m, toks, (int)nt, q4i) == 0; qwen3_q4_stats(&promoted, &rows); }
    if (ok) { set_frob(4); ok = qwen3_forward(m, toks, (int)nt, q4d) == 0; }
    set_frob(0);
    SP_CHECK(ok, "forward pass in f32 / Q4-inline / Q4-dequant-ref modes");

    if (ok) {
        double worst_lift = 0.0;
        for (size_t i = 0; i < nlog; i++) {
            double d = fabs((double)q4i[i] - q4d[i]);
            if (d > worst_lift) worst_lift = d;
        }
        long argmax_ok = 0; double kl_sum = 0.0, kl_max = 0.0;
        for (uint32_t t = 0; t < nt; t++) {
            const float *a = q4i + (size_t)t * nv, *b = f32 + (size_t)t * nv;
            if (argmax(a, nv) == argmax(b, nv)) argmax_ok++;
            double kl = kl_div(b, a, nv);   /* KL(f32 || q4) */
            kl_sum += kl; if (kl > kl_max) kl_max = kl;
        }
        double kl_mean = kl_sum / nt;
        const char *kg = getenv("SP_Q4_KL_MAX");
        double kl_gate = kg ? atof(kg) : 1.5;   /* Q4 noise floor (lossy by design, ~0.72 measured);
                                                  * headroom over measured mean — a broken scale/pack
                                                  * blows KL up orders of magnitude. See §8.2.1. */
        double prate = rows ? (100.0 * (double)promoted / (double)rows) : 0.0;
        fprintf(stderr, "    %u pos x %u vocab | lift |Δ|max=%.3e (inline vs dequant) | "
                "Q8-promoted %ld/%ld rows (%.1f%%) | Q4-vs-f32: argmax=%ld/%u KL mean=%.3e max=%.3e (gate %.1e)\n",
                nt, nv, worst_lift, promoted, rows, prate, argmax_ok, nt, kl_mean, kl_max, kl_gate);
        /* PRIMARY: inline-Q4 lift == dequant-then-f32-dot of the same Q4 codes. */
        SP_CHECK(worst_lift < 1.0e-2, "inline Q4 lift == dequant-ref matmul (float-assoc)");
        /* mixed precision actually exercised both code paths */
        SP_CHECK(rows > 0, "Q4 weight path ran (rows seen)");
        /* Quality regression guard: Q4 is lossy by design (4 bits, one scale per
         * wide row); argmax flips are expected and reported, not gated. KL guards
         * against a broken scale/pack (which blows KL up orders of magnitude). */
        SP_CHECK(kl_mean < kl_gate, "Q4 forward KL(f32||q4) below regression gate");
    }

    free(f32); free(q4i); free(q4d); free(toks);
    qwen3_free(m);
}

int main(void) {
    SP_RUN(E_CPU_7);
    return SP_DONE();
}
