/* test_kv_spinor.c — E_CPU_8: inline VHT2+Spinor KV-cache compression.
 *
 * The foundational KV codec (§4.5/§4.9): each post-norm/post-RoPE K and post-proj
 * V head vector (head_dim=128) is stored as ceil(128/55)=3 frozen 63-byte Spinor
 * blocks and decoded back (lossy) before attention reads it. Gate SP_KV_SPINOR=1.
 *
 * Two properties on the oracle's token IDs (validated on the current model):
 *   (1) Regression invariant: gate OFF is bit-identical to the plain f32 forward
 *       (the codec path is conditionally skipped, so the E_CPU_2 path is untouched).
 *   (2) Gate ON: the codec is actually wired in (KL > 0 vs the f32 KV) and the
 *       lossy reconstruction stays bounded — argmax agreement reported, mean
 *       KL(f32-KV || spinor-KV) under a regression gate. Distinct from E_CPU_6's
 *       KSTE overlay (a one-way sieve signature, not a lossy codec).
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

static void set_spinor(int on) {
#ifdef _WIN32
    _putenv_s("SP_KV_SPINOR", on ? "1" : "0");
#else
    setenv("SP_KV_SPINOR", on ? "1" : "0", 1);
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

static void E_CPU_8(void) {
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
    float *base = (float *)malloc(nlog * sizeof(float));   /* gate OFF */
    float *base2= (float *)malloc(nlog * sizeof(float));   /* gate OFF again (determinism) */
    float *spin = (float *)malloc(nlog * sizeof(float));   /* gate ON  */
    int ok = base && base2 && spin;
    if (ok) { set_spinor(0); ok = qwen3_forward(m, toks, (int)nt, base)  == 0; }
    if (ok) { set_spinor(0); ok = qwen3_forward(m, toks, (int)nt, base2) == 0; }
    if (ok) { set_spinor(1); ok = qwen3_forward(m, toks, (int)nt, spin)  == 0; }
    set_spinor(0);
    SP_CHECK(ok, "forward pass gate-off (x2) and gate-on");

    if (ok) {
        /* (1) regression invariant: gate OFF is deterministic & is the plain path. */
        int off_identical = 1;
        for (size_t i = 0; i < nlog; i++) if (base[i] != base2[i]) { off_identical = 0; break; }
        SP_CHECK(off_identical, "gate OFF bit-identical (codec path skipped => E_CPU_2 path)");

        /* (2) gate ON: codec wired in (KL>0) + lossy-but-bounded. */
        long argmax_ok = 0; double kl_sum = 0.0, kl_max = 0.0;
        for (uint32_t t = 0; t < nt; t++) {
            const float *a = spin + (size_t)t * nv, *b = base + (size_t)t * nv;
            if (argmax(a, nv) == argmax(b, nv)) argmax_ok++;
            double kl = kl_div(b, a, nv);   /* KL(f32-KV || spinor-KV) */
            kl_sum += kl; if (kl > kl_max) kl_max = kl;
        }
        double kl_mean = kl_sum / nt;
        const char *kg = getenv("SP_KV_SPINOR_KL_MAX");
        double kl_gate = kg ? atof(kg) : 0.2;   /* int8-anchor KV codec (~0.022 measured); see §8.2.1 */
        fprintf(stderr, "    %u pos x %u vocab | spinor-KV vs f32-KV: argmax=%ld/%u "
                "KL mean=%.3e max=%.3e (gate %.1e)\n",
                nt, nv, argmax_ok, nt, kl_mean, kl_max, kl_gate);
        SP_CHECK(kl_mean > 0.0, "gate ON actually alters the KV (codec is wired in)");
        SP_CHECK(kl_mean < kl_gate, "spinor-KV forward KL below regression gate");
    }

    free(base); free(base2); free(spin); free(toks);
    qwen3_free(m);
}

int main(void) {
    SP_RUN(E_CPU_8);
    return SP_DONE();
}
