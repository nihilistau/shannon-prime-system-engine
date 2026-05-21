/* test_frob.c — E_CPU_3: the Frobenius/Q8 weight path.
 *
 * Two properties, both on the same token IDs from the oracle dump:
 *   (1) Lift faithfulness (the roadmap's "identical logits to a reference fp32
 *       matmul"): the inline-lift matmul (SP_ENGINE_FROB=1, accumulate q*x then
 *       scale once) and the dequant-then-f32-matmul of the *same* Q8 weights
 *       (SP_ENGINE_FROB=2) must agree to float-associativity. Gate: max |Δlogit|.
 *   (2) Q8 model quality: the Frobenius-Q8 forward vs the engine's pure-f32
 *       forward — argmax agreement + mean KL. This is the quantization error of
 *       the engine's own Q8 scheme, diffed against the engine's own f32 path
 *       (never ggml — see roadmap §8.6.1), so the gate stays meaningful.
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

/* KL(P||Q) in nats, P=softmax(p), Q=softmax(q), full vocab, numerically stable. */
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

static int run(qwen3_model *m, const int32_t *toks, uint32_t nt, uint32_t nv, int mode, float *out) {
    (void)nv;   /* signature carries it for symmetry; the forward derives nv from the model */
    set_frob(mode);
    return qwen3_forward(m, toks, (int)nt, out);
}

static void E_CPU_3(void) {
    FILE *f = fopen(SP_QWEN3_REF, "rb");
    SP_CHECK(f != NULL, "open oracle ref.bin (token IDs)");
    if (!f) { fprintf(stderr, "    (no ref: %s — regenerate with tools/oracle)\n", SP_QWEN3_REF); return; }
    uint32_t magic = 0, nt = 0, nv = 0;
    if (fread(&magic, 4, 1, f) != 1 || fread(&nt, 4, 1, f) != 1 || fread(&nv, 4, 1, f) != 1) {
        SP_CHECK(0, "read ref header"); fclose(f); return;
    }
    int32_t *toks = (int32_t *)malloc((size_t)nt * sizeof(int32_t));
    int toks_ok = toks && fread(toks, sizeof(int32_t), nt, f) == nt;  /* logits unused here */
    fclose(f);
    SP_CHECK(toks_ok, "read ref token IDs");
    if (!toks_ok) { free(toks); return; }

    qwen3_model *m = qwen3_load(SP_QWEN3_GGUF);
    SP_CHECK(m != NULL, "qwen3_load");
    if (!m) { free(toks); return; }
    nv = m->cfg.n_vocab;

    size_t nlog = (size_t)nt * nv;
    float *f32 = (float *)malloc(nlog * sizeof(float));
    float *q1  = (float *)malloc(nlog * sizeof(float));
    float *q2  = (float *)malloc(nlog * sizeof(float));
    int ok = f32 && q1 && q2 &&
             run(m, toks, nt, nv, 0, f32) == 0 &&
             run(m, toks, nt, nv, 1, q1)  == 0 &&
             run(m, toks, nt, nv, 2, q2)  == 0;
    set_frob(0);  /* leave the env clean for any later test in-process */
    SP_CHECK(ok, "forward pass in f32 / inline-lift / dequant-ref modes");

    if (ok) {
        /* (1) lift faithfulness: inline (q1) vs dequant-ref (q2), same Q8 weights */
        double worst_lift = 0.0;
        for (size_t i = 0; i < nlog; i++) {
            double d = fabs((double)q1[i] - q2[i]);
            if (d > worst_lift) worst_lift = d;
        }
        /* (2) Q8 quality: inline-lift (q1) vs pure f32 (f32) */
        long argmax_ok = 0; double kl_sum = 0.0, kl_max = 0.0;
        for (uint32_t t = 0; t < nt; t++) {
            const float *a = q1 + (size_t)t * nv, *b = f32 + (size_t)t * nv;
            if (argmax(a, nv) == argmax(b, nv)) argmax_ok++;
            double kl = kl_div(b, a, nv);   /* KL(f32 || q8) */
            kl_sum += kl; if (kl > kl_max) kl_max = kl;
        }
        double kl_mean = kl_sum / nt;
        const char *kg = getenv("SP_FROB_KL_MAX");
        double kl_gate = kg ? atof(kg) : 5.0e-2;   /* per-ROW Q8 noise floor; see §8.2 */
        fprintf(stderr, "    %u pos x %u vocab | lift |Δ|max=%.3e (inline vs dequant) "
                "| Q8-vs-f32: argmax=%ld/%u KL mean=%.3e max=%.3e (gate %.1e)\n",
                nt, nv, worst_lift, argmax_ok, nt, kl_mean, kl_max, kl_gate);
        /* PRIMARY (roadmap "identical to a reference fp32 matmul"): the inline lift
         * and the dequant-then-f32-dot of the same Q8 weights agree to float assoc. */
        SP_CHECK(worst_lift < 1.0e-2, "inline lift == dequant-ref matmul (float-assoc)");
        /* Quality regression guard: per-row Frobenius Q8 is lossy by design (one
         * scale per wide row, vs ggml's per-32-block), so it legitimately flips a
         * few low-margin argmaxes — argmax is reported, not gated. KL guards against
         * a broken scale/quant (which would blow KL up orders of magnitude). */
        SP_CHECK(kl_mean < kl_gate, "Q8 forward KL(f32||q8) below regression gate");
    }

    free(f32); free(q1); free(q2); free(toks);
    qwen3_free(m);
}

int main(void) {
    SP_RUN(E_CPU_3);
    return SP_DONE();
}
