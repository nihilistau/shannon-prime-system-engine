/* test_avx.c — E_CPU_4: the AVX2 matmul path matches the scalar reference.
 *
 * Runs the forward pass twice on the oracle's token IDs — once with the SIMD dot
 * (default, when built with SP_ENGINE_AVX2) and once forced scalar (SP_CPU_SCALAR=1)
 * — and bounds the per-logit difference. AVX2 uses FMA + 8-wide accumulation, so it
 * is not bit-identical to the sequential scalar reduction; the gate is the realistic
 * float-reassociation floor (≪ the F16/QK-norm gap of E_CPU_2), not 0. If the build
 * has no AVX2, both runs are scalar and the difference is exactly 0.
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

static void set_scalar(int on) {
    const char *v = on ? "1" : "0";
#ifdef _WIN32
    _putenv_s("SP_CPU_SCALAR", v);
#else
    setenv("SP_CPU_SCALAR", v, 1);
#endif
}
static int argmax(const float *x, uint32_t n) {
    int a = 0; for (uint32_t j = 1; j < n; j++) if (x[j] > x[a]) a = (int)j; return a;
}

static void E_CPU_4(void) {
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
    float *sc = (float *)malloc(nlog * sizeof(float));
    float *av = (float *)malloc(nlog * sizeof(float));
    int ok = sc && av;
    if (ok) { set_scalar(1); ok = qwen3_forward(m, toks, (int)nt, sc) == 0; }
    if (ok) { set_scalar(0); ok = qwen3_forward(m, toks, (int)nt, av) == 0; }
    SP_CHECK(ok, "forward pass in scalar and AVX2 modes");

#ifdef SP_ENGINE_AVX2
    const int have_avx2 = 1;
#else
    const int have_avx2 = 0;
#endif
    if (ok) {
        double worst_abs = 0.0, worst_rel = 0.0; long argmax_ok = 0;
        for (uint32_t t = 0; t < nt; t++) {
            const float *a = av + (size_t)t * nv, *b = sc + (size_t)t * nv;
            if (argmax(a, nv) == argmax(b, nv)) argmax_ok++;
            for (uint32_t j = 0; j < nv; j++) {
                double ad = fabs((double)a[j] - b[j]);
                double scl = fabs(a[j]) > fabs(b[j]) ? fabs(a[j]) : fabs(b[j]);
                if (ad > worst_abs) worst_abs = ad;
                if (scl > 1.0 && ad / scl > worst_rel) worst_rel = ad / scl;
            }
        }
        const char *tg = getenv("SP_AVX_TOL");
        double tol = tg ? atof(tg) : 1.0e-3;   /* FMA + 8-wide reassoc floor; see §8.2 */
        fprintf(stderr, "    %u pos x %u vocab | avx2_built=%d | worst_abs=%.3e worst_rel=%.3e "
                "| argmax(avx vs scalar)=%ld/%u (tol %.1e)\n",
                nt, nv, have_avx2, worst_abs, worst_rel, argmax_ok, nt, tol);
        SP_CHECK_EQ_I64(argmax_ok, nt, "AVX2 and scalar agree on argmax at every position");
        SP_CHECK(worst_abs < tol, "AVX2 matmul matches scalar within reassociation floor");
    }
    set_scalar(0);
    free(sc); free(av); free(toks);
    qwen3_free(m);
}

int main(void) {
    SP_RUN(E_CPU_4);
    return SP_DONE();
}
