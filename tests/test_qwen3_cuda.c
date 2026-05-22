/* test_qwen3_cuda.c — M_QWEN3_CUDA: the §8.3 Phase-2-CU E-tests on Qwen3-0.6B.
 *   E_CU_1  loader (qwen3_load).
 *   E_CU_2  forward distributional gate vs the stock-llama.cpp oracle (argmax +
 *           top-5 cross + mean KL) — same gate as E_CPU_2/§8.6.1.
 *   E_CU_3  per-row Frobenius Q8 arena (SP_ARENA=q8), CUDA vs CPU.
 *   E_CU_4  cuBLAS SGEMM is the CUDA "vectorised" matmul path (exercised by all
 *           the above; sm_75 has no TF32 so it is true f32).
 *   §8.3 cross-backend: CUDA output within the precision floor of the CPU output. */
#define _CRT_SECURE_NO_WARNINGS
#include "sp/sp_test.h"
#include "sp_engine/gguf.h"
#include "sp_engine/model.h"
#include "sp_engine/arena.h"
#include "sp_engine/cuda_backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef SP_QWEN3_GGUF
#define SP_QWEN3_GGUF "Qwen3-0.6B-f16.gguf"
#endif
#ifndef SP_QWEN3_REF
#define SP_QWEN3_REF "qwen3_ref.bin"
#endif

#ifdef _WIN32
#define ENV_SET(k, v) _putenv_s((k), (v))
#define ENV_CLR(k)    _putenv_s((k), "")
#else
static void ENV_SET(const char *k, const char *v) { setenv(k, v, 1); }
static void ENV_CLR(const char *k) { unsetenv(k); }
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
static int in_set(int v, const int *s, int n) { for (int i = 0; i < n; i++) if (s[i] == v) return 1; return 0; }
static double kl_div(const float *p, const float *q, uint32_t n) {
    float pmax = p[0], qmax = q[0];
    for (uint32_t j = 1; j < n; j++) { if (p[j] > pmax) pmax = p[j]; if (q[j] > qmax) qmax = q[j]; }
    double zp = 0.0, zq = 0.0;
    for (uint32_t j = 0; j < n; j++) { zp += exp((double)p[j] - pmax); zq += exp((double)q[j] - qmax); }
    double lzp = log(zp) + pmax, lzq = log(zq) + qmax, kl = 0.0;
    for (uint32_t j = 0; j < n; j++) {
        double logP = (double)p[j] - lzp, Pj = exp(logP);
        if (Pj > 0.0) kl += Pj * (logP - ((double)q[j] - lzq));
    }
    return kl;
}

/* compare two logit blocks [nt x V]: argmax-all-match + mean KL + worst rel. */
static void compare(const char *tag, const float *a, const float *b, long nt, int V,
                    double kl_gate) {
    double worst_rel = 0, kl_sum = 0, kl_max = 0; long argmax_ok = 0;
    for (long t = 0; t < nt; t++) {
        const float *x = a + (size_t)t * V, *y = b + (size_t)t * V;
        int ax = 0, ay = 0;
        for (int j = 0; j < V; j++) {
            float ad = fabsf(x[j] - y[j]);
            float sc = fabsf(x[j]) > fabsf(y[j]) ? fabsf(x[j]) : fabsf(y[j]);
            if (sc > 1.0f && ad / sc > worst_rel) worst_rel = ad / sc;
            if (x[j] > x[ax]) ax = j;
            if (y[j] > y[ay]) ay = j;
        }
        if (ax == ay) argmax_ok++;
        double kl = kl_div(x, y, (uint32_t)V); kl_sum += kl; if (kl > kl_max) kl_max = kl;
    }
    fprintf(stderr, "    [%s] argmax=%ld/%ld worst_rel=%.3e KL mean=%.3e max=%.3e (gate %.1e)\n",
            tag, argmax_ok, nt, worst_rel, kl_sum / nt, kl_max, kl_gate);
    SP_CHECK_EQ_I64(argmax_ok, nt, "argmax matches at every position");
    SP_CHECK(kl_sum / nt < kl_gate, "mean KL below gate");
}

static void M_QWEN3_CUDA(void) {
    SP_CHECK(sp_cuda_device_count() >= 1, "CUDA device visible");
    if (sp_cuda_device_count() < 1) { fprintf(stderr, "    %s\n", sp_last_error()); return; }

    /* oracle dump: magic, nt, nv, toks[nt], logits[nt*nv] */
    FILE *f = fopen(SP_QWEN3_REF, "rb");
    SP_CHECK(f != NULL, "open qwen3_ref.bin (E_CU_2 oracle)");
    if (!f) { fprintf(stderr, "    (no ref: %s)\n", SP_QWEN3_REF); return; }
    uint32_t magic = 0, nt = 0, nv = 0;
    int hdr = (fread(&magic,4,1,f)==1 && fread(&nt,4,1,f)==1 && fread(&nv,4,1,f)==1);
    SP_CHECK(hdr && magic == 0x47474C53u && nt > 0 && nv > 0, "ref header");
    if (!hdr) { fclose(f); return; }
    int32_t *toks = (int32_t *)malloc((size_t)nt * sizeof(int32_t));
    float   *ref  = (float *)malloc((size_t)nt * nv * sizeof(float));
    int ok = toks && ref && fread(toks,sizeof(int32_t),nt,f)==nt &&
             fread(ref,sizeof(float),(size_t)nt*nv,f)==(size_t)nt*nv;
    fclose(f);
    SP_CHECK(ok, "read ref tokens + logits");
    if (!ok) { free(toks); free(ref); return; }

    /* ── f32: load (E_CU_1), forward on CPU + CUDA ── */
    ENV_CLR("SP_ARENA");
    qwen3_model *m = qwen3_load(SP_QWEN3_GGUF);
    SP_CHECK(m && m->cfg.arch == SP_ARCH_QWEN3, "qwen3 load (E_CU_1)");
    if (!m) { free(toks); free(ref); return; }
    SP_CHECK_EQ_I64(m->cfg.n_vocab, nv, "vocab matches ref");
    const int V = (int)nv;

    float *cpu = (float *)malloc((size_t)nt * V * sizeof(float));
    float *cu  = (float *)malloc((size_t)nt * V * sizeof(float));
    SP_CHECK(cpu && cu, "alloc logits");
    if (!cpu || !cu) { free(cpu); free(cu); free(toks); free(ref); qwen3_free(m); return; }

    int rc_cpu = qwen3_forward(m, toks, (int)nt, cpu);
    int rc_cu  = qwen3_forward_cuda(m, toks, (int)nt, cu);
    SP_CHECK(rc_cpu == 0 && rc_cu == 0, "qwen3 forward cpu + cuda");
    if (rc_cu) fprintf(stderr, "    sp_last_error: %s\n", sp_last_error());

    if (rc_cpu == 0 && rc_cu == 0) {
        /* E_CU_2: CUDA vs the ggml oracle — distributional (argmax+top5+KL). */
        double kl_sum = 0; long argmax_ok = 0, top5_ok = 0;
        for (uint32_t t = 0; t < nt; t++) {
            const float *gr = cu + (size_t)t * V, *rr = ref + (size_t)t * V;
            int gm = 0, rm = 0;
            for (int j = 0; j < V; j++) { if (gr[j] > gr[gm]) gm = j; if (rr[j] > rr[rm]) rm = j; }
            if (gm == rm) argmax_ok++;
            int g5[5], r5[5]; top5(gr, V, g5); top5(rr, V, r5);
            if (in_set(rm, g5, 5) && in_set(gm, r5, 5)) top5_ok++;
            kl_sum += kl_div(rr, gr, V);
        }
        fprintf(stderr, "    [E_CU_2 cuda-vs-oracle] argmax=%ld/%u top5=%ld/%u KL mean=%.3e\n",
                argmax_ok, nt, top5_ok, nt, kl_sum / nt);
        SP_CHECK_EQ_I64(argmax_ok, nt, "E_CU_2: CUDA argmax matches ggml oracle");
        SP_CHECK_EQ_I64(top5_ok, nt, "E_CU_2: top-1 each inside the other's top-5");
        SP_CHECK(kl_sum / nt < 1.0e-5, "E_CU_2: mean KL(oracle||cuda) below 1e-5");

        /* §8.3 cross-backend: CUDA vs CPU (engine's own f32). */
        compare("8.3 cuda-vs-cpu f32", cpu, cu, nt, V, 1.0e-5);
    }
    qwen3_free(m);

    /* ── E_CU_3: per-row Frobenius Q8 arena, CUDA vs CPU ── */
    ENV_SET("SP_ARENA", "q8");
    m = qwen3_load(SP_QWEN3_GGUF);
    SP_CHECK(m && m->arena && sp_arena_precision(m->arena) == 8, "qwen3 q8 arena load (E_CU_3)");
    if (m) {
        int r1 = qwen3_forward(m, toks, (int)nt, cpu);
        int r2 = qwen3_forward_cuda(m, toks, (int)nt, cu);
        SP_CHECK(r1 == 0 && r2 == 0, "qwen3 q8 forward cpu + cuda");
        if (r2) fprintf(stderr, "    sp_last_error: %s\n", sp_last_error());
        if (r1 == 0 && r2 == 0) compare("E_CU_3 q8 cuda-vs-cpu", cpu, cu, nt, V, 1.0e-5);
        qwen3_free(m);
    }
    ENV_CLR("SP_ARENA");

    free(cpu); free(cu); free(toks); free(ref);
}

int main(void) { SP_RUN(M_QWEN3_CUDA); return SP_DONE(); }
