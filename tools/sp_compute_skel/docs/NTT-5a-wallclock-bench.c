/* NTT-5a-wallclock-bench.c — informational wall-clock comparison
 *
 * Compares sp_pr_bluestein_inner vs sp_pr_inner at the overlapping N values
 * {128, 256}. NOT a pass/fail gate per PLAN-NTT-5a §"Wall-clock comparison
 * plan (informational, not a gate)". The numbers feed into
 * CLOSURE-NTT-5a.md row 5.
 *
 * Build (Windows mingw, from engine-ntt-5a root):
 *   cmake -S lib/shannon-prime-system/core/poly_ring -B C:\Temp\sp-build-pr-bench \
 *         -G Ninja -DCMAKE_C_COMPILER=gcc
 *   cd C:\Temp\sp-build-pr-bench && ninja
 *   gcc -O2 -Iinclude tools/sp_compute_skel/docs/NTT-5a-wallclock-bench.c \
 *       libsp_poly_ring.a _deps_sp/ntt_crt/libsp_ntt_crt.a -o bench-5a.exe
 *
 * Run: .\bench-5a.exe
 */
#include "sp/poly_ring.h"
#include "sp/poly_ring_bluestein.h"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

static uint64_t rng = 0x1234567890ABCDEFull;
static int32_t rng_coeff(void) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    int32_t v = (int32_t)(rng & 0x00007FFFu) - (1 << 14);
    return v;
}

static double now_sec(void) {
    struct timespec t;
    timespec_get(&t, TIME_UTC);
    return (double)t.tv_sec + (double)t.tv_nsec / 1e9;
}

int main(void) {
    const uint32_t Ns[2] = { 128u, 256u };
    const int iters = 1000;

    for (int ni = 0; ni < 2; ni++) {
        uint32_t N = Ns[ni];

        sp_pr_ctx           *direct = sp_pr_init(N);
        sp_pr_bluestein_ctx *blue   = sp_pr_bluestein_init(N);
        if (!direct || !blue) {
            fprintf(stderr, "init failed for N=%u\n", N);
            return 1;
        }

        int32_t *q = (int32_t *)malloc(sizeof(int32_t) * N);
        int32_t *k = (int32_t *)malloc(sizeof(int32_t) * N);
        for (uint32_t i = 0; i < N; i++) { q[i] = rng_coeff(); k[i] = rng_coeff(); }

        /* Warmup */
        volatile int64_t sink = 0;
        for (int it = 0; it < 50; it++) {
            sink += sp_pr_inner(direct, q, k);
            sink += sp_pr_bluestein_inner(blue, q, k);
        }

        double t0 = now_sec();
        for (int it = 0; it < iters; it++) sink += sp_pr_inner(direct, q, k);
        double t1 = now_sec();
        for (int it = 0; it < iters; it++) sink += sp_pr_bluestein_inner(blue, q, k);
        double t2 = now_sec();

        double direct_us = (t1 - t0) * 1e6 / (double)iters;
        double blue_us   = (t2 - t1) * 1e6 / (double)iters;
        printf("N=%u  sp_pr_inner=%.2f us  sp_pr_bluestein_inner=%.2f us  ratio=%.2fx\n",
               N, direct_us, blue_us, blue_us / direct_us);

        free(q); free(k);
        sp_pr_free(direct);
        sp_pr_bluestein_free(blue);
        (void)sink;
    }
    return 0;
}
