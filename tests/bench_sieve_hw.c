/* bench_sieve_hw.c — M_POUW_2: AVX-512 ternlog throughput vs Friedman sieve.
 *
 * Gate: AVX-512 ternlog candidate-mixing rate >= 5x scalar sp_sieve_evaluate
 * throughput on AVX-512F hardware.
 *
 * DISABLED path: exits 0 with REQUIRES_LIVE_MODE when AVX-512F is masked at
 * runtime (Hyper-V root partition on this Tiger Lake-B host exposes no AVX-512).
 *
 * Method:
 *   ternlog — sp_avx512_ternlog_kste_round called N_ITERS*N_TREES times
 *   sieve   — sp_sieve_evaluate on N_TREES candidates, N_ITERS iterations
 *             (frontier_n reset each outer iteration for repeatable results)
 *   ratio   = (sieve ns/cand) / (ternlog ns/tree); gate >= 5.0x.
 *
 * Run on bare-metal Linux with AVX-512F hardware to obtain the live ratio. */
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#include "sp_engine/avx512.h"
#include "sp/sp_sieve.h"

static double now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e9 + (double)t.tv_nsec;
}

static inline void do_not_elim(void *p, size_t n) {
    (void)n;
    __asm__ volatile ("" : "+m" (*(char (*)[])p));
}

#define N_TREES     64
#define N_FRONTIER  256
#define WARMUP      50
#define N_ITERS     5000
#define RATIO_GATE  5.0

int main(void) {
    sp_avx512_init();

    if (!g_avx512_caps.has_avx512f) {
        printf("M_POUW_2: REQUIRES_LIVE_MODE"
               " (DISABLED — AVX-512F not available; masked by VBS on this host)\n");
        printf("            Run on bare-metal Linux with AVX-512 hardware, or\n");
        printf("            configure Hyper-V to expose AVX-512 to the root partition.\n");
        return 0;
    }

    int i, j;

    /* ---- AVX-512 ternlog path ------------------------------------------- *
     * sp_avx512_ternlog_kste_round processes one 64-byte KSTE tree per call
     * using a single AVX-512 ternarylogic instruction (imm8=0x96, XOR3).    */
    static uint32_t trees[N_TREES][16] __attribute__((aligned(64)));
    for (i = 0; i < N_TREES; i++)
        for (j = 0; j < 16; j++)
            trees[i][j] = (uint32_t)((i * 16 + j) * 0x9E3779B9u + 1u);

    for (i = 0; i < WARMUP; i++)
        for (j = 0; j < N_TREES; j++)
            sp_avx512_ternlog_kste_round(trees[j]);

    double t0 = now_ns();
    for (i = 0; i < N_ITERS; i++)
        for (j = 0; j < N_TREES; j++)
            sp_avx512_ternlog_kste_round(trees[j]);
    double ternlog_ns = (now_ns() - t0) / (double)(N_ITERS * N_TREES);
    do_not_elim(trees, sizeof(trees));

    /* ---- Scalar sieve path --------------------------------------------- *
     * sp_sieve_evaluate with N_TREES candidates and a fresh frontier per    *
     * outer iteration.  Candidates are built from the ternlog state arrays  *
     * with frozen v1 KSTE header bytes.                                     */
    static sp_kste_tree_t     candidates[N_TREES];
    static sp_kste_tree_t     frontier[N_FRONTIER];
    static sp_sieve_event_t   events[N_TREES];
    size_t frontier_n, n_events;

    for (i = 0; i < N_TREES; i++) {
        memcpy(candidates[i].bytes, trees[i], 64);
        candidates[i].bytes[0] = 1; /* SP_KSTE_LAYOUT_VERSION */
        candidates[i].bytes[1] = 3; /* SP_KSTE_BRANCHING      */
        candidates[i].bytes[2] = 3; /* SP_KSTE_DEPTH          */
        candidates[i].bytes[3] = 0;
        candidates[i].bytes[4] = 0;
        candidates[i].bytes[5] = 0;
        candidates[i].bytes[6] = 0;
        candidates[i].bytes[7] = 0;
    }

    for (i = 0; i < WARMUP; i++) {
        frontier_n = 0; n_events = 0;
        sp_sieve_evaluate(candidates, N_TREES, frontier, &frontier_n, N_FRONTIER,
                          events, &n_events);
    }

    t0 = now_ns();
    for (i = 0; i < N_ITERS; i++) {
        frontier_n = 0; n_events = 0;
        sp_sieve_evaluate(candidates, N_TREES, frontier, &frontier_n, N_FRONTIER,
                          events, &n_events);
    }
    double sieve_ns = (now_ns() - t0) / (double)(N_ITERS * N_TREES);
    do_not_elim(frontier, sizeof(frontier));
    do_not_elim(events,   sizeof(events));

    /* ---- Report --------------------------------------------------------- */
    double ratio = sieve_ns / ternlog_ns;
    int pass     = (ratio >= RATIO_GATE);

    printf("TERNLOG: %.2f ns/tree   (%.1f Mops/s)\n",
           ternlog_ns, 1000.0 / ternlog_ns);
    printf("SIEVE:   %.2f ns/cand   (%.1f Mops/s)\n",
           sieve_ns, 1000.0 / sieve_ns);
    printf("ratio    sieve/ternlog = %.1fx [need >= %.1fx] %s\n\n",
           ratio, RATIO_GATE, pass ? "PASS" : "FAIL");

    if (pass)
        printf("M_POUW_2: PASS\n");
    else
        printf("M_POUW_2: FAIL (%.1fx < %.1fx)\n", ratio, RATIO_GATE);

    return pass ? 0 : 1;
}
