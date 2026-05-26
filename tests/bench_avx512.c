/* bench_avx512.c — M_AVX_2 throughput gate for Phase 2-CPU.AVX kernels. */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "sp_engine/avx512.h"

static double now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e9 + (double)t.tv_nsec;
}

/* Prevent the compiler from eliminating a computation whose result is in `p`. */
static inline void do_not_elim(void *p, size_t n) {
    (void)n;
    __asm__ volatile ("" : "+m" (*(char (*)[])p));
}

/* ---- Scalar reference kernels compiled WITHOUT AVX target attributes ----
 * These functions must not be inlined or the outer -O3 + -mavx512 will
 * vectorise them. __attribute__((noinline)) + target("no-avx512f") forces
 * them to use plain scalar code, giving a true baseline. */

__attribute__((noinline, target("no-avx512f,no-avx2,no-avx")))
static void scalar_kste_round_loop(uint32_t *st2, int iters) {
    int i;
    for (i = 0; i < iters; i++) {
        uint32_t tmp[16];
        int j;
        for (j = 0; j < 16; j++) tmp[j] = st2[j] ^ st2[(j+1)%16] ^ st2[(j+5)%16];
        memcpy(st2, tmp, 64);
    }
}

__attribute__((noinline, target("no-avx512f,no-avx2,no-avx")))
static void scalar_vnni_loop(const int8_t *w, const uint8_t *act,
                              const float *sc, float *out_s,
                              int rows, int cols, int iters) {
    int iter, r, k;
    for (iter = 0; iter < iters; iter++) {
        for (r = 0; r < rows; r++) {
            int32_t acc = 0;
            for (k = 0; k < cols; k++)
                acc += (int32_t)w[r*cols+k] * (int32_t)((int8_t)((int)act[k]-128));
            out_s[r] = (float)acc * sc[r];
        }
    }
}

__attribute__((noinline, target("no-avx512f,no-avx2,no-avx")))
static void scalar_ifma_loop(const uint32_t *a_if, const uint32_t *b_if,
                              uint32_t Q1, uint64_t MU1,
                              uint32_t *out_s_if, uint32_t nif, int iters) {
    int i;
    for (i = 0; i < iters; i++) {
        uint32_t k;
        for (k = 0; k < nif; k++) {
            uint64_t x = (uint64_t)a_if[k]*(uint64_t)b_if[k];
            uint64_t qhat = ((x>>29)*MU1)>>31;
            uint64_t r = x - qhat*Q1;
            if (r>=Q1) r-=Q1;
            if (r>=Q1) r-=Q1;
            out_s_if[k] = (uint32_t)r;
        }
    }
}

#define WARMUP 20
#define ITERS  5000

int main(void) {
    sp_avx512_init();

    if (!g_avx512_caps.has_avx512f) {
        printf("SKIP: no AVX-512F\n");
        return 0;
    }

    int any_fail = 0;

    /* ---- TERNLOG ---- */
    {
        static uint32_t state[16] __attribute__((aligned(64)));
        int i;
        for (i = 0; i < 16; i++) state[i] = (uint32_t)(i * 0x9E3779B9u + 1);

        for (i = 0; i < WARMUP; i++) sp_avx512_ternlog_kste_round(state);
        double t0 = now_ns();
        for (i = 0; i < ITERS; i++) {
            int j;
            for (j = 0; j < 64; j++) sp_avx512_ternlog_kste_round(state);
        }
        double avx_ns = (now_ns() - t0) / (ITERS * 64);

        uint32_t st2[16];
        memcpy(st2, state, 64);
        /* warm up the scalar path */
        scalar_kste_round_loop(st2, WARMUP);
        t0 = now_ns();
        scalar_kste_round_loop(st2, ITERS * 64);
        double scalar_ns = (now_ns() - t0) / (ITERS * 64);
        do_not_elim(st2, sizeof(st2));

        double ratio = scalar_ns / avx_ns;
        int pass = (ratio >= 16.0);
        printf("TERNLOG: scalar=%.1fns avx=%.1fns speedup=%.1fx [need>=16x] %s\n",
               scalar_ns, avx_ns, ratio, pass ? "PASS" : "FAIL");
        if (!pass) any_fail = 1;
    }

    /* ---- VNNI ---- */
    if (!g_avx512_caps.has_vnni) {
        printf("VNNI:    SKIP (no AVX-512VNNI)\n");
    } else {
        enum { ROWS = 256, COLS = 256 };
        static int8_t  w[ROWS*COLS]  __attribute__((aligned(64)));
        static uint8_t act[COLS]     __attribute__((aligned(64)));
        static float   sc[ROWS]      __attribute__((aligned(64)));
        static int32_t bias[ROWS]    __attribute__((aligned(64)));
        static float   out_v[ROWS]   __attribute__((aligned(64)));
        static float   out_s[ROWS]   __attribute__((aligned(64)));
        int i, k;

        for (i = 0; i < ROWS*COLS; i++) w[i]   = (int8_t)(i % 127);
        for (k = 0; k < COLS; k++)      act[k]  = (uint8_t)(k % 255);
        for (i = 0; i < ROWS; i++) {
            sc[i] = 0.01f;
            int32_t s = 0;
            for (k = 0; k < COLS; k++) s += w[i*COLS+k];
            bias[i] = 128 * s;
        }

        for (i = 0; i < WARMUP; i++) sp_avx512_vnni_matvec(w, act, sc, bias, ROWS, COLS, out_v);
        double t0 = now_ns();
        for (i = 0; i < ITERS; i++) sp_avx512_vnni_matvec(w, act, sc, bias, ROWS, COLS, out_v);
        double avx_ns = (now_ns() - t0) / ITERS;

        /* warm up scalar path */
        scalar_vnni_loop(w, act, sc, out_s, ROWS, COLS, WARMUP);
        t0 = now_ns();
        scalar_vnni_loop(w, act, sc, out_s, ROWS, COLS, ITERS);
        double scalar_ns = (now_ns() - t0) / ITERS;
        do_not_elim(out_s, sizeof(out_s));

        double ratio = scalar_ns / avx_ns;
        int pass = (ratio >= 3.5);
        printf("VNNI:    scalar=%.1fns avx=%.1fns speedup=%.1fx [need>=3.5x] %s\n",
               scalar_ns, avx_ns, ratio, pass ? "PASS" : "FAIL");
        if (!pass) any_fail = 1;
    }

    /* ---- IFMA ---- */
    if (!g_avx512_caps.has_ifma) {
        printf("IFMA:    SKIP (no AVX-512IFMA)\n");
    } else {
        enum { NIF = 512 };
        static uint32_t a_if[NIF], b_if[NIF], out_v_if[NIF], out_s_if[NIF];
        const uint32_t Q1  = 1073738753u;
        const uint64_t MU1 = ((uint64_t)1<<60) / Q1;
        uint32_t k;

        for (k = 0; k < NIF; k++) {
            a_if[k] = (uint32_t)((k*0x9E3779B9u+1) % Q1);
            b_if[k] = (uint32_t)((k*0x6C62272Eu+7) % Q1);
        }

        int i;
        for (i = 0; i < WARMUP; i++) sp_avx512_ifma_modmul(a_if, b_if, Q1, MU1, NIF, out_v_if);
        double t0 = now_ns();
        for (i = 0; i < ITERS; i++) sp_avx512_ifma_modmul(a_if, b_if, Q1, MU1, NIF, out_v_if);
        double avx_ns = (now_ns() - t0) / ITERS;

        /* warm up scalar path */
        scalar_ifma_loop(a_if, b_if, Q1, MU1, out_s_if, NIF, WARMUP);
        t0 = now_ns();
        scalar_ifma_loop(a_if, b_if, Q1, MU1, out_s_if, NIF, ITERS);
        double scalar_ns = (now_ns() - t0) / ITERS;
        do_not_elim(out_s_if, sizeof(out_s_if));

        double ratio = scalar_ns / avx_ns;
        int pass = (ratio >= 2.0);
        printf("IFMA:    scalar=%.1fns avx=%.1fns speedup=%.1fx [need>=2x on TGL] %s\n",
               scalar_ns, avx_ns, ratio, pass ? "PASS" : "FAIL");
        if (!pass) any_fail = 1;
    }

    return any_fail;
}
