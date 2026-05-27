/* test_avx512_persist.c — §18.5 PERSIST gates for Phase 2-CPU.AVX.
 *
 * M_AVX_PERSIST_1: wakeup median ≤200ns (WAITPKG or spin fallback).
 * M_AVX_PERSIST_2: idle CPU cycles < 5% of wall-clock TSC cycles during a
 *                  50ms idle window (WAITPKG only; skipped if no WAITPKG).
 *
 * Methodology:
 *   Latency: main (core 0) vs worker (core 1); LFENCE-serialised RDTSC on both
 *   sides; 10μs arm window; N=1000 trials; sort for median.
 *   CPU idle: QueryThreadCycleTime over 50ms wall-clock; ratio < 0.05 required.
 */
#ifndef WIN32_LEAN_AND_MEAN
#  define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include "sp_engine/avx512.h"

/* ---- TSC calibration (test-local, independent of avx512_persist.c) ------- */

static uint64_t s_tsc_hz;

static void calibrate_tsc(void) {
    LARGE_INTEGER qpc_freq, qpc_a, qpc_b;
    QueryPerformanceFrequency(&qpc_freq);
    QueryPerformanceCounter(&qpc_a);
    uint64_t tsc_a = (uint64_t)__rdtsc();
    Sleep(100);
    uint64_t tsc_b = (uint64_t)__rdtsc();
    QueryPerformanceCounter(&qpc_b);
    double sec = (double)(qpc_b.QuadPart - qpc_a.QuadPart) / (double)qpc_freq.QuadPart;
    s_tsc_hz = (uint64_t)((double)(tsc_b - tsc_a) / sec);
}

static double tsc_to_ns(uint64_t ticks) {
    return (double)ticks * 1e9 / (double)s_tsc_hz;
}

/* ---- M_AVX_PERSIST_1: wakeup latency ------------------------------------ */

#define N_TRIALS 1000

typedef struct {
    sp_avx512_persist_sentinel *s;
    uint64_t t_wake[N_TRIALS];  /* RDTSC immediately after persist_wait returns  */
    volatile int phase;         /* main increments: "start trial i"              */
    volatile int ack;           /* worker increments: "I'm about to call wait"   */
} LatCtx;

static DWORD WINAPI latency_worker(LPVOID arg) {
    LatCtx *ctx = (LatCtx *)arg;
    SetThreadAffinityMask(GetCurrentThread(), 2u);  /* core 1 */

    for (int i = 0; i < N_TRIALS; i++) {
        /* Wait for main's start signal for this trial. */
        while (__atomic_load_n(&ctx->phase, __ATOMIC_ACQUIRE) != i + 1)
            _mm_pause();
        /* Acknowledge: we are about to call persist_wait. */
        __atomic_store_n(&ctx->ack, i + 1, __ATOMIC_RELEASE);
        /* Block until main wakes us (or 5ms timeout). */
        sp_avx512_persist_wait(ctx->s, 5000000ULL);
        /* Serialising fence before timestamp: prevents RDTSC speculating ahead
         * of the pipeline restart that follows UMWAIT/store exit. */
        _mm_lfence();
        ctx->t_wake[i] = (uint64_t)__rdtsc();
    }
    return 0;
}

static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

static int run_latency_gate(void) {
    LatCtx ctx;
    memset(&ctx, 0, sizeof(ctx));

    ctx.s = sp_avx512_persist_alloc();
    if (!ctx.s) {
        printf("M_AVX_PERSIST_1: SKIP (VirtualLock failed — no SeLockMemoryPrivilege?)\n");
        return 0;
    }

    /* Warm up: touch sentinel from both cores before measurement. */
    sp_avx512_persist_wake(ctx.s);
    sp_avx512_persist_wait(ctx.s, 1000000ULL);

    uint64_t t_signal[N_TRIALS];
    memset(t_signal, 0, sizeof(t_signal));

    HANDLE hw = CreateThread(NULL, 0, latency_worker, &ctx, 0, NULL);
    SetThreadAffinityMask(GetCurrentThread(), 1u);  /* core 0 */

    /* 10μs arm window in TSC cycles — enough time for worker to call _umonitor. */
    uint64_t arm_cycles = s_tsc_hz / 100000;

    for (int i = 0; i < N_TRIALS; i++) {
        /* Tell worker to start this trial. */
        __atomic_store_n(&ctx.phase, i + 1, __ATOMIC_RELEASE);
        /* Wait for worker ack (worker is now about to call persist_wait). */
        while (__atomic_load_n(&ctx.ack, __ATOMIC_ACQUIRE) != i + 1)
            _mm_pause();
        /* Spin 10μs so _umonitor is armed inside persist_wait. */
        uint64_t spin_end = (uint64_t)__rdtsc() + arm_cycles;
        while ((uint64_t)__rdtsc() < spin_end) _mm_pause();
        /* Serialise, record signal time, fire. */
        _mm_lfence();
        t_signal[i] = (uint64_t)__rdtsc();
        _mm_lfence();
        sp_avx512_persist_wake(ctx.s);
    }

    WaitForSingleObject(hw, INFINITE);
    CloseHandle(hw);

    /* Compute latencies, discard negatives (race: main fired before arm). */
    uint64_t lats[N_TRIALS];
    int valid = 0;
    for (int i = 0; i < N_TRIALS; i++) {
        if (ctx.t_wake[i] > t_signal[i])
            lats[valid++] = ctx.t_wake[i] - t_signal[i];
    }
    if (valid < N_TRIALS / 2) {
        printf("M_AVX_PERSIST_1: FAIL (too few valid samples: %d/%d)\n",
               valid, N_TRIALS);
        sp_avx512_persist_free(ctx.s);
        return 1;
    }
    qsort(lats, (size_t)valid, sizeof(uint64_t), cmp_u64);
    double median_ns = tsc_to_ns(lats[valid / 2]);
    int pass = (median_ns <= 200.0);
    printf("M_AVX_PERSIST_1: median=%.1fns  samples=%d/%d  [need<=200ns]  %s\n",
           median_ns, valid, N_TRIALS, pass ? "PASS" : "FAIL");

    sp_avx512_persist_free(ctx.s);
    return pass ? 0 : 1;
}

/* ---- M_AVX_PERSIST_2: idle CPU consumption (WAITPKG only) --------------- */

typedef struct {
    sp_avx512_persist_sentinel *s;
    volatile int running;
} IdleCtx;

static DWORD WINAPI idle_worker(LPVOID arg) {
    IdleCtx *ctx = (IdleCtx *)arg;
    SetThreadAffinityMask(GetCurrentThread(), 2u);  /* core 1 */
    __atomic_store_n(&ctx->running, 1, __ATOMIC_RELEASE);
    /* Block for up to 200ms — main will not fire. */
    sp_avx512_persist_wait(ctx->s, 200000000ULL);
    return 0;
}

static int run_idle_gate(void) {
    if (!g_avx512_caps.has_waitpkg) {
        printf("M_AVX_PERSIST_2: SKIP (no WAITPKG — spin fallback is expected to busy)\n");
        return 0;
    }

    IdleCtx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.s = sp_avx512_persist_alloc();
    if (!ctx.s) {
        printf("M_AVX_PERSIST_2: SKIP (VirtualLock failed)\n");
        return 0;
    }

    HANDLE hw = CreateThread(NULL, 0, idle_worker, &ctx, 0, NULL);
    /* Wait until worker is running. */
    while (__atomic_load_n(&ctx.running, __ATOMIC_ACQUIRE) == 0)
        _mm_pause();
    /* Brief extra delay to ensure UMWAIT is entered. */
    Sleep(5);

    ULONG64 cyc_before = 0, cyc_after = 0;
    QueryThreadCycleTime(hw, &cyc_before);
    Sleep(50);
    QueryThreadCycleTime(hw, &cyc_after);
    uint64_t idle_cycles = (uint64_t)(cyc_after - cyc_before);
    /* Wake the worker so it exits cleanly. */
    sp_avx512_persist_wake(ctx.s);
    WaitForSingleObject(hw, 1000);
    CloseHandle(hw);
    sp_avx512_persist_free(ctx.s);

    /* Wall-clock cycles for the 50ms measurement window. */
    uint64_t wall_cycles = s_tsc_hz / 20;
    double ratio = (double)idle_cycles / (double)wall_cycles;
    int pass = (ratio < 0.05);
    printf("M_AVX_PERSIST_2: idle_cycles=%llu  wall_cycles=%llu  ratio=%.3f  "
           "[need<0.05]  %s\n",
           (unsigned long long)idle_cycles,
           (unsigned long long)wall_cycles,
           ratio,
           pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

/* ---- main ---------------------------------------------------------------- */

int main(void) {
    sp_avx512_init();
    calibrate_tsc();

    printf("TSC: %.3f GHz\n", (double)s_tsc_hz / 1e9);
    printf("WAITPKG: %s\n", g_avx512_caps.has_waitpkg ? "yes" : "no (spin fallback)");

    if (!g_avx512_caps.has_avx512f) {
        printf("SKIP: no AVX-512F — PERSIST gates not applicable\n");
        return 0;
    }

    int fail = 0;
    fail |= run_latency_gate();
    fail |= run_idle_gate();
    return fail;
}
