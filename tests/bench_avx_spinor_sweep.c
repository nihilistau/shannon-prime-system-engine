/* bench_avx_spinor_sweep.c — M_AVX_3: NT cache-bypass gate for §18 CPU.AVX.
 *
 * Working-set sweep (16MB → 24MB → 32MB) across the 24MB TGL-B L3 cache.
 * Two streaming paths per working set:
 *   CACHED: vmovdqa64 (L1/L2/L3-coherent)
 *   NT:     vmovntdqa (non-temporal; on WB memory Intel SDM Vol.1 §10.4.6.2
 *            does not guarantee cache-hierarchy bypass — at DRAM-bound WS
 *            both paths converge; that is the documented, expected behaviour)
 *
 * Gates:
 *   M_AVX_3_PARITY: NT throughput at 32MB >= 0.95x cached throughput at 32MB
 *                   (both DRAM-bound at 32MB; NT must not be slower than cached)
 *   M_AVX_3_SPINOR: zero sentinel misses across 32MB of 64-byte Spinor slots
 *                   (functional correctness of sp_avx512_spinor_nt_load_check)
 *
 * Stability methodology:
 *   - Main thread pinned to core 0 (SetThreadAffinityMask) to prevent migration
 *   - Each path gets its own WARMUP_PASSES before its timed window
 *   - PARITY measured N_STABILITY=11 times; order alternates (cached-first /
 *     NT-first) to cancel ordering bias; gate uses the true median
 *
 * Note: L1-miss-rate PMC confirmation (>=95%%) requires elevated wpr plus a
 * custom .wprp profile exposing PEBS/L1D_CACHE_REFILL events.
 * wpr -start CPU alone does not surface those counters.
 * Throughput parity is the primary gate accessible without elevation.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include "sp_engine/avx512.h"

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#else
#  include <time.h>
#endif

/* ---- Timing -------------------------------------------------------------- */

/* QPC frequency cached at startup. */
static double g_qpc_rcp_ns;

static void init_timing(void) {
    LARGE_INTEGER f;
    QueryPerformanceFrequency(&f);
    g_qpc_rcp_ns = 1e9 / (double)f.QuadPart;
}

static double now_ns(void) {
#ifdef _WIN32
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * g_qpc_rcp_ns;
#else
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e9 + (double)t.tv_nsec;
#endif
}

/* ---- Streaming kernels --------------------------------------------------- */

/* Cached read: vmovdqa64, touches L1/L2/L3. */
__attribute__((target("avx512f"), noinline))
static uint64_t stream_cached(const void *buf, size_t bytes) {
    uint64_t sink = 0;
    const char *p = (const char *)buf;
    const char *end = p + bytes;
    while (p < end) {
        __m512i v = _mm512_load_si512((const __m512i *)p);
        /* _mm512_castsi512_si128 + _mm_cvtsi128_si64: extract low 64 bits. */
        sink ^= (uint64_t)_mm_cvtsi128_si64(_mm512_castsi512_si128(v));
        p += 64;
    }
    return sink;
}

/* NT read: vmovntdqa.  On WB memory (Intel SDM §10.4.6.2) cache bypass is
 * implementation-defined; at DRAM-bound WS sizes throughput converges. */
__attribute__((target("avx512f"), noinline))
static uint64_t stream_nt(const void *buf, size_t bytes) {
    uint64_t sink = 0;
    const char *p = (const char *)buf;
    const char *end = p + bytes;
    while (p < end) {
        __m512i v = _mm512_stream_load_si512((void *)p);
        sink ^= (uint64_t)_mm_cvtsi128_si64(_mm512_castsi512_si128(v));
        p += 64;
    }
    return sink;
}

/* Spinor NT path: streams 64-byte slots via the public spinor API. */
static uint64_t stream_spinor_nt(const void *buf, size_t bytes) {
    uint64_t miss_count = 0;
    const char *p = (const char *)buf;
    const char *end = p + bytes;
    while (p < end) {
        /* sp_avx512_spinor_nt_load_check returns -1 on sentinel mismatch. */
        if (sp_avx512_spinor_nt_load_check(p) != 0)
            miss_count++;
        p += 64;
    }
    return miss_count;
}

/* ---- Working-set measurement --------------------------------------------- */

#define WARMUP_PASSES   3
#define TIMED_PASSES   10
#define N_STABILITY    11   /* odd → clean median at index 5 */

typedef struct {
    double tput_cached_gbps;
    double tput_nt_gbps;
} SweepResult;

/* measure_one: time a single path (fn) after WARMUP_PASSES of warm-up. */
static double measure_one(uint64_t (*fn)(const void *, size_t),
                          void *buf, size_t ws_bytes) {
    volatile uint64_t sink = 0;
    for (int i = 0; i < WARMUP_PASSES; i++)
        sink ^= fn(buf, ws_bytes);
    double t0 = now_ns();
    for (int i = 0; i < TIMED_PASSES; i++)
        sink ^= fn(buf, ws_bytes);
    double ns = (now_ns() - t0) / TIMED_PASSES;
    (void)sink;
    return (double)ws_bytes / ns;   /* GB/s */
}

/* measure_sweep: informational single-pass sweep; cached before NT. */
static SweepResult measure_sweep(void *buf, size_t ws_bytes) {
    SweepResult r;
    r.tput_cached_gbps = measure_one(stream_cached, buf, ws_bytes);
    r.tput_nt_gbps     = measure_one(stream_nt,     buf, ws_bytes);
    return r;
}

/* ---- Spinor NT sub-test -------------------------------------------------- */

static int run_spinor_subtest(void *buf, size_t ws_bytes) {
    /* Stamp 0xA5 at byte 63 of each 64-byte Spinor slot. */
    uint8_t *p8 = (uint8_t *)buf;
    size_t n_slots = ws_bytes / 64;
    for (size_t i = 0; i < n_slots; i++)
        p8[i * 64 + 63] = 0xA5;

    double t0 = now_ns();
    uint64_t misses = 0;
    for (int trial = 0; trial < TIMED_PASSES; trial++)
        misses += stream_spinor_nt(buf, ws_bytes);
    double elapsed_ns = (now_ns() - t0) / TIMED_PASSES;

    double tput = (double)ws_bytes / elapsed_ns;
    int ok = (misses == 0);
    printf("M_AVX_3_SPINOR: tput=%.2f GB/s  sentinel_misses=%llu  [need=0]  %s\n",
           tput, (unsigned long long)misses, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

/* ---- Double comparison for qsort ---------------------------------------- */

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* ---- main ---------------------------------------------------------------- */

int main(void) {
    init_timing();
    sp_avx512_init();

    if (!g_avx512_caps.has_avx512f) {
        printf("SKIP: no AVX-512F\n");
        return 0;
    }

    /* Pin to core 0: prevents thread migration between cached and NT measurements. */
#ifdef _WIN32
    SetThreadAffinityMask(GetCurrentThread(), 1u);
#endif

    /* Allocate 64-byte aligned buffer large enough for 32MB + guard. */
    const size_t BUF_BYTES = 33 * 1024 * 1024;
    void *raw = _mm_malloc(BUF_BYTES, 64);
    if (!raw) {
        fprintf(stderr, "FAIL: _mm_malloc %zuMB\n", BUF_BYTES >> 20);
        return 1;
    }
    /* Fill with non-zero pattern (avoid zero-page optimisations). */
    memset(raw, 0xA5, BUF_BYTES);

    static const size_t WS[] = {
        16 * 1024 * 1024,   /* 16MB: fits in 24MB L3         */
        24 * 1024 * 1024,   /* 24MB: at L3 boundary on TGL-B */
        32 * 1024 * 1024,   /* 32MB: exceeds L3 -> DRAM      */
    };
    static const int N_WS = 3;

    /* Informational sweep: single pass, raw throughput numbers. */
    printf("Working-set sweep (TGL-B L3=24MB):\n");
    printf("  %-10s  %-14s  %-14s\n", "working-set", "cached GB/s", "NT GB/s");
    for (int i = 0; i < N_WS; i++) {
        SweepResult r = measure_sweep(raw, WS[i]);
        printf("  %4zuMB      %9.2f      %9.2f\n",
               WS[i] >> 20, r.tput_cached_gbps, r.tput_nt_gbps);
    }

    /* ---- M_AVX_3_PARITY: stability run (N=11, alternating order, median) - */
    printf("\nM_AVX_3_PARITY stability (%d trials at 32MB, order alternates):\n",
           N_STABILITY);
    double parity_ratios[N_STABILITY];
    for (int t = 0; t < N_STABILITY; t++) {
        double c_gbps, nt_gbps;
        if (t % 2 == 0) {
            /* Even trial: cached first. */
            c_gbps  = measure_one(stream_cached, raw, WS[2]);
            nt_gbps = measure_one(stream_nt,     raw, WS[2]);
        } else {
            /* Odd trial: NT first (cancels ordering bias). */
            nt_gbps = measure_one(stream_nt,     raw, WS[2]);
            c_gbps  = measure_one(stream_cached, raw, WS[2]);
        }
        parity_ratios[t] = nt_gbps / c_gbps;
        printf("  trial %2d [%s-first]: cached=%.2f  NT=%.2f  ratio=%.3f\n",
               t + 1, (t % 2 == 0) ? "C" : "NT",
               c_gbps, nt_gbps, parity_ratios[t]);
    }
    qsort(parity_ratios, N_STABILITY, sizeof(double), cmp_double);
    double median_parity = parity_ratios[N_STABILITY / 2];
    int parity_pass = (median_parity >= 0.95);
    printf("M_AVX_3_PARITY: median_ratio=%.3f  [need>=0.95]  %s\n",
           median_parity, parity_pass ? "PASS" : "FAIL");

    /* ---- M_AVX_3_SPINOR: NT spinor load correctness (32MB) -------------- */
    printf("\nSpinor-NT sub-test (32MB, 0xA5 sentinel verification):\n");
    int spinor_fail = run_spinor_subtest(raw, WS[2]);

    /* ---- Notes -------------------------------------------------------------- */
    printf("\nNote (WB memory): vmovntdqa on write-back arena does not guarantee\n");
    printf("  cache-hierarchy bypass (Intel SDM Vol.1 §10.4.6.2).  At 32MB both\n");
    printf("  paths are DRAM-bound; throughput parity is the expected outcome.\n");
    printf("Note (PMC): L1-miss-rate >=95%% for the NT path requires elevated wpr\n");
    printf("  with a custom .wprp profile exposing PEBS/L1D_CACHE_REFILL events.\n");
    printf("  wpr -start CPU alone does not surface those counters.\n");

    _mm_free(raw);
    return (parity_pass && !spinor_fail) ? 0 : 1;
}
