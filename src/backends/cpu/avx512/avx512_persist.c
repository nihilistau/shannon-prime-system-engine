/* avx512_persist.c — §18.5 PERSIST: UMONITOR/UMWAIT cache-line sentinel.
 *
 * Wait primitive: on WAITPKG hardware (Tiger Lake+), _umonitor registers a
 * hardware watch on the sentinel cache-line; _umwait suspends the logical
 * processor in C0.2 sub-state until the line is dirtied or deadline fires.
 * On Zen 4 / no-WAITPKG, falls back to _mm_pause spin (same API, lower quality).
 *
 * The sentinel page is VirtualLock'd on Windows so the cache-line is always
 * resident (UMWAIT is undefined if the monitored address is paged out).
 */
#include "sp_engine/avx512.h"
#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#endif

/* Internal layout: 64-byte cache-line at the start of a VirtualLock'd 4KB page. */
struct sp_avx512_persist_sentinel {
    volatile uint32_t trigger;  /* written by _wake; monitored by _wait */
    char _pad[60];              /* pad to one full cache line             */
    /* remaining ~4032 bytes of the page unused */
};

/* ---- TSC calibration ---------------------------------------------------- */

static uint64_t g_tsc_hz;

static void calibrate_tsc_once(void) {
    if (g_tsc_hz) return;
#ifdef _WIN32
    LARGE_INTEGER qpc_freq, qpc_a, qpc_b;
    QueryPerformanceFrequency(&qpc_freq);
    QueryPerformanceCounter(&qpc_a);
    uint64_t tsc_a = (uint64_t)__rdtsc();
    Sleep(100);
    uint64_t tsc_b = (uint64_t)__rdtsc();
    QueryPerformanceCounter(&qpc_b);
    double sec = (double)(qpc_b.QuadPart - qpc_a.QuadPart) / (double)qpc_freq.QuadPart;
    g_tsc_hz = (uint64_t)((double)(tsc_b - tsc_a) / sec);
#else
    /* POSIX fallback: clock_gettime */
    struct timespec ta, tb;
    clock_gettime(CLOCK_MONOTONIC, &ta);
    uint64_t tsc_a = (uint64_t)__rdtsc();
    /* busy-wait 10ms to avoid scheduler granularity issues */
    struct timespec tc;
    do { clock_gettime(CLOCK_MONOTONIC, &tc); }
    while ((tc.tv_sec - ta.tv_sec) * 1000000000LL + (tc.tv_nsec - ta.tv_nsec) < 10000000LL);
    uint64_t tsc_b = (uint64_t)__rdtsc();
    clock_gettime(CLOCK_MONOTONIC, &tb);
    double sec = (double)(tb.tv_sec - ta.tv_sec) +
                 (double)(tb.tv_nsec - ta.tv_nsec) * 1e-9;
    g_tsc_hz = (uint64_t)((double)(tsc_b - tsc_a) / sec);
#endif
}

static uint64_t ns_to_tsc(uint64_t ns) {
    return ns * g_tsc_hz / 1000000000ULL;
}

/* ---- Allocation ---------------------------------------------------------- */

sp_avx512_persist_sentinel *sp_avx512_persist_alloc(void) {
    calibrate_tsc_once();
#ifdef _WIN32
    void *page = VirtualAlloc(NULL, 4096, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!page) return NULL;
    if (!VirtualLock(page, 4096)) {
        VirtualFree(page, 0, MEM_RELEASE);
        return NULL;
    }
    sp_avx512_persist_sentinel *s = (sp_avx512_persist_sentinel *)page;
    s->trigger = 0;
    return s;
#else
    /* POSIX: mmap + mlock */
#  include <sys/mman.h>
    void *page = mmap(NULL, 4096, PROT_READ|PROT_WRITE,
                      MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if (page == MAP_FAILED) return NULL;
    if (mlock(page, 4096) != 0) { munmap(page, 4096); return NULL; }
    sp_avx512_persist_sentinel *s = (sp_avx512_persist_sentinel *)page;
    s->trigger = 0;
    return s;
#endif
}

void sp_avx512_persist_free(sp_avx512_persist_sentinel *s) {
    if (!s) return;
#ifdef _WIN32
    VirtualUnlock(s, 4096);
    VirtualFree(s, 0, MEM_RELEASE);
#else
    munmap(s, 4096);
#endif
}

/* ---- Wait (WAITPKG path) ------------------------------------------------- */

/* Function-level target so the TU can be compiled without -mwaitpkg globally. */
__attribute__((target("waitpkg")))
static void persist_wait_waitpkg(volatile uint32_t *p, uint32_t expect,
                                  uint64_t timeout_ns) {
    uint64_t deadline = (uint64_t)__rdtsc() + ns_to_tsc(timeout_ns);
    while (__atomic_load_n(p, __ATOMIC_ACQUIRE) == expect) {
        _umonitor((void *)(uintptr_t)p);
        /* Re-check after arming the monitor — the producer may have written
         * between the load above and _umonitor; without this check we would
         * sleep past an already-fired write. */
        if (__atomic_load_n(p, __ATOMIC_ACQUIRE) != expect) break;
        _umwait(0, deadline);  /* C0.2: fastest exit */
        if ((uint64_t)__rdtsc() >= deadline) break;
    }
}

/* ---- Wait (spin fallback) ------------------------------------------------ */

static void persist_wait_spin(volatile uint32_t *p, uint32_t expect,
                               uint64_t timeout_ns) {
    uint64_t deadline = (uint64_t)__rdtsc() + ns_to_tsc(timeout_ns);
    while (__atomic_load_n(p, __ATOMIC_ACQUIRE) == expect) {
        _mm_pause();
        if ((uint64_t)__rdtsc() >= deadline) break;
    }
}

/* ---- Public API ---------------------------------------------------------- */

void sp_avx512_persist_wait(sp_avx512_persist_sentinel *s, uint64_t timeout_ns) {
    uint32_t old = __atomic_load_n(&s->trigger, __ATOMIC_ACQUIRE);
    if (g_avx512_caps.has_waitpkg)
        persist_wait_waitpkg(&s->trigger, old, timeout_ns);
    else
        persist_wait_spin(&s->trigger, old, timeout_ns);
}

void sp_avx512_persist_wake(sp_avx512_persist_sentinel *s) {
    __atomic_fetch_add(&s->trigger, 1u, __ATOMIC_RELEASE);
}
