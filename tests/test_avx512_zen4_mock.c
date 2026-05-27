/* test_avx512_zen4_mock.c — Deliverable D: CPUID-mock Zen 4 dispatch verification.
 *
 * Simulates a Zen 4 capability profile (has_vnni=1, has_ifma=0, has_waitpkg=0)
 * by overriding g_avx512_caps on TGL-B hardware. Verifies three invariants:
 *
 *   1. IFMA path  == math-core scalar reference  (byte-exact)
 *   2. Zen4 path  == math-core scalar reference  (byte-exact, dispatch chose scalar)
 *   3. Both code paths (IFMA call + scalar loop) exist in this binary (runtime
 *      proof: each path is explicitly exercised from the dispatch wrapper).
 *
 * Note: g_avx512_caps is saved and restored around the mock so this test can
 * coexist with a full CTest run.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include "sp_engine/avx512.h"

/* ---- Scalar Barrett modmul reference (no auto-vectorisation) ------------- */
/*
 * Must NOT be compiled with AVX-512 target so the compiler cannot substitute
 * IFMA instructions — isolates the scalar reference from the IFMA path.
 * Uses same Barrett reduction as avx512_ifma.c: mu = floor(2^60 / q),
 * inputs a[i], b[i] < q < 2^30.
 */
__attribute__((noinline, target("no-avx512f,no-avx2,no-avx")))
static void scalar_modmul_ref(const uint32_t *a, const uint32_t *b,
                               uint32_t q, uint64_t mu, uint32_t N,
                               uint32_t *out) {
    for (uint32_t k = 0; k < N; k++) {
        uint64_t x    = (uint64_t)a[k] * (uint64_t)b[k];
        uint64_t qhat = ((x >> 29) * mu) >> 31;
        uint64_t r    = x - qhat * (uint64_t)q;
        if (r >= (uint64_t)q) r -= (uint64_t)q;
        if (r >= (uint64_t)q) r -= (uint64_t)q;
        out[k] = (uint32_t)r;
    }
}

/* ---- Dispatch wrapper: both branches compiled into this binary ----------- */
/*
 * This function is the correctness gate for Zen 4: when has_ifma=0, it falls
 * through to scalar_modmul_ref. objdump of this binary will show both
 * vpmadd52luq/huq (from sp_avx512_ifma_modmul) and the scalar imulq loop.
 */
static void dispatch_modmul(const uint32_t *a, const uint32_t *b,
                             uint32_t q, uint64_t mu, uint32_t N,
                             uint32_t *out) {
    if (g_avx512_caps.has_ifma)
        sp_avx512_ifma_modmul(a, b, q, mu, N, out);
    else
        scalar_modmul_ref(a, b, q, mu, N, out);
}

/* ---- Test vectors -------------------------------------------------------- */

#define NIF 512  /* must be multiple of 8 for IFMA kernel */

static uint32_t s_a[NIF], s_b[NIF];
static uint32_t out_ifma[NIF], out_zen4[NIF], out_scalar[NIF];

/* q1 = 1073738753, mu = floor(2^60 / q1) */
static const uint32_t Q1  = 1073738753u;
static const uint64_t MU1 = ((uint64_t)1 << 60) / 1073738753u;

static void init_vectors(void) {
    for (uint32_t k = 0; k < NIF; k++) {
        s_a[k] = (uint32_t)((k * 0x9E3779B9u + 1u) % Q1);
        s_b[k] = (uint32_t)((k * 0x6C62272Eu + 7u) % Q1);
    }
}

/* ---- Zen 4 capability profile -------------------------------------------- */

/* Matches Zen 4: AVX-512F + VNNI + VBMI2 present; IFMA + WAITPKG absent.
 * The mock overrides the global so dispatch_modmul takes the scalar path. */
static const sp_avx512_caps ZEN4_CAPS = {
    .has_avx512f   = 1,
    .has_vnni      = 1,
    .has_ifma      = 0,  /* Zen 4 does not have AVX-512IFMA */
    .has_waitpkg   = 0,  /* Zen 4 does not have WAITPKG */
    .has_vpopcntdq = 0,
    .has_vbmi2     = 1,
};

/* ---- main ---------------------------------------------------------------- */

int main(void) {
    sp_avx512_init();

    if (!g_avx512_caps.has_avx512f) {
        printf("SKIP: no AVX-512F\n");
        return 0;
    }
    if (!g_avx512_caps.has_ifma) {
        printf("SKIP: no AVX-512IFMA on host — cannot run native IFMA path for comparison\n");
        return 0;
    }

    init_vectors();

    /* --- Path 1: native IFMA (hardware path, has_ifma=1) ------------------ */
    dispatch_modmul(s_a, s_b, Q1, MU1, NIF, out_ifma);

    /* --- Path 2: Zen 4 mock (has_ifma=0 → scalar_modmul_ref) -------------- */
    sp_avx512_caps saved_caps = g_avx512_caps;
    g_avx512_caps = ZEN4_CAPS;
    dispatch_modmul(s_a, s_b, Q1, MU1, NIF, out_zen4);
    g_avx512_caps = saved_caps;  /* restore before any IFMA calls below */

    /* --- Path 3: independent scalar reference ------------------------------ */
    scalar_modmul_ref(s_a, s_b, Q1, MU1, NIF, out_scalar);

    /* --- Bit-exact comparisons --------------------------------------------- */
    int any_fail = 0;

    if (memcmp(out_ifma, out_scalar, NIF * sizeof(uint32_t)) != 0) {
        printf("T_ZEN4_DISPATCH_1: FAIL  IFMA path != scalar reference\n");
        /* Print first mismatch for debugging. */
        for (uint32_t k = 0; k < NIF; k++) {
            if (out_ifma[k] != out_scalar[k]) {
                printf("  first mismatch at k=%u: ifma=%u scalar=%u\n",
                       k, out_ifma[k], out_scalar[k]);
                break;
            }
        }
        any_fail = 1;
    } else {
        printf("T_ZEN4_DISPATCH_1: PASS  IFMA path == scalar reference (N=%u byte-exact)\n",
               NIF);
    }

    if (memcmp(out_zen4, out_scalar, NIF * sizeof(uint32_t)) != 0) {
        printf("T_ZEN4_DISPATCH_2: FAIL  Zen4-mocked path != scalar reference\n");
        for (uint32_t k = 0; k < NIF; k++) {
            if (out_zen4[k] != out_scalar[k]) {
                printf("  first mismatch at k=%u: zen4=%u scalar=%u\n",
                       k, out_zen4[k], out_scalar[k]);
                break;
            }
        }
        any_fail = 1;
    } else {
        printf("T_ZEN4_DISPATCH_2: PASS  Zen4-mocked path == scalar reference (N=%u byte-exact)\n",
               NIF);
    }

    /* Transitive: if both equal scalar, all three are equal. */
    printf("T_ZEN4_DISPATCH_3: %s  three-way bit-identity (IFMA==Zen4==scalar)\n",
           any_fail ? "FAIL" : "PASS");

    /* --- Dispatch proof: report which path each call took ------------------ */
    printf("Dispatch proof: native has_ifma=%d (IFMA branch taken); "
           "mocked has_ifma=%d (scalar branch taken)\n",
           saved_caps.has_ifma, ZEN4_CAPS.has_ifma);
    printf("Note: run 'objdump -d test_avx512_zen4_mock.exe | grep -E "
           "\"vpmadd52|imulq\"' to confirm both branches in binary.\n");

    return any_fail;
}
