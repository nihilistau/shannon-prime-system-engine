/* sp_compute_crt_imp.c — §3-HX Sprint K v0.beta Stage 2.5 Barrett primitives.
 *
 * Scalar Barrett (2.5a) — uint64_t arithmetic on the cDSP scalar pipe,
 * mirrors the math of ptx_ntt.cuh::barrett_reduce32_ref at engine 63d7e2d.
 * Compile-time q + μ baked per prime; matches Phase 2-CU.PTX constants.
 *
 * HVX vector Barrett (2.5b) — to be added in a follow-on commit per the
 * Stage 2.5 plan amendment.  This file's HVX section will use the
 * PTX→HVX intrinsic mapping table from
 * papers/SESSION-PLAN-lat-3-hx-mode-k-beta-AMENDMENT-stage-2-5.md §1.
 */
#include <stdint.h>
#include <string.h>
#include "HAP_farf.h"
#include "sp_compute.h"

/* Frozen primes (Phase 2-CU.PTX engine 63d7e2d / Sprint K v0.beta plan §1). */
#define SP_NTT_Q1   1073738753u
#define SP_NTT_Q2   1073732609u
/* μ = floor(2^60 / q), precomputed per prime. */
#define SP_MU_Q1    1073744895u
#define SP_MU_Q2    1073751039u

/* Scalar Barrett reduction matching engine 63d7e2d:ptx_ntt.cuh
 * barrett_reduce32_ref byte-for-byte.  Algorithm:
 *   qhat = ((x >> 29) * mu) >> 31
 *   r    = x - qhat * q     // 0 <= r < 3*q < 2^32
 *   r %= q (at most 2 conditional subtracts)
 *
 * x   : i64 product (here taken as uint64_t since a,b ∈ [0, q))
 * q   : 30-bit prime
 * mu  : floor(2^60 / q), 30-bit
 * Returns: canonical r ∈ [0, q).
 */
static inline uint32_t sp_barrett_reduce32_scalar(uint64_t x, uint32_t q, uint32_t mu) {
    uint64_t qhat = ((x >> 29u) * (uint64_t)mu) >> 31u;
    uint64_t r    = x - qhat * (uint64_t)q;
    if (r >= (uint64_t)q) r -= (uint64_t)q;
    if (r >= (uint64_t)q) r -= (uint64_t)q;
    return (uint32_t)r;
}

static inline uint32_t sp_modmul_scalar_q1(uint32_t a, uint32_t b) {
    return sp_barrett_reduce32_scalar((uint64_t)a * (uint64_t)b, SP_NTT_Q1, SP_MU_Q1);
}

static inline uint32_t sp_modmul_scalar_q2(uint32_t a, uint32_t b) {
    return sp_barrett_reduce32_scalar((uint64_t)a * (uint64_t)b, SP_NTT_Q2, SP_MU_Q2);
}

/* ─────────────────────────────────────────────────────────────────────────
 * sp_compute_barrett_oracle — §3-HX Sprint K v0.beta T_BARRETT_SCALAR_ORACLE.
 *
 * Drives N test vectors (a_i, b_i) through the scalar Barrett mod-mul for
 * the selected prime (q_idx=0→q_1, q_idx=1→q_2; mode=0→scalar; mode=1→HVX
 * vector, reserved for Stage 2.5b).  Outputs r_i = (a_i * b_i) mod q.
 *
 * Inputs are u32 in [0, q).  Outputs are u32 in [0, q).  Buffer layouts are
 * native u32 little-endian (matches the Rust harness side).
 * ───────────────────────────────────────────────────────────────────────── */
int sp_compute_barrett_oracle(remote_handle64 h,
                              int q_idx, int mode,
                              const unsigned char *a_buf, int a_bufLen,
                              const unsigned char *b_buf, int b_bufLen,
                              unsigned char       *r_buf, int r_bufLen)
{
    (void)h;
    if (q_idx < 0 || q_idx > 1) return -1;
    if (mode  < 0 || mode  > 1) return -1;
    if (!a_buf || !b_buf || !r_buf) return -1;
    if (a_bufLen != b_bufLen || a_bufLen != r_bufLen) return -1;
    if ((a_bufLen % 4) != 0) return -1;
    int n = a_bufLen / 4;

    const uint32_t *a = (const uint32_t *)a_buf;
    const uint32_t *b = (const uint32_t *)b_buf;
    uint32_t       *r = (uint32_t *)      r_buf;

    if (mode == 0) {
        /* Stage 2.5a — scalar Barrett. */
        if (q_idx == 0) {
            for (int i = 0; i < n; i++) r[i] = sp_modmul_scalar_q1(a[i], b[i]);
        } else {
            for (int i = 0; i < n; i++) r[i] = sp_modmul_scalar_q2(a[i], b[i]);
        }
    } else {
        /* Stage 2.5b — HVX vector Barrett.  Placeholder: returns -1 until the
         * intrinsic chain lands in a follow-on commit per the amendment. */
        FARF(RUNTIME_ERROR, "sp_compute_barrett_oracle: mode=1 (HVX) reserved for Stage 2.5b");
        return -1;
    }

    FARF(RUNTIME_HIGH,
         "sp_compute_barrett_oracle: q_idx=%d mode=%d n=%d r[0]=%u r[%d]=%u",
         q_idx, mode, n, r[0], n-1, r[n-1]);
    return 0;
}
