/* sp_compute_crt_imp.c — §3-HX Sprint K v0.beta Stage 2.5 Barrett primitives.
 *
 * Scalar Barrett (2.5a) — uint64_t arithmetic on the cDSP scalar pipe,
 * mirrors the math of ptx_ntt.cuh::barrett_reduce32_ref at engine 63d7e2d.
 * Compile-time q + μ baked per prime; matches Phase 2-CU.PTX constants.
 *
 * HVX vector Barrett (2.5b) — 32-lane u32 Barrett reduction via the V69
 * 32×32→64 widening idiom: vmpye + vmpyoacc.  See
 *   tools/sp_compute_skel/docs/PLAN-K-beta-2-5b.md
 *   tools/sp_compute_skel/docs/HVX_BARRETT_MAPPING.md
 * for the intrinsic mapping (extended/corrected from the AMENDMENT plan §1)
 * and reference/hexagon_v69_hvx.extracted.txt:5577-5586 for the source ISA
 * note on the widening idiom.
 */
#include <stdint.h>
#include <string.h>
#include "HAP_farf.h"
#include "HAP_perf.h"
#include "hexagon_types.h"
#include "hexagon_protos.h"
#include "hvx_hexagon_protos.h"
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
 * HVX vector Barrett (Stage 2.5b) — 32 u32 lanes per HVX_Vector.
 *
 * Math identical to sp_barrett_reduce32_scalar:
 *     qhat = ((x >> 29) * mu) >> 31
 *     r    = x - qhat * q       // in [0, 3q) < 2^32
 *     r %= q via two conditional subtracts
 *
 * V69 implementation uses the 32×32→64 widening idiom:
 *     pair = vmpye(a, b)            // Vdd: lo=(a*b)&0xFFFFFFFF, hi=(a*b)>>32
 *     pair += vmpyo(a, b)           // accumulator completes the widening
 * per Hexagon HVX Programmer's Reference Manual §151.
 *
 * sp_barrett_reduce32_hvx_lane: per-32-lane Barrett, parameters splatted by
 * caller (q, q-1, mu) to amortize splat cost across the loop.
 * ───────────────────────────────────────────────────────────────────────── */
static inline HVX_Vector sp_barrett_reduce32_hvx_lane(
    HVX_Vector va, HVX_Vector vb,
    HVX_Vector vq, HVX_Vector vq_minus_1, HVX_Vector vmu)
{
    /* 1. x = a*b as a 64-bit pair (lo, hi) per word lane.
     *    Per ISA: Vdd = vmpye(Vu.w, Vv.uh) sets v[0]=lo32 (shifted form),
     *    v[1]=hi32; Vxx += vmpyo(Vu.w, Vv.h) accumulates the high-half
     *    cross-product, completing the full 64-bit widening. */
    HVX_VectorPair x_pair = Q6_W_vmpye_VwVuh(va, vb);
    x_pair = Q6_W_vmpyoacc_WVwVh(x_pair, va, vb);
    HVX_Vector x_lo = Q6_V_lo_W(x_pair);
    HVX_Vector x_hi = Q6_V_hi_W(x_pair);

    /* 2. sh = x >> 29. Since x < 2^60, sh < 2^31 — fits in u32.
     *    sh = (x_lo >> 29) | (x_hi << 3) */
    HVX_Vector sh = Q6_V_vor_VV(
        Q6_Vuw_vlsr_VuwR(x_lo, 29),
        Q6_Vw_vasl_VwR(x_hi, 3));

    /* 3. q_pair = sh * mu (u31 × u31 → u62) */
    HVX_VectorPair q_pair = Q6_W_vmpye_VwVuh(sh, vmu);
    q_pair = Q6_W_vmpyoacc_WVwVh(q_pair, sh, vmu);
    HVX_Vector q_lo = Q6_V_lo_W(q_pair);
    HVX_Vector q_hi = Q6_V_hi_W(q_pair);

    /* 4. qhat = q_pair >> 31. Since q_pair < 2^62, qhat < 2^31.
     *    qhat = (q_lo >> 31) | (q_hi << 1) */
    HVX_Vector qhat = Q6_V_vor_VV(
        Q6_Vuw_vlsr_VuwR(q_lo, 31),
        Q6_Vw_vasl_VwR(q_hi, 1));

    /* 5. qq_lo = low 32 bits of (qhat * q).  Same widening idiom; we only
     *    use the lo half because x_lo - qq_lo is exact mod 2^32 (the high
     *    parts cancel in true arithmetic). */
    HVX_VectorPair qq_pair = Q6_W_vmpye_VwVuh(qhat, vq);
    qq_pair = Q6_W_vmpyoacc_WVwVh(qq_pair, qhat, vq);
    HVX_Vector qq_lo = Q6_V_lo_W(qq_pair);

    /* 6. r0 = x_lo - qq_lo (modular sub).  r0 ∈ [0, 3q) ⊂ [0, 2^32) since
     *    q < 2^30. */
    HVX_Vector r0 = Q6_Vw_vsub_VwVw(x_lo, qq_lo);

    /* 7. Two Barrett corrections.  Unsigned compare: r > (q-1) ≡ r ≥ q. */
    HVX_VectorPred gt0 = Q6_Q_vcmp_gt_VuwVuw(r0, vq_minus_1);
    HVX_Vector    r1   = Q6_V_vmux_QVV(gt0, Q6_Vw_vsub_VwVw(r0, vq), r0);
    HVX_VectorPred gt1 = Q6_Q_vcmp_gt_VuwVuw(r1, vq_minus_1);
    HVX_Vector    r2   = Q6_V_vmux_QVV(gt1, Q6_Vw_vsub_VwVw(r1, vq), r1);

    return r2;
}

/* Run the HVX Barrett over n u32 lanes (n MUST be a multiple of 32 — one
 * HVX_Vector = 128 B = 32 u32 lanes).  Buffer pointers MUST be 128-B aligned.
 * Returns 0 on success, -1 on input violation. */
static int sp_barrett_vec_run(int q_idx,
                              const uint32_t *a, const uint32_t *b,
                              uint32_t *r, int n)
{
    if ((n % 32) != 0) return -1;
    if (((uintptr_t)a & 127u) || ((uintptr_t)b & 127u) || ((uintptr_t)r & 127u))
        return -1;

    uint32_t q  = (q_idx == 0) ? SP_NTT_Q1 : SP_NTT_Q2;
    uint32_t mu = (q_idx == 0) ? SP_MU_Q1  : SP_MU_Q2;
    HVX_Vector vq         = Q6_V_vsplat_R((int32_t)q);
    HVX_Vector vq_minus_1 = Q6_V_vsplat_R((int32_t)(q - 1u));
    HVX_Vector vmu        = Q6_V_vsplat_R((int32_t)mu);

    const HVX_Vector *va_p = (const HVX_Vector *)a;
    const HVX_Vector *vb_p = (const HVX_Vector *)b;
    HVX_Vector       *vr_p = (HVX_Vector *)      r;
    int n_vecs = n / 32;
    for (int i = 0; i < n_vecs; i++) {
        HVX_Vector va = va_p[i];
        HVX_Vector vb = vb_p[i];
        vr_p[i] = sp_barrett_reduce32_hvx_lane(va, vb, vq, vq_minus_1, vmu);
    }
    return 0;
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
        /* Stage 2.5b — HVX vector Barrett.
         * Buffer pointers from FastRPC come from DmaBuffer / RPCMEM_TRY_MAP_STATIC
         * allocations — RPC mem heap 25 produces 128-B-aligned pointers (see
         * reference-mode-d-bridge-architecture).  Vector path requires (n % 32 == 0)
         * AND 128-byte aligned pointers; falls back to -1 on either violation. */
        unsigned long long t0 = HAP_perf_get_pcycles();
        int rc = sp_barrett_vec_run(q_idx, a, b, r, n);
        unsigned long long t1 = HAP_perf_get_pcycles();
        if (rc != 0) {
            FARF(RUNTIME_ERROR,
                 "sp_compute_barrett_oracle: mode=1 vector path rejected (n=%d, alignment a=%p b=%p r=%p)",
                 n, (const void *)a, (const void *)b, (const void *)r);
            return rc;
        }
        FARF(RUNTIME_HIGH,
             "sp_compute_barrett_oracle: q_idx=%d mode=1 n=%d r[0]=%u r[%d]=%u pcycles=%llu",
             q_idx, n, r[0], n-1, r[n-1], (t1 - t0));
        return 0;
    }

    FARF(RUNTIME_HIGH,
         "sp_compute_barrett_oracle: q_idx=%d mode=%d n=%d r[0]=%u r[%d]=%u",
         q_idx, mode, n, r[0], n-1, r[n-1]);
    return 0;
}

/* ─────────────────────────────────────────────────────────────────────────
 * Stage 2.5c — HVX mod_q matmul kernel.
 *
 *   Y[b][i] = ( Σ_{k=0..d_in} X[b][k] * W[k][i] ) mod q
 *
 * All elements u32 in [0, q).  Layout row-major:
 *   X[b][k] = x[b*d_in + k]
 *   W[k][i] = w[k*d_out + i]
 *   Y[b][i] = y[b*d_out + i]
 *
 * Algorithm (per PLAN-K-beta-2-5c.md §Architectural choices, path C):
 *   Reuses sp_barrett_reduce32_hvx_lane silicon-confirmed in Stage 2.5b.
 *   Inner loop over k:
 *     x_splat = splat(X[b][k])               // 1 op (hoistable per (b,k))
 *     w_vec   = vmem(W[k] + i_out)           // 1 load
 *     prod    = barrett_reduce32_hvx_lane(   // 19 ops (SASS-audited Stage 2.5b)
 *                  x_splat, w_vec, vq, vq_m1, vmu)
 *     sum     = vadd(acc, prod)              // 1 op,  in [0, 2q-1)
 *     gte     = vcmp.gt(sum, vq_m1)          // 1 op,  predicate sum >= q
 *     acc     = vmux(gte, vsub(sum, vq), sum) // 2 ops (vsub for vmux input + vmux)
 *
 *   Total per inner-loop iteration: 19 + 5 = 24 intrinsics on the
 *   compute body, plus 1 vsplat per k and 1 vmem load.  Splats vq, vq_m1,
 *   vmu hoisted out of all loops (3 splats / invoke).
 *
 * Constraints (returns -1):
 *   q_idx ∉ {0, 1};  d_out % 32 != 0;  batch < 1 || d_in < 1 || d_out < 32;
 *   buffer pointers misaligned to 128 B;  buffer pointers null.
 *
 * Returns 0 on success, -1 on input violation.
 * ───────────────────────────────────────────────────────────────────────── */
static int sp_matmul_q_hvx(int q_idx,
                           int batch, int d_in, int d_out,
                           const uint32_t *x, const uint32_t *w, uint32_t *y)
{
    if (q_idx < 0 || q_idx > 1) return -1;
    if (batch < 1 || d_in < 1 || d_out < 32) return -1;
    if ((d_out % 32) != 0) return -1;
    if (!x || !w || !y) return -1;
    /* W and Y must be 128-B aligned for HVX vmem loads/stores at i_out
     * stride-32 boundaries.  X is u32-aligned scalar broadcasts only —
     * splat reads through scalar pipe, no alignment requirement. */
    if (((uintptr_t)w & 127u) || ((uintptr_t)y & 127u)) return -1;

    uint32_t q  = (q_idx == 0) ? SP_NTT_Q1 : SP_NTT_Q2;
    uint32_t mu = (q_idx == 0) ? SP_MU_Q1  : SP_MU_Q2;
    HVX_Vector vq    = Q6_V_vsplat_R((int32_t)q);
    HVX_Vector vq_m1 = Q6_V_vsplat_R((int32_t)(q - 1u));
    HVX_Vector vmu   = Q6_V_vsplat_R((int32_t)mu);
    HVX_Vector vzero = Q6_V_vzero();

    int n_vecs_per_row = d_out / 32;
    for (int b = 0; b < batch; b++) {
        HVX_Vector       *y_vp = (HVX_Vector *)      (y + b * d_out);
        for (int iv = 0; iv < n_vecs_per_row; iv++) {
            HVX_Vector acc = vzero;
            int i_out = iv * 32;
            for (int k = 0; k < d_in; k++) {
                /* Broadcast scalar X[b][k] into 32 lanes. */
                HVX_Vector x_splat = Q6_V_vsplat_R((int32_t)x[b * d_in + k]);
                /* Load 32 lanes W[k][i_out..i_out+32]. */
                const HVX_Vector *w_vp =
                    (const HVX_Vector *)(w + k * d_out + i_out);
                HVX_Vector w_vec = *w_vp;
                /* prod = (x_splat * w_vec) mod q, per-lane via 2.5b primitive. */
                HVX_Vector prod = sp_barrett_reduce32_hvx_lane(
                    x_splat, w_vec, vq, vq_m1, vmu);
                /* Modular add: sum = acc + prod  ∈ [0, 2q-1).
                 * Conditional subtract once if sum > q-1. */
                HVX_Vector sum = Q6_Vw_vadd_VwVw(acc, prod);
                HVX_VectorPred gte = Q6_Q_vcmp_gt_VuwVuw(sum, vq_m1);
                acc = Q6_V_vmux_QVV(gte, Q6_Vw_vsub_VwVw(sum, vq), sum);
            }
            y_vp[iv] = acc;
        }
    }
    return 0;
}

/* sp_compute_matmul_q — IDL method 11.
 *
 * primIn layout (qaic-emitted): 7 × i32 = 28 bytes
 *   [0] q_idx        (int32)
 *   [1] batch        (int32)
 *   [2] d_in         (int32)
 *   [3] d_out        (int32)
 *   [4] x_bufLen     (int32)
 *   [5] w_bufLen     (int32)
 *   [6] y_bufLen     (int32)
 *
 * Outputs (rout):
 *   y_buf:           u32 LE  (batch * d_out * 4 B)
 *   kernel_pcycles_lo, kernel_pcycles_hi:  scalar i32 each (low/high 32 bits)
 */
int sp_compute_matmul_q(remote_handle64 h,
                        int q_idx, int batch, int d_in, int d_out,
                        const unsigned char *x_buf, int x_bufLen,
                        const unsigned char *w_buf, int w_bufLen,
                        unsigned char       *y_buf, int y_bufLen,
                        int *kernel_pcycles_lo,
                        int *kernel_pcycles_hi)
{
    (void)h;
    if (q_idx < 0 || q_idx > 1) return -1;
    if (batch < 1 || d_in < 1 || d_out < 32) return -1;
    if ((d_out % 32) != 0) return -1;
    if (!x_buf || !w_buf || !y_buf) return -1;
    if (!kernel_pcycles_lo || !kernel_pcycles_hi) return -1;
    if (x_bufLen != batch * d_in * 4) return -1;
    if (w_bufLen != d_in * d_out * 4) return -1;
    if (y_bufLen != batch * d_out * 4) return -1;

    const uint32_t *x = (const uint32_t *)x_buf;
    const uint32_t *w = (const uint32_t *)w_buf;
    uint32_t       *y = (uint32_t *)      y_buf;

    unsigned long long t0 = HAP_perf_get_pcycles();
    int rc = sp_matmul_q_hvx(q_idx, batch, d_in, d_out, x, w, y);
    unsigned long long t1 = HAP_perf_get_pcycles();
    if (rc != 0) {
        FARF(RUNTIME_ERROR,
             "sp_compute_matmul_q: kernel rejected q_idx=%d B=%d D_in=%d D_out=%d (alignment w=%p y=%p)",
             q_idx, batch, d_in, d_out, (const void *)w, (void *)y);
        *kernel_pcycles_lo = 0;
        *kernel_pcycles_hi = 0;
        return rc;
    }
    unsigned long long delta = t1 - t0;
    *kernel_pcycles_lo = (int)(uint32_t)(delta & 0xFFFFFFFFu);
    *kernel_pcycles_hi = (int)(uint32_t)((delta >> 32) & 0xFFFFFFFFu);
    FARF(RUNTIME_HIGH,
         "sp_compute_matmul_q: q_idx=%d B=%d D_in=%d D_out=%d y[0]=%u y[last]=%u pcycles=%llu",
         q_idx, batch, d_in, d_out,
         y[0], y[batch * d_out - 1], delta);
    return 0;
}
