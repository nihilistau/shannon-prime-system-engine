/* sp_compute_ntt_hvx_imp.c — §4-NTT Sprint NTT.1 HVX-vectorized NTT
 * butterfly core.
 *
 * Implements `sp_compute_ntt_hvx_oracle` (IDL method 13). Same primIn
 * shape and contract as `sp_compute_ntt_oracle` (method 12, NTT.0 scalar
 * reference); algorithm identical at the math layer (negacyclic
 * Cooley-Tukey radix-2 DIT NTT mod a frozen Proth prime), with the
 * radix-2 butterflies vectorized via hand-rolled HVX intrinsics.
 *
 * Vectorization strategy:
 *   - Pre-weight (Step 1)              : SCALAR (small relative to butterflies)
 *   - Bit-reversal permutation (Step 2): SCALAR (data-dependent indices)
 *   - Stages with half >= 32           : HVX-vectorized (large-stage path)
 *   - Stages with half <  32           : SCALAR fallback (small-stage path)
 *
 * Per-stage twiddle compaction:
 *   The butterfly inner loop accesses w_fwd[k * step] with stride step =
 *   N/len. For HVX vmem loads we need stride-1 access, so each large
 *   stage compacts its twiddles into a 128-B-aligned scratch array
 *   `w_compact[half]` with w_compact[k] = w_fwd[k * step]. NTT.2 will
 *   lift this to VTCM-resident precomputed per-stage tables; the kernel
 *   here doesn't need to change when that lands — only the source
 *   pointer for the compacted twiddles.
 *
 * Per-lane Barrett:
 *   Re-declared locally as `sp_barrett_reduce32_hvx_lane_ntt1` to keep
 *   this TU's SASS audit (HVX_NTT_SASS_GATES.md) isolated from
 *   sp_compute_crt_imp.c (Stage 2.5b's audit). Math byte-identical to
 *   the silicon-confirmed K.beta.2.5b primitive — same q, mu, shifts,
 *   2-op Q6_W_vmpye + Q6_W_vmpyoacc widening idiom (per
 *   reference-hexagon-v69-32x32-widening-idiom).
 *
 * Correctness gates (T_NTT1_HVX_BIT_EXACT, T_NTT1_NO_REGRESSION):
 *   - vs `sp_compute_ntt_oracle` (method 12, NTT.0 scalar)
 *   - vs math-core `ntt_forward` per-prime channel (transitive via NTT.0)
 *   Same 600-run sweep: N ∈ {128, 256, 512} × q_idx ∈ {0, 1} × 100 seeds.
 */
#include <stdint.h>
#include <string.h>
#include "HAP_farf.h"
#include "hexagon_types.h"
#include "hexagon_protos.h"
#include "hvx_hexagon_protos.h"
#include "sp_compute.h"

/* Frozen primes (mirror sp_compute_ntt_imp.c / sp_compute_crt_imp.c). */
#define SP_NTT_Q1   1073738753u
#define SP_NTT_Q2   1073732609u
#define SP_MU_Q1    1073744895u
#define SP_MU_Q2    1073751039u
#define SP_NTT_Q_BITS 30u

/* Max N per 2-adic valuation (reference-ntt-frozen-primes-N-cap). */
#define SP_NTT_N_MAX 512

/* ─────────────────────────────────────────────────────────────────────
 * Scalar modular arithmetic (reused as-is for small-stage fallback +
 * pre-weight + bit-reversal). Byte-identical to NTT.0
 * sp_compute_ntt_imp.c:42-94.
 * ───────────────────────────────────────────────────────────────────── */
static inline uint64_t barrett_reduce(uint64_t x, uint64_t q, uint64_t mu) {
    uint64_t qhat = ((x >> (SP_NTT_Q_BITS - 1u)) * mu) >> (SP_NTT_Q_BITS + 1u);
    uint64_t r = x - qhat * q;
    if (r >= q) r -= q;
    if (r >= q) r -= q;
    return r;
}

static inline uint32_t modmul_s(uint32_t a, uint32_t b, uint32_t q, uint64_t mu) {
    uint64_t x = (uint64_t)a * (uint64_t)b;
    return (uint32_t)barrett_reduce(x, (uint64_t)q, mu);
}

static inline uint32_t modadd_s(uint32_t a, uint32_t b, uint32_t q) {
    uint32_t s = a + b;
    if (s >= q) s -= q;
    return s;
}

static inline uint32_t modsub_s(uint32_t a, uint32_t b, uint32_t q) {
    return (a >= b) ? (a - b) : (a + q - b);
}

static uint32_t modpow(uint32_t base, uint64_t e, uint32_t q, uint64_t mu) {
    uint32_t result = 1u % q;
    uint32_t b = base % q;
    while (e) {
        if (e & 1u) result = modmul_s(result, b, q, mu);
        b = modmul_s(b, b, q, mu);
        e >>= 1;
    }
    return result;
}

static uint32_t find_psi(uint32_t N, uint32_t q, uint64_t mu) {
    uint64_t exp = ((uint64_t)q - 1u) / (2u * (uint64_t)N);
    for (uint32_t a = 2u; a < q; a++) {
        uint32_t psi = modpow(a, exp, q, mu);
        if (psi == 0u) continue;
        uint32_t pN = modpow(psi, (uint64_t)N, q, mu);
        if (pN == q - 1u) return psi;
    }
    return 0u;
}

static inline uint32_t ilog2_u32(uint32_t n) {
    uint32_t l = 0;
    while ((1u << l) < n) l++;
    return l;
}

static inline uint32_t bitrev_u32(uint32_t x, uint32_t logN) {
    uint32_t r = 0;
    for (uint32_t b = 0; b < logN; b++) { r = (r << 1) | (x & 1u); x >>= 1; }
    return r;
}

/* ─────────────────────────────────────────────────────────────────────
 * HVX per-lane Barrett (§4-NTT NTT.1 local copy of K.beta.2.5b primitive).
 *
 * 32-lane u32 Barrett reduction. Math identical to scalar barrett_reduce:
 *   qhat = ((x >> 29) * mu) >> 31
 *   r    = x_lo - qhat * q   (mod 2^32)
 *   r %= q via two conditional subtracts
 *
 * Uses the V69 32×32→64 widening idiom Q6_W_vmpye + Q6_W_vmpyoacc per
 * reference-hexagon-v69-32x32-widening-idiom (SASS-confirmed Sprint
 * K v0.beta-2.5b @ 0822747 in engine-kbeta-2-5b worktree).
 * ───────────────────────────────────────────────────────────────────── */
static inline HVX_Vector sp_barrett_reduce32_hvx_lane_ntt1(
    HVX_Vector va, HVX_Vector vb,
    HVX_Vector vq, HVX_Vector vq_minus_1, HVX_Vector vmu)
{
    /* 1. x = a*b (32×32→64 per lane). */
    HVX_VectorPair x_pair = Q6_W_vmpye_VwVuh(va, vb);
    x_pair = Q6_W_vmpyoacc_WVwVh(x_pair, va, vb);
    HVX_Vector x_lo = Q6_V_lo_W(x_pair);
    HVX_Vector x_hi = Q6_V_hi_W(x_pair);

    /* 2. sh = x >> 29 = (x_lo >> 29) | (x_hi << 3) */
    HVX_Vector sh = Q6_V_vor_VV(
        Q6_Vuw_vlsr_VuwR(x_lo, 29),
        Q6_Vw_vasl_VwR(x_hi, 3));

    /* 3. q_pair = sh * mu (u31 × u31 → u62) */
    HVX_VectorPair q_pair = Q6_W_vmpye_VwVuh(sh, vmu);
    q_pair = Q6_W_vmpyoacc_WVwVh(q_pair, sh, vmu);
    HVX_Vector q_lo = Q6_V_lo_W(q_pair);
    HVX_Vector q_hi = Q6_V_hi_W(q_pair);

    /* 4. qhat = q_pair >> 31 = (q_lo >> 31) | (q_hi << 1) */
    HVX_Vector qhat = Q6_V_vor_VV(
        Q6_Vuw_vlsr_VuwR(q_lo, 31),
        Q6_Vw_vasl_VwR(q_hi, 1));

    /* 5. qq_lo = low 32 bits of (qhat * q) */
    HVX_VectorPair qq_pair = Q6_W_vmpye_VwVuh(qhat, vq);
    qq_pair = Q6_W_vmpyoacc_WVwVh(qq_pair, qhat, vq);
    HVX_Vector qq_lo = Q6_V_lo_W(qq_pair);

    /* 6. r0 = x_lo - qq_lo (modular sub mod 2^32). r0 ∈ [0, 3q). */
    HVX_Vector r0 = Q6_Vw_vsub_VwVw(x_lo, qq_lo);

    /* 7. Two Barrett corrections. */
    HVX_VectorPred gt0 = Q6_Q_vcmp_gt_VuwVuw(r0, vq_minus_1);
    HVX_Vector    r1   = Q6_V_vmux_QVV(gt0, Q6_Vw_vsub_VwVw(r0, vq), r0);
    HVX_VectorPred gt1 = Q6_Q_vcmp_gt_VuwVuw(r1, vq_minus_1);
    HVX_Vector    r2   = Q6_V_vmux_QVV(gt1, Q6_Vw_vsub_VwVw(r1, vq), r1);

    return r2;
}

/* ─────────────────────────────────────────────────────────────────────
 * HVX modular add: c = (a + b) mod q. a, b ∈ [0, q) ⇒ a+b ∈ [0, 2q-1).
 * One conditional subtract suffices.
 *   sum = a + b
 *   c   = (sum > q-1) ? (sum - q) : sum
 *   Encoded via Q6_V_vmux_QVV.
 * ───────────────────────────────────────────────────────────────────── */
static inline HVX_Vector sp_modadd_hvx_lane_ntt1(
    HVX_Vector a, HVX_Vector b, HVX_Vector vq, HVX_Vector vq_m1)
{
    HVX_Vector     sum = Q6_Vw_vadd_VwVw(a, b);
    HVX_VectorPred gt  = Q6_Q_vcmp_gt_VuwVuw(sum, vq_m1);
    HVX_Vector     sub = Q6_Vw_vsub_VwVw(sum, vq);
    return Q6_V_vmux_QVV(gt, sub, sum);
}

/* ─────────────────────────────────────────────────────────────────────
 * HVX modular sub: c = (a - b) mod q. a, b ∈ [0, q).
 *   if a >= b: c = a - b
 *   else:      c = a - b + q
 * Unsigned-wrap diff = a - b. Correct when a >= b; otherwise add q.
 * Predicate is "b > a" (lane-wise unsigned-gt).
 * ───────────────────────────────────────────────────────────────────── */
static inline HVX_Vector sp_modsub_hvx_lane_ntt1(
    HVX_Vector a, HVX_Vector b, HVX_Vector vq)
{
    HVX_Vector     diff = Q6_Vw_vsub_VwVw(a, b);
    HVX_VectorPred lt   = Q6_Q_vcmp_gt_VuwVuw(b, a);
    HVX_Vector     add  = Q6_Vw_vadd_VwVw(diff, vq);
    return Q6_V_vmux_QVV(lt, add, diff);
}

/* ─────────────────────────────────────────────────────────────────────
 * Large-stage butterfly (half >= 32). Processes a full radix-2 DIT
 * stage with the HVX inner loop.
 *
 * Per the math-core ntt_core (ntt_crt.c:241-254) algorithm, for one
 * stage with `len`, `half = len/2`, `step = N/len`:
 *
 *   for i in [0, N) step len:                # group offset
 *     widx = 0
 *     for k in [0, half):                    # within-group butterfly
 *       u = out[i+k]
 *       v = (out[i+k+half] * w[widx]) mod q
 *       out[i+k]      = (u + v) mod q
 *       out[i+k+half] = (u - v) mod q
 *       widx += step
 *
 * The HVX path processes 32 butterflies per inner iteration via the
 * compacted twiddle array.
 * ───────────────────────────────────────────────────────────────────── */
static void sp_ntt_butterfly_stage_hvx(uint32_t *out, uint32_t N,
                                       uint32_t len, uint32_t half,
                                       const uint32_t *w_compact,
                                       HVX_Vector vq, HVX_Vector vq_m1,
                                       HVX_Vector vmu)
{
    /* half is a multiple of 32 here (caller ensures half >= 32, and
     * half is always a power of 2). */
    uint32_t vecs_per_group = half / 32u;
    const HVX_Vector *w_vp = (const HVX_Vector *)w_compact;

    for (uint32_t i = 0; i < N; i += len) {
        HVX_Vector       *uvp = (HVX_Vector *)      (out + i);
        HVX_Vector       *vvp = (HVX_Vector *)      (out + i + half);
        for (uint32_t v = 0; v < vecs_per_group; v++) {
            HVX_Vector u_vec = uvp[v];
            HVX_Vector v_raw = vvp[v];
            HVX_Vector w_vec = w_vp[v];
            /* v_red = (v_raw * w_vec) mod q, 32-lane parallel. */
            HVX_Vector v_red = sp_barrett_reduce32_hvx_lane_ntt1(
                                   v_raw, w_vec, vq, vq_m1, vmu);
            /* Modular add → out[i+k];  modular sub → out[i+k+half]. */
            HVX_Vector u_out = sp_modadd_hvx_lane_ntt1(u_vec, v_red, vq, vq_m1);
            HVX_Vector u_lo  = sp_modsub_hvx_lane_ntt1(u_vec, v_red, vq);
            uvp[v] = u_out;
            vvp[v] = u_lo;
        }
    }
}

/* ─────────────────────────────────────────────────────────────────────
 * Small-stage butterfly (half < 32). Pure scalar — byte-identical to
 * NTT.0 ntt_forward_one_scalar's Step 3 inner body.
 * ───────────────────────────────────────────────────────────────────── */
static void sp_ntt_butterfly_stage_scalar(uint32_t *out, uint32_t N,
                                          uint32_t len, uint32_t half,
                                          uint32_t step,
                                          const uint32_t *w_fwd,
                                          uint32_t q, uint64_t mu)
{
    for (uint32_t i = 0; i < N; i += len) {
        uint32_t widx = 0;
        for (uint32_t k = 0; k < half; k++) {
            uint32_t u = out[i + k];
            uint32_t v = modmul_s(out[i + k + half], w_fwd[widx], q, mu);
            out[i + k]        = modadd_s(u, v, q);
            out[i + k + half] = modsub_s(u, v, q);
            widx += step;
        }
    }
}

/* ─────────────────────────────────────────────────────────────────────
 * Per-prime forward NTT — HVX path for large stages, scalar fallback for
 * small stages. Mirrors NTT.0 ntt_forward_one_scalar Step 1 + Step 2 +
 * Step 3, with Step 3 dispatched per-stage.
 * ───────────────────────────────────────────────────────────────────── */
static void ntt_forward_one_hvx(uint32_t N, uint32_t logN,
                                uint32_t q, uint64_t mu,
                                const uint32_t *psi_pow,
                                const uint32_t *w_fwd,
                                const int32_t *in,
                                uint32_t *out)
{
    /* Step 1 — pre-weight (scalar). Same as NTT.0. */
    for (uint32_t j = 0; j < N; j++) {
        int64_t v = (int64_t)in[j] % (int64_t)q;
        if (v < 0) v += (int64_t)q;
        out[j] = modmul_s((uint32_t)v, psi_pow[j], q, mu);
    }

    /* Step 2 — bit-reversal permutation (scalar). Same as NTT.0. */
    for (uint32_t i = 0; i < N; i++) {
        uint32_t j = bitrev_u32(i, logN);
        if (i < j) { uint32_t t = out[i]; out[i] = out[j]; out[j] = t; }
    }

    /* Step 3 — logN stages of radix-2 DIT butterflies.
     * Splat HVX scalars ONCE here, reuse across all large stages. */
    HVX_Vector vq    = Q6_V_vsplat_R((int32_t)q);
    HVX_Vector vq_m1 = Q6_V_vsplat_R((int32_t)(q - 1u));
    HVX_Vector vmu   = Q6_V_vsplat_R((int32_t)mu);

    /* Per-stage compaction scratch. half_max = N/2 = 256 at N=512.
     * 256 * 4 = 1024 B; comfortably stack-resident; 128-B aligned for
     * HVX vmem. NTT.2 lifts this to VTCM-resident precomputed tables. */
    uint32_t w_compact[SP_NTT_N_MAX / 2] __attribute__((aligned(128)));

    for (uint32_t len = 2; len <= N; len <<= 1) {
        uint32_t half = len >> 1;
        uint32_t step = N / len;

        if (half >= 32u) {
            /* Compact stride-step twiddles into stride-1 array. */
            for (uint32_t k = 0; k < half; k++) {
                w_compact[k] = w_fwd[k * step];
            }
            sp_ntt_butterfly_stage_hvx(out, N, len, half, w_compact,
                                       vq, vq_m1, vmu);
        } else {
            sp_ntt_butterfly_stage_scalar(out, N, len, half, step,
                                          w_fwd, q, mu);
        }
    }
}

/* ─────────────────────────────────────────────────────────────────────
 * IDL handler — sp_compute_ntt_hvx_oracle (method 13).
 *
 * primIn layout (qaic-emitted): 4 x i32 = 16 bytes — IDENTICAL to NTT.0
 *   [0] q_idx        (int32; 0 -> q_1, 1 -> q_2)
 *   [1] N            (int32; must be 128, 256, or 512)
 *   [2] data_inLen   (int32; expected N * 4)
 *   [3] data_outLen  (int32; expected N * 4)
 *
 * Returns 0 on success, -1 on any input violation.
 * ───────────────────────────────────────────────────────────────────── */
int sp_compute_ntt_hvx_oracle(remote_handle64 h,
                              int q_idx, int N,
                              const unsigned char *data_in,  int data_inLen,
                              unsigned char       *data_out, int data_outLen)
{
    (void)h;
    if (q_idx < 0 || q_idx > 1) return -1;
    if (N != 128 && N != 256 && N != 512) return -1;
    if (!data_in || !data_out) return -1;
    if (data_inLen  < N * 4) return -1;
    if (data_outLen < N * 4) return -1;

    uint32_t Nu = (uint32_t)N;
    uint32_t logN = ilog2_u32(Nu);
    uint32_t q  = (q_idx == 0) ? SP_NTT_Q1 : SP_NTT_Q2;
    uint64_t mu = (q_idx == 0) ? (uint64_t)SP_MU_Q1 : (uint64_t)SP_MU_Q2;

    /* Twiddle precompute — same algorithm as NTT.0; lifted into the same
     * stack-resident scratch (≤ 3 KB at N=512). NTT.2 will lift these
     * into VTCM-resident precomputed tables. */
    uint32_t psi_pow[SP_NTT_N_MAX] __attribute__((aligned(128)));
    uint32_t w_fwd[SP_NTT_N_MAX / 2] __attribute__((aligned(128)));

    uint32_t psi = find_psi(Nu, q, mu);
    if (psi == 0u) {
        FARF(RUNTIME_ERROR,
             "sp_compute_ntt_hvx_oracle: find_psi failed q_idx=%d N=%d", q_idx, N);
        return -1;
    }
    uint32_t omega = modmul_s(psi, psi, q, mu);

    uint32_t acc = 1u % q;
    for (uint32_t j = 0; j < Nu; j++) {
        psi_pow[j] = acc;
        acc = modmul_s(acc, psi, q, mu);
    }
    acc = 1u % q;
    uint32_t half = Nu / 2u;
    for (uint32_t j = 0; j < half; j++) {
        w_fwd[j] = acc;
        acc = modmul_s(acc, omega, q, mu);
    }

    /* Per NTT.0: caller's bytes may not be u32-aligned in the worst
     * case; memcpy through aligned local buffers avoids UB. Also
     * required for HVX vmem loads (128-B alignment via attribute). */
    int32_t  in_local[SP_NTT_N_MAX]  __attribute__((aligned(128)));
    uint32_t out_local[SP_NTT_N_MAX] __attribute__((aligned(128)));
    memcpy(in_local, data_in, (size_t)N * 4u);

    ntt_forward_one_hvx(Nu, logN, q, mu, psi_pow, w_fwd, in_local, out_local);

    memcpy(data_out, out_local, (size_t)N * 4u);

    FARF(RUNTIME_HIGH,
         "sp_compute_ntt_hvx_oracle: q_idx=%d N=%d psi=%u out[0]=%u out[N-1]=%u",
         q_idx, N, psi, out_local[0], out_local[Nu - 1u]);
    return 0;
}
