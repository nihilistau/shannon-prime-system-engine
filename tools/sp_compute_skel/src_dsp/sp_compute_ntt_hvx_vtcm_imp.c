/* sp_compute_ntt_hvx_vtcm_imp.c — §4-NTT Sprint NTT.3 VTCM-aware HVX
 * NTT butterfly core (method 17).
 *
 * Same algorithmic contract and same primIn shape as NTT.1's
 * `sp_compute_ntt_hvx_oracle` (method 13). The difference: per-call
 * `find_psi` + `psi_pow` precompute + `w_fwd` precompute + per-stage
 * stride-step → stride-1 compaction are all ELIMINATED. Instead this
 * handler consumes the NTT.2 VTCM-resident tables via the skel-internal
 * `sp_compute_ntt_twiddle_view` accessor.
 *
 * Caller contract: ARM-side daemon MUST have called `ntt_twiddle_init`
 * (method 14) at least once before invoking method 17 — otherwise this
 * handler returns -1.
 *
 * Per CLOSURE-NTT-2.md §"Per-stage compaction layout":
 *   w_fwd_stages: stage-major layout, stage s has half_s = 2^{s-1} entries
 *   at entry-offset (2^{s-1} - 1) from compacted region base.
 *
 * Inheritance from NTT.1's HVX kernel (sp_compute_ntt_hvx_imp.c):
 *   - Same per-lane Barrett primitive shape (Q6_W_vmpye + Q6_W_vmpyoacc
 *     widening pair per reference-hexagon-v69-32x32-widening-idiom)
 *   - Same modadd / modsub primitives
 *   - Same large/small stage split (half >= 32 → HVX; half < 32 → scalar)
 *   - Same scalar pre-weight + bit-reversal (CLOSURE-NTT-1.md option (i))
 *
 * Per-TU isolation note: local copies of `barrett_reduce`, `modmul_s`,
 * `modadd_s`, `modsub_s`, `bitrev_u32` and the HVX Barrett/modadd/modsub
 * primitives, to keep the SASS-audit boundary of NTT.1's TU
 * (HVX_NTT_SASS_GATES.md) un-disturbed by NTT.3 changes. Math is
 * byte-identical to NTT.1.
 */
#include <stdint.h>
#include <string.h>
#include "HAP_farf.h"
#include "hexagon_types.h"
#include "hexagon_protos.h"
#include "hvx_hexagon_protos.h"
#include "sp_compute.h"

/* Frozen primes (mirror NTT.0/1 + sp_compute_crt_imp.c). */
#define SP_NTT3_Q1   1073738753u
#define SP_NTT3_Q2   1073732609u
#define SP_NTT3_MU_Q1    1073744895u
#define SP_NTT3_MU_Q2    1073751039u
#define SP_NTT3_Q_BITS   30u

#define SP_NTT3_N_MAX    512

/* ─── Forward declaration of NTT.2's skel-internal view accessor ──────────
 * Defined in sp_compute_ntt_twiddle.c. Pure C-internal API (no IDL).
 * Struct layout mirrored here to avoid a header dependency that NTT.2
 * intentionally did not export (see CLOSURE-NTT-2.md "Memory entry
 * candidates" #2).
 */
typedef struct {
    uint32_t        N;
    uint32_t        q;
    uint64_t        mu;
    uint32_t        ninv;
    const uint32_t *psi_pow;
    const uint32_t *ipsi_pow;
    const uint32_t *w_fwd;
    const uint32_t *w_inv;
    const uint32_t *w_fwd_stages;
    const uint32_t *w_inv_stages;
} sp_tw_view_ntt3;

/* Resolved at link time against the symbol defined in sp_compute_ntt_twiddle.c.
 * Struct layout is identical to that file's `sp_tw_view` (verified via
 * matched field names + types). */
extern int sp_compute_ntt_twiddle_view(int q_idx, int N, sp_tw_view_ntt3 *out);

/* ─── Scalar modular arithmetic (small-stage fallback + pre-weight + bitrev) */
static inline uint64_t sp_ntt3_barrett_reduce(uint64_t x, uint64_t q, uint64_t mu) {
    uint64_t qhat = ((x >> (SP_NTT3_Q_BITS - 1u)) * mu) >> (SP_NTT3_Q_BITS + 1u);
    uint64_t r = x - qhat * q;
    if (r >= q) r -= q;
    if (r >= q) r -= q;
    return r;
}

static inline uint32_t sp_ntt3_modmul_s(uint32_t a, uint32_t b, uint32_t q, uint64_t mu) {
    uint64_t x = (uint64_t)a * (uint64_t)b;
    return (uint32_t)sp_ntt3_barrett_reduce(x, (uint64_t)q, mu);
}

static inline uint32_t sp_ntt3_modadd_s(uint32_t a, uint32_t b, uint32_t q) {
    uint32_t s = a + b;
    if (s >= q) s -= q;
    return s;
}

static inline uint32_t sp_ntt3_modsub_s(uint32_t a, uint32_t b, uint32_t q) {
    return (a >= b) ? (a - b) : (a + q - b);
}

static inline uint32_t sp_ntt3_ilog2(uint32_t n) {
    uint32_t l = 0;
    while ((1u << l) < n) l++;
    return l;
}

static inline uint32_t sp_ntt3_bitrev(uint32_t x, uint32_t logN) {
    uint32_t r = 0;
    for (uint32_t b = 0; b < logN; b++) { r = (r << 1) | (x & 1u); x >>= 1; }
    return r;
}

/* ─── HVX per-lane Barrett (TU-local copy of NTT.1's primitive) ───────── */
static inline HVX_Vector sp_ntt3_barrett_reduce32_hvx_lane(
    HVX_Vector va, HVX_Vector vb,
    HVX_Vector vq, HVX_Vector vq_minus_1, HVX_Vector vmu)
{
    HVX_VectorPair x_pair = Q6_W_vmpye_VwVuh(va, vb);
    x_pair = Q6_W_vmpyoacc_WVwVh(x_pair, va, vb);
    HVX_Vector x_lo = Q6_V_lo_W(x_pair);
    HVX_Vector x_hi = Q6_V_hi_W(x_pair);

    HVX_Vector sh = Q6_V_vor_VV(
        Q6_Vuw_vlsr_VuwR(x_lo, 29),
        Q6_Vw_vasl_VwR(x_hi, 3));

    HVX_VectorPair q_pair = Q6_W_vmpye_VwVuh(sh, vmu);
    q_pair = Q6_W_vmpyoacc_WVwVh(q_pair, sh, vmu);
    HVX_Vector q_lo = Q6_V_lo_W(q_pair);
    HVX_Vector q_hi = Q6_V_hi_W(q_pair);

    HVX_Vector qhat = Q6_V_vor_VV(
        Q6_Vuw_vlsr_VuwR(q_lo, 31),
        Q6_Vw_vasl_VwR(q_hi, 1));

    HVX_VectorPair qq_pair = Q6_W_vmpye_VwVuh(qhat, vq);
    qq_pair = Q6_W_vmpyoacc_WVwVh(qq_pair, qhat, vq);
    HVX_Vector qq_lo = Q6_V_lo_W(qq_pair);

    HVX_Vector r0 = Q6_Vw_vsub_VwVw(x_lo, qq_lo);

    HVX_VectorPred gt0 = Q6_Q_vcmp_gt_VuwVuw(r0, vq_minus_1);
    HVX_Vector    r1   = Q6_V_vmux_QVV(gt0, Q6_Vw_vsub_VwVw(r0, vq), r0);
    HVX_VectorPred gt1 = Q6_Q_vcmp_gt_VuwVuw(r1, vq_minus_1);
    HVX_Vector    r2   = Q6_V_vmux_QVV(gt1, Q6_Vw_vsub_VwVw(r1, vq), r1);

    return r2;
}

static inline HVX_Vector sp_ntt3_modadd_hvx_lane(
    HVX_Vector a, HVX_Vector b, HVX_Vector vq, HVX_Vector vq_m1)
{
    HVX_Vector     sum = Q6_Vw_vadd_VwVw(a, b);
    HVX_VectorPred gt  = Q6_Q_vcmp_gt_VuwVuw(sum, vq_m1);
    HVX_Vector     sub = Q6_Vw_vsub_VwVw(sum, vq);
    return Q6_V_vmux_QVV(gt, sub, sum);
}

static inline HVX_Vector sp_ntt3_modsub_hvx_lane(
    HVX_Vector a, HVX_Vector b, HVX_Vector vq)
{
    HVX_Vector     diff = Q6_Vw_vsub_VwVw(a, b);
    HVX_VectorPred lt   = Q6_Q_vcmp_gt_VuwVuw(b, a);
    HVX_Vector     add  = Q6_Vw_vadd_VwVw(diff, vq);
    return Q6_V_vmux_QVV(lt, add, diff);
}

/* ─── Large-stage butterfly (half >= 32). Identical inner-loop shape to
 * NTT.1's sp_ntt_butterfly_stage_hvx; consumes a stride-1 compacted twiddle
 * pointer. The VTCM-resident `w_fwd_stages[stage_offset_entries(s)..]`
 * already IS stride-1 (per CLOSURE-NTT-2.md compaction layout), so the
 * caller passes a pointer into that table directly — no per-call copy.
 * ───────────────────────────────────────────────────────────────────── */
static void sp_ntt3_butterfly_stage_hvx(uint32_t *out, uint32_t N,
                                        uint32_t len, uint32_t half,
                                        const uint32_t *w_compact,
                                        HVX_Vector vq, HVX_Vector vq_m1,
                                        HVX_Vector vmu)
{
    uint32_t vecs_per_group = half / 32u;
    const HVX_Vector *w_vp = (const HVX_Vector *)w_compact;

    for (uint32_t i = 0; i < N; i += len) {
        HVX_Vector       *uvp = (HVX_Vector *)      (out + i);
        HVX_Vector       *vvp = (HVX_Vector *)      (out + i + half);
        for (uint32_t v = 0; v < vecs_per_group; v++) {
            HVX_Vector u_vec = uvp[v];
            HVX_Vector v_raw = vvp[v];
            HVX_Vector w_vec = w_vp[v];
            HVX_Vector v_red = sp_ntt3_barrett_reduce32_hvx_lane(
                                   v_raw, w_vec, vq, vq_m1, vmu);
            HVX_Vector u_out = sp_ntt3_modadd_hvx_lane(u_vec, v_red, vq, vq_m1);
            HVX_Vector u_lo  = sp_ntt3_modsub_hvx_lane(u_vec, v_red, vq);
            uvp[v] = u_out;
            vvp[v] = u_lo;
        }
    }
}

/* ─── Small-stage butterfly (half < 32). Scalar; byte-identical to NTT.1.
 * Consumes the VTCM-resident w_fwd_stages too (stage-major compacted), so
 * inner loop is stride-1 (no `widx += step` step variable here vs NTT.1's
 * scalar fallback).
 * ───────────────────────────────────────────────────────────────────── */
static void sp_ntt3_butterfly_stage_scalar(uint32_t *out, uint32_t N,
                                           uint32_t len, uint32_t half,
                                           const uint32_t *w_compact,
                                           uint32_t q, uint64_t mu)
{
    for (uint32_t i = 0; i < N; i += len) {
        for (uint32_t k = 0; k < half; k++) {
            uint32_t u = out[i + k];
            uint32_t v = sp_ntt3_modmul_s(out[i + k + half], w_compact[k], q, mu);
            out[i + k]        = sp_ntt3_modadd_s(u, v, q);
            out[i + k + half] = sp_ntt3_modsub_s(u, v, q);
        }
    }
}

/* ─── Stage offset helper (matches CLOSURE-NTT-2.md §"Per-stage compaction
 * layout"): stage s has half_s = 2^{s-1} entries at entry-offset
 * (2^{s-1} - 1) from the compacted region base.
 * ───────────────────────────────────────────────────────────────────── */
static inline uint32_t sp_ntt3_stage_offset_entries(uint32_t s) {
    /* s in [1, logN]; offset = 2^{s-1} - 1 */
    return (1u << (s - 1u)) - 1u;
}

/* ─── Per-prime forward NTT (VTCM-aware) ──────────────────────────────── */
static void sp_ntt3_forward_one_hvx_vtcm(uint32_t N, uint32_t logN,
                                         uint32_t q, uint64_t mu,
                                         const uint32_t *psi_pow,
                                         const uint32_t *w_fwd_stages,
                                         const int32_t *in,
                                         uint32_t *out)
{
    /* Step 1 — pre-weight. Reads psi_pow directly from VTCM. */
    for (uint32_t j = 0; j < N; j++) {
        int64_t v = (int64_t)in[j] % (int64_t)q;
        if (v < 0) v += (int64_t)q;
        out[j] = sp_ntt3_modmul_s((uint32_t)v, psi_pow[j], q, mu);
    }

    /* Step 2 — bit-reversal permutation (scalar). */
    for (uint32_t i = 0; i < N; i++) {
        uint32_t j = sp_ntt3_bitrev(i, logN);
        if (i < j) { uint32_t t = out[i]; out[i] = out[j]; out[j] = t; }
    }

    /* Step 3 — logN stages. */
    HVX_Vector vq    = Q6_V_vsplat_R((int32_t)q);
    HVX_Vector vq_m1 = Q6_V_vsplat_R((int32_t)(q - 1u));
    HVX_Vector vmu   = Q6_V_vsplat_R((int32_t)mu);

    uint32_t stage = 1u;
    for (uint32_t len = 2u; len <= N; len <<= 1, stage++) {
        uint32_t half = len >> 1;
        uint32_t off_entries = sp_ntt3_stage_offset_entries(stage);
        const uint32_t *w_compact = w_fwd_stages + off_entries;
        if (half >= 32u) {
            sp_ntt3_butterfly_stage_hvx(out, N, len, half, w_compact,
                                        vq, vq_m1, vmu);
        } else {
            sp_ntt3_butterfly_stage_scalar(out, N, len, half, w_compact,
                                           q, mu);
        }
    }
}

/* ─── IDL handler — sp_compute_ntt_hvx_vtcm_oracle (method 17, anticipated) */

/* primIn layout (qaic-emitted): 4 x i32 = 16 bytes — IDENTICAL to NTT.0 / NTT.1
 *   [0] q_idx        (int32; 0 -> q_1, 1 -> q_2)
 *   [1] N            (int32; must be 128, 256, or 512)
 *   [2] data_inLen   (int32; expected N * 4)
 *   [3] data_outLen  (int32; expected N * 4)
 *
 * Returns 0 on success, -1 on any input violation OR if the VTCM table for
 * (q_idx, N) is not present (caller must invoke ntt_twiddle_init first).
 */
int sp_compute_ntt_hvx_vtcm_oracle(remote_handle64 h,
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

    sp_tw_view_ntt3 v;
    int rc = sp_compute_ntt_twiddle_view(q_idx, N, &v);
    if (rc != 0) {
        FARF(RUNTIME_ERROR,
             "sp_compute_ntt_hvx_vtcm_oracle: twiddle table not present "
             "q_idx=%d N=%d rc=%d -- caller must ntt_twiddle_init first",
             q_idx, N, rc);
        return -1;
    }

    /* Sanity check against frozen primes — the view should match. */
    uint32_t expected_q  = (q_idx == 0) ? SP_NTT3_Q1 : SP_NTT3_Q2;
    uint64_t expected_mu = (q_idx == 0) ? (uint64_t)SP_NTT3_MU_Q1 : (uint64_t)SP_NTT3_MU_Q2;
    if (v.q != expected_q || v.mu != expected_mu || (uint32_t)N != v.N) {
        FARF(RUNTIME_ERROR,
             "sp_compute_ntt_hvx_vtcm_oracle: view mismatch q_idx=%d N=%d "
             "v.q=%u v.mu=%lu v.N=%u",
             q_idx, N, v.q, (unsigned long)v.mu, v.N);
        return -1;
    }

    uint32_t Nu = (uint32_t)N;
    uint32_t logN = sp_ntt3_ilog2(Nu);

    /* Local aligned IO buffers. Caller may have non-128-B-aligned data. */
    int32_t  in_local[SP_NTT3_N_MAX]  __attribute__((aligned(128)));
    uint32_t out_local[SP_NTT3_N_MAX] __attribute__((aligned(128)));
    memcpy(in_local, data_in, (size_t)N * 4u);

    sp_ntt3_forward_one_hvx_vtcm(Nu, logN, v.q, v.mu,
                                 v.psi_pow, v.w_fwd_stages,
                                 in_local, out_local);

    memcpy(data_out, out_local, (size_t)N * 4u);

    FARF(RUNTIME_HIGH,
         "sp_compute_ntt_hvx_vtcm_oracle: q_idx=%d N=%d out[0]=%u out[N-1]=%u",
         q_idx, N, out_local[0], out_local[Nu - 1u]);
    return 0;
}
