/* sp_compute_ntt_intt_imp.c — §4-NTT Sprint NTT.4 — INTT HVX skel handler.
 *
 * Implements `sp_compute_intt_hvx_oracle` (IDL method 18). Negacyclic
 * inverse NTT mod a frozen Proth prime, mirroring math-core's
 * `inverse_one` (lib/shannon-prime-system/core/ntt_crt/ntt_crt.c:281-294).
 *
 * Algorithm:
 *   1. out[j] = in[j] % q                                      (scalar)
 *   2. bit-reversal permutation                                (scalar)
 *   3. logN stages of radix-2 DIT butterflies with w_inv       (HVX large,
 *                                                               scalar small)
 *   4. post-pass:  out[j] = (out[j] * ninv) * ipsi_pow[j] mod q (scalar)
 *
 * The HVX butterfly is bit-identical to NTT.1's `sp_ntt_butterfly_stage_hvx`
 * (sp_compute_ntt_hvx_imp.c:224-256) — Cooley-Tukey radix-2 DIT, same
 * Barrett primitive, same modular add/sub primitives. The only difference
 * is that NTT.4 consumes the VTCM-resident `w_inv_stages` table populated
 * by NTT.2's `sp_compute_ntt_twiddle_init` (method 14), so the per-call
 * per-stage compaction loop NTT.1 does is eliminated — we point at the
 * appropriate stage offset directly.
 *
 * Anti-contamination: NTT.1's TU is NOT modified. The HVX helpers and
 * butterfly are intentionally duplicated here to keep the NTT.1
 * HVX_NTT_SASS_GATES.md audit valid (same TU, same emitted SASS).
 *
 * VTCM prerequisite: caller MUST have called `sp_compute_ntt_twiddle_init`
 * (method 14) before the first invocation. The IDL handler returns -1 if
 * `sp_compute_ntt_twiddle_view` reports the table not present.
 */

#include <stdint.h>
#include <string.h>
#include "HAP_farf.h"
#include "hexagon_types.h"
#include "hexagon_protos.h"
#include "hvx_hexagon_protos.h"
#include "sp_compute.h"

/* Frozen primes (mirror sp_compute_ntt_imp.c / sp_compute_ntt_hvx_imp.c). */
#define SP_NTT_Q1     1073738753u
#define SP_NTT_Q2     1073732609u
#define SP_NTT_Q_BITS 30u
#define SP_NTT_N_MAX  512

/* ─────────────────────────────────────────────────────────────────────
 * Scalar modular arithmetic (small-stage fallback + pre/post passes).
 * Byte-identical to NTT.1's sp_compute_ntt_hvx_imp.c:61-94.
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
 * HVX per-lane Barrett. Byte-identical to NTT.1's
 * sp_compute_ntt_hvx_imp.c:130-176 (sp_barrett_reduce32_hvx_lane_ntt1).
 * Renamed *_intt to avoid any TU-name conflict if linker ever inlines
 * symbol audits across TUs.
 * ───────────────────────────────────────────────────────────────────── */
static inline HVX_Vector sp_barrett_reduce32_hvx_lane_intt(
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

/* HVX modular add/sub — byte-identical to NTT.1 helpers. */
static inline HVX_Vector sp_modadd_hvx_lane_intt(
    HVX_Vector a, HVX_Vector b, HVX_Vector vq, HVX_Vector vq_m1)
{
    HVX_Vector     sum = Q6_Vw_vadd_VwVw(a, b);
    HVX_VectorPred gt  = Q6_Q_vcmp_gt_VuwVuw(sum, vq_m1);
    HVX_Vector     sub = Q6_Vw_vsub_VwVw(sum, vq);
    return Q6_V_vmux_QVV(gt, sub, sum);
}

static inline HVX_Vector sp_modsub_hvx_lane_intt(
    HVX_Vector a, HVX_Vector b, HVX_Vector vq)
{
    HVX_Vector     diff = Q6_Vw_vsub_VwVw(a, b);
    HVX_VectorPred lt   = Q6_Q_vcmp_gt_VuwVuw(b, a);
    HVX_Vector     add  = Q6_Vw_vadd_VwVw(diff, vq);
    return Q6_V_vmux_QVV(lt, add, diff);
}

/* ─────────────────────────────────────────────────────────────────────
 * HVX butterfly stage. half >= 32. Math byte-identical to NTT.1's
 * sp_ntt_butterfly_stage_hvx; reads `w_compact[half]` (stride-1).
 *
 * For NTT.4 the caller passes `w_inv_stages + stage_offset` from VTCM,
 * where `stage_offset = 2^{s-1} - 1` entries (per NTT.2 layout).
 * ───────────────────────────────────────────────────────────────────── */
static void sp_intt_butterfly_stage_hvx(uint32_t *out, uint32_t N,
                                        uint32_t len, uint32_t half,
                                        const uint32_t *w_compact,
                                        HVX_Vector vq, HVX_Vector vq_m1,
                                        HVX_Vector vmu)
{
    uint32_t vecs_per_group = half / 32u;
    /* `out` is 128-byte aligned per caller (`out_local` __attribute__).
     * `w_compact` is `w_inv_stages + stage_off` from VTCM; for stage s the
     * stage-base offset within w_inv_stages is (2^{s-1}-1)*4 bytes which is
     * NOT 128-byte aligned. Use HVX_UVector* (vmemu) for w_compact loads
     * to avoid silent misalignment faults. The `out` reads/writes remain
     * aligned via HVX_Vector* (vmem). */
    const HVX_UVector *w_uvp = (const HVX_UVector *)w_compact;

    for (uint32_t i = 0; i < N; i += len) {
        HVX_Vector *uvp = (HVX_Vector *)(out + i);
        HVX_Vector *vvp = (HVX_Vector *)(out + i + half);
        for (uint32_t v = 0; v < vecs_per_group; v++) {
            HVX_Vector u_vec = uvp[v];
            HVX_Vector v_raw = vvp[v];
            HVX_Vector w_vec = w_uvp[v];   /* unaligned VTCM load */
            HVX_Vector v_red = sp_barrett_reduce32_hvx_lane_intt(
                                   v_raw, w_vec, vq, vq_m1, vmu);
            HVX_Vector u_out = sp_modadd_hvx_lane_intt(u_vec, v_red, vq, vq_m1);
            HVX_Vector u_lo  = sp_modsub_hvx_lane_intt(u_vec, v_red, vq);
            uvp[v] = u_out;
            vvp[v] = u_lo;
        }
    }
}

/* Small-stage scalar butterfly. half < 32. Byte-identical to NTT.1
 * sp_ntt_butterfly_stage_scalar (sp_compute_ntt_hvx_imp.c:258-275). */
static void sp_intt_butterfly_stage_scalar(uint32_t *out, uint32_t N,
                                           uint32_t len, uint32_t half,
                                           uint32_t step,
                                           const uint32_t *w_inv,
                                           uint32_t q, uint64_t mu)
{
    for (uint32_t i = 0; i < N; i += len) {
        uint32_t widx = 0;
        for (uint32_t k = 0; k < half; k++) {
            uint32_t u = out[i + k];
            uint32_t v = modmul_s(out[i + k + half], w_inv[widx], q, mu);
            out[i + k]        = modadd_s(u, v, q);
            out[i + k + half] = modsub_s(u, v, q);
            widx += step;
        }
    }
}

/* ─────────────────────────────────────────────────────────────────────
 * Per-prime inverse NTT — uses VTCM-resident tables from NTT.2.
 *
 * Mirrors math-core inverse_one (ntt_crt.c:281-294) with HVX butterflies
 * over `w_inv_stages` (per-stage compacted, stride-1).
 *
 * Stage-offset within w_inv_stages: stage s ∈ [1, logN] has
 *   offset_entries(s) = 2^{s-1} - 1
 *   size_entries(s)   = 2^{s-1}
 * (per sp_compute_ntt_twiddle.c:226-235).
 *
 * `w_inv` is the legacy stride-step base, retained for the small-stage
 * scalar fallback. NTT.2 also exposes this directly so we use it.
 * ───────────────────────────────────────────────────────────────────── */
static void ntt_inverse_one_hvx_vtcm(uint32_t N, uint32_t logN,
                                     uint32_t q, uint64_t mu,
                                     uint32_t ninv,
                                     const uint32_t *ipsi_pow,
                                     const uint32_t *w_inv,
                                     const uint32_t *w_inv_stages,
                                     const uint32_t *in,
                                     uint32_t *out)
{
    /* Step 1 — input reduce. Matches math-core inverse_one:285. */
    for (uint32_t j = 0; j < N; j++) {
        out[j] = in[j] % q;
    }

    /* Step 2 — bit-reversal permutation. Same as forward. */
    for (uint32_t i = 0; i < N; i++) {
        uint32_t j = bitrev_u32(i, logN);
        if (i < j) { uint32_t t = out[i]; out[i] = out[j]; out[j] = t; }
    }

    /* Step 3 — logN butterfly stages over w_inv. */
    HVX_Vector vq    = Q6_V_vsplat_R((int32_t)q);
    HVX_Vector vq_m1 = Q6_V_vsplat_R((int32_t)(q - 1u));
    HVX_Vector vmu   = Q6_V_vsplat_R((int32_t)mu);

    uint32_t stage_entries_off = 0;
    for (uint32_t s = 1u; s <= logN; s++) {
        uint32_t len  = 1u << s;
        uint32_t half = 1u << (s - 1u);
        uint32_t step = N / len;

        if (half >= 32u) {
            const uint32_t *w_compact = w_inv_stages + stage_entries_off;
            sp_intt_butterfly_stage_hvx(out, N, len, half, w_compact,
                                        vq, vq_m1, vmu);
        } else {
            sp_intt_butterfly_stage_scalar(out, N, len, half, step,
                                           w_inv, q, mu);
        }
        stage_entries_off += half;
    }

    /* Step 4 — scale by ninv and post-weight by ipsi_pow[j].
     * Matches math-core inverse_one:289-293 (modmul × modmul = two
     * Barrett reductions). Scalar; HVX vectorization deferred to a
     * future sprint per feedback-bundled-changeset (load-bearing gate
     * is round-trip correctness, not the post-pass µs). */
    for (uint32_t j = 0; j < N; j++) {
        uint32_t s_val = modmul_s(out[j], ninv, q, mu);
        out[j] = modmul_s(s_val, ipsi_pow[j], q, mu);
    }
}

/* ─────────────────────────────────────────────────────────────────────
 * VTCM view accessor declared in sp_compute_ntt_twiddle.c.
 * Not exposed via IDL — pure C-internal.
 * ───────────────────────────────────────────────────────────────────── */
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
} sp_tw_view;

extern int sp_compute_ntt_twiddle_view(int q_idx, int N, sp_tw_view *out);

/* ─────────────────────────────────────────────────────────────────────
 * IDL handler — sp_compute_intt_hvx_oracle (method 18).
 *
 * primIn layout (qaic-emitted): 4 x i32 = 16 bytes — IDENTICAL to NTT.0
 * and NTT.1.
 *   [0] q_idx        (int32; 0 -> q_1, 1 -> q_2)
 *   [1] N            (int32; must be 128, 256, or 512)
 *   [2] data_inLen   (int32; expected N * 4)
 *   [3] data_outLen  (int32; expected N * 4)
 *
 * data_in : N x u32 LE in [0, q) — forward-NTT output for one prime
 * data_out: N x u32 LE in [0, q) — inverse-NTT output for one prime
 *
 * VTCM tables MUST be initialized via ntt_twiddle_init first; if not
 * present this method returns -1.
 *
 * Returns 0 on success, -1 on any input violation or table missing.
 * ───────────────────────────────────────────────────────────────────── */
int sp_compute_intt_hvx_oracle(remote_handle64 h,
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

    sp_tw_view view;
    int rc = sp_compute_ntt_twiddle_view(q_idx, N, &view);
    if (rc != 0) {
        FARF(RUNTIME_ERROR,
             "sp_compute_intt_hvx_oracle: twiddle table missing q_idx=%d N=%d "
             "(call ntt_twiddle_init first)", q_idx, N);
        return -1;
    }

    uint32_t Nu   = (uint32_t)N;
    uint32_t logN = ilog2_u32(Nu);
    uint32_t q    = view.q;
    uint64_t mu   = view.mu;
    uint32_t ninv = view.ninv;

    /* 128-byte aligned local scratches for HVX vmem. */
    uint32_t in_local[SP_NTT_N_MAX]  __attribute__((aligned(128)));
    uint32_t out_local[SP_NTT_N_MAX] __attribute__((aligned(128)));
    memcpy(in_local, data_in, (size_t)N * 4u);

    ntt_inverse_one_hvx_vtcm(Nu, logN, q, mu, ninv,
                             view.ipsi_pow, view.w_inv, view.w_inv_stages,
                             in_local, out_local);

    memcpy(data_out, out_local, (size_t)N * 4u);

    FARF(RUNTIME_HIGH,
         "sp_compute_intt_hvx_oracle: q_idx=%d N=%d ninv=%u out[0]=%u out[N-1]=%u",
         q_idx, N, ninv, out_local[0], out_local[Nu - 1u]);
    return 0;
}
