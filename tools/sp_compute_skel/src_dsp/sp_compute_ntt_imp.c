/* sp_compute_ntt_imp.c — Sprint NTT.0 scalar Hexagon NTT oracle.
 *
 * Implements `sp_compute_ntt_oracle` (IDL method 12), a byte-exact port of
 * math-core's per-prime negacyclic Cooley-Tukey radix-2 DIT NTT
 * (`lib/shannon-prime-system/core/ntt_crt/ntt_crt.c::forward_one`) to the
 * cDSP scalar pipe.
 *
 * Algorithm (matches math-core `forward_one` for one prime):
 *   1. Pre-weight signed input: x[j] = (in[j] mod q) * psi^j  (mod q)
 *   2. Bit-reversal permutation in place
 *   3. logN stages of radix-2 DIT butterflies (u, v) -> (u + w*v, u - w*v)
 *      where w = omega^(j*stride) = w_fwd[j*stride]
 *
 * Twiddle policy (NTT.0):
 *   psi_pow[N] and w_fwd[N/2] computed per-call from psi = find_psi(N, q)
 *   into stack-resident scratch (N <= 512). NTT.2 lifts these to
 *   VTCM-resident precomputed tables.
 *
 * NO HVX (NTT.1 introduces the HVX butterfly). NO 128-bit arithmetic; the
 * Barrett primitive is the same one K.beta.2.5b silicon-confirmed in
 * `sp_compute_crt_imp.c` — same constants, same shifts.
 *
 * Correctness gate: T_NTT0_SCALAR_BIT_EXACT (this oracle vs math-core's
 * `ntt_forward` per-prime output, 0 divergences over 100 seeds *
 * N in {128, 256, 512} * 2 primes = 600 runs).
 */
#include <stdint.h>
#include <string.h>
#include "HAP_farf.h"
#include "sp_compute.h"

/* Frozen primes (mirror sp_compute_crt_imp.c definitions; do NOT include
 * that file — keep this oracle isolated for future SASS audit). */
#define SP_NTT_Q1   1073738753u
#define SP_NTT_Q2   1073732609u
#define SP_MU_Q1    1073744895u
#define SP_MU_Q2    1073751039u
#define SP_NTT_Q_BITS 30u

/* Maximum supported N per frozen-primes 2-adic valuation = 10
 * (q-1 = 2^10 * odd for both q_1 and q_2). */
#define SP_NTT_N_MAX 512

/* ---- modular arithmetic (Barrett, no 128-bit type) ---------------------
 * Byte-identical to math-core ntt_crt.c::barrett_reduce + modmul + modadd +
 * modsub + modpow. Keep the math local to this TU so audit + measurement
 * stay scoped to NTT.0 only.
 */
static inline uint64_t barrett_reduce(uint64_t x, uint64_t q, uint64_t mu) {
    uint64_t qhat = ((x >> (SP_NTT_Q_BITS - 1u)) * mu) >> (SP_NTT_Q_BITS + 1u);
    uint64_t r = x - qhat * q;
    if (r >= q) r -= q;
    if (r >= q) r -= q;
    return r;
}

static inline uint32_t modmul(uint32_t a, uint32_t b, uint32_t q, uint64_t mu) {
    uint64_t x = (uint64_t)a * (uint64_t)b;
    return (uint32_t)barrett_reduce(x, (uint64_t)q, mu);
}

static inline uint32_t modadd(uint32_t a, uint32_t b, uint32_t q) {
    uint32_t s = a + b;
    if (s >= q) s -= q;
    return s;
}

static inline uint32_t modsub(uint32_t a, uint32_t b, uint32_t q) {
    return (a >= b) ? (a - b) : (a + q - b);
}

static uint32_t modpow(uint32_t base, uint64_t e, uint32_t q, uint64_t mu) {
    uint32_t result = 1u % q;
    uint32_t b = base % q;
    while (e) {
        if (e & 1u) result = modmul(result, b, q, mu);
        b = modmul(b, b, q, mu);
        e >>= 1;
    }
    return result;
}

/* Find primitive 2N-th root psi: smallest a >= 2 with psi^N == -1 (mod q),
 * where psi = a^((q-1)/(2N)). Byte-identical to math-core find_psi. */
static uint32_t find_psi(uint32_t N, uint32_t q, uint64_t mu) {
    uint64_t exp = ((uint64_t)q - 1u) / (2u * (uint64_t)N);
    for (uint32_t a = 2u; a < q; a++) {
        uint32_t psi = modpow(a, exp, q, mu);
        if (psi == 0u) continue;
        uint32_t pN = modpow(psi, (uint64_t)N, q, mu);
        if (pN == q - 1u) return psi;
    }
    return 0u;  /* unreachable for frozen primes & N in {128,256,512} */
}

/* ---- ilog2 + bit-reversal ----------------------------------------------- */

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

/* ---- per-prime scalar NTT (one residue channel) ------------------------- */

/* `psi_pow` is psi^j for j in [0, N), `w_fwd` is omega^j for j in [0, N/2),
 * omega = psi^2.  Pre-weighting + bit-reversal + logN radix-2 DIT stages,
 * identical to math-core forward_one + ntt_core.
 */
static void ntt_forward_one_scalar(uint32_t N, uint32_t logN,
                                   uint32_t q, uint64_t mu,
                                   const uint32_t *psi_pow,
                                   const uint32_t *w_fwd,
                                   const int32_t *in,
                                   uint32_t *out)
{
    /* Step 1 — pre-weight signed input. Matches forward_one:266-270. */
    for (uint32_t j = 0; j < N; j++) {
        int64_t v = (int64_t)in[j] % (int64_t)q;
        if (v < 0) v += (int64_t)q;
        out[j] = modmul((uint32_t)v, psi_pow[j], q, mu);
    }

    /* Step 2 — bit-reversal permutation. Matches ntt_core:236-239.
     * Computed inline to avoid carrying a bitrev[] table; cheap at N <= 512. */
    for (uint32_t i = 0; i < N; i++) {
        uint32_t j = bitrev_u32(i, logN);
        if (i < j) { uint32_t t = out[i]; out[i] = out[j]; out[j] = t; }
    }

    /* Step 3 — logN stages of radix-2 DIT. Matches ntt_core:241-254. */
    for (uint32_t len = 2; len <= N; len <<= 1) {
        uint32_t half = len >> 1;
        uint32_t step = N / len;
        for (uint32_t i = 0; i < N; i += len) {
            uint32_t widx = 0;
            for (uint32_t k = 0; k < half; k++) {
                uint32_t u = out[i + k];
                uint32_t v = modmul(out[i + k + half], w_fwd[widx], q, mu);
                out[i + k]        = modadd(u, v, q);
                out[i + k + half] = modsub(u, v, q);
                widx += step;
            }
        }
    }
}

/* ---- IDL handler — sp_compute_ntt_oracle (method 12) -------------------- */

/* primIn layout (qaic-emitted): 4 x i32 = 16 bytes
 *   [0] q_idx        (int32; 0 -> q_1, 1 -> q_2)
 *   [1] N            (int32; must be 128, 256, or 512)
 *   [2] data_inLen   (int32; expected N * 4)
 *   [3] data_outLen  (int32; expected N * 4)
 *
 * data_in : N signed i32 LE.  Modular reduction is done inside (matches
 *           forward_one's "arbitrary int32" contract).
 * data_out: N u32 LE in [0, q).
 *
 * Returns 0 on success, -1 on any input violation. */
int sp_compute_ntt_oracle(remote_handle64 h,
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

    /* Compute psi + twiddles per-call into stack-resident scratch.
     * sizeof(psi_pow) = 4 * 512 = 2 KB; sizeof(w_fwd) = 4 * 256 = 1 KB.
     * Total ~3 KB; comfortably within the cDSP stack on Path B.
     * NTT.2 will lift these to VTCM-resident precomputed tables. */
    uint32_t psi_pow[SP_NTT_N_MAX];
    uint32_t w_fwd[SP_NTT_N_MAX / 2];

    uint32_t psi = find_psi(Nu, q, mu);
    if (psi == 0u) {
        FARF(RUNTIME_ERROR,
             "sp_compute_ntt_oracle: find_psi failed q_idx=%d N=%d", q_idx, N);
        return -1;
    }
    uint32_t omega = modmul(psi, psi, q, mu);

    uint32_t acc = 1u % q;
    for (uint32_t j = 0; j < Nu; j++) {
        psi_pow[j] = acc;
        acc = modmul(acc, psi, q, mu);
    }
    acc = 1u % q;
    uint32_t half = Nu / 2u;
    for (uint32_t j = 0; j < half; j++) {
        w_fwd[j] = acc;
        acc = modmul(acc, omega, q, mu);
    }

    /* Copy signed input safely (caller's bytes may not be u32-aligned in
     * the worst case; memcpy through a local buffer avoids UB).  Output
     * goes to a local buffer too; memcpy back at the end. */
    int32_t  in_local[SP_NTT_N_MAX];
    uint32_t out_local[SP_NTT_N_MAX];
    memcpy(in_local, data_in, (size_t)N * 4u);

    ntt_forward_one_scalar(Nu, logN, q, mu, psi_pow, w_fwd, in_local, out_local);

    memcpy(data_out, out_local, (size_t)N * 4u);

    FARF(RUNTIME_HIGH,
         "sp_compute_ntt_oracle: q_idx=%d N=%d psi=%u out[0]=%u out[N-1]=%u",
         q_idx, N, psi, out_local[0], out_local[Nu - 1u]);
    return 0;
}
