#include "sp_engine/avx512.h"
#include <immintrin.h>
#include <stdint.h>

/* Pointwise Barrett modmul for 30-bit inputs via AVX-512IFMA.
 *
 * For a[i], b[i] < q < 2^30:
 *   x = a*b < 2^60; madd52lo gives bits [51:0], madd52hi gives x >> 52.
 *   Reconstruct: x = (x_hi << 52) | x_lo
 *   Barrett: qhat = ((x >> 29) * mu) >> 31   where mu = floor(2^60/q)
 *            r    = x - qhat*q; two conditional subtracts.
 *
 * qhat computation uses two-instruction form (madd52lo + madd52hi) to avoid
 * precision loss: qhat_num < 2^31, mu < 2^31, product < 2^62 — bits 52-61
 * are needed after >>31, so we must capture them via madd52hi.
 *
 * N must be a multiple of 8 (8 uint64 lanes per ZMM after 32->64 widening).
 * Caller must check g_avx512_caps.has_ifma before calling on Zen 4.
 */
SP_TARGET("avx512f,avx512ifma,avx512dq")
void sp_avx512_ifma_modmul(const uint32_t *a, const uint32_t *b,
                            uint32_t q, uint64_t mu, uint32_t N, uint32_t *out) {
    __m512i vq   = _mm512_set1_epi64((int64_t)(uint64_t)q);
    __m512i vmu  = _mm512_set1_epi64((int64_t)mu);
    __m512i zero = _mm512_setzero_si512();

    for (uint32_t i = 0; i < N; i += 8) {
        __m256i a32 = _mm256_loadu_si256((const __m256i *)(a + i));
        __m256i b32 = _mm256_loadu_si256((const __m256i *)(b + i));
        __m512i va  = _mm512_cvtepu32_epi64(a32);
        __m512i vb  = _mm512_cvtepu32_epi64(b32);

        /* x = a * b: low 52 bits and high bits (x < 2^60 for 30-bit inputs) */
        __m512i x_lo = _mm512_madd52lo_epu64(zero, va, vb);
        __m512i x_hi = _mm512_madd52hi_epu64(zero, va, vb);
        /* x = (x_hi << 52) | x_lo */
        __m512i x = _mm512_or_si512(x_lo, _mm512_slli_epi64(x_hi, 52));

        /* qhat = ((x >> 29) * mu) >> 31
         * Two-instruction form to avoid precision loss:
         *   qhat_num < 2^31, mu < 2^31, product < 2^62
         *   madd52lo gives bits [51:0], madd52hi gives bits [103:52]
         *   After >> 31 we need bits [82:31], spanning both halves.
         *   qhat = (qphi << 21) | (qplo >> 31)
         */
        __m512i qhat_num = _mm512_srli_epi64(x, 29);
        __m512i qplo     = _mm512_madd52lo_epu64(zero, qhat_num, vmu);
        __m512i qphi     = _mm512_madd52hi_epu64(zero, qhat_num, vmu);
        __m512i qhat     = _mm512_or_si512(
            _mm512_slli_epi64(qphi, 21),
            _mm512_srli_epi64(qplo, 31)
        );

        /* r = x - qhat * q; two conditional subtracts */
        __m512i qhat_q = _mm512_mullo_epi64(qhat, vq);
        __m512i r      = _mm512_sub_epi64(x, qhat_q);
        __mmask8 m1    = _mm512_cmpge_epu64_mask(r, vq);
        r = _mm512_mask_sub_epi64(r, m1, r, vq);
        __mmask8 m2    = _mm512_cmpge_epu64_mask(r, vq);
        r = _mm512_mask_sub_epi64(r, m2, r, vq);

        /* narrow to uint32 and store */
        _mm256_storeu_si256((__m256i *)(out + i), _mm512_cvtepi64_epi32(r));
    }
}
