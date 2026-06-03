#include "sp_engine/avx512.h"
#include <immintrin.h>
#include <stdint.h>

/* Q8 matrix-vector multiply via AVX-512VNNI.
 *
 * DPBUSD takes u8 activations and i8 weights. The Frobenius arena stores weights
 * as signed i8; activations are also signed i8. Fix: caller passes
 *   act_u8[k] = act_i8[k] + 128  (zero-shifted to u8)
 *   bias[i]   = 128 * sum_k(w_codes[i*cols + k])  (precomputed per-row)
 * Then: (DPBUSD result - bias[i]) * scale[i] recovers the true dot product.
 *
 * rows and cols must be multiples of 64.
 * w_codes and act_u8 must be 64-byte aligned.
 */
SP_TARGET("avx512f,avx512vnni,avx512bw")
void sp_avx512_vnni_matvec(const int8_t *w_codes, const uint8_t *act_u8,
                            const float *row_scale, const int32_t *bias,
                            int rows, int cols, float *out) {
    int i;   /* hoisted: MSVC OpenMP 2.0 requires the loop var outside the for-init */
    #pragma omp parallel for
    for (i = 0; i < rows; i++) {
        __m512i acc = _mm512_setzero_si512();

        const int8_t  *wi = w_codes + (ptrdiff_t)i * cols;

        for (int k = 0; k < cols; k += 64) {
            __m512i a64 = _mm512_loadu_si512((const __m512i *)(act_u8 + k));
            __m512i w64 = _mm512_loadu_si512((const __m512i *)(wi + k));
            acc = _mm512_dpbusd_epi32(acc, a64, w64);
        }

        int32_t dot = (int32_t)_mm512_reduce_add_epi32(acc);
        out[i] = (float)(dot - bias[i]) * row_scale[i];
    }
}

/* Q8 matrix-vector with PER-32-BLOCK activation scales (Q8_0-faithful, the
 * accuracy fix that naive per-vector VNNI lacked). For each 32-element block b:
 *   act_u8[k] = round(x[k]/blk_scale[b]) + 128   (per-block scale, set by caller)
 *   int8x int8 via DPBUSD on the 32-byte block; subtract per-(row,block) bias
 *   wblk_bias[i*nblk+b] = 128*sum_{k in block} w_codes[i][k]; weight by blk_scale[b].
 *   out[i] = row_scale[i] * sum_b blk_scale[b] * (dpbusd_block - wblk_bias).
 * cols must be a multiple of 32. Needs AVX512VL (256-bit dpbusd). */
SP_TARGET("avx512f,avx512vl,avx512vnni,avx512bw")
void sp_avx512_q8blk_matvec(const int8_t *w_codes, const uint8_t *act_u8,
                            const float *blk_scale, const float *row_scale,
                            const int32_t *wblk_bias, int rows, int cols, float *out) {
    const int nblk = cols / 32;
    int i;
    #pragma omp parallel for
    for (i = 0; i < rows; i++) {
        const int8_t  *wi  = w_codes   + (ptrdiff_t)i * cols;
        const int32_t *wbs = wblk_bias + (ptrdiff_t)i * nblk;
        float accf = 0.0f;
        for (int b = 0; b < nblk; b++) {
            const int k = b * 32;
            __m256i a = _mm256_loadu_si256((const __m256i *)(act_u8 + k));
            __m256i w = _mm256_loadu_si256((const __m256i *)(wi + k));
            __m256i acc = _mm256_dpbusd_epi32(_mm256_setzero_si256(), a, w);
            __m128i lo = _mm256_castsi256_si128(acc);
            __m128i hi = _mm256_extracti128_si256(acc, 1);
            __m128i s  = _mm_add_epi32(lo, hi);
            s = _mm_hadd_epi32(s, s);
            s = _mm_hadd_epi32(s, s);
            int32_t dot = _mm_cvtsi128_si32(s);
            accf += (float)(dot - wbs[b]) * blk_scale[b];
        }
        out[i] = accf * row_scale[i];
    }
}
