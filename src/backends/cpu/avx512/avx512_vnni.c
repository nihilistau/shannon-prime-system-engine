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
        /* Deferred reduction (llama-style): keep 8 float lanes, fmadd each block's
         * scaled partials, hsum ONCE per row — no per-block cross-lane hadd. The
         * per-block scalar bias (128*sum_w) is summed separately, also deferred. */
        __m256 accv = _mm256_setzero_ps();
        float biasf = 0.0f;
        for (int b = 0; b < nblk; b++) {
            const int k = b * 32;
            __m256i a = _mm256_loadu_si256((const __m256i *)(act_u8 + k));
            __m256i w = _mm256_loadu_si256((const __m256i *)(wi + k));
            __m256i p = _mm256_dpbusd_epi32(_mm256_setzero_si256(), a, w);  /* 8 int32 partials */
            __m256  sc = _mm256_set1_ps(blk_scale[b]);
            accv = _mm256_fmadd_ps(sc, _mm256_cvtepi32_ps(p), accv);
            biasf += blk_scale[b] * (float)wbs[b];
        }
        __m128 lo = _mm256_castps256_ps128(accv);
        __m128 hi = _mm256_extractf128_ps(accv, 1);
        lo = _mm_add_ps(lo, hi);
        lo = _mm_hadd_ps(lo, lo);
        lo = _mm_hadd_ps(lo, lo);
        out[i] = (_mm_cvtss_f32(lo) - biasf) * row_scale[i];
    }
}
