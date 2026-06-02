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
    for (int i = 0; i < rows; i++) {
        __m512i acc = _mm512_setzero_si512();

        const int8_t  *wi = w_codes + (ptrdiff_t)i * cols;

        for (int k = 0; k < cols; k += 64) {
            __m512i a64 = _mm512_load_si512((const __m512i *)(act_u8 + k));
            __m512i w64 = _mm512_load_si512((const __m512i *)(wi + k));
            acc = _mm512_dpbusd_epi32(acc, a64, w64);
        }

        int32_t dot = (int32_t)_mm512_reduce_add_epi32(acc);
        out[i] = (float)(dot - bias[i]) * row_scale[i];
    }
}
