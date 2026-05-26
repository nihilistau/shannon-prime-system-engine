#include "sp_engine/avx512.h"
#include <immintrin.h>
#include <stdint.h>

/* Hot path: aligned load + prefetch next slot into L1.
 * `slot` must be 64-byte aligned (arena allocator guarantees this).
 * Prefetches slot+128 (the next spinor block in a sequential scan). */
__attribute__((target("avx512f")))
int sp_avx512_spinor_load_check(const void *slot) {
    _mm_prefetch((const char *)slot + 128, _MM_HINT_T0);
    /* _mm512_mask_load_epi64 with all-ones mask: GCC emits vmovdqa64 (W=1).
     * Plain _mm512_load_si512 / _mm512_load_epi64 emit vmovdqa32 (W=0) on GCC 15.
     * Both encode as 0x6F; only the W-bit differs (irrelevant without element masking). */
    __m512i zmm = _mm512_mask_load_epi64(_mm512_setzero_si512(), (__mmask8)0xFF, slot);
    /* Extract byte 63: lane 3 of the ZMM (bytes 48..63), then byte 15 of that lane. */
    __m128i hi      = _mm512_extracti32x4_epi32(zmm, 3);
    int     sentinel = _mm_extract_epi8(hi, 15);
    return (sentinel == 0xA5) ? 0 : -1;
}

/* Cold/NT path: non-temporal load, bypasses L1+L2.
 * Use for full-arena sweeps where blocks won't be reused soon.
 * `slot` must be 64-byte aligned. */
__attribute__((target("avx512f")))
int sp_avx512_spinor_nt_load_check(const void *slot) {
    __m512i zmm     = _mm512_stream_load_si512((void *)slot);
    __m128i hi      = _mm512_extracti32x4_epi32(zmm, 3);
    int     sentinel = _mm_extract_epi8(hi, 15);
    return (sentinel == 0xA5) ? 0 : -1;
}
