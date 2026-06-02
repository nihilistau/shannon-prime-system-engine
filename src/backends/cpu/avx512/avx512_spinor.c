#include "sp_engine/avx512.h"
#include <immintrin.h>
#include <stdint.h>

/* Hot path: aligned load + prefetch next slot into L1.
 * `slot` must be 64-byte aligned (arena allocator guarantees this).
 * Prefetches slot+128 (two blocks ahead) to hide DRAM latency in sequential scans. */
SP_TARGET("avx512f")
int sp_avx512_spinor_load_check(const void *slot) {
    _mm_prefetch((const char *)slot + 128, _MM_HINT_T0);  /* two blocks ahead */
    __m512i zmm = _mm512_load_si512((const __m512i *)slot);
    /* Extract byte 63: lane 3 of the ZMM (bytes 48..63), then byte 15 of that lane. */
    __m128i hi      = _mm512_extracti32x4_epi32(zmm, 3);
    int     sentinel = _mm_extract_epi8(hi, 15);
    return (sentinel == 0xA5) ? 0 : -1;
}

/* Cold/NT path: non-temporal load, bypasses L1+L2.
 * Use for full-arena sweeps where blocks won't be reused soon.
 * `slot` must be 64-byte aligned. */
SP_TARGET("avx512f")
int sp_avx512_spinor_nt_load_check(const void *slot) {
    __m512i zmm     = _mm512_stream_load_si512((void *)slot);
    __m128i hi      = _mm512_extracti32x4_epi32(zmm, 3);
    int     sentinel = _mm_extract_epi8(hi, 15);
    return (sentinel == 0xA5) ? 0 : -1;
}
