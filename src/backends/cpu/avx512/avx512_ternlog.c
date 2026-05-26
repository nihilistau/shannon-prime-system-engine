#include "sp_engine/avx512.h"
#include <immintrin.h>
#include <stdint.h>

/* popcnt across 8 x uint64 = 512 bits.
 * Requires AVX-512F + AVX-512VPOPCNTDQ. */
__attribute__((target("avx512f,avx512vpopcntdq")))
uint64_t sp_avx512_ternlog_popcnt512(const uint64_t *v8) {
    __m512i v   = _mm512_loadu_si512((const __m512i *)v8);
    __m512i cnt = _mm512_popcnt_epi64(v);
    return (uint64_t)_mm512_reduce_add_epi64(cnt);
}

/* KSTE mixing round on 16 x u32 (= 512 bits in one ZMM).
 * lane[i] ^= lane[(i+1)%16] ^ lane[(i+5)%16]   =>  imm8=0x96 (XOR3).
 * The permute indices for +1 and +5 rotations are frozen constants.
 * state must be 64-byte aligned. */
__attribute__((target("avx512f,avx512bw")))
void sp_avx512_ternlog_kste_round(uint32_t *state) {
    /* Rotation by +1 (mod 16): indices 1,2,...,15,0 */
    static const int32_t rot1_idx[16] __attribute__((aligned(64))) =
        {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0};
    /* Rotation by +5 (mod 16): indices 5,6,...,15,0,1,2,3,4 */
    static const int32_t rot5_idx[16] __attribute__((aligned(64))) =
        {5,6,7,8,9,10,11,12,13,14,15,0,1,2,3,4};

    __m512i a    = _mm512_load_si512((const __m512i *)state);
    __m512i idx1 = _mm512_load_si512((const __m512i *)rot1_idx);
    __m512i idx5 = _mm512_load_si512((const __m512i *)rot5_idx);
    __m512i b    = _mm512_permutexvar_epi32(idx1, a);
    __m512i c    = _mm512_permutexvar_epi32(idx5, a);
    /* ternarylogic imm=0x96: a^b^c */
    __m512i res  = _mm512_ternarylogic_epi32(a, b, c, 0x96);
    _mm512_store_si512((__m512i *)state, res);
}
