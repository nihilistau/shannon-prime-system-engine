#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "sp_engine/avx512.h"

/* ---- scalar references ---- */
static uint64_t scalar_popcnt512(const uint64_t *v) {
    uint64_t n = 0;
    for (int i = 0; i < 8; i++) n += (uint64_t)__builtin_popcountll(v[i]);
    return n;
}

/* lane[i] ^= lane[(i+1)%16] ^ lane[(i+5)%16] */
static void scalar_kste_round(uint32_t *st) {
    uint32_t tmp[16];
    for (int i = 0; i < 16; i++)
        tmp[i] = st[i] ^ st[(i+1)%16] ^ st[(i+5)%16];
    memcpy(st, tmp, 64);
}

int main(void) {
    sp_avx512_init();

    if (!g_avx512_caps.has_avx512f) {
        printf("SKIP: no AVX-512F on this CPU\n");
        return 0;
    }

    int fail = 0;

    /* T_TERNLOG_1: popcnt512 */
    {
        uint64_t v[8] = {0xDEADBEEFCAFEBABEULL, 0x1234567890ABCDEFULL,
                         0xFFFFFFFFFFFFFFFFULL, 0ULL,
                         0x0101010101010101ULL, 0xAAAAAAAAAAAAAAAAULL,
                         0x5555555555555555ULL, 1ULL};
        uint64_t ref = scalar_popcnt512(v);
        uint64_t got = sp_avx512_ternlog_popcnt512(v);
        if (ref != got) {
            printf("FAIL T_TERNLOG_1: popcnt512 ref=%llu got=%llu\n",
                   (unsigned long long)ref, (unsigned long long)got);
            fail = 1;
        } else {
            printf("PASS T_TERNLOG_1: popcnt512 = %llu\n", (unsigned long long)ref);
        }
    }

    /* T_TERNLOG_2: kste_round */
    {
        static uint32_t st_ref[16] __attribute__((aligned(64)));
        static uint32_t st_avx[16] __attribute__((aligned(64)));
        for (int i = 0; i < 16; i++) st_ref[i] = st_avx[i] = (uint32_t)(i * 0x9E3779B9u + 1);
        scalar_kste_round(st_ref);
        sp_avx512_ternlog_kste_round(st_avx);
        if (memcmp(st_ref, st_avx, 64) != 0) {
            printf("FAIL T_TERNLOG_2: kste_round mismatch\n");
            for (int i = 0; i < 16; i++)
                printf("  [%2d] ref=%08x avx=%08x %s\n", i, st_ref[i], st_avx[i],
                       st_ref[i]==st_avx[i] ? "" : "<--");
            fail = 1;
        } else {
            printf("PASS T_TERNLOG_2: kste_round 16-lane XOR3\n");
        }
    }

    /* T_SPINOR_1: load + sentinel check — valid 0xA5 sentinel */
    {
        static uint8_t slot[64] __attribute__((aligned(64)));
        int i;
        for (i = 0; i < 63; i++) slot[i] = (uint8_t)(i + 1);
        slot[63] = 0xA5;
        int r = sp_avx512_spinor_load_check(slot);
        if (r != 0) {
            printf("FAIL T_SPINOR_1: expected 0, got %d\n", r);
            fail = 1;
        } else {
            printf("PASS T_SPINOR_1: sentinel OK -> 0\n");
        }
    }

    /* T_SPINOR_2: load + sentinel check — corrupted sentinel */
    {
        static uint8_t slot[64] __attribute__((aligned(64)));
        int i;
        for (i = 0; i < 64; i++) slot[i] = 0xFF;
        int r = sp_avx512_spinor_load_check(slot);
        if (r != -1) {
            printf("FAIL T_SPINOR_2: expected -1, got %d\n", r);
            fail = 1;
        } else {
            printf("PASS T_SPINOR_2: sentinel mismatch -> -1\n");
        }
    }

    /* T_VNNI_1/T_VNNI_2: added in Task 4 */
    /* T_IFMA_1: added in Task 5 */

    return fail;
}
