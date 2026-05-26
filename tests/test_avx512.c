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

    /* T_VNNI_1: Q8 matvec 64x64 — byte-exact vs scalar */
    if (!g_avx512_caps.has_vnni) {
        printf("SKIP T_VNNI_1: no AVX-512VNNI\n");
    } else {
        enum { ROWS1 = 64, COLS1 = 64 };
        static int8_t  w1[ROWS1*COLS1] __attribute__((aligned(64)));
        static uint8_t act1[COLS1]     __attribute__((aligned(64)));
        static float   sc1[ROWS1]      __attribute__((aligned(64)));
        static int32_t bias1[ROWS1]    __attribute__((aligned(64)));
        static float   out_avx1[ROWS1], out_ref1[ROWS1];
        int i, k;

        for (i = 0; i < ROWS1*COLS1; i++) w1[i]   = (int8_t)((i * 7 + 3) % 127);
        for (k = 0; k < COLS1; k++)       act1[k]  = (uint8_t)((k * 13 + 5) % 255);
        for (i = 0; i < ROWS1; i++)       sc1[i]   = 0.01f * (float)(i + 1);
        for (i = 0; i < ROWS1; i++) {
            int32_t s = 0;
            for (k = 0; k < COLS1; k++) s += w1[i*COLS1 + k];
            bias1[i] = 128 * s;
        }
        /* scalar ref: act_i8 = act_u8 - 128 */
        for (i = 0; i < ROWS1; i++) {
            int32_t acc = 0;
            for (k = 0; k < COLS1; k++)
                acc += (int32_t)w1[i*COLS1+k] * (int32_t)((int8_t)((int)act1[k] - 128));
            out_ref1[i] = (float)acc * sc1[i];
        }
        sp_avx512_vnni_matvec(w1, act1, sc1, bias1, ROWS1, COLS1, out_avx1);

        int vnni_fail = 0;
        for (i = 0; i < ROWS1; i++) {
            float diff = out_avx1[i] - out_ref1[i];
            if (diff < -1e-3f || diff > 1e-3f) {
                printf("FAIL T_VNNI_1 row %d: ref=%.6f avx=%.6f\n", i, out_ref1[i], out_avx1[i]);
                vnni_fail = 1; fail = 1;
            }
        }
        if (!vnni_fail) printf("PASS T_VNNI_1: Q8 matvec %dx%d\n", ROWS1, COLS1);
    }

    /* T_VNNI_2: Q8 matvec 64x128 — two ZMM chunks per row */
    if (!g_avx512_caps.has_vnni) {
        printf("SKIP T_VNNI_2: no AVX-512VNNI\n");
    } else {
        enum { ROWS2 = 64, COLS2 = 128 };
        static int8_t  w2[ROWS2*COLS2] __attribute__((aligned(64)));
        static uint8_t act2[COLS2]     __attribute__((aligned(64)));
        static float   sc2[ROWS2]      __attribute__((aligned(64)));
        static int32_t bias2[ROWS2]    __attribute__((aligned(64)));
        static float   out_avx2[ROWS2], out_ref2[ROWS2];
        int i, k;

        for (i = 0; i < ROWS2*COLS2; i++) w2[i]   = (int8_t)((i * 11 + 7) % 127);
        for (k = 0; k < COLS2; k++)       act2[k]  = (uint8_t)((k * 17 + 3) % 255);
        for (i = 0; i < ROWS2; i++)       sc2[i]   = 0.005f * (float)(i + 1);
        for (i = 0; i < ROWS2; i++) {
            int32_t s = 0;
            for (k = 0; k < COLS2; k++) s += w2[i*COLS2 + k];
            bias2[i] = 128 * s;
        }
        for (i = 0; i < ROWS2; i++) {
            int32_t acc = 0;
            for (k = 0; k < COLS2; k++)
                acc += (int32_t)w2[i*COLS2+k] * (int32_t)((int8_t)((int)act2[k] - 128));
            out_ref2[i] = (float)acc * sc2[i];
        }
        sp_avx512_vnni_matvec(w2, act2, sc2, bias2, ROWS2, COLS2, out_avx2);

        int vnni_fail2 = 0;
        for (i = 0; i < ROWS2; i++) {
            float diff = out_avx2[i] - out_ref2[i];
            if (diff < -1e-3f || diff > 1e-3f) {
                printf("FAIL T_VNNI_2 row %d: ref=%.6f avx=%.6f\n", i, out_ref2[i], out_avx2[i]);
                vnni_fail2 = 1; fail = 1;
            }
        }
        if (!vnni_fail2) printf("PASS T_VNNI_2: Q8 matvec %dx%d\n", ROWS2, COLS2);
    }

    /* T_IFMA_1: added in Task 5 */

    return fail;
}
