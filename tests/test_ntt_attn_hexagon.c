/* test_ntt_attn_hexagon.c — E_HX_5: NTT-attention substitution lock for the Hexagon backend.
 *
 * Part A (the gate, host): the attention score <q,k> represented as an exact int64 integer
 * dot equals the poly-ring sp_pr_inner (coefficient 0 of the negacyclic product == sum_i q_i k_i).
 * This proves int64_dot(q,k) == sp_pr_inner(q,k) bit-for-bit at head_dim in {128,256,512} — the
 * same backend-agnostic substitution lock the closed CU/VK E_*_5 tests assert (the reason the
 * int64 representation may stand in for a CRT-NTT: int64 holds the exact result).
 *
 * Part B (deferred analog — stated honestly): the closed CU/VK E_*_5 tests additionally gate
 * their backend's int64-dot NTT-attention *forward* KL vs the f32 path. Hexagon's cDSP forward
 * uses scalar softmax attention; a poly-ring/int64-dot NTT-attention path on the V69 is the
 * deferred CRT-NTT-on-Hexagon work (the §8.5 Hexagon gate-adaptation pre-scoped E_HX_5 to the
 * substitution lock). E_HX_5 therefore closes on Part A alone — a documented adaptation that is
 * one notch thinner than CU/VK, who ran the int64-dot attention forward. Host test; no device.
 */
#define _CRT_SECURE_NO_WARNINGS
#include "sp/sp_test.h"
#include "sp/poly_ring.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

static uint32_t rng_state = 0x2545F491u;
static int32_t rand_coeff(int range) {
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 17; rng_state ^= rng_state << 5;
    return (int32_t)(rng_state % (uint32_t)(2 * range + 1)) - range;
}

static void E_HX_5(void) {
    /* Part A: int64 dot == sp_pr_inner, the Hexagon substitution lock. */
    const uint32_t Ns[3] = { 128, 256, 512 };
    long ok = 0, total = 0;
    for (int ni = 0; ni < 3; ni++) {
        uint32_t N = Ns[ni];
        sp_pr_ctx *pr = sp_pr_init(N);
        SP_CHECK(pr != NULL, "sp_pr_init(N)");
        if (!pr) continue;
        int32_t *q = (int32_t *)malloc(N * sizeof(int32_t));
        int32_t *k = (int32_t *)malloc(N * sizeof(int32_t));
        for (int trial = 0; trial < 64; trial++) {
            for (uint32_t i = 0; i < N; i++) { q[i] = rand_coeff(1 << 21); k[i] = rand_coeff(1 << 21); }
            long long d_int = 0;
            for (uint32_t i = 0; i < N; i++) d_int += (long long)q[i] * (long long)k[i];
            int64_t d_pr = sp_pr_inner(pr, q, k);
            total++;
            if (d_int == (long long)d_pr) ok++;
        }
        free(q); free(k); sp_pr_free(pr);
    }
    fprintf(stderr, "    E_HX_5 Part A: int64_dot == sp_pr_inner on %ld/%ld vectors (N in {128,256,512})\n", ok, total);
    SP_CHECK_EQ_I64(ok, total, "int64 dot equals poly-ring sp_pr_inner exactly (Hexagon substitution lock)");
}

int main(void) { SP_RUN(E_HX_5); return SP_DONE(); }
