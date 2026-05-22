/* test_compose.c — COMPOSE: the foundational compression gates ON together
 * (roadmap §8.2.2, Piece 3). The prior session validated each gate in isolation
 * (E_CPU_9 arena==FROB, GEN_KV_SPINOR block==f32-KV, GEN_KV O(n)==O(n^2)) but
 * never the three at once. This is the all-gates-on smoke:
 *
 *   SP_ARENA=q8       — packed-weight arena, matmul lifts inline from Q8 codes
 *   SP_KV_SPINOR=1    — persistent 63-byte Spinor-block KV cache
 *   qwen3_generate_kv — the O(n) persistent-KV decode loop
 *
 * Invariant: with the arena ACTIVE, the Spinor-block KV cache still produces a
 * token sequence IDENTICAL to the f32 round-trip KV reference (SP_KV_SPINOR_REF=1).
 * The weight source (arena Q8 vs the f32 reference path) and the KV layout (blocks
 * vs f32) are orthogonal axes; this proves they compose without interference.
 * Run under SP_CPU_SCALAR=1 so the only varying axis is the KV storage.
 */
#define _CRT_SECURE_NO_WARNINGS
#include "sp/sp_test.h"
#include "sp_engine/model.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef SP_QWEN3_GGUF
#define SP_QWEN3_GGUF "Qwen3-0.6B-f16.gguf"
#endif
#ifndef SP_QWEN3_REF
#define SP_QWEN3_REF "qwen3_ref.bin"
#endif

#define N_GEN_DEFAULT 8

static void set_env(const char *k, const char *v) {
#ifdef _WIN32
    _putenv_s(k, v);
#else
    setenv(k, v, 1);
#endif
}

static int32_t *read_ref_prompt(uint32_t *out_nt) {
    FILE *f = fopen(SP_QWEN3_REF, "rb");
    SP_CHECK(f != NULL, "open oracle ref.bin (token IDs)");
    if (!f) { fprintf(stderr, "    (no ref: %s)\n", SP_QWEN3_REF); return NULL; }
    uint32_t magic = 0, nt = 0, nv = 0;
    if (fread(&magic, 4, 1, f) != 1 || fread(&nt, 4, 1, f) != 1 || fread(&nv, 4, 1, f) != 1) {
        SP_CHECK(0, "read ref header"); fclose(f); return NULL;
    }
    int32_t *prompt = (int32_t *)malloc((size_t)nt * sizeof(int32_t));
    int ok = prompt && fread(prompt, sizeof(int32_t), nt, f) == nt;
    fclose(f);
    SP_CHECK(ok, "read ref token IDs");
    if (!ok) { free(prompt); return NULL; }
    *out_nt = nt;
    return prompt;
}

static void COMPOSE(void) {
    uint32_t nt = 0;
    int32_t *prompt = read_ref_prompt(&nt);
    if (!prompt) return;

    set_env("SP_ARENA", "q8");                       /* arena built at load => matmul lifts inline */
    qwen3_model *m = qwen3_load(SP_QWEN3_GGUF);
    SP_CHECK(m != NULL, "qwen3_load with SP_ARENA=q8");
    set_env("SP_ARENA", "");                          /* don't leak to a later load */
    if (!m) { free(prompt); return; }

    set_env("SP_CPU_SCALAR", "1");
    set_env("SP_KV_SPINOR", "1");

    const char *ng = getenv("SP_GEN_KV_N");
    int n_gen = ng ? atoi(ng) : N_GEN_DEFAULT;
    if (n_gen < 1) n_gen = N_GEN_DEFAULT;

    int32_t *seq_blk = (int32_t *)malloc((size_t)((int)nt + n_gen) * sizeof(int32_t));
    int32_t *seq_ref = (int32_t *)malloc((size_t)((int)nt + n_gen) * sizeof(int32_t));
    if (seq_blk && seq_ref) {
        memcpy(seq_blk, prompt, (size_t)nt * sizeof(int32_t));
        memcpy(seq_ref, prompt, (size_t)nt * sizeof(int32_t));
        set_env("SP_KV_SPINOR_REF", "0");                            /* arena + Spinor-block KV */
        int n_blk = qwen3_generate_kv(m, seq_blk, (int)nt, n_gen, -1);
        set_env("SP_KV_SPINOR_REF", "1");                            /* arena + f32 round-trip KV */
        int n_ref = qwen3_generate_kv(m, seq_ref, (int)nt, n_gen, -1);
        set_env("SP_KV_SPINOR_REF", "0");
        SP_CHECK(n_blk == (int)nt + n_gen && n_ref == n_blk, "all-gates-on run generated n_gen tokens");
        if (n_blk == n_ref) {
            int valid = 1, match = 1, first_bad = -1;
            uint32_t V = m->cfg.n_vocab;
            for (int i = 0; i < n_blk; i++) {
                if (seq_blk[i] < 0 || (uint32_t)seq_blk[i] >= V) valid = 0;
                if (seq_blk[i] != seq_ref[i]) { match = 0; if (first_bad < 0) first_bad = i; }
            }
            SP_CHECK(valid, "all generated token IDs are in vocab range");
            if (!match)
                fprintf(stderr, "    first divergence at position %d: block %d vs f32-ref %d\n",
                        first_bad, seq_blk[first_bad], seq_ref[first_bad]);
            else
                fprintf(stderr, "    %d tokens identical (arena Q8 + Spinor-block KV composes with arena Q8 + f32 KV)\n", n_gen);
            SP_CHECK(match, "with arena active, Spinor-block KV == f32 KV (gates are orthogonal)");
        }
    } else {
        SP_CHECK(0, "alloc generation buffers");
    }
    set_env("SP_KV_SPINOR", "0");
    set_env("SP_CPU_SCALAR", "0");

    free(seq_blk); free(seq_ref); free(prompt);
    qwen3_free(m);
}

int main(void) {
    SP_RUN(COMPOSE);
    return SP_DONE();
}
