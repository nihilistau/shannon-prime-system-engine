/* test_gen_kv.c — GEN_KV: persistent-KV O(n) decode vs the O(n^2) reference.
 *
 * qwen3_generate_kv maintains a position-indexed K/V cache (each token's weight
 * matmuls run once) and must produce the SAME greedy token sequence as
 * qwen3_generate (which re-prefills the whole prefix every step). The two paths
 * reduce different-length softmax sums, so logits differ by the float-
 * reassociation floor (§8.6.1); the gate is therefore SEQUENCE (argmax) identity,
 * run under SP_CPU_SCALAR=1 to remove FMA reassociation as a confound.
 *
 * Prompt = the oracle ref.bin token IDs; generate a fixed number of tokens.
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

/* Tokens to generate. The O(n) decode is fast; the cost is the O(n^2) *reference*
 * under scalar mode, so the default is small (8 consecutive argmax matches under
 * scalar is strong sequence-identity evidence). SP_GEN_KV_N overrides for a
 * thorough run (e.g. 24). */
#define N_GEN_DEFAULT 8

static void set_env(const char *k, const char *v) {
#ifdef _WIN32
    _putenv_s(k, v);
#else
    setenv(k, v, 1);
#endif
}
static void set_scalar(int on) { set_env("SP_CPU_SCALAR", on ? "1" : "0"); }

/* Read the oracle ref.bin token IDs into a freshly malloc'd buffer; returns NULL
 * (and SP_CHECK-fails) on any error. *out_nt receives the token count. */
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

static void GEN_KV(void) {
    uint32_t nt = 0;
    int32_t *prompt = read_ref_prompt(&nt);
    if (!prompt) return;

    qwen3_model *m = qwen3_load(SP_QWEN3_GGUF);
    SP_CHECK(m != NULL, "qwen3_load");
    if (!m) { free(prompt); return; }

    set_scalar(1);   /* remove FMA reassociation as a confound */

    const char *ng = getenv("SP_GEN_KV_N");
    int n_gen = ng ? atoi(ng) : N_GEN_DEFAULT;
    if (n_gen < 1) n_gen = N_GEN_DEFAULT;

    int32_t *seq_ref = (int32_t *)malloc((size_t)((int)nt + n_gen) * sizeof(int32_t));
    int32_t *seq_kv  = (int32_t *)malloc((size_t)((int)nt + n_gen) * sizeof(int32_t));
    if (seq_ref && seq_kv) {
        memcpy(seq_ref, prompt, (size_t)nt * sizeof(int32_t));
        memcpy(seq_kv,  prompt, (size_t)nt * sizeof(int32_t));
        int n_ref = qwen3_generate   (m, seq_ref, (int)nt, n_gen, -1);   /* O(n^2) reference */
        int n_kv  = qwen3_generate_kv(m, seq_kv,  (int)nt, n_gen, -1);   /* O(n) persistent-KV */
        SP_CHECK(n_ref == (int)nt + n_gen && n_kv == n_ref, "both generated n_gen tokens");
        if (n_ref == n_kv) {
            int match = 1, first_bad = -1;
            for (int i = 0; i < n_ref; i++)
                if (seq_ref[i] != seq_kv[i]) { match = 0; if (first_bad < 0) first_bad = i; }
            if (!match)
                fprintf(stderr, "    first divergence at position %d: ref %d vs kv %d\n",
                        first_bad, seq_ref[first_bad], seq_kv[first_bad]);
            else
                fprintf(stderr, "    %d generated tokens identical (O(n) KV-cache == O(n^2) reference)\n", n_gen);
            SP_CHECK(match, "persistent-KV decode reproduces the reference token sequence");
        }
    } else {
        SP_CHECK(0, "alloc generation buffers");
    }
    set_scalar(0);

    free(seq_ref); free(seq_kv); free(prompt);
    qwen3_free(m);
}

/* Piece 2 (load-bearing KV layout, roadmap §8.2.2): qwen3_generate_kv with
 * SP_KV_SPINOR=1 stores the cache as frozen 63-byte Spinor blocks and decodes on
 * read — the production §4.9 layout. decode(encode(x)) is arithmetically the same
 * as the in-place round-trip, so the block cache must produce a token sequence
 * IDENTICAL to the f32 parity reference (SP_KV_SPINOR_REF=1, the "fp32 cache for
 * parity tests only"). This is the layout-parity gate, mirroring E_CPU_9's
 * arena==FROB byte gate. The block run also prints the measured KV memory ratio. */
static void GEN_KV_SPINOR(void) {
    uint32_t nt = 0;
    int32_t *prompt = read_ref_prompt(&nt);
    if (!prompt) return;

    qwen3_model *m = qwen3_load(SP_QWEN3_GGUF);
    SP_CHECK(m != NULL, "qwen3_load");
    if (!m) { free(prompt); return; }

    set_scalar(1);
    set_env("SP_KV_SPINOR", "1");

    const char *ng = getenv("SP_GEN_KV_N");
    int n_gen = ng ? atoi(ng) : N_GEN_DEFAULT;
    if (n_gen < 1) n_gen = N_GEN_DEFAULT;

    int32_t *seq_blk = (int32_t *)malloc((size_t)((int)nt + n_gen) * sizeof(int32_t));
    int32_t *seq_ref = (int32_t *)malloc((size_t)((int)nt + n_gen) * sizeof(int32_t));
    if (seq_blk && seq_ref) {
        memcpy(seq_blk, prompt, (size_t)nt * sizeof(int32_t));
        memcpy(seq_ref, prompt, (size_t)nt * sizeof(int32_t));
        set_env("SP_KV_SPINOR_REF", "0");                                /* production block cache */
        int n_blk = qwen3_generate_kv(m, seq_blk, (int)nt, n_gen, -1);
        set_env("SP_KV_SPINOR_REF", "1");                                /* f32 round-trip parity ref */
        int n_ref = qwen3_generate_kv(m, seq_ref, (int)nt, n_gen, -1);
        set_env("SP_KV_SPINOR_REF", "0");
        SP_CHECK(n_blk == (int)nt + n_gen && n_ref == n_blk, "both generated n_gen tokens");
        if (n_blk == n_ref) {
            int match = 1, first_bad = -1;
            for (int i = 0; i < n_blk; i++)
                if (seq_blk[i] != seq_ref[i]) { match = 0; if (first_bad < 0) first_bad = i; }
            if (!match)
                fprintf(stderr, "    first divergence at position %d: block %d vs f32-ref %d\n",
                        first_bad, seq_blk[first_bad], seq_ref[first_bad]);
            else
                fprintf(stderr, "    %d generated tokens identical (Spinor-block KV == f32 round-trip)\n", n_gen);
            SP_CHECK(match, "Spinor-block KV cache == f32 round-trip reference (Piece 2 layout parity)");
        }
    } else {
        SP_CHECK(0, "alloc generation buffers");
    }
    set_env("SP_KV_SPINOR", "0");
    set_scalar(0);

    free(seq_blk); free(seq_ref); free(prompt);
    qwen3_free(m);
}

int main(void) {
    SP_RUN(GEN_KV);
    SP_RUN(GEN_KV_SPINOR);
    return SP_DONE();
}
