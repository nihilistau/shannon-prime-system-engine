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

#define N_GEN 24

static void set_scalar(int on) {
#ifdef _WIN32
    _putenv_s("SP_CPU_SCALAR", on ? "1" : "0");
#else
    setenv("SP_CPU_SCALAR", on ? "1" : "0", 1);
#endif
}

static void GEN_KV(void) {
    FILE *f = fopen(SP_QWEN3_REF, "rb");
    SP_CHECK(f != NULL, "open oracle ref.bin (token IDs)");
    if (!f) { fprintf(stderr, "    (no ref: %s)\n", SP_QWEN3_REF); return; }
    uint32_t magic = 0, nt = 0, nv = 0;
    if (fread(&magic, 4, 1, f) != 1 || fread(&nt, 4, 1, f) != 1 || fread(&nv, 4, 1, f) != 1) {
        SP_CHECK(0, "read ref header"); fclose(f); return;
    }
    int32_t *prompt = (int32_t *)malloc((size_t)nt * sizeof(int32_t));
    int ok = prompt && fread(prompt, sizeof(int32_t), nt, f) == nt;
    fclose(f);
    SP_CHECK(ok, "read ref token IDs");
    if (!ok) { free(prompt); return; }

    qwen3_model *m = qwen3_load(SP_QWEN3_GGUF);
    SP_CHECK(m != NULL, "qwen3_load");
    if (!m) { free(prompt); return; }

    set_scalar(1);   /* remove FMA reassociation as a confound */

    int32_t *seq_ref = (int32_t *)malloc((size_t)(nt + N_GEN) * sizeof(int32_t));
    int32_t *seq_kv  = (int32_t *)malloc((size_t)(nt + N_GEN) * sizeof(int32_t));
    if (seq_ref && seq_kv) {
        memcpy(seq_ref, prompt, (size_t)nt * sizeof(int32_t));
        memcpy(seq_kv,  prompt, (size_t)nt * sizeof(int32_t));
        int n_ref = qwen3_generate   (m, seq_ref, (int)nt, N_GEN, -1);   /* O(n^2) reference */
        int n_kv  = qwen3_generate_kv(m, seq_kv,  (int)nt, N_GEN, -1);   /* O(n) persistent-KV */
        SP_CHECK(n_ref == (int)nt + N_GEN && n_kv == n_ref, "both generated N_GEN tokens");
        if (n_ref == n_kv) {
            int match = 1, first_bad = -1;
            for (int i = 0; i < n_ref; i++)
                if (seq_ref[i] != seq_kv[i]) { match = 0; if (first_bad < 0) first_bad = i; }
            if (!match)
                fprintf(stderr, "    first divergence at position %d: ref %d vs kv %d\n",
                        first_bad, seq_ref[first_bad], seq_kv[first_bad]);
            else
                fprintf(stderr, "    %d generated tokens identical (O(n) KV-cache == O(n^2) reference)\n", N_GEN);
            SP_CHECK(match, "persistent-KV decode reproduces the reference token sequence");
        }
    } else {
        SP_CHECK(0, "alloc generation buffers");
    }
    set_scalar(0);

    free(seq_ref); free(seq_kv); free(prompt);
    qwen3_free(m);
}

int main(void) {
    SP_RUN(GEN_KV);
    return SP_DONE();
}
