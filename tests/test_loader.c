/* test_loader.c — E_CPU_1: GGUF loader parses + round-trips the Qwen3-0.6B
 * header, metadata, and tensor table, and validates every tensor lies within
 * the mapped file. Uses the math-core sp_test.h harness (via the submodule). */
#include "sp/sp_test.h"
#include "sp_engine/gguf.h"

#include <stdio.h>
#include <string.h>

#ifndef SP_QWEN3_GGUF
#define SP_QWEN3_GGUF "Qwen3-0.6B-f16.gguf"
#endif

static void E_CPU_1(void) {
    const char *path = SP_QWEN3_GGUF;
    gguf_ctx *g = gguf_open(path);
    SP_CHECK(g != NULL, "gguf_open(Qwen3-0.6B f16)");
    if (!g) { fprintf(stderr, "    (model not found: %s)\n", path); return; }

    SP_CHECK(gguf_version(g) >= 2 && gguf_version(g) <= 3, "GGUF version in {2,3}");
    SP_CHECK(gguf_n_tensors(g) > 0, "tensor count > 0");
    SP_CHECK(gguf_n_kv(g) > 0, "kv count > 0");
    SP_CHECK(gguf_alignment(g) >= 1, "alignment set");
    SP_CHECK(gguf_data_offset(g) % gguf_alignment(g) == 0, "data section aligned");

    const char *arch = gguf_get_str(g, "general.architecture");
    SP_CHECK(arch != NULL && arch[0] != '\0', "general.architecture present");

    /* derive the arch-prefixed config keys (robust to exact arch naming) */
    uint64_t nl = 0, nh = 0, el = 0;
    if (arch) {
        char key[160];
        snprintf(key, sizeof key, "%s.block_count", arch);
        SP_CHECK(gguf_get_u64(g, key, &nl) && nl > 0, "<arch>.block_count > 0");
        snprintf(key, sizeof key, "%s.attention.head_count", arch);
        SP_CHECK(gguf_get_u64(g, key, &nh) && nh > 0, "<arch>.attention.head_count > 0");
        snprintf(key, sizeof key, "%s.embedding_length", arch);
        SP_CHECK(gguf_get_u64(g, key, &el) && el > 0, "<arch>.embedding_length > 0");
    }

    /* every tensor: aligned offset, valid data pointer, within file bounds */
    int all_ok = 1;
    for (uint64_t i = 0; i < gguf_n_tensors(g); i++) {
        const gguf_tensor *t = gguf_tensor_at(g, i);
        if (!t || (t->offset % gguf_alignment(g)) != 0) { all_ok = 0; break; }
        if (!gguf_tensor_data(g, t)) { all_ok = 0; break; }
        if (gguf_data_offset(g) + t->offset + t->nbytes > gguf_file_size(g)) { all_ok = 0; break; }
    }
    SP_CHECK(all_ok, "every tensor aligned and within file bounds");

    /* canonical Qwen3 tensors present */
    SP_CHECK(gguf_find_tensor(g, "token_embd.weight") != NULL, "token_embd.weight present");
    SP_CHECK(gguf_find_tensor(g, "output_norm.weight") != NULL, "output_norm.weight present");

    /* round-trip / determinism: a fresh parse yields the identical header */
    gguf_ctx *g2 = gguf_open(path);
    SP_CHECK(g2 != NULL, "re-open");
    if (g2) {
        SP_CHECK(gguf_version(g2) == gguf_version(g) &&
                 gguf_n_tensors(g2) == gguf_n_tensors(g) &&
                 gguf_n_kv(g2) == gguf_n_kv(g) &&
                 gguf_data_offset(g2) == gguf_data_offset(g) &&
                 gguf_alignment(g2) == gguf_alignment(g),
                 "re-parse is byte-identical at the header level");
        gguf_close(g2);
    }

    fprintf(stderr, "    arch=%s  tensors=%llu  kv=%llu  layers=%llu  heads=%llu  dim=%llu\n",
            arch ? arch : "?", (unsigned long long)gguf_n_tensors(g),
            (unsigned long long)gguf_n_kv(g), (unsigned long long)nl,
            (unsigned long long)nh, (unsigned long long)el);
    gguf_close(g);
}

int main(void) {
    SP_RUN(E_CPU_1);
    return SP_DONE();
}
