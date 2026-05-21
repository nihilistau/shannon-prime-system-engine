/* test_tokenizer.c — TOK_DECODE: byte-level decode of the GGUF vocab.
 *
 * The oracle tokenized a known prompt into the IDs stored in ref.bin. Decoding
 * those IDs with our tokenizer must reconstruct that prompt text — a real check
 * against ground truth (not a self-round-trip). Also decodes a short greedy
 * continuation and prints it, so the generation loop now yields readable text.
 */
#define _CRT_SECURE_NO_WARNINGS
#include "sp/sp_test.h"
#include "sp_engine/model.h"
#include "sp_engine/tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef SP_QWEN3_GGUF
#define SP_QWEN3_GGUF "Qwen3-0.6B-f16.gguf"
#endif
#ifndef SP_QWEN3_REF
#define SP_QWEN3_REF "qwen3_ref.bin"
#endif

/* distinctive fragments of the ref.bin prompt
 * ("The prime factorization of an integer ... orders the natural numbers by
 *  dominance.") — see tools/oracle invocation. */
#define FRAG_A "factorization"
#define FRAG_B "dominance"

static void TOK_DECODE(void) {
    FILE *f = fopen(SP_QWEN3_REF, "rb");
    SP_CHECK(f != NULL, "open oracle ref.bin (prompt token IDs)");
    if (!f) { fprintf(stderr, "    (no ref: %s)\n", SP_QWEN3_REF); return; }
    uint32_t magic = 0, nt = 0, nv = 0;
    if (fread(&magic, 4, 1, f) != 1 || fread(&nt, 4, 1, f) != 1 || fread(&nv, 4, 1, f) != 1) {
        SP_CHECK(0, "read ref header"); fclose(f); return;
    }
    int32_t *toks = (int32_t *)malloc((size_t)nt * sizeof(int32_t));
    int ok = toks && fread(toks, sizeof(int32_t), nt, f) == nt;
    fclose(f);
    SP_CHECK(ok, "read prompt token IDs");
    if (!ok) { free(toks); return; }

    qwen3_model *m = qwen3_load(SP_QWEN3_GGUF);
    SP_CHECK(m != NULL, "qwen3_load");
    if (!m) { free(toks); return; }

    sp_tokenizer *tk = sp_tokenizer_load(m->gguf);
    SP_CHECK(tk != NULL, "sp_tokenizer_load (tokenizer.ggml.tokens)");
    if (!tk) { free(toks); qwen3_free(m); return; }
    SP_CHECK_EQ_I64(sp_tokenizer_vocab_size(tk), m->cfg.n_vocab, "vocab size matches model");

    char text[4096];
    long len = sp_tokenizer_decode(tk, toks, (int)nt, text, sizeof text);
    SP_CHECK(len > 0 && (size_t)len < sizeof text, "decode produced (untruncated) text");
    if (len > 0) {
        fprintf(stderr, "    decoded prompt: \"%s\"\n", text);
        SP_CHECK(strstr(text, FRAG_A) != NULL, "decoded prompt contains \"" FRAG_A "\"");
        SP_CHECK(strstr(text, FRAG_B) != NULL, "decoded prompt contains \"" FRAG_B "\"");
    }

    /* show the greedy continuation as text */
    int32_t *seq = (int32_t *)malloc((size_t)(nt + 12) * sizeof(int32_t));
    if (seq) {
        memcpy(seq, toks, (size_t)nt * sizeof(int32_t));
        int n = qwen3_generate(m, seq, (int)nt, 12, -1);
        if (n > (int)nt) {
            char gen[1024];
            sp_tokenizer_decode(tk, seq + nt, n - (int)nt, gen, sizeof gen);
            fprintf(stderr, "    greedy continuation: \"%s\"\n", gen);
        }
        free(seq);
    }

    sp_tokenizer_free(tk);
    free(toks);
    qwen3_free(m);
}

int main(void) {
    SP_RUN(TOK_DECODE);
    return SP_DONE();
}
