/* sp_tok_enc.c — dump token IDs for a text file using a .sp-tokenizer blob.
 *
 * CONTRACT-CHAT-FULLSTACK B3 (AUTONOMOUS RECALL) episode-capture helper. Unlike
 * sp_tok_dump (which needs a vocab GGUF — the dead GGUF lane on the 12B), this
 * loads the .sp-tokenizer blob directly via sp_tokenizer_load_tokfile (the
 * gemma4 GEMMA4_BPE lane the daemon itself uses), so the produced token-ID
 * stream is byte-identical to what /v1/chat tokenizes a passage to. The PPL
 * episode-capture harness (test_gemma4_ppl_cuda + SP_XBAR_RECALL_WRITE +
 * SP_PPL_TOKENS) consumes this stream to mint a new XBAR episode from arbitrary
 * factual text.
 *
 * BOS is auto-prepended (gemma4 forces add_bos=1), matching the PPL fixture
 * convention (first id = 2).
 *
 * usage: sp_tok_enc <tokenizer.sp-tokenizer> <text-file> [max_tokens]
 */
#define _CRT_SECURE_NO_WARNINGS
#include "sp_engine/tokenizer.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <tokenizer.sp-tokenizer> <text-file> [max_tokens]\n", argv[0]);
        return 1;
    }
    sp_tokenizer *t = sp_tokenizer_load_tokfile(argv[1]);
    if (!t) { fprintf(stderr, "sp_tokenizer_load_tokfile failed: %s\n", argv[1]); return 1; }

    FILE *f = fopen(argv[2], "rb");
    if (!f) { fprintf(stderr, "open failed: %s\n", argv[2]); return 1; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char *text = (char *)malloc((size_t)sz + 1);
    if (!text || (sz > 0 && fread(text, 1, (size_t)sz, f) != (size_t)sz)) {
        fprintf(stderr, "read failed\n"); return 1;
    }
    text[sz] = '\0'; fclose(f);

    long cap = sz + 64;
    int32_t *ids = (int32_t *)malloc((size_t)cap * sizeof(int32_t));
    if (!ids) { fprintf(stderr, "OOM\n"); return 1; }
    long n = sp_tokenizer_encode(t, text, (size_t)sz, 0, ids, (int)cap);
    if (n < 0) { fprintf(stderr, "encode failed\n"); return 1; }
    if (n > cap) n = cap;

    long maxt = (argc > 3) ? atol(argv[3]) : 0;
    if (maxt > 0 && n > maxt) n = maxt;
    for (long i = 0; i < n; i++) printf("%d\n", ids[i]);
    fprintf(stderr, "[sp_tok_enc] %ld tokens (first=%d)\n", n, n > 0 ? ids[0] : -1);

    free(ids); free(text);
    sp_tokenizer_free(t);
    return 0;
}
