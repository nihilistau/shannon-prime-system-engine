/* sp_tok_dump.c — dump token IDs for a text file using a GGUF's tokenizer.
 *
 * ETA.5b PPL-gate fixture generator: produces the whitespace-separated token-ID
 * stream the PPL harnesses consume (the test_gemma4_ppl fixture convention:
 * BOS first — sp_tokenizer_encode auto-prepends it when the GGUF sets
 * add_bos_token=1, which Gemma does). The SP tokenizer's encode parity vs the
 * llama.cpp oracle is separately gated (the 3-G4 transcode/round-trip gates);
 * this tool exists because llama-tokenize in the available reference build
 * faults on gemma4 hparams while its perplexity/bench paths run fine.
 *
 * usage: sp_tok_dump <model.gguf> <text-file> [max_tokens]
 */
#define _CRT_SECURE_NO_WARNINGS
#include "sp_engine/gguf.h"
#include "sp_engine/tokenizer.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.gguf> <text-file> [max_tokens]\n", argv[0]);
        return 1;
    }
    gguf_ctx *g = gguf_open(argv[1]);
    if (!g) { fprintf(stderr, "gguf_open failed: %s\n", argv[1]); return 1; }
    sp_tokenizer *t = sp_tokenizer_load(g);
    if (!t) { fprintf(stderr, "sp_tokenizer_load failed\n"); return 1; }

    FILE *f = fopen(argv[2], "rb");
    if (!f) { fprintf(stderr, "open failed: %s\n", argv[2]); return 1; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char *text = (char *)malloc((size_t)sz + 1);
    if (!text || (sz > 0 && fread(text, 1, (size_t)sz, f) != (size_t)sz)) {
        fprintf(stderr, "read failed\n"); return 1;
    }
    text[sz] = '\0'; fclose(f);

    long cap = sz + 64;                      /* tokens <= bytes for BPE/SPM + slack */
    int32_t *ids = (int32_t *)malloc((size_t)cap * sizeof(int32_t));
    if (!ids) { fprintf(stderr, "OOM\n"); return 1; }
    long n = sp_tokenizer_encode(t, text, (size_t)sz, 0, ids, (int)cap);
    if (n < 0) { fprintf(stderr, "encode failed\n"); return 1; }
    if (n > cap) n = cap;

    long maxt = (argc > 3) ? atol(argv[3]) : 0;
    if (maxt > 0 && n > maxt) n = maxt;
    for (long i = 0; i < n; i++) printf("%d\n", ids[i]);
    fprintf(stderr, "[sp_tok_dump] %ld tokens (first=%d)\n", n, n > 0 ? ids[0] : -1);

    free(ids); free(text);
    sp_tokenizer_free(t); gguf_close(g);
    return 0;
}
