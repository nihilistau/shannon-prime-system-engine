/* test_spm.c — SPM_ENCODE: the engine's SentencePiece ("llama") tokenizer
 * reproduces the stock llama.cpp token IDs byte-for-byte on Gemma3-1B. For each
 * fixture tests/fixtures/spm/pN.{txt,ids}: encode(txt) must equal the oracle IDs
 * (dump_tokens, add_special=parse_special=1). Also checks owning-mode load (the
 * unigram scores survive a GGUF source release). */
#define _CRT_SECURE_NO_WARNINGS
#include "sp/sp_test.h"
#include "sp_engine/gguf.h"
#include "sp_engine/tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef SP_GEMMA3_GGUF
#define SP_GEMMA3_GGUF "gemma-3-1b-it-f16.gguf"
#endif
#ifndef SP_SPM_FIXDIR
#define SP_SPM_FIXDIR "tests/fixtures/spm"
#endif

#define NFIX 5

static char *read_file(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    if (sz < 0) { fclose(f); return NULL; }
    char *b = (char *)malloc((size_t)sz + 1);
    if (!b) { fclose(f); return NULL; }
    if (sz > 0 && fread(b, 1, (size_t)sz, f) != (size_t)sz) { free(b); fclose(f); return NULL; }
    b[sz] = '\0'; *len = (size_t)sz; fclose(f);
    return b;
}

/* parse whitespace-separated decimal ints from `s` into out (cap), return count */
static int parse_ids(const char *s, int32_t *out, int cap) {
    int n = 0; char *end;
    for (;;) {
        while (*s == ' ' || *s == '\n' || *s == '\r' || *s == '\t') s++;
        if (!*s) break;
        long v = strtol(s, &end, 10);
        if (end == s) break;
        if (n < cap) out[n] = (int32_t)v;
        n++; s = end;
    }
    return n;
}

static void check_fixture(const sp_tokenizer *tok, int idx) {
    char ptxt[512], pids[512];
    snprintf(ptxt, sizeof ptxt, "%s/p%d.txt", SP_SPM_FIXDIR, idx);
    snprintf(pids, sizeof pids, "%s/p%d.ids", SP_SPM_FIXDIR, idx);
    size_t tlen = 0, ilen = 0;
    char *text = read_file(ptxt, &tlen);
    char *idss = read_file(pids, &ilen);
    char lbl[64]; snprintf(lbl, sizeof lbl, "p%d: load fixture", idx);
    SP_CHECK(text && idss, lbl);
    if (!text || !idss) { free(text); free(idss); return; }

    int32_t want[256]; int nwant = parse_ids(idss, want, 256);
    int32_t got[256];
    long ngot = sp_tokenizer_encode(tok, text, tlen, /*parse_special=*/1, got, 256);

    snprintf(lbl, sizeof lbl, "p%d: token count (%ld vs oracle %d)", idx, ngot, nwant);
    SP_CHECK_EQ_I64(ngot, nwant, lbl);
    int ok = (ngot == nwant);
    for (int i = 0; ok && i < nwant; i++) if (got[i] != want[i]) {
        fprintf(stderr, "      p%d mismatch at %d: got %d want %d\n", idx, i, got[i], want[i]);
        ok = 0;
    }
    snprintf(lbl, sizeof lbl, "p%d: byte-parity with oracle IDs", idx);
    SP_CHECK(ok, lbl);
    free(text); free(idss);
}

static void SPM_ENCODE(void) {
    gguf_ctx *g = gguf_open(SP_GEMMA3_GGUF);
    SP_CHECK(g != NULL, "open gemma3 GGUF");
    if (!g) return;
    sp_tokenizer *tok = sp_tokenizer_load(g);
    SP_CHECK(tok != NULL, "load SPM tokenizer");
    if (tok) {
        for (int i = 1; i <= NFIX; i++) check_fixture(tok, i);
        sp_tokenizer_free(tok);
    }

    /* owning mode: copy vocab + scores, drop the GGUF, still encode (scores must
     * not dangle). Re-check fixture 1. */
    sp_tokenizer *owned = sp_tokenizer_load_ex(g, 1);
    SP_CHECK(owned != NULL, "load owning SPM tokenizer");
    gguf_close(g);   /* mapping gone; owning tokenizer must survive */
    if (owned) {
        check_fixture(owned, 1);
        sp_tokenizer_free(owned);
    }
}

int main(void) { SP_RUN(SPM_ENCODE); return SP_DONE(); }
