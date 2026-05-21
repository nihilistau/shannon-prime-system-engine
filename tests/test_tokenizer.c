/* test_tokenizer.c — TOK_DECODE + TOK_ENCODE for the GPT2/Qwen2 byte-level BPE.
 *
 * TOK_DECODE: decode ref.bin's oracle IDs back to the known prompt text (a real
 * check against ground truth, not a self-round-trip).
 *
 * TOK_ENCODE: the hard gate. Encoding the known prompt string must reproduce
 * the EXACT token IDs stock llama.cpp produced (ref.bin's 31 IDs). Plus
 * round-trip (decode(encode)==prompt) and a set of byte-escaped parity cases —
 * CJK / accented Latin / Devanagari combining marks / single-digit splitting /
 * whitespace backtrack / special tokens — whose expected IDs were dumped from
 * the same stock oracle (tools/oracle/gen_encode_fixture.py).
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

/* The exact prompt the oracle tokenized into ref.bin (recovered by decoding the
 * stored IDs; see tools/oracle/gguf_peek.py). */
#define REF_PROMPT \
    "The prime factorization of an integer is the multiset of primes whose " \
    "product is that integer; this lattice of divisibility orders the natural " \
    "numbers by dominance."

#define FRAG_A "factorization"
#define FRAG_B "dominance"

static int32_t *read_ref_ids(uint32_t *n_out) {
    FILE *f = fopen(SP_QWEN3_REF, "rb");
    if (!f) { fprintf(stderr, "    (no ref: %s)\n", SP_QWEN3_REF); return NULL; }
    uint32_t magic = 0, nt = 0, nv = 0;
    if (fread(&magic, 4, 1, f) != 1 || fread(&nt, 4, 1, f) != 1 || fread(&nv, 4, 1, f) != 1) {
        fclose(f); return NULL;
    }
    int32_t *toks = (int32_t *)malloc((size_t)nt * sizeof(int32_t));
    int ok = toks && fread(toks, sizeof(int32_t), nt, f) == nt;
    fclose(f);
    if (!ok) { free(toks); return NULL; }
    *n_out = nt;
    return toks;
}

static void TOK_DECODE(void) {
    uint32_t nt = 0;
    int32_t *toks = read_ref_ids(&nt);
    SP_CHECK(toks != NULL, "read oracle ref.bin (prompt token IDs)");
    if (!toks) return;

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
        SP_CHECK(strcmp(text, REF_PROMPT) == 0, "decoded prompt equals REF_PROMPT exactly");
    }

    sp_tokenizer_free(tk);
    free(toks);
    qwen3_free(m);
}

/* parity cases: byte-escaped UTF-8 prompt + the IDs stock llama.cpp produced. */
static const int32_t IDS_ML[]      = { 97480, 374, 220, 93901, 33983, 1959, 94880, 586,
                                       12984, 20, 11, 15, 15, 13, 72858, 16744, 81705,
                                       84310, 42311, 101, 30484, 99, 43647 };
static const int32_t IDS_DIGITS[]  = { 13683, 16, 17, 18, 750, 220, 19, 20, 21, 22, 856, 15 };
static const int32_t IDS_WS[]      = { 64, 220, 293, 256, 272 };
static const int32_t IDS_SPECIAL[] = { 151644, 872, 198, 9707, 151645 };

typedef struct {
    const char    *name;
    const char    *bytes;   /* byte-escaped UTF-8, no embedded NUL (strlen is the byte count) */
    const int32_t *ids;
    int            n_ids;
} enc_case;

static void check_encode(const sp_tokenizer *tk, const char *name, const char *bytes,
                         const int32_t *exp, int n_exp) {
    int32_t got[256];
    long n = sp_tokenizer_encode(tk, bytes, strlen(bytes), 1 /*parse_special*/, got, 256);
    char msg[128];
    snprintf(msg, sizeof msg, "[%s] token count == %d (got %ld)", name, n_exp, n);
    SP_CHECK(n == n_exp, msg);
    if (n != n_exp) return;
    int match = 1;
    for (int i = 0; i < n_exp; i++) if (got[i] != exp[i]) { match = 0;
        fprintf(stderr, "    [%s] id[%d]: got %d, want %d\n", name, i, got[i], exp[i]); }
    snprintf(msg, sizeof msg, "[%s] all token IDs match the stock-llama oracle", name);
    SP_CHECK(match, msg);
}

static void TOK_ENCODE(void) {
    uint32_t nt = 0;
    int32_t *ref = read_ref_ids(&nt);
    SP_CHECK(ref != NULL, "read oracle ref.bin (prompt token IDs)");
    if (!ref) return;

    qwen3_model *m = qwen3_load(SP_QWEN3_GGUF);
    SP_CHECK(m != NULL, "qwen3_load");
    if (!m) { free(ref); return; }
    sp_tokenizer *tk = sp_tokenizer_load(m->gguf);
    SP_CHECK(tk != NULL, "sp_tokenizer_load");
    if (!tk) { free(ref); qwen3_free(m); return; }

    /* the hard gate: encode the known prompt -> exactly ref.bin's IDs */
    int32_t got[256];
    long n = sp_tokenizer_encode(tk, REF_PROMPT, strlen(REF_PROMPT), 1, got, 256);
    char msg[128];
    snprintf(msg, sizeof msg, "encode(REF_PROMPT) count == %u (got %ld)", nt, n);
    SP_CHECK(n == (long)nt, msg);
    if (n == (long)nt) {
        int match = 1;
        for (uint32_t i = 0; i < nt; i++) if (got[i] != ref[i]) { match = 0;
            fprintf(stderr, "    id[%u]: got %d, want %d\n", i, got[i], ref[i]); }
        SP_CHECK(match, "encode(REF_PROMPT) reproduces stock-llama IDs exactly");
    }

    /* round-trip: decode(encode(prompt)) == prompt */
    if (n > 0 && n <= 256) {
        char back[4096];
        long bl = sp_tokenizer_decode(tk, got, (int)n, back, sizeof back);
        SP_CHECK(bl > 0 && strcmp(back, REF_PROMPT) == 0, "decode(encode(REF_PROMPT)) round-trips");
    }

    /* Unicode + specials + whitespace parity cases */
    const enc_case cases[] = {
        { "ml",       "\xe4\xbb\xb7\xe6\xa0\xbc\x20\x69\x73\x20\xe4\xbe\xa1\xe6\xa0\xbc\x20\xe2\x80\x94\x20\x6e\x61\xc3\xaf\x76\x65\x20\xe2\x82\xac\x35\x2c\x30\x30\x2e\x20\xe4\xb8\xad\xe6\x96\x87\xe6\xb5\x8b\xe8\xaf\x95\x20\xe0\xa4\xb9\xe0\xa4\xbf\xe0\xa4\xa8\xe0\xa5\x8d\xe0\xa4\xa6\xe0\xa5\x80",
          IDS_ML, (int)(sizeof IDS_ML / sizeof *IDS_ML) },
        { "digits",   "\x61\x62\x63\x31\x32\x33\x64\x65\x66\x20\x34\x35\x36\x37\x20\x78\x30",
          IDS_DIGITS, (int)(sizeof IDS_DIGITS / sizeof *IDS_DIGITS) },
        { "ws",       "\x61\x20\x20\x62\x20\x20\x20\x63",
          IDS_WS, (int)(sizeof IDS_WS / sizeof *IDS_WS) },
        { "specials", "\x3c\x7c\x69\x6d\x5f\x73\x74\x61\x72\x74\x7c\x3e\x75\x73\x65\x72\x0a\x48\x65\x6c\x6c\x6f\x3c\x7c\x69\x6d\x5f\x65\x6e\x64\x7c\x3e",
          IDS_SPECIAL, (int)(sizeof IDS_SPECIAL / sizeof *IDS_SPECIAL) },
    };
    for (size_t i = 0; i < sizeof cases / sizeof *cases; i++)
        check_encode(tk, cases[i].name, cases[i].bytes, cases[i].ids, cases[i].n_ids);

    sp_tokenizer_free(tk);
    free(ref);
    qwen3_free(m);
}

int main(void) {
    SP_RUN(TOK_DECODE);
    SP_RUN(TOK_ENCODE);
    return SP_DONE();
}
