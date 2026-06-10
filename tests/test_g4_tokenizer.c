/* test_g4_tokenizer.c — T_G4_TOK_PARITY + T_G4_TOK_ROUNDTRIP (issue #115).
 *
 * T_G4_TOK_PARITY (exact, no tolerance): encode the 24 KB wikitext head and
 * require id-for-id equality with the verified fixture, on BOTH lanes:
 *   (a) GGUF lane     — sp_tokenizer_load on the gemma4 vocab GGUF;
 *   (b) blob lane     — sp_transcode --tok-only -> .sp-tokenizer (family tag
 *                       GEMMA4_BPE=4) -> sp_tokenizer_load_tokfile.
 *
 * GROUND TRUTH PROVENANCE (reconstructed + reverified 2026-06-10):
 *   fixtures/g4tok/wiki24k.txt = EXACTLY the first 24576 bytes of
 *   archive/eval/wikitext-103-raw/wiki.test.raw (wikitext-103 test split).
 *   fixtures/g4tok/wiki24k.ids.txt = _g4_12b_wiki_tokens.txt, the llama-dumped
 *   fixture of the gold campaign (5432 ids = BOS 2 + 5431), proven == HF
 *   tokenizer.json 5431/5431 during the 06-R8 campaign and RE-VERIFIED fresh
 *   against HF `tokenizers` 0.22.2 with the official gemma-4-12b bucket
 *   tokenizer.json: encode(wiki24k.txt, add_special=False) == fixture[1:],
 *   5431/5431, including the truncation-tail token 8475 (U+2581 "Ba").
 *
 * T_G4_TOK_ROUNDTRIP: decode(encode(x)) == x (byte-exact, BOS skipped by
 * decode) over the 24 KB corpus + adversarial strings (emoji, CJK, mixed
 * spaces/newlines/CRLF/tabs, raw control bytes incl. NUL, invalid UTF-8 ->
 * <0xNN> byte-fallback coverage). Strings containing a literal U+2581 are
 * excluded BY DESIGN: the SPM-style space escape aliases U+2581 with ' '
 * (llama.cpp has the identical property).
 *
 * Perf telemetry (no gate): encode MB/s with the hashed pair-rank table.
 *
 * Skips cleanly (like M_GEMMA4) when the out-of-tree vocab GGUF is absent. */
#define _CRT_SECURE_NO_WARNINGS
#include "sp/sp_test.h"
#include "sp/sp_model.h"
#include "sp/sp_status.h"
#include "sp_engine/gguf.h"
#include "sp_engine/tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef SP_G4VOCAB_GGUF
#define SP_G4VOCAB_GGUF "ggml-vocab-gemma-4.gguf"
#endif
#ifndef SP_G4TOK_FIXDIR
#define SP_G4TOK_FIXDIR "tests/fixtures/g4tok"
#endif
#ifndef SP_TRANSCODE_BIN
#define SP_TRANSCODE_BIN "sp_transcode"
#endif
#ifndef SP_FMT_OUT_DIR
#define SP_FMT_OUT_DIR "."
#endif
/* #115 deployment leg: the INSTALLED 12B artifacts (out-of-tree; skip if absent).
 * Env SP_G4_12B_SPTOK / SP_G4_12B_SPMODEL override (run the gate against the
 * -qat / -st / plain pairs the same way). */
#ifndef SP_G4_12B_SPTOK_DEF
#define SP_G4_12B_SPTOK_DEF "D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-tokenizer"
#endif
#ifndef SP_G4_12B_SPMODEL_DEF
#define SP_G4_12B_SPMODEL_DEF "D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-model"
#endif

#define MAX_IDS 8192

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

static int file_exists(const char *path) {
    FILE *f = fopen(path, "rb");
    if (f) { fclose(f); return 1; }
    return 0;
}

static double now_s(void) {
    struct timespec t; timespec_get(&t, TIME_UTC);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

/* shared state across the two registered test mains (one-binary pattern) */
static char  *g_text;  static size_t  g_text_len;
static int32_t g_want[MAX_IDS]; static int g_nwant;

static int load_fixtures(void) {
    char p[1024]; size_t ilen = 0;
    snprintf(p, sizeof p, "%s/wiki24k.txt", SP_G4TOK_FIXDIR);
    g_text = read_file(p, &g_text_len);
    snprintf(p, sizeof p, "%s/wiki24k.ids.txt", SP_G4TOK_FIXDIR);
    char *ids = read_file(p, &ilen);
    if (!g_text || !ids) { free(ids); return 0; }
    g_nwant = parse_ids(ids, g_want, MAX_IDS);
    free(ids);
    return g_text_len == 24576 && g_nwant == 5432;
}

/* exact id-parity of encode(corpus) vs the fixture; reports N/N. */
static void check_parity(sp_tokenizer *tk, const char *lane) {
    char lbl[128];
    int32_t *got = (int32_t *)malloc(MAX_IDS * sizeof(int32_t));
    SP_CHECK(got != NULL, "alloc ids");
    if (!got) return;

    double t0 = now_s();
    long n = sp_tokenizer_encode(tk, g_text, g_text_len, /*parse_special=*/0, got, MAX_IDS);
    double dt = now_s() - t0;

    snprintf(lbl, sizeof lbl, "[%s] token count == %d (got %ld)", lane, g_nwant, n);
    SP_CHECK(n == g_nwant, lbl);
    int match = 0, shown = 0;
    if (n == g_nwant) {
        for (int i = 0; i < g_nwant; i++) {
            if (got[i] == g_want[i]) { match++; continue; }
            if (shown++ < 8)   /* print only the first few mismatches */
                fprintf(stderr, "    [%s] id[%d]: got %d want %d\n", lane, i, got[i], g_want[i]);
        }
    }
    snprintf(lbl, sizeof lbl, "[%s] PARITY %d/%d exact vs llama-dumped==HF fixture",
             lane, match, g_nwant);
    fprintf(stderr, "    %s\n", lbl);   /* receipt: print the count pass or fail */
    SP_CHECK(match == g_nwant, lbl);
    fprintf(stderr, "    [%s] telemetry: encode %.0f KB in %.3f ms = %.1f MB/s\n",
            lane, g_text_len / 1024.0, dt * 1e3, g_text_len / (1024.0 * 1024.0) / (dt > 0 ? dt : 1));
    free(got);
}

/* decode(encode(x)) == x byte-exact (decode skips BOS). */
static void check_roundtrip(sp_tokenizer *tk, const char *name,
                            const char *bytes, size_t blen) {
    char lbl[128];
    int32_t ids[MAX_IDS];
    long n = sp_tokenizer_encode(tk, bytes, blen, 0, ids, MAX_IDS);
    snprintf(lbl, sizeof lbl, "[rt:%s] encode ok (%ld toks)", name, n);
    SP_CHECK(n >= 0 && n <= MAX_IDS, lbl);
    if (n < 0 || n > MAX_IDS) return;
    size_t cap = 4 * blen + 16;
    char *back = (char *)malloc(cap);
    SP_CHECK(back != NULL, "alloc decode buf");
    if (!back) return;
    long bl = sp_tokenizer_decode(tk, ids, (int)n, back, cap);
    snprintf(lbl, sizeof lbl, "[rt:%s] decode(encode(x)) == x (%lu bytes)",
             name, (unsigned long)blen);
    SP_CHECK(bl == (long)blen && memcmp(back, bytes, blen) == 0, lbl);
    if (bl != (long)blen)
        fprintf(stderr, "    [rt:%s] length %ld != %lu\n", name, bl, (unsigned long)blen);
    free(back);
}

static void T_G4_TOK_PARITY(void) {
    SP_CHECK(load_fixtures(), "load g4tok fixtures (24576 bytes / 5432 ids)");
    if (!g_text) return;
    if (!file_exists(SP_G4VOCAB_GGUF)) {
        fprintf(stderr, "    SKIP: vocab GGUF absent: %s\n", SP_G4VOCAB_GGUF);
        return;
    }

    /* lane (a): GGUF -> sp_tokenizer_load (family dispatch on tokenizer.ggml.model) */
    gguf_ctx *g = gguf_open(SP_G4VOCAB_GGUF);
    SP_CHECK(g != NULL, "gguf_open gemma4 vocab GGUF");
    if (g) {
        sp_tokenizer *tk = sp_tokenizer_load(g);
        SP_CHECK(tk != NULL, "sp_tokenizer_load dispatches gemma4 (model=gemma4)");
        if (tk) {
            SP_CHECK_EQ_I64(sp_tokenizer_vocab_size(tk), 262144, "vocab size 262144");
            check_parity(tk, "gguf-lane");
            sp_tokenizer_free(tk);
        }
        gguf_close(g);
    }

    /* lane (b): sp_transcode --tok-only -> family-tagged blob -> blob loader */
    char tokf[1024], cmd[3072];
    snprintf(tokf, sizeof tokf, "%s/g4vocab_rt.sp-tokenizer", SP_FMT_OUT_DIR);
    snprintf(cmd, sizeof cmd, "\"%s\" --tok-only \"%s\" \"%s\"",
             SP_TRANSCODE_BIN, SP_G4VOCAB_GGUF, tokf);
#ifdef _WIN32
    /* cmd.exe strips the outer quotes of a quoted command line; re-wrap. */
    {
        char wrapped[3300];
        snprintf(wrapped, sizeof wrapped, "\"%s\"", cmd);
        SP_CHECK(system(wrapped) == 0, "sp_transcode --tok-only (family-tagged blob)");
    }
#else
    SP_CHECK(system(cmd) == 0, "sp_transcode --tok-only (family-tagged blob)");
#endif
    sp_tokenizer *bk = sp_tokenizer_load_tokfile(tokf);
    SP_CHECK(bk != NULL, "sp_tokenizer_load_tokfile dispatches GEMMA4_BPE (type_id=4)");
    if (bk) {
        check_parity(bk, "blob-lane");
        sp_tokenizer_free(bk);
    }
}

static void T_G4_TOK_ROUNDTRIP(void) {
    if (!g_text) { SP_CHECK(load_fixtures(), "load g4tok fixtures"); }
    if (!g_text) return;
    if (!file_exists(SP_G4VOCAB_GGUF)) {
        fprintf(stderr, "    SKIP: vocab GGUF absent: %s\n", SP_G4VOCAB_GGUF);
        return;
    }
    /* blob-lane tokenizer (the production .sp-model pairing path) */
    char tokf[1024];
    snprintf(tokf, sizeof tokf, "%s/g4vocab_rt.sp-tokenizer", SP_FMT_OUT_DIR);
    sp_tokenizer *tk = file_exists(tokf) ? sp_tokenizer_load_tokfile(tokf) : NULL;
    if (!tk) {   /* PARITY may not have run first; fall back to the GGUF lane */
        gguf_ctx *g = gguf_open(SP_G4VOCAB_GGUF);
        if (g) tk = sp_tokenizer_load(g);   /* tokenizer borrows g; leak g (test exit) */
    }
    SP_CHECK(tk != NULL, "gemma4 tokenizer available");
    if (!tk) return;

    /* ground-truth decode: decode(fixture ids) == corpus (BOS skipped) */
    {
        char *back = (char *)malloc(g_text_len + 64);
        SP_CHECK(back != NULL, "alloc");
        if (back) {
            long bl = sp_tokenizer_decode(tk, g_want, g_nwant, back, g_text_len + 64);
            SP_CHECK(bl == (long)g_text_len && memcmp(back, g_text, g_text_len) == 0,
                     "decode(fixture 5432 ids) == 24 KB corpus byte-exact");
            free(back);
        }
    }

    /* corpus round-trip */
    check_roundtrip(tk, "wiki24k", g_text, g_text_len);

    /* adversarial round-trips (byte-escaped; NUL + invalid UTF-8 via explicit len) */
    static const struct { const char *name; const char *b; int len; } cases[] = {
        { "plain",     "Hello world", -1 },
        { "spaces",    "  leading, trailing  , and   multiple   spaces ", -1 },
        { "newlines",  "line1\nline2\n\n\nline3\n\n", -1 },
        { "nl-only",   "\n", -1 },
        { "nl-run",    "\n\n\n\n\n\n\n", -1 },
        { "crlf",      "a\r\nb\r\rc\n\r", -1 },
        { "tabs",      "col1\tcol2\t\tend\t", -1 },
        { "emoji",     "smile \xF0\x9F\x99\x82 rocket\xF0\x9F\x9A\x80 flag \xF0\x9F\x87\xA6\xF0\x9F\x87\xBA!", -1 },
        { "cjk",       "\xE4\xB8\xAD\xE6\x96\x87\xE6\xB5\x8B\xE8\xAF\x95 \xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E \xED\x95\x9C\xEA\xB5\xAD\xEC\x96\xB4", -1 },
        { "accents",   "na\xC3\xAFve fa\xC3\xA7" "ade \xE2\x82\xAC" "5,00 \xC4\x85\xC4\x99\xC5\x82", -1 },
        { "controls",  "\x01\x02\x03 bell\x07 esc\x1B[0m del\x7F", -1 },
        { "nul",       "a\x00b\x00\x00" "c", 6 },
        { "bad-utf8",  "ok \xFF\xFE\x80\xC0 end", 11 },
        { "mixed",     " \n mixed \t\n  space\xC2\xA0nbsp \n\n tail", -1 },
    };
    for (size_t i = 0; i < sizeof cases / sizeof *cases; i++) {
        size_t L = cases[i].len >= 0 ? (size_t)cases[i].len : strlen(cases[i].b);
        check_roundtrip(tk, cases[i].name, cases[i].b, L);
    }
    sp_tokenizer_free(tk);
}

/* T_G4_TOK_12B_PAIRED (#115 deployment leg, "12B text-in"):
 *   (1) the INSTALLED 12B .sp-tokenizer must load through the blob lane
 *       (sp_tokenizer_load_tokfile — a legacy type_id=2 blob would dispatch
 *       to the GPT2 lane and fail parity; a future unknown tag hard-errors);
 *   (2) encode(24 KB corpus) == the HF-verified fixture, 5432/5432 EXACT;
 *   (3) if the paired .sp-model is present, sp_model_load must accept the
 *       pair — i.e. header.tokenizer_hash == SHA-256(blob file) (§8 13-16).
 * Skips cleanly when the out-of-tree artifacts are absent. */
static void T_G4_TOK_12B_PAIRED(void) {
    if (!g_text) { SP_CHECK(load_fixtures(), "load g4tok fixtures"); }
    if (!g_text) return;
    const char *spt = getenv("SP_G4_12B_SPTOK");
    const char *spm = getenv("SP_G4_12B_SPMODEL");
    if (!spt) spt = SP_G4_12B_SPTOK_DEF;
    if (!spm) spm = SP_G4_12B_SPMODEL_DEF;
    if (!file_exists(spt)) {
        fprintf(stderr, "    SKIP: installed 12B blob absent: %s\n", spt);
        return;
    }
    fprintf(stderr, "    blob:  %s\n    model: %s\n", spt, spm);

    sp_tokenizer *tk = sp_tokenizer_load_tokfile(spt);
    SP_CHECK(tk != NULL, "installed 12B blob loads (family-tagged, blob lane)");
    if (!tk) return;
    SP_CHECK_EQ_I64(sp_tokenizer_vocab_size(tk), 262144, "12B blob vocab 262144");
    check_parity(tk, "12b-installed-blob");
    sp_tokenizer_free(tk);

    if (!file_exists(spm)) {
        fprintf(stderr, "    NOTE: paired .sp-model absent (%s) — pairing leg skipped\n", spm);
        return;
    }
    sp_model *h = NULL;
    sp_status st = sp_model_load(spm, spt, &h);
    if (st != SP_OK) fprintf(stderr, "    sp_model_load: %s\n", sp_last_error());
    SP_CHECK(st == SP_OK && h != NULL,
             "sp_model_load accepts the pair (tokenizer_hash == SHA-256(blob))");
    if (h) sp_model_unload(h);
}

int main(void) {
    SP_RUN(T_G4_TOK_PARITY);
    SP_RUN(T_G4_TOK_ROUNDTRIP);
    SP_RUN(T_G4_TOK_12B_PAIRED);
    return SP_DONE();
}
