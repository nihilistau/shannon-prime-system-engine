/* niah.c — C2.1 Step 3 / G1: Needle-In-A-Haystack retrieval gate for the recall
 * router + two-ring spill/recall. THE direct test of the ±1 projection router,
 * run on the DECODE path (qwen3_generate_kv) where recall/Ring-2 actually live
 * (the prefill qwen3_forward used by sp_perplexity has no recall knobs).
 *
 * Fixture is self-contained — no external NIAH dataset, no Python/HF tokenizer:
 *   1. read a raw-text haystack (Alice / wiki.test.raw),
 *   2. tokenize it with the ENGINE's own tokenizer (sp_tokenizer_encode — the
 *      same call sp_perplexity uses; "validated to reproduce stock llama.cpp IDs"),
 *   3. inject an out-of-distribution needle ("...password ... is <secret>.") at a
 *      token-space depth, truncate the haystack so the prompt hits ~N tokens,
 *   4. append the retrieval question,
 *   5. prefill + greedily decode SP_NIAH_GEN tokens via qwen3_generate_kv,
 *   6. decode the output and report whether <secret> survived.
 *
 * The recall knobs (SP_RECALL_B/R/W, SP_RING2) are read by the forward itself,
 * so a driver script sweeps depth x N x B by re-invoking this binary with env set.
 * recall OFF (SP_RECALL_B unset/0) == full-attention baseline reference.
 *
 * Env:
 *   SP_NIAH_GGUF    model GGUF            (default SP_QWEN3_GGUF compile-def)
 *   SP_NIAH_CORPUS  haystack text path    (REQUIRED; absolute is safest)
 *   SP_NIAH_N       target prompt tokens  (haystack truncated to fit; default 2048)
 *   SP_NIAH_DEPTH   needle depth percent  (0..100, default 50)
 *   SP_NIAH_GEN     tokens to generate    (default 24)
 *   SP_NIAH_SECRET  secret digits         (default 837492)
 */
#define _CRT_SECURE_NO_WARNINGS
#include "sp_engine/gguf.h"
#include "sp_engine/model.h"
#include "sp_engine/sp_model.h"   /* SP_NIAH_SP: production swivel loader */
#include "sp_engine/ring2_arm.h"  /* SP_RING2_OPTANE_DIR: physical Ring-2 store */
#include "sp_engine/tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#ifndef SP_QWEN3_GGUF
#define SP_QWEN3_GGUF "Qwen3-0.6B-f16.gguf"
#endif

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

static int env_int(const char *k, int dflt) { const char *e = getenv(k); return e ? atoi(e) : dflt; }

int main(void) {
    const char *gguf   = getenv("SP_NIAH_GGUF");   if (!gguf)   gguf   = SP_QWEN3_GGUF;
    const char *corpus = getenv("SP_NIAH_CORPUS");
    const char *secret = getenv("SP_NIAH_SECRET"); if (!secret) secret = "837492";
    int N     = env_int("SP_NIAH_N", 2048);
    int depth = env_int("SP_NIAH_DEPTH", 50);   if (depth < 0) depth = 0; if (depth > 100) depth = 100;
    int n_gen = env_int("SP_NIAH_GEN", 24);     if (n_gen < 1) n_gen = 24;

    if (!corpus) { fprintf(stderr, "[niah] set SP_NIAH_CORPUS to the haystack text path\n"); return 2; }

    size_t clen = 0; char *text = read_file(corpus, &clen);
    if (!text) { fprintf(stderr, "[niah] cannot read corpus: %s\n", corpus); return 2; }

    /* Live-Optane Ring-2 (same hook as sp_toks): SP_RING2_OPTANE_DIR registers
     * the dual-size NO_BUFFERING+IOCP store as THE ARM backend pre-decode. */
    if (getenv("SP_RING2_OPTANE_DIR")) {
        if (sp_ring2_optane_register_env())
            fprintf(stderr, "[niah] WARN: SP_RING2_OPTANE_DIR set but registration failed\n");
    }

    gguf_ctx *g = gguf_open(gguf);
    if (!g) { fprintf(stderr, "[niah] gguf_open FAIL: %s\n", gguf); free(text); return 1; }
    /* SP_NIAH_SP=<file.sp-model>: load weights via the production swivel path
     * (packed OK_Q8 arena, ~20x faster prefill than the raw-f16 reference) —
     * the same loader sp_toks uses. Tokenizer still read from the GGUF (same
     * vocab; the gguf stays the tokenizer source either way). */
    qwen3_model *m = NULL;
    const char *sp_path = getenv("SP_NIAH_SP");
    if (sp_path && sp_path[0]) {
        char tok_path[1024];
        snprintf(tok_path, sizeof(tok_path), "%s", sp_path);
        char *dot = strrchr(tok_path, '.');
        if (dot && strcmp(dot, ".sp-model") == 0) strcpy(dot, ".sp-tokenizer");
        sp_model *spm = NULL;
        if (sp_model_load(sp_path, tok_path, &spm) != SP_OK || !spm ||
            !(m = sp_model_to_qwen3(spm))) {
            fprintf(stderr, "[niah] SP_NIAH_SP load FAIL: %s\n", sp_path); return 1;
        }
        fprintf(stderr, "[niah] weights via swivel: %s (.sp-model OK_Q8 arena)\n", sp_path);
    } else {
        m = qwen3_load(gguf);
    }
    sp_tokenizer *tok = sp_tokenizer_load(g);
    if (!m || !tok) { fprintf(stderr, "[niah] load model/tokenizer FAIL\n"); return 1; }

    /* needle + question, tokenized with the model's own tokenizer (no special toks) */
    char needle[256], question[256];
    snprintf(needle, sizeof needle,
             " The secret password for the Optane vault is %s. Remember it. ", secret);
    snprintf(question, sizeof question,
             " What is the secret password for the Optane vault? The password is");

    int cap = N + 4096;
    int32_t *hay = (int32_t *)malloc((size_t)(clen + 16) * sizeof(int32_t)); /* >= max possible tokens */
    int32_t *ndl = (int32_t *)malloc(256 * sizeof(int32_t));
    int32_t *qst = (int32_t *)malloc(256 * sizeof(int32_t));
    int32_t *seq = (int32_t *)malloc((size_t)(cap + n_gen) * sizeof(int32_t));
    if (!hay || !ndl || !qst || !seq) { fprintf(stderr, "[niah] OOM\n"); return 1; }

    long nh = sp_tokenizer_encode(tok, text, clen, 0, hay, (int)(clen + 16));
    long nn = sp_tokenizer_encode(tok, needle,   strlen(needle),   0, ndl, 256);
    long nq = sp_tokenizer_encode(tok, question, strlen(question), 0, qst, 256);
    if (nh < 1 || nn < 1 || nq < 1) { fprintf(stderr, "[niah] tokenize FAIL (nh=%ld nn=%ld nq=%ld)\n", nh, nn, nq); return 1; }

    /* budget of haystack tokens so prompt ~= N: N - needle - question */
    long budget = (long)N - nn - nq;
    if (budget < 16) budget = 16;
    if (budget > nh) budget = nh;                 /* corpus shorter than requested N */
    long inj = (budget * depth) / 100;            /* needle insertion point in token space */
    if (inj > budget) inj = budget;

    /* seq = haystack[0:inj] + needle + haystack[inj:budget] + question */
    int p = 0;
    for (long i = 0; i < inj; i++)        seq[p++] = hay[i];
    for (long i = 0; i < nn; i++)         seq[p++] = ndl[i];
    for (long i = inj; i < budget; i++)   seq[p++] = hay[i];
    for (long i = 0; i < nq; i++)         seq[p++] = qst[i];
    int n_prompt = p;

    /* greedy decode on the recall/Ring-2 path */
    int n = qwen3_generate_kv(m, seq, n_prompt, n_gen, -1);
    if (n < n_prompt) { fprintf(stderr, "[niah] generate FAIL (n=%d)\n", n); return 1; }

    char out[4096];
    long ob = sp_tokenizer_decode(tok, seq + n_prompt, n - n_prompt, out, sizeof out);
    if (ob < 0) out[0] = '\0';
    int hit = (strstr(out, secret) != NULL);

    /* config line: echo the recall knobs the forward actually read */
    const char *rb = getenv("SP_RECALL_B"); const char *rw = getenv("SP_RECALL_W");
    const char *rr = getenv("SP_RECALL_R"); const char *r2 = getenv("SP_RING2");
    /* sanitize the answer for single-line logging */
    for (char *s = out; *s; s++) if (*s == '\n' || *s == '\r' || *s == '\t') *s = ' ';

    fprintf(stderr,
        "[niah] %-4s N=%d(actual=%d) depth=%d%% inj_tok=%ld  B=%s W=%s R=%s RING2=%s  answer=\"%.60s\"\n",
        hit ? "HIT" : "MISS", N, n_prompt, depth, inj,
        rb ? rb : "off", rw ? rw : "-", rr ? rr : "-", r2 ? r2 : "-", out);

    free(text); free(hay); free(ndl); free(qst); free(seq);
    sp_tokenizer_free(tok); qwen3_free(m); gguf_close(g);
    sp_ring2_optane_unregister();   /* prints read-latency + cache hit-rate stats */
    return hit ? 0 : 3;   /* 0 = retrieved, 3 = missed (driver aggregates) */
}
