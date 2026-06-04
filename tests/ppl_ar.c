/* ppl_ar.c — C2.1 Step 3 / G2: autoregressive (DECODE-path) perplexity.
 *
 * Feeds a real corpus through qwen3_ppl_decode (teacher-forced decode), so the
 * recall router + two-ring are measured EXACTLY as production generates — unlike
 * sp_perplexity, which runs the dense prefill qwen3_forward that has no recall
 * knobs. The G2 metric is the deflection:  PPL(recall on) - PPL(full attention),
 * gate target < ~2%. Run the same binary twice (recall off vs on) and diff.
 *
 * Tokenized with the ENGINE's own validated tokenizer (no Python/HF). SLOW by
 * design (a full forward + vocab projection at every position); bake detached.
 *
 * Env:
 *   SP_PPLAR_GGUF    model GGUF              (default SP_QWEN3_GGUF compile-def)
 *   SP_PPLAR_CORPUS  corpus text path        (REQUIRED; absolute is safest)
 *   SP_PPLAR_N       cap tokens scored        (default 2048)
 *   SP_PPLAR_WARM    skip cold prefix (warm)  (default N/4)
 *   (recall knobs SP_RECALL_B/R/W, SP_RING2 read by the decode forward itself)
 */
#define _CRT_SECURE_NO_WARNINGS
#include "sp_engine/gguf.h"
#include "sp_engine/model.h"
#include "sp_engine/sp_model.h"   /* SP_PPLAR_SP: production swivel loader */
#include "sp_engine/tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

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
static double now_s(void) { struct timespec t; timespec_get(&t, TIME_UTC); return (double)t.tv_sec + (double)t.tv_nsec * 1e-9; }

int main(void) {
    const char *gguf   = getenv("SP_PPLAR_GGUF");   if (!gguf)   gguf   = SP_QWEN3_GGUF;
    const char *corpus = getenv("SP_PPLAR_CORPUS");
    int N    = env_int("SP_PPLAR_N", 2048);
    int warm = env_int("SP_PPLAR_WARM", N / 4);
    if (!corpus) { fprintf(stderr, "[ppl_ar] set SP_PPLAR_CORPUS to the corpus text path\n"); return 2; }

    size_t clen = 0; char *text = read_file(corpus, &clen);
    if (!text) { fprintf(stderr, "[ppl_ar] cannot read corpus: %s\n", corpus); return 2; }

    gguf_ctx *g = gguf_open(gguf);
    if (!g) { fprintf(stderr, "[ppl_ar] gguf_open FAIL: %s\n", gguf); return 1; }
    /* SP_PPLAR_SP=<file.sp-model>: weights via the production swivel path
     * (packed OK_Q8 arena) — same hook as niah/sp_toks; tokenizer stays GGUF. */
    qwen3_model *m = NULL;
    const char *sp_path = getenv("SP_PPLAR_SP");
    if (sp_path && sp_path[0]) {
        char tok_path[1024];
        snprintf(tok_path, sizeof(tok_path), "%s", sp_path);
        char *dot = strrchr(tok_path, '.');
        if (dot && strcmp(dot, ".sp-model") == 0) strcpy(dot, ".sp-tokenizer");
        sp_model *spm = NULL;
        if (sp_model_load(sp_path, tok_path, &spm) != SP_OK || !spm ||
            !(m = sp_model_to_qwen3(spm))) {
            fprintf(stderr, "[ppl_ar] SP_PPLAR_SP load FAIL: %s\n", sp_path); return 1;
        }
        fprintf(stderr, "[ppl_ar] weights via swivel: %s (.sp-model OK_Q8 arena)\n", sp_path);
    } else {
        m = qwen3_load(gguf);
    }
    sp_tokenizer *tok = sp_tokenizer_load(g);
    if (!m || !tok) { fprintf(stderr, "[ppl_ar] load FAIL\n"); return 1; }

    int cap = N + 16;
    int32_t *toks = (int32_t *)malloc((size_t)(clen + 16) * sizeof(int32_t));
    if (!toks) { fprintf(stderr, "[ppl_ar] OOM\n"); return 1; }
    long nt = sp_tokenizer_encode(tok, text, clen, 0, toks, (int)(clen + 16));
    if (nt < 4) { fprintf(stderr, "[ppl_ar] tokenize FAIL (nt=%ld)\n", nt); return 1; }
    if (nt > N) nt = N;                       /* cap to the requested window */
    (void)cap;

    double ppl = 0.0; long n_scored = 0;
    double t0 = now_s();
    int rc = qwen3_ppl_decode(m, toks, (int)nt, warm, &ppl, &n_scored);
    double dt = now_s() - t0;
    if (rc != 0) { fprintf(stderr, "[ppl_ar] qwen3_ppl_decode FAIL (rc=%d)\n", rc); return 1; }

    const char *rb = getenv("SP_RECALL_B"); const char *rw = getenv("SP_RECALL_W");
    const char *rr = getenv("SP_RECALL_R"); const char *r2 = getenv("SP_RING2");
    fprintf(stderr,
        "[ppl_ar] PPL=%.5f  n_tok=%ld n_scored=%ld warm=%d  B=%s W=%s R=%s RING2=%s  (%.1fs)\n",
        ppl, nt, n_scored, warm, rb ? rb : "off", rw ? rw : "-", rr ? rr : "-", r2 ? r2 : "-", dt);

    free(text); free(toks);
    sp_tokenizer_free(tok); qwen3_free(m); gguf_close(g);
    return 0;
}
