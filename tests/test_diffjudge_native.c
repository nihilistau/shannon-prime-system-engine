/* test_diffjudge_native.c — G-DIFFJUDGE-NATIVE (N4).
 *
 * Runs the tag-based memory-index judge on OUR OWN native DiffusionGemma forward
 * (diffusion_gemma_forward_cuda) with a CONSTRAINED-SELECTION head, and gates
 * SELECTION FIDELITY against the PR-24423 oracle (G-DIFFJUDGE-1: recall 95.6%,
 * reject 96.0% on _needle_corpus_div).
 *
 * THE JUDGE IS A SINGLE CONSTRAINED FORWARD (not iterative denoising):
 *   prompt = judge instruction + K tagged candidate texts + the question + answer cue
 *   canvas = dg_canvas_length fill tokens (the diffusion answer surface)
 *   ONE diffusion_gemma_forward_cuda([prompt|canvas]) -> logits[n_tok x V]
 *   at the FIRST canvas position (row P) apply a HARD logit mask: every vocab
 *   logit -> -INF except the K candidate TAG tokens + the NULL token; argmax over
 *   that constrained subspace -> the selected candidate (or NULL). Pure deterministic
 *   argmax, no temperature/top-k/top-p.
 *
 * Tags are chosen to be SINGLE vocab tokens (probed at startup): the harness builds a
 * pool of short tag strings, encodes each with parse_special=0, and keeps only the
 * ones that tokenize to EXACTLY ONE id (so the mask is a single-position argmax). NULL
 * likewise resolves to a single token (" NONE" or fallback). Each query draws K
 * distinct single-token tags and maps argmax-token -> candidate index.
 *
 * Window discipline mirrors diffjudge_recall_test.py EXACTLY (same seed/K so the
 * candidate windows match the oracle): each matched query is judged against K = true
 * needle + (K-1) random distractors, position-shuffled. Foreign queries get K random
 * needles (no truth) and must select NULL.
 *
 * The model is loaded ONCE and stays resident (arena mmap); the per-query forward
 * streams weights per-layer (~<1 min each). ~140 forwards => ~2-3 hours; run detached.
 *
 * Env:
 *   SP_DJ_SPMODEL / SP_DJ_SPTOK   .sp-model / .sp-tokenizer (defaults below)
 *   SP_DJ_CORPUS                  corpus dir (default _needle_corpus_div)
 *   SP_DJ_OUT                     output log (default tests/fixtures/.../G-DIFFJUDGE-NATIVE.log)
 *   SP_DJ_SEED                    window RNG seed (default 20260621, == oracle)
 *   SP_DJ_K                       candidate window size (default 12, == oracle)
 *   SP_DJ_LIMIT / SP_DJ_FLIMIT    cap matched / foreign counts (0 = all; for a smoke run)
 *   SP_DJ_CANVAS                  canvas length override (default = model dg_canvas_length)
 *
 * PASS (G-DIFFJUDGE-NATIVE): recall@1 >= 90% (oracle 95.6%) AND foreign-reject >= 90%
 *   (oracle 96.0%). Reports the actual numbers vs the oracle + sample picks.
 */
#define _CRT_SECURE_NO_WARNINGS
#include "sp/sp_model.h"
#include "sp/model.h"
#include "sp/sp_status.h"
#include "sp_engine/cuda_backend.h"
#include "sp_engine/tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifndef SP_DJ_SPMODEL_DEF
#define SP_DJ_SPMODEL_DEF "C:/sp_models/diffusiongemma-26B-A4B.sp-model"
#endif
#ifndef SP_DJ_SPTOK_DEF
#define SP_DJ_SPTOK_DEF "C:/sp_models/diffusiongemma-26B-A4B.sp-tokenizer"
#endif

/* ───────── deterministic RNG identical to Python random.Random for OUR use ─────────
 * We do NOT need to byte-match Python's Mersenne; we need a REPRODUCIBLE window per
 * needle. We use a simple splitmix64 seeded from SP_DJ_SEED; the window is the same on
 * every run so the gate is reproducible. (The oracle's exact distractor identities
 * differ from ours, but the TASK is identical: true needle + K-1 distractors, shuffled.
 * Selection fidelity is about whether OUR forward picks the truth out of a K-window;
 * the precise distractor set is immaterial to that.) */
static uint64_t g_rng = 0;
static uint64_t rng_next(void) {
    uint64_t z = (g_rng += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static int rng_below(int n) { return (n <= 0) ? 0 : (int)(rng_next() % (uint64_t)n); }

/* ───────── corpus ───────── */
typedef struct { char id[64]; char text[1024]; char query[1024]; } needle_t;

/* extract a JSON string value for "key": "..." (handles \" and \\). Returns 1 on hit. */
static int json_str(const char *line, const char *key, char *out, size_t cap) {
    char pat[80]; snprintf(pat, sizeof pat, "\"%s\"", key);
    const char *p = strstr(line, pat);
    if (!p) return 0;
    p += strlen(pat);
    while (*p && *p != ':') p++;
    if (*p != ':') return 0;
    p++;
    while (*p == ' ' || *p == '\t') p++;
    if (*p != '"') return 0;
    p++;
    size_t o = 0;
    while (*p && *p != '"') {
        if (*p == '\\' && p[1]) {
            p++;
            char c = *p;
            if (c == 'n') c = '\n'; else if (c == 't') c = '\t';
            else if (c == 'u') { /* skip \uXXXX -> '?' */ if (o + 1 < cap) out[o++] = '?'; p += 5; continue; }
            if (o + 1 < cap) out[o++] = c;
            p++;
        } else {
            if (o + 1 < cap) out[o++] = *p;
            p++;
        }
    }
    out[o] = '\0';
    return 1;
}

static int load_needles(const char *cdir, needle_t **out) {
    char path[1024]; snprintf(path, sizeof path, "%s/corpus_manifest.jsonl", cdir);
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return -1; }
    int cap = 128, n = 0;
    needle_t *v = (needle_t *)malloc((size_t)cap * sizeof(needle_t));
    char line[8192];
    while (fgets(line, sizeof line, f)) {
        char id[64];
        if (!json_str(line, "id", id, sizeof id)) continue;
        if (strncmp(id, "ctrl", 4) == 0) continue;       /* skip the parametric control */
        if (n == cap) { cap *= 2; v = (needle_t *)realloc(v, (size_t)cap * sizeof(needle_t)); }
        memset(&v[n], 0, sizeof v[n]);
        strncpy(v[n].id, id, sizeof v[n].id - 1);
        json_str(line, "text",  v[n].text,  sizeof v[n].text);
        json_str(line, "query", v[n].query, sizeof v[n].query);
        n++;
    }
    fclose(f);
    *out = v;
    return n;
}

static int load_foreign(const char *cdir, char ***out) {
    char path[1024]; snprintf(path, sizeof path, "%s/foreign_queries.txt", cdir);
    FILE *f = fopen(path, "rb");
    if (!f) { *out = NULL; return 0; }
    int cap = 64, n = 0;
    char **v = (char **)malloc((size_t)cap * sizeof(char *));
    char line[2048];
    while (fgets(line, sizeof line, f)) {
        size_t L = strlen(line);
        while (L && (line[L-1] == '\n' || line[L-1] == '\r' || line[L-1] == ' ')) line[--L] = '\0';
        if (!L) continue;
        if (n == cap) { cap *= 2; v = (char **)realloc(v, (size_t)cap * sizeof(char *)); }
        v[n++] = _strdup(line);
    }
    fclose(f);
    *out = v;
    return n;
}

/* ───────── single-token tag pool (probed against the live tokenizer) ───────── */
/* Candidate tag surfaces. We want short, visually-distinct strings that gemma4 maps to
 * exactly ONE token. Uppercase letters and 2-letter codes are the safest. We over-supply
 * and keep only the single-token ones at startup. The leading space matters: gemma4 SPM
 * tokenizes " A" as one piece; "A" mid-sentence too. We probe with a leading space since
 * the judge prints "[TAG] text" with a space before '['. */
static const char *TAG_CANDS[] = {
    "AA","BB","CC","DD","EE","FF","GG","HH","JJ","KK","LL","MM","NN","PP","QQ","RR","SS","TT","VV","WW","XX","YY","ZZ",
    "Q1","Q2","Q3","Q4","Q5","Q6","Q7","Q8","Q9","R1","R2","R3","R4","R5","R6","R7","R8","R9",
    "Z1","Z2","Z3","Z4","Z5","Z6","Z7","Z8","Z9","Z0",
    "A","B","C","D","E","F","G","H","J","K","L","M","N","P","Q","R","S","T","U","V","W","X","Y","Z"
};
#define N_TAG_CANDS ((int)(sizeof(TAG_CANDS)/sizeof(TAG_CANDS[0])))

/* encode a string; return the single token id if it tokenizes to exactly one id, else -1.
 * We test the surface AS IT APPEARS in the prompt: bracketed "[TAG]". The mask must use the
 * token the model would WRITE for the answer, which is the tag token in isolation (the model
 * answers with the bare tag). So we probe the BARE tag with a leading space (answer context),
 * AND require the bracketed form to round-trip to that same token. Simpler + robust: require
 * the tag, with a leading space, to be a single token; use that id for the mask. */
/* the gemma tokenizer auto-prepends BOS (id 2). A surface is a "single token" if, after
 * dropping a leading BOS, exactly one content id remains. We probe BOTH the bare surface
 * and a leading-space variant (SPM word-boundary); return that single content id, else -1. */
#define DJ_BOS 2
static int single_content_id(const int32_t *ids, long n) {
    long lo = 0; if (n >= 1 && ids[0] == DJ_BOS) lo = 1;
    if (n - lo == 1) return ids[lo];
    return -1;
}
static int probe_single_token(const sp_tokenizer *tk, const char *s) {
    int32_t ids[16];
    long n = sp_tokenizer_encode(tk, s, strlen(s), 0, ids, 16);
    int id = single_content_id(ids, n);
    if (id >= 0) return id;
    /* try with a leading space (SPM word-boundary) */
    char sp[66]; snprintf(sp, sizeof sp, " %s", s);
    n = sp_tokenizer_encode(tk, sp, strlen(sp), 0, ids, 16);
    return single_content_id(ids, n);
}
/* return BOTH single-token forms of a surface: bare (as it appears inside "[TAG]") and
 * space-prefixed (as the model would WRITE it after a space). Sets *bare/*spaced to the
 * content ids (or -1). A surface qualifies for the tag pool iff at least the BARE form is a
 * single content token (so it round-trips inside the bracket); the spaced form is added to
 * the allow-mask when it too is single-token. Returns 1 if the bare form is single-token. */
static int probe_tag_forms(const sp_tokenizer *tk, const char *s, int *bare, int *spaced) {
    int32_t ids[16];
    long n = sp_tokenizer_encode(tk, s, strlen(s), 0, ids, 16);
    *bare = single_content_id(ids, n);
    char sp[66]; snprintf(sp, sizeof sp, " %s", s);
    n = sp_tokenizer_encode(tk, sp, strlen(sp), 0, ids, 16);
    *spaced = single_content_id(ids, n);
    return (*bare >= 0);
}

int main(void) {
    const char *spm = getenv("SP_DJ_SPMODEL"); if (!spm || !*spm) spm = SP_DJ_SPMODEL_DEF;
    const char *spt = getenv("SP_DJ_SPTOK");   if (!spt || !*spt) spt = SP_DJ_SPTOK_DEF;
    const char *cdir = getenv("SP_DJ_CORPUS"); if (!cdir || !*cdir) cdir = "_needle_corpus_div";
    const char *outp = getenv("SP_DJ_OUT");
    char outbuf[1024];
    if (!outp || !*outp) { snprintf(outbuf, sizeof outbuf, "tests/fixtures/chat_fullstack/G-DIFFJUDGE-NATIVE.log"); outp = outbuf; }
    uint64_t seed = 20260621; { const char *e = getenv("SP_DJ_SEED"); if (e && *e) seed = strtoull(e, NULL, 10); }
    int K = 12; { const char *e = getenv("SP_DJ_K"); if (e && *e) { int v = atoi(e); if (v > 1) K = v; } }
    int LIM = 0; { const char *e = getenv("SP_DJ_LIMIT"); if (e && *e) LIM = atoi(e); }
    int FLIM = 0; { const char *e = getenv("SP_DJ_FLIMIT"); if (e && *e) FLIM = atoi(e); }
    g_rng = seed ? seed : 1;

    FILE *log = fopen(outp, "w");
    if (!log) { fprintf(stderr, "cannot open out log %s\n", outp); return 2; }
    #define EMIT(...) do { fprintf(stdout, __VA_ARGS__); fprintf(log, __VA_ARGS__); fflush(log); fflush(stdout); } while (0)

    EMIT("# G-DIFFJUDGE-NATIVE  corpus=%s seed=%llu K=%d\n", cdir, (unsigned long long)seed, K);
    EMIT("# model=%s\n", spm);
    EMIT("# NATIVE forward = diffusion_gemma_forward_cuda + constrained-argmax head\n");

    FILE *probe = fopen(spm, "rb");
    if (!probe) { EMIT("# model absent (%s) — SKIP (exit 0)\n", spm); fclose(log); return 0; }
    fclose(probe);

    /* load corpus */
    needle_t *needles = NULL; int NN = load_needles(cdir, &needles);
    if (NN <= 0) { EMIT("FATAL: no needles loaded from %s\n", cdir); fclose(log); return 2; }
    char **foreign = NULL; int NF = load_foreign(cdir, &foreign);
    EMIT("# needles=%d foreign=%d\n", NN, NF);

    /* load tokenizer */
    sp_tokenizer *tk = sp_tokenizer_load_tokfile(spt);
    if (!tk) { EMIT("FATAL: sp_tokenizer_load_tokfile(%s) failed\n", spt); fclose(log); return 2; }
    uint32_t VOCAB = sp_tokenizer_vocab_size(tk);

    /* build the single-token tag pool. Each tag carries TWO mask ids: the BARE form (the
     * token inside "[TAG]") and the SPACE-prefixed form (how the model writes it after a
     * space). The argmax allows either; both map back to the same candidate slot. */
    int *tagtok  = (int *)malloc((size_t)N_TAG_CANDS * sizeof(int));   /* bare id (also identity) */
    int *tagtok2 = (int *)malloc((size_t)N_TAG_CANDS * sizeof(int));   /* spaced id (-1 if none) */
    const char **tagstr = (const char **)malloc((size_t)N_TAG_CANDS * sizeof(char *));
    int n_tags = 0;
    for (int i = 0; i < N_TAG_CANDS; i++) {
        int bare = -1, spaced = -1;
        if (!probe_tag_forms(tk, TAG_CANDS[i], &bare, &spaced)) continue;
        /* dedup on the bare id AND the spaced id (avoid two tags colliding on any mask id) */
        int dup = 0;
        for (int j = 0; j < n_tags; j++)
            if (tagtok[j] == bare || (spaced >= 0 && tagtok2[j] == spaced) ||
                (spaced >= 0 && tagtok[j] == spaced) || tagtok2[j] == bare) { dup = 1; break; }
        if (dup) continue;
        tagtok[n_tags] = bare; tagtok2[n_tags] = spaced; tagstr[n_tags] = TAG_CANDS[i]; n_tags++;
    }
    EMIT("# single-token tags available=%d (need K=%d)\n", n_tags, K);
    if (n_tags < K) { EMIT("FATAL: not enough single-token tags (%d < %d)\n", n_tags, K); fclose(log); return 2; }

    /* NULL token: the cue ends "(or NONE):" so NULL surface = "NONE". Carry bare + spaced. */
    int null_tok = -1, null_tok2 = -1;
    const char *null_str = "NONE";
    probe_tag_forms(tk, "NONE", &null_tok, &null_tok2);
    if (null_tok < 0) { probe_tag_forms(tk, "None", &null_tok, &null_tok2); null_str = "None"; }
    if (null_tok < 0) { probe_tag_forms(tk, "NULL", &null_tok, &null_tok2); null_str = "NULL"; }
    EMIT("# NULL surface='%s' bare=%d spaced=%d  vocab=%u\n", null_str, null_tok, null_tok2, VOCAB);
    if (null_tok < 0) { EMIT("FATAL: no single-token NULL surface\n"); fclose(log); return 2; }

    /* load model */
    sp_model *handle = NULL;
    sp_status st = sp_model_load(spm, spt, &handle);
    if (st != SP_OK || !handle) { EMIT("FATAL: sp_model_load: %s\n", sp_last_error()); fclose(log); return 2; }
    qwen3_model *m = sp_model_to_diffusion_gemma(handle);
    if (!m) { EMIT("FATAL: sp_model_to_diffusion_gemma: %s\n", sp_last_error()); sp_model_unload(handle); fclose(log); return 2; }
    const qwen3_config *c = &m->cfg;
    const int V = (int)c->n_vocab;
    int CL = (int)c->dg_canvas_length;
    { const char *e = getenv("SP_DJ_CANVAS"); if (e && *e) { int v = atoi(e); if (v > 0 && v <= CL) CL = v; } }
    EMIT("# loaded: V=%d dg_canvas_length=%d (using CL=%d)\n", V, (int)c->dg_canvas_length, CL);

    /* ── PRE-WARM the model into the OS file cache (mitigation for the cold-mmap
     * demand-fault stall that hangs diffusion_gemma_forward_cuda mid-traversal). The
     * forward streams weights through CreateFileMapping/MapViewOfFile + per-row dequant;
     * a cold mapping faults pages on demand and (on this driver) stalls. Reading the
     * whole .sp-model sequentially once (NVMe ~3 GB/s ~5s for 14 GB) populates the OS
     * standby cache so the forward's subsequent faults are cheap. Default-on; set
     * SP_DJ_NOPREWARM=1 to skip. ── */
    if (!getenv("SP_DJ_NOPREWARM")) {
        FILE *wf = fopen(spm, "rb");
        if (wf) {
            EMIT("# pre-warming model into OS cache (sequential read)...\n");
            clock_t tw = clock();
            char *wbuf = (char *)malloc(64u << 20);   /* 64 MB */
            unsigned long long total = 0; size_t r;
            if (wbuf) {
                while ((r = fread(wbuf, 1, 64u << 20, wf)) > 0) total += r;
                free(wbuf);
            }
            fclose(wf);
            EMIT("# pre-warm read %.1f GB in %.1fs\n", total / 1e9,
                 (double)(clock() - tw) / CLOCKS_PER_SEC);
        }
    }

    /* judge prompt template (mirrors the oracle JUDGE_TMPL, tag-tight) */
    /* instruction + " " + entries + " QUESTION: <q> Tag of the answer (or NONE):" */
    static const char *INSTR =
        "You are a memory index. Each entry below has a TAG in brackets. "
        "Read the QUESTION and reply with ONLY the tag of the single entry that "
        "directly answers it. If no entry answers it, reply NONE.";

    /* scratch buffers (sized for the longest prompt + canvas) */
    int max_tok = 16384;
    int32_t *toks = (int32_t *)malloc((size_t)max_tok * sizeof(int32_t));
    char *prompt = (char *)malloc(64 * 1024);
    float *logits = NULL;  /* allocated per forward (n_tok * V) */

    int hit = 0, tot = 0, frej = 0, ftot = 0;
    int n_samples_printed = 0;
    clock_t t_start = clock();

    /* one trial: build window+tags, tokenize prompt, run forward, constrained argmax.
     * returns the selected candidate index in [0,K) or K (==NULL), or -1 on error.
     * truth_idx (or -1 for foreign) is only for logging. */
    int total_trials = (LIM > 0 ? (NN < LIM ? NN : LIM) : NN) + (FLIM > 0 ? (NF < FLIM ? NF : FLIM) : NF);
    int trial_no = 0;

    /* a reusable selector */
    /* candidate set built per trial */
    int *cand = (int *)malloc((size_t)K * sizeof(int));     /* needle indices, -1 for NULL-only filler (none) */
    int *picktag = (int *)malloc((size_t)K * sizeof(int));  /* tag token per candidate slot */

    int n_matched = (LIM > 0 ? (NN < LIM ? NN : LIM) : NN);
    int n_for     = (FLIM > 0 ? (NF < FLIM ? NF : FLIM) : NF);

    for (int phase = 0; phase < 2; phase++) {
        int count = (phase == 0) ? n_matched : n_for;
        for (int qi = 0; qi < count; qi++) {
            trial_no++;
            /* ---- build candidate window ---- */
            int truth_idx = -1;
            const char *query = NULL;
            if (phase == 0) {
                /* matched: true needle = needles[qi] + K-1 distractors, shuffled */
                /* pick K-1 distinct distractor indices != qi */
                int chosen[256]; int nc = 0;
                int guard = 0;
                while (nc < K - 1 && guard < 100000) {
                    guard++;
                    int d = rng_below(NN);
                    if (d == qi) continue;
                    int dup = 0; for (int j = 0; j < nc; j++) if (chosen[j] == d) { dup = 1; break; }
                    if (dup) continue;
                    chosen[nc++] = d;
                }
                /* assemble K slots: distractors + truth, then shuffle slot order */
                int slotneedle[256];
                for (int j = 0; j < nc; j++) slotneedle[j] = chosen[j];
                slotneedle[nc] = qi;             /* truth */
                int kk = nc + 1;
                /* Fisher-Yates shuffle */
                for (int a = kk - 1; a > 0; a--) { int b = rng_below(a + 1); int t = slotneedle[a]; slotneedle[a] = slotneedle[b]; slotneedle[b] = t; }
                for (int j = 0; j < kk; j++) { cand[j] = slotneedle[j]; if (slotneedle[j] == qi) truth_idx = j; }
                for (int j = kk; j < K; j++) cand[j] = -1;
                query = needles[qi].query;
            } else {
                /* foreign: K random needles, no truth */
                int chosen[256]; int nc = 0; int guard = 0;
                while (nc < K && guard < 100000) {
                    guard++;
                    int d = rng_below(NN);
                    int dup = 0; for (int j = 0; j < nc; j++) if (chosen[j] == d) { dup = 1; break; }
                    if (dup) continue;
                    chosen[nc++] = d;
                }
                for (int j = 0; j < nc; j++) cand[j] = chosen[j];
                for (int j = nc; j < K; j++) cand[j] = -1;
                truth_idx = -1;
                query = foreign[qi];
            }

            /* ---- assign a distinct single-token tag per real candidate slot ---- */
            /* draw K distinct tags from the pool by index-shuffle */
            int tagperm[512];
            for (int j = 0; j < n_tags; j++) tagperm[j] = j;
            for (int a = n_tags - 1; a > 0; a--) { int b = rng_below(a + 1); int t = tagperm[a]; tagperm[a] = tagperm[b]; tagperm[b] = t; }
            int realK = 0; for (int j = 0; j < K; j++) if (cand[j] >= 0) realK++;
            for (int j = 0; j < K; j++) picktag[j] = (cand[j] >= 0) ? tagperm[j % n_tags] : -1;

            /* ---- build the prompt text ---- */
            char *p = prompt; size_t rem = 64 * 1024;
            int w = snprintf(p, rem, "%s ", INSTR); p += w; rem -= (size_t)w;
            for (int j = 0; j < K; j++) {
                if (cand[j] < 0) continue;
                w = snprintf(p, rem, "[%s] %s ", tagstr[picktag[j]], needles[cand[j]].text);
                p += w; rem -= (size_t)w;
            }
            w = snprintf(p, rem, "QUESTION: %s Tag of the answer (or %s):", query, null_str);
            p += w; rem -= (size_t)w;

            /* ---- tokenize prompt (parse_special on so any control surfaces map cleanly;
             * BOS auto-prepended by the gemma tokenizer) ---- */
            long np = sp_tokenizer_encode(tk, prompt, strlen(prompt), 1, toks, max_tok);
            if (np <= 0) { EMIT("[err] tokenize failed q=%d\n", trial_no); continue; }
            if (np > max_tok - CL - 4) {
                /* shouldn't happen with K=12; guard */
                EMIT("[err] prompt too long (%ld toks) q=%d\n", np, trial_no); continue;
            }
            int P = (int)np;                 /* prompt length; canvas starts at row P */
            int n_tok = P + CL;
            /* canvas fill = a benign in-vocab token (BOS=2 if present else 0) */
            int32_t fill = 2; if (fill >= V) fill = 0;
            for (int i = 0; i < CL; i++) toks[P + i] = fill;

            /* ---- run the native forward ---- */
            if (logits) free(logits);
            EMIT("# trial %d: P=%d n_tok=%d logits=%.0fMB forward...\n",
                 trial_no, P, n_tok, (double)n_tok * V * 4.0 / 1e6);
            logits = (float *)malloc((size_t)n_tok * V * sizeof(float));
            if (!logits) { EMIT("[err] logits OOM (n_tok=%d) q=%d\n", n_tok, trial_no); continue; }
            clock_t tq = clock();
            int frc = diffusion_gemma_forward_cuda(m, toks, n_tok, logits);
            double secs = (double)(clock() - tq) / CLOCKS_PER_SEC;
            if (frc != 0) { EMIT("[err] forward rc=%d (%s) q=%d\n", frc, sp_last_error(), trial_no); continue; }

            /* ---- CONSTRAINED ARGMAX at the first canvas position (row P) ---- */
            /* allowed set = for each real candidate, {bare tag id, spaced tag id}; plus
             * {NONE bare, NONE spaced}. Everything else is -INF (we just scan the allowed
             * ids). Each candidate's score = max over its two forms (the model may write
             * either the bracket-bare token or the space-led token). */
            const float *row = logits + (size_t)P * V;
            #define DJ_L(id) (((id) >= 0 && (id) < V) ? row[(id)] : -3.0e38f)
            int best_slot = -2; float best_val = -3.0e38f;
            for (int j = 0; j < K; j++) {
                if (cand[j] < 0) continue;
                int t = picktag[j];
                float lv = DJ_L(tagtok[t]);
                float lv2 = DJ_L(tagtok2[t]);
                if (lv2 > lv) lv = lv2;
                if (lv > best_val) { best_val = lv; best_slot = j; }
            }
            float null_val = DJ_L(null_tok);
            { float nv2 = DJ_L(null_tok2); if (nv2 > null_val) null_val = nv2; }
            #undef DJ_L
            int pick;        /* candidate slot index, or K for NULL */
            if (null_val > best_val) { pick = K; }   /* NULL wins */
            else pick = best_slot;

            /* ---- score ---- */
            if (phase == 0) {
                int ok = (pick == truth_idx);
                tot++; hit += ok ? 1 : 0;
                EMIT("[match] %-6s %-12s K=%d gt=%2d pick=%2d nullL=%.3f bestL=%.3f %.0fs :: %.46s\n",
                     ok ? "OK" : "MISS", needles[qi].id, realK, truth_idx,
                     (pick == K ? -1 : pick), null_val, best_val, secs, query);
                if (!ok && n_samples_printed < 12) {
                    /* sample diagnosis: what we picked vs truth */
                    int picked_needle = (pick == K) ? -2 : (pick >= 0 ? cand[pick] : -3);
                    EMIT("        sample: truth='%s' picked=%s\n",
                         needles[qi].text,
                         pick == K ? "NULL" : (picked_needle >= 0 ? needles[picked_needle].text : "?"));
                    n_samples_printed++;
                }
            } else {
                int ok = (pick == K);     /* foreign must pick NULL */
                ftot++; frej += ok ? 1 : 0;
                EMIT("[forgn] %-6s K=%d pick=%2d nullL=%.3f bestL=%.3f %.0fs :: %.46s\n",
                     ok ? "REJECT" : "FALSE", realK, (pick == K ? -1 : pick), null_val, best_val, secs, query);
            }
            /* progress heartbeat */
            double elapsed = (double)(clock() - t_start) / CLOCKS_PER_SEC;
            EMIT("# progress %d/%d  elapsed=%.0fs\n", trial_no, total_trials, elapsed);
        }
    }

    double rec = tot ? (double)hit / tot : 0.0;
    double rej = ftot ? (double)frej / ftot : 0.0;
    int g_rec = (rec >= 0.90), g_rej = (rej >= 0.90);
    EMIT("\n================ G-DIFFJUDGE-NATIVE RESULT ================\n");
    EMIT("K=%d seed=%llu   forwards=%d\n", K, (unsigned long long)seed, trial_no);
    EMIT("recall@1       : %d/%d = %.1f%%   (oracle PR-24423 = 95.6%%)\n", hit, tot, 100.0 * rec);
    EMIT("foreign-reject : %d/%d = %.1f%%   (oracle PR-24423 = 96.0%%)\n", frej, ftot, 100.0 * rej);
    EMIT("GATE recall>=90%% : %s    GATE reject>=90%% : %s\n", g_rec ? "GREEN" : "RED", g_rej ? "GREEN" : "RED");
    EMIT("G-DIFFJUDGE-NATIVE : %s\n", (g_rec && g_rej) ? "GREEN — native engine reproduces the oracle judge" : "RED — see per-query diagnosis");

    if (logits) free(logits);
    free(toks); free(prompt); free(cand); free(picktag);
    free(tagtok); free(tagtok2); free((void *)tagstr);
    sp_tokenizer_free(tk);
    sp_model_unload(handle);
    for (int i = 0; i < NF; i++) free(foreign[i]);
    free(foreign); free(needles);
    fclose(log);
    return (g_rec && g_rej) ? 0 : 1;
}
