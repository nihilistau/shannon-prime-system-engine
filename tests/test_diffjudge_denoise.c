/* test_diffjudge_denoise.c — G-DIFFJUDGE-NATIVE-full (N4-full).
 *
 * The constrained {tags,NULL} DENOISE judge on OUR native DiffusionGemma forward.
 * Replaces test_diffjudge_native's SINGLE forward+argmax with the full entropy-bound
 * denoise loop (DESIGN-diffgemma-sampler.md §N4-full): a multi-step renoise loop that
 * refines the canvas, with the answer canvas position HARD-MASKED to {tag tokens, NULL}
 * each step before dg_sample_kernel. The leading hypothesis is that the iterative
 * refinement cures the single-forward Marlock-style miss (single-forward ~14% recall;
 * oracle 95.6%).
 *
 * Each step:
 *   inv_temp = 1 / lerp(t_max=0.8, t_min=0.4, step/S)
 *   logits = forward([prompt|canvas], self_cond = prev canvas logits if step>0)
 *   MASK the answer canvas row (row P) to {tag ids, NULL ids} (-INF elsewhere)
 *   (argmax,entropy,denoiser) = dg_sample_kernel(logits, u=seeded, inv_temp)
 *   order canvas positions by ascending entropy; accept the lowest-entropy prefix while
 *     cumE <= entropy_bound (0.1); canvas[pos] = accept ? denoiser : fresh_random
 *   output canvas[pos] = argmax[pos]
 *   adaptive stop: argmax stable >= stability_threshold (2) AND mean(entropy) <
 *     confidence_threshold (0.005)
 * The reported pick = the constrained argmax at the answer canvas row (row P) of the
 * final/stable output canvas.
 *
 * TRACTABILITY-SCOPED gate (per the task): run a SUBSET — the first SP_DJ_LIMIT matched
 * div queries + SP_DJ_FLIMIT foreign (default 14 + 5) through the full denoise judge.
 * Adaptive stop keeps it to a few steps each. Per-step forward ~71s -> minutes/query ->
 * a few hours for the subset. Run DETACHED; the log auto-completes. Target: subset
 * recall jumps well above the single-forward 14% toward the oracle 95.6%.
 *
 * Env (shared with test_diffjudge_native + the denoise knobs):
 *   SP_DJ_SPMODEL / SP_DJ_SPTOK / SP_DJ_CORPUS / SP_DJ_OUT / SP_DJ_SEED / SP_DJ_K
 *   SP_DJ_LIMIT (default 14) / SP_DJ_FLIMIT (default 5)   subset caps
 *   SP_DJ_CANVAS  canvas length (default min(model, 16) — small keeps each forward fast)
 *   SP_DJ_STEPS   max denoise steps (default 12; adaptive stop usually halts earlier)
 *   SP_DJ_NOPREWARM  skip the OS-cache pre-warm read
 *   SP_DJ_NOSC    disable self-conditioning (debug; default SC ON for step>0)
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
#include <stdint.h>

/* ── N6 host-AV trap: symbolize the faulting instruction + stack via dbghelp (uses the .pdb).
 * Catches the walking PageHeap access-violation and prints the exact cuda_forward.cu line. ── */
#ifdef _WIN32
#include <windows.h>
#include <dbghelp.h>
#pragma comment(lib, "dbghelp.lib")
static LONG WINAPI sp_av_filter(EXCEPTION_POINTERS *ep) {
    DWORD code = ep->ExceptionRecord->ExceptionCode;
    fprintf(stderr, "\n=== SP-AV-FILTER exception 0x%08lx ===\n", (unsigned long)code);
    if (code == EXCEPTION_ACCESS_VIOLATION || code == 0xC0000005) {
        ULONG_PTR rw = ep->ExceptionRecord->ExceptionInformation[0];
        ULONG_PTR ad = ep->ExceptionRecord->ExceptionInformation[1];
        fprintf(stderr, "ACCESS VIOLATION on %s at address 0x%p\n",
                rw==1?"WRITE":(rw==0?"READ":"EXEC"), (void*)ad);
    }
    HANDLE proc = GetCurrentProcess();
    SymSetOptions(SYMOPT_LOAD_LINES | SYMOPT_DEFERRED_LOADS | SYMOPT_UNDNAME);
    SymInitialize(proc, NULL, TRUE);
    char sbuf[sizeof(SYMBOL_INFO) + 512];
    SYMBOL_INFO *si = (SYMBOL_INFO*)sbuf; si->SizeOfStruct = sizeof(SYMBOL_INFO); si->MaxNameLen = 500;
    IMAGEHLP_LINE64 ln; ln.SizeOfStruct = sizeof(IMAGEHLP_LINE64); DWORD col; DWORD64 disp;
    CONTEXT ctx = *ep->ContextRecord;
    STACKFRAME64 sf; memset(&sf, 0, sizeof sf);
    sf.AddrPC.Offset = ctx.Rip;   sf.AddrPC.Mode = AddrModeFlat;
    sf.AddrFrame.Offset = ctx.Rbp; sf.AddrFrame.Mode = AddrModeFlat;
    sf.AddrStack.Offset = ctx.Rsp; sf.AddrStack.Mode = AddrModeFlat;
    fprintf(stderr, "--- fault call stack (top = faulting instruction) ---\n");
    for (int i = 0; i < 40; i++) {
        if (!StackWalk64(IMAGE_FILE_MACHINE_AMD64, proc, GetCurrentThread(), &sf, &ctx,
                         NULL, SymFunctionTableAccess64, SymGetModuleBase64, NULL)) break;
        DWORD64 a = sf.AddrPC.Offset; if (!a) break;
        if (SymFromAddr(proc, a, &disp, si)) {
            if (SymGetLineFromAddr64(proc, a, &col, &ln))
                fprintf(stderr, "  #%02d %s  (%s:%lu)\n", i, si->Name, ln.FileName, (unsigned long)ln.LineNumber);
            else
                fprintf(stderr, "  #%02d %s +0x%llx\n", i, si->Name, (unsigned long long)disp);
        } else {
            fprintf(stderr, "  #%02d 0x%llx\n", i, (unsigned long long)a);
        }
    }
    fflush(stderr);
    return EXCEPTION_EXECUTE_HANDLER;
}
#endif

#ifndef SP_DJ_SPMODEL_DEF
#define SP_DJ_SPMODEL_DEF "C:/sp_models/diffusiongemma-26B-A4B.sp-model"
#endif
#ifndef SP_DJ_SPTOK_DEF
#define SP_DJ_SPTOK_DEF "C:/sp_models/diffusiongemma-26B-A4B.sp-tokenizer"
#endif

/* ───────── deterministic RNG (splitmix64) ───────── */
static uint64_t g_rng = 0;
static uint64_t rng_next(void) {
    uint64_t z = (g_rng += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static int   rng_below(int n) { return (n <= 0) ? 0 : (int)(rng_next() % (uint64_t)n); }
static float rng_uniform(void) { return (float)((rng_next() >> 11) * (1.0 / 9007199254740992.0)); }

/* ───────── corpus ───────── */
typedef struct { char id[64]; char text[1024]; char query[1024]; } needle_t;

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
            else if (c == 'u') { if (o + 1 < cap) out[o++] = '?'; p += 5; continue; }
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
        if (strncmp(id, "ctrl", 4) == 0) continue;
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

/* ───────── single-token tag pool ───────── */
static const char *TAG_CANDS[] = {
    "AA","BB","CC","DD","EE","FF","GG","HH","JJ","KK","LL","MM","NN","PP","QQ","RR","SS","TT","VV","WW","XX","YY","ZZ",
    "Q1","Q2","Q3","Q4","Q5","Q6","Q7","Q8","Q9","R1","R2","R3","R4","R5","R6","R7","R8","R9",
    "Z1","Z2","Z3","Z4","Z5","Z6","Z7","Z8","Z9","Z0",
    "A","B","C","D","E","F","G","H","J","K","L","M","N","P","Q","R","S","T","U","V","W","X","Y","Z"
};
#define N_TAG_CANDS ((int)(sizeof(TAG_CANDS)/sizeof(TAG_CANDS[0])))
#define DJ_BOS 2
static int single_content_id(const int32_t *ids, long n) {
    long lo = 0; if (n >= 1 && ids[0] == DJ_BOS) lo = 1;
    if (n - lo == 1) return ids[lo];
    return -1;
}
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
#ifdef _WIN32
    if (getenv("SP_AV_FILTER")) SetUnhandledExceptionFilter(sp_av_filter);  /* N6: off by default so cdb gets 2nd-chance */
#endif
    const char *spm = getenv("SP_DJ_SPMODEL"); if (!spm || !*spm) spm = SP_DJ_SPMODEL_DEF;
    const char *spt = getenv("SP_DJ_SPTOK");   if (!spt || !*spt) spt = SP_DJ_SPTOK_DEF;
    const char *cdir = getenv("SP_DJ_CORPUS"); if (!cdir || !*cdir) cdir = "_needle_corpus_div";
    const char *outp = getenv("SP_DJ_OUT");
    char outbuf[1024];
    if (!outp || !*outp) { snprintf(outbuf, sizeof outbuf, "tests/fixtures/chat_fullstack/G-DIFFJUDGE-NATIVE-full.log"); outp = outbuf; }
    uint64_t seed = 20260621; { const char *e = getenv("SP_DJ_SEED"); if (e && *e) seed = strtoull(e, NULL, 10); }
    int K = 12; { const char *e = getenv("SP_DJ_K"); if (e && *e) { int v = atoi(e); if (v > 1) K = v; } }
    int LIM = 14; { const char *e = getenv("SP_DJ_LIMIT");  if (e && *e) LIM = atoi(e); }
    int FLIM = 5; { const char *e = getenv("SP_DJ_FLIMIT"); if (e && *e) FLIM = atoi(e); }
    int STEPS = 12; { const char *e = getenv("SP_DJ_STEPS"); if (e && *e) { int v = atoi(e); if (v > 0) STEPS = v; } }
    int use_sc = getenv("SP_DJ_NOSC") ? 0 : 1;
    g_rng = seed ? seed : 1;

    /* denoise hyperparams (the GGUF omits eb_*; use the DESIGN defaults) */
    const float T_MAX = 0.8f, T_MIN = 0.4f;
    const float ENTROPY_BOUND = 0.1f;
    const int   STABILITY_THRESHOLD = 2;
    const float CONFIDENCE_THRESHOLD = 0.005f;

    FILE *log = fopen(outp, "w");
    if (!log) { fprintf(stderr, "cannot open out log %s\n", outp); return 2; }
    #define EMIT(...) do { fprintf(stdout, __VA_ARGS__); fprintf(log, __VA_ARGS__); fflush(log); fflush(stdout); } while (0)

    EMIT("# G-DIFFJUDGE-NATIVE-full (DENOISE judge)  corpus=%s seed=%llu K=%d\n", cdir, (unsigned long long)seed, K);
    EMIT("# model=%s\n", spm);
    EMIT("# denoise: STEPS<=%d t_max=%.2f t_min=%.2f entropy_bound=%.3f stab=%d conf=%.4f SC=%d\n",
         STEPS, T_MAX, T_MIN, ENTROPY_BOUND, STABILITY_THRESHOLD, CONFIDENCE_THRESHOLD, use_sc);

    FILE *probe = fopen(spm, "rb");
    if (!probe) { EMIT("# model absent (%s) — SKIP (exit 0)\n", spm); fclose(log); return 0; }
    fclose(probe);

    needle_t *needles = NULL; int NN = load_needles(cdir, &needles);
    if (NN <= 0) { EMIT("FATAL: no needles loaded from %s\n", cdir); fclose(log); return 2; }
    char **foreign = NULL; int NF = load_foreign(cdir, &foreign);
    EMIT("# needles=%d foreign=%d\n", NN, NF);

    sp_tokenizer *tk = sp_tokenizer_load_tokfile(spt);
    if (!tk) { EMIT("FATAL: sp_tokenizer_load_tokfile(%s) failed\n", spt); fclose(log); return 2; }
    uint32_t VOCAB = sp_tokenizer_vocab_size(tk);

    int *tagtok  = (int *)malloc((size_t)N_TAG_CANDS * sizeof(int));
    int *tagtok2 = (int *)malloc((size_t)N_TAG_CANDS * sizeof(int));
    const char **tagstr = (const char **)malloc((size_t)N_TAG_CANDS * sizeof(char *));
    int n_tags = 0;
    for (int i = 0; i < N_TAG_CANDS; i++) {
        int bare = -1, spaced = -1;
        if (!probe_tag_forms(tk, TAG_CANDS[i], &bare, &spaced)) continue;
        int dup = 0;
        for (int j = 0; j < n_tags; j++)
            if (tagtok[j] == bare || (spaced >= 0 && tagtok2[j] == spaced) ||
                (spaced >= 0 && tagtok[j] == spaced) || tagtok2[j] == bare) { dup = 1; break; }
        if (dup) continue;
        tagtok[n_tags] = bare; tagtok2[n_tags] = spaced; tagstr[n_tags] = TAG_CANDS[i]; n_tags++;
    }
    EMIT("# single-token tags available=%d (need K=%d)\n", n_tags, K);
    if (n_tags < K) { EMIT("FATAL: not enough single-token tags (%d < %d)\n", n_tags, K); fclose(log); return 2; }

    int null_tok = -1, null_tok2 = -1;
    const char *null_str = "NONE";
    probe_tag_forms(tk, "NONE", &null_tok, &null_tok2);
    if (null_tok < 0) { probe_tag_forms(tk, "None", &null_tok, &null_tok2); null_str = "None"; }
    if (null_tok < 0) { probe_tag_forms(tk, "NULL", &null_tok, &null_tok2); null_str = "NULL"; }
    EMIT("# NULL surface='%s' bare=%d spaced=%d  vocab=%u\n", null_str, null_tok, null_tok2, VOCAB);
    if (null_tok < 0) { EMIT("FATAL: no single-token NULL surface\n"); fclose(log); return 2; }

    sp_model *handle = NULL;
    sp_status st = sp_model_load(spm, spt, &handle);
    if (st != SP_OK || !handle) { EMIT("FATAL: sp_model_load: %s\n", sp_last_error()); fclose(log); return 2; }
    qwen3_model *m = sp_model_to_diffusion_gemma(handle);
    if (!m) { EMIT("FATAL: sp_model_to_diffusion_gemma: %s\n", sp_last_error()); sp_model_unload(handle); fclose(log); return 2; }
    const qwen3_config *c = &m->cfg;
    const int V = (int)c->n_vocab;
    int CL = (int)c->dg_canvas_length;
    int want_cl = 16; { const char *e = getenv("SP_DJ_CANVAS"); if (e && *e) { int v = atoi(e); if (v > 0) want_cl = v; } }
    if (want_cl < CL) CL = want_cl;        /* shorter canvas keeps each forward fast */
    if (CL < 1) CL = 1;
    EMIT("# loaded: V=%d dg_canvas_length=%d (using CL=%d)\n", V, (int)c->dg_canvas_length, CL);

    /* pre-warm the OS file cache (mitigate the cold-mmap demand-fault stall) */
    if (!getenv("SP_DJ_NOPREWARM")) {
        FILE *wf = fopen(spm, "rb");
        if (wf) {
            EMIT("# pre-warming model into OS cache...\n");
            clock_t tw = clock();
            char *wbuf = (char *)malloc(64u << 20);
            unsigned long long total = 0; size_t r;
            if (wbuf) { while ((r = fread(wbuf, 1, 64u << 20, wf)) > 0) total += r; free(wbuf); }
            fclose(wf);
            EMIT("# pre-warm read %.1f GB in %.1fs\n", total / 1e9, (double)(clock() - tw) / CLOCKS_PER_SEC);
        }
    }

    static const char *INSTR =
        "You are a memory index. Each entry below has a TAG in brackets. "
        "Read the QUESTION and reply with ONLY the tag of the single entry that "
        "directly answers it. If no entry answers it, reply NONE.";

    int max_tok = 16384;
    int32_t *toks = (int32_t *)malloc((size_t)max_tok * sizeof(int32_t));
    char *prompt = (char *)malloc(64 * 1024);

    /* per-canvas scratch */
    int   *am  = (int *)  malloc((size_t)CL * sizeof(int));
    float *ent = (float *)malloc((size_t)CL * sizeof(float));
    int   *sm  = (int *)  malloc((size_t)CL * sizeof(int));
    float *uvec = (float *)malloc((size_t)CL * sizeof(float));
    int   *canvas = (int *)malloc((size_t)CL * sizeof(int));
    int   *outc   = (int *)malloc((size_t)CL * sizeof(int));
    int   *order  = (int *)malloc((size_t)CL * sizeof(int));
    int   *cand = (int *)malloc((size_t)K * sizeof(int));
    int   *picktag = (int *)malloc((size_t)K * sizeof(int));

    int n_matched = (LIM > 0 ? (NN < LIM ? NN : LIM) : NN);
    int n_for     = (FLIM > 0 ? (NF < FLIM ? NF : FLIM) : NF);
    int total_trials = n_matched + n_for;
    int trial_no = 0;

    int hit = 0, tot = 0, frej = 0, ftot = 0;
    clock_t t_start = clock();

    /* persistent device prev-logits buffer [CL x V] for self-conditioning */
    void *prev_dev = NULL;
    if (use_sc) { prev_dev = dg_dev_alloc_f32((long)(int)c->dg_canvas_length * V); if (!prev_dev) { EMIT("# SC alloc failed -> SC OFF\n"); use_sc = 0; } }

    for (int phase = 0; phase < 2; phase++) {
        int count = (phase == 0) ? n_matched : n_for;
        for (int qi = 0; qi < count; qi++) {
            trial_no++;
            int truth_idx = -1;
            const char *query = NULL;
            if (phase == 0) {
                int chosen[256]; int nc = 0; int guard = 0;
                while (nc < K - 1 && guard < 100000) {
                    guard++; int d = rng_below(NN);
                    if (d == qi) continue;
                    int dup = 0; for (int j = 0; j < nc; j++) if (chosen[j] == d) { dup = 1; break; }
                    if (dup) continue;
                    chosen[nc++] = d;
                }
                int slotneedle[256];
                for (int j = 0; j < nc; j++) slotneedle[j] = chosen[j];
                slotneedle[nc] = qi;
                int kk = nc + 1;
                for (int a = kk - 1; a > 0; a--) { int b = rng_below(a + 1); int t = slotneedle[a]; slotneedle[a] = slotneedle[b]; slotneedle[b] = t; }
                for (int j = 0; j < kk; j++) { cand[j] = slotneedle[j]; if (slotneedle[j] == qi) truth_idx = j; }
                for (int j = kk; j < K; j++) cand[j] = -1;
                query = needles[qi].query;
            } else {
                int chosen[256]; int nc = 0; int guard = 0;
                while (nc < K && guard < 100000) {
                    guard++; int d = rng_below(NN);
                    int dup = 0; for (int j = 0; j < nc; j++) if (chosen[j] == d) { dup = 1; break; }
                    if (dup) continue;
                    chosen[nc++] = d;
                }
                for (int j = 0; j < nc; j++) cand[j] = chosen[j];
                for (int j = nc; j < K; j++) cand[j] = -1;
                truth_idx = -1;
                query = foreign[qi];
            }

            int tagperm[512];
            for (int j = 0; j < n_tags; j++) tagperm[j] = j;
            for (int a = n_tags - 1; a > 0; a--) { int b = rng_below(a + 1); int t = tagperm[a]; tagperm[a] = tagperm[b]; tagperm[b] = t; }
            int realK = 0; for (int j = 0; j < K; j++) if (cand[j] >= 0) realK++;
            for (int j = 0; j < K; j++) picktag[j] = (cand[j] >= 0) ? tagperm[j % n_tags] : -1;

            char *p = prompt; size_t rem = 64 * 1024;
            int w = snprintf(p, rem, "%s ", INSTR); p += w; rem -= (size_t)w;
            for (int j = 0; j < K; j++) {
                if (cand[j] < 0) continue;
                w = snprintf(p, rem, "[%s] %s ", tagstr[picktag[j]], needles[cand[j]].text);
                p += w; rem -= (size_t)w;
            }
            w = snprintf(p, rem, "QUESTION: %s Tag of the answer (or %s):", query, null_str);
            p += w; rem -= (size_t)w;

            long np = sp_tokenizer_encode(tk, prompt, strlen(prompt), 1, toks, max_tok);
            if (np <= 0) { EMIT("[err] tokenize failed q=%d\n", trial_no); continue; }
            if (np > max_tok - CL - 4) { EMIT("[err] prompt too long (%ld toks) q=%d\n", np, trial_no); continue; }
            int P = (int)np;
            int n_tok = P + CL;

            /* allowed answer-token mask set: each real candidate's {bare,spaced} + NULL {bare,spaced} */
            int allow[1024]; int n_allow = 0;
            for (int j = 0; j < K; j++) {
                if (cand[j] < 0) continue;
                int t = picktag[j];
                if (tagtok[t]  >= 0 && tagtok[t]  < V) allow[n_allow++] = tagtok[t];
                if (tagtok2[t] >= 0 && tagtok2[t] < V) allow[n_allow++] = tagtok2[t];
            }
            if (null_tok  >= 0 && null_tok  < V) allow[n_allow++] = null_tok;
            if (null_tok2 >= 0 && null_tok2 < V) allow[n_allow++] = null_tok2;

            /* ── seed the canvas randomly (deterministic per trial) ── */
            for (int i = 0; i < CL; i++) { int r = (int)(rng_next() % (uint64_t)V); canvas[i] = r; outc[i] = r; }
            int prev_argmax_ans = -1, held = 0;
            int n_steps_run = 0;
            float *logits = (float *)malloc((size_t)n_tok * V * sizeof(float));
            if (!logits) { EMIT("[err] logits OOM q=%d\n", trial_no); continue; }

            clock_t tq = clock();
            int have_prev = 0;                /* SC available from step>=1 */
            /* N6 Adaptive Residue-KV probe (SP_DG_PKV_STRIDE=k): every k-th denoise step,
             * flush the prefix-KV cache so the next forward MISSES -> full recompute ->
             * drift-zeroed prompt K/V. Bounded-staleness: ride the fast path between flushes,
             * refresh before the canvas->prompt leak compounds to the NaN horizon. k<=0 = off. */
            extern void dg_prefixkv_release(void);
            int pkv_stride = getenv("SP_DG_PKV_STRIDE") ? atoi(getenv("SP_DG_PKV_STRIDE")) : 0;
            for (int step = 0; step < STEPS; step++) {
                if (pkv_stride > 0 && step > 0 && (step % pkv_stride) == 0) dg_prefixkv_release();
                float tt = (STEPS > 1) ? (float)step / (float)(STEPS) : 0.0f;
                float temp = T_MAX + (T_MIN - T_MAX) * tt;   /* lerp(t_max, t_min) */
                if (temp < 1e-3f) temp = 1e-3f;
                float inv_temp = 1.0f / temp;

                for (int i = 0; i < CL; i++) toks[P + i] = canvas[i];

                int frc;
                if (use_sc && have_prev)
                    frc = diffusion_gemma_forward_cuda_sc(m, toks, n_tok, logits, (const float *)prev_dev, 1.0f / (T_MAX + (T_MIN - T_MAX) * ((float)(step-1)/(float)STEPS)));
                else
                    frc = diffusion_gemma_forward_cuda(m, toks, n_tok, logits);
                if (frc != 0) { EMIT("[err] forward rc=%d (%s) q=%d step=%d\n", frc, sp_last_error(), trial_no, step); break; }
                n_steps_run = step + 1;

                /* HARD-MASK the ANSWER canvas row (row P) to {tags,NULL} BEFORE sampling.
                 * Non-answer canvas rows stay unconstrained (free bidirectional context). */
                {
                    float *arow = logits + (size_t)P * V;
                    /* save allowed values, blast row to -INF, restore allowed */
                    float saved[1024];
                    for (int a = 0; a < n_allow; a++) saved[a] = arow[allow[a]];
                    for (int v = 0; v < V; v++) arow[v] = -3.0e38f;
                    for (int a = 0; a < n_allow; a++) arow[allow[a]] = saved[a];
                }

                /* seeded uniforms for the multinomial draw */
                for (int i = 0; i < CL; i++) uvec[i] = rng_uniform();

                /* N6-JUDGEFIX: sample the CANVAS rows [P,P+CL), not the prompt rows [0,CL).
                 * The mask (line ~448) and the SC upload (below) both offset by P*V; the sampler
                 * dropped it -> am/sm were argmax/sample of PROMPT row 0 (constant bias tok 1852),
                 * the canvas never denoised, recall pinned to 0. Align it with +P*V. */
                if (dg_sample_logits_host(logits + (size_t)P * V, CL, V, uvec, inv_temp, am, ent, sm) != 0) {
                    EMIT("[err] dg_sample rc (%s) q=%d step=%d\n", sp_last_error(), trial_no, step); break;
                }

                /* order canvas positions by ASCENDING entropy (insertion sort, CL small) */
                for (int i = 0; i < CL; i++) order[i] = i;
                for (int a = 1; a < CL; a++) { int key = order[a]; int b = a - 1;
                    while (b >= 0 && ent[order[b]] > ent[key]) { order[b+1] = order[b]; b--; } order[b+1] = key; }
                /* accept the lowest-entropy prefix while cumE <= entropy_bound (MI bound) */
                float cumE = 0.0f;
                for (int oi = 0; oi < CL; oi++) {
                    int pos = order[oi];
                    int accept = (cumE <= ENTROPY_BOUND);
                    cumE += ent[pos];
                    canvas[pos] = accept ? sm[pos] : (int)(rng_next() % (uint64_t)V);   /* renoise the uncertain */
                }
                for (int i = 0; i < CL; i++) outc[i] = am[i];     /* output = argmax canvas */

                /* SC: persist this step's RAW canvas logits to the device prev buffer.
                 * NOTE: we re-run the forward unmasked logits? We only have the masked
                 * answer row; the SC feeds the WHOLE canvas logits. To stay faithful we
                 * upload the canvas logits AS THEY WENT INTO the sampler — masked answer
                 * row included. The masked -INF entries soft-max to ~0, a benign feedback
                 * (the reference feeds raw logits; here the answer row's non-tag mass is
                 * suppressed, which is consistent with the constrained task). */
                if (use_sc) {
                    if (dg_dev_upload(prev_dev, logits + (size_t)P * V, (long)CL * V) == 0) have_prev = 1;
                }

                /* adaptive stop keyed on the ANSWER row (row 0 = the masked answer pos):
                 * the judge only cares about the answer token, and the non-answer canvas
                 * rows are unconstrained (they stay high-entropy and would block a
                 * mean-over-canvas stop forever). So stop once the ANSWER argmax is stable
                 * for STABILITY_THRESHOLD steps AND the answer-row entropy is confident.
                 * (mean_ent reported for telemetry.) */
                int ans_argmax = am[0];
                held = (prev_argmax_ans == ans_argmax) ? held + 1 : 0;
                float mean_ent = 0.0f; for (int i = 0; i < CL; i++) mean_ent += ent[i]; mean_ent /= (float)CL;
                int stop = (held >= STABILITY_THRESHOLD && ent[0] < CONFIDENCE_THRESHOLD);
                prev_argmax_ans = ans_argmax;
                if (getenv("SP_DJ_STEPTRACE"))
                    EMIT("    step %d: ans_tok=%d ans_ent=%.4f mean_ent=%.4f held=%d\n",
                         step, ans_argmax, ent[0], mean_ent, held);
                if (stop) break;
            }
            double secs = (double)(clock() - tq) / CLOCKS_PER_SEC;

            /* ── final pick = the constrained argmax at the answer row (outc[0]).
             * map the answer token -> candidate slot (or NULL). outc[0] is already
             * constrained to {tags,NULL} by the per-step mask. ── */
            int ans = outc[0];
            int pick = K;     /* default NULL */
            if (ans == null_tok || ans == null_tok2) { pick = K; }
            else {
                for (int j = 0; j < K; j++) {
                    if (cand[j] < 0) continue;
                    int t = picktag[j];
                    if (ans == tagtok[t] || ans == tagtok2[t]) { pick = j; break; }
                }
            }
            free(logits);

            if (phase == 0) {
                int ok = (pick == truth_idx);
                tot++; hit += ok ? 1 : 0;
                EMIT("[match] %-6s %-14s K=%d gt=%2d pick=%2d steps=%d ans_tok=%d %.0fs :: %.42s\n",
                     ok ? "OK" : "MISS", needles[qi].id, realK, truth_idx,
                     (pick == K ? -1 : pick), n_steps_run, ans, secs, query);
                if (!ok) EMIT("        truth='%s'\n", needles[qi].text);
            } else {
                int ok = (pick == K);
                ftot++; frej += ok ? 1 : 0;
                EMIT("[forgn] %-6s K=%d pick=%2d steps=%d ans_tok=%d %.0fs :: %.42s\n",
                     ok ? "REJECT" : "FALSE", realK, (pick == K ? -1 : pick), n_steps_run, ans, secs, query);
            }
            double elapsed = (double)(clock() - t_start) / CLOCKS_PER_SEC;
            EMIT("# progress %d/%d  recall=%d/%d reject=%d/%d  elapsed=%.0fs\n",
                 trial_no, total_trials, hit, tot, frej, ftot, elapsed);
        }
    }

    double rec = tot ? (double)hit / tot : 0.0;
    double rej = ftot ? (double)frej / ftot : 0.0;
    EMIT("\n================ G-DIFFJUDGE-NATIVE-full (SUBSET) RESULT ================\n");
    EMIT("K=%d seed=%llu STEPS<=%d CL=%d SC=%d  subset matched=%d foreign=%d\n",
         K, (unsigned long long)seed, STEPS, CL, use_sc, n_matched, n_for);
    EMIT("recall@1       : %d/%d = %.1f%%   (single-forward ~14%% ; oracle 95.6%%)\n", hit, tot, 100.0 * rec);
    EMIT("foreign-reject : %d/%d = %.1f%%   (oracle 96.0%%)\n", frej, ftot, 100.0 * rej);
    int strong = (rec >= 0.80);
    EMIT("SUBSET recall >= 80%% : %s\n", strong ? "STRONG PASS — iterative denoise cures the single-forward miss" : "below 80%% — see per-query diagnosis");

    if (prev_dev) dg_dev_free(prev_dev);
    free(toks); free(prompt); free(am); free(ent); free(sm); free(uvec);
    free(canvas); free(outc); free(order); free(cand); free(picktag);
    free(tagtok); free(tagtok2); free((void *)tagstr);
    sp_tokenizer_free(tk);
    sp_model_unload(handle);
    for (int i = 0; i < NF; i++) free(foreign[i]);
    free(foreign); free(needles);
    fclose(log);
    return 0;
}
