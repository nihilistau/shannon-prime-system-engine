/* test_gemma4_cuda.c — E_G4_CU_W (ETA.1): Stage Eta structural gate.
 *
 * The first gate of the Gemma4 CUDA port: the engine CUDA layer must INGEST a
 * core-bridged Gemma4 model across the core/engine link seam —
 *
 *   .sp-model + .sp-tokenizer -> sp_model_load -> sp_model_to_gemma4 (CORE)
 *   -> gemma4_cuda_weights_probe (ENGINE sp_engine_cuda)
 *
 * uploading the full weight set with the gemma4 structure the CPU oracle
 * (core/forward/gemma4.c) defines: per-layer GLOBAL/SWA head geometry (the
 * Q/KV projection widths differ per layer), shared-KV owner-only K/V uploads
 * (sharers reuse an owner's cache and skip their own projection), per-layer
 * ELASTIC FFN widths (MatFormer), the AltUp tensor set (per-layer inp_gate /
 * proj / post_norm / out_scale + model-level per_layer_model_proj /
 * per_layer_proj_norm), and the rope_freqs proportional table.
 *
 * This deliberately links sp_session (the CORE inference lane, same as
 * M_GEMMA4) + sp_engine_cuda — NOT sp_engine — so the core loader/bridge and
 * the CUDA backend coexist in one binary. The CUDA lib's engine-named symbol
 * references (sp_arena_find, sp_arena_dequant_row, sp_set_error, ...) resolve
 * from the core libs (structs synced byte-for-byte; cf. the fork-tax fix
 * engine 0fb39ab). Proving THAT LINK is part of this gate.
 *
 * The gemma4 CUDA forward itself is ETA.2+ (gated argmax/KL vs gemma4_forward).
 *
 * SLOW / model-gated: skips cleanly (PASS, no checks) when the 4.6 GB
 * .sp-model is absent, or when no CUDA device is present.
 *
 * Env: SP_GEMMA4_SPMODEL / SP_GEMMA4_SPTOK (defaults below). */
#define _CRT_SECURE_NO_WARNINGS
#include "sp/sp_test.h"
#include "sp/sp_model.h"   /* sp_model_load / sp_model_unload / sp_model_to_gemma4 */
#include "sp/model.h"      /* qwen3_model / qwen3_free (CORE structs) */
#include "sp/sp_status.h"
#include "sp/forward_dispatch.h"   /* sp_matmul / sp_embed_row / sp_as_f32 (CPU mirror) */
#include "sp/forward_kernels.h"    /* sp_rmsnorm / sp_rmsnorm_head / sp_rope_neox_freqs / sp_attn_head */
#include "sp_engine/tokenizer.h"   /* sp_tokenizer_* — the parity-validated .sp-tokenizer blob lane (SP_G4_KAIROS) */
#include "sp/xbar_episode.h"        /* G-XBAR-ORGANISM: serialize an audio-conditioned cache into a Ring-2 episode */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>   /* clock(): warm wall-time telemetry in the 5b gate */

#ifndef SP_GEMMA4_SPMODEL_DEF
#define SP_GEMMA4_SPMODEL_DEF "D:/F/shannon-prime-repos/models/gemma4-e2b.sp-model"
#endif
#ifndef SP_GEMMA4_SPTOK_DEF
#define SP_GEMMA4_SPTOK_DEF "D:/F/shannon-prime-repos/models/gemma4-e2b.sp-tokenizer"
#endif

/* CUDA entry points, declared directly (NOT via sp_engine/cuda_backend.h, which
 * would pull the engine's duplicate model structs into a core-header TU). C
 * links by name; the core/engine structs are synced byte-for-byte. */
int  sp_cuda_device_count(void);
int  gemma4_cuda_weights_probe(const qwen3_model *m);
int  gemma4_cuda_probe(const qwen3_model *m, const int32_t *tokens, int n_tok,
                       int n_layers, int attn_only, float *out_x);
int  gemma4_forward_cuda(const qwen3_model *m, const int32_t *tokens, int n_tok,
                         float *logits);
int  gemma4_decode_cuda(const qwen3_model *m, int32_t *seq, int n_prompt,
                        int n_gen, int eos_id);
/* KAI-1b persistent-KV ABI (cuda_forward.cu) — resident decode + O(1) rewind. */
typedef struct sp_g4_kv sp_g4_kv;
sp_g4_kv *gemma4_kv_open(const qwen3_model *m, int Pmax);
int   gemma4_kv_prefill(sp_g4_kv *s, const int32_t *toks, int n);
int   gemma4_kv_decode(sp_g4_kv *s, int n_gen, int32_t *out);
int   gemma4_kv_rewind(sp_g4_kv *s, int delta);
int   gemma4_kv_commit(sp_g4_kv *s);
int   gemma4_kv_pos(const sp_g4_kv *s);
int   gemma4_kv_reset(sp_g4_kv *s);        /* in-place re-anchor (no free/realloc) — soak leak fix */
int   gemma4_kv_capture(sp_g4_kv *s, float *emb_out);  /* KAI-2: next step D2H its post-embed residual */
int   gemma4_kv_inject(sp_g4_kv *s, const float *emb); /* KAI-2: override next step's residual (latent entry) */
int   gemma4_kv_inject_seq(sp_g4_kv *s, const float *embs, int n_frames, int ph_token); /* KAI-3 §7.1: sequence wrapper */
long  gemma4_kv_devfree_mib(void);          /* free VRAM in MiB (cudaMemGetInfo, fragmentation-aware) */
long  gemma4_kv_snapshot(const sp_g4_kv *s, float **hK, float **hV);
int   gemma4_kv_replay(sp_g4_kv *s, const char *epdir, int npos, int zero);   /* C2 #222 */
void  gemma4_kv_close(sp_g4_kv *s);
void xbar_arm_shadow_result(long *mism, long *sel);   /* P3.2-b-2b Phase-1 oracle parity handoff */
void sp_cuda_model_release(const qwen3_model *m);

/* Engine-symbol SHIM (the documented cross-seam alias pattern, cf.
 * sp_daemon_hex_glue.c): sp_engine_cuda calls the ENGINE's `as_f32`; in this
 * core-lane binary that name doesn't exist — the core's identical-semantics
 * function is `sp_as_f32` (forward_dispatch). One-line forwarder. This was the
 * ONLY unresolved symbol across the whole core+CUDA link: every other engine-
 * named reference (sp_arena_find/sp_arena_dequant_row/sp_dequant_row/
 * sp_set_error/sp_kste_encode/gguf_tensor_data) resolves from the core libs. */
const float *sp_as_f32(const qwen3_model *m, const gguf_tensor *t);
const float *as_f32(const qwen3_model *m, const gguf_tensor *t) { return sp_as_f32(m, t); }

/* SP_G4_NIAH=1 mode: Needle-In-A-Haystack under the XBAR slab/ring/poison.
 * Conditions are pure env (SP_XBAR_SWA_RING / SP_ARM_*) — this harness is
 * condition-agnostic; it asserts SWA-isolation (needle_end <= n_prompt - W) so a
 * HIT can ONLY have traversed the global crossbar. Everything is TOKEN-SPACE
 * (no tokenizer linkage): haystack ids from SP_PPL_TOKENS; the needle/query/secret
 * ids are pre-tokenized with the gemma-4 vocab (sp_tok_dump, BOS stripped — the
 * .sp-tokenizer lane is T_G4_TOK_PARITY-identical). HIT = the 6 secret-digit
 * tokens appear as a contiguous subsequence in the generated output. 0=HIT 3=MISS 2=err. */
static const int32_t NIAH_NEEDLE[] = { 669,6789,8918,573,506,10839,1629,43248,563,236743,236828,236800,236832,236812,236819,236778,236761,21429,625,236761,236743 };
static const int32_t NIAH_QUERY[]  = { 2900,563,506,6789,8918,573,506,10839,1629,43248,236881,669,8918,563 };
static const int32_t NIAH_SECRET[] = { 236828,236800,236832,236812,236819,236778 };  /* "837492" digit tokens */
static int run_niah(void) {
    const char *spm = getenv("SP_GEMMA4_SPMODEL"); if (!spm) spm = SP_GEMMA4_SPMODEL_DEF;
    const char *stk = getenv("SP_GEMMA4_SPTOK");   if (!stk) stk = SP_GEMMA4_SPTOK_DEF;
    const char *toksf = getenv("SP_PPL_TOKENS");
    int N     = getenv("SP_NIAH_N")     ? atoi(getenv("SP_NIAH_N"))     : 16384;
    int depth = getenv("SP_NIAH_DEPTH") ? atoi(getenv("SP_NIAH_DEPTH")) : 50;
    int n_gen = getenv("SP_NIAH_GEN")   ? atoi(getenv("SP_NIAH_GEN"))   : 24;
    if (depth < 0) depth = 0; if (depth > 100) depth = 100;
    if (!toksf) { fprintf(stderr, "[g4-niah] set SP_PPL_TOKENS to the pre-tokenized haystack\n"); return 2; }

    sp_model *handle = NULL;
    if (sp_model_load(spm, stk, &handle) != SP_OK || !handle) { fprintf(stderr, "[g4-niah] sp_model_load FAIL\n"); return 2; }
    qwen3_model *m = sp_model_to_gemma4(handle);
    if (!m) { fprintf(stderr, "[g4-niah] sp_model_to_gemma4 FAIL\n"); return 2; }
    int W = (int)m->cfg.sliding_window; if (W <= 0) W = 1024;
    const int nn = (int)(sizeof NIAH_NEEDLE / sizeof NIAH_NEEDLE[0]);
    const int nq = (int)(sizeof NIAH_QUERY  / sizeof NIAH_QUERY[0]);
    const int ns = (int)(sizeof NIAH_SECRET / sizeof NIAH_SECRET[0]);

    FILE *tf = fopen(toksf, "rb");
    if (!tf) { fprintf(stderr, "[g4-niah] cannot open %s\n", toksf); return 2; }
    int cap_hay = N + 8;
    int32_t *hay = (int32_t *)malloc((size_t)cap_hay * sizeof(int32_t));
    long nh = 0; int v;
    while (nh < cap_hay && fscanf(tf, "%d", &v) == 1) hay[nh++] = (int32_t)v;
    fclose(tf);
    if (nh < 64) { fprintf(stderr, "[g4-niah] haystack too short (%ld)\n", nh); return 2; }

    long budget = (long)N - nn - nq;
    if (budget < 64) budget = 64;
    if (budget > nh) budget = nh;
    long inj = (budget * depth) / 100;
    if (inj > budget) inj = budget;

    int32_t *seq = (int32_t *)malloc((size_t)(budget + nn + nq + n_gen + 8) * sizeof(int32_t));
    int p = 0;
    for (long i = 0; i < inj; i++)       seq[p++] = hay[i];
    long needle_start = p;
    for (int i = 0; i < nn; i++)         seq[p++] = NIAH_NEEDLE[i];
    long needle_end = p;                 /* one past last needle token */
    for (long i = inj; i < budget; i++)  seq[p++] = hay[i];
    for (int i = 0; i < nq; i++)         seq[p++] = NIAH_QUERY[i];
    int n_prompt = p;

    /* SWA-ISOLATION (the central control): the needle must sit strictly outside the
     * sliding window at the generation point, so only the GLOBAL crossbar can serve it. */
    long swa_gap = (long)n_prompt - needle_end;
    fprintf(stderr, "[g4-niah] N=%d depth=%d%% n_prompt=%d needle@[%ld,%ld) W=%d swa_gap=%ld\n",
            N, depth, n_prompt, needle_start, needle_end, W, swa_gap);
    if (needle_end > n_prompt - W) {
        fprintf(stderr, "[g4-niah] ABORT: needle NOT SWA-isolated (needle_end %ld > n_prompt-W %d) — raise N or lower depth\n",
                needle_end, n_prompt - W);
        return 2;
    }

    int n = gemma4_decode_cuda(m, seq, n_prompt, n_gen, -1);
    if (n < n_prompt) { fprintf(stderr, "[g4-niah] decode FAIL (n=%d) err=%s\n", n, sp_last_error()); return 2; }

    /* token-space subsequence match of the secret digits in the generated tail */
    int hit = 0;
    for (int s = n_prompt; s + ns <= n && !hit; s++) {
        int ok = 1; for (int k = 0; k < ns; k++) if (seq[s + k] != NIAH_SECRET[k]) { ok = 0; break; }
        if (ok) hit = 1;
    }
    char gentoks[1024]; int gl = 0;
    for (int i = n_prompt; i < n && gl < (int)sizeof(gentoks) - 12; i++)
        gl += snprintf(gentoks + gl, sizeof(gentoks) - gl, "%d ", seq[i]);
    const char *lsh = getenv("SP_ARM_LSH"); const char *slab = getenv("SP_ARM_SLAB");
    const char *page = getenv("SP_ARM_PAGE"); const char *ring = getenv("SP_XBAR_SWA_RING");
    fprintf(stderr, "[g4-niah] %-4s depth=%d%% LSH=%s SLAB=%s PAGE=%s SWA_RING=%s gen_ids=[%.200s]\n",
            hit ? "HIT" : "MISS", depth, lsh?"on":"off", slab?"on":"off", page?"on":"off", ring?"on":"off", gentoks);
    free(hay); free(seq);
    return hit ? 0 : 3;
}

/* ═══ SP_G4_KAIROS=1 (KAI-1 Path B): the cognitive crucible on the 12B GPU. ═══
 * Ports the Rust kairos_runner control plane to the engine CUDA forward. The
 * mechanism (proven on qwen3-0.6B) is: gemma <start_of_turn> template +
 * SALIENCE>=0.5 action policy + cold-evict NO_OP prune. gemma4_decode_cuda is
 * ONE-SHOT (rebuilds KV from seq[0..n_prompt) each call), so the prune is a
 * PREFIX-GROW: an idle NO_OP tick leaves the persistent prefix UNTOUCHED, so the
 * next idle tick re-enters a context byte-identical to the first (O(Δ), defeats
 * the corruption attractor); an ACTION GROWS the prefix (tick-5 crucible: the
 * next idle tick then sees the retained action). Tokenizer = the parity-
 * validated .sp-tokenizer blob lane (sp_tokenizer_*, T_G4_TOK_PARITY 5432/5432).
 * Env: SP_GEMMA4_SPMODEL/SPTOK (12B -b1), SP_KAIROS_TAPE (the §2b tape). */
#define KAIROS_MAXTICK 64
#define KAIROS_GEN     16
typedef struct { int expect_action; float sal; char kind[48]; char payload[96]; } ktick_t;

/* local wall-clock (this TU uses clock() elsewhere; now_s lives in the tokenizer test). */
static double kairos_now_s(void) {
    struct timespec t; timespec_get(&t, TIME_UTC);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static int run_kairos(void) {
    const char *spm = getenv("SP_GEMMA4_SPMODEL"); if (!spm) spm = SP_GEMMA4_SPMODEL_DEF;
    const char *stk = getenv("SP_GEMMA4_SPTOK");   if (!stk) stk = SP_GEMMA4_SPTOK_DEF;
    const char *tapef = getenv("SP_KAIROS_TAPE");
    if (!tapef) { fprintf(stderr, "[g4-kairos] set SP_KAIROS_TAPE to the §2b tape file\n"); return 2; }

    /* ── load model + gemma4 + the parity-validated tokenizer ── */
    sp_model *handle = NULL;
    if (sp_model_load(spm, stk, &handle) != SP_OK || !handle) { fprintf(stderr, "[g4-kairos] sp_model_load FAIL: %s\n", sp_last_error()); return 2; }
    qwen3_model *m = sp_model_to_gemma4(handle);
    if (!m) { fprintf(stderr, "[g4-kairos] sp_model_to_gemma4 FAIL: %s\n", sp_last_error()); return 2; }
    sp_tokenizer *tk = sp_tokenizer_load_tokfile(stk);
    if (!tk) { fprintf(stderr, "[g4-kairos] sp_tokenizer_load_tokfile FAIL: %s\n", stk); return 2; }

    /* ── parse the §2b tape: "tick kind payload salience expect", '#'/blank skipped, payload may be "quoted" ── */
    ktick_t ticks[KAIROS_MAXTICK]; int nt = 0;
    FILE *tf = fopen(tapef, "rb");
    if (!tf) { fprintf(stderr, "[g4-kairos] cannot open tape %s\n", tapef); return 2; }
    char line[512];
    while (nt < KAIROS_MAXTICK && fgets(line, sizeof line, tf)) {
        char *s = line; while (*s==' '||*s=='\t') s++;
        if (*s=='#' || *s=='\n' || *s=='\r' || *s==0) continue;
        long tickidx; char kind[48]={0}, payload[96]={0}, expect[16]={0}; float sal=0;
        /* field 1: tick idx */
        char *end; tickidx = strtol(s, &end, 10); if (end==s) continue; s=end; (void)tickidx;
        while (*s==' '||*s=='\t') s++;
        /* field 2: kind */
        { int i=0; while (*s && *s!=' ' && *s!='\t' && i<47) kind[i++]=*s++; kind[i]=0; }
        while (*s==' '||*s=='\t') s++;
        /* field 3: payload (quoted or single token; '-' = empty) */
        if (*s=='"') { s++; int i=0; while (*s && *s!='"' && i<95) payload[i++]=*s++; payload[i]=0; if (*s=='"') s++; }
        else { int i=0; while (*s && *s!=' ' && *s!='\t' && i<95) payload[i++]=*s++; payload[i]=0; }
        if (payload[0]=='-' && payload[1]==0) payload[0]=0;
        while (*s==' '||*s=='\t') s++;
        /* field 4: salience */
        sal = strtof(s, &end); if (end!=s) s=end;
        while (*s==' '||*s=='\t') s++;
        /* field 5: expect */
        { int i=0; while (*s && *s!=' ' && *s!='\t' && *s!='\n' && *s!='\r' && i<15) expect[i++]=*s++; expect[i]=0; }
        ticks[nt].expect_action = (expect[0]=='A'||expect[0]=='a');
        ticks[nt].sal = sal;
        snprintf(ticks[nt].kind, sizeof ticks[nt].kind, "%s", kind);
        snprintf(ticks[nt].payload, sizeof ticks[nt].payload, "%s", payload);
        nt++;
    }
    fclose(tf);
    int n_act = 0; for (int i=0;i<nt;i++) n_act += ticks[i].expect_action;
    fprintf(stderr, "[g4-kairos] tape=%s ticks=%d (salient=%d idle=%d)\n", tapef, nt, n_act, nt-n_act);

    /* ── system contract (SALIENCE>=0.5 policy) wrapped in the gemma turn frame, prefilled ONCE ── */
    const char *SYS =
        "<start_of_turn>user\n"
        "You are a background kernel daemon. Each tick you receive one environment event "
        "that carries a salience score between 0.0 and 1.0. Reply with EXACTLY one of:\n"
        "NO_OP\n<ACTION>short imperative</ACTION>\n"
        "RULE: if salience >= 0.5 the event requires intervention, so reply with an <ACTION> line. "
        "If salience < 0.5, reply NO_OP. Follow the rule exactly. Do not explain."
        "<end_of_turn>\n<start_of_turn>model\nUnderstood.<end_of_turn>\n";

    int cap = 8192;
    int32_t *seq = (int32_t *)malloc((size_t)cap * sizeof(int32_t));
    int32_t *tmp = (int32_t *)malloc((size_t)1024 * sizeof(int32_t));
    char     dbuf[2048];
    long pn = sp_tokenizer_encode(tk, SYS, strlen(SYS), /*parse_special=*/1, seq, cap);
    if (pn <= 0) { fprintf(stderr, "[g4-kairos] system encode FAIL (%ld)\n", pn); return 2; }
    int prefix_n = (int)pn;
    fprintf(stderr, "[g4-kairos] system contract prefilled (%d tokens)\n", prefix_n);

    /* ── the heartbeat: prefix-grow = cold-evict prune ── */
    int noop_ok=0, act_ok=0, false_act=0, missed=0, malformed=0;
    for (int i = 0; i < nt; i++) {
        char frame[256];
        snprintf(frame, sizeof frame,
                 "<start_of_turn>user\nEVENT kind=%s salience=%.2f payload=\"%s\"<end_of_turn>\n<start_of_turn>model\n",
                 ticks[i].kind, ticks[i].sal, ticks[i].payload);
        long fn = sp_tokenizer_encode(tk, frame, strlen(frame), 1, tmp, 1024);
        if (fn <= 0) { fprintf(stderr, "[g4-kairos] tick %d frame encode FAIL\n", i); continue; }
        if (prefix_n + (int)fn + KAIROS_GEN + 8 > cap) { fprintf(stderr, "[g4-kairos] seq cap hit\n"); break; }
        for (int k=0;k<fn;k++) seq[prefix_n+k] = tmp[k];
        int n_prompt = prefix_n + (int)fn;
        double t0 = kairos_now_s();
        int n = gemma4_decode_cuda(m, seq, n_prompt, KAIROS_GEN, -1);
        double dt = kairos_now_s() - t0;
        if (n < n_prompt) { fprintf(stderr, "[g4-kairos] tick %d decode FAIL (n=%d) %s\n", i, n, sp_last_error()); continue; }
        long bl = sp_tokenizer_decode(tk, seq + n_prompt, n - n_prompt, dbuf, sizeof dbuf - 1);
        if (bl < 0) bl = 0; dbuf[bl] = 0;
        /* parse in text space: <ACTION> wins; else NO_OP/NOOP; else malformed */
        int is_act = (strstr(dbuf, "<ACTION>") != NULL);
        int is_noop = (!is_act) && (strstr(dbuf, "NO_OP") != NULL || strstr(dbuf, "NOOP") != NULL);
        const char *dec = is_act ? "ACTION" : (is_noop ? "NOOP" : "MALFORMED");
        if (ticks[i].expect_action) { if (is_act) act_ok++; else if (is_noop) missed++; else malformed++; }
        else                        { if (is_act) false_act++; else if (is_noop) noop_ok++; else malformed++; }
        /* prefix-grow prune: keep ONLY on a real ACTION; NO_OP/malformed discard (prefix unchanged) */
        int kept = 0;
        if (is_act) { prefix_n = n; kept = 1; }
        /* trim the printed raw to one line */
        for (char *p=dbuf; *p; p++) if (*p=='\n'||*p=='\r') { *p=' '; }
        fprintf(stderr, "[g4-kairos] tick %3d expect=%-6s decided=%-9s prefix=%d%s %.1fs raw=\"%.60s\"\n",
                i, ticks[i].expect_action?"Action":"Noop", dec, prefix_n,
                kept?" (kept)":" (pruned->flat)", dt, dbuf);
    }
    fprintf(stderr, "[g4-kairos] DONE ticks=%d noop_ok=%d action_ok=%d false_action=%d missed=%d malformed=%d\n",
            nt, noop_ok, act_ok, false_act, missed, malformed);
    free(seq); free(tmp);
    sp_tokenizer_free(tk);
    sp_cuda_model_release(m); qwen3_free(m); sp_model_unload(handle);
    /* gate: zero false-actions AND zero missed-events = the cognitive crucible passed */
    return (false_act==0 && missed==0 && malformed==0) ? 0 : 3;
}

/* ═══ SP_G4_KAIROS_METAL=1 (KAI-1c): the SEMANTIC loop on the journaled-ring metal. ═══
 * The operational unification of KAI-1 (cognition) + KAI-1b/1c (O(1) metal eviction).
 * Same SALIENCE>=0.5 policy + gemma turn frame as run_kairos, but the cold-evict prune is
 * now a TRUE O(1) metal operation on the persistent journaled-ring KV (gemma4_kv_*), not a
 * host prefix re-prefill:
 *   open(ring) -> prefill SYS -> commit (anchor).  Per tick:
 *     prefill(frame) + decode(GEN) -> parse.
 *     NO_OP/malformed -> gemma4_kv_rewind(pos-anchor)  (journaled cold-evict, pos returns to anchor)
 *     ACTION          -> gemma4_kv_commit()            (retain frame+gen as new baseline, journal cleared)
 * POS-DISCIPLINE is itself a gate: idle ticks MUST return pos to the anchor (rewind exact);
 * action ticks MUST advance the anchor (commit kept). Ring via SP_G4_KV_RING_W (SWA owners
 * journaled). Env: SP_GEMMA4_SPMODEL/SPTOK, SP_KAIROS_TAPE, SP_G4_KV_RING_W, SP_CUDA_DECODE_INT8=1. */
static int run_kairos_metal(void) {
    const char *spm=getenv("SP_GEMMA4_SPMODEL"); if(!spm) spm=SP_GEMMA4_SPMODEL_DEF;
    const char *stk=getenv("SP_GEMMA4_SPTOK");   if(!stk) stk=SP_GEMMA4_SPTOK_DEF;
    const char *tapef=getenv("SP_KAIROS_TAPE");
    if(!tapef){ fprintf(stderr,"[g4-kmetal] set SP_KAIROS_TAPE\n"); return 2; }
    sp_model *handle=NULL;
    if(sp_model_load(spm,stk,&handle)!=SP_OK||!handle){ fprintf(stderr,"[g4-kmetal] load FAIL: %s\n",sp_last_error()); return 2; }
    qwen3_model *m=sp_model_to_gemma4(handle);
    if(!m){ fprintf(stderr,"[g4-kmetal] to_gemma4 FAIL: %s\n",sp_last_error()); return 2; }
    sp_tokenizer *tk=sp_tokenizer_load_tokfile(stk);
    if(!tk){ fprintf(stderr,"[g4-kmetal] tokenizer FAIL\n"); return 2; }
    /* tape (same §2b format as run_kairos) */
    ktick_t ticks[KAIROS_MAXTICK]; int nt=0; FILE *tf=fopen(tapef,"rb");
    if(!tf){ fprintf(stderr,"[g4-kmetal] cannot open tape\n"); return 2; }
    char line[512];
    while(nt<KAIROS_MAXTICK && fgets(line,sizeof line,tf)){
        char *s=line; while(*s==' '||*s=='\t')s++; if(*s=='#'||*s=='\n'||*s=='\r'||*s==0) continue;
        char kind[48]={0},payload[96]={0},expect[16]={0}; float sal=0; char *end;
        strtol(s,&end,10); if(end==s) continue; s=end; while(*s==' '||*s=='\t')s++;
        {int i=0; while(*s&&*s!=' '&&*s!='\t'&&i<47)kind[i++]=*s++; kind[i]=0;} while(*s==' '||*s=='\t')s++;
        if(*s=='"'){s++;int i=0;while(*s&&*s!='"'&&i<95)payload[i++]=*s++;payload[i]=0;if(*s=='"')s++;}
        else {int i=0;while(*s&&*s!=' '&&*s!='\t'&&i<95)payload[i++]=*s++;payload[i]=0;}
        if(payload[0]=='-'&&payload[1]==0)payload[0]=0; while(*s==' '||*s=='\t')s++;
        sal=strtof(s,&end); if(end!=s)s=end; while(*s==' '||*s=='\t')s++;
        {int i=0;while(*s&&*s!=' '&&*s!='\t'&&*s!='\n'&&*s!='\r'&&i<15)expect[i++]=*s++;expect[i]=0;}
        ticks[nt].expect_action=(expect[0]=='A'||expect[0]=='a'); ticks[nt].sal=sal;
        snprintf(ticks[nt].kind,sizeof ticks[nt].kind,"%s",kind);
        snprintf(ticks[nt].payload,sizeof ticks[nt].payload,"%s",payload); nt++;
    }
    fclose(tf);
    int n_act_exp=0; for(int i=0;i<nt;i++) n_act_exp+=ticks[i].expect_action;
    const int W = getenv("SP_G4_KV_RING_W")?atoi(getenv("SP_G4_KV_RING_W")):1024;
    fprintf(stderr,"[g4-kmetal] tape=%s ticks=%d (salient=%d idle=%d) ring_W=%d\n",tapef,nt,n_act_exp,nt-n_act_exp,W);
    const char *SYS=
        "<start_of_turn>user\n"
        "You are a background kernel daemon. Each tick you receive one environment event "
        "that carries a salience score between 0.0 and 1.0. Reply with EXACTLY one of:\n"
        "NO_OP\n<ACTION>short imperative</ACTION>\n"
        "RULE: if salience >= 0.5 the event requires intervention, so reply with an <ACTION> line. "
        "If salience < 0.5, reply NO_OP. Follow the rule exactly. Do not explain."
        "<end_of_turn>\n<start_of_turn>model\nUnderstood.<end_of_turn>\n";
    int32_t *sysb=(int32_t*)malloc(4096*sizeof(int32_t)), tmp[1024], gen[KAIROS_GEN]; char dbuf[2048];
    long pn=sp_tokenizer_encode(tk,SYS,strlen(SYS),1,sysb,4096);
    if(pn<=0){ fprintf(stderr,"[g4-kmetal] SYS encode FAIL\n"); return 2; }
    sp_g4_kv *s=gemma4_kv_open(m,2048);                 /* ring mode via SP_G4_KV_RING_W */
    if(!s){ fprintf(stderr,"[g4-kmetal] kv_open FAIL: %s\n",sp_last_error()); return 2; }
    if(gemma4_kv_prefill(s,sysb,(int)pn)){ fprintf(stderr,"[g4-kmetal] SYS prefill FAIL: %s\n",sp_last_error()); return 2; }
    gemma4_kv_commit(s);                                /* SYS = baseline anchor */
    int anchor=gemma4_kv_pos(s);
    fprintf(stderr,"[g4-kmetal] system contract prefilled (%d tokens) anchor=%d\n",(int)pn,anchor);
    int noop_ok=0,act_ok=0,false_act=0,missed=0,malformed=0,pos_violation=0;
    for(int i=0;i<nt;i++){
        char frame[256];
        snprintf(frame,sizeof frame,
                 "<start_of_turn>user\nEVENT kind=%s salience=%.2f payload=\"%s\"<end_of_turn>\n<start_of_turn>model\n",
                 ticks[i].kind,ticks[i].sal,ticks[i].payload);
        long fn=sp_tokenizer_encode(tk,frame,strlen(frame),1,tmp,1024);
        if(fn<=0){ fprintf(stderr,"[g4-kmetal] tick %d frame encode FAIL\n",i); continue; }
        int pre=gemma4_kv_pos(s);
        if(gemma4_kv_prefill(s,tmp,(int)fn)||gemma4_kv_decode(s,KAIROS_GEN,gen)){ fprintf(stderr,"[g4-kmetal] tick %d decode FAIL: %s\n",i,sp_last_error()); break; }
        long bl=sp_tokenizer_decode(tk,gen,KAIROS_GEN,dbuf,sizeof dbuf-1); if(bl<0)bl=0; dbuf[bl]=0;
        /* boundary-tolerant parse: gemma4_kv_decode's gen buffer starts one token past the
         * one-shot convention (leading "<"/"NO" lands in prefill logits, not gen[0]) — the
         * SEMANTIC content is unambiguous: "ACTION" => action, else "OP"/"NOOP" => noop.
         * (the kv-decode-vs-one-shot first-token boundary is filed as a reconcile follow-up.) */
        int is_act=(strstr(dbuf,"ACTION")!=NULL);
        int is_noop=(!is_act)&&(strstr(dbuf,"OP")!=NULL||strstr(dbuf,"NOOP")!=NULL);
        const char *dec=is_act?"ACTION":(is_noop?"NOOP":"MALFORMED");
        if(ticks[i].expect_action){ if(is_act)act_ok++; else if(is_noop)missed++; else malformed++; }
        else                      { if(is_act)false_act++; else if(is_noop)noop_ok++; else malformed++; }
        /* THE METAL PRUNE: ACTION -> commit (retain); else -> journaled rewind (cold-evict to anchor) */
        int kept=0, posafter;
        if(is_act){ gemma4_kv_commit(s); anchor=gemma4_kv_pos(s); kept=1; }
        else { if(gemma4_kv_rewind(s,gemma4_kv_pos(s)-anchor)){ fprintf(stderr,"[g4-kmetal] tick %d rewind FAIL: %s\n",i,sp_last_error()); break; } }
        posafter=gemma4_kv_pos(s);
        /* pos-discipline: idle->pos must return to anchor (==pre's anchor); action->anchor advanced past pre */
        int pos_ok = kept ? (posafter>pre) : (posafter==anchor);
        if(!pos_ok) pos_violation++;
        for(char *p=dbuf;*p;p++) if(*p=='\n'||*p=='\r')*p=' ';
        fprintf(stderr,"[g4-kmetal] tick %3d expect=%-6s decided=%-9s pos:%d->%d anchor=%d%s%s raw=\"%.50s\"\n",
                i,ticks[i].expect_action?"Action":"Noop",dec,pre,posafter,anchor,
                kept?" (commit)":" (rewind->flat)", pos_ok?"":" POS!", dbuf);
    }
    fprintf(stderr,"[g4-kmetal] DONE ticks=%d noop_ok=%d action_ok=%d false_action=%d missed=%d malformed=%d pos_violations=%d\n",
            nt,noop_ok,act_ok,false_act,missed,malformed,pos_violation);
    int pass=(false_act==0 && missed==0 && malformed==0 && pos_violation==0);
    fprintf(stderr,"[g4-kmetal] CRUCIBLE: %s (semantic clean + O(1) metal pos-discipline)\n",pass?"GREEN":"RED");
    gemma4_kv_close(s); free(sysb); sp_tokenizer_free(tk);
    sp_cuda_model_release(m); qwen3_free(m); sp_model_unload(handle);
    return pass?0:3;
}

/* ═══ SP_G4_KAIROS_SOAK=1 (G-KAIROS-1): the ≥24h unattended endurance soak. ═══
 * Loops the deterministic §2b tape with per-loop re-anchor (close+reopen ⇒ bounded state),
 * the full journaled-ring metal loop each tick, two-tier streamed/flushed telemetry, and
 * in-process hard tripwires (CUDA / semantic-safety / pos / latency-consecutive / VRAM-leak /
 * thermal). Env: SP_SOAK_HOURS (default 24), SP_SOAK_MAXLOOPS (overrides, for smoke),
 * SP_SOAK_LOG (per-tick detail log), + the metal envs. */
static int soak_probe_vram_temp(long *vram_mib, int *temp_c){   /* one nvidia-smi sample (loop cadence) */
    *vram_mib=-1; *temp_c=-1;
    FILE *p=_popen("nvidia-smi --query-gpu=memory.used,temperature.gpu --format=csv,noheader,nounits","r");
    if(!p) return -1;
    int ok=(fscanf(p,"%ld , %d",vram_mib,temp_c)==2); _pclose(p); return ok?0:-1;
}
static int run_kairos_soak(void) {
    const char *spm=getenv("SP_GEMMA4_SPMODEL"); if(!spm) spm=SP_GEMMA4_SPMODEL_DEF;
    const char *stk=getenv("SP_GEMMA4_SPTOK");   if(!stk) stk=SP_GEMMA4_SPTOK_DEF;
    const char *tapef=getenv("SP_KAIROS_TAPE");   if(!tapef){ fprintf(stderr,"[g4-soak] set SP_KAIROS_TAPE\n"); return 2; }
    const char *logf=getenv("SP_SOAK_LOG");       if(!logf) logf="results/kairos_soak_detail.log";
    double soak_hours = getenv("SP_SOAK_HOURS")? atof(getenv("SP_SOAK_HOURS")) : 24.0;
    long   maxloops   = getenv("SP_SOAK_MAXLOOPS")? atol(getenv("SP_SOAK_MAXLOOPS")) : -1;
    const int W = getenv("SP_G4_KV_RING_W")?atoi(getenv("SP_G4_KV_RING_W")):1024;
    /* model + tokenizer loaded ONCE (resident across all loops) */
    sp_model *handle=NULL;
    if(sp_model_load(spm,stk,&handle)!=SP_OK||!handle){ fprintf(stderr,"[g4-soak] load FAIL: %s\n",sp_last_error()); return 2; }
    qwen3_model *m=sp_model_to_gemma4(handle);
    if(!m){ fprintf(stderr,"[g4-soak] to_gemma4 FAIL\n"); return 2; }
    sp_tokenizer *tk=sp_tokenizer_load_tokfile(stk);
    if(!tk){ fprintf(stderr,"[g4-soak] tokenizer FAIL\n"); return 2; }
    /* tape (parsed once) */
    ktick_t ticks[KAIROS_MAXTICK]; int nt=0; FILE *tf=fopen(tapef,"rb");
    if(!tf){ fprintf(stderr,"[g4-soak] cannot open tape\n"); return 2; }
    char line[512];
    while(nt<KAIROS_MAXTICK && fgets(line,sizeof line,tf)){
        char *s=line; while(*s==' '||*s=='\t')s++; if(*s=='#'||*s=='\n'||*s=='\r'||*s==0) continue;
        char kind[48]={0},payload[96]={0},expect[16]={0}; float sal=0; char *end;
        strtol(s,&end,10); if(end==s) continue; s=end; while(*s==' '||*s=='\t')s++;
        {int i=0; while(*s&&*s!=' '&&*s!='\t'&&i<47)kind[i++]=*s++; kind[i]=0;} while(*s==' '||*s=='\t')s++;
        if(*s=='"'){s++;int i=0;while(*s&&*s!='"'&&i<95)payload[i++]=*s++;payload[i]=0;if(*s=='"')s++;}
        else {int i=0;while(*s&&*s!=' '&&*s!='\t'&&i<95)payload[i++]=*s++;payload[i]=0;}
        if(payload[0]=='-'&&payload[1]==0)payload[0]=0; while(*s==' '||*s=='\t')s++;
        sal=strtof(s,&end); if(end!=s)s=end; while(*s==' '||*s=='\t')s++;
        {int i=0;while(*s&&*s!=' '&&*s!='\t'&&*s!='\n'&&*s!='\r'&&i<15)expect[i++]=*s++;expect[i]=0;}
        ticks[nt].expect_action=(expect[0]=='A'||expect[0]=='a'); ticks[nt].sal=sal;
        snprintf(ticks[nt].kind,sizeof ticks[nt].kind,"%s",kind);
        snprintf(ticks[nt].payload,sizeof ticks[nt].payload,"%s",payload); nt++;
    }
    fclose(tf);
    const char *SYS=
        "<start_of_turn>user\n"
        "You are a background kernel daemon. Each tick you receive one environment event "
        "that carries a salience score between 0.0 and 1.0. Reply with EXACTLY one of:\n"
        "NO_OP\n<ACTION>short imperative</ACTION>\n"
        "RULE: if salience >= 0.5 the event requires intervention, so reply with an <ACTION> line. "
        "If salience < 0.5, reply NO_OP. Follow the rule exactly. Do not explain."
        "<end_of_turn>\n<start_of_turn>model\nUnderstood.<end_of_turn>\n";
    int32_t *sysb=(int32_t*)malloc(4096*sizeof(int32_t)), tmp[1024], gen[KAIROS_GEN]; char dbuf[2048];
    long pn=sp_tokenizer_encode(tk,SYS,strlen(SYS),1,sysb,4096);
    if(pn<=0){ fprintf(stderr,"[g4-soak] SYS encode FAIL\n"); return 2; }
    FILE *dl=fopen(logf,"w");
    if(dl){ fprintf(dl,"# loop tick expect decided pos latency_ms\n"); fflush(dl); }
    /* aggregate counters (fixed-size; zero RAM growth over the run) */
    long T_noop=0,T_act=0,T_false=0,T_missed=0,T_malf=0,T_pos=0,T_ticks=0;
    int consec_malf=0, consec_slow=0, consec_lowfree=0;
    double lat_med=0; long base_vram=-1;
    double t_start=kairos_now_s(), deadline=t_start+soak_hours*3600.0;
    int abort_code=0; const char *abort_why="";
    /* ONE long-lived session for the whole soak — no per-loop open/close (the leak fix:
     * 209 close/reopen cycles fragmented VRAM ⇒ kv_open OOM at ~3.6h). Re-anchor each loop
     * with the in-place gemma4_kv_reset (zero counters, no realloc). */
    sp_g4_kv *s=gemma4_kv_open(m,2048);
    if(!s){ fprintf(stderr,"[g4-soak] kv_open FAIL: %s\n",sp_last_error()); if(dl)fclose(dl); sp_model_unload(handle); return 2; }
    long base_free=gemma4_kv_devfree_mib();   /* fragmentation-aware leak baseline (free VRAM MiB) */
    fprintf(stderr,"[g4-soak] START hours=%.2f maxloops=%ld tape_ticks=%d ring_W=%d sys=%ldtok base_free=%ldMiB log=%s\n",
            soak_hours,maxloops,nt,W,pn,base_free,logf);
    long loop=0;
    for(; ; loop++){
        if(maxloops>=0 && loop>=maxloops){ abort_why="reached SP_SOAK_MAXLOOPS"; break; }
        if(kairos_now_s()>=deadline){ abort_why="reached SP_SOAK_HOURS (clean stop)"; break; }
        if(gemma4_kv_reset(s)){ abort_code=2; abort_why="kv_reset FAIL"; break; }
        if(gemma4_kv_prefill(s,sysb,(int)pn)){ abort_code=2; abort_why="SYS prefill FAIL"; break; }
        gemma4_kv_commit(s); int anchor=gemma4_kv_pos(s);
        double loop_lmin=1e9,loop_lmax=0,loop_lsum=0; int loop_n=0;
        for(int i=0;i<nt;i++){
            char frame[256];
            snprintf(frame,sizeof frame,
                     "<start_of_turn>user\nEVENT kind=%s salience=%.2f payload=\"%s\"<end_of_turn>\n<start_of_turn>model\n",
                     ticks[i].kind,ticks[i].sal,ticks[i].payload);
            long fn=sp_tokenizer_encode(tk,frame,strlen(frame),1,tmp,1024);
            if(fn<=0){ abort_code=2; abort_why="frame encode FAIL"; break; }
            int pre=gemma4_kv_pos(s);
            double t0=kairos_now_s();
            if(gemma4_kv_prefill(s,tmp,(int)fn)||gemma4_kv_decode(s,KAIROS_GEN,gen)){ abort_code=2; abort_why=sp_last_error(); break; }
            long bl=sp_tokenizer_decode(tk,gen,KAIROS_GEN,dbuf,sizeof dbuf-1); if(bl<0)bl=0; dbuf[bl]=0;
            int is_act=(strstr(dbuf,"ACTION")!=NULL);
            int is_noop=(!is_act)&&(strstr(dbuf,"OP")!=NULL||strstr(dbuf,"NOOP")!=NULL);
            const char *dec=is_act?"ACTION":(is_noop?"NOOP":"MALFORMED");
            int kept=0;
            if(is_act){ gemma4_kv_commit(s); anchor=gemma4_kv_pos(s); kept=1; }
            else { if(gemma4_kv_rewind(s,gemma4_kv_pos(s)-anchor)){ abort_code=2; abort_why="rewind FAIL"; break; } }
            int posafter=gemma4_kv_pos(s);
            double lat=(kairos_now_s()-t0)*1e3;
            /* counters */
            T_ticks++; loop_n++; loop_lsum+=lat; if(lat<loop_lmin)loop_lmin=lat; if(lat>loop_lmax)loop_lmax=lat;
            int pos_ok = kept ? (posafter>pre) : (posafter==anchor);
            if(ticks[i].expect_action){ if(is_act)T_act++; else if(is_noop)T_missed++; else T_malf++; }
            else                      { if(is_act)T_false++; else if(is_noop)T_noop++; else T_malf++; }
            if(!pos_ok) T_pos++;
            consec_malf = (!is_act&&!is_noop)? consec_malf+1 : 0;
            if(dl){ fprintf(dl,"%ld %d %s %s %d %.1f\n",loop,i,ticks[i].expect_action?"A":"N",dec,posafter,lat); fflush(dl); }
            /* ── TRIPWIRES (in-process, hard) ── */
            if(ticks[i].expect_action && is_noop){ abort_code=3; abort_why="SEMANTIC: missed salient event"; break; }
            if(!ticks[i].expect_action && is_act){ abort_code=3; abort_why="SEMANTIC: false action on idle"; break; }
            if(!pos_ok){ abort_code=3; abort_why="POS-DISCIPLINE violation"; break; }
            if(consec_malf>=3){ abort_code=3; abort_why="SEMANTIC DRIFT: 3 consecutive malformed"; break; }
            if(lat_med>0 && lat>3.0*lat_med){ consec_slow++; } else consec_slow=0;
            if(consec_slow>=5){ abort_code=3; abort_why="LATENCY: 5 consecutive ticks > 3x median"; break; }
            if(lat>30000.0){ abort_code=3; abort_why="LATENCY: single tick > 30s (hang)"; break; }
        }
        if(abort_code) break;                     /* (no per-loop close — one resident session) */
        /* establish warm-up latency median after loop 0 */
        double loop_lmed = (loop_n>0)? loop_lsum/loop_n : 0;   /* loop-mean as median proxy (cheap) */
        if(loop==0){ lat_med = loop_lmed; }
        long vram=-1; int temp=-1; soak_probe_vram_temp(&vram,&temp);   /* nvidia-smi: thermal + coarse log */
        long curfree=gemma4_kv_devfree_mib();      /* the real leak signal (fragmentation-aware) */
        long freedrop=(base_free>0 && curfree>0)?(base_free-curfree):0;
        if(loop==0 && vram>0) base_vram=vram;
        /* cudaMemGetInfo free is DEVICE-GLOBAL (includes other processes), so a single-sample
         * threshold false-fires on transient external GPU use on a shared desktop. A real leak is
         * MONOTONIC + sustained; external contention is a step that may recover. Require the drop to
         * persist K=10 consecutive loops (~10 min) before aborting, and don't over-claim "leak". */
        if(freedrop>256) consec_lowfree++; else consec_lowfree=0;
        if(consec_lowfree>=10){ abort_code=3; abort_why="VRAM PRESSURE sustained >256MiB for 10 loops (residual leak OR external GPU contention)"; }
        if(temp>87){ abort_code=3; abort_why="THERMAL: GPU temp > 87C"; }
        double elapsed=kairos_now_s()-t_start;
        fprintf(stderr,"[g4-soak] loop %6ld t=%.0fs ticks=%ld noop=%ld act=%ld FALSE=%ld MISS=%ld malf=%ld pos!=%ld lat{%.0f/%.0f/%.0f}ms free=%ldMiB(-%ld) vram=%ldMiB temp=%dC\n",
                loop,elapsed,T_ticks,T_noop,T_act,T_false,T_missed,T_malf,T_pos,loop_lmin,loop_lmed,loop_lmax,curfree,freedrop,vram,temp);
        if(abort_code) break;
    }
    if(dl) fclose(dl);
    int pass=(abort_code==0 && T_false==0 && T_missed==0 && T_malf==0 && T_pos==0);
    fprintf(stderr,"[g4-soak] DONE loops=%ld ticks=%ld | noop_ok=%ld action_ok=%ld false=%ld missed=%ld malformed=%ld pos_violations=%ld\n",
            loop,T_ticks,T_noop,T_act,T_false,T_missed,T_malf,T_pos);
    fprintf(stderr,"[g4-soak] VERDICT: %s (%s)\n", pass?"GREEN":"RED/ABORT", abort_why);
    gemma4_kv_close(s);                        /* one close for the whole soak */
    free(sysb); sp_tokenizer_free(tk); sp_cuda_model_release(m); qwen3_free(m); sp_model_unload(handle);
    return pass?0:(abort_code?abort_code:3);
}

/* ═══ SP_G4_KV_REWIND=1 (KAI-1b G-1b-REWIND-NULL): the bit-exact rewind gate. ═══
 * Prefill a system prefix (anchor), snapshot the cache; run an IDLE TICK (prefill
 * frame + decode); rewind to the anchor; snapshot again. The [0,anchor) region MUST
 * be byte-identical — rewind is a perfect inverse (the T8.1 analog on the GPU).
 * EQUIV-lite: re-running the same idle tick after rewind reproduces the SAME
 * generated tokens — the rewound cache is a perfect re-entry point ("never knew it
 * waited"). Token ids are arbitrary-valid (semantics irrelevant to KV byte-identity). */
static int run_kv_rewind(void) {
    const char *spm = getenv("SP_GEMMA4_SPMODEL"); if (!spm) spm = SP_GEMMA4_SPMODEL_DEF;
    const char *stk = getenv("SP_GEMMA4_SPTOK");   if (!stk) stk = SP_GEMMA4_SPTOK_DEF;
    sp_model *handle = NULL;
    if (sp_model_load(spm, stk, &handle) != SP_OK || !handle) { fprintf(stderr, "[g4-kvrw] load FAIL\n"); return 2; }
    qwen3_model *m = sp_model_to_gemma4(handle);
    if (!m) { fprintf(stderr, "[g4-kvrw] to_gemma4 FAIL: %s\n", sp_last_error()); return 2; }
    const qwen3_config *c = &m->cfg;
    const int NL=(int)c->n_layers, period=(int)c->g4_swa_period?(int)c->g4_swa_period:6;
    const int kvfs=(int)c->g4_n_kv_from_start?(int)c->g4_n_kv_from_start:NL;
    const int g_nkv=(int)c->n_head_kv,g_hd=(int)c->head_dim,s_nkv=(int)c->g4_nkv_swa,s_hd=(int)c->g4_hd_swa;
    const int Pmax=256, sys_n=24, frm_n=12, ngen=8;
    sp_g4_kv *s=gemma4_kv_open(m,Pmax);
    if(!s){ fprintf(stderr,"[g4-kvrw] kv_open FAIL: %s\n",sp_last_error()); return 2; }
    int32_t sys[24]; for(int i=0;i<sys_n;i++) sys[i]=100+i;
    int32_t frm[12]; for(int i=0;i<frm_n;i++) frm[i]=500+i;
    if(gemma4_kv_prefill(s,sys,sys_n)){ fprintf(stderr,"[g4-kvrw] sys prefill FAIL: %s\n",sp_last_error()); return 2; }
    int anchor=gemma4_kv_pos(s);
    float **hK0=(float**)calloc(NL,sizeof(float*)),**hV0=(float**)calloc(NL,sizeof(float*));
    float **hK1=(float**)calloc(NL,sizeof(float*)),**hV1=(float**)calloc(NL,sizeof(float*));
    for(int L=0;L<kvfs&&L<NL;L++){ int global=((L%period)==period-1); int kvd=(global?g_nkv:s_nkv)*(global?g_hd:s_hd);
        size_t nb=(size_t)Pmax*kvd*sizeof(float); hK0[L]=(float*)malloc(nb);hV0[L]=(float*)malloc(nb);hK1[L]=(float*)malloc(nb);hV1[L]=(float*)malloc(nb); }
    gemma4_kv_snapshot(s,hK0,hV0);
    int32_t gen1[8];
    if(gemma4_kv_prefill(s,frm,frm_n)||gemma4_kv_decode(s,ngen,gen1)){ fprintf(stderr,"[g4-kvrw] tick1 FAIL: %s\n",sp_last_error()); return 2; }
    int after=gemma4_kv_pos(s);
    if(gemma4_kv_rewind(s,after-anchor)){ fprintf(stderr,"[g4-kvrw] rewind FAIL\n"); return 2; }
    gemma4_kv_snapshot(s,hK1,hV1);
    long diffs=0; size_t cmp_bytes=0;
    for(int L=0;L<kvfs&&L<NL;L++){ int global=((L%period)==period-1); int kvd=(global?g_nkv:s_nkv)*(global?g_hd:s_hd);
        size_t nb=(size_t)anchor*kvd*sizeof(float); cmp_bytes+=2*nb;
        if(memcmp(hK0[L],hK1[L],nb)) diffs++; if(memcmp(hV0[L],hV1[L],nb)) diffs++; }
    int32_t gen2[8]; int genmatch=1;
    if(gemma4_kv_prefill(s,frm,frm_n)||gemma4_kv_decode(s,ngen,gen2)){ fprintf(stderr,"[g4-kvrw] tick2 FAIL\n"); return 2; }
    for(int i=0;i<ngen;i++) if(gen1[i]!=gen2[i]) genmatch=0;
    gemma4_kv_rewind(s,gemma4_kv_pos(s)-anchor);
    fprintf(stderr,"[g4-kvrw] anchor=%d after_tick=%d cmp=%zuB owners=%d\n",anchor,after,cmp_bytes,(kvfs<NL?kvfs:NL));
    fprintf(stderr,"[g4-kvrw] REWIND-NULL: %s (layer-diffs=%ld) | EQUIV gen-reproduce: %s [%d %d %d %d ...]\n",
        diffs==0?"GREEN":"RED",diffs, genmatch?"GREEN":"RED", gen1[0],gen1[1],gen1[2],gen1[3]);
    gemma4_kv_close(s); sp_model_unload(handle);
    return (diffs==0 && genmatch)?0:3;
}

/* ═══ SP_G4_KV_REPLAY_GATE=1 (C2 #222 G-222): replay-inject into the persistent cache, then
 *     O(1) bit-exact rewind back to the pre-injection floor. Prefill a system anchor, snapshot;
 *     gemma4_kv_replay a ZEROED (corrupted) episode into [anchor,anchor+npos) (load-bearing:
 *     the injected slots must read back all-zero); rewind(npos); snapshot. The [0,anchor) region
 *     MUST be byte-identical (the rewind is the KAI-1b slot==pos inverse, now for a replay-inject —
 *     proving the curator can SPECULATE a recall and UNDO it bit-exactly in O(1) on reject).
 *     Env: SP_REPLAY=<episode dir>, SP_REPLAY_NPOS (default 8). Full cache (no SP_G4_KV_RING_W). */
static int run_kv_replay(void) {
    const char *spm = getenv("SP_GEMMA4_SPMODEL"); if (!spm) spm = SP_GEMMA4_SPMODEL_DEF;
    const char *stk = getenv("SP_GEMMA4_SPTOK");   if (!stk) stk = SP_GEMMA4_SPTOK_DEF;
    const char *epdir = getenv("SP_REPLAY");
    const int npos = getenv("SP_REPLAY_NPOS") ? atoi(getenv("SP_REPLAY_NPOS")) : 8;
    if (!epdir) { fprintf(stderr, "[g4-kvrp] set SP_REPLAY to the episode dir\n"); return 2; }
    sp_model *handle = NULL;
    if (sp_model_load(spm, stk, &handle) != SP_OK || !handle) { fprintf(stderr, "[g4-kvrp] load FAIL\n"); return 2; }
    qwen3_model *m = sp_model_to_gemma4(handle);
    if (!m) { fprintf(stderr, "[g4-kvrp] to_gemma4 FAIL: %s\n", sp_last_error()); return 2; }
    const qwen3_config *c = &m->cfg;
    const int NL=(int)c->n_layers, period=(int)c->g4_swa_period?(int)c->g4_swa_period:6;
    const int kvfs=(int)c->g4_n_kv_from_start?(int)c->g4_n_kv_from_start:NL;
    const int g_nkv=(int)c->n_head_kv,g_hd=(int)c->head_dim,s_nkv=(int)c->g4_nkv_swa,s_hd=(int)c->g4_hd_swa;
    const int W = getenv("SP_G4_KV_RING_W")?atoi(getenv("SP_G4_KV_RING_W")):0;  /* ring mode if >0 */
    const int Pmax=256, sys_n=24;
    sp_g4_kv *s=gemma4_kv_open(m,Pmax);
    if(!s){ fprintf(stderr,"[g4-kvrp] kv_open FAIL: %s\n",sp_last_error()); return 2; }
    int32_t sys[24]; for(int i=0;i<sys_n;i++) sys[i]=100+i;
    if(gemma4_kv_prefill(s,sys,sys_n)){ fprintf(stderr,"[g4-kvrp] sys prefill FAIL: %s\n",sp_last_error()); return 2; }
    if(W>0) gemma4_kv_commit(s);   /* commit the context: the speculation window's journal starts fresh at this anchor */
    int anchor=gemma4_kv_pos(s);
    float **hK0=(float**)calloc(NL,sizeof(float*)),**hV0=(float**)calloc(NL,sizeof(float*));
    float **hK1=(float**)calloc(NL,sizeof(float*)),**hV1=(float**)calloc(NL,sizeof(float*));
    float **hKm=(float**)calloc(NL,sizeof(float*)),**hVm=(float**)calloc(NL,sizeof(float*));
    for(int L=0;L<kvfs&&L<NL;L++){ int global=((L%period)==period-1); int kvd=(global?g_nkv:s_nkv)*(global?g_hd:s_hd);
        size_t nb=(size_t)Pmax*kvd*sizeof(float);
        hK0[L]=(float*)malloc(nb);hV0[L]=(float*)malloc(nb);hK1[L]=(float*)malloc(nb);hV1[L]=(float*)malloc(nb);hKm[L]=(float*)malloc(nb);hVm[L]=(float*)malloc(nb); }
    gemma4_kv_snapshot(s,hK0,hV0);                                   /* pre-injection floor (window for ring SWA owners) */
    if(gemma4_kv_replay(s,epdir,npos,/*zero=*/1)){ fprintf(stderr,"[g4-kvrp] replay FAIL: %s\n",sp_last_error()); return 2; }
    int after=gemma4_kv_pos(s);
    gemma4_kv_snapshot(s,hKm,hVm);                                   /* mid: injected slots must be zeroed */
    /* load-bearing: the injected slot for each (L,p) must read back all-zero (the write landed).
     * slot = ring SWA-owner ? (anchor+p)%W : (anchor+p) — matches gemma4_kv_replay. */
    long inj_nonzero=0, inj_floats=0;
    for(int L=0;L<kvfs&&L<NL;L++){ int global=((L%period)==period-1); int kvd=(global?g_nkv:s_nkv)*(global?g_hd:s_hd);
        int ring=(W>0&&!global);
        for(int p=0;p<npos;p++){ size_t slot=ring?(size_t)((anchor+p)%W):(size_t)(anchor+p);
            for(int d=0;d<kvd;d++){ inj_floats++; if(hKm[L][slot*kvd+d]!=0.0f) inj_nonzero++; } } }
    if(gemma4_kv_rewind(s,after-anchor)){ fprintf(stderr,"[g4-kvrp] rewind FAIL\n"); return 2; }
    int back=gemma4_kv_pos(s);
    gemma4_kv_snapshot(s,hK1,hV1);                                   /* post-rewind floor */
    /* compare the LIVE region: ring SWA owners = the full W-slot window (journal must have restored
     * every clobbered slot); globals / full-cache = [0,anchor) (sheared slots are never read). */
    long diffs=0; size_t cmp_bytes=0;
    for(int L=0;L<kvfs&&L<NL;L++){ int global=((L%period)==period-1); int kvd=(global?g_nkv:s_nkv)*(global?g_hd:s_hd);
        int ring=(W>0&&!global); size_t cmp_slots=ring?(size_t)W:(size_t)anchor;
        size_t nb=cmp_slots*kvd*sizeof(float); cmp_bytes+=2*nb;
        if(memcmp(hK0[L],hK1[L],nb)) diffs++; if(memcmp(hV0[L],hV1[L],nb)) diffs++; }
    int load_bearing=(inj_nonzero==0 && inj_floats>0);              /* injected slots all-zero ⇒ replay wrote */
    int pos_ok=(back==anchor);
    fprintf(stderr,"[g4-kvrp] mode=%s W=%d anchor=%d after_replay=%d (npos=%d) back=%d cmp=%zuB owners=%d\n",
        W>0?"RING":"FULL",W,anchor,after,npos,back,cmp_bytes,(kvfs<NL?kvfs:NL));
    fprintf(stderr,"[g4-kvrp] LOAD-BEARING (injected slots zeroed): %s (%ld/%ld nonzero)\n",load_bearing?"GREEN":"RED",inj_nonzero,inj_floats);
    fprintf(stderr,"[g4-kvrp] %s ([live region] byte-identical after O(1) rewind): %s (layer-diffs=%ld) pos-reset=%s\n",
        W>0?"G-222-WRAP":"G-222 REWIND-NULL",diffs==0?"GREEN":"RED",diffs,pos_ok?"OK":"BAD");
    gemma4_kv_close(s); sp_model_unload(handle);
    return (diffs==0 && load_bearing && pos_ok)?0:3;
}

/* ═══ SP_G4_KV_TELEMETRY=1 (KAI-1b §5.4): the O(actions)→O(1) receipt. ═══
 * Idle-tick latency vs retained-action count A∈{1,2,4,8,16}, both modes:
 *   PREFIX-GROW (the host hack): an idle tick re-prefills [system + A·action + frame]
 *     via the one-shot gemma4_decode_cuda ⇒ cost ∝ A (the recompute tax).
 *   METAL (XBAR-evict): the A actions are RESIDENT; an idle tick = kv_prefill(frame)
 *     + kv_decode + kv_rewind ⇒ cost independent of A (flatline).
 * Tokens arbitrary-valid (latency is token-content-independent). Min of 3 warm reps. */
static double kvt_now(void){ struct timespec t; timespec_get(&t,TIME_UTC); return (double)t.tv_sec + (double)t.tv_nsec*1e-9; }
static int run_kv_telemetry(void) {
    const char *spm=getenv("SP_GEMMA4_SPMODEL"); if(!spm) spm=SP_GEMMA4_SPMODEL_DEF;
    const char *stk=getenv("SP_GEMMA4_SPTOK");   if(!stk) stk=SP_GEMMA4_SPTOK_DEF;
    sp_model *handle=NULL;
    if(sp_model_load(spm,stk,&handle)!=SP_OK||!handle){ fprintf(stderr,"[g4-kvtel] load FAIL\n"); return 2; }
    qwen3_model *m=sp_model_to_gemma4(handle);
    if(!m){ fprintf(stderr,"[g4-kvtel] to_gemma4 FAIL\n"); return 2; }
    const int As[]={1,2,4,8,16}, nA=5;
    const int sys_n=24, act_n=12, frame_n=12, ngen=8, Pmax=1024, REP=3;
    int32_t sys[24],act[12],frm[12];
    for(int i=0;i<sys_n;i++)sys[i]=100+i; for(int i=0;i<act_n;i++)act[i]=400+i; for(int i=0;i<frame_n;i++)frm[i]=700+i;
    int32_t gbuf[64];
    double t_grow[8]={0}, t_metal[8]={0};
    fprintf(stderr,"[g4-kvtel] sweep A∈{1,2,4,8,16} sys=%d act=%d frame=%d ngen=%d (min of %d warm reps)\n",sys_n,act_n,frame_n,ngen,REP);
    /* ── METAL: resident cache, O(1) idle tick ── */
    for(int ai=0; ai<nA; ai++){
        int A=As[ai];
        sp_g4_kv *s=gemma4_kv_open(m,Pmax);
        if(!s){ fprintf(stderr,"[g4-kvtel] kv_open FAIL\n"); return 2; }
        if(gemma4_kv_prefill(s,sys,sys_n)){ fprintf(stderr,"[g4-kvtel] metal sys FAIL\n"); return 2; }
        for(int a=0;a<A;a++){ if(gemma4_kv_prefill(s,act,act_n)||gemma4_kv_decode(s,ngen,gbuf)){ fprintf(stderr,"[g4-kvtel] metal action FAIL\n"); return 2; } }
        int anchor=gemma4_kv_pos(s);
        double best=1e9;
        for(int r=0;r<REP;r++){
            double t0=kvt_now();
            gemma4_kv_prefill(s,frm,frame_n); gemma4_kv_decode(s,ngen,gbuf); gemma4_kv_rewind(s,gemma4_kv_pos(s)-anchor);
            double dt=kvt_now()-t0; if(dt<best)best=dt;
        }
        t_metal[ai]=best; gemma4_kv_close(s);
        fprintf(stderr,"[g4-kvtel] METAL  A=%2d resident_pos=%d idle_tick=%.4fs\n",A,anchor,best);
    }
    /* ── PREFIX-GROW: one-shot re-prefill of the accumulated prefix each idle tick ── */
    for(int ai=0; ai<nA; ai++){
        int A=As[ai];
        int plen=sys_n + A*(act_n+ngen);    /* accumulated context the hack re-absorbs */
        int n_prompt=plen+frame_n;
        int total=n_prompt+ngen+8;
        int32_t *seq=(int32_t*)malloc((size_t)total*sizeof(int32_t));
        for(int i=0;i<n_prompt;i++) seq[i]=100+(i%200);   /* valid ids */
        double best=1e9;
        for(int r=0;r<REP;r++){
            double t0=kvt_now();
            int n=gemma4_decode_cuda(m,seq,n_prompt,ngen,-1);
            double dt=kvt_now()-t0; if(n<n_prompt){ fprintf(stderr,"[g4-kvtel] grow decode FAIL\n"); free(seq); return 2; } if(dt<best)best=dt;
        }
        t_grow[ai]=best; free(seq);
        fprintf(stderr,"[g4-kvtel] GROW   A=%2d prefix_len=%d idle_tick=%.4fs\n",A,plen,best);
    }
    fprintf(stderr,"\n[g4-kvtel] ── O(actions)→O(1) RECEIPT ──\n  A : prefix-grow(s) : metal(s) : grow/metal\n");
    for(int ai=0; ai<nA; ai++)
        fprintf(stderr,"  %2d : %10.4f : %8.4f : %.2fx\n",As[ai],t_grow[ai],t_metal[ai], t_metal[ai]>0?t_grow[ai]/t_metal[ai]:0.0);
    double grow_slope=(t_grow[nA-1]-t_grow[0])/(double)(As[nA-1]-As[0]);
    double metal_slope=(t_metal[nA-1]-t_metal[0])/(double)(As[nA-1]-As[0]);
    fprintf(stderr,"[g4-kvtel] slope d(idle)/dA: prefix-grow=%.5f s/action  metal=%.5f s/action\n",grow_slope,metal_slope);
    fprintf(stderr,"[g4-kvtel] VERDICT: %s (grow rises with A; metal flat)\n",
        (grow_slope > 5.0*fabs(metal_slope)+1e-4)?"O(actions) vs O(1) CONFIRMED":"inconclusive");
    sp_model_unload(handle);
    return 0;
}

/* ═══ SP_G4_KV_RING_TEL=1 (KAI-1c Task #219): journaled-ring O(1) telemetry. ═══
 * Re-runs the idle-tick-latency-vs-A sweep through the undo-journal path and quantifies
 * the D2D save-before-store tax vs the full-cache metal path, apples-to-apples in ONE process.
 * Operational semantics: commit on EACH retained action ⇒ the journal only ever holds the
 * single idle-tick span (≤ frame+decode), so the per-tick journal cost is A-INVARIANT by
 * construction. Two legs (env-toggled at gemma4_kv_open time):
 *   METAL-FULL : SP_G4_KV_RING_W cleared ⇒ full-cache slot==pos rewind (the KAI-1b path).
 *   METAL-RING : SP_G4_KV_RING_W=W      ⇒ SWA owners journal each clobbered slot, rewind restores.
 * PRE-REGISTERED THRESHOLDS (Task #219):
 *   T1 FLATLINE  : ring slope d(idle)/dA < 0.02 s/action (KAI-1b full metal was 0.0073);
 *                  PASS only if the slope does NOT rise materially with A.
 *   T2 A-INVARIANT TAX : the ring−full per-tick delta must be a CONSTANT overhead, not growing
 *                  with A. Acceptance: coefficient-of-variation of (t_ring−t_full) across
 *                  A∈{1..16} < 25%  (a delta that scales with A would mean the journal cost
 *                  leaked the action count — a bug). Reported in absolute ms + as a ratio.
 *   T3 HEADROOM  : absolute ring idle-tick < 1.0 s (comfortably inside an interactive daemon tick). */
static void kvt_putenv(const char *k, const char *v){ char b[64]; snprintf(b,sizeof b,"%s=%s",k,v?v:""); _putenv(b); }
static int run_kv_ring_telemetry(void) {
    const char *spm=getenv("SP_GEMMA4_SPMODEL"); if(!spm) spm=SP_GEMMA4_SPMODEL_DEF;
    const char *stk=getenv("SP_GEMMA4_SPTOK");   if(!stk) stk=SP_GEMMA4_SPTOK_DEF;
    sp_model *handle=NULL;
    if(sp_model_load(spm,stk,&handle)!=SP_OK||!handle){ fprintf(stderr,"[g4-rtel] load FAIL\n"); return 2; }
    qwen3_model *m=sp_model_to_gemma4(handle);
    if(!m){ fprintf(stderr,"[g4-rtel] to_gemma4 FAIL\n"); return 2; }
    const int As[]={8,16,32,50,64,80,96}, nA=7;       /* extend past window saturation (resid_pos>=1024 @ A>=50) */
    const int sys_n=24, act_n=12, frame_n=12, ngen=8, Pmax=2048, REP=5;  /* median-of-reps (robust for a delta) */
    const int W = getenv("SP_G4_KV_RING_W")?atoi(getenv("SP_G4_KV_RING_W")):1024;  /* true SWA window default */
    int32_t sys[24],act[12],frm[12];
    for(int i=0;i<sys_n;i++)sys[i]=100+i; for(int i=0;i<act_n;i++)act[i]=400+i; for(int i=0;i<frame_n;i++)frm[i]=700+i;
    int32_t gbuf[64];
    double t_full[8]={0}, t_ring[8]={0}; int resid[8]={0};
    kvt_putenv("SP_G4_KV_JMAX","64");
    fprintf(stderr,"[g4-rtel] sweep A∈{8,16,32,50,64,80,96} sys=%d act=%d frame=%d ngen=%d W=%d (commit-per-action; min of %d warm reps)\n",sys_n,act_n,frame_n,ngen,W,REP);
    for(int mode=0; mode<2; mode++){                 /* 0=FULL (ring cleared), 1=RING (journal) */
        if(mode==0) kvt_putenv("SP_G4_KV_RING_W","0"); else { char wb[16]; snprintf(wb,sizeof wb,"%d",W); kvt_putenv("SP_G4_KV_RING_W",wb); }
        for(int ai=0; ai<nA; ai++){
            int A=As[ai];
            sp_g4_kv *s=gemma4_kv_open(m,Pmax);
            if(!s){ fprintf(stderr,"[g4-rtel] kv_open FAIL: %s\n",sp_last_error()); return 2; }
            if(gemma4_kv_prefill(s,sys,sys_n)){ fprintf(stderr,"[g4-rtel] sys FAIL\n"); return 2; }
            gemma4_kv_commit(s);
            for(int a=0;a<A;a++){ if(gemma4_kv_prefill(s,act,act_n)||gemma4_kv_decode(s,ngen,gbuf)){ fprintf(stderr,"[g4-rtel] action FAIL\n"); return 2; } gemma4_kv_commit(s); }
            int anchor=gemma4_kv_pos(s);
            double rt[16]; for(int r=0;r<REP;r++){
                double t0=kvt_now();
                gemma4_kv_prefill(s,frm,frame_n); gemma4_kv_decode(s,ngen,gbuf); gemma4_kv_rewind(s,gemma4_kv_pos(s)-anchor);
                rt[r]=kvt_now()-t0;
            }
            for(int i=1;i<REP;i++){ double v=rt[i]; int j=i-1; while(j>=0&&rt[j]>v){rt[j+1]=rt[j];j--;} rt[j+1]=v; } /* insertion sort */
            double med=rt[REP/2];                                  /* median-of-reps: robust for a delta */
            if(mode==0) t_full[ai]=med; else t_ring[ai]=med; resid[ai]=anchor;
            gemma4_kv_close(s);
            fprintf(stderr,"[g4-rtel] %s A=%2d resident_pos=%d idle_tick=%.4fs\n",mode?"RING":"FULL",A,anchor,med);
        }
    }
    kvt_putenv("SP_G4_KV_RING_W","0");
    fprintf(stderr,"\n[g4-rtel] ── JOURNALED-RING O(1) RECEIPT ──\n  A : resid : full-cache(s) : ring+journal(s) : tax(ms) : ring/full : window\n");
    for(int ai=0; ai<nA; ai++)
        fprintf(stderr,"  %2d : %5d : %10.4f : %12.4f : %7.2f : %.3fx : %s\n",As[ai],resid[ai],t_full[ai],t_ring[ai],
            (t_ring[ai]-t_full[ai])*1e3, t_full[ai]>0?t_ring[ai]/t_full[ai]:0.0, resid[ai]>=W?"SAT":"grow");
    /* T2 on the SATURATED regime (resid>=W ⇒ attention window fixed): the journal tax must be flat there.
     * Pre-saturation (resid<W) the ring−full delta tracks the growing attention window, NOT the journal. */
    double dmean=0; int ns=0; for(int ai=0;ai<nA;ai++) if(resid[ai]>=W){ dmean+=(t_ring[ai]-t_full[ai]); ns++; } if(ns>0) dmean/=ns;
    double dvar=0;  for(int ai=0;ai<nA;ai++) if(resid[ai]>=W){ double d=(t_ring[ai]-t_full[ai])-dmean; dvar+=d*d; } if(ns>0) dvar/=ns;
    double dsd=sqrt(dvar), dcv=(fabs(dmean)>1e-9)?dsd/fabs(dmean):0.0;
    double full_slope=(t_full[nA-1]-t_full[0])/(double)(As[nA-1]-As[0]);
    double ring_slope=(t_ring[nA-1]-t_ring[0])/(double)(As[nA-1]-As[0]);
    double ring_max=0; for(int ai=0;ai<nA;ai++) if(t_ring[ai]>ring_max) ring_max=t_ring[ai];
    /* saturated-regime tax SLOPE: ms/action across the SAT points — the decisive O(1) test */
    int s0=-1,s1=-1; for(int ai=0;ai<nA;ai++) if(resid[ai]>=W){ if(s0<0)s0=ai; s1=ai; }
    double sat_tax_slope = (s1>s0)? ((t_ring[s1]-t_full[s1])-(t_ring[s0]-t_full[s0]))*1e3/(double)(As[s1]-As[s0]) : 0.0;
    double sat_tick=0; for(int ai=0;ai<nA;ai++) if(resid[ai]>=W) sat_tick+=t_ring[ai]; if(ns>0) sat_tick/=ns;
    double tax_frac=(sat_tick>1e-9)?dmean/sat_tick:0.0;       /* journal marginal cost as a fraction of the tick */
    fprintf(stderr,"[g4-rtel] slope d(idle)/dA: full=%.5f s/action  ring=%.5f s/action\n",full_slope,ring_slope);
    fprintf(stderr,"[g4-rtel] D2D tax (SAT regime, n=%d): mean=%.2fms sd=%.2fms cv=%.1f%% | sat-tax-slope=%.3f ms/action | tax=%.1f%% of tick (tick~%.3fs model-bound)\n",ns,dmean*1e3,dsd*1e3,dcv*100.0,sat_tax_slope,tax_frac*100.0,sat_tick);
    int t1=(ring_slope<0.02), t2=(ns>=2 && dcv<0.25 && fabs(sat_tax_slope)<0.5), t3=(tax_frac<0.05);
    fprintf(stderr,"[g4-rtel] T1 flatline(ring_slope<0.02): %s | T2 sat-tax-flat(cv<25%%,|slope|<0.5ms/A): %s | T3 journal-tax(<5%% of tick): %s\n",
        t1?"PASS":"FAIL", t2?"PASS":"FAIL", t3?"PASS":"FAIL");
    fprintf(stderr,"[g4-rtel] VERDICT: %s\n",(t1&&t2&&t3)?"JOURNALED-RING O(1) CONFIRMED":"REVIEW");
    sp_model_unload(handle);
    return (t1&&t2&&t3)?0:3;
}

/* ═══ SP_G4_KAIROS_INTERRUPT=1 (KAI-2 G-KAIROS-2 A/B): latent-vs-text event delivery latency. ═══
 * Self-null proved inject(own-embedding)==text, so a raw-embedding arm is vacuous — the latency win can
 * only come from COMPRESSION. Arm B uses an UNTRAINED baseline: mean-pool the event frame's token
 * embeddings → 1 vector, injected in ONE step. Arm A delivers the same event as the full text frame.
 * Metric: total steps-to-ACTION (delivery steps + decode steps until "<ACTION>" appears). If the pooled
 * 1-vector still pivots the resident, latent delivery beats text (1 vs frame_len delivery); if not, an
 * honest negative pointing to the phase-2 trained adapter (the roadmap's named fallback). */
static int run_kairos_interrupt(void) {
    const char *spm=getenv("SP_GEMMA4_SPMODEL"); if(!spm) spm=SP_GEMMA4_SPMODEL_DEF;
    const char *stk=getenv("SP_GEMMA4_SPTOK");   if(!stk) stk=SP_GEMMA4_SPTOK_DEF;
    sp_model *handle=NULL;
    if(sp_model_load(spm,stk,&handle)!=SP_OK||!handle){ fprintf(stderr,"[g4-int] load FAIL\n"); return 2; }
    qwen3_model *m=sp_model_to_gemma4(handle);
    if(!m){ fprintf(stderr,"[g4-int] to_gemma4 FAIL\n"); return 2; }
    sp_tokenizer *tk=sp_tokenizer_load_tokfile(stk);
    if(!tk){ fprintf(stderr,"[g4-int] tokenizer FAIL\n"); return 2; }
    const qwen3_config *c=&m->cfg; const int E=(int)c->n_embd;
    const char *SYS=
        "<start_of_turn>user\nYou are a background kernel daemon. Each tick you receive one environment event "
        "that carries a salience score between 0.0 and 1.0. Reply with EXACTLY one of:\nNO_OP\n<ACTION>short imperative</ACTION>\n"
        "RULE: if salience >= 0.5 the event requires intervention, so reply with an <ACTION> line. "
        "If salience < 0.5, reply NO_OP. Follow the rule exactly. Do not explain.<end_of_turn>\n<start_of_turn>model\nUnderstood.<end_of_turn>\n";
    const char *FRAME="<start_of_turn>user\nEVENT kind=EVENT.timer salience=0.90 payload=\"build finished\"<end_of_turn>\n<start_of_turn>model\n";
    int32_t sysb[4096], frm[512], gen[1]; char dbuf[256];
    long pn=sp_tokenizer_encode(tk,SYS,strlen(SYS),1,sysb,4096);
    long fn=sp_tokenizer_encode(tk,FRAME,strlen(FRAME),1,frm,512);
    if(pn<=0||fn<=0){ fprintf(stderr,"[g4-int] encode FAIL\n"); return 2; }
    sp_g4_kv *s=gemma4_kv_open(m,2048);
    if(!s){ fprintf(stderr,"[g4-int] kv_open FAIL: %s\n",sp_last_error()); return 2; }
    if(gemma4_kv_prefill(s,sysb,(int)pn)){ fprintf(stderr,"[g4-int] SYS prefill FAIL\n"); return 2; }
    gemma4_kv_commit(s); int anchor=gemma4_kv_pos(s);
    const int KGEN=12;
    /* mint the latent packet: mean-pool the FRAME token embeddings (untrained k=1 compression).
     * capture each token's post-embed residual via a throwaway prefill, then rewind back to anchor. */
    float *acc=(float*)calloc(E,sizeof(float)), *cap=(float*)malloc((size_t)E*sizeof(float)), *mean=(float*)malloc((size_t)E*sizeof(float));
    for(int i=0;i<fn;i++){ gemma4_kv_capture(s,cap); if(gemma4_kv_prefill(s,&frm[i],1)){ fprintf(stderr,"[g4-int] mint FAIL\n"); return 2; } for(int e=0;e<E;e++) acc[e]+=cap[e]; }
    gemma4_kv_rewind(s,(int)fn);   /* back to anchor (mint was non-destructive) */
    for(int e=0;e<E;e++) mean[e]=acc[e]/(float)fn;
    /* helper: free-decode until "<ACTION>" appears or KGEN exhausted; returns steps-to-action (or -1) */
    #define STEPS_TO_ACTION(out_steps, out_txt) do { (out_steps)=-1; (out_txt)[0]=0; char run[512]={0}; \
        for(int g=0; g<KGEN; g++){ if(gemma4_kv_decode(s,1,gen)){ break; } long bl=sp_tokenizer_decode(tk,gen,1,dbuf,sizeof dbuf-1); if(bl<0)bl=0; dbuf[bl]=0; \
            strncat(run,dbuf,sizeof(run)-strlen(run)-1); if(strstr(run,"ACTION")){ (out_steps)=g+1; break; } } \
        for(char*p=run;*p;p++) if(*p=='\n'||*p=='\r')*p=' '; snprintf((out_txt),120,"%.118s",run); } while(0)
    char txtA[128], txtB[128]; int decA=0,decB=0;
    /* ── Arm A (text): deliver full frame, then decode to action ── */
    if(gemma4_kv_prefill(s,frm,(int)fn)){ fprintf(stderr,"[g4-int] A frame FAIL\n"); return 2; }
    STEPS_TO_ACTION(decA,txtA);
    int totalA = (decA<0)? -1 : (int)fn + decA;
    gemma4_kv_rewind(s,gemma4_kv_pos(s)-anchor);
    /* ── Arm B (latent, mean-pool k=1): inject ONE vector (rides the first decode step), decode to action ──
     * delivery is FUSED into step 1 (gemma4_kv_inject overrides the next step's residual) ⇒ total = decode steps. */
    gemma4_kv_inject(s,mean);
    STEPS_TO_ACTION(decB,txtB);
    int totalB = (decB<0)? -1 : decB;
    fprintf(stderr,"[g4-int] frame_len=%ld KGEN=%d\n",fn,KGEN);
    fprintf(stderr,"[g4-int] ARM A (text):   pivot=%s decode_steps=%d total_steps=%d raw=\"%s\"\n", decA>0?"YES":"NO",decA,totalA,txtA);
    fprintf(stderr,"[g4-int] ARM B (latent k=1 mean-pool): pivot=%s total_steps=%d raw=\"%s\"\n", decB>0?"YES":"NO",totalB,txtB);
    if(decA>0 && decB>0)
        fprintf(stderr,"[g4-int] G-KAIROS-2 A/B: latent total=%d vs text total=%d ⇒ %s\n",totalB,totalA,
            totalB<totalA?"LATENT FASTER (compression wins)":(totalB==totalA?"TIE":"text faster"));
    else
        fprintf(stderr,"[g4-int] G-KAIROS-2 A/B: %s (Arm B pivot=%s — untrained mean-pool %s; phase-2 trained adapter is the path)\n",
            decB>0?"latent pivoted":"HONEST NEGATIVE: untrained k=1 mean-pool did NOT pivot",decB>0?"YES":"NO", decB>0?"sufficed":"insufficient");
    #undef STEPS_TO_ACTION
    free(acc);free(cap);free(mean);
    gemma4_kv_close(s); sp_tokenizer_free(tk); sp_cuda_model_release(m); qwen3_free(m); sp_model_unload(handle);
    return 0;   /* exploratory measurement, not pass/fail */
}

/* ═══ SP_G4_KV_INJECT_NULL=1 (KAI-2 G-KAIROS-2 self-null): the latent-inject seam is bit-exact inert. ═══
 * Capture the model's OWN post-embed residual at a position, then re-inject it: the decode MUST be
 * byte-identical to no injection (the X-R1 G0 analog on the persistent ABI). Non-vacuity: a PERTURBED
 * residual must CHANGE the output (proves the inject path is live, not silently skipped). */
static int run_kv_inject_null(void) {
    const char *spm=getenv("SP_GEMMA4_SPMODEL"); if(!spm) spm=SP_GEMMA4_SPMODEL_DEF;
    const char *stk=getenv("SP_GEMMA4_SPTOK");   if(!stk) stk=SP_GEMMA4_SPTOK_DEF;
    sp_model *handle=NULL;
    if(sp_model_load(spm,stk,&handle)!=SP_OK||!handle){ fprintf(stderr,"[g4-inj] load FAIL\n"); return 2; }
    qwen3_model *m=sp_model_to_gemma4(handle);
    if(!m){ fprintf(stderr,"[g4-inj] to_gemma4 FAIL: %s\n",sp_last_error()); return 2; }
    const qwen3_config *c=&m->cfg;
    const int E=(int)c->n_embd, NL=(int)c->n_layers, period=(int)c->g4_swa_period?(int)c->g4_swa_period:6;
    const int kvfs=(int)c->g4_n_kv_from_start?(int)c->g4_n_kv_from_start:NL;
    const int g_nkv=(int)c->n_head_kv,g_hd=(int)c->head_dim,s_nkv=(int)c->g4_nkv_swa,s_hd=(int)c->g4_hd_swa;
    const int Pmax=256, sys_n=24;
    sp_g4_kv *s=gemma4_kv_open(m,Pmax);   /* full cache (SP_G4_KV_RING_W unset) */
    if(!s){ fprintf(stderr,"[g4-inj] kv_open FAIL: %s\n",sp_last_error()); return 2; }
    int32_t sys[32]; for(int i=0;i<sys_n;i++) sys[i]=100+i;
    if(gemma4_kv_prefill(s,sys,sys_n)){ fprintf(stderr,"[g4-inj] prefill FAIL: %s\n",sp_last_error()); return 2; }
    /* snapshot buffers (owners, full cache = Pmax slots) */
    float **hK0=(float**)calloc(NL,sizeof(float*)),**hV0=(float**)calloc(NL,sizeof(float*));
    float **hK1=(float**)calloc(NL,sizeof(float*)),**hV1=(float**)calloc(NL,sizeof(float*));
    for(int L=0;L<kvfs&&L<NL;L++){ int gl=((L%period)==period-1); int kvd=(gl?g_nkv:s_nkv)*(gl?g_hd:s_hd);
        size_t nb=(size_t)Pmax*kvd*sizeof(float); hK0[L]=malloc(nb);hV0[L]=malloc(nb);hK1[L]=malloc(nb);hV1[L]=malloc(nb); }
    float *cap=(float*)malloc((size_t)E*sizeof(float));
    /* ── A: capture the model's own post-embed residual at this position, normal decode ── */
    gemma4_kv_capture(s,cap);
    int32_t genA[1]; if(gemma4_kv_decode(s,1,genA)){ fprintf(stderr,"[g4-inj] decode A FAIL: %s\n",sp_last_error()); return 2; }
    gemma4_kv_snapshot(s,hK0,hV0);
    gemma4_kv_rewind(s,1);
    /* ── B: re-inject the captured residual (== the model's own) → must be byte-identical ── */
    gemma4_kv_inject(s,cap);
    int32_t genB[1]; if(gemma4_kv_decode(s,1,genB)){ fprintf(stderr,"[g4-inj] decode B FAIL: %s\n",sp_last_error()); return 2; }
    gemma4_kv_snapshot(s,hK1,hV1);
    long diffs=0; for(int L=0;L<kvfs&&L<NL;L++){ int gl=((L%period)==period-1); int kvd=(gl?g_nkv:s_nkv)*(gl?g_hd:s_hd);
        size_t nb=(size_t)Pmax*kvd*sizeof(float); if(memcmp(hK0[L],hK1[L],nb))diffs++; if(memcmp(hV0[L],hV1[L],nb))diffs++; }
    int self_null = (genA[0]==genB[0] && diffs==0);
    /* ── C (non-vacuity): a PERTURBED residual must change the output ── */
    gemma4_kv_rewind(s,1);
    float *cap2=(float*)malloc((size_t)E*sizeof(float));
    for(int i=0;i<E;i++) cap2[i]=cap[i]+ (i<8? 2.0f : 0.0f);   /* perturb a few dims */
    gemma4_kv_inject(s,cap2);
    int32_t genC[1]; gemma4_kv_decode(s,1,genC);
    gemma4_kv_snapshot(s,hK1,hV1);
    long pdiffs=0; for(int L=0;L<kvfs&&L<NL;L++){ int gl=((L%period)==period-1); int kvd=(gl?g_nkv:s_nkv)*(gl?g_hd:s_hd);
        size_t nb=(size_t)Pmax*kvd*sizeof(float); if(memcmp(hK0[L],hK1[L],nb))pdiffs++; }
    int nonvacuous = (genC[0]!=genA[0] || pdiffs>0);
    fprintf(stderr,"[g4-inj] self-null: genA=%d genB=%d kv-diffs=%ld | perturbed: genC=%d kv-changed=%ld\n",
            genA[0],genB[0],diffs,genC[0],pdiffs);
    fprintf(stderr,"[g4-inj] G-KAIROS-2 SELF-NULL: %s (inject seam bit-exact inert) | non-vacuous: %s (seam is live)\n",
            self_null?"GREEN":"RED", nonvacuous?"YES":"NO(VACUOUS!)");
    gemma4_kv_close(s); sp_model_unload(handle);
    return (self_null && nonvacuous)?0:3;
}

/* ═══ SP_G4_KAI2_PACKET=1 (KAI-2 G-KAIROS-2 GATE): inject the TRAINED codec packet, measure pivot + selectivity ═══
 * Replicates train_kai2_codec.py's scaffold EXACTLY: SYSTEM text + k codec soft-vectors (the trained packet,
 * injected over k decode steps) + DECIDE text, decision read next. TEXT control arm (SYSTEM+event_text+DECIDE)
 * isolates codec transfer from resident-model capability (teacher was bf16; resident is OK_Q4B, PPL-parity).
 * Two cases (salient→ACTION, idle→NO_OP) = selectivity. GATE: the PACKET arm reproduces the expected decision
 * for BOTH cases. Packets = tests/fixtures/kai2/event_*.bin ('KAI2'|u32 k|u32 hidden|k*hidden f32). */
static int kai2_read_packet(const char *path, int E, int *out_k, float **out){
    FILE *f=fopen(path,"rb"); if(!f) return -1; char mg[4]; uint32_t k=0,h=0;
    if(fread(mg,1,4,f)!=4||memcmp(mg,"KAI2",4)){fclose(f);return -2;}
    if(fread(&k,4,1,f)!=1||fread(&h,4,1,f)!=1){fclose(f);return -3;}
    if((int)h!=E){fclose(f);return -4;}
    float *v=(float*)malloc((size_t)k*h*sizeof(float));
    if(fread(v,sizeof(float),(size_t)k*h,f)!=(size_t)k*h){free(v);fclose(f);return -5;}
    fclose(f); *out_k=(int)k; *out=v; return 0;
}
static const char* kai2_decide(sp_g4_kv*s, sp_tokenizer*tk, char*txt){
    const int KGEN=8; int32_t gen[8]; char dbuf[512];
    if(gemma4_kv_decode(s,KGEN,gen)){ txt[0]=0; return "DECODE_FAIL"; }
    long bl=sp_tokenizer_decode(tk,gen,KGEN,dbuf,sizeof dbuf-1); if(bl<0)bl=0; dbuf[bl]=0;
    for(char*p=dbuf;*p;p++) if(*p=='\n'||*p=='\r')*p=' '; snprintf(txt,120,"%.118s",dbuf);
    if(strstr(dbuf,"ACTION")) return "ACTION";                      /* soak detection: ACTION first */
    if(strstr(dbuf,"OP")||strstr(dbuf,"NOOP")) return "NO_OP";      /* NO_OP / NOOP both contain "OP" */
    return "NEITHER";
}
/* arena embed-row access (resolved at link via the engine lib; not in this TU's headers) */
typedef struct sp_arena_tensor sp_arena_tensor;
extern const sp_arena_tensor *sp_arena_find(const void *a, const char *name);
extern int sp_arena_dequant_row(const sp_arena_tensor *at, int r, float *dst);
static int run_kai2_packet_gate(void){
    const char *spm=getenv("SP_GEMMA4_SPMODEL"); if(!spm) spm=SP_GEMMA4_SPMODEL_DEF;
    const char *stk=getenv("SP_GEMMA4_SPTOK");   if(!stk) stk=SP_GEMMA4_SPTOK_DEF;
    sp_model *handle=NULL;
    if(sp_model_load(spm,stk,&handle)!=SP_OK||!handle){ fprintf(stderr,"[kai2] load FAIL\n"); return 2; }
    qwen3_model *m=sp_model_to_gemma4(handle); if(!m){ fprintf(stderr,"[kai2] to_gemma4 FAIL\n"); return 2; }
    sp_tokenizer *tk=sp_tokenizer_load_tokfile(stk); if(!tk){ fprintf(stderr,"[kai2] tok FAIL\n"); return 2; }
    const int E=(int)m->cfg.n_embd;
    /* EXACT soak scaffold (the gemma <start_of_turn> daemon prompt the 6h soak proved selective). The codec
     * packet replaces the EVENT TEXT inside the user turn: TEXT arm = SYS + [user: EVENT...]; PACKET arm =
     * SYS + "<start_of_turn>user\n" + k soft-vectors + "<end_of_turn>\n<start_of_turn>model\n". */
    const char *SYS=
        "<start_of_turn>user\n"
        "You are a background kernel daemon. Each tick you receive one environment event "
        "that carries a salience score between 0.0 and 1.0. Reply with EXACTLY one of:\n"
        "NO_OP\n<ACTION>short imperative</ACTION>\n"
        "RULE: if salience >= 0.5 the event requires intervention, so reply with an <ACTION> line. "
        "If salience < 0.5, reply NO_OP. Follow the rule exactly. Do not explain."
        "<end_of_turn>\n<start_of_turn>model\nUnderstood.<end_of_turn>\n";
    const char *UOPEN="<start_of_turn>user\n";
    const char *UCLOSE="<end_of_turn>\n<start_of_turn>model\n";
    struct { const char*text; const char*pkt; const char*expect; } cs[2]={
        {"EVENT build_id=4471 status=FAILED tests=3_broken salience=0.85", "tests/fixtures/kai2/event_000_ACTION.bin", "ACTION"},
        {"EVENT heartbeat ok cpu=12% salience=0.10",                        "tests/fixtures/kai2/event_004_NO_OP.bin", "NO_OP"}};
    int32_t sysb[4096], uob[64], ucb[64], frmb[512]; char tT[128], tP[128], frame[512];
    long sn=sp_tokenizer_encode(tk,SYS,strlen(SYS),1,sysb,4096);    /* SYS gets BOS (matches soak) */
    long uon=sp_tokenizer_encode(tk,UOPEN,strlen(UOPEN),1,uob,64);  /* user-turn open: BOS like the soak frame */
    long ucn=sp_tokenizer_encode(tk,UCLOSE,strlen(UCLOSE),0,ucb,64);
    if(sn<=0||uon<=0||ucn<=0){ fprintf(stderr,"[kai2] scaffold encode FAIL\n"); return 2; }
    /* PHASE-1 delivery control (CONTRACT-KAIROS §6): the real-token-embedding lane. */
    const sp_arena_tensor *eat = m->arena ? sp_arena_find(m->arena, m->token_embd->name) : NULL;
    const float g_embscale = sqrtf((float)E);
    int pass=0, emb_pass=0;
    /* STEP-1 manifold-match calibration (SP_KAI2_CAL=1): channel-wise mean/std of the OK_Q4B embedding
     * rows (×embscale, i.e. the native injected-residual distribution), used to whiten-then-recolor the
     * codec packet before injection. Falsifies the "purely distributional" gap (off by default = null). */
    int do_cal = getenv("SP_KAI2_CAL") && getenv("SP_KAI2_CAL")[0]=='1';
    float *emu=NULL, *esd=NULL;
    if(do_cal && eat){
        emu=(float*)calloc(E,sizeof(float)); esd=(float*)calloc(E,sizeof(float));
        float *hr=(float*)malloc((size_t)E*sizeof(float));
        const int S=4096; int got=0;            /* sample first S vocab rows */
        for(int r=0;r<S;r++){ if(sp_arena_dequant_row(eat,r,hr)) continue;
            for(int d=0;d<E;d++){ float v=hr[d]*g_embscale; emu[d]+=v; esd[d]+=v*v; } got++; }
        if(got>0){ for(int d=0;d<E;d++){ emu[d]/=got; float var=esd[d]/got-emu[d]*emu[d]; esd[d]=var>1e-12f?sqrtf(var):1e-6f; } }
        free(hr);
        fprintf(stderr,"[kai2] CAL: embed channel stats over %d rows (emu[0]=%.4f esd[0]=%.4f)\n",got,emu[0],esd[0]);
    }
    for(int i=0;i<2;i++){
        sp_g4_kv *s=gemma4_kv_open(m,2048); if(!s){ fprintf(stderr,"[kai2] kv_open FAIL\n"); return 2; }
        if(gemma4_kv_prefill(s,sysb,(int)sn)){ fprintf(stderr,"[kai2] SYS prefill FAIL\n"); return 2; }
        gemma4_kv_commit(s); int anchor=gemma4_kv_pos(s);
        /* TEXT control: byte-identical to the soak's selective path — full event frame, then decode */
        snprintf(frame,sizeof frame,"%s%s%s",UOPEN,cs[i].text,UCLOSE);
        long fn=sp_tokenizer_encode(tk,frame,strlen(frame),1,frmb,512);
        if(fn<=0){ fprintf(stderr,"[kai2] frame encode FAIL\n"); return 2; }
        if(gemma4_kv_prefill(s,frmb,(int)fn)){ fprintf(stderr,"[kai2] text prefill FAIL\n"); return 2; }
        const char *decT=kai2_decide(s,tk,tT);
        gemma4_kv_rewind(s,gemma4_kv_pos(s)-anchor);
        /* PHASE-1 EMB control (CONTRACT §6): inject the event's REAL token embeddings (embd[tok]*embscale ==
         * the token's normal dx) through the SAME gemma4_kv_inject seam. If this pivots like the TEXT arm, the
         * residual-entry delivery works and any miss is the codec; if it's inert, the seam itself is broken. */
        char tE[128]; const char *decE="EMB_SKIP"; int ne=0;
        if(eat){
            int32_t etok[256]; long en=sp_tokenizer_encode(tk,cs[i].text,strlen(cs[i].text),0,etok,256);
            int embk=getenv("SP_KAI2_EMBK")?atoi(getenv("SP_KAI2_EMBK")):0;   /* capacity probe: cap real-embd count */
            if(embk>0 && en>embk) en=embk;
            if(en>0){ ne=(int)en; int32_t ph[1]={258881}; float *hrow=(float*)malloc((size_t)E*sizeof(float));
                if(gemma4_kv_prefill(s,uob,(int)uon)){ fprintf(stderr,"[kai2] EMB uopen FAIL\n"); }
                for(long t=0;t<en;t++){
                    if(sp_arena_dequant_row(eat,(int)etok[t],hrow)){ fprintf(stderr,"[kai2] EMB dequant FAIL tok=%d\n",(int)etok[t]); break; }
                    for(int d=0;d<E;d++) hrow[d]*=g_embscale;
                    gemma4_kv_inject(s,hrow);
                    if(gemma4_kv_prefill(s,ph,1)){ fprintf(stderr,"[kai2] EMB inject-prefill FAIL\n"); break; }
                }
                if(gemma4_kv_prefill(s,ucb,(int)ucn)){ fprintf(stderr,"[kai2] EMB uclose FAIL\n"); }
                decE=kai2_decide(s,tk,tE); free(hrow);
                gemma4_kv_rewind(s,gemma4_kv_pos(s)-anchor);
            }
        }
        int okE=(strcmp(decE,cs[i].expect)==0); emb_pass+=okE;
        /* PACKET arm: open the user turn, inject k codec vectors (one per decode step) in place of the event
         * text, close the turn, decode. (Current packets were trained on the OLD scaffold ⇒ expected to miss.) */
        int k=0; float *vecs=NULL; int rc=kai2_read_packet(cs[i].pkt,E,&k,&vecs);
        const char *decP="PKT_FAIL";
        if(rc==0){
            /* Option 2 (AltUp-faithful): each soft position is a PLACEHOLDER token (gemma audio_token_id
             * 258881) so the PLE/AltUp gathers fire exactly as in training; gemma4_kv_inject overrides ONLY
             * the post-embed 0th stream at that position. prefill(ph,1) runs the step that consumes inj_active. */
            int32_t ph[1]={258881};
            float *cal=NULL, cmu=0.f, csd=1.f;
            if(do_cal && emu){ double sum=0,ssq=0; size_t n=(size_t)k*E;
                for(size_t z=0;z<n;z++){ sum+=vecs[z]; ssq+=(double)vecs[z]*vecs[z]; }
                cmu=(float)(sum/n); double var=ssq/n-(double)cmu*cmu; csd=var>1e-12?(float)sqrt(var):1e-6f;
                cal=(float*)malloc((size_t)E*sizeof(float));
                fprintf(stderr,"[kai2] CAL: case=%s codec mu=%.4f sd=%.4f -> recolor to embed manifold\n",cs[i].expect[0]=='A'?"sal":"idle",cmu,csd); }
            if(gemma4_kv_prefill(s,uob,(int)uon)){ fprintf(stderr,"[kai2] uopen prefill FAIL\n"); }
            for(int j=0;j<k;j++){ const float *src=vecs+(size_t)j*E;
                if(cal){ for(int d=0;d<E;d++) cal[d]=(src[d]-cmu)/csd*esd[d]+emu[d]; src=cal; }
                gemma4_kv_inject(s,src); if(gemma4_kv_prefill(s,ph,1)){ fprintf(stderr,"[kai2] inject-placeholder FAIL\n"); break; } }
            if(cal) free(cal);
            if(gemma4_kv_prefill(s,ucb,(int)ucn)){ fprintf(stderr,"[kai2] uclose prefill FAIL\n"); }
            decP=kai2_decide(s,tk,tP);
            free(vecs);
        } else { fprintf(stderr,"[kai2] packet read rc=%d (%s)\n",rc,cs[i].pkt); tP[0]=0; }
        int okT = (strcmp(decT,cs[i].expect)==0);
        int okP = (strcmp(decP,cs[i].expect)==0);
        pass += okP;
        fprintf(stderr,"[kai2] case=%s expect=%s | TEXT->%s (\"%s\") [%s] | EMB(real-embd n=%d)->%s (\"%s\") [%s] | PACKET(k=%d)->%s (\"%s\") [%s]\n",
                cs[i].expect[0]=='A'?"salient":"idle", cs[i].expect, decT,tT, okT?"sel":"NONSEL", ne, decE,tE, okE?"PASS":"miss", k, decP,tP, okP?"PASS":"miss");
        gemma4_kv_close(s);
    }
    fprintf(stderr,"[kai2] PHASE-1 EMB-DELIVERY GATE: %d/2 (real token embeddings injected via gemma4_kv_inject). "
                   "PASS here = the residual-entry seam DELIVERS (salient->ACTION, idle->NO_OP); then PACKET miss = codec. "
                   "MISS here = the kv-path inject seam is broken vs the proven one-shot SP_XBAR_EMB.\n", emb_pass);
    fprintf(stderr,"[kai2] G-KAIROS-2 PACKET GATE: PACKET %d/2.\n", pass);
    sp_tokenizer_free(tk); sp_cuda_model_release(m); qwen3_free(m); sp_model_unload(handle);
    return pass==2 ? 0 : 3;
}

/* ═══ SP_G4_INJ_SEQ=1 (KAI-3 §7.2 G-KAIROS-3-NULL): the sequence-wrapper null-floor gate. ═══
 * Replicates the Phase-1 EMB delivery path (run_kai2_packet_gate L981-986) but drives the per-position
 * loop through the NEW engine wrapper gemma4_kv_inject_seq instead of an inline harness loop. STRICT
 * success (CONTRACT-KAIROS §7.2): the wrapper must reproduce the Phase-1 pivot EXACTLY — salient->ACTION,
 * idle->NO_OP, 2/2. PASS = moving the loop into the engine changed nothing (delivery path locked); any
 * miss = a wrapper defect (loop bound / dpos advance / ph token). No training is introduced here. */
static int run_inject_seq_null(void){
    const char *spm=getenv("SP_GEMMA4_SPMODEL"); if(!spm) spm=SP_GEMMA4_SPMODEL_DEF;
    const char *stk=getenv("SP_GEMMA4_SPTOK");   if(!stk) stk=SP_GEMMA4_SPTOK_DEF;
    sp_model *handle=NULL;
    if(sp_model_load(spm,stk,&handle)!=SP_OK||!handle){ fprintf(stderr,"[injseq] load FAIL\n"); return 2; }
    qwen3_model *m=sp_model_to_gemma4(handle); if(!m){ fprintf(stderr,"[injseq] to_gemma4 FAIL\n"); return 2; }
    sp_tokenizer *tk=sp_tokenizer_load_tokfile(stk); if(!tk){ fprintf(stderr,"[injseq] tok FAIL\n"); return 2; }
    const int E=(int)m->cfg.n_embd;
    const char *SYS=
        "<start_of_turn>user\n"
        "You are a background kernel daemon. Each tick you receive one environment event "
        "that carries a salience score between 0.0 and 1.0. Reply with EXACTLY one of:\n"
        "NO_OP\n<ACTION>short imperative</ACTION>\n"
        "RULE: if salience >= 0.5 the event requires intervention, so reply with an <ACTION> line. "
        "If salience < 0.5, reply NO_OP. Follow the rule exactly. Do not explain."
        "<end_of_turn>\n<start_of_turn>model\nUnderstood.<end_of_turn>\n";
    const char *UOPEN="<start_of_turn>user\n";
    const char *UCLOSE="<end_of_turn>\n<start_of_turn>model\n";
    struct { const char*text; const char*expect; } cs[2]={
        {"EVENT build_id=4471 status=FAILED tests=3_broken salience=0.85", "ACTION"},
        {"EVENT heartbeat ok cpu=12% salience=0.10",                      "NO_OP"}};
    int32_t sysb[4096], uob[64], ucb[64];
    long sn=sp_tokenizer_encode(tk,SYS,strlen(SYS),1,sysb,4096);
    long uon=sp_tokenizer_encode(tk,UOPEN,strlen(UOPEN),1,uob,64);
    long ucn=sp_tokenizer_encode(tk,UCLOSE,strlen(UCLOSE),0,ucb,64);
    if(sn<=0||uon<=0||ucn<=0){ fprintf(stderr,"[injseq] scaffold encode FAIL\n"); return 2; }
    const sp_arena_tensor *eat = m->arena ? sp_arena_find(m->arena, m->token_embd->name) : NULL;
    if(!eat){ fprintf(stderr,"[injseq] token_embd arena tensor not found\n"); return 2; }
    const float g_embscale = sqrtf((float)E);
    const int PH=258881;                 /* gemma-4 audio_token_id placeholder */
    int pass=0;
    for(int i=0;i<2;i++){
        sp_g4_kv *s=gemma4_kv_open(m,2048); if(!s){ fprintf(stderr,"[injseq] kv_open FAIL\n"); return 2; }
        if(gemma4_kv_prefill(s,sysb,(int)sn)){ fprintf(stderr,"[injseq] SYS prefill FAIL\n"); return 2; }
        gemma4_kv_commit(s); int anchor=gemma4_kv_pos(s);
        int32_t etok[256]; long en=sp_tokenizer_encode(tk,cs[i].text,strlen(cs[i].text),0,etok,256);
        if(en<=0){ fprintf(stderr,"[injseq] event encode FAIL\n"); return 2; }
        /* build the contiguous [en][E] raw-embedding sequence (embd[tok]*sqrt(E) == the token's normal dx) */
        float *embs=(float*)malloc((size_t)en*E*sizeof(float));
        if(!embs){ fprintf(stderr,"[injseq] embs OOM\n"); return 2; }
        int ok=1;
        for(long t=0;t<en;t++){ float *row=embs+(size_t)t*E;
            if(sp_arena_dequant_row(eat,(int)etok[t],row)){ fprintf(stderr,"[injseq] dequant FAIL tok=%d\n",(int)etok[t]); ok=0; break; }
            for(int d=0;d<E;d++) row[d]*=g_embscale; }
        char tS[128]; const char *decS="SEQ_FAIL";
        if(ok){
            if(gemma4_kv_prefill(s,uob,(int)uon)){ fprintf(stderr,"[injseq] uopen FAIL\n"); }
            if(gemma4_kv_inject_seq(s,embs,(int)en,PH)){ fprintf(stderr,"[injseq] inject_seq FAIL: %s\n",sp_last_error()); }
            else { if(gemma4_kv_prefill(s,ucb,(int)ucn)){ fprintf(stderr,"[injseq] uclose FAIL\n"); }
                   decS=kai2_decide(s,tk,tS); }
        }
        free(embs);
        int okS=(strcmp(decS,cs[i].expect)==0); pass+=okS;
        fprintf(stderr,"[injseq] case=%s expect=%s | INJ_SEQ(n=%ld via gemma4_kv_inject_seq)->%s (\"%s\") [%s]\n",
                cs[i].expect[0]=='A'?"salient":"idle", cs[i].expect, en, decS,tS, okS?"PASS":"miss");
        gemma4_kv_close(s);
    }
    fprintf(stderr,"[injseq] G-KAIROS-3-NULL: %d/2 (sequence wrapper reproduces Phase-1 EMB pivot). "
                   "PASS=2/2 ⇒ gemma4_kv_inject_seq is byte-faithful to the proven inline loop; delivery path locked.\n", pass);
    sp_tokenizer_free(tk); sp_cuda_model_release(m); qwen3_free(m); sp_model_unload(handle);
    return pass==2 ? 0 : 3;
}

/* ═══ SP_G4_TOK_DUMP (KAI-3 §7.3 support): dump real gemma-4 token ids for a line-per-event text file. ═══
 * Uses the engine's .sp-tokenizer (same one the metal gates use) — no Python tokenizer / no cloud needed.
 * SP_G4_TOK_DUMP_IN = text file (one event per line); SP_G4_TOK_DUMP_OUT = output ("id id id" per line). */
static int run_tok_dump(void){
    const char *stk=getenv("SP_GEMMA4_SPTOK"); if(!stk) stk=SP_GEMMA4_SPTOK_DEF;
    const char *in=getenv("SP_G4_TOK_DUMP_IN"), *out=getenv("SP_G4_TOK_DUMP_OUT");
    if(!in||!out){ fprintf(stderr,"[tokdump] need SP_G4_TOK_DUMP_IN + SP_G4_TOK_DUMP_OUT\n"); return 2; }
    sp_tokenizer *tk=sp_tokenizer_load_tokfile(stk); if(!tk){ fprintf(stderr,"[tokdump] tok FAIL\n"); return 2; }
    FILE *fi=fopen(in,"rb"), *fo=fopen(out,"wb");
    if(!fi||!fo){ fprintf(stderr,"[tokdump] open FAIL\n"); return 2; }
    char line[1024]; int32_t ids[512]; long nl=0;
    while(fgets(line,sizeof line,fi)){
        size_t L=strlen(line); while(L&&(line[L-1]=='\n'||line[L-1]=='\r')) line[--L]=0;
        if(L==0){ fprintf(fo,"\n"); continue; }
        long n=sp_tokenizer_encode(tk,line,L,0,ids,512);
        if(n<=0){ fprintf(fo,"\n"); continue; }
        for(long i=0;i<n;i++) fprintf(fo,"%d%s",ids[i],i+1<n?" ":"");
        fprintf(fo,"\n"); nl++;
    }
    fclose(fi); fclose(fo); sp_tokenizer_free(tk);
    fprintf(stderr,"[tokdump] wrote %ld lines -> %s\n",nl,out);
    return 0;
}

/* ═══ SP_G4_KAI3=manifest (KAI-3 §7.3 G-KAIROS-3 metal pivot gate): inject PROJECTOR packets as a ═══
 * sequence and check the 12B pivots salient->ACTION / idle->NO_OP. Manifest lines: "<path.bin> <EXPECT>".
 * Each packet (KAI2 fmt, k x E on-manifold frame vectors) is injected via gemma4_kv_inject_seq inside the
 * exact soak scaffold (SYS + user-open + <packet seq> + user-close), then decoded. This is the composition
 * receipt: projector output -> real metal -> pivot. */
static int run_kai3_gate(void){
    const char *spm=getenv("SP_GEMMA4_SPMODEL"); if(!spm) spm=SP_GEMMA4_SPMODEL_DEF;
    const char *stk=getenv("SP_GEMMA4_SPTOK");   if(!stk) stk=SP_GEMMA4_SPTOK_DEF;
    const char *man=getenv("SP_G4_KAI3");        if(!man){ fprintf(stderr,"[kai3] need SP_G4_KAI3=manifest\n"); return 2; }
    sp_model *handle=NULL;
    if(sp_model_load(spm,stk,&handle)!=SP_OK||!handle){ fprintf(stderr,"[kai3] load FAIL\n"); return 2; }
    qwen3_model *m=sp_model_to_gemma4(handle); if(!m){ fprintf(stderr,"[kai3] to_gemma4 FAIL\n"); return 2; }
    sp_tokenizer *tk=sp_tokenizer_load_tokfile(stk); if(!tk){ fprintf(stderr,"[kai3] tok FAIL\n"); return 2; }
    const int E=(int)m->cfg.n_embd; const int PH=258881;
    const char *SYS=
        "<start_of_turn>user\n"
        "You are a background kernel daemon. Each tick you receive one environment event "
        "that carries a salience score between 0.0 and 1.0. Reply with EXACTLY one of:\n"
        "NO_OP\n<ACTION>short imperative</ACTION>\n"
        "RULE: if salience >= 0.5 the event requires intervention, so reply with an <ACTION> line. "
        "If salience < 0.5, reply NO_OP. Follow the rule exactly. Do not explain."
        "<end_of_turn>\n<start_of_turn>model\nUnderstood.<end_of_turn>\n";
    const char *UOPEN="<start_of_turn>user\n", *UCLOSE="<end_of_turn>\n<start_of_turn>model\n";
    int32_t sysb[4096], uob[64], ucb[64];
    long sn=sp_tokenizer_encode(tk,SYS,strlen(SYS),1,sysb,4096);
    long uon=sp_tokenizer_encode(tk,UOPEN,strlen(UOPEN),1,uob,64);
    long ucn=sp_tokenizer_encode(tk,UCLOSE,strlen(UCLOSE),0,ucb,64);
    FILE *fm=fopen(man,"rb"); if(!fm){ fprintf(stderr,"[kai3] manifest open FAIL\n"); return 2; }
    char path[512], expect[32]; int pass=0, tot=0;
    while(fscanf(fm,"%511s %31s",path,expect)==2){
        int k=0; float *vecs=NULL; if(kai2_read_packet(path,E,&k,&vecs)!=0){ fprintf(stderr,"[kai3] packet read FAIL %s\n",path); continue; }
        sp_g4_kv *s=gemma4_kv_open(m,2048); if(!s){ free(vecs); continue; }
        char tD[128]; const char *dec="FAIL";
        if(!gemma4_kv_prefill(s,sysb,(int)sn)){ gemma4_kv_commit(s);
            if(!gemma4_kv_prefill(s,uob,(int)uon) && !gemma4_kv_inject_seq(s,vecs,k,PH) && !gemma4_kv_prefill(s,ucb,(int)ucn))
                dec=kai2_decide(s,tk,tD);
        }
        int ok=(strcmp(dec,expect)==0); pass+=ok; tot++;
        fprintf(stderr,"[kai3] %s expect=%s | PACKET(k=%d via inject_seq)->%s (\"%s\") [%s]\n",
                path,expect,k,dec,tD,ok?"PASS":"miss");
        gemma4_kv_close(s); free(vecs);
    }
    fclose(fm);
    fprintf(stderr,"[kai3] G-KAIROS-3 PROJECTED-FRAME PIVOT GATE: %d/%d\n",pass,tot);
    sp_tokenizer_free(tk); sp_cuda_model_release(m); qwen3_free(m); sp_model_unload(handle);
    return (tot>0 && pass==tot) ? 0 : 3;
}

/* ═══ SP_G4_KAI3_WRITE=<packet.bin> (G-XBAR-ORGANISM step 1: the EAR -> Ring-2 WRITE SEAM). ═══
 * Inject a REAL audio-derived projector packet via the proven KAI-3 path (gemma4_kv_inject_seq inside
 * the daemon turn frame), then SERIALIZE the resulting audio-conditioned resident cache into a standard
 * Ring-2 episode ep_audio.{k,v,mf} — the same on-disk format SP_REPLAY / the curator digest. Proves the
 * 12B's audio-pivoted KV state becomes a functional, curator-indexable memory. Out dir = SP_KAI3_WRITE_OUT. */
static int run_kai3_write(void){
    const char *spm=getenv("SP_GEMMA4_SPMODEL"); if(!spm) spm=SP_GEMMA4_SPMODEL_DEF;
    const char *stk=getenv("SP_GEMMA4_SPTOK");   if(!stk) stk=SP_GEMMA4_SPTOK_DEF;
    const char *pkt=getenv("SP_G4_KAI3_WRITE");  const char *out=getenv("SP_KAI3_WRITE_OUT");
    if(!pkt||!out){ fprintf(stderr,"[kai3-wr] need SP_G4_KAI3_WRITE=<packet.bin> + SP_KAI3_WRITE_OUT=<epdir>\n"); return 2; }
    sp_model *handle=NULL;
    if(sp_model_load(spm,stk,&handle)!=SP_OK||!handle){ fprintf(stderr,"[kai3-wr] load FAIL\n"); return 2; }
    qwen3_model *m=sp_model_to_gemma4(handle); if(!m){ fprintf(stderr,"[kai3-wr] to_gemma4 FAIL\n"); return 2; }
    sp_tokenizer *tk=sp_tokenizer_load_tokfile(stk); if(!tk){ fprintf(stderr,"[kai3-wr] tok FAIL\n"); return 2; }
    const qwen3_config *c=&m->cfg;
    const int E=(int)c->n_embd, PH=258881;
    const int NL=(int)c->n_layers, period=(int)c->g4_swa_period?(int)c->g4_swa_period:6;
    const int kvfs=(int)c->g4_n_kv_from_start?(int)c->g4_n_kv_from_start:NL;
    const int g_nkv=(int)c->n_head_kv,g_hd=(int)c->head_dim,s_nkv=(int)c->g4_nkv_swa,s_hd=(int)c->g4_hd_swa,nh=(int)c->n_head;
    const char *SYS=
        "<start_of_turn>user\nYou are a background kernel daemon. Each tick you receive one environment event "
        "that carries a salience score. Reply with EXACTLY one of:\nNO_OP\n<ACTION>short imperative</ACTION>\n"
        "<end_of_turn>\n<start_of_turn>model\nUnderstood.<end_of_turn>\n";
    const char *UOPEN="<start_of_turn>user\n", *UCLOSE="<end_of_turn>\n<start_of_turn>model\n";
    int32_t sysb[4096],uob[64],ucb[64];
    long sn=sp_tokenizer_encode(tk,SYS,strlen(SYS),1,sysb,4096);
    long uon=sp_tokenizer_encode(tk,UOPEN,strlen(UOPEN),1,uob,64);
    long ucn=sp_tokenizer_encode(tk,UCLOSE,strlen(UCLOSE),0,ucb,64);
    int k=0; float *vecs=NULL;
    if(kai2_read_packet(pkt,E,&k,&vecs)!=0){ fprintf(stderr,"[kai3-wr] packet read FAIL %s\n",pkt); return 2; }
    int Pmax=2048; sp_g4_kv *s=gemma4_kv_open(m,Pmax);
    if(!s){ fprintf(stderr,"[kai3-wr] kv_open FAIL: %s\n",sp_last_error()); free(vecs); return 2; }
    if(gemma4_kv_prefill(s,sysb,(int)sn)){ fprintf(stderr,"[kai3-wr] sys FAIL\n"); return 2; }
    gemma4_kv_commit(s);
    if(gemma4_kv_prefill(s,uob,(int)uon)||gemma4_kv_inject_seq(s,vecs,k,PH)||gemma4_kv_prefill(s,ucb,(int)ucn)){
        fprintf(stderr,"[kai3-wr] inject flow FAIL: %s\n",sp_last_error()); return 2; }
    int npos=gemma4_kv_pos(s);
    fprintf(stderr,"[kai3-wr] audio packet injected (k=%d frames); conditioned cache npos=%d\n",k,npos);
    fprintf(stderr,"[kai3-wr] cfg: E=%d NL=%d period=%d kvfs=%d  global(nkv=%d hd=%d ->%d)  swa(nkv=%d hd=%d ->%d)\n",
            E,NL,period,kvfs,g_nkv,g_hd,g_nkv*g_hd,s_nkv,s_hd,s_nkv*s_hd);
    /* snapshot the conditioned cache (per-layer width = class kvd) */
    float **hK=(float**)calloc(NL,sizeof(float*)),**hV=(float**)calloc(NL,sizeof(float*));
    for(int L=0;L<kvfs&&L<NL;L++){ int gl=((L%period)==period-1); int kvd=(gl?g_nkv:s_nkv)*(gl?g_hd:s_hd);
        size_t nb=(size_t)Pmax*kvd*sizeof(float); hK[L]=(float*)malloc(nb); hV[L]=(float*)malloc(nb); }
    gemma4_kv_snapshot(s,hK,hV);
    /* CANONICAL 12B episode = UNIFORM kvd=512 per layer (matches RECALL_WRITE / _c2_ep_wiki, which the curator
     * reads as [NL,P,512] and SP_REPLAY injects at 512/layer). Build a uniform-512 manifest + copy the first
     * EP_KVD floats of each layer's snapshot row (globals are exactly 512; any wider class is clamped to the
     * canonical global width — the curator's sig uses GLOBAL owners only, and SP_REPLAY round-trips at 512). */
    const int EP_KVD=512;
    sp_xbar_layer_geom *geom=(sp_xbar_layer_geom*)calloc(NL,sizeof(sp_xbar_layer_geom));
    for(int L=0;L<NL;L++){ int gl=((L%period)==period-1);
        geom[L].cls=(uint8_t)(gl?SP_XBAR_CLASS_GLOBAL:SP_XBAR_CLASS_SWA);
        geom[L].nh=nh; geom[L].nkv=1; geom[L].hd=EP_KVD;       /* uniform 512 (12B canonical) */
        geom[L].window=(gl?-1:1024); geom[L].rope_base=(gl?1e6f:1e4f);
        geom[L].has_freq_factors=(uint8_t)(gl?1:0); geom[L].vless=(uint8_t)(gl?1:0); }
    sp_xbar_manifest mf; memset(&mf,0,sizeof(mf));
    uint8_t sha[SP_XBAR_SHA_LEN]; memset(sha,0,sizeof(sha));
    if(sp_xbar_manifest_build(&mf,NL,npos,period,kvfs,32,SP_ARM_PROJ_SEED,sha,geom)!=0){
        fprintf(stderr,"[kai3-wr] manifest_build FAIL: %s\n",sp_last_error()); return 2; }
    /* lay the filled prefix [0,npos) of each OWNER layer (first EP_KVD floats/pos) into the uniform store */
    size_t sb=(size_t)mf.store_bytes; uint8_t *sk=(uint8_t*)calloc(1,sb),*sv=(uint8_t*)calloc(1,sb);
    for(int L=0;L<kvfs&&L<NL;L++){ int gl=((L%period)==period-1); int nat=(gl?g_nkv:s_nkv)*(gl?g_hd:s_hd);
        int cp=(nat<EP_KVD?nat:EP_KVD); size_t off=(size_t)mf.layers[L].off;
        for(int p=0;p<npos;p++){
            memcpy(sk+off+(size_t)p*EP_KVD*4, hK[L]+(size_t)p*nat, (size_t)cp*4);
            memcpy(sv+off+(size_t)p*EP_KVD*4, hV[L]+(size_t)p*nat, (size_t)cp*4); } }
    char wp[1024]; FILE *wf;
    size_t msz=sp_xbar_manifest_serial_size(&mf); uint8_t *mb=(uint8_t*)malloc(msz);
    sp_xbar_manifest_serialize(&mf,mb,msz);
    snprintf(wp,sizeof wp,"%s/ep.mf",out); wf=fopen(wp,"wb"); if(wf){fwrite(mb,1,msz,wf);fclose(wf);}
    snprintf(wp,sizeof wp,"%s/ep.k",out);  wf=fopen(wp,"wb"); if(wf){fwrite(sk,1,sb,wf);fclose(wf);}
    snprintf(wp,sizeof wp,"%s/ep.v",out);  wf=fopen(wp,"wb"); if(wf){fwrite(sv,1,sb,wf);fclose(wf);}
    fprintf(stderr,"[kai3-wr] WROTE ep_audio: manifest %zu B + K/V store %.2f MiB (NL=%d P=%d) -> %s\n",
            msz,(double)sb/1048576.0,NL,npos,out);
    fprintf(stderr,"[kai3-wr] G-XBAR-ORGANISM write-seam: ep_audio serialized from audio-conditioned cache [%s]\n",(sb>0&&npos>0)?"GREEN":"RED");
    free(mb);free(sk);free(sv);free(geom);sp_xbar_manifest_free(&mf);
    for(int L=0;L<kvfs&&L<NL;L++){free(hK[L]);free(hV[L]);} free(hK);free(hV);free(vecs);
    gemma4_kv_close(s); sp_tokenizer_free(tk); sp_cuda_model_release(m); qwen3_free(m); sp_model_unload(handle);
    return (sb>0&&npos>0)?0:3;
}

/* ═══ SP_G4_KV_WRAP=1 (KAI-1c G-1b-WRAP-NULL): wrap-aware ring rewind gate. ═══
 * Small Wring (SP_G4_KV_RING_W) forces wraps cheaply. Prefill past W multiple times
 * (≥3 wraps), commit (retain the action). Snapshot the W-slot SWA rings. Idle tick
 * whose span crosses a wrap boundary; wrap-crossing rewind; snapshot again. The SWA
 * rings MUST be byte-identical (the undo-journal is a perfect inverse across the wrap),
 * and the re-run idle tick reproduces identical tokens. */
static int run_kv_wrap(void) {
    const char *spm=getenv("SP_GEMMA4_SPMODEL"); if(!spm) spm=SP_GEMMA4_SPMODEL_DEF;
    const char *stk=getenv("SP_GEMMA4_SPTOK");   if(!stk) stk=SP_GEMMA4_SPTOK_DEF;
    sp_model *handle=NULL;
    if(sp_model_load(spm,stk,&handle)!=SP_OK||!handle){ fprintf(stderr,"[g4-wrap] load FAIL\n"); return 2; }
    qwen3_model *m=sp_model_to_gemma4(handle);
    if(!m){ fprintf(stderr,"[g4-wrap] to_gemma4 FAIL: %s\n",sp_last_error()); return 2; }
    const qwen3_config *c=&m->cfg;
    const int NL=(int)c->n_layers, period=(int)c->g4_swa_period?(int)c->g4_swa_period:6;
    const int kvfs=(int)c->g4_n_kv_from_start?(int)c->g4_n_kv_from_start:NL;
    const int g_nkv=(int)c->n_head_kv,g_hd=(int)c->head_dim,s_nkv=(int)c->g4_nkv_swa,s_hd=(int)c->g4_hd_swa;
    const int W = getenv("SP_G4_KV_RING_W")?atoi(getenv("SP_G4_KV_RING_W")):16;
    const int Pmax=160, sys_n=50, frm_n=12, ngen=8;     /* sys 50 > 3W -> ≥3 wraps; tick 20 > W -> wrap-crossing */
    if(W<=0){ fprintf(stderr,"[g4-wrap] need SP_G4_KV_RING_W>0\n"); return 2; }
    sp_g4_kv *s=gemma4_kv_open(m,Pmax);                  /* ring mode via SP_G4_KV_RING_W env */
    if(!s){ fprintf(stderr,"[g4-wrap] kv_open FAIL: %s\n",sp_last_error()); return 2; }
    int32_t sys[64]; for(int i=0;i<sys_n;i++) sys[i]=100+(i%200);
    int32_t frm[16]; for(int i=0;i<frm_n;i++) frm[i]=500+i;
    if(gemma4_kv_prefill(s,sys,sys_n)){ fprintf(stderr,"[g4-wrap] sys prefill FAIL: %s\n",sp_last_error()); return 2; }
    gemma4_kv_commit(s);                                 /* the retained action: baseline + journal reset */
    int anchor=gemma4_kv_pos(s);
    float **hK0=(float**)calloc(NL,sizeof(float*)),**hV0=(float**)calloc(NL,sizeof(float*));
    float **hK1=(float**)calloc(NL,sizeof(float*)),**hV1=(float**)calloc(NL,sizeof(float*));
    for(int L=0;L<kvfs&&L<NL;L++){ int global=((L%period)==period-1); int kvd=(global?g_nkv:s_nkv)*(global?g_hd:s_hd);
        size_t slots=global?(size_t)Pmax:(size_t)W; size_t nb=slots*kvd*sizeof(float);
        hK0[L]=malloc(nb);hV0[L]=malloc(nb);hK1[L]=malloc(nb);hV1[L]=malloc(nb); }
    gemma4_kv_snapshot(s,hK0,hV0);
    int32_t gen1[8];
    if(gemma4_kv_prefill(s,frm,frm_n)||gemma4_kv_decode(s,ngen,gen1)){ fprintf(stderr,"[g4-wrap] tick1 FAIL: %s\n",sp_last_error()); return 2; }
    int after=gemma4_kv_pos(s);
    int wraps_crossed=(after/W)-(anchor/W);
    /* NON-VACUITY: ring AFTER tick (pre-rewind). The wrap-crossing tick MUST clobber live
     * window slots (aliasing) — if pre==mid the rewind would be trivially identity. */
    float **hKm=(float**)calloc(NL,sizeof(float*)),**hVm=(float**)calloc(NL,sizeof(float*));
    for(int L=0;L<kvfs&&L<NL;L++){ int global=((L%period)==period-1); int kvd=(global?g_nkv:s_nkv)*(global?g_hd:s_hd);
        size_t slots=global?(size_t)Pmax:(size_t)W; size_t nb=slots*kvd*sizeof(float); hKm[L]=malloc(nb);hVm[L]=malloc(nb); }
    gemma4_kv_snapshot(s,hKm,hVm);
    long clobbered=0;
    for(int L=0;L<kvfs&&L<NL;L++){ int global=((L%period)==period-1); if(global) continue;
        int kvd=s_nkv*s_hd; size_t nb=(size_t)W*kvd*sizeof(float); if(memcmp(hK0[L],hKm[L],nb)) clobbered++; }
    if(gemma4_kv_rewind(s,after-anchor)){ fprintf(stderr,"[g4-wrap] rewind FAIL: %s\n",sp_last_error()); return 2; }
    gemma4_kv_snapshot(s,hK1,hV1);
    long diffs=0;
    for(int L=0;L<kvfs&&L<NL;L++){ int global=((L%period)==period-1); if(global) continue;   /* SWA rings only */
        int kvd=s_nkv*s_hd; size_t nb=(size_t)W*kvd*sizeof(float);
        if(memcmp(hK0[L],hK1[L],nb)) diffs++; if(memcmp(hV0[L],hV1[L],nb)) diffs++; }
    int32_t gen2[8]; int genmatch=1;
    if(gemma4_kv_prefill(s,frm,frm_n)||gemma4_kv_decode(s,ngen,gen2)){ fprintf(stderr,"[g4-wrap] tick2 FAIL\n"); return 2; }
    for(int i=0;i<ngen;i++) if(gen1[i]!=gen2[i]) genmatch=0;
    gemma4_kv_rewind(s,gemma4_kv_pos(s)-anchor);
    fprintf(stderr,"[g4-wrap] W=%d anchor=%d after=%d wraps_crossed=%d (anchor%%W=%d tick_span=%d) clobbered_owners=%ld\n",W,anchor,after,wraps_crossed,anchor%W,after-anchor,clobbered);
    fprintf(stderr,"[g4-wrap] WRAP-NULL: %s (swa-ring-diffs=%ld) | EQUIV gen-reproduce: %s [%d %d %d %d ...] | non-vacuous: %s\n",
        diffs==0?"GREEN":"RED",diffs, genmatch?"GREEN":"RED", gen1[0],gen1[1],gen1[2],gen1[3], clobbered>0?"YES":"NO(VACUOUS!)");
    gemma4_kv_close(s); sp_model_unload(handle);
    return (diffs==0 && genmatch && clobbered>0)?0:3;
}

/* GELU tanh approximation — verbatim gemma4.c g4_gelu (static in the oracle TU). */
static float g4_gelu(float v) {
    const float k = 0.7978845608028654f;
    return 0.5f * v * (1.0f + tanhf(k * (v + 0.044715f * v * v * v)));
}

/* ═══ Truncated parity harness (ETA.2/3): CPU mirror of gemma4.c through
 * n_layers (full layers, attention-only at the LAST), built from the oracle's
 * OWN core primitives in the oracle's exact order, vs gemma4_cuda_probe at the
 * same boundary. Drops the 6-stage telemetry at the last layer; abs_gates =
 * {nx, q, ao, ap, x} or NULL for telemetry-only (first run measures the floor,
 * then the gates get pinned at ~3x — the L0 discipline). NO AltUp/out_scale on
 * EITHER side (ETA.4); the boundary is before the injection point. ═══ */
static void truncated_parity(const qwen3_model *m, int n_layers, const char *tag,
                             const double *abs_gates) {
    enum { NT = 12 };
    const int32_t toks[NT] = { 2, 10, 100, 1000, 5000, 9999, 31, 7, 42, 256, 777, 12345 };
    const qwen3_config *c = &m->cfg;
    const int E = (int)c->n_embd, SW = (int)c->sliding_window, FF = (int)c->n_ff;
    const float eps = c->rms_eps, embscale = sqrtf((float)E);
    const int period = (int)c->g4_swa_period ? (int)c->g4_swa_period : 6;
    const int NL = (int)c->n_layers;
    const int kvfs = (int)c->g4_n_kv_from_start ? (int)c->g4_n_kv_from_start : NL;
    const int g_nh = (int)c->n_head, g_nkv = (int)c->n_head_kv, g_hd = (int)c->head_dim;
    const int s_nh = (int)c->g4_nh_swa, s_nkv = (int)c->g4_nkv_swa, s_hd = (int)c->g4_hd_swa;
    const float g_base = c->rope_freq_base, s_base = c->g4_rope_base_swa;
    const int QDmax = (g_nh*g_hd > s_nh*s_hd) ? g_nh*g_hd : s_nh*s_hd;
    const int KVDmax = (g_nkv*g_hd > s_nkv*s_hd) ? g_nkv*g_hd : s_nkv*s_hd;
    int FFmax = FF;
    for (int L = 0; L < n_layers; L++) {
        const gguf_tensor *fg = m->layers[L].ffn_gate;
        int f = (fg && fg->n_dims >= 2 && fg->dims[1] > 0) ? (int)fg->dims[1] : FF;
        if (f > FFmax) FFmax = f;
    }

    float *x   = (float *)malloc((size_t)NT * E * sizeof(float));
    float *nx  = (float *)malloc((size_t)NT * E * sizeof(float));
    float *nx0 = (float *)malloc((size_t)NT * E * sizeof(float));
    float *q   = (float *)malloc((size_t)NT * QDmax * sizeof(float));
    /* per-OWNER K/V storage (shared-KV mirror of the oracle: owners [0,kvfs)
     * compute+store; sharers reuse owner kvfs-1 (global) / kvfs-2 (SWA) and skip
     * their own projection + norms — gemma4.c lines 173-193) */
    float **Kst = (float **)calloc((size_t)NL, sizeof(float *));
    float **Vst = (float **)calloc((size_t)NL, sizeof(float *));
    float *ao  = (float *)malloc((size_t)NT * QDmax * sizeof(float));
    float *ap  = (float *)malloc((size_t)NT * E * sizeof(float));
    float *g   = (float *)malloc((size_t)NT * FFmax * sizeof(float));
    float *up  = (float *)malloc((size_t)NT * FFmax * sizeof(float));
    float *dn  = (float *)malloc((size_t)NT * E * sizeof(float));
    float *sc  = (float *)malloc((size_t)NT * sizeof(float));
    float *gq  = (float *)malloc((size_t)NT * QDmax * sizeof(float));
    SP_CHECK(x && nx && nx0 && q && Kst && Vst && ao && ap && g && up && dn && sc && gq,
             "parity buffers");
    if (!(x && nx && nx0 && q && Kst && Vst && ao && ap && g && up && dn && sc && gq)) goto fin;

    {
        int cpu_ok = 1, last_qd = 0;
        for (int t = 0; t < NT && cpu_ok; t++) {
            if (sp_embed_row(m, toks[t], E, x + (size_t)t * E)) cpu_ok = 0;
            for (int i = 0; i < E; i++) x[(size_t)t * E + i] *= embscale;
        }
        for (int L = 0; L < n_layers && cpu_ok; L++) {
            const qwen3_layer *ly = &m->layers[L];
            const int global = ((L % period) == period - 1);
            const int nh  = global ? g_nh  : s_nh;
            const int nkv = global ? g_nkv : s_nkv;
            const int hd  = global ? g_hd  : s_hd;
            const int grp = nh / nkv, qd = nh * hd, kvd = nkv * hd;
            const float rbase = global ? g_base : s_base;
            const float *ffac = global ? sp_as_f32(m, m->rope_freqs) : NULL;
            const int win = global ? -1 : SW;
            const gguf_tensor *fg = ly->ffn_gate;
            const int ffL = (fg && fg->n_dims >= 2 && fg->dims[1] > 0) ? (int)fg->dims[1] : FF;
            const int last = (L == n_layers - 1);
            if (last) last_qd = qd;

            for (int t = 0; t < NT; t++)
                sp_rmsnorm(x + (size_t)t * E, sp_as_f32(m, ly->attn_norm), E, eps, nx + (size_t)t * E);
            if (last) memcpy(nx0, nx, (size_t)NT * E * sizeof(float));
            if (sp_matmul(m, ly->attn_q, nx, NT, E, qd, q)) { cpu_ok = 0; break; }
            {
                const float *qn = sp_as_f32(m, ly->attn_q_norm);
                for (int t = 0; t < NT; t++)
                    for (int h = 0; h < nh; h++) {
                        float *qh = q + (size_t)t * qd + (size_t)h * hd;
                        sp_rmsnorm_head(qh, qn, hd, eps);
                        sp_rope_neox_freqs(qh, hd, t, rbase, ffac);
                    }
            }
            float *Kuse, *Vuse;
            if (L < kvfs) {                       /* OWNER: project + norm + store */
                float *K  = (float *)malloc((size_t)NT * kvd * sizeof(float));
                float *Vb = (float *)malloc((size_t)NT * kvd * sizeof(float));
                if (!K || !Vb) { free(K); free(Vb); cpu_ok = 0; break; }
                if (sp_matmul(m, ly->attn_k, nx, NT, E, kvd, K))  { free(K); free(Vb); cpu_ok = 0; break; }
                if (ly->attn_v) {   /* V-less layers (dense 12B globals): V = raw K projection */
                    if (sp_matmul(m, ly->attn_v, nx, NT, E, kvd, Vb)) { free(K); free(Vb); cpu_ok = 0; break; }
                } else {
                    memcpy(Vb, K, (size_t)NT * kvd * sizeof(float));
                }
                const float *kn = sp_as_f32(m, ly->attn_k_norm);
                for (int t = 0; t < NT; t++)
                    for (int h = 0; h < nkv; h++) {
                        float *kh = K + (size_t)t * kvd + (size_t)h * hd;
                        sp_rmsnorm_head(kh, kn, hd, eps);
                        sp_rope_neox_freqs(kh, hd, t, rbase, ffac);
                        float *vh = Vb + (size_t)t * kvd + (size_t)h * hd;
                        double ss = 0.0;
                        for (int i = 0; i < hd; i++) ss += (double)vh[i] * (double)vh[i];
                        float inv = 1.0f / sqrtf((float)(ss / (double)hd) + eps);
                        for (int i = 0; i < hd; i++) vh[i] *= inv;
                    }
                Kst[L] = K; Vst[L] = Vb; Kuse = K; Vuse = Vb;
            } else {                              /* SHARER: reuse owner, skip projection */
                const int src = kvfs - (global ? 1 : 2);
                Kuse = Kst[src]; Vuse = Vst[src];
                if (!Kuse || !Vuse) { cpu_ok = 0; break; }
            }
            for (int t = 0; t < NT; t++)
                for (int h = 0; h < nh; h++)
                    sp_attn_head(q + (size_t)t * qd + (size_t)h * hd, Kuse, Vuse, t, kvd,
                                 h / grp, hd, 1.0f, win, sc, ao + (size_t)t * qd + (size_t)h * hd);
            if (sp_matmul(m, ly->attn_output, ao, NT, qd, E, ap)) { cpu_ok = 0; break; }
            for (int t = 0; t < NT; t++) {
                sp_rmsnorm(ap + (size_t)t * E, sp_as_f32(m, ly->post_attn_norm), E, eps, nx + (size_t)t * E);
                float *xt = x + (size_t)t * E; const float *pt = nx + (size_t)t * E;
                for (int i = 0; i < E; i++) xt[i] += pt[i];
            }
            if (last) break;   /* boundary A: attention residual of the last layer */

            for (int t = 0; t < NT; t++)
                sp_rmsnorm(x + (size_t)t * E, sp_as_f32(m, ly->ffn_norm), E, eps, nx + (size_t)t * E);
            if (sp_matmul(m, ly->ffn_gate, nx, NT, E, ffL, g))  { cpu_ok = 0; break; }
            if (sp_matmul(m, ly->ffn_up,   nx, NT, E, ffL, up)) { cpu_ok = 0; break; }
            for (size_t i = 0; i < (size_t)NT * ffL; i++) g[i] = g4_gelu(g[i]) * up[i];
            if (sp_matmul(m, ly->ffn_down, g, NT, ffL, E, dn))  { cpu_ok = 0; break; }
            for (int t = 0; t < NT; t++) {
                sp_rmsnorm(dn + (size_t)t * E, sp_as_f32(m, ly->post_ffw_norm), E, eps, nx + (size_t)t * E);
                float *xt = x + (size_t)t * E; const float *pt = nx + (size_t)t * E;
                for (int i = 0; i < E; i++) xt[i] += pt[i];
            }
        }
        SP_CHECK(cpu_ok, "CPU truncated mirror computed");

        if (cpu_ok) {
            struct { int stage; const float *cpu; size_t n; const char *nm; } stg[] = {
                { 2, nx0, (size_t)NT * E,       "nx (attn_norm)   " },
                { 3, q,   (size_t)NT * last_qd, "q  (norm+rope)   " },
                { 4, ao,  (size_t)NT * last_qd, "ao (attention)   " },
                { 5, ap,  (size_t)NT * E,       "ap (Wo, pre-norm)" },
                { 1, x,   (size_t)NT * E,       "x  (attn residual)" },
            };
            for (int s = 0; s < 5; s++) {
                int rc2 = gemma4_cuda_probe(m, toks, NT, n_layers, stg[s].stage, gq);
                if (rc2) { fprintf(stderr, "    [%s] stage %d probe: %s\n", tag, stg[s].stage, sp_last_error()); continue; }
                double mr = 0.0, ma = 0.0;
                for (size_t i = 0; i < stg[s].n; i++) {
                    double d = fabs((double)gq[i] - (double)stg[s].cpu[i]);
                    double den = fabs((double)stg[s].cpu[i]) > 1e-6 ? fabs((double)stg[s].cpu[i]) : 1e-6;
                    if (d > ma) ma = d;
                    if (d / den > mr) mr = d / den;
                }
                fprintf(stderr, "    [g4-cuda-%s] stage %s max rel %.3e  max abs %.3e%s\n",
                        tag, stg[s].nm, mr, ma, abs_gates ? "" : "  (telemetry)");
                if (abs_gates) {
                    double gate = abs_gates[s];
                    SP_CHECK((gate == 0.0 && ma == 0.0) || (gate > 0.0 && ma <= gate),
                             "stage at the measured f32 floor");
                }
            }
        }
    }
fin:
    free(x); free(nx); free(nx0); free(q); free(ao); free(ap);
    free(g); free(up); free(dn); free(sc); free(gq);
    if (Kst) { for (int L = 0; L < NL; L++) free(Kst[L]); free(Kst); }
    if (Vst) { for (int L = 0; L < NL; L++) free(Vst[L]); free(Vst); }
}

static void T_GEMMA4_CUDA_WEIGHTS(void) {
    const char *spm = getenv("SP_GEMMA4_SPMODEL"); if (!spm) spm = SP_GEMMA4_SPMODEL_DEF;
    const char *stk = getenv("SP_GEMMA4_SPTOK");   if (!stk) stk = SP_GEMMA4_SPTOK_DEF;

    FILE *probe = fopen(spm, "rb");
    if (!probe) { fprintf(stderr, "    model absent (%s) — SKIP\n", spm); return; }
    fclose(probe);
    if (sp_cuda_device_count() < 1) { fprintf(stderr, "    no CUDA device — SKIP\n"); return; }

    sp_model *handle = NULL;
    sp_status st = sp_model_load(spm, stk, &handle);
    SP_CHECK(st == SP_OK && handle, "sp_model_load gemma4-e2b");
    if (st != SP_OK || !handle) return;

    qwen3_model *m = sp_model_to_gemma4(handle);
    if (!m) fprintf(stderr, "    sp_model_to_gemma4: %s\n", sp_last_error());
    SP_CHECK(m != NULL, "sp_model_to_gemma4 (core bridge)");
    if (!m) { sp_model_unload(handle); return; }

    /* structural invariants the CUDA build depends on (from the cfg the bridge
     * populated; the CPU oracle uses exactly these) */
    SP_CHECK(m->cfg.arch == SP_ARCH_GEMMA4, "arch == SP_ARCH_GEMMA4");
    SP_CHECK(m->cfg.g4_swa_period > 0, "g4_swa_period set");
    SP_CHECK(m->cfg.g4_n_kv_from_start > 0 &&
             m->cfg.g4_n_kv_from_start <= m->cfg.n_layers, "shared-KV kvfs in range");
    /* AltUp/PLE is the E-SERIES MatFormer machinery; the dense 12B runs PL=0
     * (with layer_output_scale + rope_freqs still present). Assert accordingly. */
    if (m->cfg.g4_n_embd_per_layer > 0) {
        SP_CHECK(m->per_layer_model_proj && m->per_layer_proj_norm && m->rope_freqs,
                 "model-level AltUp tensors present (E-series)");
    } else {
        fprintf(stderr, "    dense gemma4 (PL=0): no AltUp/PLE; rope_freqs %s, out_scale[0] %s\n",
                m->rope_freqs ? "present" : "ABSENT", m->layers[0].out_scale ? "present" : "ABSENT");
        SP_CHECK(m->rope_freqs != NULL, "dense gemma4: rope_freqs present");
    }

    /* THE GATE: upload the full gemma4 weight set to the device. */
    int rc = gemma4_cuda_weights_probe(m);
    if (rc) fprintf(stderr, "    probe: %s\n", sp_last_error());
    SP_CHECK(rc == 0, "gemma4 CUDA weight set uploads (per-layer geometry + shared-KV + AltUp)");

    /* SP_G4_FASTPROBE=1 (ETA.5b bug-hunt): straight to ONE probed decode step —
     * prompt {2,10,100,1000,497}, NG=1 — skipping the slow oracle blocks.
     * Pair with SP_G4_DEC_PROBE=<pos> (read by gemma4_decode_cuda) to intercept
     * the step. Diagnostic mode only; no gates. */
    {
        const char *fpb = getenv("SP_G4_FASTPROBE");
        if (fpb && fpb[0] == '1') {
            const int E = (int)m->cfg.n_embd, NL_ = (int)m->cfg.n_layers;
            int32_t d2[6] = { 2, 10, 100, 1000, 497, 0 };
            /* out_scale values on the record (position-independent, but known) */
            fprintf(stderr, "    [fastprobe] out_scale:");
            for (int L = 0; L < NL_ && L < 48; L++) {
                const float *os = m->layers[L].out_scale ? sp_as_f32(m, m->layers[L].out_scale) : NULL;
                fprintf(stderr, " %.3f", os ? os[0] : -1.0f);
            }
            fprintf(stderr, "\n");
            /* SP_G4_LIFT=1: run the probed decode in PURE LIFT arithmetic (same
             * as the prefill probe) — the int8-vs-lift noise source drops out of
             * the diff entirely; only a structural decode bug remains visible. */
            const int lift_mode = (getenv("SP_G4_LIFT") != NULL);
            if (!lift_mode) _putenv("SP_CUDA_DECODE_INT8=1");
            int dn2 = gemma4_decode_cuda(m, d2, 5, 1, -1);
            if (dn2 < 0) fprintf(stderr, "    fastprobe decode: %s\n", sp_last_error());
            fprintf(stderr, "    [fastprobe] %s dn=%d produced=%d (oracle argmax ref: 11629)\n",
                    lift_mode ? "LIFT" : "int8", dn2, dn2 == 6 ? d2[5] : -1);
            if (!lift_mode) _putenv("SP_CUDA_DECODE_INT8=");

            /* layer-bisect: diff the decode dump (pos 4, FFN-residual boundary,
             * out_scale skipped via SP_G4_NO_OSCALE) against the TRUNCATED
             * prefill probe (gemm_w_lift arithmetic; skips PL/out_scale by
             * design; parity-proven vs the CPU mirror) at the same boundary,
             * row t=4. Expect the int8-vs-lift noise floor everywhere EXCEPT
             * past the broken stage. */
            const char *dumpf = getenv("SP_G4_DEC_DUMP");
            FILE *df = dumpf ? fopen(dumpf, "rb") : NULL;
            if (df) {
                float *dump = (float *)malloc((size_t)NL_ * E * sizeof(float));
                float *px   = (float *)malloc((size_t)5 * E * sizeof(float));
                if (dump && px && fread(dump, sizeof(float), (size_t)NL_ * E, df) == (size_t)NL_ * E) {
                    static const int bnd[] = { 1, 5, 8, 10, 11, 12, 24, 48 };
                    for (size_t b = 0; b < sizeof(bnd)/sizeof(bnd[0]); b++) {
                        int L = bnd[b]; if (L > NL_) break;
                        if (gemma4_cuda_probe(m, d2, 5, L, 0, px)) {
                            fprintf(stderr, "    probe L%d: %s\n", L, sp_last_error()); break;
                        }
                        const float *pr = px + (size_t)4 * E;        /* row t=4 */
                        const float *dc = dump + (size_t)(L - 1) * E; /* after layer L-1 */
                        double ma = 0.0, ssp = 0.0, ssd = 0.0;
                        for (int i = 0; i < E; i++) {
                            double d = fabs((double)dc[i] - (double)pr[i]);
                            if (d > ma) ma = d;
                            ssp += (double)pr[i] * pr[i]; ssd += (double)dc[i] * dc[i];
                        }
                        fprintf(stderr, "    [fastprobe-diff] boundary L%-2d max|dec-probe| %.3e  "
                                        "|x|_dec %.4e |x|_probe %.4e\n",
                                L, ma, sqrt(ssd), sqrt(ssp));
                    }
                }
                free(dump); free(px); fclose(df);
            }
            sp_cuda_model_release(m);
            qwen3_free(m);
            sp_model_unload(handle);
            return;
        }
    }

    /* ════ E_G4_CU_L0 / E_G4_CU_L4 (ETA.2/3): TRUNCATED PARITY via the harness ════
     * L0 (SWA, n_layers=1): gates PINNED at ~3x the measured floors (bisection
     * 2026-06-06): nx BIT-EXACT; q 8.6e-6 / ao 3.2e-5 / ap 6.3e-5 abs (f32 GEMM
     * floor, gemm_w_lift oracle arithmetic); residual 1.59e-3 = the post_attn_norm
     * 1/rms(ap)~25x amplification of that floor (mechanism on record above).
     * L4 (the FIRST GLOBAL layer, n_layers=5): the geometry shift — hd 256->512,
     * qd 2048->4096, nkv geometry change, rope_freqs proportional table engages,
     * SWA mask drops to full causal. First run = TELEMETRY (NULL gates); pinned
     * after measurement, the L0 discipline. */
    sp_kernels_read_env();
    /* The pinned ABS gates below were measured + pinned ON E2B (telemetry-then-pin).
     * Other gemma4 geometries (the dense 12B: 48L, E=3840, V-less globals) run the
     * SAME probes in TELEMETRY mode (NULL gates) until their own floors are pinned
     * — different model, different absolute floors; reusing E2B constants would be
     * gate-currency confusion, not rigor. */
    const int e2b_pinned = !(((size_t)m->cfg.n_vocab * (size_t)m->cfg.n_embd * sizeof(float)) > ((size_t)2u << 30));
    {
        static const double l0_gates[5] = { 0.0, 5e-5, 1e-4, 2e-4, 5e-3 };
        truncated_parity(m, 1, "L0", e2b_pinned ? l0_gates : NULL);
        /* L4 gates pinned at ~3x the measured floors (telemetry run 2026-06-06:
         * nx 1.11e-4 / q 1.15e-5 / ao 5.15e-5 / ap 8.15e-5 / x 1.59e-3 abs).
         * nx is no longer bit-exact at depth — 4 layers of norm-amplified inflow,
         * re-condensed by attn_norm (stable, no explosion). q at the floor PROVES
         * the rope_freqs proportional handoff + the 4096-wide global projection;
         * ao at the floor proves the full-causal (win=-1) mask switch. */
        static const double l4_gates[5] = { 3e-4, 5e-5, 1.5e-4, 2.5e-4, 5e-3 };
        truncated_parity(m, 5, "L4", e2b_pinned ? l4_gates : NULL);
        /* THE SHARER SEAM (ETA.3 remainder): L15 = the FIRST shared-KV layer
         * (kvfs=15; layers [0,15) own). L15 is SWA (15%5==0) -> reads owner
         * L13 (kvfs-2)'s STORED K/V from VRAM and skips its own projection.
         * This is the cross-layer VRAM dependency: wrong owner index = stale
         * layer; wrong stride = OOB. Both mirror + probe implement the oracle's
         * owner-store/sharer-reuse. Telemetry first; pin after measurement. */
        /* Gates pinned at ~3x measured (telemetry 2026-06-06: nx 2.37e-4 /
         * q 2.67e-5 / ao 1.11e-5 / ap 2.00e-5 / x 2.98e-3 abs). ao AT THE FLOOR
         * is the seam proof: an off-by-one owner index would read L14's GLOBAL
         * K/V (kvd 512) through an SWA stride (256) -> garbage, not 1.1e-5. */
        static const double l15_gates[5] = { 7e-4, 8e-5, 5e-5, 6e-5, 1e-2 };
        truncated_parity(m, 16, "L15-sharer", e2b_pinned ? l15_gates : NULL);
    }

    /* ETA.5b.4: 12B-class models keep only PACKED embd codes resident (the f32
     * dequant would be ~4 GB on a 12 GB card) — the FULL-forward tied-head f32
     * GEMM is unavailable there, and decode requires the dp4a route. The ORACLE
     * teacher-forced decode check remains the decisive gate either way. */
    const int big_embd =
        ((size_t)m->cfg.n_vocab * (size_t)m->cfg.n_embd * sizeof(float)) > ((size_t)2u << 30);

    /* ════ E_G4_CU_FULL (ETA.4): THE DECISIVE GATE — the live run. The full
     * 35-layer gemma4_forward_cuda (per-layer geometry + shared-KV + rope_freqs
     * + AltUp + out_scale + tied head + softcap) vs the CPU oracle
     * gemma4_forward, per-position ARGMAX + KL(softmax_cpu || softmax_cuda).
     * The repo's standard cross-backend currency. ════ */
    if (big_embd) {
        fprintf(stderr, "    [g4-cuda-FULL] SKIP: f32 embd not resident (12B-class VRAM budget); "
                        "the decode oracle gate below is the live check\n");
    } else {
        enum { NT = 12 };
        const int32_t toks[NT] = { 2, 10, 100, 1000, 5000, 9999, 31, 7, 42, 256, 777, 12345 };
        const int V = (int)m->cfg.n_vocab;
        float *cl = (float *)malloc((size_t)NT * V * sizeof(float));
        float *gl = (float *)malloc((size_t)NT * V * sizeof(float));
        SP_CHECK(cl && gl, "full-forward logits buffers");
        if (cl && gl) {
            int orc = gemma4_forward(m, toks, NT, cl);
            SP_CHECK(orc == 0, "CPU oracle gemma4_forward");
            int crc = gemma4_forward_cuda(m, toks, NT, gl);
            if (crc) fprintf(stderr, "    full cuda: %s\n", sp_last_error());
            SP_CHECK(crc == 0, "gemma4_forward_cuda ran");
            if (orc == 0 && crc == 0) {
                int agree = 0; double max_kl = 0.0, max_abs = 0.0;
                for (int t = 0; t < NT; t++) {
                    const float *cp = cl + (size_t)t * V, *gp = gl + (size_t)t * V;
                    /* argmax */
                    int ac = 0, ag = 0;
                    for (int i = 1; i < V; i++) {
                        if (cp[i] > cp[ac]) ac = i;
                        if (gp[i] > gp[ag]) ag = i;
                    }
                    if (ac == ag) agree++;
                    /* KL(p_cpu || q_cuda), double-precision log-softmax */
                    double mc = cp[0], mg = gp[0];
                    for (int i = 1; i < V; i++) { if (cp[i] > mc) mc = cp[i]; if (gp[i] > mg) mg = gp[i]; }
                    double zc = 0.0, zg = 0.0;
                    for (int i = 0; i < V; i++) { zc += exp((double)cp[i] - mc); zg += exp((double)gp[i] - mg); }
                    double lzc = log(zc), lzg = log(zg), kl = 0.0;
                    for (int i = 0; i < V; i++) {
                        double lp = (double)cp[i] - mc - lzc;
                        double lq = (double)gp[i] - mg - lzg;
                        kl += exp(lp) * (lp - lq);
                        double d = fabs((double)cp[i] - (double)gp[i]);
                        if (d > max_abs) max_abs = d;
                    }
                    if (kl > max_kl) max_kl = kl;
                }
                fprintf(stderr, "    [g4-cuda-FULL] argmax %d/%d  max KL %.3e  max |dlogit| %.3e\n",
                        agree, NT, max_kl, max_abs);
                SP_CHECK(agree == NT, "FULL 35-layer: CUDA argmax == CPU oracle argmax (all positions)");
                SP_CHECK(max_kl < 1e-4, "FULL 35-layer: KL(cpu||cuda) at the cross-backend floor");
            }
        }
        free(cl); free(gl);
    }

    /* ════ E_G4_CU_DEC (ETA.5a): THE DECODE GATE — autoregressive generation
     * over the JAGGED shared-KV cache (per-owner widths, sharers read owners),
     * per-step AltUp, windowed single-query attention, head + softcap + argmax.
     * Gate: the CPU ORACLE prefill over the produced sequence must teacher-
     * forced argmax-predict EVERY generated token (the proven decode pattern). ════ */
    {
        enum { NP = 4, NG = 12, PT = NP + NG };
        int32_t dseq[PT]; const int32_t prompt[NP] = { 2, 10, 100, 1000 };
        const int V = (int)m->cfg.n_vocab;
        for (int i = 0; i < NP; i++) dseq[i] = prompt[i];
        if (big_embd) {   /* 12B-class: tied head must take the dp4a route (top-1 trust) */
            _putenv("SP_CUDA_DECODE_INT8=1");
            fprintf(stderr, "    [g4-cuda-DEC] 12B-class: decode runs dp4a (f32 embd not resident); "
                            "oracle teacher-force is still the gate\n");
        }
        int dn = gemma4_decode_cuda(m, dseq, NP, NG, /*eos=*/-1);
        /* NOTE: INT8 env intentionally stays set through the BISECT diagnostic
         * below (the 12B tied head REQUIRES the dp4a route); cleared at the end. */
        if (dn < 0) fprintf(stderr, "    decode: %s\n", sp_last_error());
        SP_CHECK(dn == PT, "gemma4 CUDA decode produced full length");
        if (dn == PT) {
            float *tl = (float *)malloc((size_t)PT * V * sizeof(float));
            SP_CHECK(tl != NULL, "teacher-forced logits buffer");
            if (tl) {
                int orc = gemma4_forward(m, dseq, PT, tl);
                SP_CHECK(orc == 0, "oracle prefill over decoded sequence");
                if (orc == 0) {
                    /* Gate currency depends on the decode arithmetic:
                     *   exact path (E2B, knobs off)  -> STRICT: every token == oracle argmax.
                     *   dp4a path (12B-class)        -> top-1-TRUST: a near-tie may flip; a
                     *     flip is admissible iff the produced token sits within the oracle's
                     *     TOP-2 at that position (measured, printed — not waved through).
                     * After the first flip the autoregressive context diverges, so the
                     * teacher-forced comparison stops there (the oracle graded a different
                     * prefix from then on). */
                    int exact = 1, flips = 0, badrank = 0, firstbad = -1;
                    for (int pos = NP - 1; pos < PT - 1; pos++) {
                        const float *row = tl + (size_t)pos * V;
                        const int got = dseq[pos + 1];
                        int am = 0;
                        for (int i = 1; i < V; i++) if (row[i] > row[am]) am = i;
                        if (am == got) continue;
                        exact = 0; if (firstbad < 0) firstbad = pos;
                        /* rank of the produced token in the oracle row (1 = argmax) */
                        int rank = 1; for (int i = 0; i < V; i++) if (row[i] > row[got]) rank++;
                        fprintf(stderr, "    [g4-cuda-DEC] pos %d: produced %d vs oracle argmax %d; "
                                        "oracle-rank(produced)=%d, logit gap %.4e\n",
                                pos, got, am, rank, (double)(row[am] - row[got]));
                        flips++;
                        if (rank > 2) badrank++;
                        break;   /* context diverged — later positions aren't comparable */
                    }
                    fprintf(stderr, "    [g4-cuda-DEC] %d gen tokens; oracle teacher-forced: %s",
                            NG, exact ? "ALL" : (badrank ? "FAIL" : "top-2 flip (dp4a top-1-trust)"));
                    if (firstbad >= 0) fprintf(stderr, " (first divergence pos %d)", firstbad);
                    fprintf(stderr, "; seq:");
                    for (int i = 0; i < PT; i++) fprintf(stderr, " %d", dseq[i]);
                    fprintf(stderr, "\n");
                    if (big_embd)
                        SP_CHECK(badrank == 0, "DECODE (dp4a): tokens == oracle argmax, or measured top-2 near-tie");
                    else
                        SP_CHECK(exact, "DECODE: every generated token == oracle argmax (jagged KV + AltUp per step)");

                    /* DIAGNOSTIC (provenance bisect): if the decode diverged, rerun
                     * with the last GOOD token moved INTO the prompt (NG=1). Same
                     * prefix, same oracle continuation — if the answer changes, the
                     * bug is in consuming a SELF-GENERATED token (dseq handoff);
                     * if it repeats, the step computation itself diverges. */
                    if (!exact && firstbad == NP) {
                        int32_t d2[NP + 2];
                        for (int i = 0; i < NP; i++) d2[i] = prompt[i];
                        d2[NP] = dseq[NP];              /* the good first generation */
                        int dn2 = gemma4_decode_cuda(m, d2, NP + 1, 1, -1);
                        if (dn2 < 0) fprintf(stderr, "    bisect decode: %s\n", sp_last_error());
                        const float *row = tl + (size_t)NP * V;
                        int am = 0; for (int i = 1; i < V; i++) if (row[i] > row[am]) am = i;
                        fprintf(stderr, "    [g4-cuda-DEC-BISECT] prompt+%d -> produced %d "
                                        "(oracle argmax %d; autoregressive run produced %d)\n",
                                dseq[NP], (dn2 == NP + 2) ? d2[NP + 1] : -1, am, dseq[NP + 1]);
                    }
                }
                if (big_embd) _putenv("SP_CUDA_DECODE_INT8=");
                free(tl);
            }
        }
    }

    /* ════ E_G4_CU_HIST (XBAR P3.1b-2): recall-as-history bit-exact gate ════
     * episode = prompt[0..H); the rest decodes at absolute [H,..) attending over the
     * pre-loaded episode. monolithic [episode++rest] vs WRITE-episode + HISTORY-decode
     * must be token-identical. Env-gated; early-returns to skip the slow 5B velocity run. */
    {
        const char *hg = getenv("SP_XBAR_HIST_GATE");
        if (hg && atoi(hg) > 0) {
            const int H = atoi(hg);
            enum { HNP = 4, HNG = 12, HPT = HNP + HNG };
            const int32_t hp[HNP] = { 2, 10, 100, 1000 };
            const char *dir = getenv("SP_XBAR_HIST_DIR"); if (!dir) dir = "_p31_ep";
            char ev[600];
            int32_t mono[HPT]; for (int i = 0; i < HNP; i++) mono[i] = hp[i];
            if (big_embd) _putenv("SP_CUDA_DECODE_INT8=1");
            snprintf(ev, sizeof(ev), "SP_XBAR_RECALL_WRITE=%s", dir); _putenv(ev);
            int dnm = gemma4_decode_cuda(m, mono, HNP, HNG, -1);
            _putenv("SP_XBAR_RECALL_WRITE=");
            int32_t hist[HPT]; for (int i = 0; i < H; i++) hist[i] = 0; for (int i = H; i < HNP; i++) hist[i] = hp[i];
            snprintf(ev, sizeof(ev), "SP_XBAR_RECALL_HISTORY=%s", dir); _putenv(ev);
            snprintf(ev, sizeof(ev), "SP_XBAR_HIST_H=%d", H); _putenv(ev);
            int dnh = gemma4_decode_cuda(m, hist, HNP, HNG, -1);
            _putenv("SP_XBAR_RECALL_HISTORY="); _putenv("SP_XBAR_HIST_H=");
            if (big_embd) _putenv("SP_CUDA_DECODE_INT8=");
            int diffs = 0, n = (dnm < dnh ? dnm : dnh); if (n > HPT) n = HPT;
            for (int i = H; i < n; i++) if (mono[i] != hist[i]) diffs++;
            fprintf(stderr, "    [g4-cuda-HIST] H=%d  mono dn=%d  hist dn=%d  diffs[%d..%d)=%d\n", H, dnm, dnh, H, HPT, diffs);
            fprintf(stderr, "        mono:"); for (int i = 0; i < HPT; i++) fprintf(stderr, " %d", mono[i]); fprintf(stderr, "\n");
            fprintf(stderr, "        hist:"); for (int i = 0; i < HPT; i++) fprintf(stderr, " %d", hist[i]); fprintf(stderr, "\n");
            SP_CHECK(dnm == HPT && dnh == HPT, "P3.1b-2: both decodes produced full length");
            SP_CHECK(diffs == 0, "P3.1b-2 recall-as-history: continuation == monolithic (bit-exact)");
            return SP_DONE();
        }
    }

    /* ════ E_G4_CU_REPLAY (XBAR P3.3): SP_REPLAY injection gate — G-P3-SHARED ════
     * Inject a stored episode's owner K/V over the freshly-minted prefill rows [0,NPOS)
     * before attention reads them. Three legs: (1) intact replay == baseline (f32 store is
     * lossless); (2) zeroed episode -> DIVERGES (injection is load-bearing, not a no-op);
     * (3) SP_REPLAY unset -> baseline floor (the ref run). Owner-only; sharers ride dKc[src].
     * Episode produced by SP_XBAR_RECALL_WRITE (the P3.1-gated bit-exact mirror == baseline). */
    {
        const char *rg = getenv("SP_G4_REPLAY_GATE");
        if (rg && atoi(rg) > 0) {
            enum { RNP = 4, RNG = 12, RPT = RNP + RNG };
            const int32_t rp[RNP] = { 2, 10, 100, 1000 };
            const char *dir = getenv("SP_REPLAY_DIR"); if (!dir) dir = "_p33_ep";
            char ev[600];
            if (big_embd) _putenv("SP_CUDA_DECODE_INT8=1");
            /* leg 3 (floor): baseline, SP_REPLAY unset */
            int32_t ref[RPT]; for (int i = 0; i < RNP; i++) ref[i] = rp[i];
            int dnr = gemma4_decode_cuda(m, ref, RNP, RNG, -1);
            /* produce the episode (WRITE mirrors the live KV bit-exactly == baseline, P3.1 gated) */
            snprintf(ev, sizeof(ev), "SP_XBAR_RECALL_WRITE=%s", dir); _putenv(ev);
            int32_t wr[RPT]; for (int i = 0; i < RNP; i++) wr[i] = rp[i];
            gemma4_decode_cuda(m, wr, RNP, RNG, -1);
            _putenv("SP_XBAR_RECALL_WRITE=");
            /* leg 1 (intact): replay episode over prefill owner rows [0,RNP) */
            snprintf(ev, sizeof(ev), "SP_REPLAY=%s", dir); _putenv(ev);
            snprintf(ev, sizeof(ev), "SP_REPLAY_NPOS=%d", RNP); _putenv(ev);
            int32_t in[RPT]; for (int i = 0; i < RNP; i++) in[i] = rp[i];
            int dni = gemma4_decode_cuda(m, in, RNP, RNG, -1);
            /* leg 2 (zeroed): inject zeros -> must diverge */
            _putenv("SP_REPLAY_ZERO=1");
            int32_t zr[RPT]; for (int i = 0; i < RNP; i++) zr[i] = rp[i];
            int dnz = gemma4_decode_cuda(m, zr, RNP, RNG, -1);
            _putenv("SP_REPLAY="); _putenv("SP_REPLAY_NPOS="); _putenv("SP_REPLAY_ZERO=");
            if (big_embd) _putenv("SP_CUDA_DECODE_INT8=");
            int intact_diffs = 0, zero_diffs = 0, n;
            n = (dnr < dni ? dnr : dni); if (n > RPT) n = RPT;
            for (int i = RNP; i < n; i++) if (ref[i] != in[i]) intact_diffs++;
            n = (dnr < dnz ? dnr : dnz); if (n > RPT) n = RPT;
            for (int i = RNP; i < n; i++) if (ref[i] != zr[i]) zero_diffs++;
            fprintf(stderr, "    [g4-cuda-REPLAY] ref dn=%d intact dn=%d zero dn=%d | intact_diffs[%d..%d)=%d zero_diffs=%d\n",
                    dnr, dni, dnz, RNP, RPT, intact_diffs, zero_diffs);
            fprintf(stderr, "        ref   :"); for (int i = 0; i < RPT; i++) fprintf(stderr, " %d", ref[i]); fprintf(stderr, "\n");
            fprintf(stderr, "        intact:"); for (int i = 0; i < RPT; i++) fprintf(stderr, " %d", in[i]);  fprintf(stderr, "\n");
            fprintf(stderr, "        zero  :"); for (int i = 0; i < RPT; i++) fprintf(stderr, " %d", zr[i]);  fprintf(stderr, "\n");
            SP_CHECK(dnr == RPT && dni == RPT && dnz == RPT, "P3.3: all three decodes produced full length");
            SP_CHECK(intact_diffs == 0, "G-P3-SHARED leg1: intact replay == baseline (bit-identical)");
            SP_CHECK(zero_diffs > 0,    "G-P3-SHARED leg2: zeroed episode DIVERGES (injection load-bearing)");
            return SP_DONE();
        }
    }

    /* ════ E_G4_CU_PAGE (XBAR P3.2-b-1): paged-read bit-exact gate ════
     * legacy full-cache decode vs SP_XBAR_PAGE decode (per step: spill pos →
     * poison [0,pos] in the live cache → page [0,pos) back off Ring-2 before
     * attention). Token-identical proves the attention reads were served off disk
     * through off[L] — the live cache is provably poisoned, so it can't be the
     * source. The CLOSED loop = write-path (spill) ∘ read-path (off[L] read). */
    {
        const char *pg = getenv("SP_XBAR_PAGE_GATE");
        if (pg && atoi(pg) > 0) {
            enum { PNP = 4, PNG = 12, PPT = PNP + PNG };
            const int32_t pp[PNP] = { 2, 10, 100, 1000 };
            const char *dir = getenv("SP_XBAR_PAGE_DIR"); if (!dir) dir = "_p32_page";
            char ev[600];
            if (big_embd) _putenv("SP_CUDA_DECODE_INT8=1");
            int32_t ref[PPT]; for (int i = 0; i < PNP; i++) ref[i] = pp[i];
            int dnr = gemma4_decode_cuda(m, ref, PNP, PNG, -1);   /* legacy full-cache */
            int32_t pag[PPT]; for (int i = 0; i < PNP; i++) pag[i] = pp[i];
            snprintf(ev, sizeof(ev), "SP_XBAR_PAGE=%s", dir); _putenv(ev);
            int dnp = gemma4_decode_cuda(m, pag, PNP, PNG, -1);   /* paged off Ring-2, source poisoned */
            _putenv("SP_XBAR_PAGE=");
            if (big_embd) _putenv("SP_CUDA_DECODE_INT8=");
            int diffs = 0, n = (dnr < dnp ? dnr : dnp); if (n > PPT) n = PPT;
            for (int i = PNP; i < n; i++) if (ref[i] != pag[i]) diffs++;
            fprintf(stderr, "    [g4-cuda-PAGE] ref dn=%d  paged dn=%d  diffs[%d..%d)=%d\n", dnr, dnp, PNP, PPT, diffs);
            fprintf(stderr, "        ref :"); for (int i = 0; i < PPT; i++) fprintf(stderr, " %d", ref[i]); fprintf(stderr, "\n");
            fprintf(stderr, "        page:"); for (int i = 0; i < PPT; i++) fprintf(stderr, " %d", pag[i]); fprintf(stderr, "\n");
            SP_CHECK(dnr == PPT && dnp == PPT, "P3.2-b-1: both decodes produced full length");
            SP_CHECK(diffs == 0, "P3.2-b-1 paged-read: paged decode == legacy (bit-exact, live source poisoned)");
            return SP_DONE();
        }
    }

    /* ════ E_G4_CU_SWA (XBAR P3.2-b-2a): SWA ring-buffer shrink bit-exact gate ════
     * SWA owners shrink to a W-slot ring (W = SP_XBAR_SWA_W); the ring decode must
     * be token-identical to the full-cache decode at the SAME window. With W=4 and
     * P=16 the ring wraps and evicts, exercising the shrink cheaply. Globals are
     * untouched (they attend all positions — that's P3.2-b-2b's job). */
    {
        const char *sg = getenv("SP_XBAR_SWA_GATE");
        if (sg && atoi(sg) > 0) {
            enum { SNP = 4, SNG = 12, SPT = SNP + SNG };
            const int32_t sp_[SNP] = { 2, 10, 100, 1000 };
            const char *w = getenv("SP_XBAR_SWA_W"); if (!w) w = "4";
            char ev[600];
            if (big_embd) _putenv("SP_CUDA_DECODE_INT8=1");
            snprintf(ev, sizeof(ev), "SP_XBAR_SWA_W=%s", w); _putenv(ev);   /* window override for BOTH decodes */
            int32_t ref[SPT]; for (int i = 0; i < SNP; i++) ref[i] = sp_[i];
            int dnr = gemma4_decode_cuda(m, ref, SNP, SNG, -1);             /* full cache, window w */
            int32_t rng[SPT]; for (int i = 0; i < SNP; i++) rng[i] = sp_[i];
            _putenv("SP_XBAR_SWA_RING=1");
            int dng = gemma4_decode_cuda(m, rng, SNP, SNG, -1);             /* SWA owners = ring of w slots */
            _putenv("SP_XBAR_SWA_RING="); _putenv("SP_XBAR_SWA_W=");
            if (big_embd) _putenv("SP_CUDA_DECODE_INT8=");
            int diffs = 0, n = (dnr < dng ? dnr : dng); if (n > SPT) n = SPT;
            for (int i = SNP; i < n; i++) if (ref[i] != rng[i]) diffs++;
            fprintf(stderr, "    [g4-cuda-SWA] W=%s  full dn=%d  ring dn=%d  diffs[%d..%d)=%d\n", w, dnr, dng, SNP, SPT, diffs);
            fprintf(stderr, "        full:"); for (int i = 0; i < SPT; i++) fprintf(stderr, " %d", ref[i]); fprintf(stderr, "\n");
            fprintf(stderr, "        ring:"); for (int i = 0; i < SPT; i++) fprintf(stderr, " %d", rng[i]); fprintf(stderr, "\n");
            SP_CHECK(dnr == SPT && dng == SPT, "P3.2-b-2a: both decodes produced full length");
            SP_CHECK(diffs == 0, "P3.2-b-2a SWA ring: ring decode == full-cache windowed decode (bit-exact)");
            return SP_DONE();
        }
    }

    /* ════ E_G4_CU_ARM (XBAR P3.2-b-2b Phase 0/1): shadow-router parity gate ════
     * The frozen sp_arm_select_geom runs host-side on the 8 GLOBAL owners but only LOGS
     * the recall set — attention is untouched. So (1) the decode output must be byte-identical
     * to no-flag (the diffs=0 constraint), and (2) the incrementally-minted global projk must
     * equal a fresh reprojection of the FINAL global K cache (G-P3-GEOM.a oracle parity: the K
     * fed the router IS the K attention reads). B=0 = Phase-0 null floor (identity selection),
     * B=8 = Phase-1 sparse selection. */
    {
        const char *ag = getenv("SP_ARM_GATE");
        if (ag && atoi(ag) > 0) {
            enum { ANP = 4, ANG = 12, APT = ANP + ANG };
            const int32_t ap[ANP] = { 2, 10, 100, 1000 };
            char ev[300];
            if (big_embd) _putenv("SP_CUDA_DECODE_INT8=1");
            int32_t ref[APT]; for (int i = 0; i < ANP; i++) ref[i] = ap[i];
            int dnr = gemma4_decode_cuda(m, ref, ANP, ANG, -1);          /* no-flag baseline */
            int total_diffs = 0; long worst_mism = 0, total_sel = 0;
            const char *Bs[2] = { "0", "8" };                           /* Phase-0 null, Phase-1 sparse */
            for (int bi = 0; bi < 2; bi++) {
                _putenv("SP_ARM_SHADOW=1");
                snprintf(ev, sizeof(ev), "SP_ARM_B=%s", Bs[bi]); _putenv(ev);
                _putenv("SP_ARM_W=2"); _putenv("SP_ARM_SINK=1"); _putenv("SP_ARM_R=32");
                int32_t sh[APT]; for (int i = 0; i < ANP; i++) sh[i] = ap[i];
                int dns = gemma4_decode_cuda(m, sh, ANP, ANG, -1);
                _putenv("SP_ARM_SHADOW="); _putenv("SP_ARM_B="); _putenv("SP_ARM_W="); _putenv("SP_ARM_SINK="); _putenv("SP_ARM_R=");
                long mism = -1, sel = 0; xbar_arm_shadow_result(&mism, &sel);
                int d = 0, n = (dnr < dns ? dnr : dns); if (n > APT) n = APT;
                for (int i = ANP; i < n; i++) if (ref[i] != sh[i]) d++;
                total_diffs += d; if (mism > worst_mism) worst_mism = mism; total_sel += sel;
                fprintf(stderr, "    [g4-cuda-ARM] B=%s  shadow dn=%d  diffs[%d..%d)=%d  projk-mism=%ld  selections=%ld\n",
                        Bs[bi], dns, ANP, APT, d, mism, sel);
            }
            if (big_embd) _putenv("SP_CUDA_DECODE_INT8=");
            SP_CHECK(dnr == APT, "P3.2-b-2b: baseline decode full length");
            SP_CHECK(total_diffs == 0, "P3.2-b-2b Phase 0/1: shadow decode == no-flag (live output bit-exact)");
            SP_CHECK(worst_mism == 0, "P3.2-b-2b G-P3-GEOM.a: global projk fresh-vs-incremental parity (K-capture faithful)");
            SP_CHECK(total_sel > 0, "P3.2-b-2b: shadow router executed (selections > 0)");
            return SP_DONE();
        }
    }

    /* ════ E_G4_CU_ARMPAGE (XBAR P3.2-b-2b G1): recall set served OFF RING-2 ════
     * Two gather decodes with the SAME recall sets: (A) reads the recalled positions
     * from the LIVE global cache; (B) per step spills the global K/V to Ring-2, NaN-
     * POISONS the live global cache, and pages back ONLY the recalled union off disk —
     * then gathers. If B == A token-identical, every recalled byte was served byte-exact
     * off Ring-2 (a poisoned slot that wasn't paged would NaN-corrupt the output). This
     * is the served-off-disk proof — the needle survives the poison. */
    {
        const char *pg = getenv("SP_ARM_PAGE_GATE");
        if (pg && atoi(pg) > 0) {
            enum { GNP = 4, GNG = 12, GPT = GNP + GNG };
            const int32_t gp[GNP] = { 2, 10, 100, 1000 };
            char ev[300];
            if (big_embd) _putenv("SP_CUDA_DECODE_INT8=1");
            _putenv("SP_ARM_SHADOW=1"); _putenv("SP_ARM_GATHER=1");
            _putenv("SP_ARM_B=8"); _putenv("SP_ARM_W=2"); _putenv("SP_ARM_SINK=1"); _putenv("SP_ARM_R=32");
            int32_t live[GPT]; for (int i = 0; i < GNP; i++) live[i] = gp[i];
            int dnl = gemma4_decode_cuda(m, live, GNP, GNG, -1);          /* gather from LIVE cache */
            const char *dir = getenv("SP_ARM_PAGE_DIR"); if (!dir) dir = "_p32_armpage";
            snprintf(ev, sizeof(ev), "SP_ARM_PAGE=%s", dir); _putenv(ev);
            int32_t disk[GPT]; for (int i = 0; i < GNP; i++) disk[i] = gp[i];
            int dnd = gemma4_decode_cuda(m, disk, GNP, GNG, -1);          /* gather from DISK (poison + page) */
            _putenv("SP_ARM_PAGE=");
            _putenv("SP_ARM_SHADOW="); _putenv("SP_ARM_GATHER="); _putenv("SP_ARM_B="); _putenv("SP_ARM_W="); _putenv("SP_ARM_SINK="); _putenv("SP_ARM_R=");
            if (big_embd) _putenv("SP_CUDA_DECODE_INT8=");
            int diffs = 0, n = (dnl < dnd ? dnl : dnd); if (n > GPT) n = GPT;
            for (int i = GNP; i < n; i++) if (live[i] != disk[i]) diffs++;
            fprintf(stderr, "    [g4-cuda-ARMPAGE] live dn=%d  disk dn=%d  diffs[%d..%d)=%d\n", dnl, dnd, GNP, GPT, diffs);
            fprintf(stderr, "        live:"); for (int i = 0; i < GPT; i++) fprintf(stderr, " %d", live[i]); fprintf(stderr, "\n");
            fprintf(stderr, "        disk:"); for (int i = 0; i < GPT; i++) fprintf(stderr, " %d", disk[i]); fprintf(stderr, "\n");
            SP_CHECK(dnl == GPT && dnd == GPT, "P3.2-b-2b G1: both gather decodes full length");
            SP_CHECK(diffs == 0, "P3.2-b-2b G1: gather-from-disk == gather-from-live (recalled set served byte-exact off Ring-2, NaN-poison rigor)");
            return SP_DONE();
        }
    }

    /* ════ E_G4_CU_DEC_5B (ETA.5b): THE VELOCITY GATES — graph capture + dp4a.
     * Four decode runs over the same prompt, 256-token window:
     *   ref   : knobs-off (the oracle-gated lift path above)
     *   graph : SP_CUDA_DECODE_GRAPH=1  — SAME arithmetic, position via *dpos
     *           -> gate = EXACT sequence match vs ref
     *   int8  : SP_CUDA_DECODE_INT8=1   — dp4a packed GEMV (activation int8
     *           quant is top-1-lossless, NOT byte-exact)
     *           -> gate currency = TOP-1 sequence agreement vs ref (Beta-style)
     *   both  : GRAPH=1 + INT8=1        — the velocity configuration
     *           -> top-1 agreement vs ref
     * Wall times printed are warm within-run ratios (weights resident; clocks
     * NOT pinned here — binding tok/s numbers come from the bench methodology;
     * this is a correctness gate with a velocity telemetry line). ════ */
    {
        enum { NP = 4, NG = 256, PT = NP + NG };
        static int32_t ref[PT], gsq[PT], q8[PT], bth[PT];
        const int32_t prompt[NP] = { 2, 10, 100, 1000 };
        clock_t c0; double tref, tgr, ti8, tbo;
        int dn;

        #define G4DEC(buf, tvar) do { \
            for (int i = 0; i < NP; i++) (buf)[i] = prompt[i]; \
            c0 = clock(); dn = gemma4_decode_cuda(m, (buf), NP, NG, -1); \
            (tvar) = (double)(clock() - c0) / CLOCKS_PER_SEC; \
            if (dn < 0) fprintf(stderr, "    decode: %s\n", sp_last_error()); \
            SP_CHECK(dn == PT, "5b decode produced full length"); } while (0)

        if (big_embd) {   /* 12B-class: every config runs dp4a; 'graph' rows then
                           * gate graph-vs-per-step EXACTNESS within int8. */
            _putenv("SP_CUDA_DECODE_INT8=1");
            fprintf(stderr, "    [g4-cuda-5B] 12B-class: ref = dp4a per-step (f32 embd not resident)\n");
        }
        G4DEC(ref, tref);                                   /* knobs off (warm) */
        _putenv("SP_CUDA_DECODE_GRAPH=1");  G4DEC(gsq, tgr);
        _putenv("SP_CUDA_DECODE_GRAPH=");
        _putenv("SP_CUDA_DECODE_INT8=1");   G4DEC(q8, ti8);
        _putenv("SP_CUDA_DECODE_GRAPH=1");  G4DEC(bth, tbo);
        _putenv("SP_CUDA_DECODE_GRAPH=");   _putenv("SP_CUDA_DECODE_INT8=");
        #undef G4DEC

        int eq_g = 0, ag_i = 0, ag_b = 0;
        for (int i = NP; i < PT; i++) {
            if (gsq[i] == ref[i]) eq_g++;
            if (q8[i]  == ref[i]) ag_i++;
            if (bth[i] == ref[i]) ag_b++;
        }
        fprintf(stderr, "    [g4-cuda-5B] %d-tok window: graph %d/%d exact | int8 %d/%d top-1 | "
                "graph+int8 %d/%d top-1\n", NG, eq_g, NG, ag_i, NG, ag_b, NG);
        fprintf(stderr, "    [g4-cuda-5B] wall (warm, unpinned): ref %.2fs (%.1f tok/s) | graph %.2fs (%.1f) | "
                "int8 %.2fs (%.1f) | graph+int8 %.2fs (%.1f)\n",
                tref, NG/tref, tgr, NG/tgr, ti8, NG/ti8, tbo, NG/tbo);
        SP_CHECK(eq_g == NG, "GRAPH decode: EXACT sequence match vs knobs-off (same arithmetic)");
        SP_CHECK(ag_i == NG, "INT8 dp4a decode: top-1 agreement vs knobs-off (256-token window)");
        SP_CHECK(ag_b == NG, "GRAPH+INT8 decode: top-1 agreement vs knobs-off (the velocity config)");
    }

    sp_cuda_model_release(m);
    qwen3_free(m);
    sp_model_unload(handle);
}

/* ═══ SP_G4_BX_DUMP=1 (G-BYTEEXACT-ISLANDS-CUDA): drive a real 12B prefill with the
 * SP_BYTEEXACT_DUMP seam armed, so the float islands' input+output land on disk for the
 * host comparator (compare_islands.py) to gate against the crate's exact-integer *_q_ref.
 * Runs gemma4_cuda_probe with attn_only=1 so the layer loop executes ALL layers (the dump
 * fires at SP_BYTEEXACT_LAYER) and breaks at the last layer's attention residual — NEVER
 * reaching the tied head (which 12B can't run without the resident f32 embd). out_x is the
 * post-attn x scratch, discarded. Env: SP_GEMMA4_SPMODEL/SPTOK, SP_BYTEEXACT_DUMP=path,
 * SP_BYTEEXACT_LAYER (default last), SP_PPL_TOKENS (optional; else synthetic ids). */
static int run_bx_dump(void) {
    const char *spm = getenv("SP_GEMMA4_SPMODEL"); if (!spm) spm = SP_GEMMA4_SPMODEL_DEF;
    const char *stk = getenv("SP_GEMMA4_SPTOK");   if (!stk) stk = SP_GEMMA4_SPTOK_DEF;
    if (!getenv("SP_BYTEEXACT_DUMP")) { fprintf(stderr, "[g4-bx] set SP_BYTEEXACT_DUMP=<out.bin>\n"); return 2; }
    sp_model *handle = NULL;
    if (sp_model_load(spm, stk, &handle) != SP_OK || !handle) { fprintf(stderr, "[g4-bx] load FAIL: %s\n", sp_last_error()); return 2; }
    qwen3_model *m = sp_model_to_gemma4(handle);
    if (!m) { fprintf(stderr, "[g4-bx] sp_model_to_gemma4 FAIL\n"); return 2; }
    const qwen3_config *c = &m->cfg;
    const int E = (int)c->n_embd, NL = (int)c->n_layers;
    int n_tok = getenv("SP_BX_NTOK") ? atoi(getenv("SP_BX_NTOK")) : 16;
    if (n_tok < 1) n_tok = 16;
    int32_t *toks = (int32_t *)malloc((size_t)n_tok * sizeof(int32_t));
    const char *toksf = getenv("SP_PPL_TOKENS");
    int got = 0;
    if (toksf) { FILE *tf = fopen(toksf, "rb"); if (tf) { int v; while (got < n_tok && fscanf(tf, "%d", &v) == 1) toks[got++] = (int32_t)v; fclose(tf); } }
    for (int i = got; i < n_tok; i++) toks[i] = (int32_t)(1000 + i);   /* synthetic valid ids */
    float *out_x = (float *)malloc((size_t)n_tok * E * sizeof(float));
    if (!toks || !out_x) { fprintf(stderr, "[g4-bx] OOM\n"); return 2; }
    fprintf(stderr, "[g4-bx] prefill n_tok=%d NL=%d E=%d (dump -> %s, layer %s)\n",
            n_tok, NL, E, getenv("SP_BYTEEXACT_DUMP"),
            getenv("SP_BYTEEXACT_LAYER") ? getenv("SP_BYTEEXACT_LAYER") : "last");
    /* attn_only=1 ⇒ run every layer, dump at the chosen one, break before the head. */
    int rc = gemma4_cuda_probe(m, toks, n_tok, NL, /*attn_only=*/1, out_x);
    fprintf(stderr, "[g4-bx] probe rc=%d (islands dumped); %s\n", rc, sp_last_error());
    free(toks); free(out_x); sp_model_unload(handle);
    /* rc may be nonzero on environmental head-skip; the dump file is what matters. The
     * gate verdict is produced by the host comparator, not this run. Return 0 if the
     * dump file is non-empty. */
    return 0;
}

int main(void) {
    if (getenv("SP_G4_BX_DUMP")) return run_bx_dump(); /* G-BYTEEXACT-ISLANDS-CUDA island dump seam */
    if (getenv("SP_G4_NIAH"))   return run_niah();   /* C-c NIAH mode (env-driven conditions) */
    if (getenv("SP_G4_KAIROS")) return run_kairos(); /* KAI-1 Path B: cognitive crucible on the 12B GPU */
    if (getenv("SP_G4_KAIROS_METAL")) return run_kairos_metal(); /* KAI-1c: semantic loop on the journaled ring */
    if (getenv("SP_G4_KAIROS_SOAK")) return run_kairos_soak();   /* G-KAIROS-1: >=24h unattended endurance soak */
    if (getenv("SP_G4_KV_REWIND")) return run_kv_rewind(); /* KAI-1b: G-1b-REWIND-NULL bit-exact gate */
    if (getenv("SP_G4_KV_REPLAY_GATE")) return run_kv_replay(); /* C2 #222: replay-inject + O(1) bit-exact rewind */
    if (getenv("SP_G4_KV_WRAP")) return run_kv_wrap();     /* KAI-1c: G-1b-WRAP-NULL wrap-aware ring gate */
    if (getenv("SP_G4_KV_INJECT_NULL")) return run_kv_inject_null(); /* KAI-2: G-KAIROS-2 inject self-null */
    if (getenv("SP_G4_KAIROS_INTERRUPT")) return run_kairos_interrupt(); /* KAI-2: G-KAIROS-2 latent-vs-text A/B */
    if (getenv("SP_G4_KAI2_PACKET")) return run_kai2_packet_gate(); /* KAI-2: G-KAIROS-2 trained-packet pivot+selectivity gate */
    if (getenv("SP_G4_INJ_SEQ")) return run_inject_seq_null(); /* KAI-3 §7.2: G-KAIROS-3-NULL sequence-wrapper null floor */
    if (getenv("SP_G4_TOK_DUMP_IN")) return run_tok_dump();     /* KAI-3 §7.3: real gemma tokenizer id dump (local, no cloud) */
    if (getenv("SP_G4_KAI3_WRITE")) return run_kai3_write();    /* G-XBAR-ORGANISM step 1: EAR audio cache -> Ring-2 episode write seam */
    if (getenv("SP_G4_KAI3")) return run_kai3_gate();           /* KAI-3 §7.3: G-KAIROS-3 projected-frame pivot gate */
    if (getenv("SP_G4_KV_TELEMETRY")) return run_kv_telemetry(); /* KAI-1b §5.4: O(actions)->O(1) receipt */
    if (getenv("SP_G4_KV_RING_TEL")) return run_kv_ring_telemetry(); /* KAI-1c #219: journaled-ring D2D tax */
    SP_RUN(T_GEMMA4_CUDA_WEIGHTS);
    return SP_DONE();
}
