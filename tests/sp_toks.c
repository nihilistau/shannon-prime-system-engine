/* sp_toks.c — SPEED_BASELINE: time qwen3_generate_kv decode throughput (tok/s).
 *
 * Loads the f16 GGUF via qwen3_load, warms once to page weights in, then times
 * n_gen persistent-KV decode steps and prints tok/s. Compare vs llama.cpp on the
 * same model. Measures the SP forward AS-IS (the WIRE-gap scalar/f32 path); the
 * gap vs llama.cpp is what SPEED_WIRE_CPU (packed Q8/Q4 + VNNI) must close.
 *
 * Env: SP_TOKS_N overrides the timed token count (default 32). SP_CPU_SCALAR=1
 *      forces the scalar path; default uses SP's best current CPU path.
 */
#include "sp_engine/model.h"
#include "sp_engine/sp_model.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#ifndef SP_QWEN3_GGUF
#define SP_QWEN3_GGUF "Qwen3-0.6B-f16.gguf"
#endif

static double now_s(void) {
    struct timespec t;
    timespec_get(&t, TIME_UTC);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static int argmax_row(const float *row, int n) {
    int bi = 0; float bv = row[0];
    for (int i = 1; i < n; i++) if (row[i] > bv) { bv = row[i]; bi = i; }
    return bi;
}

int main(void) {
    const char *ng = getenv("SP_TOKS_N");
    int n_gen = ng ? atoi(ng) : 32;
    if (n_gen < 1) n_gen = 32;

    /* Production path: SP_TOKS_SP=<file.sp-model> loads via the reducing-converter
     * .sp-model + swivel adapter (sp_model_load -> sp_model_to_qwen3): packed OK_Q8
     * arena, gguf==NULL, zero quant inflation. The paired .sp-tokenizer is the same
     * path with the extension swapped. Falls back to the raw-GGUF reference loader
     * (qwen3_load) when SP_TOKS_SP is unset. */
    sp_model *spm = NULL;
    qwen3_model *m = NULL;
    const char *sp_path = getenv("SP_TOKS_SP");
    if (sp_path) {
        char tok_path[1024];
        snprintf(tok_path, sizeof(tok_path), "%s", sp_path);
        char *dot = strrchr(tok_path, '.');
        if (dot && strcmp(dot, ".sp-model") == 0) strcpy(dot, ".sp-tokenizer");
        sp_status st = sp_model_load(sp_path, tok_path, &spm);
        if (st != SP_OK || !spm) { fprintf(stderr, "[sp_toks] sp_model_load FAIL (%d): %s\n", (int)st, sp_path); return 1; }
        m = sp_model_to_qwen3(spm);
        if (!m) { fprintf(stderr, "[sp_toks] sp_model_to_qwen3 FAIL: %s\n", sp_path); return 1; }
        fprintf(stderr, "[sp_toks] loaded via swivel: %s (.sp-model OK_Q8 arena, zero-inflation)\n", sp_path);
    } else {
        m = qwen3_load(SP_QWEN3_GGUF);
        if (!m) { fprintf(stderr, "[sp_toks] load FAIL: %s\n", SP_QWEN3_GGUF); return 1; }
    }

    const int n_prompt = 4;
    int32_t *seq = (int32_t *)malloc((size_t)(n_prompt + n_gen) * sizeof(int32_t));
    if (!seq) { fprintf(stderr, "[sp_toks] OOM\n"); return 1; }

    seq[0] = 1; seq[1] = 2; seq[2] = 3; seq[3] = 4;
    (void)qwen3_generate_kv(m, seq, n_prompt, 2, -1);   /* warm: page weights in */

    /* MTP ceiling probe: does forwarding K tokens cost ~the same wall as 1?
     * If yes, the weight read is amortized across K -> speculative MTP buys ~K×
     * tok/s on accept (one weight pass verifies K draft tokens). */
    if (getenv("SP_MTP_CEIL")) {
        const int V = (int)m->cfg.n_vocab, maxK = 8, iters = 8;
        int32_t *tk = (int32_t *)malloc((size_t)maxK * sizeof(int32_t));
        float *lg = (float *)malloc((size_t)maxK * (size_t)V * sizeof(float));
        if (tk && lg) {
            for (int i = 0; i < maxK; i++) tk[i] = i + 1;
            (void)qwen3_forward(m, tk, maxK, lg);   /* warm */
            const int Ks[4] = {1, 2, 4, 8};
            fprintf(stderr, "[mtp-ceil] forwarding K tokens in one pass (matmul weight-read amortization):\n");
            for (int ki = 0; ki < 4; ki++) {
                int K = Ks[ki];
                double t0 = now_s();
                for (int it = 0; it < iters; it++) (void)qwen3_forward(m, tk, K, lg);
                double dt = (now_s() - t0) / iters;
                fprintf(stderr, "[mtp-ceil] K=%d : %.2f ms/forward  =  %.2f ms/token  (ceiling %.2fx vs K=1-per-token)\n",
                        K, dt * 1000.0, dt * 1000.0 / K, 0.0);
            }
        }
        free(tk); free(lg); free(seq); return 0;
    }

    /* MTP speculative decode (T8): prompt-lookup draft -> ONE batched verify
     * forward -> byte-exact greedy accept -> corrected token -> O(1) advance.
     * Acceptance is argmax equality, so the output is BIT-IDENTICAL to plain
     * greedy decode (the regression invariant). Reports accept rate + forwards
     * saved; the wall-clock win = (forwards saved) x the batched-forward ceiling
     * (1.71x at K=8, measured by SP_MTP_CEIL), realized once verify reuses KV. */
    if (getenv("SP_MTP")) {
        const int V = (int)m->cfg.n_vocab, N = 48, K = 8, NG = 2;
        const int cap = n_prompt + N + K + 8;
        int32_t *gd = (int32_t *)malloc((size_t)cap * sizeof(int32_t));
        int32_t *mp = (int32_t *)malloc((size_t)cap * sizeof(int32_t));
        float *lg = (float *)malloc((size_t)cap * (size_t)V * sizeof(float));
        if (gd && mp && lg) {
            /* greedy reference: 1 forward per token */
            int L = n_prompt; for (int i = 0; i < n_prompt; i++) gd[i] = i + 1;
            int gfwd = 0;
            while (L < n_prompt + N) { (void)qwen3_forward(m, gd, L, lg); gfwd++;
                gd[L] = argmax_row(lg + (size_t)(L - 1) * V, V); L++; }
            /* MTP: draft K via prompt-lookup, verify in one batched forward */
            L = n_prompt; for (int i = 0; i < n_prompt; i++) mp[i] = i + 1;
            int mfwd = 0; long acc_sum = 0, acc_steps = 0;
            while (L < n_prompt + N) {
                int32_t draft[16]; int Kd = 0;
                for (int j = L - NG - 1; j >= 0; j--) {            /* rightmost NG-gram match */
                    int mt = 1; for (int g = 0; g < NG; g++) if (mp[j + g] != mp[L - NG + g]) { mt = 0; break; }
                    if (mt) { for (int d = 0; d < K && j + NG + d < L; d++) draft[Kd++] = mp[j + NG + d]; break; }
                }
                int cl = L; for (int d = 0; d < Kd; d++) mp[cl + d] = draft[d]; cl += Kd;
                (void)qwen3_forward(m, mp, cl, lg); mfwd++;
                int na = 0;                                        /* accept longest argmax-matching prefix */
                for (int i = 0; i < Kd; i++) { if (argmax_row(lg + (size_t)(L - 1 + i) * V, V) == draft[i]) na++; else break; }
                mp[L + na] = argmax_row(lg + (size_t)(L - 1 + na) * V, V);   /* corrected token */
                L += na + 1; acc_sum += na; acc_steps++;
            }
            int identical = 1; for (int i = 0; i < n_prompt + N; i++) if (gd[i] != mp[i]) { identical = 0; break; }
            fprintf(stderr, "[mtp] N=%d K=%d : greedy_forwards=%d  mtp_forwards=%d  mean_accept=%.2f/%d  bit_identical_to_greedy=%d\n",
                    N, K, gfwd, mfwd, acc_steps ? (double)acc_sum / acc_steps : 0.0, K, identical);
            fprintf(stderr, "[mtp] forwards saved = %.2fx fewer; realized wall-win = that x the batched ceiling (~1.7x at K=8 with KV-reuse verify)\n",
                    mfwd ? (double)gfwd / mfwd : 0.0);
        }
        free(gd); free(mp); free(lg); free(seq); return 0;
    }

    /* MTP KV-reuse (T8 production path): qwen3_mtp_decode reuses the persistent
     * K/V cache across the batched verify, so the forward-count reduction turns
     * into WALL-CLOCK. Baseline K=0 = incremental-KV greedy on the SAME cache
     * substrate (apples-to-apples); K=8 = prompt-lookup speculation. Asserts the
     * two token streams are byte-identical, then reports tok/s for both. */
    if (getenv("SP_MTP_KV")) {
        const int N = 96, K = 8, NG = 2;
        /* SP_PROMPT_IDS=<file of whitespace-separated token IDs>: use a REAL
         * tokenized prompt instead of the synthetic [1,2,3,4]. A real prompt makes
         * the model emit natural (non-degenerate) text, so prompt-lookup accept is
         * honest, not inflated by a repetitive warm-up sequence. */
        int rp_n = 0; int32_t rp[512];
        const char *rp_path = getenv("SP_PROMPT_IDS");
        if (rp_path) {
            FILE *f = fopen(rp_path, "r");
            if (f) { while (rp_n < 512 && fscanf(f, "%d", &rp[rp_n]) == 1) rp_n++; fclose(f); }
            if (rp_n < 2) { fprintf(stderr, "[mtp-kv] bad SP_PROMPT_IDS: %s\n", rp_path); return 1; }
        }
        int np = rp_n ? rp_n : n_prompt;
        int32_t *g0 = (int32_t *)malloc((size_t)(np + N) * sizeof(int32_t));
        int32_t *gk = (int32_t *)malloc((size_t)(np + N) * sizeof(int32_t));
        if (g0 && gk) {
            for (int i = 0; i < np; i++) { int32_t t = rp_n ? rp[i] : (i + 1); g0[i] = t; gk[i] = t; }
            int n_prompt = np;  /* shadow: the MTP run uses the real prompt length */
            long f0 = 0, fk = 0, as = 0, ast = 0, dummy = 0;
            /* warm both paths once (page weights, prime allocator) */
            (void)qwen3_mtp_decode(m, g0, n_prompt, 8, -1, 0, NG, &dummy, &dummy, &dummy);
            (void)qwen3_mtp_decode(m, gk, n_prompt, 8, -1, K, NG, &dummy, &dummy, &dummy);
            for (int i = 0; i < n_prompt; i++) { int32_t t = rp_n ? rp[i] : (i + 1); g0[i] = t; gk[i] = t; }

            double tb0 = now_s();
            int nb0 = qwen3_mtp_decode(m, g0, n_prompt, N, -1, 0, NG, &f0, &dummy, &dummy);
            double db0 = now_s() - tb0;

            double tbk = now_s();
            int nbk = qwen3_mtp_decode(m, gk, n_prompt, N, -1, K, NG, &fk, &as, &ast);
            double dbk = now_s() - tbk;

            int identical = (nb0 == nbk);
            for (int i = 0; identical && i < nb0; i++) if (g0[i] != gk[i]) identical = 0;

            fprintf(stderr, "[mtp-kv] greedy(K=0): %d tok in %.3fs = %.2f tok/s  (forwards=%ld)\n",
                    nb0 - n_prompt, db0, db0 > 0 ? (nb0 - n_prompt) / db0 : 0.0, f0);
            fprintf(stderr, "[mtp-kv] MTP(K=%d):   %d tok in %.3fs = %.2f tok/s  (forwards=%ld, mean_accept=%.2f/%d)\n",
                    K, nbk - n_prompt, dbk, dbk > 0 ? (nbk - n_prompt) / dbk : 0.0, fk,
                    ast ? (double)as / ast : 0.0, K);
            fprintf(stderr, "[mtp-kv] speedup = %.2fx wall-clock   bit_identical_to_greedy=%d   forwards %ld->%ld (%.2fx fewer)\n",
                    db0 > 0 && dbk > 0 ? db0 / dbk : 0.0, identical, f0, fk, fk ? (double)f0 / fk : 0.0);
        }
        free(g0); free(gk); free(seq); return 0;
    }

    seq[0] = 1; seq[1] = 2; seq[2] = 3; seq[3] = 4;
    double t0 = now_s();
    int n = qwen3_generate_kv(m, seq, n_prompt, n_gen, -1);
    double dt = now_s() - t0;

    fprintf(stderr,
        "[sp_toks] gen %d tokens in %.3f s = %.2f tok/s (prompt=%d, total=%d, model=%s)\n",
        n_gen, dt, (dt > 0 ? n_gen / dt : 0.0), n_prompt, n, SP_QWEN3_GGUF);
    /* token IDs for the top-1 accuracy gate (diff VNNI vs scalar oracle) */
    fprintf(stderr, "[sp_toks] tokens:");
    for (int i = n_prompt; i < n && i < n_prompt + 24; i++) fprintf(stderr, " %d", seq[i]);
    fprintf(stderr, "\n");
    free(seq);
    return 0;
}
