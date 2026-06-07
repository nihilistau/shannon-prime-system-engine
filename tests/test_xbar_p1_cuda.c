/* test_xbar_p1_cuda.c — XBAR-P1 Inception Probe driver (CONTRACT-XBAR-P1).
 *
 * A deliberately thin gemma4 CUDA decode runner: load artifact, read a
 * token-id prompt fixture, decode N greedy tokens (or teacher-force score),
 * write the full sequence out. ALL probe behavior lives in the engine's
 * SP_XBAR_* knobs (cuda_forward.cu) and passes through this process's
 * environment — the runner adds nothing, so every arm of the probe is the
 * SAME binary with different env (the isolation discipline).
 *
 * Env:
 *   SP_XBAR_SPMODEL / SP_XBAR_SPTOK  artifact (default: the B1 06-R10 artifact)
 *   SP_XBAR_PROMPT     token-id file (whitespace-separated; BOS first)
 *   SP_XBAR_NGEN       greedy tokens to generate (default 64)
 *   SP_XBAR_OUT        write the full token sequence, one id per line
 *   SP_XBAR_SCORE_FIRST  if set: teacher-forced scoring over the WHOLE token
 *                        file from this position (engine SP_G4_SCORE lane);
 *                        prints nll/count/ppl — the G2 coherence currency.
 *   SP_XBAR_AT/ROW/CAPTURE/SPLICE/MASK/POSFREE/RESID/RANKS/TOKENS  pass through.
 *
 * 12B head is not f32-resident -> the dp4a route is forced (SP_CUDA_DECODE_INT8=1),
 * same configuration the 06-R10 citable number was gated on. Graph stays OFF
 * (the engine declines it when XBAR knobs are set; we don't set it).
 */
#define _CRT_SECURE_NO_WARNINGS
#include "sp/sp_test.h"
#include "sp/sp_model.h"
#include "sp/model.h"
#include "sp/sp_status.h"
#include "sp/forward_dispatch.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef SP_XBAR_SPMODEL_DEF
#define SP_XBAR_SPMODEL_DEF "D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-model"
#endif
#ifndef SP_XBAR_SPTOK_DEF
#define SP_XBAR_SPTOK_DEF "D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-tokenizer"
#endif

int  sp_cuda_device_count(void);
int  gemma4_decode_cuda(const qwen3_model *m, int32_t *seq, int n_prompt,
                        int n_gen, int eos_id);
void gemma4_score_result(double *nll, long *cnt);
void sp_cuda_model_release(const qwen3_model *m);

/* the documented cross-seam alias (cf. test_gemma4_cuda.c) */
const float *as_f32(const qwen3_model *m, const gguf_tensor *t) { return sp_as_f32(m, t); }

static long read_tokens(const char *path, int32_t **out) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    long cap = 4096, n = 0;
    int32_t *a = (int32_t *)malloc((size_t)cap * sizeof(int32_t));
    if (!a) { fclose(f); return -1; }
    int v;
    while (fscanf(f, "%d", &v) == 1) {
        if (n >= cap) { cap *= 2; a = (int32_t *)realloc(a, (size_t)cap * sizeof(int32_t)); if (!a) { fclose(f); return -1; } }
        a[n++] = v;
    }
    fclose(f);
    *out = a;
    return n;
}

static void XBAR_P1_RUN(void) {
    const char *spm = getenv("SP_XBAR_SPMODEL"); if (!spm) spm = SP_XBAR_SPMODEL_DEF;
    const char *spt = getenv("SP_XBAR_SPTOK");   if (!spt) spt = SP_XBAR_SPTOK_DEF;
    const char *pp  = getenv("SP_XBAR_PROMPT");
    if (!pp) { fprintf(stderr, "    SP_XBAR_PROMPT unset — SKIP\n"); return; }
    FILE *probe = fopen(spm, "rb");
    if (!probe) { fprintf(stderr, "    model absent (%s) — SKIP\n", spm); return; }
    fclose(probe);
    if (sp_cuda_device_count() < 1) { fprintf(stderr, "    no CUDA device — SKIP\n"); return; }

    const int n_gen = getenv("SP_XBAR_NGEN") ? atoi(getenv("SP_XBAR_NGEN")) : 64;
    const char *outp = getenv("SP_XBAR_OUT");
    const char *scf  = getenv("SP_XBAR_SCORE_FIRST");

    int32_t *toks = NULL;
    long nt = read_tokens(pp, &toks);
    SP_CHECK(nt > 1, "prompt fixture read");
    if (nt <= 1) { free(toks); return; }

    sp_model *handle = NULL;
    SP_CHECK(sp_model_load(spm, spt, &handle) == SP_OK && handle, "sp_model_load");
    if (!handle) { free(toks); return; }
    qwen3_model *m = sp_model_to_gemma4(handle);
    if (!m) fprintf(stderr, "    bridge: %s\n", sp_last_error());
    SP_CHECK(m != NULL, "sp_model_to_gemma4");
    if (!m) { sp_model_unload(handle); free(toks); return; }

    _putenv("SP_CUDA_DECODE_INT8=1");          /* 12B tied head needs the dp4a route */

    if (scf) {
        /* ── G2 lane: teacher-forced scoring over the whole fixture ── */
        char env[32]; snprintf(env, sizeof env, "SP_G4_SCORE=%s", scf);
        _putenv(env);
        int dn = gemma4_decode_cuda(m, toks, (int)nt, 0, -1);
        _putenv("SP_G4_SCORE=");
        SP_CHECK(dn >= 0, "teacher-forced score decode");
        if (dn >= 0) {
            double nll; long cnt;
            gemma4_score_result(&nll, &cnt);
            SP_CHECK(cnt > 0, "scored positions > 0");
            if (cnt > 0)
                fprintf(stderr, "    [xbar-score] nll=%.8f n=%ld ppl=%.6f (first=%s, n_tok=%ld)\n",
                        nll, cnt, exp(nll / (double)cnt), scf, nt);
        }
    } else {
        /* ── decode lane: B0 / G0 / Arm A / Arm B (env decides) ── */
        const long P = nt + n_gen;
        int32_t *seq = (int32_t *)malloc((size_t)P * sizeof(int32_t));
        SP_CHECK(seq != NULL, "sequence buffer");
        if (seq) {
            for (long i = 0; i < nt; i++) seq[i] = toks[i];
            int dn = gemma4_decode_cuda(m, seq, (int)nt, n_gen, -1);
            SP_CHECK(dn > 0, "gemma4_decode_cuda");
            if (dn > 0) {
                fprintf(stderr, "    [xbar-gen] n_prompt=%ld n_out=%d tail:", nt, dn);
                for (int i = (int)nt; i < dn; i++) fprintf(stderr, " %d", seq[i]);
                fprintf(stderr, "\n");
                if (outp) {
                    FILE *of = fopen(outp, "w");
                    SP_CHECK(of != NULL, "open SP_XBAR_OUT");
                    if (of) {
                        for (int i = 0; i < dn; i++) fprintf(of, "%d\n", seq[i]);
                        fclose(of);
                    }
                }
            } else fprintf(stderr, "    decode: %s\n", sp_last_error());
            free(seq);
        }
    }

    free(toks);
    sp_cuda_model_release(m);
    qwen3_free(m);
    sp_model_unload(handle);
}

int main(void) {
    SP_RUN(XBAR_P1_RUN);
    return SP_DONE();
}
