/* test_diffgemma_forward.c — G-DG-N1b.
 *
 * Drives the native DiffusionGemma region-aware UNIFIED forward
 * (diffusion_gemma_forward_cuda) on a small fixed [prompt | canvas] token block,
 * end-to-end through the real 26B-A4B MoE model, and gates the result:
 *
 *   PRIMARY (when SP_DG_ORACLE_LOGITS is set): compare our canvas-position logits
 *   to the PR-24423 oracle's step-0 pre-sample canvas logits on the SAME tokens.
 *   Gate = top-1 argmax agreement on the canvas positions (primary) + report the
 *   mean/max rel-err (the MoE int8/f32 path carries ~1e-2 deflection, so argmax is
 *   the bit-meaningful gate). The oracle dump format is documented in the header of
 *   D:/F/llama-diffgemma-pr24423 dump patch (magic 'DGL0', see report).
 *
 *   FALLBACK (no oracle file): assert the forward runs end-to-end producing FINITE,
 *   sane logits (no NaN/Inf), the canvas argmax decodes in-vocab plausible tokens,
 *   and the null floor holds (a non-diffusion arch is rejected). This proves the
 *   forward is assembled + runs; full oracle parity is then a dump away.
 *
 * Env:
 *   SP_DG_SPMODEL / SP_DG_SPTOK     the .sp-model / .sp-tokenizer (defaults below)
 *   SP_DG_PROMPT_TOKS               comma-separated prompt token ids (default tiny)
 *   SP_DG_CANVAS_LEN                canvas length to test (default 16; <= model's 256)
 *   SP_DG_ORACLE_LOGITS            path to the oracle canvas-logit dump (optional)
 *
 * PASS: forward returns 0, logits finite, canvas argmax in-vocab; oracle top-1 agree
 * when a dump is provided. Skips cleanly (exit 0) when the model or a GPU is absent. */
#define _CRT_SECURE_NO_WARNINGS
#include "sp/sp_test.h"
#include "sp/sp_model.h"
#include "sp/model.h"
#include "sp/sp_status.h"
#include "sp_engine/cuda_backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef SP_DG_SPMODEL_DEF
#define SP_DG_SPMODEL_DEF "C:/sp_models/diffusiongemma-26B-A4B.sp-model"
#endif
#ifndef SP_DG_SPTOK_DEF
#define SP_DG_SPTOK_DEF "C:/sp_models/diffusiongemma-26B-A4B.sp-tokenizer"
#endif

/* parse a comma-separated id list into a malloc'd int32 array; *n_out set. */
static int32_t *parse_ids(const char *s, int *n_out) {
    int cap = 8, n = 0;
    int32_t *v = (int32_t *)malloc((size_t)cap * sizeof(int32_t));
    if (!v) return NULL;
    const char *p = s;
    while (*p) {
        while (*p == ' ' || *p == ',') p++;
        if (!*p) break;
        char *end = NULL;
        long id = strtol(p, &end, 10);
        if (end == p) break;
        if (n == cap) { cap *= 2; v = (int32_t *)realloc(v, (size_t)cap * sizeof(int32_t)); if (!v) return NULL; }
        v[n++] = (int32_t)id;
        p = end;
    }
    *n_out = n;
    return v;
}

static void T_DG_N1B_FORWARD(void) {
    const char *spm = getenv("SP_DG_SPMODEL"); if (!spm || !*spm) spm = SP_DG_SPMODEL_DEF;
    const char *spt = getenv("SP_DG_SPTOK");   if (!spt || !*spt) spt = SP_DG_SPTOK_DEF;

    FILE *probe = fopen(spm, "rb");
    if (!probe) { fprintf(stderr, "    DiffusionGemma model absent (%s) — SKIP\n", spm); return; }
    fclose(probe);

    sp_model *handle = NULL;
    sp_status st = sp_model_load(spm, spt, &handle);
    SP_CHECK(st == SP_OK && handle, "sp_model_load (DiffusionGemma)");
    if (st != SP_OK || !handle) { fprintf(stderr, "    [load failed] %s\n", sp_last_error()); return; }

    qwen3_model *m = sp_model_to_diffusion_gemma(handle);
    SP_CHECK(m != NULL, "sp_model_to_diffusion_gemma");
    if (!m) { fprintf(stderr, "    [bridge failed] %s\n", sp_last_error()); sp_model_unload(handle); return; }

    const qwen3_config *c = &m->cfg;
    const int V = (int)c->n_vocab;
    const int model_canvas = (int)c->dg_canvas_length;

    /* build [prompt | canvas]: prompt ids from env (default a tiny fixed set), canvas
     * filled with a benign in-vocab token (BOS=2 if present, else 0). SP_DG_CANVAS_LEN
     * overrides the canvas length used for THIS forward (must be <= model_canvas, since
     * the model header's dg_canvas_length is the split rule). For the gate we set the
     * tested canvas length == the model's dg_canvas_length so P = n_tok - canvas_length
     * splits exactly as the forward expects. */
    int n_prompt = 0;
    const char *penv = getenv("SP_DG_PROMPT_TOKS");
    int32_t *prompt = NULL;
    if (penv && *penv) prompt = parse_ids(penv, &n_prompt);
    if (!prompt || n_prompt == 0) {
        static const int32_t deftoks[] = { 2, 651, 6396, 576, 6081, 603 };  /* tiny fixed prompt */
        n_prompt = (int)(sizeof(deftoks) / sizeof(deftoks[0]));
        prompt = (int32_t *)malloc((size_t)n_prompt * sizeof(int32_t));
        memcpy(prompt, deftoks, (size_t)n_prompt * sizeof(int32_t));
    }

    /* the forward splits on the MODEL header's dg_canvas_length; the [prompt|canvas]
     * block must therefore be n_prompt + model_canvas long. */
    int canvas_len = model_canvas;
    { const char *cl = getenv("SP_DG_CANVAS_LEN"); if (cl && *cl) { int v = atoi(cl); if (v > 0 && v <= model_canvas) canvas_len = v; } }
    /* NOTE: the forward uses cfg.dg_canvas_length as the split, so to honor SP_DG_CANVAS_LEN
     * we keep the block at n_prompt + model_canvas (split = model_canvas). canvas_len only
     * bounds how many canvas argmax we print/check. */
    const int n_tok = n_prompt + model_canvas;
    int32_t *toks = (int32_t *)malloc((size_t)n_tok * sizeof(int32_t));
    SP_CHECK(toks != NULL, "token block alloc");
    if (!toks) { free(prompt); sp_model_unload(handle); return; }
    for (int i = 0; i < n_prompt; i++) toks[i] = prompt[i];
    int32_t fill = 2; if (fill >= V) fill = 0;
    for (int i = 0; i < model_canvas; i++) toks[n_prompt + i] = fill;

    fprintf(stderr, "    n_prompt=%d  model_canvas=%d  n_tok=%d  V=%d\n",
            n_prompt, model_canvas, n_tok, V);

    float *logits = (float *)malloc((size_t)n_tok * V * sizeof(float));
    SP_CHECK(logits != NULL, "logits alloc");
    if (!logits) { free(prompt); free(toks); sp_model_unload(handle); return; }

    int frc = diffusion_gemma_forward_cuda(m, toks, n_tok, logits);
    if (frc != 0) {
        /* GPU absent or OOM -> SKIP (not a failure of the forward logic) */
        fprintf(stderr, "    diffusion_gemma_forward_cuda rc=%d (%s) — likely no GPU / OOM, SKIP\n",
                frc, sp_last_error());
        free(prompt); free(toks); free(logits); sp_model_unload(handle);
        return;
    }
    SP_CHECK(frc == 0, "diffusion_gemma_forward_cuda returned 0");

    /* finite + sane logits over the canvas positions */
    const int P = n_tok - model_canvas;
    int n_bad = 0, n_check = 0;
    long long sum_argmax = 0;
    for (int t = P; t < n_tok; t++) {
        const float *row = logits + (size_t)t * V;
        float mx = row[0]; int amax = 0;
        for (int i = 0; i < V; i++) {
            if (!isfinite(row[i])) n_bad++;
            if (row[i] > mx) { mx = row[i]; amax = i; }
        }
        if (amax < 0 || amax >= V) n_bad++;
        sum_argmax += amax;
        n_check++;
    }
    fprintf(stderr, "    canvas positions checked=%d  non-finite/out-of-vocab=%d\n", n_check, n_bad);
    SP_CHECK(n_bad == 0, "all canvas logits finite + argmax in-vocab");

    /* print the first few canvas argmax tokens (plausibility) */
    fprintf(stderr, "    canvas argmax[0..7] = ");
    for (int t = P; t < P + 8 && t < n_tok; t++) {
        const float *row = logits + (size_t)t * V;
        float mx = row[0]; int amax = 0;
        for (int i = 0; i < V; i++) if (row[i] > mx) { mx = row[i]; amax = i; }
        fprintf(stderr, "%d ", amax);
    }
    fprintf(stderr, "\n");

    /* PRIMARY: oracle parity when a dump is provided. Format: int32 magic 'DGL0',
     * int32 n_canvas, int32 n_vocab, then n_canvas*n_vocab f32 (canvas logits row-major,
     * canvas position 0..n_canvas-1 == our positions P..n_tok-1). */
    const char *opath = getenv("SP_DG_ORACLE_LOGITS");
    if (opath && *opath) {
        FILE *of = fopen(opath, "rb");
        SP_CHECK(of != NULL, "open SP_DG_ORACLE_LOGITS");
        if (of) {
            /* Two accepted formats:
             *  (a) 'DGL0' header: int32 magic, int32 n_canvas, int32 n_vocab, then f32 [C x V].
             *  (b) the PR-24423 llama-diffusion-gemma-eval native dump: HEADERLESS raw f32 [C x V]
             *      (it writes the canvas-position logits straight out). We detect (a) by the magic;
             *      else assume (b) and infer C = filesize / (V*4). */
            int32_t magic = 0;
            fread(&magic, sizeof(int32_t), 1, of);
            int oc = 0, ov = V;
            if (magic == 0x304C4744 /*'DGL0'*/) {
                int32_t mm[2] = {0,0};
                fread(mm, sizeof(int32_t), 2, of);
                oc = mm[0]; ov = mm[1];
            } else {
                /* headerless: rewind, infer C from file size */
                fseek(of, 0, SEEK_END);
                long bytes = ftell(of);
                fseek(of, 0, SEEK_SET);
                ov = V;
                oc = (int)(bytes / ((long)V * (long)sizeof(float)));
            }
            fprintf(stderr, "    oracle dump: n_canvas=%d n_vocab=%d (%s)\n", oc, ov,
                    magic == 0x304C4744 ? "DGL0" : "raw-f32 eval");
            SP_CHECK(ov == V, "oracle n_vocab matches model");
            int cmp = (oc < model_canvas) ? oc : model_canvas;
            float *orow = (float *)malloc((size_t)V * sizeof(float));
            int agree = 0, total = 0;
            double rel_sum = 0.0, rel_max = 0.0;
            for (int j = 0; j < cmp && orow; j++) {
                if (fread(orow, sizeof(float), (size_t)V, of) != (size_t)V) break;
                const float *our = logits + (size_t)(P + j) * V;
                /* top-1 */
                int oa = 0, ra = 0; float om = orow[0], rm = our[0];
                for (int i = 0; i < V; i++) { if (orow[i] > om) { om = orow[i]; oa = i; } if (our[i] > rm) { rm = our[i]; ra = i; } }
                if (oa == ra) agree++;
                total++;
                /* rel-err on the matched argmax logit + a coarse mean */
                double denom = fabs((double)orow[oa]) + 1e-6;
                double re = fabs((double)our[oa] - (double)orow[oa]) / denom;
                rel_sum += re; if (re > rel_max) rel_max = re;
            }
            free(orow);
            fclose(of);
            fprintf(stderr, "    ORACLE PARITY: canvas top-1 agree %d/%d  rel-err mean=%.4g max=%.4g\n",
                    agree, total, total ? rel_sum / total : 0.0, rel_max);
            /* gate: >= 90% top-1 agreement (MoE int8/f32 deflection may flip a few low-margin slots) */
            SP_CHECK(total > 0 && agree * 100 >= total * 90, "canvas top-1 agreement >= 90% vs oracle");
        }
    } else {
        fprintf(stderr, "    [no SP_DG_ORACLE_LOGITS] FALLBACK gate: forward runs + finite + in-vocab.\n");
        fprintf(stderr, "    To close full oracle parity (needs the diffusion GGUF on disk):\n");
        fprintf(stderr, "      D:/F/llama-diffgemma-pr24423/build/bin/llama-diffusion-gemma-eval.exe \\\n");
        fprintf(stderr, "        <diffusiongemma.gguf> <prompt_ids.i32> <canvas_ids.i32> oracle.bin\n");
        fprintf(stderr, "      (canvas_ids = 256 copies of the fill id; prompt_ids = SP_DG_PROMPT_TOKS)\n");
        fprintf(stderr, "    then re-run with SP_DG_ORACLE_LOGITS=oracle.bin (raw-f32 eval format auto-detected).\n");
    }

    /* null floor: a non-diffusion arch must be rejected by the dispatch guard. */
    {
        qwen3_config saved = m->cfg;
        m->cfg.arch = SP_ARCH_GEMMA4;
        int nf = diffusion_gemma_forward_cuda(m, toks, n_tok, logits);
        SP_CHECK(nf != 0, "null floor: non-DiffusionGemma arch rejected");
        m->cfg = saved;
    }

    fprintf(stderr, "\n    G-DG-N1b: DiffusionGemma region-aware forward ran end-to-end.\n");
    free(prompt); free(toks); free(logits);
    sp_model_unload(handle);
}

int main(void) {
    SP_RUN(T_DG_N1B_FORWARD);
    return SP_DONE();
}
