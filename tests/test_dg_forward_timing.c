/* test_dg_forward_timing.c — N5c surgical instrument.
 * Loads DiffusionGemma once, runs SP_DGT_NFWD single diffusion forwards over a tiny
 * [prompt|canvas] token array, and cudaEvent-times EACH forward. No corpus, no judge,
 * no denoise loop -> bounded (~1 forward), zero runaway risk (the test_diffjudge_*
 * harnesses ignore SP_DJ_LIMIT and run the whole 90+50 corpus). The first forward warms
 * the reservoir clone cache; later forwards are steady-state. Honors SP_DG_RESERVOIR.
 *   SP_DGT_SPMODEL / SP_DGT_SPTOK : model + tokenizer (defaults below)
 *   SP_DGT_CANVAS  (default 8)    : canvas length (small keeps each forward fast)
 *   SP_DGT_NFWD    (default 3)    : how many timed forwards
 */
#include "sp/model.h"
#include "sp/sp_status.h"
#include "sp_engine/cuda_backend.h"
#include "sp_engine/tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <cuda_runtime.h>

#ifndef SP_DGT_SPMODEL_DEF
#define SP_DGT_SPMODEL_DEF "C:/sp_models/diffusiongemma-26B-A4B.sp-model"
#endif
#ifndef SP_DGT_SPTOK_DEF
#define SP_DGT_SPTOK_DEF "C:/sp_models/diffusiongemma-26B-A4B.sp-tokenizer"
#endif

int main(void) {
    const char *spm = getenv("SP_DGT_SPMODEL"); if (!spm || !*spm) spm = SP_DGT_SPMODEL_DEF;
    const char *spt = getenv("SP_DGT_SPTOK");   if (!spt || !*spt) spt = SP_DGT_SPTOK_DEF;
    int canvas = 8; { const char *e = getenv("SP_DGT_CANVAS"); if (e && *e) { int v = atoi(e); if (v > 0) canvas = v; } }
    int nfwd   = 3; { const char *e = getenv("SP_DGT_NFWD");   if (e && *e) { int v = atoi(e); if (v > 0) nfwd = v; } }
    const char *res = getenv("SP_DG_RESERVOIR"); res = (res && *res && *res != '0') ? "ON" : "off";

    sp_tokenizer *tk = sp_tokenizer_load_tokfile(spt);
    if (!tk) { fprintf(stderr, "FATAL: tokenizer load: %s\n", sp_last_error()); return 2; }

    sp_model *handle = NULL;
    sp_status st = sp_model_load(spm, spt, &handle);
    if (st != SP_OK || !handle) { fprintf(stderr, "FATAL: sp_model_load: %s\n", sp_last_error()); return 2; }
    qwen3_model *m = sp_model_to_diffusion_gemma(handle);
    if (!m) { fprintf(stderr, "FATAL: sp_model_to_diffusion_gemma: %s\n", sp_last_error()); sp_model_unload(handle); return 2; }
    const qwen3_config *c = &m->cfg;
    const int V = (int)c->n_vocab;
    int CL = (int)c->dg_canvas_length; if (canvas < CL) CL = canvas; if (CL < 1) CL = 1;
    printf("# DGT: V=%d CL=%d nfwd=%d reservoir=%s\n", V, CL, nfwd, res); fflush(stdout);

    int32_t toks[2048];
    long np = sp_tokenizer_encode(tk, "The capital of France is", 24, 1, toks, 64);
    if (np <= 0) { fprintf(stderr, "FATAL: encode\n"); sp_model_unload(handle); return 2; }
    /* SP_DGT_NTOK: target total n_tok (prefill-width timing). Tile the PROMPT region (dynamic,
     * safe) up to target-CL, then CL canvas-fill. The weight-GEMM batch dim = n_tok regardless
     * of the prompt/canvas split, so this faithfully exercises the prefill matmul cost. */
    int target = 0; { const char *e = getenv("SP_DGT_NTOK"); if (e && *e) { int v = atoi(e); if (v > 0) target = v; } }
    if (target > 0) {
        int cap = 2048 - CL;
        int want = target - CL; if (want < (int)np) want = (int)np; if (want > cap) want = cap;
        int base = (int)np;
        for (int i = base; i < want; i++) toks[i] = toks[(i - base) % base];   /* tile prompt tokens */
        np = want;
    }
    int n_tok = (int)np + CL;
    if (n_tok > 2048) n_tok = 2048;
    for (int i = (int)np; i < n_tok; i++) toks[i] = 1;        /* canvas fill = valid id */

    float *logits = (float *)malloc((size_t)n_tok * V * sizeof(float));
    if (!logits) { fprintf(stderr, "FATAL: logits OOM\n"); sp_model_unload(handle); return 2; }

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int f = 0; f < nfwd; f++) {
        cudaEventRecord(e0, 0);
        int rc = diffusion_gemma_forward_cuda(m, toks, n_tok, logits);
        cudaEventRecord(e1, 0); cudaEventSynchronize(e1);
        float ms = 0.0f; cudaEventElapsedTime(&ms, e0, e1);
        printf("[DGT] forward %d/%d  rc=%d  n_tok=%d  %.1f ms  (reservoir=%s)\n",
               f + 1, nfwd, rc, n_tok, ms, res); fflush(stdout);
        if (rc != 0) { fprintf(stderr, "forward rc=%d: %s\n", rc, sp_last_error()); break; }
    }
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    free(logits); sp_tokenizer_free(tk); sp_model_unload(handle);
    return 0;
}