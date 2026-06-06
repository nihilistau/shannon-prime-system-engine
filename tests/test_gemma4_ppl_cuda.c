/* test_gemma4_ppl_cuda.c — M_GEMMA4_CUDA_PPL: the ETA.5b PPL gate.
 *
 * The 12B shootout number (SP 34.2 vs llama.cpp-CUDA 31.29 tok/s, +9.3%) is
 * NOT citable until this gate closes: the SP artifact squeezes the source's
 * Q6_K tensors into Q4 codes — fewer bytes read (part of the speed win), more
 * weight-quant error than llama.cpp's mixed K-quants. This gate measures the
 * wikitext perplexity of the SP 12B artifact on the GPU eval path
 * (SP_G4_SCORE teacher-forced scoring, dp4a head) and compares it against the
 * llama.cpp-CUDA perplexity of the SOURCE GGUF over the SAME token budget.
 *
 * PROTOCOL (mirrors llama-perplexity + the M_GEMMA4 sp_perplexity convention):
 * chunks of n_ctx tokens, BOS re-anchored at each chunk's position 0, scored
 * positions [n_ctx/2, n_ctx-1) per chunk, full f64 log-softmax NLL.
 * Token-parity: the fixture is a token-ID dump of the SAME text (sp_tok_dump,
 * the gated-parity SP tokenizer; llama-perplexity tokenizes the same file with
 * the same gemma vocab).
 *
 * COMPARISON CURRENCY: the DELTA between the two engines on identical text is
 * the gate quantity — the absolute wikitext PPL of an instruction-tuned
 * channel-format model is high for BOTH engines and cancels.
 *
 * Env:
 *   SP_GEMMA4_SPMODEL / SP_GEMMA4_SPTOK  the SP artifact (12B)
 *   SP_PPL_TOKENS   token-id fixture (REQUIRED; whitespace-separated IDs, BOS first)
 *   SP_PPL_NCTX     chunk length        (default 512 — match the llama run)
 *   SP_PPL_CHUNKS   chunk count         (default 8   — match the llama run)
 *   SP_PPL_ORACLE   llama.cpp PPL on the same text/protocol (default unset -> report-only)
 *   SP_PPL_GATE     rel tolerance vs oracle (default 8e-2 — the M_GEMMA4 smoke-bound currency)
 */
#define _CRT_SECURE_NO_WARNINGS
#include "sp/sp_test.h"
#include "sp/sp_model.h"
#include "sp/model.h"
#include "sp/sp_status.h"
#include "sp/forward_dispatch.h"   /* sp_as_f32 (the cross-seam shim target) */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef SP_GEMMA4_SPMODEL_DEF
#define SP_GEMMA4_SPMODEL_DEF "D:/F/shannon-prime-repos/models/gemma4-12b.sp-model"
#endif
#ifndef SP_GEMMA4_SPTOK_DEF
#define SP_GEMMA4_SPTOK_DEF "D:/F/shannon-prime-repos/models/gemma4-12b.sp-tokenizer"
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

static void M_GEMMA4_CUDA_PPL(void) {
    const char *spm = getenv("SP_GEMMA4_SPMODEL"); if (!spm) spm = SP_GEMMA4_SPMODEL_DEF;
    const char *spt = getenv("SP_GEMMA4_SPTOK");   if (!spt) spt = SP_GEMMA4_SPTOK_DEF;
    const char *tokpath = getenv("SP_PPL_TOKENS");
    if (!tokpath) { fprintf(stderr, "    SP_PPL_TOKENS unset — SKIP\n"); return; }
    FILE *probe = fopen(spm, "rb");
    if (!probe) { fprintf(stderr, "    model absent (%s) — SKIP\n", spm); return; }
    fclose(probe);
    if (sp_cuda_device_count() < 1) { fprintf(stderr, "    no CUDA device — SKIP\n"); return; }

    const int n_ctx    = getenv("SP_PPL_NCTX")   ? atoi(getenv("SP_PPL_NCTX"))   : 512;
    const int n_chunks = getenv("SP_PPL_CHUNKS") ? atoi(getenv("SP_PPL_CHUNKS")) : 8;
    const double oracle = getenv("SP_PPL_ORACLE") ? atof(getenv("SP_PPL_ORACLE")) : 0.0;
    const double gate   = getenv("SP_PPL_GATE")   ? atof(getenv("SP_PPL_GATE"))   : 8.0e-2;

    int32_t *toks = NULL;
    long nt = read_tokens(tokpath, &toks);
    SP_CHECK(nt >= (long)n_ctx * n_chunks, "token fixture covers the chunk budget");
    if (nt < (long)n_ctx * n_chunks) { free(toks); return; }

    sp_model *handle = NULL;
    SP_CHECK(sp_model_load(spm, spt, &handle) == SP_OK && handle, "sp_model_load (12B)");
    if (!handle) { free(toks); return; }
    qwen3_model *m = sp_model_to_gemma4(handle);
    if (!m) fprintf(stderr, "    bridge: %s\n", sp_last_error());
    SP_CHECK(m != NULL, "sp_model_to_gemma4");
    if (!m) { sp_model_unload(handle); free(toks); return; }

    const int first = n_ctx / 2;
    const int32_t bos = toks[0];
    char scorenv[32];
    snprintf(scorenv, sizeof scorenv, "SP_G4_SCORE=%d", first);
    _putenv(scorenv);
    _putenv("SP_CUDA_DECODE_INT8=1");      /* 12B head requires the dp4a route */

    int32_t *chunk = (int32_t *)malloc((size_t)n_ctx * sizeof(int32_t));
    SP_CHECK(chunk != NULL, "chunk buffer");
    double nll = 0.0; long count = 0; int rc = 0;
    for (int c = 0; c < n_chunks && chunk; c++) {
        long start = (long)c * n_ctx;
        chunk[0] = bos;
        for (int k = 1; k < n_ctx; k++) chunk[k] = toks[start + k];
        int dn = gemma4_decode_cuda(m, chunk, n_ctx, 0, -1);
        if (dn < 0) { fprintf(stderr, "    score decode: %s\n", sp_last_error()); rc = 1; break; }
        double cn; long cc;
        gemma4_score_result(&cn, &cc);
        nll += cn; count += cc;
        fprintf(stderr, "    [g4-ppl] chunk %d: ppl %.4f (n=%ld)\n", c + 1, exp(cn / (double)cc), cc);
        fflush(stderr);
    }
    _putenv("SP_G4_SCORE=");
    _putenv("SP_CUDA_DECODE_INT8=");
    SP_CHECK(rc == 0 && count > 0, "teacher-forced scoring over all chunks");

    /* SP_PPL_ORACLE_DIFF=1 (bug hunt): run the CPU ORACLE (gemma4_forward, the
     * exact-lift arithmetic, softcap applied internally) over chunk 0 and print
     * the POST-cap max/target logits at the first scored positions — the
     * reference half of the logit intercept — plus the host-recomputed chunk
     * NLL under BOTH loop-bound conventions (ours: targets [first, n_ctx);
     * M_GEMMA4: targets [first+1, n_ctx)) to reconcile the scorer. CPU 12B is
     * hours — use on E2B-class models only. */
    if (getenv("SP_PPL_ORACLE_DIFF") && chunk) {
        const int V = (int)m->cfg.n_vocab;
        float *ol = (float *)malloc((size_t)n_ctx * (size_t)V * sizeof(float));
        if (ol) {
            chunk[0] = bos;
            for (int k = 1; k < n_ctx; k++) chunk[k] = toks[k];
            if (gemma4_forward(m, chunk, n_ctx, ol) == 0) {
                for (int p = first - 1; p < first + 4 && p < n_ctx - 1; p++) {
                    const float *lg = ol + (size_t)p * V;
                    const int tgt = chunk[p + 1];
                    float mx = lg[0]; int mi = 0;
                    for (int i = 1; i < V; i++) if (lg[i] > mx) { mx = lg[i]; mi = i; }
                    fprintf(stderr, "    [g4-oracle-dbg] pos %d POST-cap: max %.4f (id %d) target[%d] %.4f\n",
                            p, (double)mx, mi, tgt, (double)lg[tgt]);
                }
                double nA = 0.0, nB = 0.0; long cA = 0, cB = 0;
                for (int p = first - 1; p < n_ctx - 1; p++) {
                    const float *lg = ol + (size_t)p * V;
                    const int tgt = chunk[p + 1];
                    double mx = lg[0];
                    for (int i = 1; i < V; i++) if (lg[i] > mx) mx = lg[i];
                    double se = 0.0;
                    for (int i = 0; i < V; i++) se += exp((double)lg[i] - mx);
                    double nll1 = -((double)lg[tgt] - mx - log(se));
                    nA += nll1; cA++;                       /* ours: p in [first-1, n_ctx-1) */
                    if (p >= first) { nB += nll1; cB++; }   /* M_GEMMA4: p in [first, n_ctx-1) */
                }
                fprintf(stderr, "    [g4-oracle-dbg] CPU-oracle chunk-0 PPL: ours-bounds %.4f (n=%ld) | "
                                "M_GEMMA4-bounds %.4f (n=%ld)\n",
                        exp(nA / (double)cA), cA, exp(nB / (double)cB), cB);
            } else {
                fprintf(stderr, "    [g4-oracle-dbg] CPU oracle forward FAILED: %s\n", sp_last_error());
            }
            free(ol);
        }
    }

    if (count > 0) {
        double ppl = exp(nll / (double)count);
        if (oracle > 0.0) {
            double rel = (ppl - oracle) / oracle;
            fprintf(stderr, "    [g4-ppl] SP PPL=%.4f  llama.cpp PPL=%.4f  rel-diff=%+.3f%% "
                            "(gate %.1f%%, n_scored=%ld, n_ctx=%d, chunks=%d)\n",
                    ppl, oracle, 100.0 * rel, 100.0 * gate, count, n_ctx, n_chunks);
            SP_CHECK((rel < 0 ? -rel : rel) < gate,
                     "M_GEMMA4_CUDA_PPL: the Q4-squeezed artifact holds within the smoke bound vs llama.cpp");
        } else {
            fprintf(stderr, "    [g4-ppl] SP PPL=%.4f (report-only; set SP_PPL_ORACLE to gate) "
                            "n_scored=%ld n_ctx=%d chunks=%d\n", ppl, count, n_ctx, n_chunks);
        }
    }

    free(chunk); free(toks);
    sp_cuda_model_release(m);
    qwen3_free(m);
    sp_model_unload(handle);
}

int main(void) {
    SP_RUN(M_GEMMA4_CUDA_PPL);
    return SP_DONE();
}
