/* test_gemma3_cuda.c — M_GEMMA3_CUDA: the CUDA Gemma3 f32 forward matches the
 * CPU f32 forward on identical tokens (§8.3 cross-backend gate). Distributional,
 * same rationale as M_GEMMA3_CPU/§8.6.1: per-head QK-RMSNorm amplifies tiny
 * reduction-order (cuBLAS-vs-scalar) differences, so we gate on argmax + mean KL
 * and report the worst per-logit rel-diff rather than hard-bounding every logit. */
#define _CRT_SECURE_NO_WARNINGS
#include "sp/sp_test.h"
#include "sp_engine/gguf.h"
#include "sp_engine/model.h"
#include "sp_engine/tokenizer.h"
#include "sp_engine/cuda_backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef SP_GEMMA3_GGUF
#define SP_GEMMA3_GGUF "gemma-3-1b-it-f16.gguf"
#endif

/* KL(P||Q) in nats, P=softmax(p), Q=softmax(q). */
static double kl_div(const float *p, const float *q, uint32_t n) {
    float pmax = p[0], qmax = q[0];
    for (uint32_t j = 1; j < n; j++) { if (p[j] > pmax) pmax = p[j]; if (q[j] > qmax) qmax = q[j]; }
    double zp = 0.0, zq = 0.0;
    for (uint32_t j = 0; j < n; j++) { zp += exp((double)p[j] - pmax); zq += exp((double)q[j] - qmax); }
    double logzp = log(zp) + pmax, logzq = log(zq) + qmax, kl = 0.0;
    for (uint32_t j = 0; j < n; j++) {
        double logP = (double)p[j] - logzp, Pj = exp(logP);
        if (Pj > 0.0) kl += Pj * (logP - ((double)q[j] - logzq));
    }
    return kl;
}

static void M_GEMMA3_CUDA(void) {
    SP_CHECK(sp_cuda_device_count() >= 1, "CUDA device visible");
    if (sp_cuda_device_count() < 1) { fprintf(stderr, "    %s\n", sp_last_error()); return; }

    gguf_ctx *g = gguf_open(SP_GEMMA3_GGUF);
    SP_CHECK(g != NULL, "open gemma3 GGUF");
    if (!g) return;
    qwen3_model *m = qwen3_load(SP_GEMMA3_GGUF);
    sp_tokenizer *tok = sp_tokenizer_load(g);
    SP_CHECK(m && tok, "load model + tokenizer");
    if (!m || !tok) { sp_tokenizer_free(tok); qwen3_free(m); gguf_close(g); return; }
    SP_CHECK(m->cfg.arch == SP_ARCH_GEMMA3, "arch gemma3");

    const char *prompt = "The capital of France is Paris, and the Eiffel Tower stands there.";
    int32_t toks[128];
    long nt = sp_tokenizer_encode(tok, prompt, strlen(prompt), /*parse_special=*/0, toks, 128);
    SP_CHECK(nt > 1 && nt <= 128, "tokenize prompt");
    if (nt < 2) { sp_tokenizer_free(tok); qwen3_free(m); gguf_close(g); return; }
    const int V = (int)m->cfg.n_vocab;
    fprintf(stderr, "    n_tok=%ld vocab=%d\n", nt, V);

    float *cpu = (float *)malloc((size_t)nt * V * sizeof(float));
    float *cu  = (float *)malloc((size_t)nt * V * sizeof(float));
    SP_CHECK(cpu && cu, "alloc logits");
    if (!cpu || !cu) { free(cpu); free(cu); sp_tokenizer_free(tok); qwen3_free(m); gguf_close(g); return; }

    int rc_cpu = gemma3_forward(m, toks, (int)nt, cpu);
    SP_CHECK(rc_cpu == 0, "cpu gemma3_forward");
    int rc_cu = gemma3_forward_cuda(m, toks, (int)nt, cu);
    SP_CHECK(rc_cu == 0, "cuda gemma3_forward_cuda");
    if (rc_cu != 0) fprintf(stderr, "    sp_last_error: %s\n", sp_last_error());

    if (rc_cpu == 0 && rc_cu == 0) {
        double worst_abs = 0, worst_rel = 0, kl_sum = 0, kl_max = 0;
        long argmax_ok = 0;
        for (long t = 0; t < nt; t++) {
            const float *a = cpu + (size_t)t * V, *b = cu + (size_t)t * V;
            int am_c = 0, am_g = 0;
            for (int j = 0; j < V; j++) {
                float ad = fabsf(a[j] - b[j]);
                float sc = fabsf(a[j]) > fabsf(b[j]) ? fabsf(a[j]) : fabsf(b[j]);
                if (ad > worst_abs) worst_abs = ad;
                if (sc > 1.0f && ad / sc > worst_rel) worst_rel = ad / sc;
                if (a[j] > a[am_c]) am_c = j;
                if (b[j] > b[am_g]) am_g = j;
            }
            if (am_c == am_g) argmax_ok++;
            double kl = kl_div(a, b, (uint32_t)V);   /* KL(cpu || cuda) */
            kl_sum += kl; if (kl > kl_max) kl_max = kl;
        }
        double kl_mean = kl_sum / (double)nt;
        fprintf(stderr, "    %ld pos x %d vocab | worst_abs=%.3e worst_rel=%.3e "
                "| argmax=%ld/%ld | KL(cpu||cuda) mean=%.3e max=%.3e\n",
                nt, V, worst_abs, worst_rel, argmax_ok, nt, kl_mean, kl_max);
        SP_CHECK_EQ_I64(argmax_ok, nt, "CUDA argmax matches CPU at every position");
        SP_CHECK(kl_mean < 1.0e-5, "mean KL(cpu||cuda) below 1e-5 nats");
        SP_CHECK(worst_rel < 1.0e-2, "worst per-logit rel-diff below 1e-2 (gross-bug guard)");
    }

    free(cpu); free(cu);
    sp_tokenizer_free(tok); qwen3_free(m); gguf_close(g);
}

int main(void) { SP_RUN(M_GEMMA3_CUDA); return SP_DONE(); }
