/* test_gemma3_vulkan.c — M_GEMMA3_VULKAN: the Vulkan Gemma3 forward matches the
 * CPU forward on identical tokens (§8.3 cross-backend gate). Three scenarios:
 *   (f32) plain GGUF f32 weights — VK.1.
 *   (q8 ) per-row Frobenius Q8 arena (SP_ARENA=q8) — VK.2.
 *   (q4 ) per-row Frobenius Q4 mixed-precision arena (SP_ARENA=q4) — VK.3: rows
 *         whose Q4 round-trip error exceeds SP_Q4_PROMOTE are stored Q8, so this
 *         exercises BOTH branches of the device decode shader dequant_arena.
 * For q8/q4 the arena codes are built CPU-side at load, so Vulkan device-decode vs
 * CPU matmul_arena differ only by float reassociation + GEMM reduction order.
 * Distributional gate (argmax + mean KL), worst per-logit rel-diff reported. */
#define _CRT_SECURE_NO_WARNINGS
#include "sp/sp_test.h"
#include "sp_engine/gguf.h"
#include "sp_engine/model.h"
#include "sp_engine/arena.h"
#include "sp_engine/tokenizer.h"
#include "sp_engine/vulkan_backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef SP_GEMMA3_GGUF
#define SP_GEMMA3_GGUF "gemma-3-1b-it-f16.gguf"
#endif

#ifdef _WIN32
#define ENV_SET(k, v) _putenv_s((k), (v))
#define ENV_CLR(k)    _putenv_s((k), "")
#else
static void ENV_SET(const char *k, const char *v) { setenv(k, v, 1); }
static void ENV_CLR(const char *k) { unsetenv(k); }
#endif

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

static void run_compare(const char *tag, const char *arena_mode) {
    fprintf(stderr, "  [%s]\n", tag);
    if (arena_mode) ENV_SET("SP_ARENA", arena_mode); else ENV_CLR("SP_ARENA");

    gguf_ctx *g = gguf_open(SP_GEMMA3_GGUF);
    SP_CHECK(g != NULL, "open gemma3 GGUF");
    if (!g) { ENV_CLR("SP_ARENA"); return; }
    qwen3_model *m = qwen3_load(SP_GEMMA3_GGUF);
    sp_tokenizer *tok = sp_tokenizer_load(g);
    SP_CHECK(m && tok, "load model + tokenizer");
    if (!m || !tok) { sp_tokenizer_free(tok); qwen3_free(m); gguf_close(g); ENV_CLR("SP_ARENA"); return; }
    SP_CHECK(arena_mode ? (m->arena != NULL) : (m->arena == NULL), "arena state matches scenario");
    if (arena_mode) {
        int want = (arena_mode[1] == '8') ? 8 : 4;
        SP_CHECK_EQ_I64(sp_arena_precision(m->arena), want, "arena precision matches scenario");
        if (want == 4)
            fprintf(stderr, "    q4 mixed-precision: %ld/%ld rows promoted to Q8\n",
                    sp_arena_promoted(m->arena), sp_arena_total_rows(m->arena));
    }

    const char *prompt = "The capital of France is Paris, and the Eiffel Tower stands there.";
    int32_t toks[128];
    long nt = sp_tokenizer_encode(tok, prompt, strlen(prompt), 0, toks, 128);
    SP_CHECK(nt > 1 && nt <= 128, "tokenize prompt");
    const int V = (int)m->cfg.n_vocab;

    float *cpu = (float *)malloc((size_t)nt * V * sizeof(float));
    float *vk  = (float *)malloc((size_t)nt * V * sizeof(float));
    SP_CHECK(cpu && vk, "alloc logits");
    if (nt < 2 || !cpu || !vk) { free(cpu); free(vk); sp_tokenizer_free(tok); qwen3_free(m); gguf_close(g); ENV_CLR("SP_ARENA"); return; }

    int rc_cpu = gemma3_forward(m, toks, (int)nt, cpu);
    int rc_vk  = gemma3_forward_vulkan(m, toks, (int)nt, vk);
    SP_CHECK(rc_cpu == 0, "cpu gemma3_forward");
    SP_CHECK(rc_vk == 0, "vulkan gemma3_forward_vulkan");
    if (rc_vk != 0) fprintf(stderr, "    sp_last_error: %s\n", sp_last_error());

    if (rc_cpu == 0 && rc_vk == 0) {
        double worst_abs = 0, worst_rel = 0, kl_sum = 0, kl_max = 0;
        long argmax_ok = 0;
        for (long t = 0; t < nt; t++) {
            const float *a = cpu + (size_t)t * V, *b = vk + (size_t)t * V;
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
            double kl = kl_div(a, b, (uint32_t)V);
            kl_sum += kl; if (kl > kl_max) kl_max = kl;
        }
        double kl_mean = kl_sum / (double)nt;
        fprintf(stderr, "    %ld pos x %d vocab | worst_abs=%.3e worst_rel=%.3e "
                "| argmax=%ld/%ld | KL(cpu||vulkan) mean=%.3e max=%.3e\n",
                nt, V, worst_abs, worst_rel, argmax_ok, nt, kl_mean, kl_max);
        SP_CHECK_EQ_I64(argmax_ok, nt, "Vulkan argmax matches CPU at every position");
        SP_CHECK(kl_mean < 1.0e-5, "mean KL(cpu||vulkan) below 1e-5 nats");
        SP_CHECK(worst_rel < 1.0e-2, "worst per-logit rel-diff below 1e-2");
    }

    free(cpu); free(vk);
    sp_tokenizer_free(tok); qwen3_free(m); gguf_close(g);
    ENV_CLR("SP_ARENA");
}

static void M_GEMMA3_VULKAN(void) {
    SP_CHECK(sp_vulkan_device_count() >= 1, "Vulkan device visible");
    if (sp_vulkan_device_count() < 1) { fprintf(stderr, "    %s\n", sp_last_error()); return; }
    /* Q4-scoped on a VRAM-bounded desktop GPU (mirrors M_GEMMA3_CUDA): Gemma3-1B f32/Q8 over-capacity
     * a 12GB GPU; M_QWEN3_VULKAN covers f32/Q8/Q4 at 0.6B; Q4 mixed-precision is the production path
     * and its cross-backend identity (vk-vs-cpu KL ~1e-11) is the load-bearing relocation validation. */
    run_compare("Q4 mixed-precision Frobenius arena (VK.3)", "q4");
}

int main(void) { SP_RUN(M_GEMMA3_VULKAN); return SP_DONE(); }
