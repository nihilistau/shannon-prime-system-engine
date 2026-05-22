/* test_ntt_attn_vulkan.c — E_VK_5: NTT-attention on Vulkan.
 *
 * Part A (substitution lock, CPU): the Vulkan NTT-attn shader computes the score
 * <q,k> as an exact int64 integer dot of the quantized head vectors. The CPU
 * E_CPU_5 path uses the poly-ring sp_pr_inner (coeff 0 of the negacyclic product
 * == sum_i q_i k_i). Both are the SAME exact integer; this test proves
 * int64_dot(q,k) == sp_pr_inner(q,k) bit-for-bit at head_dim in {128,256,512},
 * which is why the GPU may skip the CRT-NTT (int64 holds the result).
 *
 * Part B (gate, Vulkan): qwen3_forward_vulkan with SP_ENGINE_NTT_ATTN=1 vs the f32
 * path — mean KL <= 1e-7 (the T_PR_2 / E_CPU_5 tolerance). KL must also be > 0,
 * which proves the NTT branch actually fired (not a silent f32 fallback). */
#define _CRT_SECURE_NO_WARNINGS
#include "sp/sp_test.h"
#include "sp/poly_ring.h"
#include "sp_engine/gguf.h"
#include "sp_engine/model.h"
#include "sp_engine/vulkan_backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef SP_QWEN3_GGUF
#define SP_QWEN3_GGUF "Qwen3-0.6B-f16.gguf"
#endif
#ifndef SP_QWEN3_REF
#define SP_QWEN3_REF "qwen3_ref.bin"
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
    double lzp = log(zp) + pmax, lzq = log(zq) + qmax, kl = 0.0;
    for (uint32_t j = 0; j < n; j++) {
        double logP = (double)p[j] - lzp, Pj = exp(logP);
        if (Pj > 0.0) kl += Pj * (logP - ((double)q[j] - lzq));
    }
    return kl;
}

static uint32_t rng_state = 0x2545F491u;
static int32_t rand_coeff(int range) {
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 17; rng_state ^= rng_state << 5;
    return (int32_t)(rng_state % (uint32_t)(2 * range + 1)) - range;
}

static void E_VK_5(void) {
    /* ── Part A: int64 dot == sp_pr_inner ── */
    const uint32_t Ns[3] = { 128, 256, 512 };
    long ok = 0, total = 0;
    for (int ni = 0; ni < 3; ni++) {
        uint32_t N = Ns[ni];
        sp_pr_ctx *pr = sp_pr_init(N);
        SP_CHECK(pr != NULL, "sp_pr_init(N)");
        if (!pr) continue;
        int32_t *q = (int32_t *)malloc(N * sizeof(int32_t));
        int32_t *k = (int32_t *)malloc(N * sizeof(int32_t));
        for (int trial = 0; trial < 64; trial++) {
            for (uint32_t i = 0; i < N; i++) { q[i] = rand_coeff(1 << 21); k[i] = rand_coeff(1 << 21); }
            long long d_int = 0;
            for (uint32_t i = 0; i < N; i++) d_int += (long long)q[i] * (long long)k[i];
            int64_t d_pr = sp_pr_inner(pr, q, k);
            total++;
            if (d_int == (long long)d_pr) ok++;
        }
        free(q); free(k); sp_pr_free(pr);
    }
    fprintf(stderr, "    Part A: int64_dot == sp_pr_inner on %ld/%ld random vectors (N in {128,256,512})\n", ok, total);
    SP_CHECK_EQ_I64(ok, total, "int64 dot equals poly-ring sp_pr_inner exactly");

    /* ── Part B: Vulkan NTT-attn vs Vulkan f32 forward, mean KL <= 1e-7, KL > 0. ── */
    SP_CHECK(sp_vulkan_device_count() >= 1, "Vulkan device visible");
    if (sp_vulkan_device_count() < 1) return;

    FILE *f = fopen(SP_QWEN3_REF, "rb");
    SP_CHECK(f != NULL, "open qwen3_ref.bin for tokens");
    if (!f) return;
    uint32_t magic = 0, nt = 0, nv = 0;
    if (!(fread(&magic,4,1,f)==1 && fread(&nt,4,1,f)==1 && fread(&nv,4,1,f)==1) || nt == 0) { fclose(f); SP_CHECK(0, "ref header"); return; }
    int32_t *toks = (int32_t *)malloc((size_t)nt * sizeof(int32_t));
    int ok2 = toks && fread(toks, sizeof(int32_t), nt, f) == nt;
    fclose(f);
    SP_CHECK(ok2, "read ref tokens");
    if (!ok2) { free(toks); return; }

    ENV_CLR("SP_ARENA"); ENV_CLR("SP_ENGINE_NTT_ATTN");
    qwen3_model *m = qwen3_load(SP_QWEN3_GGUF);
    SP_CHECK(m && m->cfg.arch == SP_ARCH_QWEN3, "qwen3 load");
    if (!m) { free(toks); return; }
    const int V = (int)m->cfg.n_vocab;
    float *base = (float *)malloc((size_t)nt * V * sizeof(float));
    float *ntt  = (float *)malloc((size_t)nt * V * sizeof(float));
    SP_CHECK(base && ntt, "alloc logits");
    if (!base || !ntt) { free(base); free(ntt); free(toks); qwen3_free(m); return; }

    int rb = qwen3_forward_vulkan(m, toks, (int)nt, base);    /* f32 attention */
    ENV_SET("SP_ENGINE_NTT_ATTN", "1");
    int rn = qwen3_forward_vulkan(m, toks, (int)nt, ntt);     /* NTT (exact-integer) attention */
    ENV_CLR("SP_ENGINE_NTT_ATTN");
    SP_CHECK(rb == 0 && rn == 0, "vulkan forward f32 + ntt");
    if (rn) fprintf(stderr, "    sp_last_error: %s\n", sp_last_error());

    if (rb == 0 && rn == 0) {
        double kl_sum = 0, kl_max = 0; long argmax_ok = 0;
        for (uint32_t t = 0; t < nt; t++) {
            const float *a = base + (size_t)t * V, *b = ntt + (size_t)t * V;
            int ax = 0, bx = 0;
            for (int j = 0; j < V; j++) { if (a[j] > a[ax]) ax = j; if (b[j] > b[bx]) bx = j; }
            if (ax == bx) argmax_ok++;
            double kl = kl_div(a, b, (uint32_t)V); kl_sum += kl; if (kl > kl_max) kl_max = kl;
        }
        double kl_mean = kl_sum / nt;
        fprintf(stderr, "    Part B: argmax=%ld/%u KL(f32||ntt) mean=%.3e max=%.3e (gate 1e-7)\n",
                argmax_ok, nt, kl_mean, kl_max);
        SP_CHECK_EQ_I64(argmax_ok, nt, "NTT-attn argmax matches f32 at every position");
        SP_CHECK(kl_mean < 1.0e-7, "mean KL(f32||ntt) <= 1e-7 (T_PR_2)");
        SP_CHECK(kl_max > 0.0, "KL > 0 => NTT-attn branch actually fired (no silent f32 fallback)");
    }
    free(base); free(ntt); free(toks); qwen3_free(m);
}

int main(void) { SP_RUN(E_VK_5); return SP_DONE(); }
