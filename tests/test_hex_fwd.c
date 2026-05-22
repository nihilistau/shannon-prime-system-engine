/* test_hex_fwd.c — M_GEMMA3_HEXAGON: the cDSP layers-forward matches the CPU
 * forward on identical tokens, Q8 arena (the HX.3a gate, on the phone). Mirrors
 * M_GEMMA3_CUDA: run gemma3_forward (CPU) and gemma3_forward_hexagon (layers on
 * the cDSP, embed+head host) on the same tokens + same Q8 arena, compare argmax +
 * mean KL. Both are scalar f32 over the same Q8 codes, so they should agree
 * tightly (the cDSP matmul was already proven bit-exact). Fixed token IDs — no
 * tokenizer/corpus needed. Android exe; pushed + run via adb. */
#define _CRT_SECURE_NO_WARNINGS
#include "sp_engine/model.h"
#include "sp_engine/hexagon_backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef SP_GEMMA3_GGUF
#define SP_GEMMA3_GGUF "gemma-3-1b-it-f16.gguf"
#endif

static double kl_div(const float *p, const float *q, int n) {
    float pmax = p[0], qmax = q[0];
    for (int j = 1; j < n; j++) { if (p[j] > pmax) pmax = p[j]; if (q[j] > qmax) qmax = q[j]; }
    double zp = 0.0, zq = 0.0;
    for (int j = 0; j < n; j++) { zp += exp((double)p[j] - pmax); zq += exp((double)q[j] - qmax); }
    double lzp = log(zp) + pmax, lzq = log(zq) + qmax, kl = 0.0;
    for (int j = 0; j < n; j++) {
        double logP = (double)p[j] - lzp, Pj = exp(logP);
        if (Pj > 0.0) kl += Pj * (logP - ((double)q[j] - lzq));
    }
    return kl;
}

int main(void) {
    putenv((char *)"SP_ARENA=q8");          /* Q8 arena (hexagon path needs it) */
    putenv((char *)"SP_BACKEND=cpu");       /* CPU ref pass first */

    qwen3_model *m = qwen3_load(SP_GEMMA3_GGUF);
    if (!m || m->cfg.arch != SP_ARCH_GEMMA3 || !m->arena) {
        printf("HEX_FWD: load/arena FAIL (need %s + Q8 arena)\n", SP_GEMMA3_GGUF);
        return 1;
    }
    const int V = (int)m->cfg.n_vocab;
    int32_t toks[] = { 2, 1234, 5678, 9012, 100, 200 };
    int nt = (int)(sizeof(toks) / sizeof(toks[0]));
    printf("HEX_FWD: n_tok=%d vocab=%d\n", nt, V);

    float *cpu = (float *)malloc((size_t)nt * V * sizeof(float));
    float *hex = (float *)malloc((size_t)nt * V * sizeof(float));
    if (!cpu || !hex) { printf("HEX_FWD: alloc FAIL\n"); return 1; }

    int rc_cpu = gemma3_forward(m, toks, nt, cpu);          /* CPU Q8 reference */
    int rc_hex = gemma3_forward_hexagon(m, toks, nt, hex);  /* layers on cDSP */
    printf("HEX_FWD: gemma3_forward rc=%d  gemma3_forward_hexagon rc=%d\n", rc_cpu, rc_hex);
    if (rc_cpu || rc_hex) {
        printf("HEX_FWD: %s\n", sp_last_error());
        printf("HEX_FWD FAIL (forward error)\n");
        return 1;
    }

    double worst_rel = 0, kl_sum = 0, kl_max = 0;
    long argmax_ok = 0;
    for (int t = 0; t < nt; t++) {
        const float *a = cpu + (size_t)t * V, *b = hex + (size_t)t * V;
        int ax = 0, bx = 0;
        for (int j = 0; j < V; j++) {
            float ad = fabsf(a[j] - b[j]);
            float sc = fabsf(a[j]) > fabsf(b[j]) ? fabsf(a[j]) : fabsf(b[j]);
            if (sc > 1.0f && ad / sc > worst_rel) worst_rel = ad / sc;
            if (a[j] > a[ax]) ax = j;
            if (b[j] > b[bx]) bx = j;
        }
        if (ax == bx) argmax_ok++;
        double kl = kl_div(a, b, V); kl_sum += kl; if (kl > kl_max) kl_max = kl;
    }
    printf("HEX_FWD: argmax=%ld/%d worst_rel=%.3e KL(cpu||hex) mean=%.3e max=%.3e\n",
           argmax_ok, nt, worst_rel, kl_sum / nt, kl_max);

    int ok = (argmax_ok == nt) && (kl_sum / nt < 1e-5) && (worst_rel < 1e-2);
    printf(ok ? "HEX_FWD OK (cDSP Q8 layers match CPU Q8)\n" : "HEX_FWD FAIL\n");
    free(cpu); free(hex); qwen3_free(m);
    return ok ? 0 : 1;
}
