/* test_moe_parity_cuda.c — G_N5A_MOE_PARITY
 *
 * Byte-exact (pure-f32) parity gate for the DiffusionGemma (arch_id 9) CUDA MoE FFN
 * (gemma4_moe_ffn_cuda in src/backends/cuda/cuda_forward.cu).
 *
 * Synthetic f32 fixture (deterministic xorshift weights) — no .sp-model, no OK_Q4B
 * quant noise, so the gate isolates the MoE algorithm: router prep -> f32 logit GEMV
 * -> full softmax -> top-NU by prob -> renorm -> per-expert GeGLU (GELU-tanh) ->
 * weighted accumulate. The CPU reference here IS the oracle (mirrors qwen36.c moe_ffn
 * arithmetic + the DiffusionGemma router-prep + GELU-tanh + first-half=gate split).
 *
 * GATE: selected-expert indices IDENTICAL CPU<->CUDA  AND  rel-err ||d||/||cpu|| < 1e-3
 * (pure-f32 path should be << 1e-5).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/* the additive CUDA entry under test */
extern int gemma4_moe_ffn_cuda(int E, int NE, int NU, int FF, float eps,
                               const float *gate_inp, const float *gate_inp_scale,
                               const float *gate_up_exps, const float *down_exps,
                               const float *hidden, float *out, int *sel_out);

/* deterministic uniform-ish weights in [-0.5,0.5) from a 32-bit xorshift stream. */
static uint32_t s_rng = 0;
static void seed_rng(uint32_t s) { s_rng = s ? s : 0x9e3779b9u; }
static float frand(void) {
    s_rng ^= s_rng << 13; s_rng ^= s_rng >> 17; s_rng ^= s_rng << 5;
    return ((float)(s_rng & 0xffffffu) / (float)0x1000000u) - 0.5f;
}

static float gelu_tanh(float x) {
    const float k = 0.7978845608028654f;
    float th = tanhf(k * (x + 0.044715f * x * x * x));
    return 0.5f * x * (1.0f + th);
}

/* CPU oracle: identical algorithm to gemma4_moe_ffn_cuda. */
static void moe_cpu(int E, int NE, int NU, int FF, float eps,
                    const float *gate_inp, const float *scale,
                    const float *gate_up_exps, const float *down_exps,
                    const float *hidden, float *out, int *sel) {
    float *x   = (float *)malloc((size_t)E * sizeof(float));   /* rms_norm(hidden) */
    float *tmp = (float *)malloc((size_t)E * sizeof(float));   /* router input */
    float *lg  = (float *)malloc((size_t)NE * sizeof(float));
    float *gu  = (float *)malloc((size_t)FF * 2 * sizeof(float));
    float *hh  = (float *)malloc((size_t)FF * sizeof(float));
    float *de  = (float *)malloc((size_t)E * sizeof(float));
    int   *idx = (int *)malloc((size_t)NU * sizeof(int));
    float *wt  = (float *)malloc((size_t)NU * sizeof(float));
    char  *used = (char *)malloc((size_t)NE);

    /* x = rms_norm(hidden) */
    double ss = 0.0; for (int i = 0; i < E; i++) ss += (double)hidden[i] * hidden[i];
    float inv = 1.0f / sqrtf((float)(ss / (double)E) + eps);
    for (int i = 0; i < E; i++) x[i] = hidden[i] * inv;
    /* router input prep: tmp = x * 1/sqrt(E) * scale[i] */
    float rscale = 1.0f / sqrtf((float)E);
    for (int i = 0; i < E; i++) tmp[i] = x[i] * rscale * scale[i];
    /* router GEMV */
    for (int o = 0; o < NE; o++) {
        const float *wo = gate_inp + (size_t)o * E;
        float a = 0.0f; for (int i = 0; i < E; i++) a += tmp[i] * wo[i];
        lg[o] = a;
    }
    /* softmax over all NE */
    float mx = lg[0]; for (int i = 1; i < NE; i++) if (lg[i] > mx) mx = lg[i];
    double se = 0.0;
    for (int i = 0; i < NE; i++) { lg[i] = expf(lg[i] - mx); se += lg[i]; }
    for (int i = 0; i < NE; i++) lg[i] = (float)(lg[i] / se);
    /* top-NU by prob (strict-greater, lowest index on tie) + renorm */
    memset(used, 0, (size_t)NE);
    float wsum = 0.0f;
    for (int k = 0; k < NU; k++) {
        int best = -1; float bv = -1.0f;
        for (int i = 0; i < NE; i++) if (!used[i] && lg[i] > bv) { bv = lg[i]; best = i; }
        used[best] = 1; idx[k] = best; wt[k] = bv; wsum += bv;
    }
    for (int k = 0; k < NU; k++) wt[k] = (wt[k] / wsum) * 1.0f;
    if (sel) for (int k = 0; k < NU; k++) sel[k] = idx[k];
    /* expert dispatch + weighted accumulate */
    memset(out, 0, (size_t)E * sizeof(float));
    for (int k = 0; k < NU; k++) {
        int e = idx[k];
        const float *gu_e = gate_up_exps + (size_t)e * (size_t)(FF * 2) * E;
        const float *dn_e = down_exps    + (size_t)e * (size_t)E * FF;
        /* gate_up = gu_e @ x -> [FF*2] */
        for (int o = 0; o < FF * 2; o++) {
            const float *wr = gu_e + (size_t)o * E;
            float a = 0.0f; for (int i = 0; i < E; i++) a += wr[i] * x[i];
            gu[o] = a;
        }
        /* h = gelu(gate) * up, gate = gu[0:FF], up = gu[FF:2FF] */
        for (int i = 0; i < FF; i++) hh[i] = gelu_tanh(gu[i]) * gu[FF + i];
        /* de = dn_e @ h -> [E] */
        for (int o = 0; o < E; o++) {
            const float *wr = dn_e + (size_t)o * FF;
            float a = 0.0f; for (int i = 0; i < FF; i++) a += wr[i] * hh[i];
            de[o] = a;
        }
        for (int i = 0; i < E; i++) out[i] += wt[k] * de[i];
    }
    free(x); free(tmp); free(lg); free(gu); free(hh); free(de);
    free(idx); free(wt); free(used);
}

int main(void) {
    /* small synthetic dims for speed (algorithm is dim-agnostic). */
    const int E = 64, NE = 8, NU = 4, FF = 16;
    const float eps = 1e-6f;

    seed_rng(0xC0FFEEu);
    float *gate_inp   = (float *)malloc((size_t)NE * E * sizeof(float));
    float *scale      = (float *)malloc((size_t)E * sizeof(float));
    float *gate_up    = (float *)malloc((size_t)NE * FF * 2 * E * sizeof(float));
    float *down       = (float *)malloc((size_t)NE * E * FF * sizeof(float));
    float *hidden     = (float *)malloc((size_t)E * sizeof(float));
    for (size_t i = 0; i < (size_t)NE * E; i++)          gate_inp[i] = frand();
    for (size_t i = 0; i < (size_t)E; i++)               scale[i]    = 0.8f + 0.4f * (frand() + 0.5f); /* ~[0.8,1.2] */
    for (size_t i = 0; i < (size_t)NE * FF * 2 * E; i++) gate_up[i]  = frand();
    for (size_t i = 0; i < (size_t)NE * E * FF; i++)     down[i]     = frand();
    for (int i = 0; i < E; i++)                          hidden[i]   = 2.0f * frand();

    float *out_cpu = (float *)malloc((size_t)E * sizeof(float));
    float *out_gpu = (float *)malloc((size_t)E * sizeof(float));
    int sel_cpu[16], sel_gpu[16];

    moe_cpu(E, NE, NU, FF, eps, gate_inp, scale, gate_up, down, hidden, out_cpu, sel_cpu);

    int rc = gemma4_moe_ffn_cuda(E, NE, NU, FF, eps, gate_inp, scale, gate_up, down,
                                 hidden, out_gpu, sel_gpu);
    if (rc != 0) { printf("G_N5A_MOE_PARITY: FAIL (cuda entry returned %d)\n", rc); return 1; }

    /* expert-selection match */
    int sel_match = 1;
    for (int k = 0; k < NU; k++) if (sel_cpu[k] != sel_gpu[k]) sel_match = 0;
    printf("selected experts CPU:");
    for (int k = 0; k < NU; k++) printf(" %d", sel_cpu[k]);
    printf("  GPU:");
    for (int k = 0; k < NU; k++) printf(" %d", sel_gpu[k]);
    printf("   match=%s\n", sel_match ? "YES" : "NO");

    /* L2 / rel-err */
    double l2d = 0.0, l2c = 0.0;
    for (int i = 0; i < E; i++) {
        double d = (double)out_cpu[i] - out_gpu[i];
        l2d += d * d; l2c += (double)out_cpu[i] * out_cpu[i];
    }
    double l2 = sqrt(l2d);
    double rel = l2 / (sqrt(l2c) + 1e-30);
    printf("L2(diff)=%.6e  ||cpu||=%.6e  rel-err=%.6e\n", l2, sqrt(l2c), rel);

    int pass = sel_match && (rel < 1e-3);
    printf("G_N5A_MOE_PARITY: %s  (dims E=%d NE=%d NU=%d FF=%d)\n",
           pass ? "PASS" : "FAIL", E, NE, NU, FF);

    free(gate_inp); free(scale); free(gate_up); free(down); free(hidden);
    free(out_cpu); free(out_gpu);
    return pass ? 0 : 1;
}
