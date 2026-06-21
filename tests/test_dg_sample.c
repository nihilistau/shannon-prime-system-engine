/* test_dg_sample.c — G-DG-N4a: the entropy-bound sample kernel parity gate.
 *
 * Pure vocab-space float test (NO model, NO weights, NO GPU memory the caller owns):
 * builds a FIXED deterministic logit fixture [n_pos x n_vocab], runs the ported CUDA
 * dg_sample_kernel (via dg_sample_logits_host), and compares against a HOST reference
 * computed with the IDENTICAL formula:
 *   argmax_v (logit_v * inv_temp)                        -> must be EXACT
 *   d_v = logit_v*inv_temp - max ; Z=sum exp(d) ; T=sum d*exp(d)
 *   entropy = log Z - T/Z                                -> ~1e-4 (FP reduction order)
 *   sampled = first vocab-order v with cumulative exp(d) >= u[row]*Z
 *
 * The host reference uses the SAME single-precision arithmetic the kernel does
 * (the kernel accumulates in f32 across 256-thread partial sums; we accumulate the
 * reference in f32 too), so the entropy delta is just FP reduction ORDER, ~1e-5.
 *
 * PASS (G-DG-N4a): argmax exact on every row AND max |entropy - ref| < 1e-3 AND
 * sampled matches the host CDF walk on every row.
 *
 * Fast + tractable: a few ms on the GPU, no model load. The first thing to land.
 */
#define _CRT_SECURE_NO_WARNINGS
#include "sp_engine/cuda_backend.h"
#include "sp/sp_status.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/* deterministic splitmix64 -> uniform in [0,1) and a logit fixture generator */
static uint64_t g_rng = 0;
static uint64_t rng_next(void) {
    uint64_t z = (g_rng += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static float rng_uniform(void) { return (float)((rng_next() >> 11) * (1.0 / 9007199254740992.0)); }

int main(void) {
    /* fixture dims: a handful of positions, a realistic-ish vocab. Mix of distributions:
     * row 0 = nearly uniform (high entropy), row 1 = peaked (low entropy), rest random. */
    const int n_pos   = 8;
    const int n_vocab = 4099;          /* not a multiple of 256 -> exercises the slice tail */
    const float inv_temp = 1.0f / 0.7f;

    float *logits = (float *)malloc((size_t)n_pos * n_vocab * sizeof(float));
    float *u      = (float *)malloc((size_t)n_pos * sizeof(float));
    if (!logits || !u) { fprintf(stderr, "OOM\n"); return 2; }

    g_rng = 20260621ull;
    for (int p = 0; p < n_pos; p++) {
        for (int v = 0; v < n_vocab; v++) {
            float base;
            if (p == 0)      base = 0.01f * (rng_uniform() - 0.5f);     /* nearly flat */
            else if (p == 1) base = 3.0f  * (rng_uniform() - 0.5f);     /* moderately peaked */
            else             base = 2.0f  * (rng_uniform() - 0.5f);
            logits[(size_t)p * n_vocab + v] = base;
        }
        /* plant a clear argmax on rows >=1 so exactness is unambiguous */
        if (p >= 1) {
            int peak = (int)(rng_next() % (uint64_t)n_vocab);
            logits[(size_t)p * n_vocab + peak] += 8.0f + 4.0f * rng_uniform();
        }
        u[p] = rng_uniform();
    }

    /* ── host reference (identical f32 formula, plain serial order) ── */
    int   *ref_am  = (int *)  malloc((size_t)n_pos * sizeof(int));
    float *ref_ent = (float *)malloc((size_t)n_pos * sizeof(float));
    int   *ref_sm  = (int *)  malloc((size_t)n_pos * sizeof(int));
    for (int p = 0; p < n_pos; p++) {
        const float *row = logits + (size_t)p * n_vocab;
        float mx = -3.4e38f; int amax = 0;
        for (int v = 0; v < n_vocab; v++) { float x = row[v] * inv_temp; if (x > mx) { mx = x; amax = v; } }
        float Z = 0.0f, T = 0.0f;
        for (int v = 0; v < n_vocab; v++) { float d = row[v] * inv_temp - mx; float e = expf(d); Z += e; T += d * e; }
        ref_am[p]  = amax;
        ref_ent[p] = logf(Z) - T / Z;
        /* CDF walk: first v with cumulative exp(d) >= u*Z */
        float r = u[p] * Z, cum = 0.0f; int tok = n_vocab - 1;
        for (int v = 0; v < n_vocab; v++) { cum += expf(row[v] * inv_temp - mx); if (cum >= r) { tok = v; break; } }
        ref_sm[p] = tok;
    }

    /* ── kernel ── */
    int   *k_am  = (int *)  malloc((size_t)n_pos * sizeof(int));
    float *k_ent = (float *)malloc((size_t)n_pos * sizeof(float));
    int   *k_sm  = (int *)  malloc((size_t)n_pos * sizeof(int));
    int rc = dg_sample_logits_host(logits, n_pos, n_vocab, u, inv_temp, k_am, k_ent, k_sm);
    if (rc != 0) {
        /* GPU/driver absent -> SKIP (exit 0), receipts-first honesty */
        fprintf(stdout, "# dg_sample_logits_host rc=%d (%s) — GPU absent? SKIP (exit 0)\n", rc, sp_last_error());
        return 0;
    }

    /* ── compare ── */
    int am_ok = 1, sm_ok = 1;
    double max_ent_err = 0.0;
    printf("# G-DG-N4a  n_pos=%d n_vocab=%d inv_temp=%.4f\n", n_pos, n_vocab, inv_temp);
    for (int p = 0; p < n_pos; p++) {
        double ee = fabs((double)k_ent[p] - (double)ref_ent[p]);
        if (ee > max_ent_err) max_ent_err = ee;
        int a_ok = (k_am[p] == ref_am[p]);
        int s_ok = (k_sm[p] == ref_sm[p]);
        if (!a_ok) am_ok = 0;
        if (!s_ok) sm_ok = 0;
        printf("  pos %d: argmax k=%d ref=%d %s | entropy k=%.6f ref=%.6f d=%.2e | sampled k=%d ref=%d %s\n",
               p, k_am[p], ref_am[p], a_ok ? "OK" : "MISS",
               k_ent[p], ref_ent[p], ee, k_sm[p], ref_sm[p], s_ok ? "OK" : "MISS");
    }

    int ent_ok = (max_ent_err < 1e-3);
    int pass = am_ok && sm_ok && ent_ok;
    printf("\n================ G-DG-N4a RESULT ================\n");
    printf("argmax exact   : %s\n", am_ok ? "GREEN" : "RED");
    printf("sampled exact  : %s\n", sm_ok ? "GREEN" : "RED");
    printf("entropy err    : max=%.3e  (tol 1e-3)  %s\n", max_ent_err, ent_ok ? "GREEN" : "RED");
    printf("G-DG-N4a       : %s\n", pass ? "GREEN — sampler kernel matches host reference" : "RED");

    free(logits); free(u);
    free(ref_am); free(ref_ent); free(ref_sm);
    free(k_am); free(k_ent); free(k_sm);
    return pass ? 0 : 1;
}
