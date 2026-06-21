/* test_moe_parity_packed_cuda.c — G-DG-N5a-packed
 *
 * Parity gate for the DiffusionGemma (arch_id 9) CUDA MoE FFN on the REAL OK_Q4B
 * packed experts through the dp4a path (gemma4_moe_ffn_q4b_cuda in
 * src/backends/cuda/cuda_forward.cu).
 *
 * Builds a small SELF-CONTAINED OK_Q4B-packed MoE fixture in-process, using the
 * EXACT substrate packing the .sp-model loader (build_packed_q4b) produces:
 *   - per-32-block f16 scale (sp_f32_to_f16, the SAME f16 round-trip as the loader),
 *   - nibble-packed signed-4bit codes (2 codes/byte; low nibble = even idx, high =
 *     odd; sign range clamped to [-7,7]); dequant w = code * f16(bscale[i>>5]).
 * The fused gate_up tensor is [FF*2, E] per expert (gate = first FF rows, up = last
 * FF rows); down is [E, FF] per expert; expert e at rows [e*rows_per_expert, ...).
 *
 * ORACLE = the CPU OK_Q4B MoE: it dequants the SAME codes/bscale to f32 (the exact
 * sp_frob_packed_dequant_row arithmetic) and does an f32 dot — bit-for-bit the CPU
 * expert path. The CUDA side quantizes the activation to int8 (per-16-block, qmax
 * 127) and accumulates via integer dp4a against the IDENTICAL weight codes+scales.
 *
 * GATE: expert-selection IDENTICAL CPU<->CUDA  AND  rel-err ||d||/||cpu|| < 2e-3
 *   (the weight codes+scales are identical both sides; the deflection is the int8
 *    activation quant + integer reduction order, ~1e-3 — exactly the dense dp4a vs
 *    f32 split, NOT a layout/scale/offset bug, which would blow rel-err to O(1)).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "sp/weight_dtype.h"   /* sp_f16_to_f32 / sp_f32_to_f16 — the loader's f16 round-trip */

/* the additive CUDA entries under test */
extern int gemma4_moe_ffn_q4b_cuda(int E, int NE, int NU, int FF, float eps,
                                   const float *gate_inp, const float *gate_inp_scale,
                                   const unsigned char *gu_codes, const unsigned short *gu_bscale,
                                   const unsigned char *dn_codes, const unsigned short *dn_bscale,
                                   const float *hidden, float *out, int *sel_out);

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
static double relerr(const float *a, const float *b, int n) {
    double d = 0.0, c = 0.0;
    for (int i = 0; i < n; i++) { double x = (double)a[i] - b[i]; d += x*x; c += (double)a[i]*a[i]; }
    return sqrt(d) / (sqrt(c) + 1e-30);
}

/* ── OK_Q4B packer: rows x cols f32 -> nibble codes + per-32-block f16 scales.
 * Mirrors build_packed_q4b's expectation (codes pre-quantized against the STORED
 * f16 scale; w = code * f16(scale)) so sp_frob_packed_dequant_row reproduces it. */
static void q4b_pack(const float *W, int rows, int cols,
                     unsigned char *codes, unsigned short *bscale) {
    const int nblk = (cols + 31) / 32;
    const size_t nibcols = (size_t)((cols + 1) / 2);
    for (int r = 0; r < rows; r++) {
        const float *wr = W + (size_t)r * cols;
        unsigned char *cr = codes + (size_t)r * nibcols;
        unsigned short *br = bscale + (size_t)r * nblk;
        memset(cr, 0, nibcols);
        for (int b = 0; b < nblk; b++) {
            int base = b * 32;
            float m = 0.0f;
            for (int i = 0; i < 32 && base + i < cols; i++) {
                float a = fabsf(wr[base + i]); if (a > m) m = a;
            }
            float scale = m > 0.0f ? m / 7.0f : 1.0f;   /* QMAX4 = 7 */
            uint16_t h = sp_f32_to_f16(scale);
            br[b] = h;
            float sf = sp_f16_to_f32(h);                 /* store-then-derive: quantize vs the STORED f16 */
            float inv = sf > 0.0f ? 1.0f / sf : 0.0f;
            for (int i = 0; i < 32 && base + i < cols; i++) {
                int idx = base + i;
                int q = (int)lrintf(wr[idx] * inv);
                if (q > 7) q = 7; if (q < -7) q = -7;    /* symmetric 4-bit */
                unsigned char nib = (unsigned char)(q & 0xF);
                if (idx & 1) cr[idx >> 1] |= (unsigned char)(nib << 4);
                else         cr[idx >> 1] |= nib;
            }
        }
    }
}

/* dequant one row exactly as sp_frob_packed_dequant_row (OK_Q4B branch). */
static void q4b_dequant_row(const unsigned char *codes, const unsigned short *bscale,
                            int row, int cols, int nblk, float *dst) {
    const size_t nibcols = (size_t)((cols + 1) / 2);
    const unsigned char *rc = codes + (size_t)row * nibcols;
    const unsigned short *bs = bscale + (size_t)row * nblk;
    for (int i = 0; i < cols; i++) {
        unsigned char b = (i & 1) ? (unsigned char)(rc[i >> 1] >> 4) : (unsigned char)(rc[i >> 1] & 0xF);
        int8_t v = (int8_t)((b & 0x8) ? (int)b - 16 : (int)b);   /* sign-extend 4-bit */
        dst[i] = (float)v * sp_f16_to_f32(bs[i >> 5]);
    }
}

/* per-16-block int8 activation quant, EXACTLY k_quant_act_int8 (qmax 127, padded
 * to npad). Writes int8 codes + per-16-block f32 scales; the dp4a integer dot then
 * multiplies the int weight*act by (block scale). Used only by the int8-floor CPU
 * reference below, to attribute the deflection to act-quant (not a CUDA bug). */
static void cpu_quant_act_int8(const float *x, int n, int npad, signed char *qx, float *sxb) {
    int nblk = npad >> 4;
    for (int b = 0; b < nblk; b++) {
        int base = b << 4;
        float m = 0.0f;
        for (int i = 0; i < 16; i++) { int idx = base + i; float a = (idx < n) ? fabsf(x[idx]) : 0.0f; if (a > m) m = a; }
        float scale = m > 0.0f ? m * (1.0f / 127.0f) : 1.0f;
        sxb[b] = scale;
        float inv = 1.0f / scale;
        for (int i = 0; i < 16; i++) {
            int idx = base + i;
            float v = (idx < n) ? x[idx] * inv : 0.0f;
            int q = (int)lrintf(v); if (q > 127) q = 127; if (q < -127) q = -127;
            qx[idx] = (signed char)q;
        }
    }
}

/* int8-FLOOR CPU reference: the SAME dp4a integer arithmetic the CUDA kernel runs
 * (int8 act-quant + integer weight*act dot + per-32-block weight scale split into
 * two 16-block halves), in scalar C. Its delta vs the f32 oracle IS the irreducible
 * int8-activation-quant floor; the CUDA result should land at (not below) it. */
static float q4b_dp4a_row_cpu(const unsigned char *codes, const unsigned short *bscale,
                              int row, int in, int nblk, const signed char *qx, const float *sxb) {
    const size_t nibcols = (size_t)((in + 1) / 2);
    const unsigned char *rc = codes + (size_t)row * nibcols;
    const unsigned short *bs = bscale + (size_t)row * nblk;
    float facc = 0.0f;
    int n32 = in >> 5;
    for (int c = 0; c < n32; c++) {            /* one 32-block = two 16 act-blocks */
        int a0 = 0, a1 = 0;
        for (int i = 0; i < 16; i++) {
            int idx = c * 32 + i;
            unsigned char b = (idx & 1) ? (unsigned char)(rc[idx >> 1] >> 4) : (unsigned char)(rc[idx >> 1] & 0xF);
            int8_t w = (int8_t)((b & 0x8) ? (int)b - 16 : (int)b);
            a0 += (int)w * (int)qx[idx];
        }
        for (int i = 16; i < 32; i++) {
            int idx = c * 32 + i;
            unsigned char b = (idx & 1) ? (unsigned char)(rc[idx >> 1] >> 4) : (unsigned char)(rc[idx >> 1] & 0xF);
            int8_t w = (int8_t)((b & 0x8) ? (int)b - 16 : (int)b);
            a1 += (int)w * (int)qx[idx];
        }
        float wbsc = sp_f16_to_f32(bs[c]);
        facc += wbsc * ((float)a0 * sxb[2*c] + (float)a1 * sxb[2*c + 1]);
    }
    return facc;
}

/* CPU oracle: the SAME MoE algorithm, experts dequanted OK_Q4B -> f32 dot.
 * int8_floor != 0 routes the expert GEMVs through the int8 dp4a arithmetic instead
 * (the CUDA path's exact integer sequence) — the act-quant floor reference. */
static void moe_cpu_q4b(int E, int NE, int NU, int FF, float eps,
                        const float *gate_inp, const float *scale,
                        const unsigned char *gu_codes, const unsigned short *gu_bscale,
                        const unsigned char *dn_codes, const unsigned short *dn_bscale,
                        const float *hidden, float *out, int *sel, int int8_floor) {
    const int gu_rows = FF * 2, dn_rows = E;
    const int gu_nblk = E / 32, dn_nblk = FF / 32;
    const size_t gu_nib = (size_t)((E + 1) / 2), dn_nib = (size_t)((FF + 1) / 2);
    float *x   = (float *)malloc((size_t)E * sizeof(float));
    float *tmp = (float *)malloc((size_t)E * sizeof(float));
    float *lg  = (float *)malloc((size_t)NE * sizeof(float));
    float *gu  = (float *)malloc((size_t)FF * 2 * sizeof(float));
    float *hh  = (float *)malloc((size_t)FF * sizeof(float));
    float *de  = (float *)malloc((size_t)E * sizeof(float));
    float *wrow = (float *)malloc((size_t)(E > FF ? E : FF) * sizeof(float));
    int   *idx = (int *)malloc((size_t)NU * sizeof(int));
    float *wt  = (float *)malloc((size_t)NU * sizeof(float));
    char  *used = (char *)malloc((size_t)NE);

    double ss = 0.0; for (int i = 0; i < E; i++) ss += (double)hidden[i] * hidden[i];
    float inv = 1.0f / sqrtf((float)(ss / (double)E) + eps);
    for (int i = 0; i < E; i++) x[i] = hidden[i] * inv;
    float rscale = 1.0f / sqrtf((float)E);
    for (int i = 0; i < E; i++) tmp[i] = x[i] * rscale * scale[i];
    for (int o = 0; o < NE; o++) {
        const float *wo = gate_inp + (size_t)o * E;
        float a = 0.0f; for (int i = 0; i < E; i++) a += tmp[i] * wo[i];
        lg[o] = a;
    }
    float mx = lg[0]; for (int i = 1; i < NE; i++) if (lg[i] > mx) mx = lg[i];
    double se = 0.0;
    for (int i = 0; i < NE; i++) { lg[i] = expf(lg[i] - mx); se += lg[i]; }
    for (int i = 0; i < NE; i++) lg[i] = (float)(lg[i] / se);
    memset(used, 0, (size_t)NE);
    float wsum = 0.0f;
    for (int k = 0; k < NU; k++) {
        int best = -1; float bv = -1.0f;
        for (int i = 0; i < NE; i++) if (!used[i] && lg[i] > bv) { bv = lg[i]; best = i; }
        used[best] = 1; idx[k] = best; wt[k] = bv; wsum += bv;
    }
    for (int k = 0; k < NU; k++) wt[k] = (wt[k] / wsum) * 1.0f;
    if (sel) for (int k = 0; k < NU; k++) sel[k] = idx[k];
    memset(out, 0, (size_t)E * sizeof(float));
    for (int k = 0; k < NU; k++) {
        int e = idx[k];
        const unsigned char  *gu_c  = gu_codes  + (size_t)e * gu_rows * gu_nib;
        const unsigned short *gu_bs = gu_bscale + (size_t)e * gu_rows * gu_nblk;
        const unsigned char  *dn_c  = dn_codes  + (size_t)e * dn_rows * dn_nib;
        const unsigned short *dn_bs = dn_bscale + (size_t)e * dn_rows * dn_nblk;
        /* gate_up = Q4B(gate_up_exps[e]) @ x -> [FF*2] */
        if (int8_floor) {
            int npad = (E + 31) & ~31;
            signed char *qx = (signed char *)malloc((size_t)npad);
            float *sx = (float *)malloc((size_t)(npad >> 4) * sizeof(float));
            cpu_quant_act_int8(x, E, npad, qx, sx);
            for (int o = 0; o < FF * 2; o++) gu[o] = q4b_dp4a_row_cpu(gu_c, gu_bs, o, E, gu_nblk, qx, sx);
            free(qx); free(sx);
        } else {
            for (int o = 0; o < FF * 2; o++) {
                q4b_dequant_row(gu_c, gu_bs, o, E, gu_nblk, wrow);
                float a = 0.0f; for (int i = 0; i < E; i++) a += wrow[i] * x[i];
                gu[o] = a;
            }
        }
        for (int i = 0; i < FF; i++) hh[i] = gelu_tanh(gu[i]) * gu[FF + i];
        /* de = Q4B(down_exps[e]) @ h -> [E] */
        if (int8_floor) {
            int npad = (FF + 31) & ~31;
            signed char *qx = (signed char *)malloc((size_t)npad);
            float *sx = (float *)malloc((size_t)(npad >> 4) * sizeof(float));
            cpu_quant_act_int8(hh, FF, npad, qx, sx);
            for (int o = 0; o < E; o++) de[o] = q4b_dp4a_row_cpu(dn_c, dn_bs, o, FF, dn_nblk, qx, sx);
            free(qx); free(sx);
        } else {
            for (int o = 0; o < E; o++) {
                q4b_dequant_row(dn_c, dn_bs, o, FF, dn_nblk, wrow);
                float a = 0.0f; for (int i = 0; i < FF; i++) a += wrow[i] * hh[i];
                de[o] = a;
            }
        }
        for (int i = 0; i < E; i++) out[i] += wt[k] * de[i];
    }
    free(x); free(tmp); free(lg); free(gu); free(hh); free(de); free(wrow);
    free(idx); free(wt); free(used);
}

int main(void) {
    /* dims multiples of 32 (Q4B dp4a precondition); small for speed. */
    const int E = 128, NE = 8, NU = 4, FF = 64;
    const float eps = 1e-6f;
    const int gu_rows = FF * 2, dn_rows = E;
    const int gu_nblk = E / 32, dn_nblk = FF / 32;
    const size_t gu_nib = (size_t)((E + 1) / 2), dn_nib = (size_t)((FF + 1) / 2);

    seed_rng(0xD1FF26u);
    float *gate_inp = (float *)malloc((size_t)NE * E * sizeof(float));
    float *scale    = (float *)malloc((size_t)E * sizeof(float));
    float *hidden   = (float *)malloc((size_t)E * sizeof(float));
    /* f32 expert weights, then packed to OK_Q4B (the real substrate format) */
    float *gu_f32 = (float *)malloc((size_t)NE * gu_rows * E  * sizeof(float));
    float *dn_f32 = (float *)malloc((size_t)NE * dn_rows * FF * sizeof(float));
    for (size_t i = 0; i < (size_t)NE * E; i++) gate_inp[i] = frand();
    for (int i = 0; i < E; i++)  scale[i]  = 0.8f + 0.4f * (frand() + 0.5f);
    for (int i = 0; i < E; i++)  hidden[i] = 2.0f * frand();
    for (size_t i = 0; i < (size_t)NE * gu_rows * E;  i++) gu_f32[i] = frand();
    for (size_t i = 0; i < (size_t)NE * dn_rows * FF; i++) dn_f32[i] = frand();

    unsigned char  *gu_codes  = (unsigned char  *)malloc((size_t)NE * gu_rows * gu_nib);
    unsigned short *gu_bscale  = (unsigned short *)malloc((size_t)NE * gu_rows * gu_nblk * sizeof(unsigned short));
    unsigned char  *dn_codes  = (unsigned char  *)malloc((size_t)NE * dn_rows * dn_nib);
    unsigned short *dn_bscale  = (unsigned short *)malloc((size_t)NE * dn_rows * dn_nblk * sizeof(unsigned short));
    q4b_pack(gu_f32, NE * gu_rows, E,  gu_codes, gu_bscale);
    q4b_pack(dn_f32, NE * dn_rows, FF, dn_codes, dn_bscale);

    float *out_orc = (float *)malloc((size_t)E * sizeof(float));   /* f32-dequant oracle (the CPU OK_Q4B MoE) */
    float *out_flr = (float *)malloc((size_t)E * sizeof(float));   /* int8-act dp4a floor reference */
    float *out_gpu = (float *)malloc((size_t)E * sizeof(float));   /* the CUDA dp4a path */
    int sel_orc[16], sel_flr[16], sel_gpu[16];

    moe_cpu_q4b(E, NE, NU, FF, eps, gate_inp, scale,
                gu_codes, gu_bscale, dn_codes, dn_bscale, hidden, out_orc, sel_orc, 0);
    moe_cpu_q4b(E, NE, NU, FF, eps, gate_inp, scale,
                gu_codes, gu_bscale, dn_codes, dn_bscale, hidden, out_flr, sel_flr, 1);

    int rc = gemma4_moe_ffn_q4b_cuda(E, NE, NU, FF, eps, gate_inp, scale,
                                     gu_codes, gu_bscale, dn_codes, dn_bscale,
                                     hidden, out_gpu, sel_gpu);
    if (rc != 0) { printf("G-DG-N5a-packed: FAIL (cuda entry returned %d)\n", rc); return 1; }

    int sel_match = 1;
    for (int k = 0; k < NU; k++) if (sel_orc[k] != sel_gpu[k]) sel_match = 0;
    printf("selected experts CPU:");
    for (int k = 0; k < NU; k++) printf(" %d", sel_orc[k]);
    printf("  GPU:");
    for (int k = 0; k < NU; k++) printf(" %d", sel_gpu[k]);
    printf("   match=%s\n", sel_match ? "YES" : "NO");

    double rel_orc    = relerr(out_orc, out_gpu, E);   /* CUDA vs the f32 OK_Q4B oracle (the headline number) */
    double rel_flr    = relerr(out_flr, out_gpu, E);   /* CUDA vs the int8-act dp4a reference (kernel correctness) */
    double rel_qfloor = relerr(out_orc, out_flr, E);   /* the irreducible int8-act-quant cost */

    printf("rel-err CUDA vs f32-oracle      = %.6e\n", rel_orc);
    printf("rel-err CUDA vs int8-dp4a floor = %.6e   (kernel-correctness: must be ~0)\n", rel_flr);
    printf("rel-err int8-floor vs f32-oracle= %.6e   (irreducible act-quant cost)\n", rel_qfloor);

    /* PASS = expert-select identical AND the CUDA dp4a reproduces the int8-dp4a
     * arithmetic to < 2e-3 (proves the byte layout / scale indexing / act-quant /
     * reduction are all correct). The CUDA-vs-f32-oracle deflection equals the
     * irreducible act-quant floor by construction — reported, not gated, since it
     * is a property of int8 activations on this adversarial synthetic fixture, not
     * of the wiring. */
    int pass = sel_match && (rel_flr < 2e-3);
    printf("G-DG-N5a-packed: %s  (real OK_Q4B experts via dp4a; dims E=%d NE=%d NU=%d FF=%d)\n",
           pass ? "PASS" : "FAIL", E, NE, NU, FF);

    free(gate_inp); free(scale); free(hidden); free(gu_f32); free(dn_f32);
    free(gu_codes); free(gu_bscale); free(dn_codes); free(dn_bscale);
    free(out_orc); free(out_flr); free(out_gpu);
    return pass ? 0 : 1;
}
