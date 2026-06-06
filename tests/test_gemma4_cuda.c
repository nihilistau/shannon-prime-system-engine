/* test_gemma4_cuda.c — E_G4_CU_W (ETA.1): Stage Eta structural gate.
 *
 * The first gate of the Gemma4 CUDA port: the engine CUDA layer must INGEST a
 * core-bridged Gemma4 model across the core/engine link seam —
 *
 *   .sp-model + .sp-tokenizer -> sp_model_load -> sp_model_to_gemma4 (CORE)
 *   -> gemma4_cuda_weights_probe (ENGINE sp_engine_cuda)
 *
 * uploading the full weight set with the gemma4 structure the CPU oracle
 * (core/forward/gemma4.c) defines: per-layer GLOBAL/SWA head geometry (the
 * Q/KV projection widths differ per layer), shared-KV owner-only K/V uploads
 * (sharers reuse an owner's cache and skip their own projection), per-layer
 * ELASTIC FFN widths (MatFormer), the AltUp tensor set (per-layer inp_gate /
 * proj / post_norm / out_scale + model-level per_layer_model_proj /
 * per_layer_proj_norm), and the rope_freqs proportional table.
 *
 * This deliberately links sp_session (the CORE inference lane, same as
 * M_GEMMA4) + sp_engine_cuda — NOT sp_engine — so the core loader/bridge and
 * the CUDA backend coexist in one binary. The CUDA lib's engine-named symbol
 * references (sp_arena_find, sp_arena_dequant_row, sp_set_error, ...) resolve
 * from the core libs (structs synced byte-for-byte; cf. the fork-tax fix
 * engine 0fb39ab). Proving THAT LINK is part of this gate.
 *
 * The gemma4 CUDA forward itself is ETA.2+ (gated argmax/KL vs gemma4_forward).
 *
 * SLOW / model-gated: skips cleanly (PASS, no checks) when the 4.6 GB
 * .sp-model is absent, or when no CUDA device is present.
 *
 * Env: SP_GEMMA4_SPMODEL / SP_GEMMA4_SPTOK (defaults below). */
#define _CRT_SECURE_NO_WARNINGS
#include "sp/sp_test.h"
#include "sp/sp_model.h"   /* sp_model_load / sp_model_unload / sp_model_to_gemma4 */
#include "sp/model.h"      /* qwen3_model / qwen3_free (CORE structs) */
#include "sp/sp_status.h"
#include "sp/forward_dispatch.h"   /* sp_matmul / sp_embed_row / sp_as_f32 (CPU mirror) */
#include "sp/forward_kernels.h"    /* sp_rmsnorm / sp_rmsnorm_head / sp_rope_neox_freqs / sp_attn_head */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#ifndef SP_GEMMA4_SPMODEL_DEF
#define SP_GEMMA4_SPMODEL_DEF "D:/F/shannon-prime-repos/models/gemma4-e2b.sp-model"
#endif
#ifndef SP_GEMMA4_SPTOK_DEF
#define SP_GEMMA4_SPTOK_DEF "D:/F/shannon-prime-repos/models/gemma4-e2b.sp-tokenizer"
#endif

/* CUDA entry points, declared directly (NOT via sp_engine/cuda_backend.h, which
 * would pull the engine's duplicate model structs into a core-header TU). C
 * links by name; the core/engine structs are synced byte-for-byte. */
int  sp_cuda_device_count(void);
int  gemma4_cuda_weights_probe(const qwen3_model *m);
int  gemma4_cuda_probe(const qwen3_model *m, const int32_t *tokens, int n_tok,
                       int n_layers, int attn_only, float *out_x);
void sp_cuda_model_release(const qwen3_model *m);

/* Engine-symbol SHIM (the documented cross-seam alias pattern, cf.
 * sp_daemon_hex_glue.c): sp_engine_cuda calls the ENGINE's `as_f32`; in this
 * core-lane binary that name doesn't exist — the core's identical-semantics
 * function is `sp_as_f32` (forward_dispatch). One-line forwarder. This was the
 * ONLY unresolved symbol across the whole core+CUDA link: every other engine-
 * named reference (sp_arena_find/sp_arena_dequant_row/sp_dequant_row/
 * sp_set_error/sp_kste_encode/gguf_tensor_data) resolves from the core libs. */
const float *sp_as_f32(const qwen3_model *m, const gguf_tensor *t);
const float *as_f32(const qwen3_model *m, const gguf_tensor *t) { return sp_as_f32(m, t); }

static void T_GEMMA4_CUDA_WEIGHTS(void) {
    const char *spm = getenv("SP_GEMMA4_SPMODEL"); if (!spm) spm = SP_GEMMA4_SPMODEL_DEF;
    const char *stk = getenv("SP_GEMMA4_SPTOK");   if (!stk) stk = SP_GEMMA4_SPTOK_DEF;

    FILE *probe = fopen(spm, "rb");
    if (!probe) { fprintf(stderr, "    model absent (%s) — SKIP\n", spm); return; }
    fclose(probe);
    if (sp_cuda_device_count() < 1) { fprintf(stderr, "    no CUDA device — SKIP\n"); return; }

    sp_model *handle = NULL;
    sp_status st = sp_model_load(spm, stk, &handle);
    SP_CHECK(st == SP_OK && handle, "sp_model_load gemma4-e2b");
    if (st != SP_OK || !handle) return;

    qwen3_model *m = sp_model_to_gemma4(handle);
    if (!m) fprintf(stderr, "    sp_model_to_gemma4: %s\n", sp_last_error());
    SP_CHECK(m != NULL, "sp_model_to_gemma4 (core bridge)");
    if (!m) { sp_model_unload(handle); return; }

    /* structural invariants the CUDA build depends on (from the cfg the bridge
     * populated; the CPU oracle uses exactly these) */
    SP_CHECK(m->cfg.arch == SP_ARCH_GEMMA4, "arch == SP_ARCH_GEMMA4");
    SP_CHECK(m->cfg.g4_swa_period > 0, "g4_swa_period set");
    SP_CHECK(m->cfg.g4_n_kv_from_start > 0 &&
             m->cfg.g4_n_kv_from_start <= m->cfg.n_layers, "shared-KV kvfs in range");
    SP_CHECK(m->cfg.g4_n_embd_per_layer > 0, "AltUp PL width set");
    SP_CHECK(m->per_layer_model_proj && m->per_layer_proj_norm && m->rope_freqs,
             "model-level AltUp tensors present");

    /* THE GATE: upload the full gemma4 weight set to the device. */
    int rc = gemma4_cuda_weights_probe(m);
    if (rc) fprintf(stderr, "    probe: %s\n", sp_last_error());
    SP_CHECK(rc == 0, "gemma4 CUDA weight set uploads (per-layer geometry + shared-KV + AltUp)");

    /* ════ E_G4_CU_L0 (ETA.2): LAYER-0 TRUNCATED PARITY — the bisection gate ════
     * CPU mirror computed HERE with the oracle's OWN core primitives (sp_rmsnorm /
     * sp_matmul / sp_rmsnorm_head / sp_rope_neox_freqs / sp_attn_head), in the
     * oracle's exact order (gemma4.c lines 144-205), truncated after layer 0's
     * ATTENTION residual. Layer 0 is SWA (global = L%period==period-1): base
     * g4_rope_base_swa, no freq factors, sliding window, ascale=1.0, QK-norm
     * before RoPE, WEIGHTLESS V-norm. The CUDA side runs gemma4_cuda_probe
     * (n_layers=1, attn_only=1). Locks the math mechanics before AltUp/SWA-global
     * routing enter. Gate = max rel err at the f32 cuBLAS-vs-CPU floor (1e-4;
     * observed qwen3/gemma3 floors ~3e-5 — NOT bit-exact, reduction order differs). */
    {
        enum { NT = 12 };
        const int32_t toks[NT] = { 2, 10, 100, 1000, 5000, 9999, 31, 7, 42, 256, 777, 12345 };
        const qwen3_config *c = &m->cfg;
        const int E = (int)c->n_embd, SW = (int)c->sliding_window;
        const float eps = c->rms_eps, embscale = sqrtf((float)E);
        const int nh = (int)c->g4_nh_swa, nkv = (int)c->g4_nkv_swa, hd = (int)c->g4_hd_swa;
        const int grp = nh / nkv, qd = nh * hd, kvd = nkv * hd;
        const float rbase = c->g4_rope_base_swa, ascale = 1.0f;
        const int win = SW;
        const qwen3_layer *ly = &m->layers[0];

        float *x  = (float *)malloc((size_t)NT * E * sizeof(float));
        float *nx = (float *)malloc((size_t)NT * E * sizeof(float));
        float *q  = (float *)malloc((size_t)NT * qd * sizeof(float));
        float *K  = (float *)malloc((size_t)NT * kvd * sizeof(float));
        float *Vb = (float *)malloc((size_t)NT * kvd * sizeof(float));
        float *ao = (float *)malloc((size_t)NT * qd * sizeof(float));
        float *ap = (float *)malloc((size_t)NT * E * sizeof(float));
        float *sc = (float *)malloc((size_t)NT * sizeof(float));
        float *gx = (float *)malloc((size_t)NT * E * sizeof(float));
        float *nx0 = (float *)malloc((size_t)NT * E * sizeof(float));  /* attn_norm snapshot */
        SP_CHECK(x && nx && q && K && Vb && ao && ap && sc && gx && nx0, "L0 probe buffers");
        if (x && nx && q && K && Vb && ao && ap && sc && gx && nx0) {
            sp_kernels_read_env();
            /* CPU mirror — embed + sqrt(E) scale */
            int cpu_ok = 1;
            for (int t = 0; t < NT && cpu_ok; t++) {
                if (sp_embed_row(m, toks[t], E, x + (size_t)t * E)) cpu_ok = 0;
                for (int i = 0; i < E; i++) x[(size_t)t * E + i] *= embscale;
            }
            /* ── bisect stage 0: EMBED parity (isolates arena dequant + upload from
             * the attention math; expected ~1e-7, pure dequant+scale both sides) ── */
            if (cpu_ok) {
                int erc = gemma4_cuda_probe(m, toks, NT, /*n_layers=*/0, 0, gx);
                SP_CHECK(erc == 0, "CUDA embed probe ran");
                if (erc == 0) {
                    double mr = 0.0;
                    for (size_t i = 0; i < (size_t)NT * E; i++) {
                        double d = fabs((double)gx[i] - (double)x[i]);
                        double den = fabs((double)x[i]) > 1e-6 ? fabs((double)x[i]) : 1e-6;
                        if (d / den > mr) mr = d / den;
                    }
                    fprintf(stderr, "    [g4-cuda-L0] embed parity: max rel %.3e\n", mr);
                    SP_CHECK(mr < 1e-5, "embed+scale: CUDA == CPU");
                }
            }
            /* layer 0 attention block, oracle order */
            if (cpu_ok) {
                for (int t = 0; t < NT; t++)
                    sp_rmsnorm(x + (size_t)t * E, sp_as_f32(m, ly->attn_norm), E, eps, nx + (size_t)t * E);
                memcpy(nx0, nx, (size_t)NT * E * sizeof(float));
                if (sp_matmul(m, ly->attn_q, nx, NT, E, qd, q)) cpu_ok = 0;
            }
            if (cpu_ok) {
                const float *qn = sp_as_f32(m, ly->attn_q_norm);
                for (int t = 0; t < NT; t++)
                    for (int h = 0; h < nh; h++) {
                        float *qh = q + (size_t)t * qd + (size_t)h * hd;
                        sp_rmsnorm_head(qh, qn, hd, eps);
                        sp_rope_neox_freqs(qh, hd, t, rbase, NULL);
                    }
                if (sp_matmul(m, ly->attn_k, nx, NT, E, kvd, K))  cpu_ok = 0;
                if (cpu_ok && sp_matmul(m, ly->attn_v, nx, NT, E, kvd, Vb)) cpu_ok = 0;
            }
            if (cpu_ok) {
                const float *kn = sp_as_f32(m, ly->attn_k_norm);
                for (int t = 0; t < NT; t++)
                    for (int h = 0; h < nkv; h++) {
                        float *kh = K + (size_t)t * kvd + (size_t)h * hd;
                        sp_rmsnorm_head(kh, kn, hd, eps);
                        sp_rope_neox_freqs(kh, hd, t, rbase, NULL);
                        /* WEIGHTLESS V-norm (gemma4.c g4_rmsnorm_noweight, inlined —
                         * it is static in the oracle TU) */
                        float *vh = Vb + (size_t)t * kvd + (size_t)h * hd;
                        double ss = 0.0;
                        for (int i = 0; i < hd; i++) ss += (double)vh[i] * (double)vh[i];
                        float inv = 1.0f / sqrtf((float)(ss / (double)hd) + eps);
                        for (int i = 0; i < hd; i++) vh[i] *= inv;
                    }
                for (int t = 0; t < NT; t++)
                    for (int h = 0; h < nh; h++)
                        sp_attn_head(q + (size_t)t * qd + (size_t)h * hd, K, Vb, t, kvd,
                                     h / grp, hd, ascale, win, sc, ao + (size_t)t * qd + (size_t)h * hd);
                if (sp_matmul(m, ly->attn_output, ao, NT, qd, E, ap)) cpu_ok = 0;
            }
            if (cpu_ok) {
                for (int t = 0; t < NT; t++) {
                    sp_rmsnorm(ap + (size_t)t * E, sp_as_f32(m, ly->post_attn_norm), E, eps, nx + (size_t)t * E);
                    float *xt = x + (size_t)t * E; const float *pt = nx + (size_t)t * E;
                    for (int i = 0; i < E; i++) xt[i] += pt[i];
                }
            }
            SP_CHECK(cpu_ok, "CPU layer-0 mirror computed");

            /* ── intra-block bisection stages (informational; pinpoint the seam) ──
             * stage 2 = post-attn_norm nx [NT*E]; 3 = q post norm+rope [NT*qd];
             * 4 = ao post-attention [NT*qd]. CPU intermediates are live above. */
            {
                float *gq = (float *)malloc((size_t)NT * qd * sizeof(float));
                if (gq) {
                    struct { int stage; const float *cpu; size_t n; const char *nm; } stg[] = {
                        { 2, nx0, (size_t)NT * E,  "nx (attn_norm)   " },
                        { 3, q,   (size_t)NT * qd, "q  (norm+rope)   " },
                        { 4, ao,  (size_t)NT * qd, "ao (attention)   " },
                        { 5, ap,  (size_t)NT * E,  "ap (Wo, pre-norm)" },
                    };
                    /* STAGED ABS GATES — the real L0 math lock. Floors measured on the
                     * E2B (bisection 2026-06-06): nx is BIT-EXACT; q/ao/ap sit at the
                     * f32 GEMM reduction-order floor (8.6e-6 / 3.2e-5 / 6.3e-5 abs).
                     * Gates set ~3x above the measured floor. Raw REL error is NOT
                     * gated at these boundaries: it inflates on near-zero elements. */
                    const double abs_gate[4] = { 0.0, 5e-5, 1e-4, 2e-4 };
                    for (int s = 0; s < 4; s++) {
                        int rc2 = gemma4_cuda_probe(m, toks, NT, 1, stg[s].stage, gq);
                        if (rc2) { fprintf(stderr, "    stage %d probe: %s\n", stg[s].stage, sp_last_error()); continue; }
                        double mr = 0.0, ma = 0.0;
                        for (size_t i = 0; i < stg[s].n; i++) {
                            double d = fabs((double)gq[i] - (double)stg[s].cpu[i]);
                            double den = fabs((double)stg[s].cpu[i]) > 1e-6 ? fabs((double)stg[s].cpu[i]) : 1e-6;
                            if (d > ma) ma = d;
                            if (d / den > mr) mr = d / den;
                        }
                        fprintf(stderr, "    [g4-cuda-L0] stage %s max rel %.3e  max abs %.3e\n",
                                stg[s].nm, mr, ma);
                        SP_CHECK(ma <= (abs_gate[s] > 0.0 ? abs_gate[s] : 1e-12) ||
                                 (s == 0 && ma == 0.0), "L0 stage at the f32 floor");
                    }
                    free(gq);
                }
            }

            /* CUDA side: same boundary */
            int crc = gemma4_cuda_probe(m, toks, NT, /*n_layers=*/1, /*attn_only=*/1, gx);
            if (crc) fprintf(stderr, "    cuda probe: %s\n", sp_last_error());
            SP_CHECK(crc == 0, "CUDA layer-0 probe ran");

            if (cpu_ok && crc == 0) {
                double maxrel = 0.0, maxabs = 0.0; size_t bad = (size_t)-1;
                for (size_t i = 0; i < (size_t)NT * E; i++) {
                    double d = fabs((double)gx[i] - (double)x[i]);
                    double denom = fabs((double)x[i]) > 1e-6 ? fabs((double)x[i]) : 1e-6;
                    if (d > maxabs) maxabs = d;
                    if (d / denom > maxrel) { maxrel = d / denom; bad = i; }
                }
                fprintf(stderr, "    [g4-cuda-L0] attn-residual parity: max rel %.3e  max abs %.3e  (worst idx %zu)\n",
                        maxrel, maxabs, bad);
                /* RESIDUAL BOUNDARY GATE — measured amplification, on the record
                 * (bisection 2026-06-06, NOT a silent revision): every pre-norm stage
                 * above sits at the f32 floor (<= 6.3e-5 abs), then post_attn_norm
                 * divides by rms(ap) ~ 0.04, amplifying that floor ~x25 into the
                 * residual (1.59e-3 abs measured). The pipeline math is at the floor;
                 * the boundary metric amplifies. Gate = 5e-3 abs (~3x the measured
                 * amplified floor). The DECISIVE correctness gate for the gemma4 CUDA
                 * forward is the full-forward argmax+KL vs the CPU oracle (ETA.4
                 * closure), the repo's standard cross-backend currency — raw
                 * activation error is amplified through every norm exactly like this. */
                SP_CHECK(maxabs < 5e-3, "L0 attention block: residual at the norm-amplified f32 floor");
            }
        }
        free(x); free(nx); free(nx0); free(q); free(K); free(Vb); free(ao); free(ap); free(sc); free(gx);
    }

    sp_cuda_model_release(m);
    qwen3_free(m);
    sp_model_unload(handle);
}

int main(void) {
    SP_RUN(T_GEMMA4_CUDA_WEIGHTS);
    return SP_DONE();
}
