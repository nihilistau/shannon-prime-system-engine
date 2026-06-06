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
int  gemma4_forward_cuda(const qwen3_model *m, const int32_t *tokens, int n_tok,
                         float *logits);
int  gemma4_decode_cuda(const qwen3_model *m, int32_t *seq, int n_prompt,
                        int n_gen, int eos_id);
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

/* GELU tanh approximation — verbatim gemma4.c g4_gelu (static in the oracle TU). */
static float g4_gelu(float v) {
    const float k = 0.7978845608028654f;
    return 0.5f * v * (1.0f + tanhf(k * (v + 0.044715f * v * v * v)));
}

/* ═══ Truncated parity harness (ETA.2/3): CPU mirror of gemma4.c through
 * n_layers (full layers, attention-only at the LAST), built from the oracle's
 * OWN core primitives in the oracle's exact order, vs gemma4_cuda_probe at the
 * same boundary. Drops the 6-stage telemetry at the last layer; abs_gates =
 * {nx, q, ao, ap, x} or NULL for telemetry-only (first run measures the floor,
 * then the gates get pinned at ~3x — the L0 discipline). NO AltUp/out_scale on
 * EITHER side (ETA.4); the boundary is before the injection point. ═══ */
static void truncated_parity(const qwen3_model *m, int n_layers, const char *tag,
                             const double *abs_gates) {
    enum { NT = 12 };
    const int32_t toks[NT] = { 2, 10, 100, 1000, 5000, 9999, 31, 7, 42, 256, 777, 12345 };
    const qwen3_config *c = &m->cfg;
    const int E = (int)c->n_embd, SW = (int)c->sliding_window, FF = (int)c->n_ff;
    const float eps = c->rms_eps, embscale = sqrtf((float)E);
    const int period = (int)c->g4_swa_period ? (int)c->g4_swa_period : 6;
    const int NL = (int)c->n_layers;
    const int kvfs = (int)c->g4_n_kv_from_start ? (int)c->g4_n_kv_from_start : NL;
    const int g_nh = (int)c->n_head, g_nkv = (int)c->n_head_kv, g_hd = (int)c->head_dim;
    const int s_nh = (int)c->g4_nh_swa, s_nkv = (int)c->g4_nkv_swa, s_hd = (int)c->g4_hd_swa;
    const float g_base = c->rope_freq_base, s_base = c->g4_rope_base_swa;
    const int QDmax = (g_nh*g_hd > s_nh*s_hd) ? g_nh*g_hd : s_nh*s_hd;
    const int KVDmax = (g_nkv*g_hd > s_nkv*s_hd) ? g_nkv*g_hd : s_nkv*s_hd;
    int FFmax = FF;
    for (int L = 0; L < n_layers; L++) {
        const gguf_tensor *fg = m->layers[L].ffn_gate;
        int f = (fg && fg->n_dims >= 2 && fg->dims[1] > 0) ? (int)fg->dims[1] : FF;
        if (f > FFmax) FFmax = f;
    }

    float *x   = (float *)malloc((size_t)NT * E * sizeof(float));
    float *nx  = (float *)malloc((size_t)NT * E * sizeof(float));
    float *nx0 = (float *)malloc((size_t)NT * E * sizeof(float));
    float *q   = (float *)malloc((size_t)NT * QDmax * sizeof(float));
    /* per-OWNER K/V storage (shared-KV mirror of the oracle: owners [0,kvfs)
     * compute+store; sharers reuse owner kvfs-1 (global) / kvfs-2 (SWA) and skip
     * their own projection + norms — gemma4.c lines 173-193) */
    float **Kst = (float **)calloc((size_t)NL, sizeof(float *));
    float **Vst = (float **)calloc((size_t)NL, sizeof(float *));
    float *ao  = (float *)malloc((size_t)NT * QDmax * sizeof(float));
    float *ap  = (float *)malloc((size_t)NT * E * sizeof(float));
    float *g   = (float *)malloc((size_t)NT * FFmax * sizeof(float));
    float *up  = (float *)malloc((size_t)NT * FFmax * sizeof(float));
    float *dn  = (float *)malloc((size_t)NT * E * sizeof(float));
    float *sc  = (float *)malloc((size_t)NT * sizeof(float));
    float *gq  = (float *)malloc((size_t)NT * QDmax * sizeof(float));
    SP_CHECK(x && nx && nx0 && q && Kst && Vst && ao && ap && g && up && dn && sc && gq,
             "parity buffers");
    if (!(x && nx && nx0 && q && Kst && Vst && ao && ap && g && up && dn && sc && gq)) goto fin;

    {
        int cpu_ok = 1, last_qd = 0;
        for (int t = 0; t < NT && cpu_ok; t++) {
            if (sp_embed_row(m, toks[t], E, x + (size_t)t * E)) cpu_ok = 0;
            for (int i = 0; i < E; i++) x[(size_t)t * E + i] *= embscale;
        }
        for (int L = 0; L < n_layers && cpu_ok; L++) {
            const qwen3_layer *ly = &m->layers[L];
            const int global = ((L % period) == period - 1);
            const int nh  = global ? g_nh  : s_nh;
            const int nkv = global ? g_nkv : s_nkv;
            const int hd  = global ? g_hd  : s_hd;
            const int grp = nh / nkv, qd = nh * hd, kvd = nkv * hd;
            const float rbase = global ? g_base : s_base;
            const float *ffac = global ? sp_as_f32(m, m->rope_freqs) : NULL;
            const int win = global ? -1 : SW;
            const gguf_tensor *fg = ly->ffn_gate;
            const int ffL = (fg && fg->n_dims >= 2 && fg->dims[1] > 0) ? (int)fg->dims[1] : FF;
            const int last = (L == n_layers - 1);
            if (last) last_qd = qd;

            for (int t = 0; t < NT; t++)
                sp_rmsnorm(x + (size_t)t * E, sp_as_f32(m, ly->attn_norm), E, eps, nx + (size_t)t * E);
            if (last) memcpy(nx0, nx, (size_t)NT * E * sizeof(float));
            if (sp_matmul(m, ly->attn_q, nx, NT, E, qd, q)) { cpu_ok = 0; break; }
            {
                const float *qn = sp_as_f32(m, ly->attn_q_norm);
                for (int t = 0; t < NT; t++)
                    for (int h = 0; h < nh; h++) {
                        float *qh = q + (size_t)t * qd + (size_t)h * hd;
                        sp_rmsnorm_head(qh, qn, hd, eps);
                        sp_rope_neox_freqs(qh, hd, t, rbase, ffac);
                    }
            }
            float *Kuse, *Vuse;
            if (L < kvfs) {                       /* OWNER: project + norm + store */
                float *K  = (float *)malloc((size_t)NT * kvd * sizeof(float));
                float *Vb = (float *)malloc((size_t)NT * kvd * sizeof(float));
                if (!K || !Vb) { free(K); free(Vb); cpu_ok = 0; break; }
                if (sp_matmul(m, ly->attn_k, nx, NT, E, kvd, K))  { free(K); free(Vb); cpu_ok = 0; break; }
                if (sp_matmul(m, ly->attn_v, nx, NT, E, kvd, Vb)) { free(K); free(Vb); cpu_ok = 0; break; }
                const float *kn = sp_as_f32(m, ly->attn_k_norm);
                for (int t = 0; t < NT; t++)
                    for (int h = 0; h < nkv; h++) {
                        float *kh = K + (size_t)t * kvd + (size_t)h * hd;
                        sp_rmsnorm_head(kh, kn, hd, eps);
                        sp_rope_neox_freqs(kh, hd, t, rbase, ffac);
                        float *vh = Vb + (size_t)t * kvd + (size_t)h * hd;
                        double ss = 0.0;
                        for (int i = 0; i < hd; i++) ss += (double)vh[i] * (double)vh[i];
                        float inv = 1.0f / sqrtf((float)(ss / (double)hd) + eps);
                        for (int i = 0; i < hd; i++) vh[i] *= inv;
                    }
                Kst[L] = K; Vst[L] = Vb; Kuse = K; Vuse = Vb;
            } else {                              /* SHARER: reuse owner, skip projection */
                const int src = kvfs - (global ? 1 : 2);
                Kuse = Kst[src]; Vuse = Vst[src];
                if (!Kuse || !Vuse) { cpu_ok = 0; break; }
            }
            for (int t = 0; t < NT; t++)
                for (int h = 0; h < nh; h++)
                    sp_attn_head(q + (size_t)t * qd + (size_t)h * hd, Kuse, Vuse, t, kvd,
                                 h / grp, hd, 1.0f, win, sc, ao + (size_t)t * qd + (size_t)h * hd);
            if (sp_matmul(m, ly->attn_output, ao, NT, qd, E, ap)) { cpu_ok = 0; break; }
            for (int t = 0; t < NT; t++) {
                sp_rmsnorm(ap + (size_t)t * E, sp_as_f32(m, ly->post_attn_norm), E, eps, nx + (size_t)t * E);
                float *xt = x + (size_t)t * E; const float *pt = nx + (size_t)t * E;
                for (int i = 0; i < E; i++) xt[i] += pt[i];
            }
            if (last) break;   /* boundary A: attention residual of the last layer */

            for (int t = 0; t < NT; t++)
                sp_rmsnorm(x + (size_t)t * E, sp_as_f32(m, ly->ffn_norm), E, eps, nx + (size_t)t * E);
            if (sp_matmul(m, ly->ffn_gate, nx, NT, E, ffL, g))  { cpu_ok = 0; break; }
            if (sp_matmul(m, ly->ffn_up,   nx, NT, E, ffL, up)) { cpu_ok = 0; break; }
            for (size_t i = 0; i < (size_t)NT * ffL; i++) g[i] = g4_gelu(g[i]) * up[i];
            if (sp_matmul(m, ly->ffn_down, g, NT, ffL, E, dn))  { cpu_ok = 0; break; }
            for (int t = 0; t < NT; t++) {
                sp_rmsnorm(dn + (size_t)t * E, sp_as_f32(m, ly->post_ffw_norm), E, eps, nx + (size_t)t * E);
                float *xt = x + (size_t)t * E; const float *pt = nx + (size_t)t * E;
                for (int i = 0; i < E; i++) xt[i] += pt[i];
            }
        }
        SP_CHECK(cpu_ok, "CPU truncated mirror computed");

        if (cpu_ok) {
            struct { int stage; const float *cpu; size_t n; const char *nm; } stg[] = {
                { 2, nx0, (size_t)NT * E,       "nx (attn_norm)   " },
                { 3, q,   (size_t)NT * last_qd, "q  (norm+rope)   " },
                { 4, ao,  (size_t)NT * last_qd, "ao (attention)   " },
                { 5, ap,  (size_t)NT * E,       "ap (Wo, pre-norm)" },
                { 1, x,   (size_t)NT * E,       "x  (attn residual)" },
            };
            for (int s = 0; s < 5; s++) {
                int rc2 = gemma4_cuda_probe(m, toks, NT, n_layers, stg[s].stage, gq);
                if (rc2) { fprintf(stderr, "    [%s] stage %d probe: %s\n", tag, stg[s].stage, sp_last_error()); continue; }
                double mr = 0.0, ma = 0.0;
                for (size_t i = 0; i < stg[s].n; i++) {
                    double d = fabs((double)gq[i] - (double)stg[s].cpu[i]);
                    double den = fabs((double)stg[s].cpu[i]) > 1e-6 ? fabs((double)stg[s].cpu[i]) : 1e-6;
                    if (d > ma) ma = d;
                    if (d / den > mr) mr = d / den;
                }
                fprintf(stderr, "    [g4-cuda-%s] stage %s max rel %.3e  max abs %.3e%s\n",
                        tag, stg[s].nm, mr, ma, abs_gates ? "" : "  (telemetry)");
                if (abs_gates) {
                    double gate = abs_gates[s];
                    SP_CHECK((gate == 0.0 && ma == 0.0) || (gate > 0.0 && ma <= gate),
                             "stage at the measured f32 floor");
                }
            }
        }
    }
fin:
    free(x); free(nx); free(nx0); free(q); free(ao); free(ap);
    free(g); free(up); free(dn); free(sc); free(gq);
    if (Kst) { for (int L = 0; L < NL; L++) free(Kst[L]); free(Kst); }
    if (Vst) { for (int L = 0; L < NL; L++) free(Vst[L]); free(Vst); }
}

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

    /* ════ E_G4_CU_L0 / E_G4_CU_L4 (ETA.2/3): TRUNCATED PARITY via the harness ════
     * L0 (SWA, n_layers=1): gates PINNED at ~3x the measured floors (bisection
     * 2026-06-06): nx BIT-EXACT; q 8.6e-6 / ao 3.2e-5 / ap 6.3e-5 abs (f32 GEMM
     * floor, gemm_w_lift oracle arithmetic); residual 1.59e-3 = the post_attn_norm
     * 1/rms(ap)~25x amplification of that floor (mechanism on record above).
     * L4 (the FIRST GLOBAL layer, n_layers=5): the geometry shift — hd 256->512,
     * qd 2048->4096, nkv geometry change, rope_freqs proportional table engages,
     * SWA mask drops to full causal. First run = TELEMETRY (NULL gates); pinned
     * after measurement, the L0 discipline. */
    sp_kernels_read_env();
    {
        static const double l0_gates[5] = { 0.0, 5e-5, 1e-4, 2e-4, 5e-3 };
        truncated_parity(m, 1, "L0", l0_gates);
        /* L4 gates pinned at ~3x the measured floors (telemetry run 2026-06-06:
         * nx 1.11e-4 / q 1.15e-5 / ao 5.15e-5 / ap 8.15e-5 / x 1.59e-3 abs).
         * nx is no longer bit-exact at depth — 4 layers of norm-amplified inflow,
         * re-condensed by attn_norm (stable, no explosion). q at the floor PROVES
         * the rope_freqs proportional handoff + the 4096-wide global projection;
         * ao at the floor proves the full-causal (win=-1) mask switch. */
        static const double l4_gates[5] = { 3e-4, 5e-5, 1.5e-4, 2.5e-4, 5e-3 };
        truncated_parity(m, 5, "L4", l4_gates);
        /* THE SHARER SEAM (ETA.3 remainder): L15 = the FIRST shared-KV layer
         * (kvfs=15; layers [0,15) own). L15 is SWA (15%5==0) -> reads owner
         * L13 (kvfs-2)'s STORED K/V from VRAM and skips its own projection.
         * This is the cross-layer VRAM dependency: wrong owner index = stale
         * layer; wrong stride = OOB. Both mirror + probe implement the oracle's
         * owner-store/sharer-reuse. Telemetry first; pin after measurement. */
        /* Gates pinned at ~3x measured (telemetry 2026-06-06: nx 2.37e-4 /
         * q 2.67e-5 / ao 1.11e-5 / ap 2.00e-5 / x 2.98e-3 abs). ao AT THE FLOOR
         * is the seam proof: an off-by-one owner index would read L14's GLOBAL
         * K/V (kvd 512) through an SWA stride (256) -> garbage, not 1.1e-5. */
        static const double l15_gates[5] = { 7e-4, 8e-5, 5e-5, 6e-5, 1e-2 };
        truncated_parity(m, 16, "L15-sharer", l15_gates);
    }

    /* ════ E_G4_CU_FULL (ETA.4): THE DECISIVE GATE — the live run. The full
     * 35-layer gemma4_forward_cuda (per-layer geometry + shared-KV + rope_freqs
     * + AltUp + out_scale + tied head + softcap) vs the CPU oracle
     * gemma4_forward, per-position ARGMAX + KL(softmax_cpu || softmax_cuda).
     * The repo's standard cross-backend currency. ════ */
    {
        enum { NT = 12 };
        const int32_t toks[NT] = { 2, 10, 100, 1000, 5000, 9999, 31, 7, 42, 256, 777, 12345 };
        const int V = (int)m->cfg.n_vocab;
        float *cl = (float *)malloc((size_t)NT * V * sizeof(float));
        float *gl = (float *)malloc((size_t)NT * V * sizeof(float));
        SP_CHECK(cl && gl, "full-forward logits buffers");
        if (cl && gl) {
            int orc = gemma4_forward(m, toks, NT, cl);
            SP_CHECK(orc == 0, "CPU oracle gemma4_forward");
            int crc = gemma4_forward_cuda(m, toks, NT, gl);
            if (crc) fprintf(stderr, "    full cuda: %s\n", sp_last_error());
            SP_CHECK(crc == 0, "gemma4_forward_cuda ran");
            if (orc == 0 && crc == 0) {
                int agree = 0; double max_kl = 0.0, max_abs = 0.0;
                for (int t = 0; t < NT; t++) {
                    const float *cp = cl + (size_t)t * V, *gp = gl + (size_t)t * V;
                    /* argmax */
                    int ac = 0, ag = 0;
                    for (int i = 1; i < V; i++) {
                        if (cp[i] > cp[ac]) ac = i;
                        if (gp[i] > gp[ag]) ag = i;
                    }
                    if (ac == ag) agree++;
                    /* KL(p_cpu || q_cuda), double-precision log-softmax */
                    double mc = cp[0], mg = gp[0];
                    for (int i = 1; i < V; i++) { if (cp[i] > mc) mc = cp[i]; if (gp[i] > mg) mg = gp[i]; }
                    double zc = 0.0, zg = 0.0;
                    for (int i = 0; i < V; i++) { zc += exp((double)cp[i] - mc); zg += exp((double)gp[i] - mg); }
                    double lzc = log(zc), lzg = log(zg), kl = 0.0;
                    for (int i = 0; i < V; i++) {
                        double lp = (double)cp[i] - mc - lzc;
                        double lq = (double)gp[i] - mg - lzg;
                        kl += exp(lp) * (lp - lq);
                        double d = fabs((double)cp[i] - (double)gp[i]);
                        if (d > max_abs) max_abs = d;
                    }
                    if (kl > max_kl) max_kl = kl;
                }
                fprintf(stderr, "    [g4-cuda-FULL] argmax %d/%d  max KL %.3e  max |dlogit| %.3e\n",
                        agree, NT, max_kl, max_abs);
                SP_CHECK(agree == NT, "FULL 35-layer: CUDA argmax == CPU oracle argmax (all positions)");
                SP_CHECK(max_kl < 1e-4, "FULL 35-layer: KL(cpu||cuda) at the cross-backend floor");
            }
        }
        free(cl); free(gl);
    }

    /* ════ E_G4_CU_DEC (ETA.5a): THE DECODE GATE — autoregressive generation
     * over the JAGGED shared-KV cache (per-owner widths, sharers read owners),
     * per-step AltUp, windowed single-query attention, head + softcap + argmax.
     * Gate: the CPU ORACLE prefill over the produced sequence must teacher-
     * forced argmax-predict EVERY generated token (the proven decode pattern). ════ */
    {
        enum { NP = 4, NG = 12, PT = NP + NG };
        int32_t dseq[PT]; const int32_t prompt[NP] = { 2, 10, 100, 1000 };
        const int V = (int)m->cfg.n_vocab;
        for (int i = 0; i < NP; i++) dseq[i] = prompt[i];
        int dn = gemma4_decode_cuda(m, dseq, NP, NG, /*eos=*/-1);
        if (dn < 0) fprintf(stderr, "    decode: %s\n", sp_last_error());
        SP_CHECK(dn == PT, "gemma4 CUDA decode produced full length");
        if (dn == PT) {
            float *tl = (float *)malloc((size_t)PT * V * sizeof(float));
            SP_CHECK(tl != NULL, "teacher-forced logits buffer");
            if (tl) {
                int orc = gemma4_forward(m, dseq, PT, tl);
                SP_CHECK(orc == 0, "oracle prefill over decoded sequence");
                if (orc == 0) {
                    int ok = 1, firstbad = -1;
                    for (int pos = NP - 1; pos < PT - 1 && ok; pos++) {
                        const float *row = tl + (size_t)pos * V;
                        int am = 0;
                        for (int i = 1; i < V; i++) if (row[i] > row[am]) am = i;
                        if (am != dseq[pos + 1]) { ok = 0; firstbad = pos; }
                    }
                    fprintf(stderr, "    [g4-cuda-DEC] %d gen tokens; oracle teacher-forced match: %s",
                            NG, ok ? "ALL" : "FAIL");
                    if (!ok) fprintf(stderr, " (first bad pos %d)", firstbad);
                    fprintf(stderr, "; seq:");
                    for (int i = 0; i < PT; i++) fprintf(stderr, " %d", dseq[i]);
                    fprintf(stderr, "\n");
                    SP_CHECK(ok, "DECODE: every generated token == oracle argmax (jagged KV + AltUp per step)");
                }
                free(tl);
            }
        }
    }

    sp_cuda_model_release(m);
    qwen3_free(m);
    sp_model_unload(handle);
}

int main(void) {
    SP_RUN(T_GEMMA4_CUDA_WEIGHTS);
    return SP_DONE();
}
