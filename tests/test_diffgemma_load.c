/* test_diffgemma_load.c — G-DG-N1 (N1a LOADER portion).
 *
 * Loads the transcoded DiffusionGemma .sp-model (arch_id == 9) through the new
 * bridge sp_model_to_diffusion_gemma() and asserts it maps with NO missing-tensor
 * or shape error, then prints the resolved geometry + a few sample tensor shapes.
 * This de-risks the format side of the native DiffusionGemma port. It does NOT run
 * a forward (that is N1b, blocked on a CUDA MoE backbone) — pure load + print.
 *
 * Env (overridable):
 *   SP_DG_SPMODEL   the DiffusionGemma .sp-model   (default below)
 *   SP_DG_SPTOK     the paired .sp-tokenizer        (default below)
 *
 * PASS: sp_model_load + sp_model_to_diffusion_gemma succeed (non-NULL), the printed
 * geometry matches the inspected model, and the sample per-layer/global tensors
 * resolve with the expected dims. Skips cleanly (exit 0) if the model is absent. */
#define _CRT_SECURE_NO_WARNINGS
#include "sp/sp_test.h"
#include "sp/sp_model.h"
#include "sp/model.h"
#include "sp/sp_status.h"

#include <stdio.h>
#include <stdlib.h>

#ifndef SP_DG_SPMODEL_DEF
#define SP_DG_SPMODEL_DEF "C:/sp_models/diffusiongemma-26B-A4B.sp-model"
#endif
#ifndef SP_DG_SPTOK_DEF
#define SP_DG_SPTOK_DEF "C:/sp_models/diffusiongemma-26B-A4B.sp-tokenizer"
#endif

/* Print a tensor's resolved dims (from the synth gguf_tensor the bridge built).
 * Returns 1 if found+non-NULL, else 0 (and the caller's SP_CHECK fails). */
static int show_dims(const char *label, const gguf_tensor *t) {
    if (!t) { fprintf(stderr, "    %-28s = <NULL>\n", label); return 0; }
    fprintf(stderr, "    %-28s ndims=%u dims=[", label, t->n_dims);
    for (uint32_t d = 0; d < t->n_dims && d < 8u; d++)
        fprintf(stderr, "%s%llu", d ? "," : "", (unsigned long long)t->dims[d]);
    fprintf(stderr, "]\n");
    return 1;
}

static void T_DG_N1A_LOAD(void) {
    const char *spm = getenv("SP_DG_SPMODEL"); if (!spm || !*spm) spm = SP_DG_SPMODEL_DEF;
    const char *spt = getenv("SP_DG_SPTOK");   if (!spt || !*spt) spt = SP_DG_SPTOK_DEF;

    FILE *probe = fopen(spm, "rb");
    if (!probe) { fprintf(stderr, "    DiffusionGemma model absent (%s) — SKIP\n", spm); return; }
    fclose(probe);

    fprintf(stderr, "    model=%s\n    tok  =%s\n", spm, spt);

    sp_model *handle = NULL;
    sp_status st = sp_model_load(spm, spt, &handle);
    SP_CHECK(st == SP_OK && handle, "sp_model_load (DiffusionGemma)");
    if (st != SP_OK || !handle) {
        fprintf(stderr, "    [load failed] status=%d %s\n", (int)st, sp_last_error());
        return;
    }

    sp_arch_info ai;
    SP_CHECK(sp_model_arch(handle, &ai) == SP_OK, "sp_model_arch");
    SP_CHECK_EQ_I64(ai.arch_id, SP_ARCH_ID_DIFFUSION_GEMMA, "arch_id == 9 (DIFFUSION_GEMMA)");

    qwen3_model *m = sp_model_to_diffusion_gemma(handle);
    SP_CHECK(m != NULL, "sp_model_to_diffusion_gemma (no missing-tensor/shape error)");
    if (!m) {
        fprintf(stderr, "    [bridge failed] %s\n", sp_last_error());
        sp_model_unload(handle);
        return;
    }

    const qwen3_config *c = &m->cfg;
    fprintf(stderr, "\n    === DiffusionGemma resolved geometry ===\n");
    fprintf(stderr, "    arch_id            = %u\n", ai.arch_id);
    fprintf(stderr, "    n_layers           = %u\n", c->n_layers);
    fprintf(stderr, "    n_embd (hidden)    = %u\n", c->n_embd);
    fprintf(stderr, "    n_ff (dense MLP)   = %u\n", c->n_ff);
    fprintf(stderr, "    n_head / n_head_kv = %u / %u  (GLOBAL geom)\n", c->n_head, c->n_head_kv);
    fprintf(stderr, "    head_dim           = %u\n", c->head_dim);
    fprintf(stderr, "    n_vocab            = %u\n", c->n_vocab);
    fprintf(stderr, "    SWA  hd/nh/nkv     = %u / %u / %u  rope_base_swa=%g\n",
            c->g4_hd_swa, c->g4_nh_swa, c->g4_nkv_swa, c->g4_rope_base_swa);
    fprintf(stderr, "    swa_window         = %u  swa_period=%u\n", c->sliding_window, c->g4_swa_period);
    fprintf(stderr, "    rope_freq_base     = %g  rms_eps=%g\n", c->rope_freq_base, c->rms_eps);
    fprintf(stderr, "    g4_n_embd_per_layer= %u  (0 = NO AltUp/PLE)\n", c->g4_n_embd_per_layer);
    fprintf(stderr, "    g4_n_kv_from_start = %u  logit_softcap=%g  tied=%d\n",
            c->g4_n_kv_from_start, c->g4_logit_softcap, c->tied_embedding);
    fprintf(stderr, "    canvas_length      = %u\n", c->dg_canvas_length);
    fprintf(stderr, "    MoE n_expert       = %u  n_expert_used=%u  n_ff_exp=%u\n",
            c->q36_n_expert, c->q36_n_expert_used, c->q36_n_ff_exp);

    /* geometry assertions (against the inspected model) */
    SP_CHECK_EQ_I64(c->n_layers, 30, "n_layers == 30");
    SP_CHECK_EQ_I64(c->n_embd,   2816, "n_embd == 2816");
    SP_CHECK_EQ_I64(c->head_dim, 512, "head_dim == 512");
    SP_CHECK_EQ_I64(c->dg_canvas_length, 256, "canvas_length == 256");
    SP_CHECK_EQ_I64(c->q36_n_expert, 128, "n_expert == 128");
    SP_CHECK_EQ_I64(c->q36_n_expert_used, 8, "n_expert_used == 8");
    SP_CHECK_EQ_I64(c->q36_n_ff_exp, 704, "n_ff_exp == 704");
    SP_CHECK_EQ_I64(c->g4_n_embd_per_layer, 0, "no AltUp/PLE (n_embd_per_layer==0)");

    fprintf(stderr, "\n    === sample tensor shapes ===\n");
    SP_CHECK(show_dims("token_embd",                   m->token_embd),                 "token_embd resolved");
    SP_CHECK(show_dims("blk.0.attn_q",                 m->layers[0].attn_q),           "blk.0.attn_q resolved");
    SP_CHECK(show_dims("blk.0.ffn_gate_up_exps",       m->layers[0].ffn_gate_up_exps), "blk.0.ffn_gate_up_exps resolved (fused)");
    SP_CHECK(show_dims("blk.0.ffn_down_exps",          m->layers[0].ffn_down_exps),    "blk.0.ffn_down_exps resolved");
    SP_CHECK(show_dims("blk.0.ffn_gate_inp (router)",  m->layers[0].ffn_gate_inp),     "blk.0.ffn_gate_inp resolved");
    SP_CHECK(show_dims("blk.0.enc_out_scale",          m->layers[0].enc_out_scale),    "blk.0.enc_layer_output_scale resolved");
    SP_CHECK(show_dims("blk.0.pre_ffw_norm_2",         m->layers[0].pre_ffw_norm_2),   "blk.0.pre_ffw_norm_2 resolved");
    SP_CHECK(show_dims("self_cond_gate",               m->self_cond_gate),             "self_cond_gate resolved");
    SP_CHECK(show_dims("self_cond_pre_norm",           m->self_cond_pre_norm),         "self_cond_pre_norm resolved");
    SP_CHECK(show_dims("rope_freqs",                   m->rope_freqs),                 "rope_freqs resolved");

    /* the fused gate|up second dim must be n_ff_exp*2; down first dim must be n_ff_exp */
    if (m->layers[0].ffn_gate_up_exps && m->layers[0].ffn_gate_up_exps->n_dims >= 3)
        SP_CHECK_EQ_I64(m->layers[0].ffn_gate_up_exps->dims[1], (long long)c->q36_n_ff_exp * 2,
                        "ffn_gate_up_exps fused dim == n_ff_exp*2 (1408)");
    if (m->layers[0].ffn_gate_up_exps && m->layers[0].ffn_gate_up_exps->n_dims >= 3)
        SP_CHECK_EQ_I64(m->layers[0].ffn_gate_up_exps->dims[2], c->q36_n_expert,
                        "ffn_gate_up_exps expert dim == n_expert (128)");

    /* V-less global layers: at least one of the 30 layers must OMIT attn_v (the
     * model has 25/30 V tensors). Spot-check that the bridge left those NULL. */
    int v_present = 0, v_absent = 0;
    for (uint32_t L = 0; L < c->n_layers; L++) {
        if (m->layers[L].attn_v) v_present++; else v_absent++;
    }
    fprintf(stderr, "\n    attn_v: present=%d absent=%d (V-less global layers)\n", v_present, v_absent);
    SP_CHECK_EQ_I64(v_present, 25, "attn_v present on 25 layers");
    SP_CHECK_EQ_I64(v_absent,  5,  "attn_v absent (V-less) on 5 layers");

    fprintf(stderr, "\n    G-DG-N1 (N1a LOADER): DiffusionGemma loaded clean, no missing tensor.\n");
    sp_model_unload(handle);
}

int main(void) {
    SP_RUN(T_DG_N1A_LOAD);
    return SP_DONE();
}
