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

#include <stdio.h>
#include <stdlib.h>

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

    sp_cuda_model_release(m);
    qwen3_free(m);
    sp_model_unload(handle);
}

int main(void) {
    SP_RUN(T_GEMMA4_CUDA_WEIGHTS);
    return SP_DONE();
}
