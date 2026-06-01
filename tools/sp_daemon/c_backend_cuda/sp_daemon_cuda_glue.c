/* sp_daemon_cuda_glue.c -- C glue between sp_l1.h's sp_forward_dispatch_fn ABI
 * and the engine's CUDA PTX whole-forward entry points (cuda_forward.cu:497,724).
 *
 * Sprint WIRE-CUDA: symmetric to sp_daemon_hex_glue.c. The daemon's host
 * binary, when SP_DAEMON_BACKEND=cuda is set, calls
 *   sp_session_register_forward_backend(s, NULL, sp_daemon_cuda_forward).
 * On each subsequent sp_prefill_chunk, the math-core session routes through
 * this glue (sp_l1.h:#6) which forwards to the engine's CUDA backend.
 *
 * Arch routing: the CUDA backend exports BOTH gemma3_forward_cuda (CU.1-4)
 * and qwen3_forward_cuda (CU.5) plus qwen3_forward_cuda_ex (CU.5+E_CU_6
 * KV-tree variant, not used by daemon). The glue inspects m->cfg.arch and
 * routes accordingly. This is wider than the hex glue (gemma3-only) because
 * the CUDA backend already supports both arches in ctest (M_QWEN3_CUDA
 * passes; M_GEMMA3_CUDA has a known OOM bug on this host -- see ctest log).
 *
 * Type-safety: the L1 ABI passes the qwen3_model pointer as `const void *`
 * to keep sp_l1.h independent of sp/model.h (which lives in the engine
 * include tree). Cast back here. Same arch enum (SP_ARCH_GEMMA3 /
 * SP_ARCH_QWEN3) on both math-core and engine sides.
 *
 * Error handling: gemma3_forward_cuda / qwen3_forward_cuda return 0 on
 * success, non-zero on error. sp_prefill_chunk maps non-zero to
 * SP_EBADSTATE + preserves sp_last_error via the sp_set_error calls already
 * inside the CUDA backend.
 *
 * No kernel-name shim: the CUDA backend's cuda_forward.cu calls math-core's
 * sp_matmul / sp_embed_row / sp_as_f32 directly via forward_dispatch.h --
 * the same symbols the daemon already links from libsp_forward_dispatch.a.
 * cpu_overlay.c is irrelevant on CUDA.
 *
 * Lifecycle hook: sp_cuda_model_release frees device-resident weights cached
 * by the engine's static (g_w in cuda_forward.cu:729). Called at daemon
 * shutdown via release_for_model (cuda_forward_dispatch.rs).
 */
#include "sp_engine/cuda_backend.h"   /* gemma3_forward_cuda + qwen3_forward_cuda + sp_cuda_model_release */
#include "sp/model.h"                  /* qwen3_model + sp_arch_t + SP_ARCH_GEMMA3 / SP_ARCH_QWEN3 */
#include <stdint.h>

extern void sp_set_error(const char *msg);

/* The L1 forward-dispatch ABI from sp/sp_l1.h #6. Forward-declared here so the
 * glue stays free of the math-core include path (the daemon Rust side bindgens
 * sp_l1.h directly; this C TU just needs the signature to match). */
typedef int (*sp_forward_dispatch_fn_t)(
    void *handle, const void *qm_opaque,
    const int32_t *tokens, int n_tok, float *logits);

/* Forward-dispatch entry point exposed to Rust via #[link]. The L1 contract
 * (sp_l1.h:#6) guarantees:
 *   - handle is the opaque pointer the daemon passed at register time. Ignored
 *     here -- the CUDA backend caches its device-resident weight blob via
 *     a static (g_w in cuda_forward.cu) keyed on the model pointer.
 *   - qm_opaque is the session-owned qwen3_model* (math-core's sp_session.c
 *     reconstructs it via sp_model_to_gemma3 / sp_model_to_qwen3; pointer
 *     remains valid for the session's lifetime).
 *   - tokens/n_tok/logits per the standard engine forward signature
 *     (src/forward/ppl.c:19).
 * Returns 0 on success, non-zero on error.
 *
 * Arch routing:
 *   SP_ARCH_GEMMA3 -> gemma3_forward_cuda  (CU.1-4)
 *   SP_ARCH_QWEN3  -> qwen3_forward_cuda   (CU.5)
 *   else           -> sp_set_error + return -1
 *
 * SP_ARCH_QWEN25 is not yet covered by the CUDA backend (qwen25_forward_cuda
 * does not exist in cuda_forward.cu); routing to qwen3_forward_cuda would
 * mismatch the arch check inside the kernel and fail; surface as error here. */
int sp_daemon_cuda_forward(void *handle,
                           const void *qm_opaque,
                           const int32_t *tokens,
                           int n_tok,
                           float *logits) {
    (void)handle;  /* opaque; the engine owns its own statics */
    const qwen3_model *qm = (const qwen3_model *)qm_opaque;
    if (!qm) { sp_set_error("cuda: NULL qwen3_model"); return -1; }
    switch (qm->cfg.arch) {
        case SP_ARCH_GEMMA3:
            return gemma3_forward_cuda(qm, tokens, n_tok, logits);
        case SP_ARCH_QWEN3:
            return qwen3_forward_cuda(qm, tokens, n_tok, logits);
        default:
            sp_set_error("cuda: unsupported arch (only SP_ARCH_GEMMA3 / SP_ARCH_QWEN3)");
            return -1;
    }
}

/* Lifecycle hook: when the daemon tears down its cuda backend (or the model
 * unloads), release the cached device-resident weight blob. Mirrors
 * sp_hexagon_model_release. */
void sp_daemon_cuda_release(const void *qm_opaque) {
    const qwen3_model *qm = (const qwen3_model *)qm_opaque;
    if (qm) sp_cuda_model_release(qm);
}
