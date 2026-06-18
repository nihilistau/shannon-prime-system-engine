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
#include "sp/forward_dispatch.h"      /* sp_matmul / sp_embed_row / sp_as_f32 / sp_kernels_read_env */
#include "sp/model.h"                 /* qwen3_model + sp_arch_t + SP_ARCH_GEMMA3 / SP_ARCH_QWEN3 + gguf_tensor */
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
        case SP_ARCH_GEMMA4:
            /* byte-exact bridge step 1 (CONTRACT-BYTEEXACT §8): route the real
             * Gemma-4-12B forward through the universal crate's register_forward_backend
             * hook. Same prefill-only contract + signature as gemma3_forward_cuda;
             * entry exported in cuda_backend.h:77. Needs the wire_cuda build to gate. */
            return gemma4_forward_cuda(qm, tokens, n_tok, logits);
        case SP_ARCH_QWEN3:
            return qwen3_forward_cuda(qm, tokens, n_tok, logits);
        default:
            sp_set_error("cuda: unsupported arch (only SP_ARCH_GEMMA3 / SP_ARCH_GEMMA4 / SP_ARCH_QWEN3)");
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

/* ── Sprint WIRE-CUDA-DECODE-GEMMA4: persistent-KV decode glue ──────────────
 *
 * The prefill dispatcher above (sp_daemon_cuda_forward) is PREFILL-ONLY: it
 * re-runs the full forward over the accumulated history and, for a 12B OK_Q4B
 * model, the tied full-vocab head is reserved for the DECODE path, so driving
 * decode through it trips `g4 probe: FULL head needs the f32 embd`
 * (cuda_forward.cu:1627). These six glue fns adapt the new
 * sp_kvdecode_dispatch_fn table (WIRE-CUDA-DECODE-GEMMA4.md §2) onto the
 * `gemma4_kv_*` C ABI already compiled into THIS lib from cuda_forward.cu
 * (declared inline in tests/test_gemma4_cuda.c:65-79; defined extern "C" at
 * cuda_forward.cu ~3478-3960). The opaque handle is an `sp_g4_kv*`.
 *
 * gemma4_kv_* is the engine null floor (byte-untouched). The ONLY additive
 * engine symbol this glue needs is the logits-returning decode step (§3.1
 * option A): gemma4_kv_decode_logits — NOT yet in cuda_forward.cu. Until it
 * lands at INTEGRATION the `step` body below is a TODO(WIRE-CUDA-DECODE) stub
 * that surfaces a clear error rather than silently arg-maxing.
 *
 * Forward-declared here (the glue stays free of the engine include path beyond
 * cuda_backend.h; these match the test-file prototypes exactly). sp_g4_kv is
 * opaque on this side — we only ever pass the pointer through. */
typedef struct sp_g4_kv sp_g4_kv;
extern sp_g4_kv *gemma4_kv_open(const qwen3_model *m, int Pmax);
extern int   gemma4_kv_prefill(sp_g4_kv *s, const int32_t *toks, int n);
extern int   gemma4_kv_rewind(sp_g4_kv *s, int delta);
extern int   gemma4_kv_pos(const sp_g4_kv *s);
extern void  gemma4_kv_close(sp_g4_kv *s);
/* TODO(WIRE-CUDA-DECODE): additive logits-returning step (addendum §3.1.A).
 * Declared here so the glue compiles against the intended symbol; the engine
 * definition is the INTEGRATION step. Until then `step` returns an error.
 *   extern int gemma4_kv_decode_logits(sp_g4_kv *s, int32_t token, float *logits); */

/* open(qm, pmax) -> sp_g4_kv* (as void*). NULL on failure (sp_last_error set by
 * gemma4_kv_open). pmax = max resident position budget. Requires the model to
 * be SP_ARCH_GEMMA4 and SP_CUDA_DECODE_INT8=1 in the env (tied head). */
void *sp_daemon_cuda_kvdecode_open(const void *qm_opaque, int pmax) {
    const qwen3_model *qm = (const qwen3_model *)qm_opaque;
    if (!qm) { sp_set_error("cuda kvdecode open: NULL qwen3_model"); return 0; }
    if (qm->cfg.arch != SP_ARCH_GEMMA4) {
        sp_set_error("cuda kvdecode open: persistent-KV decode is SP_ARCH_GEMMA4 only");
        return 0;
    }
    return (void *)gemma4_kv_open(qm, pmax);
}

/* prefill(handle, tokens, n): ingest history into the resident cache. 0 ok. */
int sp_daemon_cuda_kvdecode_prefill(void *handle, const int32_t *tokens, int n_tok) {
    sp_g4_kv *s = (sp_g4_kv *)handle;
    if (!s || !tokens || n_tok <= 0) { sp_set_error("cuda kvdecode prefill: bad args"); return -1; }
    return gemma4_kv_prefill(s, tokens, n_tok);
}

/* decode_step(handle, token, logits): forward ONE token at the live dpos, write
 * the full-vocab logits row [n_vocab] for the NEXT position, advance dpos.
 *
 * TODO(WIRE-CUDA-DECODE): wire to the additive gemma4_kv_decode_logits once it
 * exists in cuda_forward.cu (addendum §3.1 option A + §7 step 1):
 *   return gemma4_kv_decode_logits(s, token, logits);
 * Until then surface a clear, non-silent error (the gate stays RED, by design —
 * no fake greedy path that would lose L2-side sampling). */
int sp_daemon_cuda_kvdecode_step(void *handle, int32_t token, float *logits) {
    sp_g4_kv *s = (sp_g4_kv *)handle;
    if (!s || !logits) { sp_set_error("cuda kvdecode step: bad args"); return -1; }
    (void)token;
    sp_set_error("cuda kvdecode step: TODO(WIRE-CUDA-DECODE) — needs additive "
                 "gemma4_kv_decode_logits (WIRE-CUDA-DECODE-GEMMA4.md §3.1.A/§7.1)");
    return -1;
}

/* rewind(handle, n): O(1) cold-evict (dpos -= n). 0 ok. */
int sp_daemon_cuda_kvdecode_rewind(void *handle, int n) {
    sp_g4_kv *s = (sp_g4_kv *)handle;
    if (!s || n < 0) { sp_set_error("cuda kvdecode rewind: bad args"); return -1; }
    return gemma4_kv_rewind(s, n);
}

/* position(handle): current dpos, -1 on NULL. */
int sp_daemon_cuda_kvdecode_position(const void *handle) {
    const sp_g4_kv *s = (const sp_g4_kv *)handle;
    return gemma4_kv_pos(s);
}

/* close(handle): free the resident cache. NULL-safe. */
void sp_daemon_cuda_kvdecode_close(void *handle) {
    sp_g4_kv *s = (sp_g4_kv *)handle;
    if (s) gemma4_kv_close(s);
}

/* ── Engine-kernel shim over math-core forward_dispatch ────────────────────
 *
 * cuda_forward.cu reaches for `sp_engine/kernels.h` entry points by unprefixed
 * names (as_f32, and indirectly matmul/embed_row via the engine's host-side
 * weight-upload path). In the daemon-link build we DO NOT link cpu_overlay.c
 * (it would duplicate sp_kernels_read_env + qwen3_q4_stats already present in
 * math-core's libsp_forward_dispatch). Instead, expose the engine names as
 * thin aliases over math-core's sp_* variants. Same TU as the L1 dispatcher
 * so the link graph is one extra .o, matching the WIRE-HEX glue shape. */

int matmul(const qwen3_model *m, const gguf_tensor *W,
           const float *X, int n_tok, int in, int out, float *Y) {
    return sp_matmul(m, W, X, n_tok, in, out, Y);
}

int embed_row(const qwen3_model *m, int32_t tok, int E, float *dst) {
    return sp_embed_row(m, tok, E, dst);
}

const float *as_f32(const qwen3_model *m, const gguf_tensor *t) {
    return sp_as_f32(m, t);
}

/* sp_kernels_read_env: math-core's forward_dispatch.c already exports this
 * name; cuda_forward.cu's call site resolves to the math-core symbol directly,
 * no shim entry needed here. Mirrors the WIRE-HEX glue (sp_daemon_hex_glue.c). */
