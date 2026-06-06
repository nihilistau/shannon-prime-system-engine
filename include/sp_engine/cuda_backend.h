/* cuda_backend.h — C-ABI surface of the CUDA backend (Phase 2-CU).
 *
 * Built only when SP_ENGINE_WITH_CUDA. The forward entry point mirrors the CPU
 * gemma3_forward exactly (same signature/logits layout); a CUDA forward_fn is
 * selected when SP_BACKEND=cuda. Every cudaError_t / cublasStatus_t that escapes
 * a call here is wrapped to SP_ECUDA with the detail in sp_last_error() (the
 * frozen L1 ABI error contract, sp_status.h).
 */
#ifndef SP_ENGINE_CUDA_BACKEND_H
#define SP_ENGINE_CUDA_BACKEND_H

#include "sp_engine/sp_status.h"
#include "sp_engine/model.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Number of CUDA devices visible (0 on none or driver error; detail in
 * sp_last_error on error). Cheap toolchain/link smoke. */
int sp_cuda_device_count(void);

/* Device `dev` info: copies the name into `name` (capacity name_cap, always
 * NUL-terminated when name_cap>0) and writes the compute capability to
 * *sm_major/*sm_minor (when non-NULL). Returns SP_OK or SP_ECUDA. */
sp_status sp_cuda_device_info(int dev, char *name, int name_cap,
                              int *sm_major, int *sm_minor);

/* Gemma3 f32 forward on CUDA. Same contract as the CPU gemma3_forward: prefill,
 * causal; writes logits[n_tokens * n_vocab] position-major. Requires a model
 * loaded with arch == SP_ARCH_GEMMA3. Honors the packed-weight arena (Q8/Q4)
 * when m->arena is set, else the GGUF f32/f16 path. Returns 0 on success,
 * nonzero on error (sp_last_error has CUDA detail). [CU.1] */
int gemma3_forward_cuda(const qwen3_model *m, const int32_t *tokens, int n_tokens,
                        float *logits);

/* Qwen3 f32/Q8/Q4 forward on CUDA. Same contract as the CPU qwen3_forward:
 * prefill, causal; logits[n_tokens * n_vocab] position-major. Requires arch ==
 * SP_ARCH_QWEN3. Honors the packed-weight arena when m->arena is set. [CU.5] */
int qwen3_forward_cuda(const qwen3_model *m, const int32_t *tokens, int n_tokens,
                       float *logits);

/* As qwen3_forward_cuda, but if kv_trees is non-NULL each post-norm/post-RoPE K
 * head-vector is KSTE-encoded to its 64-byte signature (E_CU_6) via the host
 * sp_kste_encode. kv_trees holds n_layers * n_tokens * n_kv_heads entries,
 * indexed ((L*n_tokens + t)*n_kv_heads + h). Pass NULL to skip (== forward). */
int qwen3_forward_cuda_ex(const qwen3_model *m, const int32_t *tokens, int n_tokens,
                          float *logits, sp_kste_tree_t *kv_trees);

/* Autoregressive KV-cache decode on the GPU (Beta). seq[0..n_prompt) is the
 * prompt; greedy-argmax continuations are written into seq[n_prompt..n_prompt+
 * n_gen). KV stays resident in VRAM across steps (one token/step, single-query
 * attention over the cached span). Returns final length (n_prompt+produced) or
 * -1. argmax sequence matches the CPU qwen3_generate_kv (knobs off). */
int qwen3_decode_cuda(const qwen3_model *m, int32_t *seq, int n_prompt,
                      int n_gen, int eos_id);

/* Release any cached device-resident weights for model `m` (called from
 * qwen3_free when the CUDA backend is built). No-op if nothing cached. */
void sp_cuda_model_release(const qwen3_model *m);

#ifdef __cplusplus
}
#endif
#endif /* SP_ENGINE_CUDA_BACKEND_H */
