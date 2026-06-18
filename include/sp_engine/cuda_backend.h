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

/* ETA.1 (Stage Eta, Gemma4): structural weight-upload probe. Builds the full
 * Gemma4 CUDA weight set (per-layer global/SWA Q-KV widths, shared-KV owner
 * skips, elastic per-layer FFN, AltUp tensor set, rope_freqs) for a
 * core-bridged model (sp_model_load -> sp_model_to_gemma4) and prints the
 * resolved geometry. 0 on success. The gemma4 forward itself lands in ETA.2+. */
int gemma4_cuda_weights_probe(const qwen3_model *m);

/* ETA.2: truncatable Gemma4 CUDA prefill probe (the bisection harness).
 * n_layers=0 -> embed+scale only; attn_only=1 -> stop after layer n_layers-1's
 * attention residual; else after its FFN residual. Downloads the residual
 * stream x [n_tok x E] at the boundary. AltUp/out_scale (ETA.4) + head/softcap
 * are NOT in the probe path yet — boundaries stop before them. */
int gemma4_cuda_probe(const qwen3_model *m, const int32_t *tokens, int n_tok,
                      int n_layers, int attn_only, float *out_x);

/* ETA.4: the official Gemma4 CUDA prefill — full forward (per-layer geometry +
 * shared-KV + proportional RoPE + AltUp injection + per-layer out_scale + tied
 * head + final-logit softcap). logits = [n_tok x n_vocab], all positions.
 * Gated argmax+KL vs the CPU oracle gemma4_forward (E_G4_CU_FULL). */
int gemma4_forward_cuda(const qwen3_model *m, const int32_t *tokens, int n_tok,
                        float *logits);

/* ETA.5a: Gemma4 autoregressive CUDA decode (host-driven, oracle arithmetic).
 * Jagged per-OWNER KV cache (global 512-wide / SWA 256-wide rows; sharers
 * allocate nothing), per-step AltUp, windowed single-query attention, tied head
 * + softcap, greedy argmax. seq[0..n_prompt) in, continuations written into
 * seq[n_prompt..n_prompt+n_gen). Returns final length or -1. Gate E_G4_CU_DEC:
 * the oracle prefill must teacher-forced-predict every generated token. */
int gemma4_decode_cuda(const qwen3_model *m, int32_t *seq, int n_prompt,
                       int n_gen, int eos_id);

/* WIRE-CUDA-DECODE-GEMMA4 §3.1.A: logits-returning persistent-KV decode step
 * (the additive sibling of the KAI-1b gemma4_kv_decode). Forwards ONE token at
 * the resident cache's live position through the same g4_kv_step, then D2H-copies
 * the post-head full-vocab logits row [n_vocab] instead of the internal argmax —
 * so the universal daemon's L1 kvdecode verb owns sampling. `s` is the opaque
 * sp_g4_kv* from gemma4_kv_open (the rest of the gemma4_kv_* lifecycle ABI is
 * declared inline in tests/test_gemma4_cuda.c / forward-declared in the daemon
 * glue; only this additive symbol is surfaced in the public backend header).
 * `logits` is caller-allocated [n_vocab] f32. 0 on success; null floor decode
 * paths (gemma4_kv_decode / gemma4_decode_cuda) stay byte-untouched. */
typedef struct sp_g4_kv sp_g4_kv;
int gemma4_kv_decode_logits(sp_g4_kv *s, int32_t token, float *logits);

/* CONTRACT-CHAT-FULLSTACK B1: per-session byte-exact "auditable mode" toggle on
 * the resident KV-decode cache. on!=0 sets the device d_bx_flag (exact-integer
 * islands) AND routes the resident-decode attention through k_attn_decode_win_bx
 * (the dual-prime CRT-NTT exact-integer dot); on==0 restores the float Stage-A
 * path (byte-identical null floor). Callable per request under the cache Mutex
 * (the chat path sets on=1 at request start, on=0 at end). 0 on success. */
int gemma4_kv_byteexact_set(sp_g4_kv *s, int on);

/* CONTRACT-CHAT-FULLSTACK B2 (§6d-b): replay a stored episode's owner K/V directly
 * into the resident KV-decode cache at [dpos, dpos+npos) and advance dpos — the
 * persistent-ABI SP_REPLAY recall (C2 #222), rolling a prior memory into the live
 * turn. zero!=0 injects a ZEROED (corrupted) episode = the reject control. Reject
 * is the O(1) gemma4_kv_rewind(npos) inverse (ring-aware journal under SWA-ring).
 * `epdir` holds ep.mf/ep.k/ep.v (the xbar episode serialization). 0 on success;
 * null floor decode paths stay byte-untouched (callable only when a turn names an
 * episode). */
int gemma4_kv_replay(sp_g4_kv *s, const char *epdir, int npos, int zero);

/* Release any cached device-resident weights for model `m` (called from
 * qwen3_free when the CUDA backend is built). No-op if nothing cached. */
void sp_cuda_model_release(const qwen3_model *m);

#ifdef __cplusplus
}
#endif
#endif /* SP_ENGINE_CUDA_BACKEND_H */
