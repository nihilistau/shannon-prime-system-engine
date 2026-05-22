/* model.h — Qwen3 model: config + weight binding over a loaded GGUF.
 *
 * 2-CPU.B step 1: read the architecture hyperparameters and bind every weight
 * tensor to its gguf_tensor (data stays paged in the mapping). The forward pass
 * (next) dequantizes on demand via sp_dequant_row. Qwen3-0.6B specifics:
 * GQA (head_count 16 / head_count_kv 8), head_dim 128 (so Q proj is
 * n_head*head_dim = 2048 wide while K/V are n_head_kv*head_dim = 1024),
 * per-head Q/K RMSNorm, SwiGLU FFN, untied output, RoPE base 1e6.
 */
#ifndef SP_ENGINE_MODEL_H
#define SP_ENGINE_MODEL_H

#include "sp_engine/gguf.h"
#include "sp/kste.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Model architecture. Selects the forward pass + which optional config fields and
 * per-layer tensors are populated. Default 0 = Qwen3 (calloc-zeroed configs and
 * existing callers get the right value with no churn). */
typedef enum { SP_ARCH_QWEN3 = 0, SP_ARCH_GEMMA3 = 1 } sp_arch_t;

typedef struct {
    sp_arch_t arch;           /* SP_ARCH_QWEN3 (default) | SP_ARCH_GEMMA3 */
    uint32_t n_layers;        /* {arch}.block_count                      */
    uint32_t n_embd;          /* {arch}.embedding_length                 */
    uint32_t n_ff;            /* {arch}.feed_forward_length              */
    uint32_t n_head;          /* {arch}.attention.head_count             */
    uint32_t n_head_kv;       /* {arch}.attention.head_count_kv (GQA)    */
    uint32_t head_dim;        /* {arch}.attention.key_length             */
    uint32_t n_vocab;         /* from token_embd / output rows           */
    uint32_t context_length;
    uint32_t sliding_window;  /* gemma3 local-attn window; 0 = none      */
    float    rope_freq_base;  /* {arch}.rope.freq_base (gemma3: global)  */
    float    rms_eps;         /* {arch}.attention.layer_norm_rms_epsilon */
    int      has_qk_norm;     /* per-head Q/K RMSNorm present            */
    int      tied_embedding;  /* output.weight absent -> reuse token_embd*/
} qwen3_config;

typedef struct {
    const gguf_tensor *attn_norm;     /* [n_embd]                        */
    const gguf_tensor *attn_q;        /* [n_embd, n_head*head_dim]       */
    const gguf_tensor *attn_k;        /* [n_embd, n_head_kv*head_dim]    */
    const gguf_tensor *attn_v;        /* [n_embd, n_head_kv*head_dim]    */
    const gguf_tensor *attn_output;   /* [n_head*head_dim, n_embd]       */
    const gguf_tensor *attn_q_norm;   /* [head_dim] or NULL              */
    const gguf_tensor *attn_k_norm;   /* [head_dim] or NULL              */
    const gguf_tensor *post_attn_norm;/* gemma3 post_attention_norm; NULL on qwen3 */
    const gguf_tensor *ffn_norm;      /* [n_embd]                        */
    const gguf_tensor *ffn_gate;      /* [n_embd, n_ff]                  */
    const gguf_tensor *ffn_up;        /* [n_embd, n_ff]                  */
    const gguf_tensor *ffn_down;      /* [n_ff, n_embd]                  */
    const gguf_tensor *post_ffw_norm; /* gemma3 post_ffw_norm; NULL on qwen3 */
} qwen3_layer;

struct sp_arena;   /* sp_engine/arena.h — packed-weight arena (Phase 1a) */

typedef struct qwen3_model {
    gguf_ctx          *gguf;          /* owned */
    qwen3_config       cfg;
    const gguf_tensor *token_embd;    /* [n_embd, n_vocab]               */
    const gguf_tensor *output_norm;   /* [n_embd]                        */
    const gguf_tensor *output;        /* [n_embd, n_vocab] (==token_embd if tied) */
    qwen3_layer       *layers;        /* n_layers */
    struct sp_arena   *arena;         /* packed Q8/Q4 matmul weights when SP_ARENA set; else NULL */
    /* source-release state (Phase 1b): after qwen3_release_source the GGUF data
     * mapping is unmapped; norms are served from owned f32 copies and the
     * embedding + all matmul weights from the arena. */
    int                 released;
    const gguf_tensor **norm_src;     /* [n_norm] keys (norm tensors) */
    float             **norm_buf;     /* [n_norm] owned f32 copies */
    int                 n_norm;
} qwen3_model;

/* Open the GGUF, read the qwen3 config, and bind every weight. Returns NULL on
 * a missing/inconsistent tensor or a non-qwen3 architecture. */
qwen3_model *qwen3_load(const char *path);
void         qwen3_free(qwen3_model *m);

/* Release the F16 GGUF source (Phase 1b, §4.8). Requires a packed arena that
 * includes the embedding (SP_ARENA + SP_ARENA_EMBED). Copies the norm weights to
 * owned f32, then unmaps the GGUF data — after this the forward reads only the
 * arena (matmul weights + embedding) and the owned norms, so peak memory drops
 * to the packed footprint. A tokenizer, if used afterwards, must have been loaded
 * owning (sp_tokenizer_load_ex(g,1)) before this call. Returns 0 on success.
 * qwen3_load performs this automatically when SP_ARENA_RELEASE=1. */
int          qwen3_release_source(qwen3_model *m);

/* f32 reference forward pass over a token-ID sequence (prefill, causal).
 * Writes logits for every position into `logits` (caller-allocated,
 * n_tokens * n_vocab, position-major). Returns 0 on success.
 * SP_ENGINE_BACKEND=cpu scalar reference path; correctness gate E_CPU_2. */
int qwen3_forward(const qwen3_model *m, const int32_t *tokens, int n_tokens,
                  float *logits);

/* Gemma3 f32 reference forward pass (prefill, causal). Same logits layout/return
 * as qwen3_forward. Requires a model loaded with arch == SP_ARCH_GEMMA3. The
 * Gemma deltas vs Qwen3: embedding scale (×√n_embd), sandwich norms (post-attn /
 * post-ffw on the residual branch), GeGLU FFN, local/global sliding-window attn
 * with dual RoPE base, tied LM head. Correctness gate M_GEMMA3_CPU. */
int gemma3_forward(const qwen3_model *m, const int32_t *tokens, int n_tokens,
                   float *logits);

/* As qwen3_forward, but if `kv_trees` is non-NULL it additionally KSTE-encodes
 * every cached K head-vector (the KSTE KV-cache overlay, gated in production by
 * SP_KSTE_KV=1; E_CPU_6). Each post-norm/post-RoPE K head-vector is quantized to
 * int32 and encoded to its 64-byte signature. `kv_trees` must hold
 * n_layers * n_tokens * n_head_kv entries, indexed
 * ((L*n_tokens + t)*n_head_kv + h). Pass NULL to skip (== qwen3_forward). */
int qwen3_forward_ex(const qwen3_model *m, const int32_t *tokens, int n_tokens,
                     float *logits, sp_kste_tree_t *kv_trees);

/* Greedy (argmax) autoregressive generation. `seq` holds `n_prompt` prompt token
 * IDs and must have capacity for at least n_prompt + n_gen entries; generated
 * tokens are appended in place. Stops after n_gen tokens, or earlier if `eos_id`
 * (>= 0) is produced. Returns the new total sequence length, or < 0 on error.
 *
 * Reference implementation: each step re-runs qwen3_forward over the whole prefix
 * and takes the argmax of the last position's logits. Correct by construction
 * (it reuses the E_CPU_2-validated forward) at O(n^2) cost; a persistent-KV-cache
 * decode is a later optimization to be validated for token-identity against this. */
int qwen3_generate(const qwen3_model *m, int32_t *seq, int n_prompt, int n_gen,
                   int eos_id);

/* Persistent-KV O(n) greedy decode. Same signature/semantics as qwen3_generate,
 * but maintains a position-indexed K/V cache so each token's weight matmuls run
 * once (O(n) total) instead of re-prefilling the prefix every step (O(n^2)).
 * Stores K/V post-RoPE; honors SP_ENGINE_FROB / SP_CPU_SCALAR / SP_KV_SPINOR.
 * Greedy output matches qwen3_generate up to the float-reassociation floor, so
 * GEN_KV gates on sequence (argmax) identity, not bit-equal logits. */
int qwen3_generate_kv(const qwen3_model *m, int32_t *seq, int n_prompt, int n_gen,
                      int eos_id);

/* Q4 calibration stats from the most recent forward on the Q4 weight path
 * (SP_ENGINE_FROB=3 or 4): how many weight rows the mixed-precision calibration
 * promoted to Q8, out of the total rows seen. Both 0 otherwise. (E_CPU_7.) */
void qwen3_q4_stats(long *promoted, long *rows);

/* ── dequantization (forward pass reads weights through these) ── */
float    sp_f16_to_f32(uint16_t h);
/* Round f32 to IEEE half (round-to-nearest-even). Used by the ggml-faithful
 * validation path: ggml downcasts matmul activations (src1) to F16 when the
 * weight (src0) is F16, so SP_ENGINE_F16_ACT=1 mimics that to match the oracle
 * bit-for-bit under matched precision. */
uint16_t sp_f32_to_f16(float f);
/* Dequantize `n` elements of a tensor row starting at `src` (a pointer into the
 * mapping) of the given ggml_type into `dst` (f32). Supports F32, F16, Q8_0.
 * Returns 0 on success, nonzero for an unsupported type. */
int   sp_dequant_row(const void *src, uint32_t ggml_type, int n, float *dst);

#ifdef __cplusplus
}
#endif
#endif /* SP_ENGINE_MODEL_H */
