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

typedef struct {
    uint32_t n_layers;        /* qwen3.block_count                       */
    uint32_t n_embd;          /* qwen3.embedding_length                  */
    uint32_t n_ff;            /* qwen3.feed_forward_length               */
    uint32_t n_head;          /* qwen3.attention.head_count              */
    uint32_t n_head_kv;       /* qwen3.attention.head_count_kv (GQA)     */
    uint32_t head_dim;        /* qwen3.attention.key_length              */
    uint32_t n_vocab;         /* from token_embd / output rows           */
    uint32_t context_length;
    float    rope_freq_base;  /* qwen3.rope.freq_base                    */
    float    rms_eps;         /* qwen3.attention.layer_norm_rms_epsilon  */
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
    const gguf_tensor *ffn_norm;      /* [n_embd]                        */
    const gguf_tensor *ffn_gate;      /* [n_embd, n_ff]                  */
    const gguf_tensor *ffn_up;        /* [n_embd, n_ff]                  */
    const gguf_tensor *ffn_down;      /* [n_ff, n_embd]                  */
} qwen3_layer;

typedef struct {
    gguf_ctx          *gguf;          /* owned */
    qwen3_config       cfg;
    const gguf_tensor *token_embd;    /* [n_embd, n_vocab]               */
    const gguf_tensor *output_norm;   /* [n_embd]                        */
    const gguf_tensor *output;        /* [n_embd, n_vocab] (==token_embd if tied) */
    qwen3_layer       *layers;        /* n_layers */
} qwen3_model;

/* Open the GGUF, read the qwen3 config, and bind every weight. Returns NULL on
 * a missing/inconsistent tensor or a non-qwen3 architecture. */
qwen3_model *qwen3_load(const char *path);
void         qwen3_free(qwen3_model *m);

/* f32 reference forward pass over a token-ID sequence (prefill, causal).
 * Writes logits for every position into `logits` (caller-allocated,
 * n_tokens * n_vocab, position-major). Returns 0 on success.
 * SP_ENGINE_BACKEND=cpu scalar reference path; correctness gate E_CPU_2. */
int qwen3_forward(const qwen3_model *m, const int32_t *tokens, int n_tokens,
                  float *logits);

/* As qwen3_forward, but if `kv_trees` is non-NULL it additionally KSTE-encodes
 * every cached K head-vector (the KSTE KV-cache overlay, gated in production by
 * SP_KSTE_KV=1; E_CPU_6). Each post-norm/post-RoPE K head-vector is quantized to
 * int32 and encoded to its 64-byte signature. `kv_trees` must hold
 * n_layers * n_tokens * n_head_kv entries, indexed
 * ((L*n_tokens + t)*n_head_kv + h). Pass NULL to skip (== qwen3_forward). */
int qwen3_forward_ex(const qwen3_model *m, const int32_t *tokens, int n_tokens,
                     float *logits, sp_kste_tree_t *kv_trees);

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
