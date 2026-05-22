# Gemma3-1B CPU bring-up — SP1: loader + forward pass + distributional gate

**Date:** 2026-05-22
**Phase:** 2-CPU (engine), second model architecture. Roadmap T_FRO_4 / §9 (CPU × Gemma 3 cell).
**Status:** design approved; spec for SP1 (the first of three sub-projects).

---

## Why this work / decomposition

Adding Gemma3-1B as the second CPU architecture, CPU-first so the validated forward
pass becomes the reference the CUDA/Vulkan/Hexagon ports mirror. The full goal
(arch + PPL + SentencePiece) is genuinely 3 independent subsystems, sequenced by
dependency; each gets its own spec → plan → build cycle:

- **SP1 (this spec)** — Gemma3 loader + forward pass + distributional correctness gate,
  validated against the llama.cpp oracle on **oracle-provided token IDs** (no tokenizer
  needed). All architectural correctness lives here.
- **SP2** — PPL gate (T_FRO_4: within 0.1% of the fp16 reference), on an oracle-tokenized
  corpus. Small once SP1 is correct.
- **SP3** — SentencePiece unigram tokenizer (encode via Viterbi over the 262144-piece
  model + byte fallback; decode). Independent of SP1/SP2 *correctness* (they use oracle
  IDs); needed only for end-to-end text. Comparable in size to the GPT2-BPE effort.

This spec covers **SP1 only**.

## Confirmed Gemma3-1B configuration (from the GGUF, arch=`gemma3`)

| field | value |
|---|---|
| block_count | 26 |
| embedding_length (n_embd) | 1152 |
| feed_forward_length | 6912 |
| attention.head_count | 4 |
| attention.head_count_kv | 1 (GQA 4:1) |
| attention.key_length / value_length (head_dim) | 256 (Q proj→1024, K/V proj→256) |
| attention.sliding_window | 512 |
| rope.freq_base | 1 000 000 (global layers) |
| rms_eps | 1e-6 |
| context_length | 32768 |
| vocab (token_embd rows) | 262144 |
| LM head | **tied** (no `output.weight`; LM head = `token_embd`) |
| tokenizer | SentencePiece (`tokenizer.ggml.model=llama`, unigram, 0 merges), bos=2 eos=1, add_bos=1 |

Per-layer tensors (`blk.N.*`): `attn_norm`, `attn_q`/`attn_k`/`attn_v`, `attn_q_norm`[256],
`attn_k_norm`[256], `attn_output`, `post_attention_norm`, `ffn_norm`, `ffn_gate`/`ffn_up`/
`ffn_down`, `post_ffw_norm`. Top level: `token_embd`, `output_norm`.

## Architecture deltas vs the validated Qwen3 forward pass

1. **Embedding scale:** `x = embed(tok) · sqrt(n_embd)` (= √1152).
2. **Gemma RMSNorm:** `out[i] = x[i]/rms(x) · (1 + w[i])` — the `(1 + weight)` offset (Qwen3
   uses `w[i]`). Used by all 6 norm sites (attn_norm, post_attention_norm, ffn_norm,
   post_ffw_norm, attn_q_norm, attn_k_norm, output_norm).
3. **Sandwich norms (4 per layer), residual adds AFTER the post-norms:**
   - `x = x + post_attention_norm( attn_output · Attn( … attn_norm(x) … ) )`
   - `x = x + post_ffw_norm( GeGLU( ffn_norm(x) ) )`
4. **QK-norm:** per-head RMSNorm (Gemma form) over head_dim=256 on Q and K, before RoPE
   (same shape as Qwen3, so the same precision-floor reasoning applies — see Gate).
5. **GeGLU:** `ffn_down( gelu(ffn_gate·h) ⊙ (ffn_up·h) )`, gelu-tanh approximation.
6. **Local/global attention:** local layers use sliding-window-512 masking + RoPE θ≈10000;
   global layers (≈ every 6th) use full causal attention + RoPE θ=1e6. Exact pattern,
   local θ, and query scale are pinned via the oracle (below).
7. **Query scale:** `1/sqrt(head_dim)` = `1/sqrt(256)` (no `query_pre_attn_scalar` key ⇒
   defaults to head_dim).
8. **Tied LM head:** logits = `token_embd · final_norm(x)`.

## Module structure

Extract the backend-agnostic kernels currently `static` in `src/forward/forward.c` into a
shared **`src/forward/kernels.{h,c}`** with identical signatures and behavior:
`dot_f32` (+ the AVX2/scalar gate), `rmsnorm`, `rmsnorm_head`, `rope_neox`, `matmul` /
`matmul_arena`, and the GQA causal-attention core. `forward.c` (Qwen3) includes the header
and is otherwise unchanged. A new **`src/forward/gemma3.c`** builds the Gemma3 layer loop on
those kernels, adding the Gemma-specific pieces: `rmsnorm_gemma` (the `1+w` variant),
`gelu_tanh`, the embedding scale, the sandwich-norm wiring, and the local/global +
sliding-window attention selection.

- The **kernel extraction is behavior-preserving**; the existing Qwen3 regression
  (E_CPU_2/3/4/5/6, GEN_*, E_CPU_7..10, COMPOSE — 16/16) is the guard.
- Gemma3 reuses the loader (`gguf.c`) and the model layer (`model.c`) extended for
  `arch=gemma3` (config parse + tensor binding; tied head; head_dim from key_length).
- Gemma3 reuses the arena (`SP_FROB_ARENA_LAYOUT_VERSION`) and the Spinor KV codec
  unchanged — they are arch-independent.

## Loader / config (model.c, gguf.c)

- Detect `general.architecture == "gemma3"`; parse the gemma3.* hparams above.
- Bind the per-layer tensors incl. the 4 norms + QK-norm; head_dim from `key_length`
  (NOT n_embd/n_head — they differ: 1152/4=288 ≠ 256).
- Tied head: `output` tensor = `token_embd` when `output.weight` is absent.
- A `GEMMA3_BIND` test mirrors Qwen3's `MODEL_BIND`: config matches the known arch,
  all 26 layers bound, shapes consistent, tied head detected, embedding dequant finite.

## Oracle (resolves the ambiguities empirically)

Extend the existing `tools/oracle/` harness (links the clean llama.cpp at
`shannon-prime-lattice-llama`) to Gemma3:
- `dump_logits` already arch-agnostic — produce `gemma3_ref.bin` (token IDs + per-position
  logits) for a fixed prompt; f32 KV + flash-attn off (apples-to-apples).
- `dump_layers` extended to Gemma3 checkpoints (attn_norm out, Qcur/Kcur post-QKnorm,
  post-RoPE, kqv, post_attention_norm out, ffn checkpoints, result_norm) so each delta
  is verified in isolation, exactly as E_CPU_2 was.
- These pin: (a) the global-layer pattern, (b) local RoPE θ, (c) query scale, (d) the gelu
  variant. For a **short (<512-token) prompt** the sliding window does not bite, so local
  vs global differ only by RoPE θ — the simplest first target. The window only matters for
  SP2's longer PPL corpus.

## Validation gate (SP1 deliverable)

Distributional, per `reference-ecpu2-qknorm-precision-gate`: QK-norm + the sandwich norms
over 26 layers amplify the F16-matmul precision floor, so a per-logit bit-match vs ggml is
not achievable. The gate, on the **pure-f32** engine path vs the oracle `gemma3_ref.bin`:
- argmax match (top-1) over all positions,
- top-5 set cross-match,
- mean `KL(ggml ‖ engine) < 1e-5` nats (same bar as E_CPU_2; threshold overridable via
  `SP_KL_MAX`).

Test name **`M_GEMMA3_CPU`** (the SP1 gate; aligns with roadmap M_<family>_<backend>_2).
Engine gate knobs stay default-OFF (pure-f32 reference), preserving the regression invariant.

## Out of scope for SP1

- SentencePiece tokenizer (SP3) — SP1 reads oracle token IDs from `gemma3_ref.bin`.
- PPL loop (SP2).
- Quantized Gemma3 paths (Q8/Q4 arena, Spinor KV) on Gemma3 — they are arch-independent and
  validated on Qwen3; a Gemma3 composability pass can follow once the f32 forward is correct.
- The other backends (CUDA/Vulkan/Hexagon) — they mirror this CPU reference later.

## Test/build commands

```
scripts\build\build-cpu.bat
ctest --test-dir build-cpu -R "GEMMA3|M_GEMMA3" --output-on-failure
ctest --test-dir build-cpu --output-on-failure   # full regression incl. Qwen3 guard
```

## Success criteria

1. `GEMMA3_BIND` green (loader/config correct).
2. `M_GEMMA3_CPU` green (forward pass distributionally matches the oracle).
3. Full Qwen3 regression still 16/16 (kernel extraction behavior-preserving).
4. MSVC `/W4` clean.
