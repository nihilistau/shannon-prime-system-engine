---
type: design
title: Gemma3-1B CPU Forward Pass (SP1) Implementation Plan
description: "Goal: Bring up the Gemma3-1B forward pass on the CPU backend and prove it distributionally matches the llama.cpp oracle (argmax + top-5 + mean KL < 1e-5), reusing the existing engine kernels."
tags: [design]
timestamp: 2026-05-22T03:08:40Z
resource: ./docs/superpowers/plans/2026-05-22-gemma3-cpu-forward.md
sp_status: ACTIVE
sp_gate: none
sp_commit: TBD
sp_repro: none
---

# Gemma3-1B CPU Forward Pass (SP1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring up the Gemma3-1B forward pass on the CPU backend and prove it distributionally matches the llama.cpp oracle (argmax + top-5 + mean KL < 1e-5), reusing the existing engine kernels.

**Architecture:** Extract the backend-agnostic kernels currently `static` in `src/forward/forward.c` into a shared `src/forward/kernels.{h,c}` (behavior-preserving; the Qwen3 16/16 regression is the guard). Add `src/forward/gemma3.c` implementing the Gemma3 layer loop on those kernels with the Gemma deltas (embedding scale, `(1+w)` RMSNorm, sandwich norms, GeGLU, QK-norm, local/global sliding-window attention, dual RoPE θ, tied LM head). The few empirical constants (global-layer pattern, local RoPE θ, query scale, gelu variant) are pinned by extending the oracle harness and matching per-layer checkpoints — the same methodology that closed E_CPU_2.

**Tech Stack:** C11, MSVC (VS2019 BT) + Ninja, AVX2/AVX512; math-core submodule (`sp_frobenius`, `sp_vht2`); the clean llama.cpp oracle at `D:\F\shannon-prime-repos\shannon-prime-lattice-llama`; `sp/sp_test.h` harness.

**Reference spec:** `docs/superpowers/specs/2026-05-22-gemma3-cpu-forward-design.md`.

---

## File structure

| File | Responsibility |
|---|---|
| `src/forward/kernels.h` (create) | Declarations for the shared backend-agnostic kernels + the env-knob accessor. |
| `src/forward/kernels.c` (create) | Definitions moved verbatim from `forward.c`: the env-knob state + `read_env_knobs`, `dot_f32`, `rmsnorm`, `rmsnorm_head`, `rope_neox`, `matmul`/`matmul_arena`, `row_bytes`, `as_f32`, `embed_row`, the GQA attention helper. |
| `src/forward/forward.c` (modify) | Qwen3 forward; now `#include "kernels.h"`, drops the moved statics. Behavior unchanged. |
| `src/forward/gemma3.c` (create) | Gemma3 layer loop + Gemma-specific helpers (`rmsnorm_gemma`, `gelu_tanh`) + `gemma3_forward`. |
| `include/sp_engine/model.h` (modify) | `arch` enum on `qwen3_config`/model; Gemma3 hparams; `gemma3_forward` decl. |
| `src/forward/model.c` (modify) | Detect `arch=gemma3`, parse gemma3.* hparams, bind the 26-layer tensors incl. sandwich + QK norms, tied head. |
| `tests/test_gemma3_bind.c` (create) | `GEMMA3_BIND` — config + binding correctness. |
| `tests/test_gemma3_forward.c` (create) | `M_GEMMA3_CPU` — distributional gate vs `gemma3_ref.bin`. |
| `tools/oracle/dump_logits.cpp` (reuse) | Produce `gemma3_ref.bin` (arch-agnostic already). |
| `tools/oracle/dump_layers.cpp` (modify) | Add Gemma3 per-layer checkpoints. |
| `tests/CMakeLists.txt` (modify) | Register `GEMMA3_BIND`, `M_GEMMA3_CPU`; add `SP_GEMMA3_GGUF`/`SP_GEMMA3_REF` cache vars. |

---

## Task 1: Extract shared kernels (behavior-preserving)

**Files:**
- Create: `src/forward/kernels.h`, `src/forward/kernels.c`
- Modify: `src/forward/forward.c` (remove the moved statics, add `#include "sp_engine/kernels.h"` — or a local `"kernels.h"`)
- Modify: `src/CMakeLists.txt` (add `forward/kernels.c` to the `sp_engine` sources)

The goal is a pure move: every function keeps its exact signature and body. `forward.c`'s
`matmul`/`dot_f32` depend on file-scope env-knob globals (`g_scalar`, `g_f16_act`, `g_frob`,
`g_q4_promote`, `g_ntt_attn`, `g_kv_spinor`, `g_kv_spinor_ref`) and `read_env_knobs()`; these
move to `kernels.c` so both `forward.c` and `gemma3.c` share one copy. The KV-spinor helpers
(`kv_spinor_roundtrip`) and the format-lock `_Static_assert`s stay in `forward.c` (Qwen3/KV
specific) unless `gemma3.c` needs them later.

- [ ] **Step 1: Identify the exact extraction set.** From `src/forward/forward.c`, the
  backend-agnostic kernels to move: the env-knob globals + `read_env_knobs`, `dot_f32`,
  `rmsnorm`, `rmsnorm_head`, `rope_neox`, `row_bytes`, `as_f32`, `sp_dequant_row` wrapper
  usage, `matmul`, `matmul_arena`, `embed_row`. The GQA softmax-attention is currently
  inlined in `qwen3_forward_ex`; extract it into a reusable
  `kernels_gqa_attention(q, kc, vc, ...)` helper (see Step 4) so both arches call it.
- [ ] **Step 2: Create `src/forward/kernels.h`** declaring the moved functions + the env
  accessor. Match the existing signatures exactly. Include guards `SP_ENGINE_KERNELS_H`.
  Declare `void sp_read_env_knobs(void);` and getters only if a caller outside kernels.c
  needs a knob (gemma3.c needs the scalar/frob behavior implicitly through `matmul`, so no
  getters required initially).
- [ ] **Step 3: Create `src/forward/kernels.c`** with the moved bodies verbatim (cut from
  forward.c). Keep the `#if defined(SP_ENGINE_AVX2)` include of `<immintrin.h>` and the AVX
  paths. Move the `read_env_knobs` rename to `sp_read_env_knobs` (update both call sites in
  forward.c).
- [ ] **Step 4: Extract the GQA attention core.** Define in kernels.c:

```c
/* Causal GQA softmax attention for one query head h over cached K/V at positions
 * [0, pos], reading KC/VC laid out [s*KVD + kvh*HD]. `win` < 0 means full causal;
 * win >= 0 masks to a sliding window of that size (s >= pos-win+1). Writes HD floats
 * to `out`. `sc` is caller scratch of length >= pos+1. */
void kernels_attn_head(const float *qh, const float *KC, const float *VC,
                       int pos, int KVD, int kvh, int HD, float ascale, int win,
                       float *sc, float *out);
```

  Body = the existing per-head loop from `qwen3_forward_ex` / `qwen3_generate_kv` (max-shift
  softmax, dot, weighted V-sum), plus a window guard `int s0 = (win >= 0 && pos - win + 1 > 0)
  ? pos - win + 1 : 0;` and loop `for (s = s0; s <= pos; s++)`. Refactor the Qwen3 forward to
  call it with `win = -1` (full causal) — this is the behavior-preserving change the Qwen3
  regression validates.
- [ ] **Step 5: Modify `forward.c`** to `#include "sp_engine/kernels.h"`, delete the moved
  bodies, and call `kernels_attn_head(..., /*win=*/-1, ...)` in the attention loops.
- [ ] **Step 6: Build.**

Run: `scripts\build\build-cpu.bat`
Expected: `BUILD_EXIT=0`, MSVC `/W4` clean, no unresolved symbols.

- [ ] **Step 7: Full Qwen3 regression (the behavior-preserving guard).**

Run: `ctest --test-dir build-cpu --output-on-failure`
Expected: `100% tests passed ... 16` — identical to pre-extraction. If any of
E_CPU_2/4/5/GEN_*/COMPOSE drift, the extraction changed behavior; fix before continuing.

- [ ] **Step 8: Commit.**

```bash
git add src/forward/kernels.h src/forward/kernels.c src/forward/forward.c src/CMakeLists.txt
git commit -m "[lat-2-CPU] Extract shared forward kernels into kernels.{h,c} (Qwen3 16/16 unchanged)"
```

---

## Task 2: Gemma3 config + loader + GEMMA3_BIND

**Files:**
- Modify: `include/sp_engine/model.h` (arch tag + Gemma3 hparams + `gemma3_forward` decl)
- Modify: `src/forward/model.c` (gemma3 detect/parse/bind; tied head)
- Create: `tests/test_gemma3_bind.c`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Write the failing test `tests/test_gemma3_bind.c`.** Mirror `test_model.c`.
  Assert the known Gemma3-1B config and binding:

```c
#define _CRT_SECURE_NO_WARNINGS
#include "sp/sp_test.h"
#include "sp_engine/model.h"
#include <stdio.h>
#ifndef SP_GEMMA3_GGUF
#define SP_GEMMA3_GGUF "gemma-3-1b-it-f16.gguf"
#endif
static void GEMMA3_BIND(void) {
    qwen3_model *m = qwen3_load(SP_GEMMA3_GGUF);   /* shared loader, arch-dispatched */
    SP_CHECK(m != NULL, "gemma3 load");
    if (!m) return;
    const qwen3_config *c = &m->cfg;
    SP_CHECK(c->arch == SP_ARCH_GEMMA3,        "arch == gemma3");
    SP_CHECK_EQ_I64(c->n_layers, 26,           "26 layers");
    SP_CHECK_EQ_I64(c->n_embd, 1152,           "n_embd 1152");
    SP_CHECK_EQ_I64(c->n_ff, 6912,             "n_ff 6912");
    SP_CHECK_EQ_I64(c->n_head, 4,              "4 q heads");
    SP_CHECK_EQ_I64(c->n_head_kv, 1,           "1 kv head");
    SP_CHECK_EQ_I64(c->head_dim, 256,          "head_dim 256");
    SP_CHECK_EQ_I64(c->n_vocab, 262144,        "vocab 262144");
    SP_CHECK_EQ_I64(c->sliding_window, 512,    "sliding window 512");
    SP_CHECK(c->tied_embeddings,               "tied LM head");
    SP_CHECK(m->output == m->token_embd,       "output aliases token_embd");
    int bound = 1;
    for (uint32_t i = 0; i < c->n_layers; i++) {
        const qwen3_layer *L = &m->layers[i];
        if (!L->attn_norm || !L->attn_q || !L->attn_k || !L->attn_v ||
            !L->attn_q_norm || !L->attn_k_norm || !L->attn_output ||
            !L->post_attn_norm || !L->ffn_norm || !L->ffn_gate ||
            !L->ffn_up || !L->ffn_down || !L->post_ffw_norm) { bound = 0; break; }
    }
    SP_CHECK(bound, "all 26 layers fully bound (incl. sandwich + QK norms)");
    qwen3_free(m);
}
int main(void) { SP_RUN(GEMMA3_BIND); return SP_DONE(); }
```

- [ ] **Step 2: Add the config/struct fields in `include/sp_engine/model.h`.** Add an arch
  enum and the new fields (additive; Qwen3 leaves them defaulted):

```c
typedef enum { SP_ARCH_QWEN3 = 0, SP_ARCH_GEMMA3 = 1 } sp_arch_t;
/* in qwen3_config: */
sp_arch_t arch;
uint32_t  sliding_window;   /* gemma3 local-attn window; 0 = none */
int       tied_embeddings;  /* 1 if LM head aliases token_embd */
/* in qwen3_layer: */
const gguf_tensor *post_attn_norm;   /* gemma3 post_attention_norm */
const gguf_tensor *post_ffw_norm;    /* gemma3 post_ffw_norm */
```

  Declare `int gemma3_forward(const qwen3_model *m, const int32_t *tokens, int n_tok, float *logits);`.
- [ ] **Step 3: Extend `src/forward/model.c` `qwen3_load`** to dispatch on
  `general.architecture`. For `"gemma3"`: set `arch=SP_ARCH_GEMMA3`; read `gemma3.block_count`,
  `gemma3.embedding_length`, `gemma3.feed_forward_length`, `gemma3.attention.head_count`,
  `.head_count_kv`, `.key_length` (→head_dim), `.sliding_window`, `gemma3.rope.freq_base`,
  `gemma3.attention.layer_norm_rms_epsilon`, vocab from `token_embd` rows. Bind per-layer
  `blk.N.{attn_norm,attn_q,attn_k,attn_v,attn_q_norm,attn_k_norm,attn_output,
  post_attention_norm,ffn_norm,ffn_gate,ffn_up,ffn_down,post_ffw_norm}` and top-level
  `token_embd`, `output_norm`. Tied head: if `output.weight` absent set `tied_embeddings=1`,
  `m->output = m->token_embd`.
- [ ] **Step 4: Register the test in `tests/CMakeLists.txt`:**

```cmake
set(SP_GEMMA3_GGUF
    "D:/Files/Models/Mine/gemma-3-1b-it/gemma-3-1b-it-f16/gemma-3-1b-it-f16.gguf"
    CACHE FILEPATH "Gemma3-1B f16 GGUF for the engine tests")
add_executable(test_gemma3_bind test_gemma3_bind.c)
target_link_libraries(test_gemma3_bind PRIVATE sp_engine)
target_compile_definitions(test_gemma3_bind PRIVATE SP_GEMMA3_GGUF="${SP_GEMMA3_GGUF}")
if(MSVC)
  target_compile_options(test_gemma3_bind PRIVATE /W4)
else()
  target_compile_options(test_gemma3_bind PRIVATE -Wall -Wextra)
endif()
add_test(NAME GEMMA3_BIND COMMAND test_gemma3_bind)
```

- [ ] **Step 5: Build + run GEMMA3_BIND.**

Run: `scripts\build\build-cpu.bat && ctest --test-dir build-cpu -R GEMMA3_BIND --output-on-failure`
Expected: PASS (all config + binding checks).

- [ ] **Step 6: Commit.**

```bash
git add include/sp_engine/model.h src/forward/model.c tests/test_gemma3_bind.c tests/CMakeLists.txt
git commit -m "[lat-2-CPU] Gemma3 config parse + loader bind + GEMMA3_BIND (tied head, sandwich+QK norms)"
```

---

## Task 3: Oracle — gemma3_ref.bin + per-layer checkpoints

**Files:**
- Reuse: `tools/oracle/dump_logits.{cpp,sh}` (arch-agnostic) → produce `gemma3_ref.bin`
- Modify: `tools/oracle/dump_layers.cpp` (add gemma3 checkpoints)

The oracle is the ground truth for the empirical constants (global-layer pattern, local RoPE θ,
query scale, gelu variant). Build it against the clean llama.cpp at
`D:\F\shannon-prime-repos\shannon-prime-lattice-llama` (gcc/Ninja, its own repo — not committed
here).

- [ ] **Step 1: Produce `gemma3_ref.bin`.** Run the existing `dump_logits` against the Gemma3
  GGUF on a short fixed prompt (< 512 tokens so the sliding window does not bite — local vs
  global then differ only by RoPE θ). Place it at the `SP_GEMMA3_REF` path used by Task 4.

Run: `tools/oracle/dump_logits.sh <gemma3.gguf> @prompt.txt gemma3_ref.bin`
Expected: a header (magic, n_tok, n_vocab=262144) + token IDs + per-position logits.

- [ ] **Step 2: Extend `dump_layers.cpp`** to register Gemma3 checkpoint tensors via the sched
  eval callback: `attn_norm` out, `Qcur`/`Kcur` post-QK-norm, post-RoPE, `kqv`,
  `post_attention_norm` out, `ffn_norm` out, GeGLU intermediate, `post_ffw_norm` out,
  `result_norm`. These let each Gemma delta be verified in isolation in Task 4.
- [ ] **Step 3: Capture the empirical constants.** From the oracle (its config + the layer
  checkpoints) record, into a short note in the spec or a comment in `gemma3.c`:
  (a) which layer indices use full attention (the global pattern; expected `il % 6 == 5`),
  (b) the local RoPE θ (expected 10000) vs global (1e6),
  (c) the query pre-attention scale (expected `1/sqrt(256)`),
  (d) the gelu variant (expected tanh approximation).
- [ ] **Step 4: Commit the oracle changes (engine repo) + the fixture.**

```bash
git add tools/oracle/dump_layers.cpp
git commit -m "[lat-2-CPU] Oracle: Gemma3 per-layer checkpoints for the forward bring-up"
```
(`gemma3_ref.bin` lives at the model dir / `SP_GEMMA3_REF`, like `qwen3_ref.bin` — not committed.)

---

## Task 4: Gemma3 forward pass + helpers

**Files:**
- Create: `src/forward/gemma3.c`
- Modify: `src/CMakeLists.txt` (add `forward/gemma3.c`)

**Constants pinned from the oracle reference (`shannon-prime-lattice-llama/src/models/gemma3.cpp`
+ `llama-hparams.h`), confirmed before coding — no discovery needed:**
- **RMSNorm:** the oracle uses plain `LLM_NORM_RMS` (`x/rms·w`) for ALL Gemma norms (the `(1+w)`
  is baked into the GGUF weights at conversion). ⇒ Gemma3 reuses the engine's **existing shared
  `rmsnorm`/`rmsnorm_head`** — NO `(1+w)` variant (adding 1 would double-count).
- **Embedding scale:** `×sqrtf(n_embd)` = √1152, applied to the token embeddings.
- **QK-norm:** `rmsnorm_head(q, attn_q_norm)`, `rmsnorm_head(k, attn_k_norm)` over head_dim=256,
  BEFORE RoPE (V is neither normed nor roped).
- **RoPE base:** global layers 1e6 (`rope.freq_base`); local/SWA layers **10000** (llama default
  `rope_freq_base_train_swa`, since the GGUF has no override). NEOX rope, n_rot = head_dim.
- **Global-layer pattern:** SWA period 6 ⇒ layer `il` is GLOBAL (full attn, base 1e6) when
  `(il+1)%6==0` (il = 5,11,17,23); the other 22 layers are local (sliding window 512, base 10000).
- **Query scale:** `f_attention_scale = 1/sqrt(head_dim)` = `1/sqrt(256)` (the 1B branch), applied
  to scores (engine `ascale`); build_attn kq_scale is 1.0.
- **Residual:** post-norm THEN add — `sa_out = inpL + post_attn_norm(wo·attn)`, then
  `out = sa_out + post_ffw_norm(ffn(ffn_norm(sa_out)))`.
- **No softcap** (final_logit_softcapping absent ⇒ 0). **Tied head** (`m->output == token_embd`).

- [ ] **Step 1: Gemma helper in `gemma3.c`** (only GELU is new; norms reuse the shared kernels):

```c
/* GELU, tanh approximation (Gemma FFN, LLM_FFN_GELU): matches ggml_gelu. */
static float gelu_tanh(float x) {
    const float k = 0.7978845608028654f; /* sqrt(2/pi) */
    return 0.5f * x * (1.0f + tanhf(k * (x + 0.044715f * x * x * x)));
}
```

  Use the shared `rmsnorm`/`rmsnorm_head` (validated for Qwen3 against the same oracle) for every
  Gemma norm site — they are the exact `LLM_NORM_RMS` arithmetic the oracle applies.
- [ ] **Step 2: Write `gemma3_forward`** on the shared kernels. Skeleton (prefill, pure-f32
  reference path; arch deltas inline). Sliding-window/global selected per layer:

```c
int gemma3_forward(const qwen3_model *m, const int32_t *tokens, int n_tok, float *logits) {
    sp_read_env_knobs();
    const qwen3_config *c = &m->cfg;
    const int E=(int)c->n_embd, FF=(int)c->n_ff, HD=(int)c->head_dim;
    const int NH=(int)c->n_head, NKV=(int)c->n_head_kv, group=NH/NKV;
    const int QD=NH*HD, KVD=NKV*HD, V=(int)c->n_vocab, SW=(int)c->sliding_window;
    const float eps=c->rms_eps, gbase=c->rope_freq_base, lbase=10000.0f;  /* Task 3 confirms */
    const float ascale=1.0f/sqrtf((float)HD), embscale=sqrtf((float)E);
    /* allocate x[n_tok*E], per-position k/v caches [n_tok*KVD], scratch q/sc/ao/... */
    /* embed: x[t] = token_embd[tok]·embscale  (use embed_row then scale) */
    for (L=0; L<n_layers; L++) {
        int global = ((L % 6) == 5);              /* Task 3 confirms the pattern */
        float rbase = global ? gbase : lbase;
        int win = global ? -1 : SW;               /* full vs sliding window */
        /* attn: nx = rmsnorm_gemma(x, attn_norm); q/k/v = matmul; per-head
           rmsnorm_gemma_head(q,attn_q_norm)/(k,attn_k_norm); rope_neox(q,k,pos,rbase);
           kernels_attn_head(..., win, ...); ao = matmul(attn_output);
           ao = rmsnorm_gemma(ao, post_attn_norm); x += ao  (residual after post-norm) */
        /* ffn: nx = rmsnorm_gemma(x, ffn_norm);
           g = matmul(ffn_gate, nx); u = matmul(ffn_up, nx);
           for i: g[i] = gelu_tanh(g[i]) * u[i];
           dn = matmul(ffn_down, g); dn = rmsnorm_gemma(dn, post_ffw_norm); x += dn */
    }
    /* final: nx = rmsnorm_gemma(x, output_norm); logits = matmul(token_embd, nx) (tied) */
}
```

  Implement the elided pieces concretely following the Qwen3 `qwen3_forward_ex` structure (same
  allocation/matmul patterns), substituting the Gemma deltas above. The LM head uses the tied
  `m->output` (= `token_embd`); `matmul` already reads it as `[in=E, out=V]`.
- [ ] **Step 3: Per-checkpoint debug against the oracle (the empirical loop).** Build, then for
  the Task-3 prompt compare each engine checkpoint to `dump_layers` output, layer by layer:
  attn_norm out → Qcur/Kcur post-QK-norm → post-RoPE → kqv → post_attention_norm out → ffn
  checkpoints → result_norm. Each must match to the f32 precision floor (~1e-4 rel, per §8.6.1).
  A mismatch localizes the bug to one delta (norm offset, scale, θ, gelu, residual placement).
  Iterate until the final-layer result_norm matches.
- [ ] **Step 4: Build.**

Run: `scripts\build\build-cpu.bat`
Expected: `BUILD_EXIT=0`, `/W4` clean.

- [ ] **Step 5: Commit.**

```bash
git add src/forward/gemma3.c src/CMakeLists.txt
git commit -m "[lat-2-CPU] Gemma3 forward pass: sandwich norms, (1+w) RMSNorm, GeGLU, local/global SWA, tied head"
```

---

## Task 5: M_GEMMA3_CPU distributional gate + full regression

**Files:**
- Create: `tests/test_gemma3_forward.c`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Write `tests/test_gemma3_forward.c`** mirroring `test_forward.c` (E_CPU_2):
  read `gemma3_ref.bin` (token IDs + ref logits), run `gemma3_forward` on the IDs, compute over
  all positions: argmax match, top-5 cross, mean `KL(ggml‖engine)`. Reuse the `top5`/`kl_div`
  helpers from `test_forward.c` (copy or factor into a shared test header). Gate:

```c
SP_CHECK(argmax_match == n_pos,             "argmax matches oracle at every position");
SP_CHECK(top5_cross  == n_pos,             "top-5 set matches at every position");
double kl_max = getenv("SP_KL_MAX") ? atof(getenv("SP_KL_MAX")) : 1e-5;
SP_CHECK(kl_mean < kl_max,                  "mean KL(ggml||engine) < 1e-5 nats");
```

- [ ] **Step 2: Register in `tests/CMakeLists.txt`:**

```cmake
set(SP_GEMMA3_REF
    "D:/Files/Models/Mine/gemma-3-1b-it/gemma-3-1b-it-f16/gemma3_ref.bin"
    CACHE FILEPATH "Gemma3 oracle logit dump")
add_executable(test_gemma3_forward test_gemma3_forward.c)
target_link_libraries(test_gemma3_forward PRIVATE sp_engine)
target_compile_definitions(test_gemma3_forward PRIVATE
    SP_GEMMA3_GGUF="${SP_GEMMA3_GGUF}" SP_GEMMA3_REF="${SP_GEMMA3_REF}")
if(MSVC)
  target_compile_options(test_gemma3_forward PRIVATE /W4)
else()
  target_compile_options(test_gemma3_forward PRIVATE -Wall -Wextra)
endif()
add_test(NAME M_GEMMA3_CPU COMMAND test_gemma3_forward)
```

- [ ] **Step 3: Build + run the gate.**

Run: `scripts\build\build-cpu.bat && ctest --test-dir build-cpu -R "GEMMA3|M_GEMMA3" --output-on-failure`
Expected: GEMMA3_BIND + M_GEMMA3_CPU PASS (argmax/top5 all positions, KL mean < 1e-5).

- [ ] **Step 4: Full regression (Qwen3 guard + new Gemma3 gates).**

Run: `ctest --test-dir build-cpu --output-on-failure`
Expected: `100% tests passed` — the 16 Qwen3/foundational tests + GEMMA3_BIND + M_GEMMA3_CPU (18).

- [ ] **Step 5: Commit.**

```bash
git add tests/test_gemma3_forward.c tests/CMakeLists.txt
git commit -m "[lat-2-CPU] M_GEMMA3_CPU: Gemma3 forward distributionally matches the oracle (argmax/top5/KL)"
```

---

## Self-review notes

- **Spec coverage:** loader/config (Task 2), forward deltas — embed scale, (1+w) RMSNorm,
  sandwich norms, GeGLU, QK-norm, local/global SWA, tied head (Task 4); oracle resolution of
  ambiguities (Task 3); distributional gate (Task 5); kernel reuse via extraction (Task 1).
  Out-of-scope items (SPM, PPL, quant Gemma3, other backends) are not tasked — correct.
- **Empirical constants** (global pattern, local θ, query scale, gelu variant) are deliberately
  resolved in Task 3 and consumed in Task 4 rather than hard-asserted up front — this is an
  architecture bring-up, not pre-known logic.
- **Type consistency:** `sp_arch_t`, `c->arch`, `c->head_dim`, `c->sliding_window`,
  `c->tied_embeddings`, `L->post_attn_norm`, `L->post_ffw_norm`, `gemma3_forward`,
  `kernels_attn_head`, `sp_read_env_knobs` used consistently across tasks.

## Success criteria

1. `GEMMA3_BIND` green. 2. `M_GEMMA3_CPU` green. 3. Full regression green incl. unchanged
Qwen3 16. 4. MSVC `/W4` clean.
