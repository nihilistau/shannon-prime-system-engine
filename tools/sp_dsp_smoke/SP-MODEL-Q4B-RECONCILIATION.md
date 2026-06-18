# SP-MODEL Q4B Reconciliation — the `.sp-model` weight-loading story for the universal crate

**Status (2026-06-18): DECIDED — (B). The universal crate does NOT need its own OK_Q4B
loader.** The CUDA byte-exact path consumes the engine's already-resident
`qwen3_model*` device weights; the crate's HVX Q8-tile loader keeps its OK_Q8 scope.
This closes CONTRACT-BYTEEXACT-forward §8 step 3 as a decision (no loader code).

---

## 1. The question (CONTRACT-BYTEEXACT-forward.md §8, bridge step 3)

> Reconcile the `.sp-model` loader to OK_Q4B (crate currently reads OK_Q8 single
> tiles) — **or** pass the engine's resident `qwen3_model*` / `g_w` device weights
> rather than re-loading.

Two designs:

- **(A)** Extend the crate's `tools/sp_dsp_smoke/src/sp_model_layer.rs` to also read
  `SP_DT_OK_Q4B` (dtype id 13): decode 4-bit nibble-packed codes `[-7,7]` + the
  per-32-block f16 scales from the `.bscale` sibling (`SP_DT_BLOCK_SCALE_FP16`,
  id 14) into i16/f32.
- **(B)** The CUDA/engine path passes the resident `qwen3_model*` and the crate does
  NOT re-load. The crate's `sp_model_layer.rs` Q4B gap is then an **HVX-track-only**
  concern, not a universal-path blocker.

## 2. What each backend actually loads (the evidence)

### CUDA path — Q4B is decoded entirely engine-side, never in the crate

The crate's forward backend seam is `register_forward_backend` →
`cuda_forward_dispatch.rs` (`sp_wire_cuda_forward_dispatch`) →
`c_backend_cuda/sp_daemon_cuda_glue.c` (`sp_daemon_cuda_forward`). The glue receives
the model as **`const void *qm_opaque`** and arch-routes
(`sp_daemon_cuda_glue.c:78-92`):

```c
case SP_ARCH_GEMMA4: return gemma4_forward_cuda(qm, tokens, n_tok, logits);
```

The `qwen3_model*` behind `qm_opaque` is built **by the engine**, not the crate:
`sp_model_to_gemma4` (declared `lib/.../include/sp/sp_model.h:181`, impl
`lib/.../core/session/sp_model_bridge.c` / `sp_session.c`) reconstructs it from the
`.sp-model` mmap, with the OK_Q4B codes + `.bscale` decoded into the math-core
**arena** (`core/arena/arena.c`).

`gemma4_forward_cuda` then uploads those packed arena tensors to the GPU and caches
them in the static `g_w` keyed on the model pointer
(`src/backends/cuda/cuda_forward.cu`):

- `g_w` declaration + key check: `cuda_forward.cu:1081`, `:1359`, `:1436`
  (`if (g_w.key != m) { free_weights(&g_w); build_weights(m, &g_w); }`).
- `build_weights` → `build_w` (`:1186`) → for an arena tensor calls
  `upload_packed` (`:1123`), which uploads `pt->codes` and — for OK_Q4B —
  `pt->bscale` (the per-32-block f16 scales) directly (`:1141-1144`).

So on the CUDA byte-exact path the OK_Q4B nibbles + per-block f16 `bscale` are read
by **engine C (`arena.c` + `cuda_forward.cu`)**, uploaded once, and cached device-
resident in `g_w`. The crate's job is to pass `qm_opaque` through the glue — it never
sees a Q4B byte. **Re-loading OK_Q4B in Rust would duplicate the engine decode for
zero benefit on this path** (and risk a second, divergent dequant of the same bytes —
exactly what byte-exactness forbids).

### HVX path — the crate's loader is deliberately OK_Q8-only

`tools/sp_dsp_smoke/src/sp_model_layer.rs` is the Hexagon/HVX-side tile loader. It
reads ONE `SP_DT_OK_Q8` (id 10) weight tile + its per-row `SP_DT_FROBENIUS_SCALE_FP32`
(id 12) companion and dequants Q8→i16 for the Halide matmul
(`sp_model_layer.rs:19-20, 137-145`). It hard-errors on any non-Q8 dtype by design
(`:137-140`). The driver `sp_model_layer_smoke.rs` runs on-device (`#[cfg(target_os
= "android")]`) against a Qwen3-0.6B `.sp-model`.

This is the HVX Sprint-I/J line — a single-tile DSP correctness vehicle, not a
12B-gemma4 production loader. The trusted gemma4-12B weight is OK_Q4B, which **no
HVX gate consumes today**.

## 3. Decision: (B)

**The universal/CUDA byte-exact path passes the engine's resident `qwen3_model*`;
the crate does NOT re-load OK_Q4B.** Rationale:

1. **Single source of truth for the Q4B decode.** The trusted gemma4-12B path
   (`sp_transcode --st` → OK_Q4B `.sp-model` → `sp_model_to_gemma4` → arena →
   `g_w`) already decodes codes + per-32-block f16 `bscale` in engine C. Byte-
   exactness is a property of *one* canonical arithmetic; a parallel Rust decoder
   would be a second implementation to keep bit-identical for no gain.
2. **The seam already carries the model, not the bytes.** The L1 ABI deliberately
   passes `qwen3_model*` as `const void *qm_opaque` (`sp_l1.h §6`,
   `sp_daemon_cuda_glue.c:46-48`). The crate is L2 (orchestrator + scalar bit-exact
   *reference*); L1/CUDA owns the resident weights and `g_w` caching. Passing the
   handle is the intended architecture, not a workaround.
3. **No near-term gate needs a crate-side Q4B loader.** The open byte-exact gates
   (CONTRACT-BYTEEXACT §5/§8) are the **island** reconciliations
   (RMSNorm/softmax/GELU/RoPE — already in-crate as `sp_islands_q_ref.rs`,
   G-ISLANDS-Q-REF GREEN) and the end-to-end "byte-exact-when-off" logit diff
   (§8 step 5). Those exercise `gemma4_forward_cuda` over the **engine-loaded**
   weights; none requires the crate to parse OK_Q4B.
4. **Cost asymmetry.** (A) is real work (nibble unpack + f16→f32 + per-32-block
   scale-apply + a host gate) whose only consumer would be a hypothetical HVX
   gemma4-12B path that does not exist and is not on the near-term roadmap. (B) is
   zero code.

## 4. Where OK_Q4B is read, by backend (the map)

| Backend | Loads OK_Q4B? | Reader | Notes |
|---|---|---|---|
| CUDA (byte-exact / 12B) | yes | engine C: `sp_model_to_gemma4` → `core/arena/arena.c` → `cuda_forward.cu` `upload_packed`/`build_weights`, cached in `g_w` keyed on `m` | crate passes `qm_opaque`; never touches Q4B bytes |
| Universal crate (L2) | **no — by decision (B)** | n/a | consumes the resident `qwen3_model*` via `sp_session_register_forward_backend` → glue |
| HVX / Hexagon (`sp_model_layer.rs`) | no | crate Rust: OK_Q8 single tile only (id 10 + FROBENIUS_SCALE_FP32 id 12) | Sprint-I/J DSP correctness vehicle on Qwen3-0.6B; hard-errors on non-Q8 |

## 5. When (B) would need revisiting → (A)

The decision flips to (A) — a crate-side OK_Q4B loader — only if a future sprint
puts the **gemma4-12B OK_Q4B weight onto the HVX/DSP backend** (i.e. an HVX gemma4
matmul that must read the same nibble+`bscale` format the CUDA arena reads). At that
point `sp_model_layer.rs` would gain an additive `read_layer_w_gate_tile_q4b`
(decode id-13 codes `[-7,7]` + id-14 per-32-block f16 `bscale`, dequant to i16),
mirroring the existing Q8 reader + a host `*_ref_test` gate, and gated against the
engine `upload_packed` dequant for bit-exactness. **No such sprint is scheduled**
(the HVX line is single-tile Qwen3-0.6B; the gemma4-12B campaign is CUDA). Until
then, this gap is documentation, not debt.

## 6. Format reference (for the future-(A) implementer)

From `lib/shannon-prime-system/include/sp/sp_model.h §5`:

- `SP_DT_OK_Q4B = 13` — int4 codes `[-7,7]`, nibble-packed; per-32-block f16 scales
  live in the `<weight>.bscale` sibling.
- `SP_DT_BLOCK_SCALE_FP16 = 14` — `<weight>.bscale`: `f16[rows * ceil(cols/32)]`.
- `SP_DT_OK_Q8 = 10` (what the crate reads today) + `SP_DT_FROBENIUS_SCALE_FP32 = 12`
  per-row scale companion.

The crate's existing `SpTensorEntry` already surfaces `dtype_id`, `dims`,
`offset_in_data`, `block_size`, `block_count` (`sp_model_layer.rs:39-48`), so a
future Q4B reader has the table fields it needs; only the data-region decode is new.
