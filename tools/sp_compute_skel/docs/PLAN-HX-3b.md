# PLAN — HX.3b (HVX-vectorize sp_hex_forward inner matmul kernels)

**Sprint:** Phase 2-HX.3b (the perf-flip sprint)
**Date:** 2026-05-31
**Worktree:** `D:\F\shannon-prime-repos\engine-hx-3b`
**Branch:** `sprint/hx-3b` (base ba76c69 → effectively the WIRE-HEX-FINISH state at 943a9f4)
**Sub-tag candidate:** `lat-phase-2-hx-3b-hvx-vectorized` (PASS) / `lat-phase-2-hx-3b-attempted` (FAIL)
**Status:** Plan-commit. Surfacing upstream-required architectural-decision deltas vs the dispatch prompt.

---

## Stage 0 — Mandatory pre-read (cited)

| # | File | Cited line(s) | What it tells us |
|---|------|---------------|------------------|
| 1 | `src/backends/hexagon/dsp/sp_hex_imp.c` | `:251-348` | `sp_hex_forward`. **Already HVX-vectorized.** Calls `hx_matmul_q8` (line 209-227) which uses `hx_dot_q8_hvx` (`:76-99`) under `__HVX__`. The 7 matmul call sites per layer are at lines 299, 300, 301, 316, 326, 327, 329. The bottleneck is NOT "scalar f32 placeholder" (per prompt headline) — it is **per-32-element scalar int8→f32 widen at line 83** + **per-row 5-step horizontal reduce at lines 88-94** + **per-row qf32→sf convert at line 94**. Whole-forward `qurt_hvx_lock` already in place at line 262. |
| 2 | `tools/sp_compute_skel/docs/CLOSURE-WIRE-HEX-FINISH.md` | `:13-32` headline, `:185-191` decomposition, `:314-318` HX.3b candidate | 3.63× slower at prefill (0.406 vs 1.473 tok/s). Decode invariant. The closure's HX.3b hypothesis: swap to `Q6_W_vmpye_VwVuh + Q6_W_vmpyoacc_WVwVh` widening. **Reality check:** that idiom is for 32×32→64 u32 widening (Barrett); the matmul primitive we actually want is `Q6_Vw_vrmpyacc_VwVubVb` (int8×int8 → int32 four-per-lane vector dot). |
| 3 | `tools/sp_compute_skel/src_dsp/sp_compute_crt_imp.c` | `sp_matmul_q_hvx` `:247-293`, IDL method `sp_compute_matmul_q` `:295+` | K.beta.2.5c silicon-confirmed mod_q matmul. **CRITICAL:** operates on `uint32_t` Z_q operands (NOT signed Q8 codes) and produces `uint32_t` Z_q residues (NOT f32 results). The mathematical operation it performs is `Y[b][i] = (sum_k X[b][k] * W[k][i]) mod q`. **For Q8 weight × f32 activation matmul reconstruction this is the WRONG primitive** — see §"Upstream-surfaced decision" below. |
| 4 | `tools/sp_compute_skel/src_dsp/sp_compute_crt_imp.c` | `sp_barrett_reduce32_hvx_lane` `:74-123` | K.beta.2.5b silicon-confirmed Barrett mod-reduce — used as the inner reduce of `sp_matmul_q_hvx`. SASS-audited Sprint K v0.beta. Not directly applicable to Q8 dot kernels (no need to mod-reduce when reconstructing real-valued matmul). |
| 5 | `lib/shannon-prime-system/core/forward/forward.c` | `:115-204` (NTT-attention overlay) | The math-core NTT-attention overlay uses `sp_pr_inner` (`:197`). Path: f32 → int32 (via `SP_NTT_ATTN_SCALE` quantization, `:192`) → `sp_pr_inner` returns **EXACT int64 dot product** via negacyclic poly product coefficient 0 → f32 (`:198`). Works **only because the integer dot magnitude is bounded under M/2** (one prime suffices, no Garner). Source-of-truth for "how to use Z_q dot to reconstruct f32 dot." |
| 6 | `lib/shannon-prime-system/include/sp/frobenius_lift.h` | `:148-167` `sp_frob_packed_tensor` | The arena packed Q8 layout is **per-row int8 codes + per-row f32 scale**. Reconstruction: `v_hat = code * (scale / 127)`. **Not** NTT residues. Q8 codes are signed int8 in [-127, +127]. |
| 7 | `src/backends/hexagon/sp_hex_host.c` | `hx_pack_q8` `:53-63`, `gemma3_forward_hexagon` `:112-152` | Host packs Q8 codes (line 59 `memcpy(dst, pt->codes, ...)`) + per-row f32 scales (line 61 `memcpy(scales, pt->row_scale, ...)`) into rpcmem blob. DSP-side `hx_matmul_q8` (sp_hex_imp.c:209-227) reads codes + scales directly. Forward called via `sp_hex_forward` FastRPC method at host:137-140. |
| 8 | `tools/sp_daemon/c_backend/sp_daemon_hex_glue.c` (confirmed `sp_compute_skel` and `sp_hex_skel` are SEPARATE skels) | Skel binary names: `libsp_hex_skel.so` (28 KB) vs `libsp_compute_skel.so` (separate). Two distinct FastRPC entry-point sets. **Option B (inter-skel dispatch) is operationally complex** — would require sp_hex_skel to open a FastRPC handle to sp_compute_skel from inside its forward, paying double-marshalling tax. |
| 9 | Memory entries | `reference-hexagon-v69-32x32-widening-idiom`, `reference-vtcm-per-stage-misalignment`, `reference-fastrpc-concurrent-dispatch` | 32×32→64 widening is Barrett-shape (u32×u32). For Q8 dot we want different intrinsics. VTCM not relevant for matmul (the staging cost wouldn't amortize for one-shot per-prefill use). FastRPC concurrent-dispatch not relevant (single-thread per forward; no cross-island sharding in this sprint). |

### Additional pre-read (operational)

- `/Qualcomm/Hexagon_SDK/5.5.6.0/tools/HEXAGON_Tools/8.7.06/Tools/libnative/include/hvx_hexagon_protos.h` — confirms `Q6_Vw_vrmpyacc_VwVubVb` exists (int32 vector accumulator += [ubyte × byte] dot-of-4-per-lane). This is the silicon primitive for int8 dot-product with int32 accumulation.
- ADB device check: `R5CT22445JA device` — S22U connected.
- Existing skel artifact: `src/backends/hexagon/dsp/hexagon_Release_toolv87_v69/libsp_hex_skel.so` 28 KB sha256 `d3d12782...` (the WIRE-HEX-FINISH artifact, on-device since 2026-05-31 15:34).

---

## Upstream-surfaced architectural decision (DOES NOT silently follow the prompt's Option A)

Per `feedback-no-silent-gate-revisions` ("if implementation can't meet spec'd gate, surface UPSTREAM first; do not silently revise gate, retreat to higher-level API, defer to unrelated phase, or tune fixtures until a number passes"), the prompt's recommended path needs an explicit deviation surfaced **before** Stage 1.

### What the prompt asked for

> **Option A**: copy K.beta.2.5c mod_q_matmul INTO `src/backends/hexagon/dsp/sp_hex_imp.c`...
> **Path 1**: keep dequant boundary inside the forward; each layer does `Q8_weights → Z_q (via Frobenius lift on-the-fly) → mod_q_matmul → Z_q result → fp32 (dequant)`.

### Why it is mathematically broken as stated

`sp_matmul_q_hvx` computes `Y[b][i] = (Σ_k X[b][k] * W[k][i]) mod q`. The result is a **residue modulo a 30-bit prime**, not the real-valued matmul. To reconstruct the real-valued result you need either:

  (a) **Garner CRT recombination over ≥2 primes** — adds a second full `sp_matmul_q_hvx` invocation plus a Garner reconstruction step. The k-side compute halved (mod ops are cheaper than fp mpy), but the doubled total compute + Garner overhead is unlikely to flip the perf ordering at ctx=16, AND requires shipping the second prime's weight conversion + memory layout, AND requires Garner CRT bytewise determinism (re-validates `reference-lattice-decode-determinism`).
  
  (b) **The "fits in one prime" exception** — only works when |Σ_k X*W| < q/2. For Gemma3-1B's E=1152 / FF=6912 matmuls with Q8 int8 codes (|code|≤127) and bounded activations, the inner product magnitude IS small enough (~127² * 1152 ≈ 1.86M << 2^30 ≈ 1.07G). But this requires using **integer activations** (not f32), which is a different path from "Frobenius lift on-the-fly."

### What is actually feasible in HX.3b scope

Three architecturally-honest paths exist. Listing in order of decreasing risk:

| Path | Description | Estimated risk | Expected speedup |
|------|-------------|----------------|------------------|
| **HX.3b-α** | Optimize existing `hx_dot_q8_hvx` to use **`Q6_Vw_vrmpyacc_VwVubVb`** (int8×int8 → int32-accum 4-per-lane vrmpy). Quantize activations to int8 inside the forward via per-tensor scale; matmul stays in int32 accumulator; multiply by `row_scale[j] * act_scale / 127` at the end. **One horizontal reduce per row** (down from 5-step qf32 reduce). **Zero scalar widen.** Pure HVX inner loop. | LOW — uses existing weight layout (no new packing); uses existing arena Q8 scales; only adds activation quantization (one extra pass per matmul). | 4-10× per inner-loop iteration (vrmpy is 4 mul-adds per HVX inst vs 1 for qf32_vmpy). At ctx=16 likely 2-4× wall-clock speedup. **Could flip perf ordering.** |
| **HX.3b-β** | Lift `sp_pr_inner`-style exact-integer dot to Q8 matmul: quantize activations to int32, run `int32 × int8 → int64 accum dot` mod q, output exact integer dot, reconstruct f32 via scales. Use HVX `Q6_W_vmpye_VwVuh` widening idiom for the int32 × int8 mul. | MEDIUM — requires int32 activation buffer (4× memory of f32 nope, same), but matmul ops are widening 32→64-bit not vrmpy-natural; expected to be slower than HX.3b-α at this granularity due to per-lane horizontal reduce still being needed. | 1.5-3× speedup over current; may not flip ordering. |
| **HX.3b-γ (Option A as literally specified)** | Drop in `sp_matmul_q_hvx` with single prime + on-the-fly Frobenius lift. **Mathematically corrupts the matmul** unless Garner-recombined; with single prime, results are mod-q residues mapping to garbage logits. | HIGH — guaranteed to fail T_HX3B_DECODE_DETERMINISM. Mathematically broken. | N/A — the gate it must pass is mathematically unreachable. |

### Sprint decision: **HX.3b-α**

Per the prompt's own guidance ("If the wiring exposes a real architectural blocker ... STOP, document the specific blocker, surface UPSTREAM. Don't paper over it"), the canonical HVX integer dot-product primitive for Q8 matmul is the **vrmpy family**, not the mod_q matmul family. The 32×32→64 widening idiom is for Barrett mod-reduce; it does NOT compose into a faster Q8 matmul because Q8 matmul has no mod-reduce step.

The decision is also consistent with `feedback-lattice-baseline-is-prior-lattice` — the comparison baseline is the prior lattice impl (current HVX qf32 `hx_dot_q8_hvx`), not cuBLAS. HX.3b-α improves on the prior lattice impl using the silicon-native primitive (vrmpy) that aligns with the algorithm (int8 dot with int32 accum + final scalar reconstruction).

**This deviation is surfaced UPSTREAM here in the plan-commit, before any code is written, per discipline.** If operator wants HX.3b-γ literally as specified (mod_q matmul with single prime) for "wiring exercise" purposes anyway, the closure can document the broken-determinism gate as INTENTIONAL with operator sign-off — but this plan will NOT silently revise it.

---

## Option A/B/C (sub-decision: where do the HVX kernels live?)

**Decision: Option A** — kernels live inside `sp_hex_imp.c` (same skel). Justification:

  - The new kernel (HX.3b-α `hx_matmul_q8_vrmpy`) is **not** a copy of `sp_matmul_q_hvx` (those are different primitives), so the "kernel duplication" objection of Option A doesn't apply. There is no shared kernel to consolidate yet.
  - Option B (inter-skel FastRPC) doubles marshalling tax per matmul — fatal at ctx=16.
  - Option C (unify skels) is a structural refactor that competes with the perf-flip mandate; defer to follow-on.

## Path 1/2 (sub-decision: integer end-to-end?)

**Decision: Path 1** with a Path-1b variant — keep f32 boundary in/out of `sp_hex_forward`, but the inner matmul kernel quantizes activations to int8 **just-in-time inside the kernel** (one pass per matmul). This is the "**Frobenius lift on activations**" pattern from `reference-zero-copy-invariant` applied to live activations rather than packed weights. Activation int8 quantization is fast (one scale + cast per element); reconstruction multiplies row_scale × act_scale at the end.

Path 2 (integer end-to-end through the whole forward) is the natural follow-on if HX.3b-α succeeds and we want to amortize the per-matmul quantization cost.

---

## Scope (what ships)

1. **New HVX kernel `hx_matmul_q8_vrmpy`** in `sp_hex_imp.c`. Uses `Q6_Vw_vrmpyacc_VwVubVb` for int8 × int8 → int32 accumulation, quantizes activations to int8 per-matmul. Replaces the existing `hx_matmul_q8` callsite-by-callsite under a compile-time gate (`SP_HEX_VRMPY_MATMUL`) so we can A/B without re-flashing.

2. **Replace one matmul callsite (attn_q) first**, run the bit-exact gate on-device, then expand.

3. **Skel rebuild + push to S22U.** Same `scripts/build/build-hexagon.bat dsp` flow.

4. **Re-run tok/s.** Same `timed_chat.sh` harness from WIRE-HEX-FINISH.

5. **Document the architectural-decision deltas honestly** in the closure (this plan + closure together = the auditable record).

---

## Per-stage commits

| Stage | Description | Commit message |
|-------|-------------|----------------|
| 0 | This plan-commit | `[plan] HX.3b — HVX-vectorize sp_hex_forward matmul via vrmpy (HX.3b-α, Option A, Path 1); architectural deviation from mod_q surfaced` |
| 1 | Add `hx_matmul_q8_vrmpy` kernel + compile-time gate. Host-only sanity check (the kernel compiles for hexagon target; offline DSP smoke if scripts available). | `[HX.3b Stage 1] hx_matmul_q8_vrmpy kernel (vrmpy int8x int8 → int32 accum); gated by SP_HEX_VRMPY_MATMUL` |
| 2 | Replace attn_q matmul callsite under gate. Skel rebuild. Push to device. On-device decode-determinism check on first matmul only (the rest stay on qf32 path). | `[HX.3b Stage 2] sp_hex_forward attn_q → vrmpy; T_HX3B_DECODE_DETERMINISM-1of7 PASS` |
| 3 | If Stage 2 PASS, replace remaining 6 matmul callsites. | `[HX.3b Stage 3] sp_hex_forward all 7 matmuls → vrmpy; T_HX3B_DECODE_DETERMINISM-7of7 PASS` |
| 4 | Skel rebuild + push + full decode-determinism gate (32-token sequence equality vs WIRE-HEX-FINISH baseline). | `[HX.3b Stage 4] full vrmpy skel pushed; bit-exact 32-token sequence vs WIRE-HEX-FINISH baseline` |
| 5 | tok/s measurement; closure with HEADLINE TABLE + gate dispositions. | `[HX.3b Stage 5] closure + tok/s measurement (Gemma3-1B prefill X.XX tok/s vs 1.47 ref / 0.41 hex-qf32)` |

---

## Substantive gates

### T_HX3B_HVX_KERNEL_LINKED
- **Methodology:** `hexagon-llvm-objdump -d libsp_hex_skel.so | grep -E "vrmpy|vmem"` should show vrmpy + vmem ops in sp_hex_forward's disassembly.
- **Pass:** ≥10 HVX vector instructions including ≥1 vrmpy in sp_hex_forward (the matmul kernel is hoisted statically; vrmpy appears under that function's prologue).
- **Fail handling:** If vrmpy doesn't lower (e.g., codegen falls back to scalar emulation), surface upstream — try inline asm or report nvcc-equivalent codegen blocker.

### T_HX3B_DECODE_DETERMINISM
- **Methodology:** drive identical prompt `[2,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500]` + 32 decode tokens through hex-vrmpy daemon vs WIRE-HEX-FINISH baseline (qf32). Compare 32-token output sequence (the alternating `\n` `</b>` `\n` `**` ... pattern from WIRE-HEX-FINISH closure §"Bit-exactness verification").
- **Pass:** Identical first 32 decoded token IDs.
- **Fail handling:** If argmax diverges, surface UPSTREAM. Specifically: report (a) which token diverged first, (b) the logit deltas at divergence, (c) whether activation-quantization-scale tuning can recover determinism (likely yes; may need per-tensor scale calibration rather than fixed scale).

### T_HX3B_TOKS_FLIPPED
- **Methodology:** same `timed_chat.sh` 16-prefill + 32-decode methodology as WIRE-HEX-FINISH. 2 reps per config (variance from WIRE-HEX-FINISH was <0.5%).
- **Pass:** hex-vrmpy prefill tok/s ≥ 1.473 (fp32 reference baseline).
- **Fail handling (per prompt):** surface UPSTREAM with one of three dispositions:
  1. FastRPC marshalling tax dominates at ctx=16 (file: NTT.6 long-context becomes measurement target)
  2. vrmpy didn't deliver expected per-iteration speedup (file: per-instruction `HAP_perf_get_pcycles` breakdown)
  3. Activation-quantization overhead ate the matmul savings (file: Path 2 integer-end-to-end becomes the answer)

### T_HX3B_HONEST_TABLE (purely operational)
- **Pass:** Three-row tok/s table in closure headline:
  | Config | Prefill tok/s | Decode tok/s |
  |---|---|---|
  | fp32 reference | 1.473 | 1.094 |
  | hex qf32 (WIRE-HEX-FINISH baseline) | 0.406 | 1.083 |
  | hex vrmpy (HX.3b) | ??? | ??? |

---

## Workflow discipline notes

1. **One variable per commit.** Per `feedback-bundled-changeset-root-cause-ambiguity`. Stage 1 = add kernel without enabling. Stage 2 = enable for one callsite. Stage 3 = enable for the rest.
2. **NO SILENT GATE REVISIONS.** Per `feedback-no-silent-gate-revisions`. The architectural deviation from mod_q matmul is surfaced UPSTREAM here, not silently massaged.
3. **Anti-contamination strict.** `engine-hx-3b` worktree only. DO NOT modify K.beta.2.5c `sp_compute_crt_imp.c` (it's the canonical mod_q reference; only READ from it). DO NOT modify `sp_pr_*` (NTT poly-ring surfaces).
4. **Lead with the reference.** Per `feedback-lead-with-reference-then-theory`. The reference for `Q6_Vw_vrmpyacc_VwVubVb` is the Hexagon HVX intrinsic header + the public V69 HVX PRM. No theory-first reasoning about "what should work" — the intrinsic is silicon-documented.

---

## Closure deliverables (will land in CLOSURE-HX-3b.md)

1. HEADLINE TABLE (three rows).
2. T_HX3B_DECODE_DETERMINISM result (identical sequences or honest tolerance documentation).
3. T_HX3B_HVX_KERNEL_LINKED disassembly excerpt.
4. Architectural decisions taken (this plan + any further deviations).
5. Per-stage build commands.
6. Skel pre/post hashes.
7. Wall-clock breakdown.
8. Honest interpretation.
9. Files changed.
10. Commits on `sprint/hx-3b`.
11. Sub-tag (PASS / FAIL).
12. What's NOT done.
13. What unblocks.
14. Worktree status.

`git push -u origin sprint/hx-3b` at end.

---

## Risk table

| Risk | Mitigation |
|------|------------|
| `Q6_Vw_vrmpyacc_VwVubVb` operates on **unsigned×signed** byte; Q8 codes are signed×signed. Sign handling at the boundary. | Cast pattern: weights stay signed Q8 (vector treated as signed via second operand); activations quantized as `uint8_t` with bias-128 trick (subtract 128 from each ubyte post-mul via a constant correction term). Standard llama.cpp-style int8 dot pattern. |
| Activation quantization adds a per-matmul pass; if cost > matmul savings, no win | Measure separately via `HAP_perf_get_pcycles` in Stage 2; if activation quant dominates, hoist scale calibration once per-token rather than per-matmul. |
| vrmpy lowering may not be ideal on V69 codegen tier | If disassembly shows scalar fallback, inline asm fallback per `reference-nvcc-paired-register-bug` discipline. |
| Decode determinism may diverge by ULP because the int8 quantization rounding is not bit-equal to qf32 mul | Per `reference-lattice-decode-determinism`: the gate is **argmax equality**, not bit-equality of logits. ULP-level divergence is acceptable iff argmax-preserved. If argmax diverges, surface UPSTREAM with the specific token + logit-delta evidence. |
| Skel rebuild requires SDK chain; previous WIRE-HEX-FINISH used the same chain successfully on this host | Same `scripts/build/build-hexagon.bat dsp` flow; same toolchain (hexagon_Release_toolv87_v69). |

---

## Memory entry candidates (post-closure)

- `reference-hexagon-vrmpy-q8-matmul-pattern` — capture the in-vector int8 dot pattern (vrmpy + per-row-scale reconstruction + activation-quant-scale recombination) as the canonical Q8-on-HVX kernel template for future Hexagon-targeted matmul work.
- `reference-hx-3b-arch-decision-rod_q-vs-vrmpy` — capture why mod_q matmul is NOT the right primitive for real-valued matmul reconstruction without CRT cascade. Avoid future "wire mod_q in" missteps.

---

## What this plan DOES NOT promise

- A "1× or better" tok/s flip at ctx=16. The prompt set this gate, but per the upstream-surfaced architectural reality (HX.3b-α uses a different primitive than the prompt's Option A spec), the realistic speedup envelope is 2-4× over current hex-qf32 (= 0.8-1.6 tok/s prefill), which **might** clear the 1.473 fp32 ref bar but is not guaranteed.
- A flip at ctx=16 if FastRPC marshalling tax dominates the per-matmul vrmpy savings.
- Bit-equal logits across configs. Argmax equality only, per `reference-lattice-decode-determinism`.

If the gate fails: closure documents honestly; sub-tag becomes `lat-phase-2-hx-3b-attempted` with named blocker; user has a measured number + a specific reason rather than a flattering hand-wave.
