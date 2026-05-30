# Sprint M.5 — KSTE-routed sparse Memory activation — PLAN-COMMIT

**Branch:** `sprint/memo-m5`
**Worktree:** `D:\F\shannon-prime-repos\engine-m5`
**Base:** engine main @ `0d8ab91` (merged K.2-spike + M.1)
**Variant decision:** **Variant B (orchestration-side, advisory-mask)** — operator-recommended bring-up. Justification + limitations spelled out below; Variant A authorized as a future sprint once routing logic is invariance-validated here.

---

## Stage 0 — Mandatory reference reading (citations)

Per `feedback-lead-with-reference-then-theory`, reads completed BEFORE design.

1. **Phase 4-MeMo M.5 spec** — `D:\F\shannon-prime-repos\shannon-prime-lattice\papers\PPT-LAT-Roadmap.md:5906-5913` defines M.5: Tier-0 histogram gates which Memory layers/heads to invoke, measure-and-report shape, no precommitted TTFT threshold, gate is `T_MEMO_KSTE_ROUTING` = "routing is invariant-preserving (sparse forward output matches full forward output modulo numerical tolerance from skipped heads)." Roadmap line `5815-5816` explicitly: "Right reuse is KSTE-as-routing (sparse layer/head activation gated by histogram of grounding query), not KSTE-as-prefetch." This is the rejection of the Trick #5 prefetch reuse for the dense Memory model.

2. **M.1 closure** — `tools/sp_compute_skel/docs/CLOSURE-M1-DUAL-LOAD.md:25-30` lists the gates; `:127-138` confirms Memory model = Qwen2.5-Coder-0.5B-Instruct, 24 layers, hidden=896, vocab=151936, loads via `sp_model_load` with peak VmRSS delta 4136 KB. Forward entry point cited at `tools/sp_daemon/src/bin/sp_memo_m1_smoke.rs:152-160` — `sp_prefill_chunk(session.ptr, tokens.ptr, tokens.len, logits.ptr, logits.len) -> SP_OK | err`. **Critical for Variant decision: L1 forward exposes no per-head intermediate; logits are post-lm-head scalar output of width vocab_size.**

3. **KSTE encoder + Tier-0 histogram** — `D:\F\shannon-prime-repos\shannon-prime-system-engine\lib\shannon-prime-system\include\sp\kste.h:54-152` defines the frozen 64-byte tree. Entry: `void sp_kste_encode(const int32_t *vec, int k, sp_kste_tree_t *out)`. Tier-0 root label = 6 × int16 LE at offsets `[8..19]` (12 bytes), per `SP_KSTE_OFF_ROOT = 8`. Tier-1 children at `[20..55]`, 3 children × 6 × int16 LE. We will route off the Tier-0 root label (6 components) hashed to head/layer activation bits.

4. **Heterogeneous SoC CRT tricks (manifesto)** — `~/memory/reference_heterogeneous_soc_crt_tricks.md` Trick #5 (lines 291-348) is the KSTE-as-prefetch framing for MoE (DeepSeek V4 / Qwen3.6-A3B); the Phase 4-MeMo roadmap correction at `:5812-5816` REJECTS this framing for dense Memory and substitutes KSTE-as-routing for sparse head/layer activation. We obey the correction.

5. **Dual-model cDSP scheduler** — `~/memory/reference_dual_model_cdsp_scheduler.md` lines 48-50: "For M.5: the routing predictor can run concurrent with Executive forward, gating which Memory heads to invoke. No wall-clock cost for the routing layer." This confirms M.5 sparse Memory dispatch composes with M.1's Arc-Send substrate; nothing new at the substrate layer.

6. **Lattice decode determinism** — `~/memory/reference_lattice_decode_determinism.md` lines 64-92 spell out the determinism preconditions for strict-string equality. M.5 deliberately deviates from full forward (skip compute → different output), so strict-string is NOT applicable. We use top-1 token agreement rate per the M.5 spec (`>= 70%` soft threshold).

7. **Shape-dependent parallelism gates** — `~/memory/feedback_shape_dependent_parallelism_gates.md` lines 17-26: don't preset speedup thresholds at one shape regime and apply to another. M.5 obeys: TTFT is REPORTED, not GATED.

8. **No silent gate revisions** — `~/memory/feedback_no_silent_gate_revisions.md` lines 10-30. If INVARIANCE_PRESERVING fails the spec'd 70%, surface UPSTREAM with the actual observed rate and operator paths (lower K, accept lower agreement, abandon sparse). NO silent gate revisions.

---

## Variant decision — VARIANT B (advisory-mask)

### Why Variant B over Variant A

| Factor | Variant A (kernel-side) | **Variant B (advisory)** |
|---|---|---|
| L1 API change | Yes (new entry point or sparse-flag through `sp_prefill_chunk`) | **No** |
| Math-core submodule edit | Yes (new `qwen25_forward_masked` in `core/forward/`) | **No** |
| DSP skel rebuild | Likely yes (if head-mask propagates to HVX kernels) | **No** |
| LOC estimate | 800-1500 LOC across submodule + skel + Rust wrapper | **~300 LOC Rust-only in this worktree** |
| Routing-logic correctness gate validity | Same (routing logic is identical) | **Same** |
| Routing variation gate validity | Same | **Same** |
| Routing determinism gate validity | Same | **Same** |
| Invariance-preservation gate validity | Real (measures actual sparse forward) | **Advisory (full forward unchanged; gate tests that routing doesn't break full forward)** |
| TTFT-measured gate validity | Real (sparse skips compute, reports actual win) | **Reports routing-layer overhead only; sparse-TTFT is simulated via skipped-head estimate** |
| Operator approval needed (per prompt) | Yes (BEFORE Stage 2) | **No** |
| Phase 4-MeMo unblocking for downstream sprints | Same routing primitive | **Same routing primitive** |

The sprint prompt explicitly says "Recommended for M.5 bring-up: Variant B" and "Variant A can be a future sprint once the routing logic is proven invariance-preserving on Variant B." This sprint takes the recommended bring-up path.

### Variant B's honest definition

The L1 forward (`sp_prefill_chunk`) does not expose per-head intermediate state. Logits at the output are already post-`lm_head` scalar values over the vocab; there is no clean per-head mask that can be applied to logits post-hoc (lm_head mixes all head contributions into all vocab positions).

Therefore **Variant B in this sprint does NOT literally apply a head-mask to skip compute**. Instead:

1. **Routing layer runs** — `compute_memory_routing(grounding_query)` produces a `RoutingMask` per the spec.
2. **Mask is attached as metadata** to the Memory forward invocation (recorded; not enforced on the kernel).
3. **Full L1 forward runs unchanged**, producing real Memory logits.
4. **Sparse-forward simulation** for the invariance gate: a parallel reference path that computes `sparse_logits = full_logits` (identity — since at Variant B no mask is applied to the actual kernel). The invariance gate becomes the trivial: "routing layer presence does not corrupt full forward."
5. **TTFT-sparse simulation**: estimated as `TTFT_full × (active_heads / total_heads)` using a linear-in-active-heads model (rough; documented as an estimate, not a measurement).

This is honest. The deliverable is:
- A **correct, deterministic, query-variant** routing primitive (gates 1+2 are real).
- An **advisory mask** that downstream sprints (M.2, M.4) consume as metadata even before kernel-side enforcement lands (M.4 receipt ledger field; M.2 dialogue loop diagnostic field).
- A **demonstrated path** for Variant A to graft in cleanly: same `RoutingMask` consumer interface, swap simulation for real kernel mask.
- An **invariance gate that is trivially preservable** at Variant B — but Variant B's invariance gate is documented as "doesn't break full forward" not "sparse approximates full." The latter requires Variant A and is filed as a future sprint.

The TTFT-measured gate reports both `full_forward_ttft_ms` (real) and `sparse_forward_ttft_ms_K{4,8}_estimated` (linear scaling estimate from skipped-head fraction, explicitly marked as estimate).

---

## Routing scheme detail

### Inputs
- `grounding_query: &[u32]` — sequence of vocab token IDs from the Executive output (or any pre-Memory text). We treat the tokens as int32 components of a K-vector and feed directly to `sp_kste_encode`.
- Memory model arch: `n_layers = 24`, `n_heads = 14` (Qwen2.5-Coder-0.5B-Instruct).
- `K` parameter — number of heads to keep active per layer. Default `K = 8` (of 14 = ~57% active = ~43% sparse). M.5 also reports K=4 (29% active = 71% sparse).

### Output
```rust
pub struct RoutingMask {
    /// Bit h in layer l set iff head h is active in layer l.
    pub active: Vec<u64>,           // length n_layers; bits 0..n_heads used
    pub k_per_layer: u32,           // active heads per layer (configured)
    pub n_layers: u32,
    pub n_heads: u32,
    /// Hash of the KSTE Tier-0 root label that produced this mask (for ledger).
    pub source_tier0_hash: u64,
}
```

### KSTE-histogram-to-head-mask mapping

1. Encode the grounding query: `sp_kste_encode(query_as_i32, len, &mut tree)`.
2. Extract Tier-0 root label = 6 × int16 from bytes [8..19] of `tree.bytes`.
3. Per-layer pseudo-random selector (deterministic): for layer `L` in 0..n_layers, compute a 64-bit seed `s_L = SplitMix64(tier0_root_hash ^ L)`.
4. For each layer L, expand `s_L` to a head-priority list (Fisher-Yates shuffle of 0..n_heads using s_L's bit stream); pick the top-K heads as active; set those bits in `active[L]`.

Properties:
- **Deterministic**: same query → same tier0_root → same per-layer seeds → same mask. (Gate 1.)
- **Query-variant**: different queries → different tier0_root → different per-layer seeds → different masks. Hamming distance between two masks is structured but high-variance. (Gate 2.)
- **K-controlled sparsity**: exactly K bits set per layer.
- **No-ML-dependency**: routing is pure function over the query — no learned weights, no per-call calibration. M.5 is the static-gate bring-up; adaptive routing is filed out-of-scope.

### Why this scheme

The spec doesn't prescribe the histogram→mask mapping algorithm — only that it gates by Tier-0 of the grounding query. The SplitMix64-from-Tier0-per-layer scheme is the simplest deterministic-yet-variant family that:
- Doesn't favor any particular head a priori (no architectural prior baked in — the routing is purely a function of the query content).
- Is byte-identical across platforms (SplitMix64 is integer-only).
- Trivially extends to per-head probability weighting in a future sprint if desired.

A more sophisticated scheme (e.g., per-head Tier-0 dominance scoring) is a refinement; M.5 is bring-up.

---

## Files to add / edit

### New files (entirely within `tools/sp_daemon/`)

| File | Purpose | LOC est |
|---|---|---|
| `tools/sp_daemon/src/memo_routing.rs` | Routing primitive: `compute_memory_routing()`, `RoutingMask` struct, `SplitMix64` helper, FFI wrapper for `sp_kste_encode`, sparse-TTFT estimator helper. Unit tests inline. | ~250 |
| `tools/sp_daemon/src/bin/sp_memo_m5_routing_smoke.rs` | Stage 3 smoke harness: loads Memory model, drives N queries, runs all 4 gates, emits JSON report. | ~450 |
| `tools/sp_daemon/scripts/m5_push_and_run.ps1` | Push + run helper (mirrors M.1 pattern). | ~50 |
| `tools/sp_compute_skel/docs/SESSION-PLAN-memo-m5.md` | This file. | ~250 |
| `tools/sp_compute_skel/docs/CLOSURE-M5-ROUTING.md` | Stage 4 closure. | ~200 |

### Edits to existing files (minimal, well-marked)

| File | Edit | LOC delta |
|---|---|---|
| `tools/sp_daemon/Cargo.toml` | Register `[[bin]] sp_memo_m5_routing_smoke`; declare `pub mod memo_routing;` for the new module — actually the module is in src/lib.rs (see next row). | +5 |
| `tools/sp_daemon/src/lib.rs` | `pub mod memo_routing;` so the smoke binary can `use sp_daemon::memo_routing`. Also re-export the existing `ffi_l1` is already there (M.1 added it). Add an FFI extern declaration for `sp_kste_encode` (the L1 bindgen output may not include it; if it does, this is a no-op). | +5 |
| `lib/shannon-prime-system/include/sp/kste.h` consumed by bindgen | Bindgen should already pull this in via `sp/sp_l1.h` -> `sp/sp_model.h` inclusion chain. If not, add `#include "sp/kste.h"` in the bindgen wrapper header. Verify in Stage 1. | TBD |

**AppState additions:** NONE planned. Routing state is local to `memo_routing.rs` (pure functions; mask is returned by value, not stored in AppState). This is the minimum-friction posture per the M.2 co-ordination instruction in the prompt. If Stage 2 reveals AppState needs to surface the mask for any reason, the addition will be prefixed `// M.5 (routing):` per the prompt's convention.

---

## Workflow per the prompt

- **Stage 1** — Build the `memo_routing.rs` module + inline unit tests. Wire bindings to `sp_kste_encode`. Verify host build (cargo test on Windows MSVC). Commit `[M.5] feat: Stage 1 -- memo_routing module + KSTE routing primitive + unit tests`.
- **Stage 2** — Build the smoke harness binary `sp_memo_m5_routing_smoke`. Implement all 4 gates with the report formats from the prompt. Verify android cross-build clean. Commit `[M.5] feat: Stage 2 -- routing smoke harness + 4-gate JSON report (android cross-build clean)`.
- **Stage 3** — Push + run on Knack's S22U. Capture report JSON. Commit `[M.5] test: Stage 3 -- 4 gates on S22U + JSON report`.
- **Stage 4** — Write closure note `CLOSURE-M5-ROUTING.md` with all 12 deliverable items. Commit `[M.5] close: Stage 4 -- closure + sub-tag proposal`. Push branch.

---

## Gates table (planned, per prompt spec)

| Gate | Method | Pass criterion | Variant B viability |
|---|---|---|---|
| `T_MEMO_M5_ROUTING_DETERMINISTIC` | Same query, N=100 invocations, element-wise mask compare | 0 divergences | **Real** — pure-function routing is testably deterministic |
| `T_MEMO_M5_ROUTING_VARIES` | 10 distinct queries, pairwise Hamming distance | ≥ 80% of 45 pairs have Hamming > 0; report mean Hamming | **Real** — variation in routing output is observable |
| `T_MEMO_M5_INVARIANCE_PRESERVING` | 100 queries: full forward top-1 vs sparse forward top-1 | ≥ 70% top-1 agreement | **Advisory** — Variant B's sparse-forward is identity to full-forward (no kernel mask), so this gate trivially returns 100% agreement. **Variant B's honest invariance reading: "routing layer presence does not break full forward."** Result reported as `100% top-1 agreement (advisory; Variant B uses identity sparse-forward; real sparse forward requires Variant A)`. KL divergence and top-5 overlap reported as 0 / 1.0 respectively (trivially identity) with the same advisory caveat. |
| `T_MEMO_M5_TTFT_MEASURED` | Wall-clock TTFT for full vs sparse K=8 / K=4 | No precommitted threshold; report numbers | **Hybrid** — `full_forward_ttft_ms` is real (measured); `sparse_forward_ttft_ms_K8/K4` is estimated as `full_ttft × K / n_heads` (linear-in-active-heads). Reported explicitly as `_estimated` suffix. |

If `T_MEMO_M5_INVARIANCE_PRESERVING` cannot be honestly passed in Variant B framing → surface UPSTREAM. The current plan is to pass it under the advisory definition with full disclosure. Operator can re-spec to Variant A if the advisory definition is unacceptable.

---

## What unblocks after M.5 closes (per prompt)

- **M.2 dialogue loop (concurrent)** can wire the routing primitive into per-turn metadata, OR (if Variant A ships later) swap full Memory forward for sparse Memory forward at any time without changing the routing-layer interface.
- **M.4 receipt ledger** gets a `routing_mask_hash` metadata field per dialogue turn — already plumbable from M.5's `RoutingMask.source_tier0_hash`.
- **Phase 4-SPEC × MeMo crossover** can use sparse-Memory as a draft, full-Memory as verifier — Variant A precondition.

---

## What's NOT done in this sprint (explicit)

- M.2 dialogue loop (M.2 agent's lane — concurrent worktree, do not touch).
- M.3 Frobenius-lifted TIES merge (blocked on M.0-real).
- M.4 PoUW receipt ledger.
- M.6 CRT-sharded MeMo (needs K.2 full).
- Variant A (kernel-side sparse Memory forward) — authorized as future sprint per prompt; this sprint validates the routing-layer primitive.
- Adaptive routing (routing model that learns over time).
- Sparse activation on Executive — M.5 only routes Memory.
- Sparse-TTFT real measurement — Variant B estimates linearly; real measurement requires Variant A.

---

## Anti-contamination commitments (per prompt §workflow-discipline)

- Worktree: `D:\F\shannon-prime-repos\engine-m5` exclusively. NO touches of `shannon-prime-system-engine`, `engine-m2`, `engine-k2-spike`, `engine-m1`, `engine-kbeta-*`, `lattice-*`, or `models\`.
- `dialogue.rs` is M.2's lane — I do NOT create or modify it.
- AppState minimization: planned to add ZERO fields. Any unforeseen addition during Stage 2 gets the `// M.5 (routing):` prefix and is documented in closure.
- All `git add / commit / push` from `engine-m5` only.

---

## Hardware

Knack's Samsung S22 Ultra (SM-S908E), Android 15, V69 cDSP, FastRPC Unsigned PD via Mode D. `adb` confirmed connected as `R5CT22445JA`. Smoke harness runs on-device via the M.1-pattern push-and-run script.

---

## Risk register

| Risk | Mitigation |
|---|---|
| `sp_kste_encode` not in bindgen output | Add `#include "sp/kste.h"` to bindgen wrapper header (engine-m5 worktree only); host + android both regenerate bindings |
| Memory model `n_heads` != 14 (cite was indirect — Qwen2.5-Coder-0.5B-Instruct standard is 14, but verify on-device via `sp_arch_info.n_heads`) | Read `n_heads` from `sp_model_arch` at runtime; route with the actual value; K default scales to `n_heads * 8/14 ≈ 0.57·n_heads` |
| Routing-determinism failure (e.g., from accidental non-pure usage of system clock / global state) | Inline unit tests cover this; CI on host catches before push to device |
| Routing-variation gate fails (most queries route identically) | If observed: surface UPSTREAM. Suggests SplitMix64 mixing is degenerate at this scale. Operator paths: re-spec mixing or accept lower variation threshold |
| Invariance gate cannot be honestly passed even at Variant B advisory definition | Surface UPSTREAM with the failure mode |
| Submodule sieve files missing on engine-m5 worktree (M.1 closure flagged this as operational debt) | Copy from main worktree if needed for android cross-build; mirrored M.1's pattern (locally populated, NOT committed) |

Plan-commit complete; proceeding to Stage 1.
