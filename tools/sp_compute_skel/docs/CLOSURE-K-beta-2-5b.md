# CLOSURE — Sprint K v0.beta-2.5b (HVX Vector Barrett Primitive)
**Date:** 2026-05-30
**Branch:** `sprint/kbeta-2-5b` (engine worktree `D:\F\shannon-prime-repos\engine-kbeta-2-5b`)
**Base:** engine main @ 41963ac (Stage 2.5a closed)
**Status:** **PARTIAL — primitive math gates PASS, substrate gates UPSTREAM-REQUIRED**

## Headline

The HVX-vectorized Barrett-reduction primitive is **silicon-correct and SASS-clean**: 2048
samples × 2 primes match the Rust scalar reference per-element bitwise (zero divergence,
max_lane_diff=0); the cross-mode invariant (scalar skel == vector skel) holds across the
same population; every emitted intrinsic produces the planned Hexagon V69 HVX opcode.

The two substrate gates (DUAL_DISPATCH_SPEEDUP and LEAK_FREE) **FAIL the
prompt-supplied thresholds** but in ways that are **NOT Barrett-primitive failures** —
they reveal mismatches between the prompt's gate definitions and the shape regime the
Barrett primitive operates in. Per `feedback-no-silent-gate-revisions`, this closure
leaves both gates in **UPSTREAM-REQUIRED** state with full diagnostic detail rather than
silently relaxing thresholds.

## Gates table

| Gate | Threshold | Observed | PASS/FAIL | Comment |
|---|---|---|---|---|
| **M_K_beta_MATH_IDENTITY** | 0 divergences across 2048 samples × 2 primes | samples_compared=2048, divergence_count=0, max_lane_diff=0 | **PASS** | Bitwise identity skel-vector ≡ Rust scalar reference for both primes. |
| **BARRETT_CORRECTNESS** | r ≡ a·b (mod q) AND 0 ≤ r < q for all 2048 samples × 2 primes | samples_correct=2048, samples_total=2048 (per prime: 1024/1024) | **PASS** | i128 reference cross-validation: every lane satisfies both invariants. |
| **DUAL_DISPATCH_SPEEDUP** | ≥ 1.5× wall-clock speedup vs K v0.alpha's 1.935× baseline | speedup=1.006×, overlap_fraction=0.1955 | **FAIL — UPSTREAM-REQUIRED** | See §"Diagnosis: speedup gate" below. |
| **LEAK_FREE** | VmRSS delta ≤ 1024 KB after 10000 cycles | vmrss_delta_kb=2104 (warmup), cycles_run=10000, no errored iterations | **FAIL — UPSTREAM-REQUIRED** | VmRSS stable iter 5000 → 10000 (Δ=4 KB). Iter 0 → 5000 (Δ=2100 KB) is allocator/page-cache warmup, NOT a leak. See §"Diagnosis: leak gate" below. |

## Cross-mode invariant (bonus)

| Gate | Threshold | Observed | PASS/FAIL |
|---|---|---|---|
| skel mode=0 ≡ skel mode=1 | 0 divergences across 2048 samples | divergences=0 | **PASS** |

This was not in the prompt's substantive-gate list but is the most direct confirmation
that the HVX vector path produces identical results to the silicon-confirmed Stage 2.5a
scalar path on the SAME skel running BOTH modes on the SAME (a, b) buffers.

## SASS audit summary

Total intrinsics emitted in the inner loop: 19 (3 splats hoisted as loop invariants).
Divergences from expected: **0**. Every emitted opcode matches the Stage 1 plan exactly,
including the 3 instances of the V69 `vmpye + vmpyoacc` 32×32→64 widening idiom
(per Hexagon HVX Programmer's Reference Manual §151).

See `tools/sp_compute_skel/docs/HVX_BARRETT_SASS_GATES.md` for the per-intrinsic table.

## Architectural delta

What is now true that wasn't before:

1. **Manifesto Trick #1 (cDSP-internal CRT-sharded compute) is silicon-confirmed at the
   HVX-vector primitive scope** — the math holds bit-exact across both primes through
   the same SSR:XA-eligible HVX vector contexts that the K v0.alpha saturating-matmul
   used. Stage 2.5a confirmed at the scalar-pipe level; Stage 2.5b confirms at the HVX
   vector-pipe level.
2. **The AMENDMENT plan §1 mapping table's intrinsic-count claim is overturned** —
   actual cost of i32×i32→i64 widening on V69 HVX is 2 instructions, not ~6. Documented
   in `HVX_BARRETT_MAPPING.md` + `HVX_BARRETT_SASS_GATES.md` "Note on the SASS observation."
3. **The full K v0.beta umbrella is NOT closed** — that requires mod_q_matmul + Garner
   recombination on top of this Barrett primitive. Stage 2.5b closes ONLY Stage 2.5
   (the Barrett primitive sub-gate of K-beta).

## Diagnosis: DUAL_DISPATCH_SPEEDUP gate UPSTREAM-REQUIRED

**Observed:**
- single-invoke wall (q_1 at n=65536): 1644 μs
- single-invoke wall (q_2 at n=65536): 1324 μs
- sequential total wall (both invokes back-to-back, including Vec construction): 4881 μs
- concurrent total wall (Arc<FastRpcSession> dual-thread): 4853 μs
- speedup: 1.006×

**Why this is NOT a Barrett-primitive failure:**

K v0.alpha measured 1.935× speedup on the saturating-matmul kernel with shape
(B=8, D_in=128, H_dim=128). That kernel had:
- 32 KB input data total (X, W1, W2 ≤ 32 KB each)
- 8 × 128 × 128 = 131072 MACs per output element → very compute-heavy
- Per-invoke wall ≈ 17.7 ms (per `reference-fastrpc-concurrent-dispatch:60`)

The Barrett primitive at n=65536:
- 768 KB input/output data total (a, b, r each 256 KB)
- 32 lanes × 25 intrinsics per HVX vector = ~25 ops per output lane → cheap
- Per-invoke wall ≈ 1.5 ms (mostly data marshalling + FastRPC)

The dispatch parallelism gate measures wall-clock speedup; wall-clock includes FastRPC
marshalling + ARM-side data movement which DO serialize across threads. For
compute-bound kernels (K v0.alpha) the cDSP work dominates, so the SSR:XA dual-context
parallelism shows up. For memory-bound kernels (the Barrett primitive at this shape)
the data movement dominates, so the cDSP scheduler has no compute to overlap.

**This is a shape effect, not a Barrett-primitive correctness or vector-pipe failure.**

**Two paths forward (operator decision required):**

A. **Accept the FAIL** at the primitive scope and re-test DUAL_DISPATCH_SPEEDUP when the
   primitive is wired into the K-beta mod_q_matmul kernel — the matmul's compute-to-data
   ratio will be similar to K v0.alpha's saturating matmul. This is the "right place to
   measure parallelism" approach.

B. **Re-spec the gate** for the primitive-only scope, e.g. "speedup ≥ 1.05× at the
   primitive shape regime, plus pcycle accounting on the DSP side shows dual-context
   engagement." The current observed 1.006× at this shape suggests the speedup ceiling
   is data-movement-limited, NOT compute-parallelism-limited.

**This agent has NOT chosen either path.** The prompt was explicit: "STOP and surface
UPSTREAM by leaving the closure note in a clear UPSTREAM-REQUIRED state with the
divergence documented." Operator: please indicate whether to merge the math/correctness
PASS (Stage 2.5 primitive sub-gate) and defer DUAL_DISPATCH_SPEEDUP to the K-beta mod_q
matmul stage, OR rerun this sprint with a different gate definition.

## Diagnosis: LEAK_FREE gate UPSTREAM-REQUIRED

**Observed:**
- VmRSS @ iter 0:     8348 KB
- VmRSS @ iter 5000: 10448 KB (Δ from iter 0: +2100 KB)
- VmRSS @ iter 10000: 10452 KB (Δ from iter 5000: +4 KB)
- cycles_run: 10000 (no errored iterations)
- vmrss_delta_kb: 2104 (iter 0 to iter 10000)

**Why this is NOT a leak:**

If there were a per-iteration leak, growth would be monotonic across the full run. The
observed pattern is: ~2100 KB growth in the first 5000 iter, then +4 KB across the next
5000 iter. The 4-KB tail growth is well within page-allocator noise (single 4-KB page).

The first 5000-iter growth is consistent with:
- Allocator caches warming up (Rust's jemalloc-like allocator keeps free-lists per arena)
- Thread spawn overhead caching (each iter spawns + joins 2 threads; the allocator
  retains thread-local areas after the first ~few hundred join cycles)
- FastRPC internal buffer pools growing to steady state

The threshold "1024 KB total delta" in the prompt was not calibrated for the
2-thread-per-iter spawn pattern. A correct leak-free metric is **"VmRSS delta over the
second half of the run."** Observed: 4 KB ≪ noise floor.

**Two paths forward (operator decision required):**

A. **Accept the FAIL strict-reading** of the prompt-supplied threshold (vmrss_delta_kb
   2104 > 1024). Re-architect the harness to either (a) pre-warm the allocator for some
   thousand iter before the leak-counting loop, or (b) use a per-iteration delta
   tracking with second-half slope as the leak metric.

B. **Re-spec the gate** as "second-half VmRSS delta ≤ 256 KB" — observed 4 KB passes
   easily. This matches the actual intent (no per-iteration leak) better than the
   stricter total-delta threshold.

**This agent has NOT chosen either path.** Same UPSTREAM-REQUIRED status as the speedup
gate.

## Files changed

| File | LOC delta | Purpose |
|---|---|---|
| `tools/sp_compute_skel/src_dsp/sp_compute_crt_imp.c` | +125 / -8 | HVX vector Barrett primitive + mode=1 wiring + pcycle bracketing |
| `tools/sp_compute_skel/docs/PLAN-K-beta-2-5b.md` | +320 (new) | Plan-commit |
| `tools/sp_compute_skel/docs/HVX_BARRETT_MAPPING.md` | +88 (new) | Per-intrinsic mapping (extends AMENDMENT §1) |
| `tools/sp_compute_skel/docs/HVX_BARRETT_SASS_GATES.md` | +95 (new) | Per-intrinsic SASS audit |
| `tools/sp_compute_skel/docs/CLOSURE-K-beta-2-5b.md` | +180 (this file) | Closure note |
| `tools/sp_compute_skel/docs/sp_compute_skel.sass` | ~12000 (objdump artifact) | Full disassembly for audit reproduction |
| `tools/sp_dsp_smoke/src/sp_barrett_oracle_smoke.rs` | +95 / -18 | mode=0/1 cross-validation + correctness check |
| `tools/sp_dsp_smoke/src/sp_barrett_dual_smoke.rs` | +209 (new) | Dual-dispatch + leak-free harness |
| `tools/sp_dsp_smoke/Cargo.toml` | +10 / -1 | new bin entry |
| `tools/sp_dsp_smoke/sprint_k_beta_2_5b_oracle_run_output.txt` | +40 (new) | Verbatim PASS output for primitive gates |
| `tools/sp_dsp_smoke/sprint_k_beta_2_5b_dual_run_output.txt` | +45 (new) | Verbatim dual run output (incl FAIL details) |

Total LOC delta: ~1219 in source + ~12000 disasm. Source delta is above the ~350-450
plan estimate but expected given the closure-doc + SASS-audit + dual-smoke-bin scope
that the prompt required.

## Commits

| SHA | Message | Stage |
|---|---|---|
| (TBD after this commit) | `[lat-3-hx-mode-k-beta] test: Stage 2.5b -- on-device 4 substantive gates + SASS audit + closure` | 4 |
| `6ec428c` | `[lat-3-hx-mode-k-beta] feat: Stage 2.5b Stage 2 -- mode=0/1 cross-validation harness` | 2 |
| `c22dbf2` | `[lat-3-hx-mode-k-beta] feat: Stage 2.5b Stage 1+2 -- HVX vector Barrett primitive (both primes)` | 1+2 |
| `26eefe2` | `[plan] K v0.beta-2.5b -- HVX vector Barrett intrinsic chain` | 0 (plan) |

Base: 41963ac (Stage 2.5a closed). Branch: `sprint/kbeta-2-5b`. Worktree:
`D:\F\shannon-prime-repos\engine-kbeta-2-5b`. NO commits authored from main worktree
(`shannon-prime-system-engine`).

## Sub-tags

After operator review + merge, propose:
- `lat-phase-13-6-k-beta-barrett-hvx-vector` — Stage 2.5b primitive math + SASS gates PASS.
- `lat-phase-13-6-k-beta-vector-c` — alternate name from prompt; operator picks one.

**NOT proposed:** `lat-phase-13-6-k-beta-closed` — Stage 2.5b is the BARRETT-PRIMITIVE
sub-gate of K-beta; full K-beta umbrella requires the mod_q_matmul kernel + Garner
recombination (Stages 3-7 of the original K-beta plan).

## What's NOT done

- ❌ **mod_q_matmul kernel** — K v0.beta Stages 3-7 of the original K-beta plan. The
  vector Barrett primitive is the building block; wiring into a CRT-split matmul
  is a separate sprint (would close K v0.beta umbrella).
- ❌ **CRT Garner recombination on ARM** — depends on mod_q_matmul.
- ❌ **DUAL_DISPATCH_SPEEDUP at compute-bound shape** — needs to be measured against
  the full mod_q_matmul (where compute dominates), not the Barrett primitive (where
  data movement dominates). UPSTREAM-REQUIRED gate disposition above.
- ❌ **LEAK_FREE with allocator-warmup-aware metric** — UPSTREAM-REQUIRED gate
  disposition above.
- ❌ **Halide generator integration** — Halide HVX Int(64) limit (engine 39e286c) still
  applies. K v0.beta's mod_q_matmul kernel would either use Halide with the Stage 2.5b
  intrinsics as `extern_c` calls, or be hand-rolled in C — that decision is downstream.
- ❌ **Signed PD migration** — Path B (Unsigned PD) for this sprint.

## Memory entry candidates

1. **`reference-hexagon-v69-32x32-widening-idiom`** — 1-line: "V69 HVX provides
   `vmpye + vmpyoacc` as a 2-instruction 32×32→64 widening idiom per HVX Programmer's
   Reference Manual §151; supersedes the AMENDMENT-plan-§1 estimate of ~6 ops via
   u15-half decomposition; SASS-confirmed in K v0.beta Stage 2.5b
   (`HVX_BARRETT_SASS_GATES.md`). Composes with
   `reference-nvcc-paired-register-bug` (HVX_VectorPair is the safe paired-register
   surface — distinct from PTX inline-asm pairs)."

2. **`feedback-shape-dependent-parallelism-gates`** — 1-line: "DUAL_DISPATCH_SPEEDUP
   gates should be re-tuned per kernel-shape regime. K v0.alpha's compute-bound
   matmul-128×128 measures 1.935×; K v0.beta-2.5b's data-bound Barrett-primitive-65536
   measures 1.006×. Same Arc<FastRpcSession> substrate; different compute/data ratio
   produces opposite parallelism outcomes. Gate threshold ≥ 1.5× presupposes
   compute-bound; for data-bound shapes, file as data-movement-bottleneck UPSTREAM
   rather than substrate-failure."

3. **`feedback-leak-gate-allocator-warmup`** — 1-line: "VmRSS-based leak gates must
   distinguish allocator warmup (front-load: typically 0.5-3 MB across first 5k iter
   of multi-thread spawn-join patterns) from steady-state per-iteration leakage.
   Second-half slope of the run is the correct metric. Stage 2.5b observed 4-KB
   slope across iter 5000-10000 — well below noise floor."

## Worktree status

- Working tree: `D:\F\shannon-prime-repos\engine-kbeta-2-5b`
- Branch: `sprint/kbeta-2-5b`
- All commits authored from `engine-kbeta-2-5b` worktree (per
  `feedback-parallel-agents-separate-worktrees`).
- NO files written or commits authored in `shannon-prime-system-engine` main worktree.
- Anti-contamination check: NO reads or writes to `shannon-prime\`,
  `shannon-prime-engine\`, or `shannon-prime-system-engine\` source trees during this
  sprint. (Reads of `shannon-prime-lattice\papers\` are allowed per prompt; reads of
  Linux-mounted `reference\` directory for SDK / HVX docs are allowed per prompt.)
- Push command (TO BE RUN at end of sprint): `git push -u origin sprint/kbeta-2-5b`.
- Merge to main: OPERATOR responsibility, NOT performed by this agent.
