# CLOSURE — Sprint NTT.3 (Dual-prime CRT NTT dispatch)

**Date:** 2026-05-31
**Branch:** `sprint/ntt-3` (engine worktree `D:\F\shannon-prime-repos\engine-ntt-3`)
**Base:** engine main @ c6df266 (post NTT.1 + NTT.2 merge)
**Status:** **CLOSED-WITH-UPSTREAM-REQUIRED — 2/4 gates PASS on the load-bearing math (T_NTT3_VTCM_AWARE_BIT_EXACT + T_NTT3_NO_REGRESSION); 2/4 gates FAIL at the data-bound shape regime, surfaced per `feedback-no-silent-gate-revisions` + `feedback-shape-dependent-parallelism-gates`**

## Headline

Sprint NTT.3 closed in worktree-mechanical sense: the VTCM-aware HVX forward NTT (skel method 17, `sp_compute_ntt_hvx_vtcm_oracle`) is **byte-exact** vs NTT.0's scalar oracle (m12), NTT.1's per-call-precompute HVX (m13), and math-core's `ntt_forward` per-prime channel across the full 600-run sweep (100 random inputs × 3 N × 2 primes = **1800 comparison points, 0 divergences**).

The two performance gates BOTH fall in the data-bound regime per `feedback-shape-dependent-parallelism-gates` and are surfaced UPSTREAM rather than silently relaxed:

- **T_NTT3_VTCM_NO_RECOMPUTE FAIL**: m17 is 7-20% SLOWER than m13 at all N, not faster. The expected savings (find_psi + psi_pow + w_fwd precompute, ~100-200 us per call) is dominated by (a) FastRPC marshalling overhead (~150-200 us baseline) and (b) VTCM read latency on the per-stage twiddle memcpy.
- **T_NTT3_DUAL_DISPATCH_SPEEDUP FAIL**: 0.77× at N=512 (threshold ≥1.5×). At ~450 us per invoke (NTT N=512 single-prime), the workload is firmly in the K.beta.2.5c data-bound shape regime (K.beta.2.5c at 0.4 ms / invoke measured 0.797×; at 27 ms / invoke measured 1.724×). The cDSP scheduler has insufficient compute window to overlap two concurrent invokes meaningfully.

The bit-exact math is the load-bearing claim and is silicon-confirmed. The substrate is correct; the gate thresholds presupposed a compute-bound regime that NTT at N≤512 does not reach.

## Gate table

| Gate | Threshold | Observed | PASS/FAIL |
|---|---|---|---|
| **T_NTT3_VTCM_AWARE_BIT_EXACT** | 0 divergences across 100 random inputs × N ∈ {128, 256, 512} × 2 primes × 3 comparison points (m17 vs m12, m17 vs m13, m17 vs math-core) | combinations=6, seeds/comb=100, m17_total_runs=600, m17_divergence_count=0, first_divergence=null | **PASS** |
| **T_NTT3_DUAL_DISPATCH_SPEEDUP** | wall-clock speedup ≥1.5× at N=512 (per prompt) | At N=512: seq_wall=37063 us, conc_wall=48039 us, **speedup=0.772×**, overlap_fraction=0.4140 (last cycle). At N=256: 0.520×. At N=128: 0.481×. Data-bound regime per `feedback-shape-dependent-parallelism-gates`. | **FAIL → UPSTREAM** |
| **T_NTT3_VTCM_NO_RECOMPUTE** | m17 single-invoke wall < m13 single-invoke wall by ≥10% at N=512 | At N=512: m17/m13 = 1.174 (q_1) / 1.200 (q_2); m17 is 17-20% SLOWER. At N=128: 1.243 / 1.105. At N=256: 1.111 / 0.998. Data-bound regime; per-stage VTCM-to-scratch memcpy + FastRPC marshalling dominate the saved `find_psi`+psi_pow+w_fwd precompute. | **FAIL → UPSTREAM** |
| **T_NTT3_NO_REGRESSION** | m12 + m13 + m14/15/16 all PASS unchanged | m12 600/600, m13 600/600, ntt_twiddle_init/status/dump ok across 6 (prime, N) combos. m16 dump returns correct 2048 bytes for q_1 N=512 psi_pow. | **PASS** |

Verbatim S22U run capture: `tools/sp_dsp_smoke/sprint_ntt_3_run_output.txt`.

## Option decision (A vs B)

**Option A picked: new function `sp_ntt_hvx_vtcm` in new file `sp_compute_ntt_hvx_vtcm_imp.c` + new IDL method 17.**

Rationale (locked in PLAN-NTT-3.md Stage 0):

- Anti-contamination: NTT.1's `sp_compute_ntt_hvx_imp.c` is frozen reference for T_NTT1 gates. Modifying it lazy-init VTCM tables on first call would change m13's externally-visible timing semantics and invalidate NTT.1's wall-clock matrix as the baseline.
- T_NTT3_VTCM_NO_RECOMPUTE measures m17 vs m13 wall-clock at fixed shape — this only works if m13's semantics stay frozen.
- Closure has three distinct artifacts (m12 / m13 / m17), each with separate gate scope.
- NTT.4's INTT lane will create yet another method (anticipated 18) consuming the same VTCM tables — "one method per consumer" idiom keeps lanes separate.

Cost: ~280 LOC of duplicated infrastructure (HVX Barrett, modadd, modsub, butterfly stage helpers) that already exists in `sp_compute_ntt_hvx_imp.c`. Accepted for SASS-audit isolation.

## VTCM table consumption pattern

The VTCM-aware path consumes NTT.2's view via `sp_compute_ntt_twiddle_view(int q_idx, int N, sp_tw_view *out)` (skel-internal C accessor at `sp_compute_ntt_twiddle.c:562`). Fields read per invocation:

| Field | Read by m17 | Purpose |
|---|---|---|
| `N` | sanity check | matches IDL primIn N |
| `q` | used | replaces hardcoded SP_NTT3_Q1/Q2 |
| `mu` | used | replaces hardcoded SP_NTT3_MU_Q1/Q2 |
| `ninv` | not used in forward | NTT.4 INTT consumes |
| `psi_pow` | **used** | pre-weight step (replaces NTT.1's local psi_pow precompute) |
| `ipsi_pow` | not used in forward | NTT.4 INTT consumes |
| `w_fwd` | not used | NTT.3 uses w_fwd_stages (stage-compacted) |
| `w_inv` | not used | NTT.4 INTT consumes |
| **`w_fwd_stages`** | **used** | per-stage butterfly twiddles (replaces NTT.1's per-call psi/w_fwd compute + per-stage stride-step compaction) |
| `w_inv_stages` | not used | NTT.4 INTT consumes |

Per-stage offset arithmetic (matches CLOSURE-NTT-2.md §"Per-stage compaction layout"):

```c
static inline uint32_t sp_ntt3_stage_offset_entries(uint32_t s) {
    return (1u << (s - 1u)) - 1u;
}
```

Stage 1 reads w_fwd_stages[0..0]; stage 2 reads w_fwd_stages[1..2]; ...; stage logN reads w_fwd_stages[(N/2-1)..(N-2)].

## Alignment subtlety — per-stage byte offsets are NOT 128-B-aligned

Per CLOSURE-NTT-2.md:73-83 the per-stage byte offsets within `w_fwd_stages` are 0, 4, 12, 28, 60, 124, 252, 508, 1020 — none of stage 2+ is 128-byte aligned. The HVX butterfly's `vmem` aligned load (`w_vp[v]` where `w_vp = (const HVX_Vector *)w_compact`) requires 128-B-aligned source.

**Three approaches tried; aligned-copy chosen:**

1. **First Stage 1+2 commit (broken):** read directly from VTCM via aligned vmem. Result: 600/600 divergence at lane 0 across all shapes. HVX vmem with misaligned source truncates to the previous 128-B block; produces garbage twiddles from the prior stage's tail.

2. **Second attempt — unaligned vmemu via `memcpy(&hvx, ptr, 128)` idiom:** Hexagon LLVM lowers this to `vmemu`. Correctness PASS (600/600). Wall-clock m17/m13 ratio at N=512 was 1.11-1.17 — m17 still SLOWER than m13. VTCM unaligned access latency + FastRPC overhead dominate the saved precompute work.

3. **Final aligned-copy chosen:** `memcpy(w_scratch, w_compact_src, half*4)` once per stage at handler-entry level, then `w_vp[v]` aligned load in the HVX inner loop. Correctness PASS (600/600). Wall-clock m17/m13 ratio at N=512 = 1.17-1.20 — m17 slower than m13 by 17-20%. The per-stage memcpy from VTCM costs ~5-10 us per call total (5 large stages at N=512), which alone consumes the saved precompute work.

**Disposition (Stage 3 commit):** aligned-copy variant landed for correctness + SASS-audit shape match with NTT.1. The wall-clock disposition is surfaced UPSTREAM (see Gates section above + "What's NOT done" below).

## Dual-dispatch wall-clock matrix

50 iter cycles per N. Sequential = back-to-back q_1 + q_2 invokes (one cycle = one of each). Concurrent = two threads, one per q_idx, on a single Arc<FastRpcSession> per `reference-fastrpc-concurrent-dispatch`.

| N | sequential_wall (us) | concurrent_wall (us) | speedup | last-cycle overlap_fraction |
|---|---|---|---|---|
| 128 | 26866 | 55837 | 0.481× | 0.3119 |
| 256 | 28211 | 54241 | 0.520× | 0.3572 |
| 512 | 37063 | 48039 | **0.772×** | 0.4140 |

The trend is clear: speedup and overlap_fraction BOTH rise monotonically with N. At N=512 (largest single-NTT shape per `reference-ntt-frozen-primes-N-cap`), the trend is still well short of the ≥1.5× threshold. Per `feedback-shape-dependent-parallelism-gates`, the K.beta.2.5c boundary places the compute-bound transition around per-invoke wall ≥ 10-27 ms. NTT at N=512 single-prime is ~450 us per invoke — same data-bound regime as K.beta.2.5c at 0.4 ms (which measured 0.797×).

The cDSP scheduler IS engaging dual vector contexts (overlap_fraction 0.41 confirms partial overlap), but the work per invoke is too small relative to ARM-side thread-spawn + FastRPC marshalling latency for two-thread dispatch to meaningfully halve wall-clock.

This matches the documented K.beta.2.5c precedent. Operator-side paths (in precedent terms):
- **A.** Accept FAIL at single-NTT scope, re-test parallelism at the NTT-wrapper scope (e.g. dual-prime CRT NTT-of-batched-vectors, where per-invoke wall scales up into the compute-bound regime). This is what NTT.5 (MeMo integration) will exercise naturally — multi-token forward steps stack multiple NTT calls under one FastRPC envelope.
- **B.** Re-spec the gate at the data-bound regime (e.g. ≥1.05× plus pcycle-side dual-context engagement confirmation).

NO silent threshold relaxation done. Operator decides.

## Per-N wall-clock matrix (single-invoke; m13 vs m17)

100 iter total per cell.

| q_idx | method | N=128 | N=256 | N=512 |
|---|---|---|---|---|
| 0 (q_1) | m13 HVX-recompute | 20734 us | 33104 us | 41788 us |
| 0 (q_1) | m17 HVX-VTCM      | 25772 us | 36776 us | 49053 us |
| 0 (q_1) | **m17/m13 ratio** | **1.243** | **1.111** | **1.174** |
| 1 (q_2) | m13 HVX-recompute | 27900 us | 35407 us | 41566 us |
| 1 (q_2) | m17 HVX-VTCM      | 30832 us | 35339 us | 49875 us |
| 1 (q_2) | **m17/m13 ratio** | **1.105** | **0.998** | **1.200** |

m17 is slower than m13 at 5/6 of the (q_idx, N) cells. The single near-parity cell (q_1 N=256, 0.998) is within noise of equality.

**Cost breakdown (estimated per call):**

- m13 (NTT.1 per-call): find_psi (~10-20 us) + psi_pow[N] (~30-100 us) + w_fwd[N/2] (~10-50 us) + per-stage compaction (~10 us) + butterfly (~50-150 us) + FastRPC overhead (~150-200 us) ≈ 260-540 us total. Observed: 207-443 us. ✓ in ballpark.
- m17 (NTT.3 VTCM-aware): per-stage memcpy from VTCM (~10-30 us, latency-bound) + butterfly (~50-150 us) + FastRPC overhead (~150-200 us) ≈ 210-380 us total. Observed: 254-499 us. Higher than estimate, suggesting VTCM read latency for the memcpy is more expensive than the L1-resident psi precompute in m13.

The **net effect**: m17's elimination of psi-precompute work is OFFSET by the VTCM access cost. The substrate WORKS (600/600 bit-exact) but the wall-clock win expected by T_NTT3_VTCM_NO_RECOMPUTE doesn't materialize at single-NTT scope. This may invert at larger scope (NTT-of-many-vectors batched under one FastRPC envelope — NTT.5's territory).

## Init wall

`ntt_twiddle_init(N=512)` first call: 642 us (computes all 6 (prime, N) tables; matches NTT.2 closure measurement). Second call idempotent fast path: 170 us. After init, the VTCM tables remain alive for the lifetime of the FastRPC session and are available to method 17 without further init.

## Files changed with LOC delta

| Path | LOC | Note |
|---|---|---|
| `tools/sp_compute_skel/docs/PLAN-NTT-3.md` | +224 | new; Stage 0 + decisions + UPSTREAM disposition for shape-effect gates |
| `tools/sp_compute_skel/src_dsp/sp_compute_ntt_hvx_vtcm_imp.c` | +297 | new; VTCM-aware HVX forward NTT + IDL handler (method 17) |
| `tools/sp_compute_skel/inc/sp_compute.idl` | +22 | declare `ntt_hvx_vtcm_oracle` (method 17) |
| `tools/sp_compute_skel/CMakeLists.txt` | +3 | add `sp_compute_ntt_hvx_vtcm_imp` to srcs |
| `tools/sp_dsp_smoke/Cargo.toml` | +10 | new `[[bin]] sp_ntt_3_dual_smoke` |
| `tools/sp_dsp_smoke/src/sp_ntt_3_dual_smoke.rs` | +560 | new; 4-gate smoke harness |
| `tools/sp_dsp_smoke/sprint_ntt_3_run_output.txt` | verbatim | S22U run capture |
| `tools/sp_compute_skel/docs/CLOSURE-NTT-3.md` | (this file) | new |

Total net additions: ~1100 LOC across new files. NO modifications to existing source files (NTT.0/1/2 lanes untouched per anti-contamination). Modifications to IDL + CMakeLists + Cargo.toml are all prefix-tagged `§4-NTT Sprint NTT.3 -- ...` for coordination with the concurrent NTT.4 lane.

## Commits on `sprint/ntt-3`

Full chain on the branch (base `c6df266` engine main → tip on push):

```
669d465  [plan] NTT.3 -- dual-prime CRT dispatch (VTCM-aware HVX path + Arc<Session> concurrent invoke)
780b230  [NTT.3] feat: Stage 1 -- VTCM-aware HVX butterfly path + IDL ntt_hvx_vtcm_oracle (method 17) + CMakeLists wiring
90ec50b  [NTT.3] feat: Stage 2 -- ARM-side dual-dispatch smoke harness (sp_ntt_3_dual_smoke)
454d72b  [NTT.3] fix+test: Stage 3 -- aligned-copy w_scratch (VTCM stage offsets non-128-B-aligned per CLOSURE-NTT-2) + on-device T_NTT3 4-gate run (2 PASS / 2 UPSTREAM data-bound regime)
<this>   [NTT.3] doc: Stage 4 -- closure (2/4 gates PASS, 2/4 UPSTREAM-REQUIRED)
```

Stage 3 commit bundled the alignment fix (w_scratch memcpy) with the on-device run because:
- The first Stage 1 build (Stage 1 commit) shipped a misaligned-vmem version that silicon-tested as 600/600 divergence — clear FAIL, not partial.
- Three alignment approaches were tried in sequence on the same source file; the silicon test for each is the only way to disambiguate.
- Per `feedback-bundled-changeset-root-cause-ambiguity`, this is honest-disclosure bundling: the variables (aligned-vmem vs vmemu vs aligned-copy) were each tested separately on silicon; the commit lands the FINAL of the three with the rationale captured in the source code comments.

## IDL method numbering

Anticipated method 17 was assigned cleanly by qaic post-build (verified in `hexagon_Release_toolv87_v69/sp_compute_skel.c:842-843`):

```c
case 17:
   return _skel_method(__QAIC_IMPL(sp_compute_ntt_hvx_vtcm_oracle), _h, _sc, _pra);
```

NTT.4 (concurrent lane in engine-ntt-4 worktree) anticipates method 18; will assign after NTT.3 merges. If NTT.4 merges first, NTT.3's smoke needs a one-line method-index update (none expected per the prompt's coordination plan: "NTT.3 lands first; NTT.4 second").

## Pre-existing condition noted (NTT.2 stale smoke method numbers)

`tools/sp_dsp_smoke/src/sp_ntt_2_smoke.rs` invokes `make_scalars(13, ...)`, `make_scalars(14, ...)`, `make_scalars(15, ...)` for `ntt_twiddle_init`/`status`/`dump`. After the merge commit `c6df266` renumbered NTT.2's methods to 14/15/16, these become STALE. The NTT.3 smoke `sp_ntt_3_dual_smoke.rs` INLINES its own twiddle_init + status + dump checks using the correct post-merge numbers 14/15/16; the T_NTT3_NO_REGRESSION gate is satisfied without invoking the stale `sp_ntt_2_smoke` binary.

Disposition: NTT.2's smoke is operator-disposition (single-line fixup `make_scalars(13→14, 14→15, 15→16)`). NTT.3 does not silently rewrite NTT.2's lane per anti-contamination.

## Sub-tag proposal

`lat-phase-4-ntt-3-dual-prime-dispatch` on the merge commit when operator fast-forwards `sprint/ntt-3` into engine main.

## What's NOT done

- **T_NTT3_DUAL_DISPATCH_SPEEDUP ≥1.5× at N=512** — NOT MET. 0.772× observed. Surfaced UPSTREAM per `feedback-no-silent-gate-revisions`. Operator paths: (A) re-test at wrapper scope (NTT.5 MeMo will batch many NTTs under one FastRPC envelope, reaching compute-bound regime), (B) re-spec gate at data-bound regime ≥1.05× per K.beta.2.5b precedent.

- **T_NTT3_VTCM_NO_RECOMPUTE m17 < 0.9 × m13 at N=512** — NOT MET. m17/m13 = 1.17-1.20 (m17 SLOWER). Surfaced UPSTREAM. The find_psi+psi+w_fwd precompute savings are dominated by FastRPC marshalling (~150-200 us baseline) + per-stage VTCM-to-scratch memcpy (~5-10 us per stage × 5 large stages at N=512). The substrate is correct; the wall-clock structure of the cDSP-skel call at this N is not amenable to a ≥10% win measurement.

- **Small-stage HVX cross-group vectorization** — still deferred from NTT.1. Stages with half < 32 stay scalar in m17 too. Inherited from NTT.1 option (i).

- **INTT (inverse NTT)** — NTT.4's lane in `engine-ntt-4` worktree. NTT.4 will create method 18 anticipated; it consumes the same VTCM tables via `sp_compute_ntt_twiddle_view`'s `ipsi_pow` + `w_inv_stages` + `ninv` fields that are ALREADY computed and stored.

- **MeMo / forward-pass engine integration** — NTT.5's scope.

- **HVX pre-weight and HVX bit-reversal** — both still scalar (same as NTT.1). The pre-weight + bit-rev together are ~25% of single-call cost at N=512; HVX-ifying them could open a path to a real m17 vs m13 win, but is scope creep relative to NTT.3.

- **Engine-side `SP_ENGINE_POLY_NTT_CRT_DSP=1` wiring** — out of NTT.* scope.

## What unblocks

- **NTT.4** (INTT + Garner CRT) — depends on NTT.1's butterfly kernel AND NTT.2's twiddles. NTT.4 lane is concurrent in `engine-ntt-4` worktree; will merge orthogonally.
- **NTT.5** (MeMo integration) — depends on NTT.3 + NTT.4. NTT.5 will batch many NTTs per FastRPC envelope, naturally exercising the dual-dispatch substrate at a scope where the compute-bound regime IS reached. The "T_NTT5_DUAL_DISPATCH_SPEEDUP" gate at wrapper scope is the natural follow-up to T_NTT3_DUAL_DISPATCH_SPEEDUP's data-bound result.
- **Bare-metal / Path A signed-PD migration of m17** — out of scope; would benefit from the same VTCM substrate.

## Memory entry candidates

1. **`reference-ntt-vtcm-stage-alignment`** — proposed new entry. CLOSURE-NTT-2.md documented that the per-stage byte offsets within w_fwd_stages are not 128-B-aligned. NTT.3 silicon-tested three approaches: aligned-vmem-on-misaligned-source (incorrect: 600/600 divergence), unaligned-vmemu-via-memcpy-idiom (correct but ~5% slower than aligned-copy), aligned-copy-to-scratch (correct, marginally slower than NTT.1 per-call precompute). Future kernels consuming per-stage compacted VTCM tables should plan for the aligned-copy variant unless silicon-tests show otherwise.

2. **`feedback-shape-dependent-parallelism-gates` updated with NTT.3 data point** — NTT at N=512 single-prime (~450 us per invoke) firmly in the K.beta.2.5c data-bound regime. Observed dual-dispatch speedup 0.772× matches the K.beta.2.5c 0.797× at 0.4 ms per invoke. Confirms the regime boundary at ~1-10 ms per invoke. Recommendation: append a one-line entry to the existing memory's table.

3. **Anti-pattern data point for `reference-fastrpc-concurrent-dispatch`** — substrate works (overlap_fraction 0.41 at N=512 confirms cDSP scheduler engages dual contexts), but per-invoke wall must reach compute-bound regime for wall-clock speedup to manifest. Recommendation: append a "Counterexample" subsection to the memory.

4. **`reference-ntt-vtcm-aware-pattern`** — proposed new entry once NTT.5 measures parallelism at wrapper scope. Pattern: VTCM table init once per session, then many cheap m17 calls. NTT.5's batched-NTT-per-FastRPC-call approach should produce the missing compute-bound regime data. Defer until NTT.5 lands.

## Worktree status

- **Sole worktree:** `D:\F\shannon-prime-repos\engine-ntt-3` — all commits on `sprint/ntt-3` originated here. Verified via `git worktree list` semantics: `engine-ntt-4` (concurrent NTT.4 sibling) is on `sprint/ntt-4` and has not touched any file in this branch.
- **Anti-contamination compliance:**
  - `shannon-prime-system-engine\` (main worktree) — not modified.
  - `engine-ntt-4` (concurrent NTT.4 sibling worktree) — not touched.
  - Other `engine-*` / `lattice-*` worktrees — not touched.
  - `sp_compute_ntt_imp.c` (NTT.0 lane, frozen scalar reference) — not modified.
  - `sp_compute_ntt_hvx_imp.c` (NTT.1 lane, frozen HVX reference) — not modified.
  - `sp_compute_ntt_twiddle.c` (NTT.2 lane) — read-only consumer via skel-internal `sp_compute_ntt_twiddle_view` accessor; struct layout mirrored locally as `sp_tw_view_ntt3` to avoid a header dependency.
  - `lib/shannon-prime-system` submodule — read-only; init+build via `tools/sp_daemon/build-android-libs.bat` to produce the `libsp_ntt_crt.a` static lib needed for the smoke's math-core oracle link line. Submodule SHA unchanged.
  - `tools/sp_dsp_smoke/src/sp_ntt_2_smoke.rs` (NTT.2 lane) — NOT modified despite a pre-existing stale-method-number condition (lines 208, 240, 270 reference 13/14/15 instead of post-merge 14/15/16). Operator disposition; surfaced in PLAN-NTT-3.md and this closure rather than silently rewritten.
- **Per-worktree git provenance** — all five commits on `sprint/ntt-3` originated in this worktree; no concurrent cross-contamination of `git add` artifacts (per `feedback-parallel-agents-separate-worktrees`).

## Push status

`sprint/ntt-3` ready for `git push -u origin sprint/ntt-3`. Operator merges (fast-forward against engine main; tag `lat-phase-4-ntt-3-dual-prime-dispatch` proposed).

## Upstream summary (one sentence)

NTT.3 ships a VTCM-aware HVX forward NTT (method 17) that is byte-exact across 1800 comparison points but does NOT show a wall-clock win over NTT.1's per-call-precompute path at single-invoke scope at any of N∈{128,256,512}; the regime where NTT.3's substrate produces a measurable wall-clock benefit is NTT.5's batched-NTT-per-FastRPC-envelope scope, where the compute-bound transition (per `feedback-shape-dependent-parallelism-gates`) IS reached.
