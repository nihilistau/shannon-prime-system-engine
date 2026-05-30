# CLOSURE — Sprint NTT.1 (HVX-vectorized NTT butterfly core)

**Date:** 2026-05-30
**Branch:** `sprint/ntt-1` (engine worktree `D:\F\shannon-prime-repos\engine-ntt-1`)
**Base:** engine main @ f834bff (NTT.0 closed)
**Status:** **CLOSED — all 4 substantive gates PASS on Knack's S22U**

## Headline

Sprint NTT.1 CLOSED. The HVX-vectorized NTT butterfly (skel method
13, `sp_compute_ntt_hvx_oracle`) is byte-exact against math-core's
canonical C reference AND against NTT.0's scalar Hexagon oracle
across the full 6 combinations × 100 random seeds = **600 runs**.
On Knack's S22 Ultra (V69 cDSP, Path B Unsigned PD), the HVX path
ships at 0.860–0.946× wall-clock vs the NTT.0 scalar floor across
all 3 N × 2 primes, with the largest wins at N=128 (0.860–0.877×)
and a meaningful **win at N=512 for both primes** (0.946× q_1,
0.876× q_2). The SASS audit confirms the steady-state inner loop is
a 7-packet software-pipelined VLIW body emitting **32 HVX intrinsics
+ 3 hoisted splats** with **zero divergences** from the planned
opcode table. NTT.0 (skel method 12) PASSes 600/600 unchanged.

NTT.2 (twiddle VTCM staging) and NTT.3 (dual-prime concurrent
dispatch) are now both unblocked on the validated HVX-vectorized
butterfly floor.

## Gate table

| Gate | Threshold | Observed | PASS/FAIL |
|---|---|---|---|
| **T_NTT1_HVX_BIT_EXACT** | 0 divergences across 100 random inputs × N ∈ {128, 256, 512} × 2 primes, ntt_hvx_oracle vs ntt_oracle AND vs math-core ntt_forward per-prime | combinations=6, seeds/comb=100, total_runs=600, divergence_count=0, max_diff_per_prime={q_1: 0, q_2: 0}, max_diff_per_N={128: 0, 256: 0, 512: 0}, first_divergence=null | **PASS** |
| **T_NTT1_SASS_AUDIT** | every emitted HVX intrinsic in the steady-state inner loop produces the planned V69 opcode | total intrinsics in inner loop=32 (+3 hoisted splats), divergences=0; SASS doc: `tools/sp_compute_skel/docs/HVX_NTT_SASS_GATES.md` | **PASS** |
| **T_NTT1_WALL_CLOCK_WIN** | HVX path runs faster than scalar path at N=512 (largest shape, most HVX benefit) | At N=512: m13/m12 ratio = 0.946 (q_1) and 0.876 (q_2) — HVX wins at both primes. Full matrix below. | **PASS** |
| **T_NTT1_NO_REGRESSION** | ntt_oracle (method 12, NTT.0 scalar) re-runs 600/600 PASS unchanged | combinations=6, total_runs=600, divergence_count=0, first_divergence=null | **PASS** |

Verbatim S22U run capture: `tools/sp_dsp_smoke/sprint_ntt_1_run_output.txt`.

## Small-stage decision (option (i) — scalar fallback)

Per `PLAN-NTT-1.md` §"Small-stage scalar fallback (half < 32)":
stages with `half < 32` (i.e. `len ∈ {2, 4, 8, 16, 32}` for N=512 →
5 stages; `len ∈ {2, 4, 8, 16}` for N=256 → 4 stages; `len ∈ {2, 4,
8, 16}` for N=128 → 4 stages, since half=32 falls in the HVX path
when len=64) use the byte-identical NTT.0 scalar butterfly inner
body.

**Rationale:** option (ii) full cross-group HVX vectorization with
shuffles is feasible but requires non-trivial register packing for
groups whose interleave doesn't align to 32-lane boundaries.
Shipping option (i) keeps NTT.1's scope contained and yields
demonstrable wall-clock wins (above) on the strength of the large
stages alone. NTT.4 or NTT.5 can lift small-stages to HVX
cross-group vectorization once the foundation is silicon-proven —
which it now is.

**This is NOT a silent gate revision per `feedback-no-silent-gate-
revisions`:** the prompt itself recommended option (i) as the v1
default, and the PLAN-NTT-1 commit named the decision explicitly
before any implementation. Option (ii) is filed as deferred
follow-on, not "discovered to be hard during implementation."

## Per-stage twiddle compaction (placeholder for NTT.2)

Each large stage (half ≥ 32) compacts its stride-`step` twiddles into
a stride-1 scratch array `w_compact[half]` via a scalar pass:

```c
for (uint32_t k = 0; k < half; k++) {
    w_compact[k] = w_fwd[k * step];
}
```

This is **inline in the NTT.1 kernel** (see
`sp_compute_ntt_hvx_imp.c::ntt_forward_one_hvx` lines 292-298).
`w_compact` is a 128-B-aligned stack-resident scratch
(`uint32_t w_compact[SP_NTT_N_MAX / 2] __attribute__((aligned(128)))`,
1024 B at N=512).

**NTT.2's lane**: lift this to VTCM-resident precomputed per-stage
tables (i.e. one allocation per N × prime, all stages laid out
back-to-back). The kernel pointer to `w_compact` is the only thing
that changes; the inner-loop SASS is unchanged.

## SASS audit summary

**File:** `tools/sp_compute_skel/docs/HVX_NTT_SASS_GATES.md` —
per-intrinsic table mirroring `HVX_BARRETT_SASS_GATES.md`.

Function `ntt_forward_one_hvx` at `0x8030`. Steady-state inner loop
at `0x8248..0x82c8` — 7-packet software-pipelined `loop0` body.
All three `static inline` helpers
(`sp_barrett_reduce32_hvx_lane_ntt1`, `sp_modadd_hvx_lane_ntt1`,
`sp_modsub_hvx_lane_ntt1`) inlined into the butterfly stage; butterfly
stage in turn inlined into `ntt_forward_one_hvx`.

| Metric | Value |
|---|---|
| Inner-loop HVX intrinsics emitted | **32** |
| Hoisted splats (vq, vq_m1, vmu) | **3** |
| Divergences from planned opcode | **0** |
| `vmpye + vmpyoacc` widening pairs | **3** (Barrett a×b, sh×mu, qhat×q) |
| `loop0` packets | 7 |
| VLIW co-issue patterns observed | vcmp+vsub (Barrett correction); vadd+vsub+vcmp (modadd/modsub); vmux+vmux+vmem.new (stores fold into compute) |

## Wall-clock matrix (100 iters, microseconds total)

| q_idx | method | N=128 | N=256 | N=512 |
|---|---|---|---|---|
| 0 (q_1) | m12 scalar | 33055 us | 34834 us | 45721 us |
| 0 (q_1) | m13 HVX    | 28417 us | 32926 us | 43235 us |
| 0 (q_1) | **m13/m12 ratio** | **0.860** | **0.945** | **0.946** |
| 1 (q_2) | m12 scalar | 30458 us | 34639 us | 48014 us |
| 1 (q_2) | m13 HVX    | 26720 us | 31904 us | 42056 us |
| 1 (q_2) | **m13/m12 ratio** | **0.877** | **0.921** | **0.876** |

Per-iter (HVX path): ~267-284 us at N=128, ~319-329 us at N=256,
~420-432 us at N=512.

Note that wall-clock is dominated by FastRPC marshalling overhead (~150
us per invoke baseline observed in K.beta.2.5b smoke runs). The
kernel-internal speedup is much larger than the wall-clock ratio
suggests; the apparent 0.876-0.946× ratio reflects amortization of a
mostly-fixed marshalling cost over a variable-size compute body. NTT.2
(VTCM-staged twiddles, eliminates per-call `find_psi` + compaction
overhead) and NTT.3 (concurrent dispatch, halves marshalling cost per
useful kernel call) will both improve this metric — the
butterfly-internal SASS density (this sprint's audit) is the silicon
upper bound.

## Files changed with LOC delta

| Path | LOC | Note |
|---|---|---|
| `tools/sp_compute_skel/docs/PLAN-NTT-1.md` | +398 | new file; Stage 0 reference reading + 4-stage plan |
| `tools/sp_compute_skel/src_dsp/sp_compute_ntt_hvx_imp.c` | +345 | new file; HVX butterfly + IDL handler |
| `tools/sp_compute_skel/inc/sp_compute.idl` | +18 | ntt_hvx_oracle method 13 declaration |
| `tools/sp_compute_skel/CMakeLists.txt` | +2 | add sp_compute_ntt_hvx_imp to srcs |
| `tools/sp_dsp_smoke/src/sp_ntt_1_smoke.rs` | +334 | new file; 3-gate smoke harness |
| `tools/sp_dsp_smoke/Cargo.toml` | +9 | sp_ntt_1_smoke [[bin]] entry |
| `tools/sp_dsp_smoke/sprint_ntt_1_run_output.txt` | +76 | verbatim S22U run capture |
| `tools/sp_compute_skel/docs/HVX_NTT_SASS_GATES.md` | +131 | new file; per-intrinsic SASS audit |
| `tools/sp_compute_skel/docs/CLOSURE-NTT-1.md` | (this file) | new file; closure |

Total net additions: ~1310 LOC across new files. NO modifications to
existing files except the IDL + CMakeLists + Cargo.toml additions
(all three are prefix-tagged for coordination with NTT.2). NO
modifications to math-core sources (read-only reference). NO
modifications to `sp_compute_ntt_imp.c` (NTT.0 lane, frozen scalar
reference).

## Commits on `sprint/ntt-1`

Full chain on the branch (base `f834bff` engine main → tip on push):

```
58b13d7  [plan] NTT.1 -- HVX butterfly core (large-stage vectorization, small-stage scalar fallback, per-stage compacted twiddles)
302868d  [NTT.1] feat: Stages 1+2 -- HVX butterfly + IDL ntt_hvx_oracle (method 13) + CMakeLists wiring
f52b6b6  [NTT.1] test: Stage 3 -- on-device 3-gate smoke (600/600 HVX bit-exact, 600/600 NTT.0 no-regression, HVX wall-clock win at N=512 both primes)
<this>   [NTT.1] doc: Stage 4 -- closure + HVX NTT SASS audit (T_NTT1_SASS_AUDIT)
```

**Stages 1 and 2 bundled into one commit** with explicit disclosure
per `feedback-bundled-changeset-root-cause-ambiguity`: the IDL
declaration, handler impl, and CMakeLists wiring naturally land
together because splitting them produces an intermediate
non-buildable state; the bundled variables are all on the "wire-up"
path, no functional one-variable-at-a-time question to disentangle.
The functional gate is Stage 3 on-device correctness — which the
single Stage 1+2 commit can be examined against in isolation.

## Sub-tag proposal

`lat-phase-4-ntt-1-hvx-butterfly` on the merge commit when operator
fast-forwards `sprint/ntt-1` into engine main.

## What's NOT done

- **Option (ii) small-stage HVX cross-group vectorization** — the
  half < 32 stages still run scalar. The cross-group shuffles
  required for HVX vectorization of those stages add complexity
  without changing the architectural picture. Filed for NTT.4 or
  NTT.5 follow-on.
- **VTCM-resident precomputed twiddle tables** — NTT.2's lane.
  Currently `psi_pow`, `w_fwd`, and the per-stage `w_compact` are
  stack-resident scratch (~3 KB total at N=512), reset per call.
- **Dual-prime concurrent dispatch** — NTT.3's lane. The skel
  handler takes one `q_idx` at a time; NTT.3 wraps two
  `Arc<FastRpcSession>` threads each invoking `ntt_hvx_oracle` with
  different `q_idx`.
- **INTT (inverse NTT)** — NTT.4's lane. The kernel here only does
  forward; inverse reuses the same butterfly with `w_inv[]` swapped
  for `w_fwd[]` plus the post-weight + `ninv` scale.
- **HVX pre-weight and HVX bit-reversal** — both are O(N) and small
  relative to the O(N log N) butterfly work; the wall-clock
  measurement above shows the HVX wins entirely from the butterfly
  body, not from these steps. NTT.4 or NTT.5 can vectorize them if
  measurement shows they dominate.
- **CRT recombination (Garner)** — NTT.4's lane (post-INTT).

## What unblocks

- **NTT.2** (twiddle VTCM staging) — already independently in
  progress in `engine-ntt-2`; will merge cleanly with NTT.1 since
  the two sprints don't share source files except IDL + CMakeLists
  + Cargo.toml, where prefix discipline (`§4-NTT Sprint NTT.X --`)
  was maintained on both sides.
- **NTT.3** (dual-prime concurrent dispatch) — needs **both** NTT.1
  and NTT.2 closed; depends on:
  - NTT.1's HVX kernel (this sprint — DONE)
  - NTT.2's VTCM-resident twiddles (so two concurrent invocations
    don't redundantly recompute per-call twiddles)
- **NTT.4** (INTT + Garner CRT) — depends on NTT.1's butterfly
  kernel (this sprint) and NTT.2's twiddle staging. Reuses
  `sp_ntt_butterfly_stage_hvx` with `w_inv[]` swapped in.

## Memory entry candidates

1. **`reference-hexagon-v69-32x32-widening-idiom` is now extended with
   NTT.1 silicon confirmation** — the K.beta.2.5b primitive's 2-op
   widening idiom remains silicon-correct when called inside a
   higher-level kernel (NTT.1 butterfly + modadd + modsub chain).
   Three widening pairs per Barrett invocation, all clean.
   Recommendation: append a one-line note to the existing memory's
   "How this was confirmed" section pointing to NTT.1.

2. **`reference-ntt-1-hvx-butterfly-pattern`** — proposed new entry.
   Documents the pattern for future HVX kernels that need:
   - A 32-lane modular-add primitive (`vadd + vcmp.gt + vsub + vmux`)
   - A 32-lane modular-sub primitive (`vsub + vcmp.gt(b, a) + vadd +
     vmux`)
   - Per-stage twiddle compaction (stride-N to stride-1 for vmem
     loads)
   Plus the SASS-confirmed observation that the compiler will pack
   modadd + modsub into 2-3 VLIW packets when both are in the same
   inner loop. Suggest filing when operator triages.

3. **NTT-internal observation about FastRPC marshalling
   amortization** — the wall-clock matrix above shows that HVX
   speedups are partially masked by ~150 us / invoke FastRPC
   overhead. This is a generalizable insight for future per-prime,
   per-tensor, per-tile cDSP kernels. Could compose into the
   existing `feedback-shape-dependent-parallelism-gates` memory's
   "data-bound vs compute-bound" section. Not load-bearing enough
   on its own to warrant a separate entry; suggest appending.

## Worktree status

- **Sole worktree:** `D:\F\shannon-prime-repos\engine-ntt-1` — all
  commits on `sprint/ntt-1` originated here. Verified via
  `git worktree list` semantics: `engine-ntt-2` (concurrent NTT.2
  sibling) is on `sprint/ntt-2` and has not touched any file in
  this branch.
- **Anti-contamination compliance:**
  - `shannon-prime-system-engine\` (main worktree) — not modified.
  - Other `engine-*` worktrees (`engine-ntt-2` parallel sibling,
    `engine-ntt-0` NTT.0 source, all other engine-*) — not touched.
  - `shannon-prime-system\core\ntt_crt\` (math-core) — READ-ONLY
    reference; submodule not even checked out in this worktree
    (NTT.0's worktree was used for cross-reference reading).
  - `sp_compute_ntt_imp.c` (NTT.0 lane, frozen scalar reference) —
    not modified.
  - Anticipated NTT.2 file `sp_compute_ntt_twiddle.c` — never
    created or referenced.
  - `models\` artifacts — not accessed.
  - `papers\PPT-LAT-Roadmap.md` — not modified.
- **Per-worktree git provenance** — all four commits on
  `sprint/ntt-1` originated in this worktree; no concurrent
  cross-contamination of `git add` artifacts (per
  `feedback-parallel-agents-separate-worktrees`).
