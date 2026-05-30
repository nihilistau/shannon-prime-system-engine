# CLOSURE — Sprint NTT.0 (Scalar Hexagon NTT, Reference Port)

**Date:** 2026-05-30
**Branch:** `sprint/ntt-0` (engine worktree `D:\F\shannon-prime-repos\engine-ntt-0`)
**Base:** engine main @ 833abbe (ledger-autowire closed)
**Status:** **CLOSED — T_NTT0_SCALAR_BIT_EXACT 600/600 PASS at N ∈ {128, 256, 512}**

## Headline

Sprint NTT.0 CLOSED. `T_NTT0_SCALAR_BIT_EXACT` passes 600/600 on Knack's
S22 Ultra (V69 cDSP, Path B Unsigned PD), 0 divergences across 100
random inputs × N ∈ {128, 256, 512} × 2 frozen primes (q_1, q_2). The
on-Hexagon scalar NTT oracle (`sp_compute_ntt_oracle`, skel method 12)
is byte-exact against math-core's canonical C reference
`ntt_forward`. NTT.1 (HVX butterfly) and NTT.2 (twiddle VTCM staging)
are now parallel-dispatchable on the validated scalar floor.

This closure supersedes the prior UPSTREAM-REQUIRED closure (commit
`6238980` on this branch) per operator-side Path A disposition. See
**Recovery note** at the end.

## Gate table

| Gate | Threshold | Observed | PASS/FAIL |
|---|---|---|---|
| **T_NTT0_SCALAR_BIT_EXACT** | 0 divergences across 100 random inputs × N ∈ {128, 256, 512} × 2 primes vs math-core `ntt_forward` per-prime output | combinations=6, seeds/comb=100, total_runs=600, divergence_count=0, max_diff_per_prime={q_1:0, q_2:0}, max_diff_per_N={128:0, 256:0, 512:0}, first_divergence=null, wall_time=0.40 s | **PASS** |

Full numeric report: `tools/sp_daemon/scripts/ntt_0_full_report.json`.
Verbatim run capture:  `tools/sp_daemon/scripts/ntt_0_full_run.txt`.

## Architectural commitments compliance

Phase 4-NTT block (lattice `papers/PPT-LAT-Roadmap.md`) — commitments
relevant to NTT.0:

- [x] **#1 (math-core canonical NTT is the reference)** — Hexagon
      scalar oracle is byte-exact against `ntt_forward` per-prime
      output across all 600 runs. Same Cooley-Tukey radix-2 DIT
      negacyclic algorithm; same `psi^j` pre-weight; same Barrett
      primitive; same bit-reversal step.
- [x] **#2 (FROZEN primes)** — both `q_1 = 1073738753`,
      `μ_1 = 1073744895` and `q_2 = 1073732609`, `μ_2 = 1073751039`
      embedded as compile-time constants in
      `sp_compute_ntt_imp.c:32-35`. NOT modified.
- [x] **#3 (no 128-bit arithmetic on Hexagon)** — the scalar oracle
      uses `uint64_t` end-to-end, same Barrett shift constants
      (`(Q_BITS-1)=29`, `(Q_BITS+1)=31`) as math-core and the K.beta
      cDSP path. Verified at source.
- [x] **#5 (N ladder)** — implemented exactly per operator Path A:
      N ∈ {128, 256, 512}. Larger N rejected at the IDL boundary
      with `return -1`, matching math-core's `ntt_init` ABI.
- [x] **#6 (single-prime per-channel pipe)** — skel takes one
      `q_idx`, computes ONE residue channel. CRT combination is
      caller's responsibility (matches math-core's per-prime
      `forward_one` design, not its dual-prime `ntt_forward` wrapper).
- [x] **#7 (scalar-first, vector follow-up)** — NO HVX intrinsics
      in `sp_compute_ntt_imp.c`. Plain scalar pipe. HVX butterfly is
      NTT.1 scope.

Commitments #4 (CRT residues), #8 (long-context tiling) are
out-of-scope for NTT.0; deferred to NTT.3 / NTT.5 / NTT.6.

## Math-core reference signature

```c
// lib/shannon-prime-system/include/sp/ntt_crt.h
ntt_ctx *ntt_init(uint32_t N);                                      // N ∈ {128, 256, 512}
void     ntt_free(ntt_ctx *ctx);
void     ntt_forward(const ntt_ctx *ctx, const int32_t *in,
                     uint32_t *out1, uint32_t *out2);               // dual-prime output
```

Internal per-prime path (matched byte-for-byte by NTT.0 Hexagon
scalar):

```c
// lib/shannon-prime-system/core/ntt_crt/ntt_crt.c:259-272
static void forward_one(const ntt_ctx *ctx, const prime_ctx *pc,
                        const int32_t *in, uint32_t *out);
```

`out1[]` = output of `forward_one` with `pc = &ctx->p1` (mod q_1).
`out2[]` = output of `forward_one` with `pc = &ctx->p2` (mod q_2).

## Hexagon implementation signature

```c
// tools/sp_compute_skel/src_dsp/sp_compute_ntt_imp.c
int sp_compute_ntt_oracle(remote_handle64 h,
                          int q_idx, int N,
                          const unsigned char *data_in,  int data_inLen,
                          unsigned char       *data_out, int data_outLen);
```

`q_idx == 0` selects (q_1, μ_1); `q_idx == 1` selects (q_2, μ_2).
`N` ∈ {128, 256, 512}; any other value returns -1. `data_in` is N
signed i32 LE (arbitrary range — modular reduction is done inside,
matching math-core's `forward_one:267-268` contract). `data_out` is
N u32 LE in `[0, q)`.

Twiddle policy: `psi`, `psi_pow[N]`, `w_fwd[N/2]` computed per-call
from `find_psi(N, q)` into stack-resident scratch (≤ 3 KB at N=512).
Matches math-core's `prime_setup` algorithm. NTT.2 lifts these to
VTCM-resident precomputed tables.

## IDL method signature

```idl
// tools/sp_compute_skel/inc/sp_compute.idl
long ntt_oracle(in long q_idx, in long N,
                in  sequence<octet> data_in,
                rout sequence<octet> data_out);
```

Method index 12 (verified at
`hexagon_Release_toolv87_v69/sp_compute_skel.c::sp_compute_skel_invoke`
case 12 → `_skel_method(__QAIC_IMPL(sp_compute_ntt_oracle), ...)`).
Scalars MAKEX: method=12, inbufs=2, outbufs=1, inhandles=0,
outhandles=0.

`primIn` layout (16 B): `[q_idx i32, N i32, data_inLen i32,
data_outLen i32]`.

## Files changed

| Path | LOC | Note |
|---|---|---|
| `tools/sp_compute_skel/docs/PLAN-NTT-0.md` | +39 | `[plan-amend]` block documenting Path A disposition + Stage 1 reduction |
| `tools/sp_compute_skel/src_dsp/sp_compute_ntt_imp.c` | +203 | new file; scalar Hexagon NTT oracle |
| `tools/sp_compute_skel/inc/sp_compute.idl` | +24 | `ntt_oracle` method 12 |
| `tools/sp_compute_skel/CMakeLists.txt` | +1 | add `sp_compute_ntt_imp` to srcs |
| `tools/sp_dsp_smoke/build.rs` | +43 | new file; links math-core `sp_ntt_crt` for android |
| `tools/sp_dsp_smoke/Cargo.toml` | +9 | `sp_ntt_0_smoke` `[[bin]]` entry |
| `tools/sp_dsp_smoke/src/sp_ntt_0_smoke.rs` | +198 | new file; T_NTT0_SCALAR_BIT_EXACT smoke harness |
| `tools/sp_daemon/scripts/ntt_0_full_report.json` | +1 | machine-parseable gate result |
| `tools/sp_daemon/scripts/ntt_0_full_run.txt` | +40 | verbatim S22U run capture |
| `tools/sp_compute_skel/docs/CLOSURE-NTT-0.md` | (rewrite) | this file; supersedes UPSTREAM-REQUIRED |

## Commits on `sprint/ntt-0`

Full chain on the branch (base `833abbe` engine main → tip on push):

```
8143fe5  [plan] NTT.0 - scalar Hexagon NTT (Cooley-Tukey radix-2 DIT, negacyclic, q_1+q_2) - UPSTREAM-REQUIRED on N ladder
6238980  [NTT.0] stage 0 closure - UPSTREAM-REQUIRED on N ladder
0c7ddaa  [plan-amend] NTT.0 -- Path A selected; Stage 1 reduced to math-core C FFI (drop Rust port)
0e841a7  [NTT.0] feat: Stage 2 -- scalar Hexagon NTT + IDL ntt_oracle (N in {128, 256, 512}, q_1 + q_2)
4f27ff9  [NTT.0] test: Stage 3 -- T_NTT0_SCALAR_BIT_EXACT on S22U (600/600 PASS, ...)
<this>   [NTT.0] doc: Stage 4 -- closure supersedes UPSTREAM-REQUIRED; T_NTT0 PASS after operator Path A disposition
```

The first two commits (`8143fe5`, `6238980`) were the prior agent's
honest UPSTREAM-REQUIRED stop. They are PRESERVED on this branch as
historical context, NOT rewritten — per
`feedback-bundled-changeset-root-cause-ambiguity` (and the rewind-
discipline lesson from M.4: never silently overwrite a prior agent's
work).

## Sub-tag proposal

`lat-phase-4-ntt-0-scalar-hexagon` on the merge commit when operator
fast-forwards `sprint/ntt-0` into engine main.

## What's NOT done

- **NTT.1 (HVX butterfly)** — vectorize the radix-2 DIT inner loop.
  Reuses `sp_barrett_reduce32_hvx_lane` from K.beta.2.5b
  (silicon-confirmed). Substrate is the validated scalar floor in
  this sprint; gate is HVX vs scalar bitwise.
- **NTT.2 (twiddle VTCM staging)** — lift `psi_pow[]` + `w_fwd[]`
  out of per-call stack scratch into VTCM-resident precomputed
  tables. Gate is wall-clock improvement at fixed correctness.
- **NTT.3 (dual-prime concurrent dispatch)** — the scalar oracle
  takes one `q_idx` at a time; NTT.3 composes two
  `Arc<FastRpcSession>` threads for q_1 and q_2 in parallel per
  `reference-fastrpc-concurrent-dispatch` + Trick #1 of
  `reference-heterogeneous-soc-crt-tricks`.
- **NTT.4 (INTT + Garner CRT)** — inverse transform + 60-bit
  recombine. Math-core has `ntt_inverse` + `ntt_crt_recombine`
  already; the Hexagon scalar port mirrors the same pattern.
- **NTT.5 / NTT.6 (poly-ring attention + long-context tiling)** —
  the payload sprints. Reframed in the corrected Phase 4-NTT block
  to TILED N=512 transforms rather than monolithic N=ctx.

## What unblocks

NTT.1 and NTT.2 are now parallel-dispatchable. Both require:
- a separate engine worktree per agent (per
  `feedback-parallel-agents-separate-worktrees`)
- the scalar oracle in this sprint as the bitwise correctness anchor
- math-core's `ntt_forward` per-prime channel as the cross-validation
  reference (same FFI binding path established in
  `tools/sp_dsp_smoke/build.rs`).

NTT.3 unlocks once at least one of {NTT.1, NTT.2} ships — concurrent
dispatch only makes wall-clock sense once the kernel itself is fast
enough that marshalling doesn't dominate (see
`feedback-shape-dependent-parallelism-gates`).

## Memory entry candidates

1. **`reference-ntt-frozen-primes-N-cap`** (already written by
   operator as part of the e927f6f roadmap correction). Confirmed
   silicon-true: math-core `ntt_init(N=1024)` returns NULL; only
   N ∈ {128, 256, 512} construct a context. Operator already locked
   this; nothing to add.

2. **`reference-math-core-ntt-twiddle-ordering`** — proposed new
   entry. NTT.1 + NTT.2 will need to know:
   - `psi_pow[j]` is `psi^j` for `j ∈ [0, N)` (forward pre-weight)
   - `w_fwd[j]` is `omega^j` for `j ∈ [0, N/2)` where `omega = psi^2`
   - Bit-reversal permutation runs on `out[]` AFTER pre-weight,
     BEFORE the logN butterfly stages (math-core
     `ntt_crt.c:236-239` + `ntt_core` step ordering).
   - `widx` indexing in the butterfly inner loop uses
     `step = N / len` stride; widx accumulates within an `(i, len)`
     block (math-core lines 244-252). NTT.1 HVX path must match the
     same widx walk or the gate fails.
   Single 200-char index line + a 2-page detail doc; suggest filing
   when the operator triages.

3. **`reference-mathcore-ffi-from-sp-dsp-smoke`** — proposed.
   `sp_dsp_smoke/build.rs` now establishes the FFI pattern: link
   math-core static libs (`sp_ntt_crt` for NTT.0) into the
   aarch64-android cross-build via the same `engine_root /
   build-android-libs / core / <module>` convention as
   `sp_daemon/build.rs`. NTT.4 + NTT.5 will reuse this build-path
   resolution. If math-core grows transitive deps in a future
   sprint, this build.rs needs a MODULES table like sp_daemon's.

## Worktree status

- **Sole worktree:** `D:\F\shannon-prime-repos\engine-ntt-0` — all
  commits on `sprint/ntt-0` originated here. Verified via
  `git worktree list` semantics: no other concurrent agent touched
  this branch.
- **Anti-contamination compliance:**
  - `shannon-prime-system-engine\` (main worktree) — not modified.
  - Other `engine-*` / `lattice-*` worktrees — not touched.
  - `shannon-prime-system\core\ntt_crt\` (math-core) — READ-ONLY
    reference; never modified.
  - `models\` artifacts — not accessed by this sprint.
  - `papers\PPT-LAT-Roadmap.md` — NOT modified. Operator owns the
    roadmap; the corrected N ladder landed via operator commit
    `e927f6f` on lattice main, NOT by this agent.

## Recovery note

This closure was written by a continuation agent after the prior
agent surfaced UPSTREAM-REQUIRED on the N ladder during Stage 0
reference reading. The prior agent honored
`feedback-no-silent-gate-revisions` and stopped rather than silently
shipping at the (impossible) N ∈ {256, 1024, 4096} ladder.

**Operator disposition:** Path A. Lattice main `e927f6f`
("[Phase 4-NTT block CORRECTED + NTT.0 UPSTREAM-REQUIRED resolved
Path A]") corrected the roadmap to N ∈ {128, 256, 512} and reframed
long context as TILED N=512 NTT blocks. Two memory entries were
written by the operator alongside the roadmap correction:
`reference-ntt-frozen-primes-N-cap` and
`feedback-lead-with-reference-then-theory` (updated).

The continuation agent (this closure's author):
1. Re-read `PLAN-NTT-0.md` to confirm the Stage 1-4 conditional plan
   already targets N ∈ {128, 256, 512} (it does — math-core's ABI
   was already the source of truth).
2. Filed a small `[plan-amend]` block adjusting Stage 1 from "Rust
   port + ref-self-check" to "math-core C FFI directly via build.rs
   static-lib link" — closer to the continuation prompt's intent
   and removes a redundant sub-gate.
3. Executed Stages 2, 3, 4 in the per-stage commit cadence (no
   bundled changesets; one variable per stage).
4. PASS reported here on `sprint/ntt-0` tip; ready for operator
   review + merge.

**The prior agent's UPSTREAM-REQUIRED commit (`6238980`) is preserved
on the branch.** This was the right call — without that stop, the
sprint would have shipped silently at an architecturally-impossible
gate, exactly the failure mode `feedback-no-silent-gate-revisions`
exists to prevent.
