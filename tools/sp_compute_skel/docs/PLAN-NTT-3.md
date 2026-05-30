# PLAN — Sprint NTT.3 (Dual-prime CRT NTT dispatch)

**Branch:** `sprint/ntt-3` (worktree `D:\F\shannon-prime-repos\engine-ntt-3`)
**Base:** engine main @ c6df266 (post NTT.1 + NTT.2 merge)
**Concurrent sibling:** NTT.4 (INTT + Garner) in `engine-ntt-4` worktree

## Stage 0 — Mandatory reference reading (with file:line citations)

1. **NTT.1 closure** — `tools/sp_compute_skel/docs/CLOSURE-NTT-1.md:42-83`,
   per-stage compaction placeholder for NTT.2 documented at lines 62-83. HVX
   butterfly signature `ntt_forward_one_hvx` at `sp_compute_ntt_hvx_imp.c:281-328`,
   IDL handler `sp_compute_ntt_hvx_oracle` at lines 341-399.

2. **NTT.2 closure** — `tools/sp_compute_skel/docs/CLOSURE-NTT-2.md:43-91`,
   VTCM layout matrix at lines 50-57. `psi_pow_off`, `w_fwd_off`,
   `w_fwd_stages_off` are the canonical sub-table offsets within each arena.

3. **NTT.2 skel implementation** — `tools/sp_compute_skel/src_dsp/sp_compute_ntt_twiddle.c:539-583`
   defines the skel-internal C accessor `sp_compute_ntt_twiddle_view(int q_idx, int N, sp_tw_view *out)`
   returning a struct with all six sub-table pointers (`psi_pow`, `ipsi_pow`,
   `w_fwd`, `w_inv`, `w_fwd_stages`, `w_inv_stages`) plus `N`, `q`, `mu`, `ninv`.
   NOT exposed via IDL — pure C-internal API designed for NTT.1/NTT.4 consumers.

4. **K.beta.2.5c dual-dispatch reference** — `tools/sp_dsp_smoke/src/sp_matmul_q_dual_smoke.rs:49`
   (`let sess = Arc::new(sess)`), lines 154-167 (clone + spawn pattern with
   `sess_a = sess.clone()`, `sess_b = sess.clone()`, two threads invoking the
   same method with different `q_idx`), lines 178-186 (overlap_fraction
   computation: `ta0.max(tb0)` vs `ta1.min(tb1)`).

5. **`reference-fastrpc-concurrent-dispatch`** — `Arc<FastRpcSession>` is the
   substrate: "wrap in Arc, NOT Mutex, for concurrent dispatch. Multiple ARM
   threads can call `sess.invoke(&self, ...)` on ONE Arc<FastRpcSession>
   concurrently."  Parallelism discriminator MUST be wall-clock, not pcycle
   ratio.

6. **`reference-dual-model-cdsp-scheduler`** — Trick #1 is kernel-agnostic.
   "Two concurrent Arc<FastRpcSession> invokes from ARM-side threads ... Both
   kernels target HVX vector contexts ... Per-invoke compute is large enough
   that wall-clock dominates marshalling (compute-bound regime)."

7. **`feedback-shape-dependent-parallelism-gates`** — empirical regime
   boundary at K.beta.2.5c: ~0.4 ms / invoke @ B=8 D=128 D_out=128 → 0.797×
   speedup (data-bound); ~27 ms / invoke @ B=8 D=1024 D_out=512 → 1.724×
   speedup (compute-bound). NTT.1's per-invoke wall at N=512 = 420-432 µs
   (per CLOSURE-NTT-1.md:115-118) is much closer to the data-bound K.beta.2.5c
   regime (0.4 ms) than the compute-bound regime (27 ms).

8. **`feedback-no-silent-gate-revisions`** — surface UPSTREAM before relaxing
   a gate. **`feedback-lead-with-reference-then-theory`** — read the reference
   code first; this plan-commit cites file:line for every reference I touch.

## Option decision — A vs B

**Option A (RECOMMENDED, picked):** new function `sp_ntt_hvx_vtcm` in new
file `sp_compute_ntt_hvx_vtcm_imp.c`. Reads VTCM-resident tables via
`sp_compute_ntt_twiddle_view`. New IDL method `ntt_hvx_vtcm_oracle`
(anticipated method 17) routes to it.

**Rationale (vs Option B which would modify `ntt_hvx_oracle` method 13
in-place):**

- **Anti-contamination** — NTT.1's `sp_compute_ntt_hvx_imp.c` is frozen
  reference for the T_NTT1 gates. Modifying it lazy-init VTCM tables on
  first call changes the function's externally-visible timing semantics and
  invalidates NTT.1's wall-clock matrix as a baseline. Option A keeps NTT.1
  intact.
- **NTT.3 gate measures VTCM-aware variant explicitly** — T_NTT3_VTCM_NO_RECOMPUTE
  compares method 17 (VTCM-aware) vs method 13 (per-call recompute) wall-clock
  at fixed shape. This only works cleanly if method 13's semantics don't
  change.
- **Closure semantics are clean** — three distinct artifacts to validate
  (m12 scalar / m13 HVX-recompute / m17 HVX-VTCM); each one has a separate
  gate scope.
- **NTT.4 (INTT) lane parallel** — NTT.4 will create yet another method
  (anticipated 18) consuming the same VTCM-resident `w_inv_stages`. The
  "one method per consumer" idiom keeps lanes separate.

**Cost:** ~280 LOC of duplicated infrastructure (HVX Barrett, modadd,
modsub, butterfly stage helpers) that already exists in
`sp_compute_ntt_hvx_imp.c`. Acceptable for SASS-audit isolation and
correctness gate isolation.

## IDL method anticipation

| Method | Handler | Lane | Status |
|---|---|---|---|
| 12 | `sp_compute_ntt_oracle` | NTT.0 | landed |
| 13 | `sp_compute_ntt_hvx_oracle` | NTT.1 | landed |
| 14 | `sp_compute_ntt_twiddle_init` | NTT.2 | landed |
| 15 | `sp_compute_ntt_twiddle_status` | NTT.2 | landed |
| 16 | `sp_compute_ntt_twiddle_dump` | NTT.2 | landed |
| **17** | **`sp_compute_ntt_hvx_vtcm_oracle`** | **NTT.3 (this sprint)** | **anticipated** |
| 18 | (anticipated) `sp_compute_ntt_intt_oracle` | NTT.4 (concurrent) | NTT.4's lane |

NTT.3 prefix-tags all IDL + CMakeLists + Cargo.toml additions with
`§4-NTT Sprint NTT.3 -- dual-prime CRT dispatch (method 17)` per the
prefix-discipline rule in the prompt. NTT.4 will use method 18 with its own
prefix. If renumbering is required at merge time, the agent who lands second
renumbers; this lane lands first.

## VTCM table consumption pattern

The VTCM-aware `sp_ntt_hvx_vtcm` reads, per `(q_idx, N)` invocation:

```
sp_tw_view v;
if (sp_compute_ntt_twiddle_view(q_idx, N, &v) != 0) return -1;
// v.psi_pow      → pre-weight table, stride 1, N × u32
// v.w_fwd_stages → compacted per-stage forward twiddles, stride 1,
//                  (N-1) × u32, layout per CLOSURE-NTT-2.md stage-major
// v.q, v.mu, v.ninv → modular primitives (ninv unused in forward path)
```

Pre-weight step uses `v.psi_pow[j]` directly (eliminates the per-call
`find_psi` + `psi_pow` precompute loop in NTT.1 lines 364-376).

Per-stage butterfly uses `v.w_fwd_stages` with explicit stage-offset
arithmetic per CLOSURE-NTT-2.md:71-83:

```
offset_stage[s]  = (1 << (s-1)) - 1   entries from compacted region base
size_stage[s]    = (1 << (s-1))       entries
```

This eliminates the per-call stride-step → stride-1 compaction loop in
NTT.1 lines 318-320 (which would have to run on every NTT call).

Bit-reversal permutation stays scalar (data-dependent indices; no VTCM
benefit). Stages with `half < 32` stay scalar (option (i) inherited from
NTT.1).

## ARM-side dual-dispatch pattern

Per `reference-fastrpc-concurrent-dispatch` + K.beta.2.5c idiom:

```rust
let sess = Arc::new(FastRpcSession::new(URI)?);
// Prime VTCM tables once.
invoke_twiddle_init(&sess, 512)?;

// Sequential baseline.
let seq_start = Instant::now();
let (out_q1, _) = invoke_ntt_hvx_vtcm(&sess, 0, N, &input)?;
let (out_q2, _) = invoke_ntt_hvx_vtcm(&sess, 1, N, &input)?;
let seq_wall = seq_start.elapsed();

// Concurrent dispatch.
let sess_a = sess.clone();
let sess_b = sess.clone();
let conc_start = Instant::now();
let h_a = std::thread::spawn(move || invoke_ntt_hvx_vtcm(&sess_a, 0, N, &input_a));
let h_b = std::thread::spawn(move || invoke_ntt_hvx_vtcm(&sess_b, 1, N, &input_b));
let r_a = h_a.join()?;
let r_b = h_b.join()?;
let conc_wall = conc_start.elapsed();
```

Overlap fraction = `(min(ta_end, tb_end) - max(ta_start, tb_start)) / conc_wall`
per K.beta.2.5c sp_matmul_q_dual_smoke.rs:178-186.

## Gate threshold notes

**T_NTT3_DUAL_DISPATCH_SPEEDUP threshold:** prompt specifies ≥1.5× at N=512.
Per `feedback-shape-dependent-parallelism-gates`, NTT at N=512 per-invoke wall
~420-450 µs falls in the data-bound regime by K.beta.2.5c's empirical
boundary (0.4 ms = data-bound 0.797×; 27 ms = compute-bound 1.724×). The
VTCM-aware variant will be FASTER per invoke than the recompute variant
(eliminating find_psi + psi_pow + w_fwd + compaction; that's the point of
T_NTT3_VTCM_NO_RECOMPUTE), which pushes the parallelism gate further into
data-bound territory.

**Disposition per `feedback-no-silent-gate-revisions`:** I run the gate
exactly as specified. If ≥1.5× is met, PASS. If <1.5×, the closure REPORTS
the observed number, the per-N matrix, and surfaces UPSTREAM as
"T_NTT3_DUAL_DISPATCH_SPEEDUP fell in data-bound regime per
feedback-shape-dependent-parallelism-gates — operator path A or B as in
K.beta.2.5b/c precedent." NO silent threshold relaxation. NO closure-claim
of victory based on a different metric.

## Plan-commit deliverable

This file (`PLAN-NTT-3.md`) + 4-stage commit chain on `sprint/ntt-3`:

- Stage 0 (this commit): plan, references, decision, gate disposition.
- Stage 1: VTCM-aware HVX butterfly (new file `sp_compute_ntt_hvx_vtcm_imp.c`)
  + IDL method 17 + CMakeLists wiring + single-invoke correctness in smoke.
- Stage 2: ARM-side dual-dispatch smoke harness `sp_ntt_3_dual_smoke.rs`.
- Stage 3: on-device 4-gate runs (correctness + speedup + no-recompute +
  no-regression).
- Stage 4: closure `CLOSURE-NTT-3.md`.

Push at end. Operator merges.

## Co-tenancy with NTT.4

NTT.4 in `engine-ntt-4` worktree owns the INTT kernel and Garner CRT
recombiner. Per the prompt:

- I anticipate method 17 in IDL + Cargo.toml prefix tags.
- NTT.4 anticipates method 18.
- Predictable conflict points: IDL declaration order + skel CMakeLists src
  list + Cargo.toml `[[bin]]` order. All prefix-tagged for trivial conflict
  resolution at merge.

NTT.4 must NOT modify any file in `sp_compute_ntt_hvx_vtcm_imp.c` (NTT.3's
new file) or NTT.3's smoke binary. NTT.3 must NOT touch
`sp_compute_ntt_imp.c` (NTT.0 frozen), `sp_compute_ntt_hvx_imp.c` (NTT.1
frozen), or `sp_compute_ntt_twiddle.c` (NTT.2's lane; consumed read-only via
the view API).

## Pre-existing condition to surface

`tools/sp_dsp_smoke/src/sp_ntt_2_smoke.rs` was authored against IDL methods
13/14/15 (NTT.2's pre-merge expectation per CLOSURE-NTT-2.md:184-187),
but the merge commit `c6df266` renumbered NTT.2's methods to 14/15/16 to
avoid colliding with NTT.1's method 13. The smoke file's `make_scalars(13, ...)`,
`make_scalars(14, ...)`, `make_scalars(15, ...)` lines are STALE post-merge.
This blocks the T_NTT3_NO_REGRESSION check via the existing NTT.2 smoke binary.

**Disposition:** the NTT.3 smoke binary `sp_ntt_3_dual_smoke.rs` will
INLINE its own twiddle_init + status sanity check for the no-regression gate,
NOT depend on running the stale `sp_ntt_2_smoke`. The stale smoke is flagged
in CLOSURE-NTT-3.md as operator-disposition material; not silently rewritten
since it's NTT.2's lane.
