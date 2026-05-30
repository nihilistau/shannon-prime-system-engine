# CLOSURE-NTT-4.md — §4-NTT Sprint NTT.4 — INTT + ARM Garner round-trip

## Headline

NTT.4 closes the polynomial-multiplication round-trip end-to-end byte-exact
vs math-core on Knack's S22U cDSP via Path B (Unsigned PD). HVX-vectorized
INTT consumes NTT.2's VTCM-resident `w_inv_stages` / `ipsi_pow` tables;
ARM-side signed Garner CRT recombines per-prime INTT outputs into the
signed centered polynomial product mod `M = q_1 * q_2`. All three substantive
gates PASS first-class with HVX path enabled.

Worktree: `D:\F\shannon-prime-repos\engine-ntt-4` on `sprint/ntt-4`.
Base: engine main @ c6df266 (post-NTT.2 merge).
Tip:  b2dabb9.

## Gates table

| Gate | Methodology | Pass criteria | Observed | Verdict |
|------|-------------|--------------|----------|---------|
| T_NTT4_INTT_BIT_EXACT | INTT(NTT(x)) ?= x mod q per prime; 3 N × 2 primes × 100 seeds; on-device method 17 vs Rust-impl math-core port | 0 divergences in 600 runs | divergence_count=0; max_diff_per_prime=q1:0 q2:0; elapsed=0.63 s | **PASS** |
| T_NTT4_GARNER_SIGNED_BIT_EXACT | `garner_combine_q1_q2_signed` vs centering formula on 1000 random pairs + 5 boundary cases (0, 1, M/2, M/2+1, M-1) | 0 divergences; output ∈ (-M/2, M/2]; both halves covered | 1005 pairs, 0 divergences, both halves covered; full test suite 5/5 ok | **PASS** |
| T_NTT4_POLY_MUL_EXACT | device fwd → ARM Barrett pointwise → device INTT → ARM signed Garner; vs math-core `ntt_init + ntt_forward × 2 + ntt_pointwise_mul + ntt_inverse`; 3 N × 4 seeds | 0 divergences in 12 runs | divergence_count=0 at N=128/256/512 (4 seeds each); elapsed=0.04 s | **PASS** |

## Butterfly kernel reuse — design choice + rationale

PLAN-NTT-4 considered extracting NTT.1's `sp_ntt_butterfly_stage_hvx` into a
shared header (`sp_compute_ntt_hvx_shared.h`) so both forward and INTT TUs
could `#include` it. **Final design diverged**: NTT.1's TU was left strictly
untouched (anti-contamination), and NTT.4 received a duplicated copy of the
HVX butterfly + helpers under intt-specific names (`sp_barrett_reduce32_hvx_lane_intt`,
`sp_modadd_hvx_lane_intt`, `sp_modsub_hvx_lane_intt`,
`sp_intt_butterfly_stage_hvx`, `sp_intt_butterfly_stage_scalar`). ~80 LOC
duplicated; consciously chosen over header extraction because:

- HVX_NTT_SASS_GATES.md audits NTT.1's TU specifically; sharing the
  butterfly across TUs would require re-auditing both consumers and the
  shared header.
- The duplicate compiles byte-identically (same Q6 intrinsics, same Barrett
  primitive math) — verified by `hexagon-nm` on the resulting `.so`.
- Future Sprint NTT.5+ can promote to a shared header in one focused
  refactor if a third consumer emerges; deferring now keeps the
  bundled-changeset penalty zero.

`w_inv_stages` drop-in worked AS DESIGNED. Per-stage offset arithmetic
(`stage_off_entries(s) = 2^(s-1) - 1` entries from base of `w_inv_stages`)
matches NTT.2's compaction layout (sp_compute_ntt_twiddle.c:226-235). NTT.4's
butterfly reads `w_inv_stages + stage_off` directly from VTCM via
`HVX_UVector*` (unaligned vmemu) — necessary because the stage-base offsets
are 4-byte aligned but NOT 128-byte aligned (e.g. `w_inv_stages_off = 8188`
within the N=512 arena, and stage 6 lands at byte 8312 = 8188 + 31*4 from
arena base, which is 64.94*128). The misalignment-load bug was caught by
the Stage 1 round-trip gate failing 600/600 the first time with
`HVX_Vector*` cast; fixed in the same commit by switching to `HVX_UVector*`.

## Signed Garner design — centering + test

The math-core `garner_one` (ntt_crt.c:303-317) returns a signed centered
`int64_t` via:

```c
uint64_t r = (uint64_t)x1 + (uint64_t)q1 * (uint64_t)t;
int64_t v = (int64_t)r;
if (v > M / 2) v -= M;
return v;
```

NTT.4 mirrors this in Rust as `garner_combine_q1_q2_signed`:

```rust
let r: u64 = (a as u64) + (q1 as u64) * t;
let mut v: i64 = r as i64;
if v > half_m { v -= m as i64; }
v
```

K.beta.2.5c's existing `garner_combine_q1_q2` (returns `Vec<u64>` in `[0, M)`)
is UNTOUCHED. The signed sibling adds 19 LOC of post-cast centering plus
an in-source `#[cfg(test)]` unit test that drives 1005 pairs (1000 random +
5 corner cases at 0, 1, M/2, M/2+1, M-1) through both `garner_combine_q1_q2`
(unsigned reference) and `garner_combine_q1_q2_signed`, asserts the
centering relationship, asserts output range `(-M/2, M/2]`, and asserts both
halves of the output range are covered. Full module test suite ran:
`test result: ok. 5 passed; 0 failed` (4 pre-existing + 1 new).

End-to-end Stage 3 transitively confirms byte-exact match vs math-core's
`ntt_crt_recombine` (which calls `garner_one`) by composition: if signed
Garner deviated from math-core's, the polymul end-to-end output would
differ from math-core's `ntt_inverse`. It doesn't.

## Wall-clock benchmark (informational)

N=512 polynomial multiplication round-trip via FastRPC, 100 iters:

```
total wall = 265.25 ms
per iter   = 2652.5 us
```

Composition per iter:
- 4× forward NTT (method 13) — ARM→cDSP FastRPC dispatch + HVX butterfly
- 2× pointwise multiply — ARM-side Barrett scalar
- 2× INTT (method 17) — ARM→cDSP FastRPC dispatch + HVX butterfly + scale + post-pass
- 1× ARM-side signed Garner — pure ARM loop

8 FastRPC calls dominate; per-call ~330 us. Includes marshalling +
session-shared scheduler dispatch. NTT.5 (MeMo integration) will batch
multiple polymul-shaped operations via Arc<FastRpcSession> dual-thread
dispatch (reference-fastrpc-concurrent-dispatch) which should reduce per-poly
wall by ~2× per Trick #1 silicon-confirmed precedent.

This is **NOT** a benchmark gate (per feedback-shape-dependent-parallelism-
gates; correctness-only at Stage 3). Reported for future-sprint planning
context.

## Files changed with LOC delta

(vs c6df266 base)

| File | Action | LOC |
|------|--------|-----|
| tools/sp_compute_skel/CMakeLists.txt | MOD | +3 |
| tools/sp_compute_skel/docs/PLAN-NTT-4.md | NEW | +217 |
| tools/sp_compute_skel/docs/CLOSURE-NTT-4.md | NEW (this file) | (closure) |
| tools/sp_compute_skel/inc/sp_compute.idl | MOD | +32 |
| tools/sp_compute_skel/src_dsp/sp_compute_ntt_intt_imp.c | NEW | +350 |
| tools/sp_daemon/scripts/ntt_4_intt_run.txt | NEW | +41 |
| tools/sp_daemon/scripts/ntt_4_polymul_run.txt | NEW | +58 |
| tools/sp_dsp_smoke/Cargo.toml | MOD | +19 |
| tools/sp_dsp_smoke/src/sp_matmul_q_ref.rs | MOD | +108 |
| tools/sp_dsp_smoke/src/sp_ntt_4_intt_smoke.rs | NEW | +468 |
| tools/sp_dsp_smoke/src/sp_ntt_4_polymul_smoke.rs | NEW | +351 |
| **Total** | | **+1647** |

## Commits on `sprint/ntt-4`

```
97503f7  [plan] NTT.4 — INTT + ARM signed Garner round-trip (HVX butterfly reuse via w_inv_stages)
83376a9  [NTT.4] feat: Stage 1 -- HVX INTT skel + IDL method 17 (T_NTT4_INTT_BIT_EXACT 600/600 PASS)
1936d0c  [NTT.4] feat: Stage 2 -- ARM-side signed Garner sibling (T_NTT4_GARNER_SIGNED_BIT_EXACT PASS)
b2dabb9  [NTT.4] test: Stage 3 -- end-to-end polynomial multiplication smoke on S22U (T_NTT4_POLY_MUL_EXACT 12/12 PASS)
<this>   [NTT.4] doc: Stage 4 -- closure (3/3 gates PASS)
```

## Sub-tag

`lat-phase-4-ntt-4-intt-garner` (recommended; operator tags at merge).

## IDL method assignment + NTT.3 coordination

NTT.4's `intt_hvx_oracle` lands at **qaic case 17** in this worktree
(verified via the generated `sp_compute_skel.c` dispatcher in
`hexagon_Release_toolv87_v69/`):

```c
case 17:
return _skel_method(__QAIC_IMPL(sp_compute_intt_hvx_oracle), _h, _sc, _pra);
```

NTT.3's worktree (separate, concurrent) anticipates **method 17** for
`ntt_hvx_vtcm_oracle`. **Predictable IDL method collision.**

Resolution at merge time: the **second-landing lane renumbers**. Conventional
expectation (per sprint spec) is NTT.3 lands first, NTT.4 bumps to method 18.
If NTT.4 lands first, NTT.3 bumps to method 18. The IDL doc comment in
sp_compute.idl explicitly marks `intt_hvx_oracle` as "method 18 — anticipated
slot" to telegraph intent for whoever rebases.

The smoke harnesses (`sp_ntt_4_intt_smoke.rs` and `sp_ntt_4_polymul_smoke.rs`)
hard-code `INTT_METHOD = 17` to match THIS worktree's emitted dispatcher.
At merge time, if renumbered to 18, this constant flips in two places
(grep for `INTT_METHOD`).

## What's NOT done (deferred)

- **NTT.5 MeMo integration** — wire forward NTT (NTT.1+2+3) + INTT+Garner
  (this sprint) into MeMo's attention path. Natural next sprint;
  unblocked by NTT.4's polymul primitive.
- **NTT.6 long-context benchmark** — tiled N=512 NTTs for ctx ≥ 1024
  per `reference-ntt-frozen-primes-N-cap`. Asymptotic O(N log N) decoupling
  via tiling; requires NTT.5 wiring first.
- **HVX vectorization of INTT post-pass (Step 4)** — `out[j] *= ninv *
  ipsi_pow[j] mod q` is currently scalar (N modmul pairs per call).
  At N=512, that's ~1024 Barrett reductions in the scalar pipe;
  could be vectorized identically to NTT.1's pre-weight. Deferred per
  feedback-bundled-changeset (load-bearing gate is round-trip correctness,
  not post-pass µs). Estimated 15-20% wall-time win at N=512.
- **Small-stage HVX** — stages with `half < 32` (stages 1-5 at N=128;
  1-5 at N=256; 1-5 at N=512) use scalar Barrett. These stages are
  ~1.5-2% of total butterfly cost at N=512 (since half is small) so
  HVX-izing them gives small wins. NTT.1 made the same trade-off; NTT.4
  inherits it.
- **Path B → Path A migration / Signed PD** — NTT.4 runs in Unsigned PD via
  DSPRPC_CONTROL_UNSIGNED_MODULE. Production target (per
  reference-mode-d-bridge-architecture, reference-signed-pd-developer-path):
  Mode D Signed PD with VTCM pinned + real-time priority unlocked.
  Deferred to the Mode D sprint (post-NTT.5 / pre-PPT-scaling-mission).

## Bugs caught + fixes applied during the sprint

Documented in commit messages and below for retention:

1. **Function-name mismatch with qaic dispatcher.** Initially named INTT
   impl `sp_compute_ntt_intt_hvx_oracle`. qaic emits dispatcher referencing
   `sp_compute_<idl_name>` = `sp_compute_intt_hvx_oracle`. The shared lib
   linked with `U sp_compute_intt_hvx_oracle` undefined; would have crashed
   at first runtime invoke. Caught via `hexagon-nm libsp_compute_skel.so`
   before any device call. Renamed before any further iteration.

2. **VTCM stage-base misalignment for HVX vmem.** `w_inv_stages` lands at
   byte offset `12*N + 4*(N/2) = 14*N - 4` from VTCM arena base
   (8188 bytes for N=512). 4-byte aligned but NOT 128-byte aligned.
   Initial code cast `(const HVX_Vector *)(w_inv_stages + stage_off)` and
   read via aligned `vmem`; produced garbage at all stages (600/600
   divergences in Stage 1's first HVX run). Switched to `HVX_UVector*`
   (unaligned `vmemu`) for w_compact loads; `out` reads/writes stay aligned
   via `HVX_Vector*`. Captured in source comment at
   sp_compute_ntt_intt_imp.c:159-164.

3. **NTT.2 smoke harness has stale method numbers.** NTT.2's
   `sp_ntt_2_smoke.rs` calls `make_scalars(13, ...)` etc. for
   `ntt_twiddle_init/status/dump`, but post-renumber merge those are at
   case 14/15/16. The NTT.2 closure run was performed BEFORE NTT.1's
   merge (when those methods were genuinely at 13/14/15). NOT NTT.4's
   contamination scope to fix, but flagged for whichever sprint next
   touches sp_dsp_smoke. NTT.4's own smokes correctly use 14 for
   `ntt_twiddle_init`.

4. **Linearity-trick decomposition for per-prime INTT isolation does not
   work as naively expected.** I initially tried to extract t1 from
   math-core's `ntt_inverse(c_q1, zeros)` to compare against device r_q1.
   The recombine `r = t1 + q1 * ((t2 - t1) * Q1_INV_MOD_Q2 mod q2)` with
   t2 = 0 gives `r = t1 + q1 * (-t1 * Q1_INV_MOD_Q2 mod q2)`, NOT just t1.
   Confirmed by reasoning + arithmetic. The correct isolation requires
   either exposing math-core's static `inverse_one` (out of scope) or
   verifying composition (which is what Stage 3 does cleanly). Diagnostic
   code attempting this was deleted before final commit; lesson logged
   for future sprints needing per-prime INTT verification: use composition
   gates, not extraction.

## What unblocks

- **NTT.5 (MeMo integration)** — the polynomial multiplication primitive
  is the load-bearing op for MeMo's negacyclic attention. Forward NTT
  (NTT.1+2+3) AND INTT+Garner (this sprint) compose into the full
  pipeline. NTT.5 wires `polymul(K_block, V_block)` calls into the
  attention forward path. Primary risks: VTCM lifecycle across sequential
  layers, and Arc<FastRpcSession> shared across the K/V/MoE/scoring lanes.

- **Lattice-side polynomial arithmetic primitives** — anything in the
  lattice that needs `poly_mul mod (x^N+1)` mod q (NTT.5 attention,
  TS sieve products, Spinor receipt block products) can now consume the
  full ARM↔cDSP↔ARM round-trip with byte-exact semantics vs the math-core
  reference.

## Memory entry candidates

```
- [Reference: NTT.4 INTT VTCM unaligned-load idiom](reference_ntt4_intt_vtcm_unaligned.md)
  — `w_inv_stages` stage-base offsets are 4-byte aligned but not 128-byte
  aligned (e.g. byte 8188 within N=512 arena). HVX_Vector* aligned vmem
  produces garbage; HVX_UVector* unaligned vmemu produces byte-exact output.
  Applies to any future HVX consumer of per-stage compacted twiddles directly
  from VTCM (NTT.5 MeMo attention, NTT.6 tiled long-context).

- [Reference: qaic function-naming convention](reference_qaic_function_naming.md)
  — qaic dispatcher emits `sp_compute_<idl_method_name>`. IDL method
  `intt_hvx_oracle` → C function `sp_compute_intt_hvx_oracle`. Mismatch
  produces U-undefined linker symbol that slides through SHARED lib link
  with default visibility. ALWAYS verify with `hexagon-nm libsp_compute_skel.so`
  after first link to catch silent name-mismatch crashes-on-first-invoke.

- [Project: Phase 4-NTT.4 INTT + Garner polymul round-trip CLOSED](project_phase4_ntt4_closed.md)
  — sprint/ntt-4 @ b2dabb9, tag candidate lat-phase-4-ntt-4-intt-garner.
  All 3 gates PASS on S22U V69 cDSP Path B Unsigned PD. ~2.65 ms per
  N=512 polymul round-trip (informational). Closes the polynomial-
  multiplication primitive end-to-end byte-exact vs math-core's
  ntt_inverse. Unblocks NTT.5 MeMo integration.
```

## Worktree status

```
D:\F\shannon-prime-repos\engine-ntt-4   sprint/ntt-4   b2dabb9 (+ this closure commit)
```

Operator action: review + merge sprint/ntt-4 → main; tag
`lat-phase-4-ntt-4-intt-garner`. Resolve IDL method 17 ↔ 18 collision
against NTT.3 at merge time (renumber the second-landing branch). Update
NTT.4 smoke `INTT_METHOD` constant if renumbered.
