# PLAN — Sprint NTT.0 (Scalar Hexagon NTT, Reference Port)

**Date:** 2026-05-30
**Branch:** `sprint/ntt-0` (engine worktree `D:\F\shannon-prime-repos\engine-ntt-0`)
**Base:** engine main @ 833abbe (ledger-autowire closed)
**Status:** **STAGE 0 — UPSTREAM-REQUIRED — math-core ABI conflicts with prompt+roadmap N ladder**

## Headline

Stage 0 reference reading is complete. The substantive math gate as specified
(`T_NTT0_SCALAR_BIT_EXACT` at N ∈ {256, 1024, 4096} × 2 frozen primes) is
**mathematically impossible against math-core's portable C reference** and is
formally rejected by math-core's frozen ABI. Per
`feedback-no-silent-gate-revisions`, this plan is filed in UPSTREAM-REQUIRED
state pending operator disposition on which N ladder NTT.0 ships at. No
implementation work has been started; the plan and the diagnosis are the
deliverable of this stop.

## Stage 0 reference-read citations (mandatory plan opener)

1. **Math-core canonical NTT — primary reference.**
   - `lib/shannon-prime-system/core/ntt_crt/ntt_crt.c` at math-core
     `aeecdbae` (submodule pinned by engine main @ 833abbe).
   - Forward path: `forward_one(ctx, pc, in, out)` at lines 259-272 →
     `ntt_core(ctx, pc, x, wtab)` at lines 229-255.
   - Public API: `ntt_forward(ctx, in, out1, out2)` at lines 274-278.
     Takes one signed `int32_t *in` of length N, emits TWO residue
     vectors `uint32_t *out1` (mod q1) and `uint32_t *out2` (mod q2).
   - Init: `ntt_init(N)` at lines 188-215. Allocates `prime_ctx`
     containing `psi_pow[N]`, `ipsi_pow[N]`, `w_fwd[N/2]`, `w_inv[N/2]`
     per prime, plus `bitrev[N]`, `q1_inv_mod_q2`, `M = q1*q2`.
   - Algorithm (negacyclic Cooley-Tukey radix-2 DIT):
     - Step 1 (line 266-270 in `forward_one`): pre-weight `in[j]` by
       `psi^j` (the psi pre-weight is what makes this negacyclic;
       psi is a primitive 2N-th root with `psi^N == -1`).
     - Step 2 (lines 235-239 in `ntt_core`): in-place bit-reversal
       permutation using cached `bitrev[]`.
     - Step 3 (lines 241-254): logN stages of radix-2 DIT butterflies
       `(u, v) = (a + w·b, a - w·b)` with `w = wtab[widx]`, where
       `wtab = w_fwd[]` for forward = `omega^j` for j ∈ [0, N/2),
       `omega = psi^2` is a primitive N-th root.
   - Barrett primitive (lines 72-78): `barrett_reduce(x, q, mu)` with
     `mu = floor(2^60/q)`, shifts (Q_BITS-1)=29 and (Q_BITS+1)=31.
     Single multiply-then-conditional-subtract twice. Identical
     formula to cDSP scalar version (item #3 below).

2. **Phase 4-NTT roadmap block** —
   `D:\F\shannon-prime-repos\shannon-prime-lattice\papers\PPT-LAT-Roadmap.md`
   lines 5960-6177. Architectural commitments lines 5993-6044
   (commitments #1-#8). Sprint NTT.0 spec lines 6048-6055.
   Cross-references the math-core `sp_ntt` (line 5997) as the
   negacyclic reference. **Commitment #5 (line 6020) specs the
   target ladder N ∈ {256, 1024, 4096, 16384} and the NTT.0 gate
   (line 6052-6054) specs N ∈ {256, 1024, 4096}. These ladders
   conflict with the frozen-primes constraint — see Diagnosis
   below.**

3. **K.beta.2.5b scalar Barrett primitive** —
   `tools/sp_compute_skel/src_dsp/sp_compute_crt_imp.c` lines 24-56.
   - Frozen prime constants (lines 25-29):
     `SP_NTT_Q1 = 1073738753u`, `SP_NTT_Q2 = 1073732609u`,
     `SP_MU_Q1 = 1073744895u`, `SP_MU_Q2 = 1073751039u`.
   - Function signature (line 42):
     `static inline uint32_t sp_barrett_reduce32_scalar(uint64_t x,
     uint32_t q, uint32_t mu)`.
   - Algorithm (lines 43-47) is byte-identical to math-core's
     `barrett_reduce`: `qhat = (x>>29)*mu >> 31`, `r = x - qhat*q`,
     two conditional subtracts. Reusable as-is for the NTT butterfly
     modular-multiply inner loop.
   - Per-prime wrappers `sp_modmul_scalar_q1` (line 50) and
     `sp_modmul_scalar_q2` (line 54).

4. **K.beta.2.5b/c cascade pattern** —
   - `tools/sp_compute_skel/docs/CLOSURE-K-beta-2-5b.md` lines 1-65
     (scalar oracle → HVX vector kernel; cross-mode bitwise gate as
     the load-bearing correctness claim).
   - `tools/sp_compute_skel/docs/CLOSURE-K-beta-2-5c.md` (not opened
     here in full; pattern is the same — scalar reference, then
     vector kernel, then dual-dispatch). The discipline for NTT.0
     mirrors 2.5a: the scalar pipe oracle on the cDSP is the
     SOURCE OF TRUTH that NTT.1 (HVX butterfly) validates against.

5. **`reference-halide-hvx-int64-limitation`** — confirms NTT.1+
   uses hand-rolled HVX intrinsics, not Halide. Out-of-scope for
   NTT.0 (scalar); future NTT.1 will reuse `sp_barrett_reduce32_hvx_lane`
   from `sp_compute_crt_imp.c:74-122`.

6. **Discipline rules (re-read before drafting):**
   - `feedback-lead-with-reference-then-theory` — this plan's
     opener cites file:line BEFORE designing.
   - `feedback-no-silent-gate-revisions` — the N-ladder conflict
     surfaces UPSTREAM in this plan, NOT silently relaxed.
   - `feedback-bundled-changeset-root-cause-ambiguity` — staged
     commits planned (1 = math-core port verify, 2 = skel + IDL,
     3 = smoke + on-device gates, 4 = closure).
   - `feedback-parallel-agents-separate-worktrees` — sole worktree
     `engine-ntt-0`; no cross-contamination.
   - `reference-fastrpc-concurrent-dispatch` — relevant to NTT.3,
     NOT NTT.0 (single-thread oracle). Cited as future-sprint
     context.
   - **The mesh-canonical-order Stage-0 lesson** — read the actual
     code, surface UPSTREAM when reality differs from the spec
     rather than building on the wrong premise. This plan IS
     that pattern.

7. **`include/sp/ntt_crt.h`** at math-core `aeecdbae` (header
   covering the canonical NTT). Lines 53-55:
   > "N must be one of {128,256,512}; any other value (including
   > 1024, which the frozen primes cannot support) returns NULL."
   This is the frozen ABI rejection of N=1024.

8. **`papers/SESSION-CLOSED-lat-1.md` line 62**:
   > "Roadmap §4.2/4.3/7.2/7.3/7.8 + Theory §2.3: N capped to
   > {128,256,512}; the frozen primes admit no 2N-th root at
   > N=1024. Primes kept frozen (dominance-verification + DHT
   > topology depend on the exact residues). User-confirmed
   > decision."
   The operator-confirmed cap is on the record from a prior
   session; the Phase 4-NTT block filed 2026-05-30 did not
   re-read this and specs an impossible ladder.

## Diagnosis: the N-ladder conflict (load-bearing)

### The math

Negacyclic NTT over `Z_q[x]/(x^N + 1)` requires a primitive 2N-th
root of unity in `Z_q`, i.e., an element `psi` with `ord(psi) = 2N`.
By Fermat, the multiplicative group `(Z_q)^*` has order `q-1`, and
a 2N-th root exists iff `2N | (q - 1)`.

For the frozen primes:
- `q_1 - 1 = 1073738752 = 2^10 × 1048573`. The 2-adic valuation is
  exactly 10.
- `q_2 - 1 = 1073732609 - 1 = 1073732608 = 2^10 × 1048567`. Same
  2-adic valuation.

Therefore `2N | (q - 1)` for both primes iff `2N ≤ 2^10`, i.e.,
**N ≤ 512**. N=1024 (needs 2N=2048 = 2^11) and N=4096 (needs
2N=8192 = 2^13) both fail this divisibility constraint.

### Where this is rejected by math-core in code

`lib/shannon-prime-system/core/ntt_crt/ntt_crt.c:188-215`
`ntt_init(uint32_t N)`:
- Line 189: `if (N != 128u && N != 256u && N != 512u) return NULL;`

For N values outside {128, 256, 512}, the init returns NULL
unconditionally — even N=1024 cannot construct a context.

If line 189 were patched, `find_psi(N, q, mu)` at lines 116-127 would
iterate the search loop without finding a primitive 2N-th root (the
math forbids one), then return 0 → `prime_setup` returns 0 →
`ntt_init` cleans up + returns NULL.

### Where this conflict is acknowledged on the record

- `include/sp/ntt_crt.h:53-55` (the frozen public API doc).
- `papers/SESSION-CLOSED-lat-1.md:62` (user-confirmed cap from a
  prior session).
- The Phase 4-NTT roadmap block (filed 2026-05-30) does NOT
  acknowledge this — commitment #5 (lines 6020-6025) lists
  `N ∈ {256, 1024, 4096, 16384}` and Sprint NTT.0 gate
  (lines 6053-6054) lists `N ∈ {256, 1024, 4096}` without
  flagging the conflict with the frozen primes.

### What this means for NTT.0

The NTT.0 gate as specified — "0 divergences across 100 random
inputs × 3 N values × 2 primes vs math-core C reference" — cannot
be implemented because math-core's reference does not exist at
N ∈ {1024, 4096}. Any "agreement" at those N values would be
between a Hexagon scalar implementation and itself (no oracle),
not between Hexagon and math-core.

This is exactly the failure mode `feedback-no-silent-gate-revisions`
prohibits. Three forbidden silent-revision paths:

1. **Silent N-ladder shrink to {128, 256, 512}** — would close
   the gate but mute the architectural conflict; future agents
   would not know NTT.5/.6 cannot reach ctx=4096 on the frozen
   primes alone.
2. **Silent N-ladder shift to {256, 512, ...}** then "we'll get
   1024+ in a follow-up" — same opacity.
3. **Silent prime change** to admit larger N (e.g., a 32-bit Proth
   prime with 2-adic valuation ≥ 13) — violates architectural
   commitment #2 (FROZEN primes) and breaks K.beta.2.5c Garner
   constants, dominance verification, DHT topology, and every
   cross-backend bit-identity claim shipped to date.

## UPSTREAM-REQUIRED disposition options

The operator needs to pick one of the following paths before NTT.0
proceeds to Stage 1. Each is a substantive architectural decision.

### Path A — Re-spec NTT.0 to N ∈ {128, 256, 512}

Match the frozen-primes constraint. Ship NTT.0 as the scalar oracle
at the only N values where the math-core reference exists. Surface
to roadmap: Phase 4-NTT commitments #5 and Sprint NTT.0 (lines 6020
and 6053) get an amendment block dated 2026-05-30 acknowledging
the user-confirmed cap from `SESSION-CLOSED-lat-1.md:62`.

**Consequences for downstream sprints:**
- NTT.1 (HVX butterfly): also gated to N ≤ 512 with the frozen
  primes alone. The 32-lane HVX vector still applies; just the
  outer N loop is shorter.
- NTT.2 (twiddle VTCM staging): table size is small even at N=512
  (2 KB per prime); the VTCM budget gate is trivially passed.
  Reframe the gate.
- NTT.3 (dual-prime CRT dispatch): unchanged in shape; per-prime
  N still ≤ 512.
- NTT.4 (INTT + Garner): unchanged.
- **NTT.5 / NTT.6 (the MeMo + long-context payload): this is the
  real impact.** ctx=4096 mobile chat target was the headline.
  N=512 with the frozen primes means polynomial-ring attention
  can wrap N=512 polynomials at a time but not directly compute
  ctx=4096 in a single NTT. The pre-existing math-core Phase 7
  pattern (`reference-lattice-decode-determinism` and the
  persistent K-cache phase) handles this via tiling — multiple
  N=512 NTT slices composed into longer context. NTT.5 would
  need to wire the tiling, not just a single NTT.
- The "630× speedup at ctx=8192" framing in the roadmap block
  (lines 5973-5977) becomes ctx-dependent on the tiling
  efficiency, not a single-NTT property.

### Path B — Add a third prime to admit larger N

Pick a 30-bit Proth prime with 2-adic valuation ≥ 13 (allowing
N=4096) or ≥ 15 (allowing N=16384). Candidate set is small —
the canonical sweep would search 2^13+1, 5·2^13+1, ..., 2^14+1, ...
up to 30 bits. The new prime joins q_1, q_2 in a TRIPLE-prime CRT.

**Consequences:**
- **VIOLATES architectural commitment #2** ("FROZEN primes") unless
  the FROZEN list is amended to include the new prime as a
  capacity-extension prime. Either way is a Phase-level decision,
  not a sprint-level one.
- Cascading invalidation: K.beta.2.5c Garner constants
  (`Q1_INV_MOD_Q2 = 894602413`, `M = q1*q2`), `SP_NTT_M`, every
  dominance-verification residue, DHT topology constants would
  need to be triple-prime-aware. This is months of work.
- Path B is NOT a NTT.0-scope option. Surface to a separate
  Phase 4-NTT-PRIME-EXTENSION sprint with full impact analysis.

### Path C — Defer NTT.0 closure pending operator clarification

Operator confirms which N ladder NTT.0 ships at. Stop here.

## Recommendation (operator-decides)

**Path A** with a roadmap amendment block. The current spec was filed
without re-reading the frozen `ntt_crt.h` ABI; the user-confirmed
N cap from `SESSION-CLOSED-lat-1.md:62` predates the Phase 4-NTT
block. The most consistent path is to honor the FROZEN primes
commitment (locked, load-bearing) and amend the N ladder
commitment (filed 2026-05-30, not yet locked by any shipping
code) to match math-core's reality.

This is the same shape as the K.beta.2.5b operator-disposition
table: gate FAIL is preserved on the record, architectural reason
is named, corrective forward path is named, future sprint scopes
shift accordingly.

## If Path A is chosen — the rest of the plan

Below is the scoped implementation plan assuming operator selects
Path A. NOT to be executed without operator disposition.

## [plan-amend 2026-05-30 — Path A selected by operator]

**Operator disposition:** Path A. The roadmap was corrected at lattice
`main @ e927f6f` ("[Phase 4-NTT block CORRECTED + NTT.0 UPSTREAM-REQUIRED
resolved Path A]"). N ladder is now `{128, 256, 512}` everywhere; long
context (ctx ≥ 1024) is reframed as TILED N=512 NTT blocks for NTT.6.
Two memory entries written: `reference-ntt-frozen-primes-N-cap` +
`feedback-lead-with-reference-then-theory` (updated).

**Stage 1 scope reduction (continuation-agent decision):** the original
Stage 1 in this plan introduced a host-runnable Rust port of math-core
plus a `sp_ntt_ref_test` self-check binary. The continuation prompt asks
to use math-core's C reference DIRECTLY via the existing static-lib link
path (the L3.FG cross-compile pattern that `sp_daemon/build.rs` already
implements). `sp_dsp_smoke` does not currently have a build.rs — it
will get one mirroring `sp_daemon/build.rs`, linking math-core's
`sp_ntt_crt` (and the modules it depends on transitively).

**Why this is better than the prior Stage 1 plan:**
1. Single source of truth — math-core's C reference is the oracle on
   the Rust side too; no Rust port to drift out of date.
2. Removes a sub-gate (T_NTT0_REF_SELF_CHECK is no longer needed; the
   Rust port doesn't exist to validate).
3. Mirrors the K.beta.2.5c oracle pattern: skel-side scalar invariant
   compared against host-side math-core C — but now the host side IS
   math-core, not a Rust port of it.

**Stage renumbering:** Stage 1 is reduced from "Rust reference port +
ref-self-check binary" to "build.rs wiring + bindgen of `ntt_crt.h`".
Stages 2/3/4 unchanged. Gate `T_NTT0_REF_SELF_CHECK` is dropped (the
Rust port it referenced no longer exists); the only gate is
`T_NTT0_SCALAR_BIT_EXACT` (Hexagon scalar vs math-core C, 600 runs).

**Risk:** sp_ntt_crt depends transitively on a small subset of math-core
modules. The plan adds JUST `sp_ntt_crt` (plus any direct transitive
need surfaced at link time). If the dependency graph requires more
modules than expected, that surfaces at link time as a clear error
(not a silent miscompare), which is recoverable.

### Stage 1 — Math-core reference port verification

Goal: produce a Rust-side scalar reference NTT that's byte-identical
to math-core's `ntt_forward` so the on-device smoke can cross-validate
without requiring math-core to cross-compile to aarch64-android.

Pattern reuses `sp_matmul_q_ref.rs` (Sprint K.beta.2.5c Stage 1's
host-side scalar reference). The math is portable (no `__int128`);
the Rust port is 1:1 with math-core's `ntt_crt.c` algorithm:
- `prime_ctx` → Rust `PrimeCtx { q, mu, psi, ninv, psi_pow, ipsi_pow,
  w_fwd, w_inv }`
- `ntt_ctx`   → Rust `NttCtx { n, log_n, bitrev, p1, p2,
  q1_inv_mod_q2, m }`
- `barrett_reduce` → Rust `barrett_reduce(x: u64, q: u64, mu: u64)
  -> u64`
- `ntt_core` → Rust `ntt_core(&self, pc: &PrimeCtx, x: &mut [u32],
  wtab: &[u32])`
- `forward_one` → Rust `forward_one(&self, pc, in_buf: &[i32],
  out: &mut [u32])`

Gate `T_NTT0_REF_SELF_CHECK`:
- Drive Rust reference + math-core C reference on the same 100 random
  inputs at N ∈ {128, 256, 512}; compare `out1[]` + `out2[]`
  element-wise. 0 divergences expected (Rust port is byte-identical
  math). This stage's gate is INDEPENDENT of the Hexagon scalar
  oracle and proves the Rust reference is faithful to math-core
  before the on-device round.
- Host-runnable binary `sp_ntt_ref_test` (cargo `[[bin]]`) following
  `sp_matmul_q_ref_test.rs` pattern. Runs on Windows/Linux x86 with
  no DSP needed.

Math-core C reference link: cleanest path is to add math-core as
a `[build-dependencies]` `cc` crate compilation step OR shell out to
a pre-built `ntt_gen_fixture` binary from `core/ntt_crt/ntt_gen_fixture.c`
to dump reference vectors into a `.bin` then read those from the
Rust test. Decision deferred to Stage 1 (latter is simpler — no
build-system churn).

Commit: `[NTT.0] stage 1 — Rust scalar NTT reference + ref-self-check binary`

### Stage 2 — cDSP-side `sp_ntt_scalar` + IDL `ntt_oracle`

New file: `tools/sp_compute_skel/src_dsp/sp_compute_ntt_imp.c` (split
from the existing `sp_compute_crt_imp.c` since NTT is structurally
larger than Barrett primitive — separation aids future SASS audit
in NTT.1).

Function signature:

```c
// sp_compute_ntt_scalar: scalar negacyclic NTT for one prime,
// matching math-core's ntt_forward per-prime path (the per-prime
// pipe inside ntt_crt.c::ntt_forward).
//
// q_idx selects (Q1, MU_Q1) or (Q2, MU_Q2) — frozen primes.
// N must be in {128, 256, 512}; returns -1 otherwise.
// data_in: N signed i32 LE; data_out: N u32 LE in [0, q).
//
// Twiddle tables computed per-call inside this function for
// NTT.0 (correctness over speed). NTT.2 lifts twiddles to a
// preallocated VTCM-resident table.
int sp_compute_ntt_scalar(int q_idx, int N,
                          const int32_t *data_in,
                          uint32_t *data_out);
```

Twiddle computation per-call: scratch arrays sized for N=512 max
(2 KB for psi_pow, 1 KB for w_fwd) declared on the stack or via
`HAP_request_VTCM`. Decision per Sprint G recipe — declare as static
inside the function with N=512 sizing; NTT only uses N first
entries.

IDL method (append to `sp_compute.idl` after `matmul_q`):

```idl
// ntt_oracle — Sprint NTT.0 T_NTT0_SCALAR_BIT_EXACT
//
// Drives N signed i32 inputs through scalar negacyclic NTT mod q_{1,2}.
//   q_idx=0 → q_1, q_idx=1 → q_2
//   N ∈ {128, 256, 512}; returns -1 otherwise
// data_in:  N × i32 LE (caller's raw coefficients, reduced mod q skel-side)
// data_out: N × u32 LE in [0, q)
long ntt_oracle(in long q_idx, in long N,
                in  sequence<octet> data_in,
                rout sequence<octet> data_out);
```

Method index: 12 (after `matmul_q` at index 11). Verify against
qaic-emitted method table.

`sp_compute_imp.c::sp_compute_handler` dispatch entry added (skel
handler table). Build via `tools/sp_compute_skel/build.cmd`.

Commit: `[NTT.0] stage 2 — sp_compute_ntt_scalar + ntt_oracle IDL`

### Stage 3 — Smoke harness + on-device gates

New binary: `tools/sp_dsp_smoke/src/sp_ntt_oracle_smoke.rs` mirroring
`sp_barrett_oracle_smoke.rs:30-78`.

For each `q_idx ∈ {0, 1}` and `N ∈ {128, 256, 512}`:
  For each of 100 seeds:
    1. Generate `data_in: [i32; N]` with values uniform on a
       wide signed range (mimics `forward_one`'s "arbitrary int32"
       contract; the modular reduction at `forward_one:267-268`
       handles all signed values).
    2. Compute Rust reference via Stage 1 port:
       `rust_out: [u32; N] = ntt_ref.forward_q(data_in, q_idx)`.
    3. Invoke skel `ntt_oracle(q_idx, N, &data_in, &mut hex_out)`.
    4. Compare `rust_out[i] == hex_out[i]` for all i; record
       divergences.

Pass criterion: 0 divergences across 6 × 100 = 600 runs.

Report fields (per the prompt closure spec):
- `combinations_tested`: 6 (2 primes × 3 N values)
- `seeds_per_combination`: 100
- `total_runs`: 600
- `divergence_count`: 0 (expected)
- `max_lane_diff_observed_per_prime`: {q_1: 0, q_2: 0} (expected)
- `max_lane_diff_observed_per_N`: {128: 0, 256: 0, 512: 0} (expected)

Output captured to `tools/sp_dsp_smoke/sprint_ntt_0_run_output.txt`.

Commit: `[NTT.0] stage 3 — sp_ntt_oracle_smoke + T_NTT0_SCALAR_BIT_EXACT on-device run`

### Stage 4 — Closure

`tools/sp_compute_skel/docs/CLOSURE-NTT-0.md` per the prompt's 13-item
spec.

Commit: `[NTT.0] stage 4 — closure (CLOSURE-NTT-0.md)`

## Files NOT touched (anti-contamination)

- `shannon-prime\` (archive)
- `shannon-prime-engine\` (archive)
- `shannon-prime-system-engine\` (engine main worktree)
- All other `engine-*` / `lattice-*` worktrees
- `shannon-prime-system\core\ntt_crt\` (math-core's NTT — READ-ONLY
  reference; the Rust port and Hexagon scalar are in engine repo)
- `papers/PPT-LAT-Roadmap.md` — IF Path A is chosen, the operator
  amends the roadmap block; this agent does NOT silently amend it.
  The agent's plan + closure note the conflict and the disposition.

## Branch + push policy

- Worktree: `D:\F\shannon-prime-repos\engine-ntt-0` (sole; no
  cross-contamination).
- Branch: `sprint/ntt-0` (already created from main @ 833abbe).
- Push at end: `git push -u origin sprint/ntt-0`.
- Operator handles merge after review (NOT this agent).

## STOP

Per `feedback-no-silent-gate-revisions` and the discipline rule
inherited from the mesh-canonical-order Stage-0 lesson, this plan
stops at UPSTREAM-REQUIRED. Stage 1 implementation does NOT
proceed until operator picks a disposition.
