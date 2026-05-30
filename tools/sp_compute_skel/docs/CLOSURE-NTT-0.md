# CLOSURE — Sprint NTT.0 (Scalar Hexagon NTT, Reference Port)

**Date:** 2026-05-30
**Branch:** `sprint/ntt-0` (engine worktree `D:\F\shannon-prime-repos\engine-ntt-0`)
**Base:** engine main @ 833abbe (ledger-autowire closed)
**Status:** **STAGE 0 ONLY — UPSTREAM-REQUIRED — math-core ABI conflicts with prompt+roadmap N ladder**

## Headline

Stage 0 reference reading is complete and surfaces an architectural
conflict that **blocks Stages 1-3 implementation** until operator
disposition: the NTT.0 substantive gate `T_NTT0_SCALAR_BIT_EXACT`
is specified at N ∈ {256, 1024, 4096} × 2 frozen primes, but the
math-core canonical C reference rejects N ∈ {1024, 4096} as
mathematically impossible with the frozen primes. Closure left in
UPSTREAM-REQUIRED state per `feedback-no-silent-gate-revisions`,
matching the K.beta.2.5b operator-disposition pattern.

## Gate table

| Gate | Threshold | Observed | PASS/FAIL | Comment |
|---|---|---|---|---|
| **T_NTT0_SCALAR_BIT_EXACT** | 0 divergences across 100 random inputs × 3 N values × 2 primes vs math-core C reference | NOT EXECUTED | **UPSTREAM-REQUIRED** | Reference (math-core's `ntt_forward`) does not exist at N ∈ {1024, 4096}. See Diagnosis below. |

## Math-core NTT signature (the reference this port matches)

- **Path:** `lib/shannon-prime-system/core/ntt_crt/ntt_crt.c` at math-core
  `aeecdbae` (submodule pinned by engine main @ 833abbe).
- **Public API:** `void ntt_forward(const ntt_ctx *ctx, const int32_t *in,
  uint32_t *out1, uint32_t *out2);` (lines 274-278). Takes one signed
  `int32_t` input vector of length N, writes TWO residue-domain output
  vectors (mod q1 and mod q2).
- **Allocator:** `ntt_ctx *ntt_init(uint32_t N);` (lines 188-215).
  Restricts N to {128, 256, 512} at line 189; returns NULL otherwise.
- **Inner algorithm:** Cooley-Tukey radix-2 DIT negacyclic NTT
  (`ntt_core` lines 229-255). Pre-weight by `psi^j` (`forward_one`
  lines 266-270) where `psi` is a primitive 2N-th root of unity
  (`find_psi` lines 116-127).
- **Barrett primitive:** `barrett_reduce(x, q, mu)` at lines 72-78
  with `mu = floor(2^60 / q)`. Math-identical to cDSP scalar Barrett
  at `tools/sp_compute_skel/src_dsp/sp_compute_crt_imp.c:42-48`.

## Hexagon scalar implementation

NOT WRITTEN. Stage 2 was blocked by the Stage 0 UPSTREAM-REQUIRED
disposition. Planned signature documented in `PLAN-NTT-0.md` (Stage 2
section).

## IDL method

NOT WRITTEN. Planned signature documented in `PLAN-NTT-0.md` (Stage 2
section).

## Architecture compliance (commitments from Phase 4-NTT block)

| # | Commitment | Status |
|---|---|---|
| 1 | Negacyclic NTT (primitive 2N-th root, not N-th) | **COMPLIES** (plan honors; math-core is already negacyclic). |
| 2 | FROZEN primes (q_1, q_2, M, Q1_INV_MOD_Q2) | **COMPLIES** (plan honors; primes are not changed). |
| 3 | Barrett in inner loop | **COMPLIES IN PLAN** (reuse `sp_barrett_reduce32_scalar`). |
| 4 | Cooley-Tukey radix-2 DIT | **COMPLIES IN PLAN** (matches math-core). |
| 5 | N ladder N ∈ {256, 1024, 4096, 16384} | **CONFLICTS WITH #2.** N=1024 and beyond are mathematically impossible with the frozen primes (max N=512). See Diagnosis. |
| 6 | Scalar Hexagon reference before vectorizing | **COMPLIES IN PLAN** (NTT.0 IS the scalar oracle). |
| 7 | Shape-regime-aware parallelism gates | N/A for NTT.0 (single-thread oracle; relevant to NTT.3). |
| 8 | MeMo integration is THE deliverable | N/A for NTT.0 (relevant to NTT.5). |

**The conflict between commitments #2 and #5 is the architectural
finding that puts this sprint in UPSTREAM-REQUIRED.** Both cannot
hold simultaneously; #2 is locked by every previously-shipped cross-
backend bit-identity gate (Phase 4-9, K.beta.2.5c Garner constants,
DHT topology). #5 was filed 2026-05-30 in the same roadmap block as
NTT.0 and is not yet locked by any shipping code.

## Diagnosis (the load-bearing finding)

### Mathematical statement

Negacyclic NTT over `Z_q[x]/(x^N + 1)` requires a primitive 2N-th
root of unity in `(Z_q)^*`. Such a root exists iff `2N | (q - 1)`.
For the frozen primes:

- `q_1 - 1 = 1073738752 = 2^10 × 1048573` (2-adic valuation = 10)
- `q_2 - 1 = 1073732608 = 2^10 × 1048567` (2-adic valuation = 10)

Therefore the maximum N admitting a 2N-th root in BOTH primes
simultaneously is **N = 512** (`2N = 2^10 = 1024 ≤ q-1`).

### Where the rejection lives in code

`lib/shannon-prime-system/core/ntt_crt/ntt_crt.c:189`:

```c
if (N != 128u && N != 256u && N != 512u) return NULL;
```

`include/sp/ntt_crt.h:53-55`:

> "N must be one of {128,256,512}; any other value (**including
> 1024, which the frozen primes cannot support**) returns NULL."

If the gate at line 189 were patched, `find_psi(N=1024, q, mu)` at
lines 116-127 would exhaust the search loop without finding a 2048-th
root (none exists), then return 0 → `prime_setup` returns 0 →
`ntt_init` cleans up and returns NULL.

### Where the conflict is on the record (prior sessions)

- `papers/SESSION-CLOSED-lat-1.md:62` — user-confirmed cap from a
  prior session: "N capped to {128,256,512}; the frozen primes
  admit no 2N-th root at N=1024. Primes kept frozen
  (dominance-verification + DHT topology depend on the exact
  residues). **User-confirmed decision.**"

The Phase 4-NTT roadmap block (filed 2026-05-30) does not
acknowledge this prior decision — commitments #2 + #5 are mutually
incompatible by construction.

### What this means for NTT.0 specifically

The substantive gate `T_NTT0_SCALAR_BIT_EXACT` says "0 divergences
... vs math-core C reference." For N ∈ {1024, 4096} math-core's
reference returns NULL — there is no oracle to compare against. Any
"agreement" at those N values would be Hexagon-scalar-vs-itself, not
Hexagon-vs-math-core, and would not satisfy the gate as written.

## Disposition options (operator-decides)

Three paths enumerated in `PLAN-NTT-0.md` "UPSTREAM-REQUIRED
disposition options". Summary:

- **Path A — Re-spec NTT.0 to N ∈ {128, 256, 512}.** Honors FROZEN
  primes (locked); amends N ladder commitment (not yet locked).
  Downstream NTT.5/.6 ctx=4096 mobile chat target shifts to a
  tiling strategy (multiple N=512 NTT slices), not a single
  large-N NTT. The "630× speedup at ctx=8192" framing becomes
  tile-composition-dependent.

- **Path B — Add a third prime to admit larger N.** Triple-prime
  CRT. Violates commitment #2 unless the FROZEN list is amended.
  Cascading invalidation of K.beta.2.5c Garner constants, every
  dominance-verification residue, DHT topology constants. NOT
  NTT.0-scope; needs a separate Phase 4-NTT-PRIME-EXTENSION
  sprint with full impact analysis.

- **Path C — Defer NTT.0 closure pending operator clarification.**

**Agent recommendation: Path A** with a dated roadmap amendment
block, matching the K.beta.2.5b operator-disposition pattern: gate
FAIL preserved on the record, architectural reason named, corrective
forward path named. Path A also preserves the most cross-backend
bit-identity guarantees (no prime change, no Garner-constants
change, no dominance-verification ripple).

## Files changed (this sprint)

| File | LOC delta | Purpose |
|---|---|---|
| `tools/sp_compute_skel/docs/PLAN-NTT-0.md` | +427 | Stage 0 plan with file:line citations + UPSTREAM diagnosis + Stage 1-4 plan conditional on Path A. |
| `tools/sp_compute_skel/docs/CLOSURE-NTT-0.md` | +this file | Stage 0 closure in UPSTREAM-REQUIRED state. |

Total new code shipped on cDSP / Rust: **0 lines.** This is by
design — surfacing UPSTREAM before writing code is exactly the
mesh-canonical-order Stage-0 discipline.

## Commits on `sprint/ntt-0`

- `8143fe5` — `[plan] NTT.0 — scalar Hexagon NTT (Cooley-Tukey
  radix-2 DIT, negacyclic, q_1+q_2) — UPSTREAM-REQUIRED on N
  ladder`
- (this commit) — `[NTT.0] stage 0 closure — UPSTREAM-REQUIRED
  on N ladder; no implementation work pending operator
  disposition`

## Proposed sub-tag

NO TAG. Closure is UPSTREAM-REQUIRED, not PASS. Tags
(`lat-phase-4-ntt-0-scalar-hexagon` or similar) wait for
operator disposition + Stages 1-3 completion.

## What's NOT done

Stages 1, 2, 3 (Rust reference, cDSP `sp_compute_ntt_scalar` +
`ntt_oracle` IDL, smoke harness + on-device gate) — all blocked
pending operator disposition. The work-detail is documented in
`PLAN-NTT-0.md` so a follow-up agent (or this agent on a second
pass) can execute promptly once the N ladder is locked.

NTT.1 (HVX butterfly), NTT.2 (twiddle VTCM staging), NTT.3 (dual-
prime CRT dispatch), NTT.4 (INTT + Garner), NTT.5 (MeMo
integration), NTT.6 (long-context benchmark) — all explicitly
out of scope per the prompt; NTT.0 was the foundation. None of
them is unblocked by this stop.

## What unblocks (after operator disposition + Path A execution)

If Path A is chosen and Stages 1-3 ship:

- NTT.1 + NTT.2 can dispatch in parallel (one agent on HVX
  butterfly intrinsics, one on twiddle VTCM staging).
- NTT.3 needs NTT.1 + NTT.2 both closed.
- NTT.4 + NTT.5 + NTT.6 chain from NTT.3.

If Path B is chosen, NTT.0 unblocks only after the
Phase 4-NTT-PRIME-EXTENSION sprint completes (months of cascading
work); NTT.1+ unblock from there.

If Path C, this sprint stays open.

## Memory entry candidates

1. **`reference-ntt-frozen-primes-N-cap`** — record the algebraic
   constraint and the existing ABI cap:
   > Negacyclic NTT over Z_q[x]/(x^N+1) requires 2N | (q-1).
   > Frozen primes q_1=1073738753, q_2=1073732609 have
   > 2-adic valuation 10 in (q-1), capping N at 512. Math-core
   > `ntt_init` (`lib/shannon-prime-system/core/ntt_crt/ntt_crt.c:189`)
   > + header `include/sp/ntt_crt.h:53-55` enforce this cap. Any
   > Phase 4-NTT roadmap spec at N ≥ 1024 conflicts with FROZEN
   > primes commitment and is mathematically impossible without a
   > prime change. User-confirmed decision per
   > `papers/SESSION-CLOSED-lat-1.md:62`. Re-read this before
   > spec'ing any future NTT N ladder.

2. **`feedback-roadmap-block-must-re-read-frozen-abi`** — pattern
   feedback: roadmap blocks that span FROZEN ABI surfaces must
   re-read the ABI header before drafting commitments. The
   Phase 4-NTT block filed 2026-05-30 specced N ∈ {1024, 4096}
   without re-reading `include/sp/ntt_crt.h:53-55` from the same
   day's math-core HEAD. Stage 0 reference-reading by the NTT.0
   agent caught the conflict before any code shipped. The mesh-
   canonical-order Stage-0 lesson (operator-side memory entry
   error caught by reading the actual struct) repeats: read the
   actual frozen surface BEFORE drafting derivative spec.

## Worktree status

- Worktree: `D:\F\shannon-prime-repos\engine-ntt-0` (sole).
- Branch: `sprint/ntt-0` (committed locally; will push at end).
- No commits authored from `shannon-prime-system-engine` main
  worktree. No cross-worktree contamination.
- math-core submodule initialized at `aeecdbae` (READ-ONLY
  reference; not modified).
- Anti-contamination held: zero edits to `shannon-prime\`,
  `shannon-prime-engine\`, math-core's NTT sources, or any
  other engine-*/lattice-* worktree.

## STOP — operator action required

Pick a disposition (Path A / Path B / Path C). On disposition,
the follow-up agent (or this agent on a second pass) executes
the Stage 1-4 plan documented in `PLAN-NTT-0.md` against the
locked N ladder.
