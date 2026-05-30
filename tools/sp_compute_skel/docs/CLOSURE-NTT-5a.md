# CLOSURE-NTT-5a.md — Sprint NTT.5a Host-side Bluestein wrapper for math-core poly_ring

## Headline

NTT.5a ships the host-side Bluestein wrapper for math-core's negacyclic
polynomial ring, extending the admissible degree set from {128, 256, 512}
(direct `sp_pr_init`) to {2, 4, 8, 16, 32, 64, 128, 256} (Bluestein-wrapped
`sp_pr_bluestein_init`). All four substantive gates PASS bit-exact on the
first implementation pass: 800/800 inner-vs-schoolbook runs with 0
divergences, 200/200 inner-vs-`sp_pr_inner` runs at the overlap N values
with 0 divergences, 400/400 mul-vs-schoolbook runs with 0 per-coefficient
divergences, and 11/11 inadmissible N values returning NULL. Combined with
the four existing T_PR_* tests still green, the test executable reports
**4150 checks / 0 failures** end-to-end.

Math-core ABI is unchanged — additive only. `poly_ring.h`, `poly_ring.c`,
`ntt_crt.h`, `ntt_crt.c`, `forward.c` were left untouched per the spec's
anti-contamination scope. NTT.5b will wire `sp_pr_bluestein_init(HD)` into
`forward.c:115-116`, extend the L1 ABI, and stand up the Hexagon backend
counterpart.

- Worktree:  `D:\F\shannon-prime-repos\engine-ntt-5a` on `sprint/ntt-5a`.
- Engine base: `fec6fe3` (post NTT.4 merge); engine tip: `1311ad2`.
- Submodule (math-core) base: `aeecdbae5a41e35d009fb2c82046ca3bbcd33454`
  on detached HEAD; new branch `sprint/ntt-5a` tip: `7662b2d`.
- 9 commits across both repos (4 engine, 5 submodule) — see §"Commits".

## Chirp formula derivation in Z_q (Approach A, chosen)

Two Bluestein formulations were considered; PLAN-NTT-5a.md §"Chirp formula
derivation in Z_q" derives both. The chosen form (Approach A) reuses
math-core's negacyclic length-M NTT as a length-(2N-1) linear convolver via
zero-padding, then folds back to length-N cyclic and post-untwists. The
chirp-z form (Approach B) was rejected on simplicity + cost grounds (4
inner NTT calls per chirp wrap × 2 chirp wraps + extra pointwise multiplies
≈ 4× the work of Approach A's 6 inner NTT calls; both produce bit-exact
identical results).

The pipeline per Bluestein call:

  **Step 1 — per-prime psi-twist (in coefficient domain).**
  For each prime qP ∈ {SP_NTT_Q1, SP_NTT_Q2}, search a primitive 2N-th root
  ψ_NP (Fermat-style modpow primitive search, same as `find_psi` in
  `ntt_crt.c:116-127`). Form the outer-twisted, zero-padded input:

      a_tw_qP[j] = (a[j] · ψ_NP^j) mod qP   for j ∈ [0, N)
                 = 0                          for j ∈ [N, M)

  The identity that justifies the twist (full derivation in
  `core/poly_ring/poly_ring_bluestein.c:45-66`):

      C_k = sum_{i+j ≡ k (mod N)} (a_i ψ_NP^i) (b_j ψ_NP^j)
          = sum_{i+j=k}   a_i b_j ψ_NP^k        (no wrap → ψ_NP^k = ψ_NP^{i+j})
          + sum_{i+j=k+N} a_i b_j ψ_NP^{k+N}     (negacyclic wrap)
          = ψ_NP^k · (sum_{i+j=k} a_i b_j  −  sum_{i+j=k+N} a_i b_j)
            since ψ_NP^N = −1.
          = ψ_NP^k · c_k.

  So `c_k = ψ_NP^{-k} · C_k mod qP`, where C is the length-N CYCLIC
  convolution of (twisted a, twisted b).

  **Step 2 — length-M linear convolution via math-core's CRT NTT.**
  Math-core's `ntt_forward(int32_t *in, ...)` fuses both primes from a
  single int32 input. Since our outer twist produces DIFFERENT residues
  per prime, we call `ntt_forward` TWICE per operand — once with the q1-
  twisted input (keeping only the q1 channel output, the q2 channel goes
  to scratch), once with the q2-twisted input (vice versa). The pointwise
  multiply and inverse run unchanged:

      ntt_forward(inner, a_pad_q1, a_res_q1, scratch_q2);   // keep a_res_q1
      ntt_forward(inner, a_pad_q2, scratch_q1, a_res_q2);   // keep a_res_q2
      ntt_forward(inner, b_pad_q1, b_res_q1, scratch_q2);
      ntt_forward(inner, b_pad_q2, scratch_q1, b_res_q2);
      ntt_pointwise_mul(inner, a_res_q1, a_res_q2,
                                b_res_q1, b_res_q2,
                                c_res_q1, c_res_q2);
      ntt_inverse(inner, c_res_q1, c_res_q2, D);            // signed int64

  Because a_pad_qP and b_pad_qP vanish for index ≥ N, the highest nonzero
  linear-conv index is 2N−2 < M, so the length-M wraparound term (x^M = −1
  sign flip) contributes zero. D is therefore the linear convolution of the
  CRT-combined integer representatives x = CRT(a_tw_q1, a_tw_q2),
  y = CRT(b_tw_q1, b_tw_q2) — signed-centered int64.

  Cost: 4 × `ntt_forward` + 1 × `ntt_pointwise_mul` + 1 × `ntt_inverse`
  per Bluestein call (vs `sp_pr_mul`'s 2 + 1 + 1). The 2× forward cost is
  the price of math-core's "fuse both primes" forward API; an exposed
  per-prime forward would halve it. Out-of-scope for NTT.5a.

  **Step 3 — fold length-(2N−1) → length-N cyclic, per prime.**
  Reduce D back to per-prime residues, then fold:

      C_qP[k] = (D[k] mod qP + (k+N < M ? D[k+N] : 0) mod qP) mod qP
                                                              for k ∈ [0, N).

  For our M choices (smallest M ∈ {128, 256, 512} with M ≥ 2N), `k + N` is
  always in [N, 2N) ⊆ [0, M), so the conditional always picks D[k+N].

  **Step 4 — per-prime post-untwist by ψ_NP^{-k}.**
  In place:

      c_qP[k] = (C_qP[k] · ψ_NP^{-k}) mod qP.

  **Step 5 — local Garner CRT recombine to signed centered int64.**
  Math-core's `ntt_crt_recombine(ctx, x1, x2, out)` iterates over
  `ctx->N` entries — which for us is the **inner M**, not the outer N.
  Calling it directly would over-write the caller's length-N output buffer
  by M − N entries. Instead we drive a local `pr_blue_garner` loop over
  exactly N coefficients (`core/poly_ring/poly_ring_bluestein.c:138-159`).
  The local Garner algorithm mirrors `ntt_crt.c:303-316` line for line; the
  q1^{-1} mod q2 constant is precomputed in `sp_pr_bluestein_init` via our
  own modular primitives (which are necessary anyway for the outer-N psi
  search).

## Admissible-N analysis

The full table from PLAN-NTT-5a §"Bluestein admissible-N analysis", carried
forward unchanged:

  | N    | inner M (smallest ∈ {128,256,512} with M ≥ 2N) | result |
  | ---- | ---------------------------------------------- | ------ |
  | 2    | 128 (math-core min; 2N=4 oversampled to 128)   | admit  |
  | 4    | 128                                            | admit  |
  | 8    | 128                                            | admit  |
  | 16   | 128                                            | admit  |
  | 32   | 128                                            | admit  |
  | 64   | 128                                            | admit  |
  | 128  | 256                                            | admit  |
  | 256  | 512                                            | admit  |
  | 512  | (would need M=1024 — frozen primes can't)      | reject |
  | non-power-of-2 (3, 5, 6, 7, 9, 96, 100, 384, ...) | (no 2N-th root in Z_q) | reject |
  | 1024 | (exceeds 2-adic valuation cap)                 | reject |

Bluestein-admissible set = {2, 4, 8, 16, 32, 64, 128, 256}.

N=512 reject is intentional per spec: direct `sp_pr_init(512)` is more
efficient (M=512 native, no zero-padding overhead).

Inadmissible-N rejection at `bluestein_inner_M`
(`core/poly_ring/poly_ring_bluestein.c:131-145`); init returns NULL.

## Gates table

| Gate | Methodology | Pass criteria | Observed | Verdict |
|------|-------------|---------------|----------|---------|
| T_NTT5A_BLUESTEIN_BIT_EXACT_VS_SCHOOLBOOK | `sp_pr_bluestein_inner(N, q, k)` vs schoolbook `sum_i q[i]·k[i]`; N ∈ {2,4,8,16,32,64,128,256}; 100 random seeds per N; coeff ∈ [-2^14, 2^14) | 0 divergences in 800 runs | 0/800 divergences; 0/100 per-N for all 8 N values | **PASS** |
| T_NTT5A_VS_SP_PR_INNER_BIT_EXACT | `sp_pr_bluestein_inner` vs `sp_pr_inner` on overlap N ∈ {128, 256}; 100 random seeds per N | 0 divergences in 200 runs | 0/200 divergences | **PASS** |
| T_NTT5A_BLUESTEIN_MUL_BIT_EXACT | `sp_pr_bluestein_mul` vs schoolbook negacyclic O(N^2) reference; N ∈ {2..256}; 50 seeds per N; per-coefficient compare | 0 divergences across 400 runs × N coefficients | 0/{100,200,400,800,1600,3200,6400,12800} coeff divergences for N ∈ {2,4,8,16,32,64,128,256} | **PASS** |
| T_NTT5A_NULL_FOR_INADMISSIBLE_N | `sp_pr_bluestein_init(N)` for N ∈ {1, 3, 5, 6, 7, 9, 96, 100, 384, 512, 1024} | All return NULL | 11/11 NULL | **PASS** |

Additional sanity tests now green:
- T_NTT5A_SCHOOLBOOK_CROSSCHECK (NTT.5a oracle agrees with the existing
  `schoolbook_negacyclic` from `poly_ring_test.c:54-72` on overlap
  N ∈ {128, 256}): **PASS**.
- T_PR_1, T_PR_2, T_PR_3, T_PR_4 (existing math-core gates, untouched):
  all **PASS** (worst KL still 0.000e+00 over 64 rounds).

Total: **4150 checks / 0 failures**.

No gates failed. No silent gate revisions. No banned propositions surfaced
(no mixed-radix, no Good-Thomas, no zero-pad to next admissible
`sp_pr_init` N, no third prime, no Cooley-Tukey at non-power-of-2 N).

## Wall-clock comparison (informational, not a gate)

Bench source: `tools/sp_compute_skel/docs/NTT-5a-wallclock-bench.c`. Run on
Knack's Windows host with mingw-gcc 15.2 -O2; 1000 iterations per
operation; 50-iter warmup; random coeff ∈ [-2^14, 2^14).

  | N   | sp_pr_inner | sp_pr_bluestein_inner | ratio |
  | --- | ----------- | --------------------- | ----- |
  | 128 | 67.55 μs    | 235.79 μs             | 3.49× |
  | 256 | 138.03 μs   | 550.00 μs             | 3.98× |

Bluestein is ~3.5–4× slower per inner-product on the overlap N values. The
factor decomposes roughly as:
- 2× from `ntt_forward` being called twice per operand (math-core's API
  fuses both primes; we discard one channel per call). An exposed per-
  prime forward in math-core would halve this.
- 0.5–1× from per-call malloc/free of the involution kstar + product
  buffers in `sp_pr_bluestein_inner`. Caching these into the context is a
  trivial NTT.5b optimization.
- The remainder from the manual fold + untwist + Garner loop in
  coefficient domain.

This was expected and is in line with PLAN-NTT-5a §"Wall-clock comparison
plan". NTT.5b can engineer the constant down (cache scratch on context,
expose per-prime forward, or specialize sp_pr_bluestein_inner to a
"compute-c_0-only" fast path that skips the fold + untwist for k > 0). For
HD=64 (Qwen3-0.6B, Qwen2.5-Coder-0.5B), the absolute wall-clock is even
better than the N=128 row above (inner M=128 for any N ≤ 64), so the
production overhead is ~235 μs per per-head per (t, s) attention pair.
That's tolerable for the overlay activated by `g_ntt_attn` per
`forward.c:115`.

## Public API documentation

Final `include/sp/poly_ring_bluestein.h` contents are reproduced in full at
`commit f29fb68` (the initial header drop; unchanged since). Key
surface:

```c
typedef struct sp_pr_bluestein_ctx sp_pr_bluestein_ctx;

sp_pr_bluestein_ctx *sp_pr_bluestein_init(uint32_t N);
void                 sp_pr_bluestein_free(sp_pr_bluestein_ctx *ctx);
uint32_t             sp_pr_bluestein_degree(const sp_pr_bluestein_ctx *ctx);

int64_t sp_pr_bluestein_inner(sp_pr_bluestein_ctx *ctx,
                              const int32_t *q, const int32_t *k);

void    sp_pr_bluestein_mul(sp_pr_bluestein_ctx *ctx,
                            const int32_t *a, const int32_t *b,
                            int64_t *out);
```

Signatures match `sp_pr_init`/`sp_pr_free`/`sp_pr_degree`/`sp_pr_inner`/
`sp_pr_mul` shape-for-shape (modulo opaque-type rename), so NTT.5b can swap
in `sp_pr_bluestein_*` at `forward.c:115-211` with a single typedef +
function-pointer indirection or a conditional path on HD.

The contract documentation in the header covers:
- Admissible N set (and explicit NULL behavior for N ∈ {1, non-power-of-2,
  512, > 256}).
- Coefficient bit-exactness invariant (|c_k| < M_full/2 ⇒ exact in Z;
  satisfied for any caller |coeff| < 2^14 and N ≤ 256 by 2^36 ≪ 2^59).
- Thread-safety policy (one context per thread, same as `sp_pr_ctx`).
- The cross-references to `reference-ntt-frozen-primes-N-cap` and
  `reference-ntt-bluestein-arbitrary-n-escape` for the banned alternatives
  (mixed-radix, Good-Thomas, zero-pad-to-next-admissible-N, third prime).

## Files changed

NEW (math-core submodule):

  | File | LOC |
  | ---- | --- |
  | `include/sp/poly_ring_bluestein.h` | 86 |
  | `core/poly_ring/poly_ring_bluestein.c` | 497 (after include tidy) |

EXTEND (math-core submodule):

  | File | LOC delta |
  | ---- | --------- |
  | `core/poly_ring/CMakeLists.txt` | +0/-0 (one source listed) |
  | `core/poly_ring/poly_ring_test.c` | +283 (5 new test functions, schoolbook helpers, kBluesteinNs table, rng_coeff_blue, new main entries) |

NEW (engine repo):

  | File | LOC |
  | ---- | --- |
  | `tools/sp_compute_skel/docs/PLAN-NTT-5a.md` | 424 |
  | `tools/sp_compute_skel/docs/CLOSURE-NTT-5a.md` | (this file) |
  | `tools/sp_compute_skel/docs/NTT-5a-wallclock-bench.c` | 62 |

Submodule diff stat (`git diff --stat aeecdba..sprint/ntt-5a` in submodule):
  4 files changed, 868 insertions(+), 1 deletion(-)

Engine repo diff stat (`git diff --stat fec6fe3..sprint/ntt-5a`):
  2 files changed, 425 insertions(+), 1 deletion(-) (the +/- is in
  PLAN-NTT-5a.md + the submodule pointer; the closure + bench files land
  in a separate trailing commit before push)

## Commits on sprint/ntt-5a

Engine repo (`engine-ntt-5a`):

  | SHA | Message |
  | --- | ------- |
  | `3ea3678` | `[plan] NTT.5a -- Bluestein wrapper for arbitrary power-of-2 HD <= 256 in math-core poly_ring` |
  | `32c1e41` | `[NTT.5a] Stage 1 submodule bump: shannon-prime-system -> f29fb68 (Bluestein skeleton + oracle)` |
  | `3cd87ec` | `[NTT.5a] Stage 2 submodule bump: Bluestein init/free + outer-N psi tables` |
  | `ead8e3e` | `[NTT.5a] Stage 3+4 submodule bump: shannon-prime-system Bluestein inner+mul; 4/4 gates PASS` |
  | `1311ad2` | `[NTT.5a] submodule bump: tidy unused include` |
  | (one more for closure + bench, lands before push) |

Submodule (`shannon-prime-system`):

  | SHA | Message |
  | --- | ------- |
  | `f29fb68` | `[NTT.5a] Stage 1: Bluestein header + schoolbook oracle + skeleton (3/9 fails expected -- init stub)` |
  | `701b495` | `[NTT.5a] Stage 2: real init/free + outer-N psi tables + inner ntt_ctx (inner/mul still stubs)` |
  | `51d73bf` | `[NTT.5a] Stage 3+4: Bluestein inner + mul (twist/convolve/fold/untwist/garner); 4/4 gates PASS bit-exact` |
  | `7662b2d` | `[NTT.5a] tidy: drop unused string.h include in poly_ring_bluestein.c` |

**Stage 3+4 bundling disclosure** (per
`feedback-bundled-changeset-root-cause-ambiguity`): the spec's planned
Stage 3 (inner + 2 gates) and Stage 4 (mul + 1 gate) were committed as a
single change (`51d73bf` / `ead8e3e`) because `sp_pr_bluestein_inner` is a
2-line shim over `sp_pr_bluestein_mul` (involution + read c_0; same pattern
as `poly_ring.c:95-104`). The two functions share the same pipeline
(`pr_blue_twist_input` → `pr_blue_convolve_M` → `pr_blue_fold` →
`pr_blue_untwist` → manual Garner). Splitting the implementation would
have either (a) duplicated the pipeline inline in `sp_pr_bluestein_inner`,
or (b) introduced a no-op `sp_pr_bluestein_inner` returning 0 in one
commit and re-routed it through `sp_pr_bluestein_mul` in the next, which
is artificial. The bundled commit body explicitly lists the three gates
that light up together (Stage 3's two + Stage 4's one) and the single
algorithm root cause (the per-prime psi-twist + length-M zero-pad linear
conv pipeline). No "root cause ambiguity" — the three gates pass together
because they exercise the same pipeline.

## Sub-tag candidate

`lat-phase-4-ntt-5a-host-bluestein`. Operator applies post-merge.

## What's NOT done (NTT.5b scope)

- **L1 ABI extension.** The Bluestein context exists only in math-core;
  the L1 host-DSP ABI (`sp_session.h`, `sp_status.h`, `tools/sp_compute_skel/inc/*`)
  doesn't yet have a method-id for invoking it remotely. NTT.5b adds
  method 19 (or whatever's next) for `sp_pr_bluestein_*` cross-process
  dispatch.

- **Hexagon backend.** No HVX kernel work. The host-side math-core path is
  validated; mapping the outer-N psi-twist + fold + manual Garner to V69
  HVX is NTT.5b scope. The inner length-M NTT is already
  silicon-confirmed (NTT.0–NTT.4 lat-phase-4-ntt-4-intt-garner tag),
  so NTT.5b only needs the wrapper layer on-device.

- **forward.c wire-up.** `core/forward/forward.c:115-116` still uses
  `sp_pr_init((uint32_t)HD)` and silently returns NULL for HD ∉ {128,
  256, 512}. NTT.5b replaces this with a dispatch on HD: direct
  `sp_pr_init` for HD ∈ {128, 256, 512}, Bluestein
  `sp_pr_bluestein_init` for HD ∈ {2, 4, 8, 16, 32, 64} (covers Qwen3-0.6B
  and Qwen2.5-Coder-0.5B which both have HD=64), or fall back to direct
  integer dot-product with Barrett for HD with odd factors.

- **HD with odd factors > 1.** Per
  `reference-ntt-bluestein-arbitrary-n-escape:38-46`, Bluestein cannot
  help for N=96, 192, 288, 384, etc. with our current primes (2N would
  need to divide q − 1, but q − 1 has no factor of 3 or 5 in the odd
  part). The SP-aligned answer is direct integer dot-product with
  Barrett. NOT in NTT.5a or NTT.5b scope; document in NTT.5b spec text.

- **Performance engineering.** The 4× `ntt_forward` per call, the
  per-call malloc/free of kstar in `sp_pr_bluestein_inner`, and the
  "compute the whole length-N product even when only c_0 is read" cost
  are all easy NTT.5b optimizations. Wall-clock numbers in §"Wall-clock
  comparison" set the baseline.

- **aarch64-android cross-compile link smoke.** Spec asked for a Stage 5
  cross-compile sanity. The new TU is portable C11 (no platform
  intrinsics or inline asm — `grep -E '__x86_64__|_M_X64|__SSE|__AVX|<intrin|asm volatile|__builtin_' core/poly_ring/poly_ring_bluestein.c`
  reports nothing). Structural cross-compile readiness is therefore
  trivially satisfied; the actual NDK toolchain wasn't reached from
  this worktree without an environment script bring-up. Documented as
  deferred per spec ("downgrade to documented as deferred").

## What unblocks (NTT.5b and downstream)

- **NTT.5b** can dispatch with the Bluestein math validated host-side:
  any deviation on V69 HVX from the bit-exact host reference is a
  Hexagon-side bug, not a math design question. The Stage-3 gate is the
  trusted oracle for NTT.5b's on-device gate.

- **Qwen3-0.6B and Qwen2.5-Coder-0.5B HD=64 NTT-attention overlay** is
  no longer silently failing — once NTT.5b wires it in. Verified on
  host: `sp_pr_bluestein_init(64)` returns a valid context;
  `sp_pr_bluestein_inner(64, q, k)` bit-exactly matches the integer dot
  product.

- **Phase 4-NTT theory completeness for HD ∈ {2..256, ≠ 512} on
  Shannon-Prime's frozen primes.** Mixed-radix and Good-Thomas remain
  banned (operator-side fence in
  `reference-ntt-bluestein-arbitrary-n-escape:48-54`); the host-side
  Bluestein wrapper IS the SP-philosophy-aligned escape for power-of-2 N.

## Memory entry candidates

After operator merge:

1. **Update `reference-ntt-bluestein-arbitrary-n-escape`** with a closing
   line: "NTT.5a implemented this in math-core 2026-05-31 with 4/4 gates
   PASS bit-exact across N ∈ {2..256}; sub-tag `lat-phase-4-ntt-5a-host-
   bluestein`. NTT.5b extends to L1 ABI + Hexagon + forward.c wire-up."

2. **New `reference-poly-ring-bluestein-host-pipeline`** (one-liner index):
   "Host-side Bluestein wrapper for negacyclic R_q poly_ring uses Approach
   A: per-prime psi-twist + zero-pad to length-M ∈ {128,256,512} → math-
   core CRT NTT as a length-(2N−1) linear convolver (4× ntt_forward, 1×
   pointwise, 1× ntt_inverse) → per-prime fold to length-N cyclic →
   per-prime ψ_NP^{-k} untwist → local Garner CRT. Public surface only,
   no per-prime butterfly reimplementation. 3.5–4× slower than direct
   sp_pr_inner at N=128/256 (driven by 2× ntt_forward calls per operand);
   optimizable in NTT.5b."

3. **New `reference-math-core-ntt-crt-recombine-bound`** (one-liner
   index): "`ntt_crt_recombine(ctx, x1, x2, out)` iterates exactly
   `ctx->N` entries (the context's NTT length), not a caller-passed N.
   Calling it with arrays sized at outer-degree N when the NTT is sized
   at inner-degree M (M > N, as in Bluestein wrapping) overruns the
   output buffer. Use a local Garner loop over the actual coefficient
   count instead. Caught NTT.5a 2026-05-31 during Stage 3+4 implementation."

These three are the candidates; operator decides which to commit.

## Worktree status

```
D:\F\shannon-prime-repos\engine-ntt-5a   (engine)
  ├── branch:   sprint/ntt-5a
  ├── base:     fec6fe3 (engine main, post NTT.4 merge)
  ├── tip:      1311ad2 (pre-closure; closure commit lands next)
  ├── 5 commits ahead of base (plan + 4 submodule bumps + tidy)
  └── push:     git push -u origin sprint/ntt-5a

D:\F\shannon-prime-repos\engine-ntt-5a\lib\shannon-prime-system   (math-core submodule)
  ├── branch:   sprint/ntt-5a (created from detached HEAD at aeecdba)
  ├── base:     aeecdbae5a41e35d009fb2c82046ca3bbcd33454
  ├── tip:      7662b2d
  ├── 4 commits ahead of base (Stage 1 + Stage 2 + Stage 3+4 + tidy)
  └── push:     git push -u origin sprint/ntt-5a
```

Anti-contamination check: no edits in any other engine-* or lattice-*
worktree (`grep -r "sp_pr_bluestein" D:/F/shannon-prime-repos/` outside
`engine-ntt-5a` returns nothing). Files untouched in this worktree:
`include/sp/poly_ring.h`, `core/poly_ring/poly_ring.c`,
`include/sp/ntt_crt.h`, `core/ntt_crt/ntt_crt.c`, `core/forward/forward.c`.

Push commands (operator runs after closure ack):

```
cd D:\F\shannon-prime-repos\engine-ntt-5a\lib\shannon-prime-system
git push -u origin sprint/ntt-5a

cd D:\F\shannon-prime-repos\engine-ntt-5a
git push -u origin sprint/ntt-5a
```

Operator merges; applies `lat-phase-4-ntt-5a-host-bluestein` sub-tag.
