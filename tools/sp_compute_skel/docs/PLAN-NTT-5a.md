# PLAN-NTT-5a.md — Sprint NTT.5a Host-side Bluestein wrapper for math-core poly_ring

## Stage 0 — Mandatory pre-read citations

1. `reference-ntt-bluestein-arbitrary-n-escape` memory (read in full):
   `spaces/55dd71db-d563-4af9-a9ce-bc9d22ab62ff/memory/reference_ntt_bluestein_arbitrary_n_escape.md:1-119`
   Mathematical constraint: Bluestein wrapping STILL requires a primitive 2N-th
   root of unity in Z_q for the negacyclic-to-cyclic pre-weight step. For
   Shannon-Prime's frozen Proth primes (q1, q2) with v_2(q_i - 1) = 10, this
   admits 2N ∈ {2, 4, 8, ..., 1024}, hence Bluestein-admissible negacyclic-N
   set = {2, 4, 8, 16, 32, 64, 128, 256} (NOT 512: 2·512 = 1024 = 2^10 is OK,
   but 512 is already covered by direct sp_pr_init — spec explicit). Explicit
   bans: mixed-radix, Good-Thomas, zero-padding HD to next admissible
   sp_pr_init N, adding a third prime, Cooley-Tukey at non-power-of-2 N.
   Surface UPSTREAM if any of these surface from analysis.

2. `reference-ntt-frozen-primes-N-cap` companion memory:
   `spaces/55dd71db-d563-4af9-a9ce-bc9d22ab62ff/memory/reference_ntt_frozen_primes_N_cap.md:1-79`
   q1 = 1073738753, q2 = 1073732609; both have v_2(q-1) = 10, so max negacyclic
   N = 512 with current primes. Math-core's `ntt_init(N)` enforces
   N ∈ {128, 256, 512} at `lib/shannon-prime-system/core/ntt_crt/ntt_crt.c:189`.
   This means the **inner Bluestein NTT length M** must also be in
   {128, 256, 512}, not arbitrarily chosen.

3. `reference-sp-uses-phi-extensively` memory was referenced in the dispatch
   but not located in the indexed memory listing at session start. Stage 0
   acknowledgement: SP already uses the golden ratio φ in Fibonacci-Prime DHT,
   Fibonacci KV sub-sampling, `SP_ROPE_PHI`, Halton/Sobol QMC. Bluestein here
   is purely a discrete-NTT host-side wrapper; no φ-based "new framework" is
   proposed.

4. Math-core poly_ring API:
   `lib/shannon-prime-system/include/sp/poly_ring.h:1-82` (full file)
   `lib/shannon-prime-system/core/poly_ring/poly_ring.c:1-137` (full file)
   - `sp_pr_init(N)` accepts N ∈ {128, 256, 512} only (`poly_ring.h:43-46`,
     `poly_ring.c:31-32` defers to `ntt_init`).
   - `sp_pr_inner(ctx, q, k)` returns int64; uses involution
     `k*_0 = k_0, k*_j = -k_{N-j}` then reads coefficient 0 of negacyclic
     product (poly_ring.h:62-66, poly_ring.c:82-104).
   - `sp_pr_mul(ctx, a, b, out)` writes signed-centered int64 result to N
     entries of `out`, must not alias inputs (poly_ring.h:55-60).
   - Coefficient range invariant for bit-exactness: |<q,k>| < M/2 where
     M = q1·q2 ~ 2^60 (poly_ring.h:24, poly_ring.c:93).

5. Math-core CRT NTT (CALLER only, do not modify):
   `lib/shannon-prime-system/core/ntt_crt/ntt_crt.c:188-215` `ntt_init` —
   rejects N ∉ {128,256,512}, sets up twiddle tables per prime.
   `lib/shannon-prime-system/core/ntt_crt/ntt_crt.c:259-278` `forward_one` +
   `ntt_forward` — psi pre-weight then in-place Cooley-Tukey.
   `lib/shannon-prime-system/core/ntt_crt/ntt_crt.c:281-294` `inverse_one` /
   `lib/shannon-prime-system/core/ntt_crt/ntt_crt.c:323-336` `ntt_inverse` —
   inverse NTT + N^{-1} scaling + post-weight + CRT recombine.
   `lib/shannon-prime-system/core/ntt_crt/ntt_crt.c:340-351`
   `ntt_pointwise_mul` — per-residue multiply in NTT domain.
   `lib/shannon-prime-system/core/ntt_crt/ntt_crt.c:303-321` `ntt_crt_recombine`
   — standalone signed Garner CRT.
   Public surface: `lib/shannon-prime-system/include/sp/ntt_crt.h:46-91`,
   exposing `SP_NTT_Q1`, `SP_NTT_Q2`, `SP_NTT_M`, `ntt_init`, `ntt_free`,
   `ntt_forward`, `ntt_inverse`, `ntt_pointwise_mul`, `ntt_crt_recombine`.

6. Forward.c NTT-attention call site:
   `lib/shannon-prime-system/core/forward/forward.c:115-120` —
   `pr = sp_pr_init((uint32_t)HD)`; comment marks the existing restriction
   "head_dim must be in {128,256,512}". Silently fails for Qwen3-0.6B /
   Qwen2.5-Coder-0.5B which both have HD=64. NTT.5b will swap in
   `sp_pr_bluestein_init(HD)` here (out of NTT.5a scope).
   `lib/shannon-prime-system/core/forward/forward.c:188-211` —
   `sp_pr_inner(pr, qi, ki)` returning int64 per (t, h, s) attention pair.
   The Bluestein API matches this signature exactly so NTT.5b can drop in.

## Chirp formula derivation in Z_q

Two equivalent formulations of "Bluestein-wrapped negacyclic length-N
convolution via length-M inner NTT" exist. I derive both, prove their
equivalence, and choose the implementation form on simplicity grounds.

### Approach A — psi-twist + zero-pad linear convolution (chosen)

We want negacyclic length-N product over R = Z_q[x]/(x^N + 1):

    c_k = sum_{i+j=k}      a_i b_j         (i,j ∈ [0,N), i+j < N)
        - sum_{i+j=k+N}    a_i b_j         (i+j wraps; x^N = -1 sign flip)

**Step 1 — psi-twist (negacyclic → cyclic).** Let ψ_N be a primitive 2N-th
root of unity in Z_q (ψ_N^N = -1, ψ_N^{2N} = 1). Define

    A_j = a_j · ψ_N^j  mod q,   B_j = b_j · ψ_N^j  mod q.

Their length-N cyclic convolution C satisfies, for k ∈ [0,N):

    C_k = sum_{i+j ≡ k (mod N)} A_i B_j
        = sum_{i+j=k}    a_i b_j ψ_N^{i+j}     +    sum_{i+j=k+N} a_i b_j ψ_N^{i+j}
        = sum_{i+j=k}    a_i b_j ψ_N^k         +    sum_{i+j=k+N} a_i b_j ψ_N^{k+N}
        = ψ_N^k · [ sum_{i+j=k} a_i b_j  -  sum_{i+j=k+N} a_i b_j ]    (∵ ψ_N^N = -1)
        = ψ_N^k · c_k.

Therefore   **c_k = C_k · ψ_N^{-k}   mod q.**

Pre-weight = `ψ_N^j`; post-weight = `ψ_N^{-k}`. Identical mechanism to
math-core's existing `forward_one`/`inverse_one` (ntt_crt.c:259-294), just
with `N` replaced by our shorter Bluestein degree.

**Step 2 — cyclic length-N → linear length-(2N-1) → length-M negacyclic NTT.**
Zero-pad A and B to length M, where M ∈ {128, 256, 512} is the smallest
admissible inner-NTT length satisfying M ≥ 2N. Set

    Ã_i = A_i  for i < N,    Ã_i = 0  for N ≤ i < M.
    B̃_i = B_i  for i < N,    B̃_i = 0  for N ≤ i < M.

Compute the math-core length-M **negacyclic** product

    D = Ã ⊛_{neg, M} B̃,  i.e.  D_k = sum_{i+j=k} Ã_i B̃_j  -  sum_{i+j=k+M} Ã_i B̃_j.

Because Ã and B̃ vanish for index ≥ N, the highest non-zero index in the
linear product Ã*B̃ is (N-1) + (N-1) = 2N-2 < M. Therefore the i+j=k+M
wraparound term is identically zero for all k ∈ [0, M). So

    D_k = (Ã * B̃)_k     for k ∈ [0, 2N-1)
    D_k = 0              for k ∈ [2N-1, M).

This makes the length-M negacyclic NTT act as a **pure linear convolver** of
the zero-padded inputs. Math-core's `ntt_forward(M)`, `ntt_pointwise_mul`,
`ntt_inverse` give us D bit-exactly.

**Step 3 — fold linear length-(2N-1) → cyclic length-N.** The cyclic length-N
convolution C is recovered by folding the linear product:

    C_k = D_k + D_{k+N}     for k ∈ [0, N-1)
    C_{N-1} = D_{N-1}       (D_{2N-1} index does not exist; D[2N-1..M) = 0)

We can encode this uniformly as

    C_k = D_k + (k+N < M ? D_{k+N} : 0)     for k ∈ [0, N).

For our choice M ≥ 2N: when N=2N (the boundary case), the maximum index of D
that contributes is 2N-2 = N + (N-2), so D_{k+N} for k ∈ [0, N-1) reaches
indices [N, 2N-1), all of which are within [0, M). D_{N-1+N} = D_{2N-1} = 0
by the linear-conv argument. So the formula collapses to the conditional
above, exact.

**Step 4 — post-untwist.** Apply `c_k = C_k · ψ_N^{-k} mod q`.

**Step 5 — CRT recombine and sign-center.** Carry out steps 1-4 per prime
(q1, q2), then call `ntt_crt_recombine` to produce signed-centered int64
output ∈ (-M_full/2, M_full/2] where M_full = q1·q2.

This is the **chosen implementation form**.

### Approach B — classical chirp-z Bluestein (equivalent; reference only)

The textbook Bluestein chirp transform writes (for cyclic length-N transform
with N-th root χ):

    Y_k = sum_j A_j χ^{jk}
        = χ^{k²/2} sum_j (A_j χ^{j²/2}) χ^{-(k-j)²/2}     using jk = (j² + k² - (k-j)²)/2.

In Z_q (odd prime), the "/2" is well-defined via 2^{-1} mod q for **integer**
exponents. For half-integer exponents j²/2 when j² is odd, we sidestep by
using a 2N-th root η with η² = χ, so χ^{j²/2} = η^{j²} is always defined.

**This is a transform, not a convolution.** To use it for our negacyclic
**convolution** c = a ⊛ b, we would:

  - Bluestein-transform A = ψ-twisted a, B = ψ-twisted b to F_A, F_B;
  - Pointwise multiply F_A · F_B;
  - Bluestein-inverse-transform back to C;
  - Post-untwist by ψ_N^{-k}.

The pointwise-multiply step requires a length-N **cyclic** transform pair
(forward + inverse), which Bluestein gives via TWO chirp wraps (forward and
inverse). Each chirp wrap is itself a length-M cyclic convolution implemented
via the inner NTT. **Total work: 4 chirp wraps + 4 inner length-M NTTs +
2 pointwise multiplies + 1 outer pointwise multiply ≈ 4× the work of
Approach A** (which uses 3 inner NTTs total: 2 forward + 1 inverse, exact
mirror of `sp_pr_mul`).

**Equivalence.** Both approaches compute the cyclic length-N convolution C
of two length-N inputs in Z_q, then post-untwist. Their results are bit-exact
identical (cyclic convolution is unique up to representation). Approach A is
strictly more efficient and reuses math-core's existing convolution
primitives unchanged.

### Decision

**Approach A.** Rationale:
- Reuses math-core's negacyclic length-M NTT as a length-(2N-1) linear
  convolver via zero-padding — a well-known idiom, no new chirp tables.
- 3 inner length-M transforms per Bluestein call (matches sp_pr_mul cost
  shape), vs Approach B's 4.
- No half-integer-exponent ambiguity in Z_q (only ψ_N twist needed, which
  math-core already validates internally via `find_psi`).
- Bit-exact correctness is structurally obvious from the linear-conv→fold
  identity.

The spec's chirp framing is captured in Approach B above as the alternative
form for documentation completeness. The closure document will quote the
chosen chirp formula (Approach A's psi-twist + zero-pad fold) explicitly.

## Bluestein admissible-N analysis

Constraints:
- Approach A pre-weight needs ψ_N (primitive 2N-th root), so 2N | (q-1).
  For our frozen primes, this means 2N ∈ {2, 4, 8, ..., 1024}, equivalently
  N ∈ {1, 2, 4, ..., 512}.
- Inner NTT length M must be in {128, 256, 512} (math-core's `ntt_init`
  constraint).
- Linear convolver correctness: M ≥ 2N.
- Spec: Bluestein covers {2, 4, ..., 256}; N=512 stays on direct
  `sp_pr_init` (which uses M=512 cyclic natively, no zero-padding overhead).

Resulting M choice per N:

  | N    | smallest M satisfying M ≥ 2N and M ∈ {128,256,512} |
  | ---- | -------------------------------------------------- |
  | 2    | 128 (2N = 4, but math-core minimum is 128)         |
  | 4    | 128                                                |
  | 8    | 128                                                |
  | 16   | 128                                                |
  | 32   | 128                                                |
  | 64   | 128                                                |
  | 128  | 256                                                |
  | 256  | 512                                                |
  | 512  | (NOT BLUESTEIN — direct sp_pr_init)                |

N=1 has no useful interpretation as polynomial product (trivial 1-element
result), and the spec explicitly lists {2..256}.

Rejected N values that must return NULL: {1, 3, 5, 6, 7, 9, 96, 100, 384,
512, 1024} per spec. 512 is rejected from the Bluestein API because the
direct path is more efficient — Bluestein at N=512 would need M=1024, which
math-core cannot provide (would violate frozen-primes cap). All other
non-power-of-2 values are rejected because 2N would not divide (q-1).

## Files expected to change

NEW:
- `lib/shannon-prime-system/core/poly_ring/poly_ring_bluestein.c`
- `lib/shannon-prime-system/include/sp/poly_ring_bluestein.h`

EXTEND:
- `lib/shannon-prime-system/core/poly_ring/poly_ring_test.c`
- `lib/shannon-prime-system/core/poly_ring/CMakeLists.txt`

NEW (closure):
- `tools/sp_compute_skel/docs/CLOSURE-NTT-5a.md`

UNTOUCHED (anti-contamination):
- `lib/shannon-prime-system/include/sp/poly_ring.h`
- `lib/shannon-prime-system/core/poly_ring/poly_ring.c`
- `lib/shannon-prime-system/include/sp/ntt_crt.h`
- `lib/shannon-prime-system/core/ntt_crt/ntt_crt.c`
- `lib/shannon-prime-system/core/forward/forward.c`
- All other engine-*/lattice-* worktrees

## Public API (final shape)

```c
/* poly_ring_bluestein.h */
#ifndef SP_POLY_RING_BLUESTEIN_H
#define SP_POLY_RING_BLUESTEIN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sp_pr_bluestein_ctx sp_pr_bluestein_ctx;

sp_pr_bluestein_ctx *sp_pr_bluestein_init(uint32_t N);
void                 sp_pr_bluestein_free(sp_pr_bluestein_ctx *ctx);
uint32_t             sp_pr_bluestein_degree(const sp_pr_bluestein_ctx *ctx);

int64_t sp_pr_bluestein_inner(sp_pr_bluestein_ctx *ctx,
                              const int32_t *q, const int32_t *k);

void    sp_pr_bluestein_mul(sp_pr_bluestein_ctx *ctx,
                            const int32_t *a, const int32_t *b,
                            int64_t *out);

#ifdef __cplusplus
}
#endif

#endif /* SP_POLY_RING_BLUESTEIN_H */
```

## Multi-file stage plan

**Stage 1 — Schoolbook reference + unit test (no Bluestein impl yet).**
Add `bluestein_schoolbook_inner` and `bluestein_schoolbook_negacyclic_mul`
helpers in the test file. These are the trusted oracle: simple O(N²) loop
identical in structure to the existing `schoolbook_negacyclic` in
poly_ring_test.c (lines 54-72), but parameterized over the Bluestein
admissible N range. Verify the new helpers cross-check against the existing
schoolbook on overlapping N ∈ {128, 256}. Header + skeleton init/free
return NULL. CMakeLists wired to compile the new TU but no behavior tests
yet.

**Stage 2 — Context init + chirp/twiddle precompute, init/free/degree.**
- Allocate inner `ntt_ctx *ntt` via `ntt_init(M)` per admissible N
  (M-table from analysis above).
- Precompute `psi_pow[N]` (primitive 2N-th root powers for negacyclic→cyclic
  pre-weight) per prime. We don't have direct access to math-core's internal
  `find_psi`/`prime_setup`, so we re-derive ψ_N from the public surface:
  call `ntt_init(N')` for a smaller helper context where N' is in
  {128, 256, 512} that admits a 2N-th root — wait, this is circular.
- **Alternative:** carry out ψ_N derivation in poly_ring_bluestein.c using
  the same primitive search math-core uses (Fermat-style modpow), restricted
  to public constants `SP_NTT_Q1`, `SP_NTT_Q2`. This is the chosen path.
  Self-contained modular arithmetic helpers (modpow, modinv, modmul) live
  in the new TU; ~50 LOC of arithmetic, no dependency on ntt_crt internals.
- Precompute `ipsi_pow[N]` = ψ_N^{-j} for post-untwist.
- N-table lookup for inner length M and the small zero-pad scratch buffers
  (input length M signed int32 padded with zeros; per-prime residue
  vectors length M; recombined product length M int64).

Init returns NULL for N ∉ {2, 4, 8, 16, 32, 64, 128, 256}. Free is no-op
for NULL.

**Stage 3 — sp_pr_bluestein_inner.** Implement involution-then-mul-then-c0.
Per-prime: pad-and-psi-twist, ntt_forward, pointwise mul, ntt_inverse,
fold, post-untwist (only at k=0 to recover c_0). Then signed Garner CRT
recombine for k=0 only. Gates:
- T_NTT5A_BLUESTEIN_BIT_EXACT_VS_SCHOOLBOOK
- T_NTT5A_VS_SP_PR_INNER_BIT_EXACT (only N ∈ {128, 256})

**Stage 4 — sp_pr_bluestein_mul.** Full-result variant. Per-prime: pad-and-
psi-twist, ntt_forward, pointwise mul, ntt_inverse, fold, post-untwist for
all k ∈ [0, N), then CRT recombine all N coefficients. Gate:
- T_NTT5A_BLUESTEIN_MUL_BIT_EXACT.

Implementation note: `sp_pr_bluestein_inner` can be expressed in terms of
`sp_pr_bluestein_mul` (same involution trick as poly_ring.c:95-104), but
inner reads only c_0 so we keep an inner-only fast path that skips the
length-N post-untwist + recombine for k>0.

**Stage 5 — T_NTT5A_NULL_FOR_INADMISSIBLE_N + closure.**

**Anti-contamination one-variable-per-stage discipline.** Each stage commits
ONE behavior change. Stage 1 = oracle. Stage 2 = init only (init/free return
NULL or context skeleton). Stage 3 = inner + 2 gates. Stage 4 = mul + 1 gate.
Stage 5 = NULL-for-bad-N gate + closure.

## Coefficient range invariant for bit-exactness

Schoolbook reference inputs drawn from |coeff| < 2^14 (spec value).
With N ≤ 256 and |coeff| < 2^14:
- |single product| < 2^28
- |linear-conv coefficient| < N · 2^28 ≤ 256 · 2^28 = 2^36
- |inner product| < N · 2^28 ≤ 2^36
- M_full = q1·q2 ≈ 2^60 → M_full/2 ≈ 2^59 ≫ 2^36

Centered recovery is exact. No overflow into the wraparound window.

(Sanity: the existing poly_ring_test uses |coeff| < 2^23 at N ≤ 512; with
N=256 this gives max accumulator 2^55, still inside 2^59. Our tighter
spec'd range gives more headroom.)

## Wall-clock comparison plan (informational, not a gate)

Stage 4 closure measures:
- `sp_pr_bluestein_inner(N=128)` vs `sp_pr_inner(N=128)`, both averaged over
  1000 calls; expect Bluestein 2-4× slower (zero-pad doubles inner-NTT
  length, no engineering optimizations).
- Same at N=256.
This is documentation, not pass/fail. NTT.5b will exercise wall-clock as a
target on real hardware.

## Build / test discipline

- Host-only: linux x86 via WSL ubuntu, OR Windows MSVC standalone.
  Build path: `cmake -S lib/shannon-prime-system/core/poly_ring -B build-pr -G "Ninja"`.
- Standalone module build (the `sp_add_module` helper supports it via
  `add_subdirectory(../ntt_crt ...)` per cmake/sp_module.cmake:65-73).
- `ctest -V -R T_PR` runs all four T_PR_* + the new four T_NTT5A_* tests.
- aarch64-android cross-compile sanity check Stage 5 (link smoke; no
  on-device run) — gated only on the host build succeeding and the new
  TU compiling under `cmake/toolchain-hexagon.cmake` if accessible.
  If the cross toolchain is not accessible from worktree, downgrade to
  "documented as deferred"; this is a follow-on sanity check, not a gate.

## Banned propositions audit

None of the chosen approach invokes any of:
- mixed-radix NTT (Approach A is plain radix-2 length-M)
- Good-Thomas (irrelevant; no prime factorization of N involved)
- zero-padding HD to next admissible sp_pr_init N (Bluestein result is
  for the original N, not for a padded N; the zero-pad is inside the
  inner length-M and the OUTPUT is folded back to length N)
- adding a third prime (we use the same frozen q1, q2)
- Cooley-Tukey at non-power-of-2 N (the inner NTT is at M ∈ {128,256,512})

Surface UPSTREAM if any gate failure analysis suggests a banned approach.

## Closure structure (preview)

`tools/sp_compute_skel/docs/CLOSURE-NTT-5a.md`:
1. Headline
2. Chirp formula derivation (this PLAN's "Approach A" section, recapped)
3. Admissible-N analysis (this PLAN's table)
4. Gates table (4 gates with PASS/FAIL + observed numbers)
5. Wall-clock comparison (informational)
6. Public API documentation (final header file contents)
7. Files changed with LOC delta
8. Commits on `sprint/ntt-5a`
9. Sub-tag candidate: `lat-phase-4-ntt-5a-host-bluestein`
10. NOT done — L1 ABI extension (NTT.5b), Hexagon backend (NTT.5b),
    forward.c wire-up, non-power-of-2 HD
11. Unblocks — NTT.5b can dispatch with Bluestein math validated host-side
12. Memory entry candidates
13. Worktree status

## Worktree

`D:\F\shannon-prime-repos\engine-ntt-5a` on branch `sprint/ntt-5a`.
Base: engine main @ fec6fe3 (post NTT.4 merge, confirmed via `git log`).
Submodule `lib/shannon-prime-system` at aeecdbae5a41e35d009fb2c82046ca3bbcd33454.
Anti-contamination strict: no edits in other engine-* or lattice-*
worktrees.

Push at end: `git push -u origin sprint/ntt-5a`. Operator merges.

Sub-tag candidate: `lat-phase-4-ntt-5a-host-bluestein`.
