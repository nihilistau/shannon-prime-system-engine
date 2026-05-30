# PLAN-NTT-4.md — §4-NTT Sprint NTT.4 — INTT + ARM Garner round-trip

Plan commit for Sprint NTT.4. Worktree: `D:\F\shannon-prime-repos\engine-ntt-4`
on `sprint/ntt-4` (base: engine main @ c6df266, post-NTT.2 merge).

## Stage 0 — Mandatory reference citations

### 1. Math-core `inverse_one`
`lib/shannon-prime-system/core/ntt_crt/ntt_crt.c:281-294`

```c
/* inverse transform for one prime, producing residues in [0,q). */
static void inverse_one(const ntt_ctx *ctx, const prime_ctx *pc,
                        const uint32_t *in, uint32_t *out) {
    uint32_t N = ctx->N;
    uint32_t q = pc->q;
    uint64_t mu = pc->mu;

    for (uint32_t j = 0; j < N; j++) out[j] = in[j] % q;
    ntt_core(ctx, pc, out, pc->w_inv);
    /* scale by N^{-1}, then post-weight coeff j by psi^{-j} */
    for (uint32_t j = 0; j < N; j++) {
        uint32_t s = modmul(out[j], pc->ninv, q, mu);
        out[j] = modmul(s, pc->ipsi_pow[j], q, mu);
    }
}
```

INTT structure: input-reduce → forward butterfly kernel with `w_inv` → scale by
`ninv` → multiply by `ipsi_pow[j]`. The butterfly kernel is the SAME function
`ntt_core` used by forward — it doesn't know which twiddle set it processes.

### 2. Math-core `garner_one` (signed centered output)
`lib/shannon-prime-system/core/ntt_crt/ntt_crt.c:303-317`

```c
static inline int64_t garner_one(const ntt_ctx *ctx, uint32_t x1, uint32_t x2) {
    uint32_t q1 = ctx->p1.q;
    uint32_t q2 = ctx->p2.q;
    uint64_t mu2 = ctx->p2.mu;

    uint32_t d = modsub(x2 % q2, x1 % q2, q2);            /* (x2 - x1) mod q2 */
    uint32_t t = (uint32_t)barrett_reduce((uint64_t)d * ctx->q1_inv_mod_q2,
                                          (uint64_t)q2, mu2);
    uint64_t r = (uint64_t)x1 + (uint64_t)q1 * (uint64_t)t; /* < M */
    int64_t v = (int64_t)r;
    int64_t M = (int64_t)ctx->M;
    if (v > M / 2) v -= M;
    return v;
}
```

Centering logic: lift unsigned residue `r ∈ [0, M)` then if `r > M/2`, subtract
`M` to land in `(-M/2, M/2]`.

### 3. Math-core `ntt_inverse` (end-to-end INTT + Garner)
`lib/shannon-prime-system/core/ntt_crt/ntt_crt.c:323-338`

```c
void ntt_inverse(const ntt_ctx *ctx, const uint32_t *in1, const uint32_t *in2,
                 int64_t *out) {
    uint32_t N = ctx->N;
    uint32_t *t1 = malloc(sizeof(uint32_t) * N);
    uint32_t *t2 = malloc(sizeof(uint32_t) * N);
    if (!t1 || !t2) { free(t1); free(t2); return; }

    inverse_one(ctx, &ctx->p1, in1, t1);
    inverse_one(ctx, &ctx->p2, in2, t2);
    ntt_crt_recombine(ctx, t1, t2, out);

    free(t1);
    free(t2);
}
```

Signature: takes two u32 input residue vectors (post-pointwise-mul), produces
one signed-centered i64 output vector.

### 4. NTT.1 HVX butterfly stage function
`tools/sp_compute_skel/src_dsp/sp_compute_ntt_hvx_imp.c:228-256`

```c
static void sp_ntt_butterfly_stage_hvx(uint32_t *out, uint32_t N,
                                       uint32_t len, uint32_t half,
                                       const uint32_t *w_compact,
                                       HVX_Vector vq, HVX_Vector vq_m1,
                                       HVX_Vector vmu);
```

The HVX butterfly kernel is direction-agnostic — it accepts a stride-1
twiddle pointer `w_compact[half]` (where each stage's twiddles are packed
contiguously). NTT.4 calls it with `w_inv_stages` from VTCM instead of
recomputing per-stage `w_compact`.

Decision: extract helpers + butterfly into shared header
`sp_compute_ntt_hvx_shared.h`. NTT.1's TU shrinks (helpers move) but
emits byte-identical SASS for the butterfly stage. NTT.4's TU includes
the same header. SASS audit gates remain valid.

### 5. NTT.2 VTCM table accessor
`tools/sp_compute_skel/src_dsp/sp_compute_ntt_twiddle.c:441-462`
(function `sp_compute_ntt_twiddle_view`)

```c
typedef struct {
    uint32_t        N;
    uint32_t        q;
    uint64_t        mu;
    uint32_t        ninv;
    const uint32_t *psi_pow;
    const uint32_t *ipsi_pow;
    const uint32_t *w_fwd;
    const uint32_t *w_inv;
    const uint32_t *w_fwd_stages;
    const uint32_t *w_inv_stages;
} sp_tw_view;

int sp_compute_ntt_twiddle_view(int q_idx, int N, sp_tw_view *out);
```

All INTT tables (`ipsi_pow`, `w_inv`, `w_inv_stages`, `ninv`) are
precomputed and ready — NTT.4 just consumes them.

Per-stage compaction layout (sp_compute_ntt_twiddle.c:226-235):
```
offset_stage[s] = 2^{s-1} - 1 entries from base of w_inv_stages
size_stage[s]   = 2^{s-1} entries
```

NTT.4 indexes `w_inv_stages + stage_offset` for each stage and passes it
verbatim to the existing butterfly kernel — no per-call compaction loop.

### 6. K.beta.2.5c unsigned Garner (DO NOT MODIFY)
`tools/sp_dsp_smoke/src/sp_matmul_q_ref.rs:73-92`

NTT.4 ADDS a sibling `garner_combine_q1_q2_signed(r1, r2) -> Vec<i64>`
matching math-core's `garner_one` centering. K.beta.2.5c's existing
gates continue to consume the unsigned version unchanged.

### 7. reference-ntt-frozen-primes-N-cap
N ∈ {128, 256, 512}.

### 8. Discipline references
- `feedback-no-silent-gate-revisions`
- `feedback-lead-with-reference-then-theory`
- `feedback-bundled-changeset-root-cause-ambiguity`

## Coordination with NTT.3

NTT.3 anticipates method 17; NTT.4 anticipates method 18 for
`intt_hvx_oracle`. Different files; closure documents anticipated slot.

## Scope

### Stage 1 — HVX INTT skel (method 18)

- NEW: `sp_compute_ntt_hvx_shared.h` — extracted HVX helpers + butterfly.
- MODIFIED: `sp_compute_ntt_hvx_imp.c` — replace inline helpers with
  `#include "sp_compute_ntt_hvx_shared.h"`. Functional zero-delta.
- NEW: `sp_compute_ntt_intt_imp.c` — INTT IDL handler.
  1. `out[j] = in[j] % q`
  2. bit-reversal permutation
  3. logN stages of butterflies with `w_inv_stages` from VTCM
  4. post-pass scalar: `out[j] = (out[j] * ninv * ipsi_pow[j]) mod q`
- MODIFIED: IDL — `intt_hvx_oracle` (method 18).
- MODIFIED: CMakeLists.txt — add new TU.

Gate: T_NTT4_INTT_BIT_EXACT.

### Stage 2 — Signed Garner ARM-side

- MODIFIED: `sp_matmul_q_ref.rs` — add `garner_combine_q1_q2_signed`
  AFTER the existing unsigned function. Add unit test.

Gate: T_NTT4_GARNER_SIGNED_BIT_EXACT (1000 pairs, 0 divergences).

### Stage 3 — End-to-end polynomial multiplication smoke

- NEW: `sp_ntt_4_polymul_smoke.rs` — driver:
  1. random a, b
  2. forward NTT (method 13) per-prime
  3. pointwise mul per-prime (ARM-side Barrett)
  4. INTT (method 18) per-prime
  5. ARM signed Garner → i64 output
  6. host-side math-core `ntt_inverse` for reference
  7. element-wise compare
- MODIFIED: Cargo.toml — `[[bin]]` entry.

Gate: T_NTT4_POLY_MUL_EXACT (12 runs at 3 N × 4 seeds; 0 divergences).

### Stage 4 — Closure

`CLOSURE-NTT-4.md`.

## Math-core FFI extension for Stage 3

Add `ntt_inverse` and `ntt_pointwise_mul` extern declarations alongside
existing `ntt_init/ntt_forward/ntt_free`. The static lib already
exports them per the .c signatures.

## Risk register

- R1: HVX shared header SASS audit. Mitigation: `static inline`
  helpers; consumer SASS unchanged.
- R2: NTT.3 method-17 collision. Anticipated; closure documents.
- R3: math-core `ntt_inverse` FFI exposure. Function has global
  linkage (no `static`); linking works.
- R4: VTCM-table prerequisite. INTT depends on `ntt_twiddle_init`;
  smoke harness calls it at start.

## Stage commit messages (planned)

1. `[plan] NTT.4 — INTT + ARM signed Garner round-trip (HVX butterfly reuse via w_inv_stages)`
2. `[NTT.4] feat: Stage 1 — HVX INTT skel + IDL method 18 (T_NTT4_INTT_BIT_EXACT)`
3. `[NTT.4] feat: Stage 2 — ARM-side signed Garner sibling (T_NTT4_GARNER_SIGNED_BIT_EXACT)`
4. `[NTT.4] test: Stage 3 — end-to-end polynomial multiplication smoke on S22U (T_NTT4_POLY_MUL_EXACT)`
5. `[NTT.4] doc: Stage 4 — closure`
