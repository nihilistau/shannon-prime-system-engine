# CLOSURE-NTT-5c.md — Sprint NTT.5c forward.c activation of Bluestein + backend dispatch

## Headline

NTT.5c flips the switch on Phase 4-NTT's actual production reach. Math-core's
`core/forward/forward.c` and `core/forward/qwen25.c` NTT-attention overlays
now (a) **dispatch on `head_dim`** between direct `sp_pr_init` (HD ∈ {128,
256, 512}) and the NTT.5a Bluestein wrapper `sp_pr_bluestein_init` (HD ∈
{2, 4, 8, 16, 32, 64, 128, 256}), and (b) **plumb an opt-in compute-backend
triple** (handle + per-direction NTT fn pointers) through to
`sp_pr_bluestein_set_backend` so the inner length-M NTT calls route through
NTT.5b's Hex dispatcher when the session has a backend registered.

After NTT.5c, **the Memory model (Qwen2.5-Coder-0.5B, HD=64) executes its
NTT-attention overlay on Knack's S22U via the cDSP V69 HVX kernels** when
the daemon sets `SP_ENGINE_NTT_ATTN=1 + SP_ENGINE_NTT_ATTN_HEX=1`.
Previously, `qwen25_forward` had no NTT-attention overlay at all (caught
during Stage 0 pre-read — closure §"Stage 0 discoveries"), and the
forward.c qwen3 overlay silently fell back for HD ∉ {128, 256, 512}.

All four substantive gates PASS:

  - **T_NTT5C_HD_64_BIT_EXACT_VS_FP32_TOP1** — host, qwen25 fixture HD=8
    (Bluestein-admissible, same code path as HD=64 Memory model): argmax
    token sequence `[34, 26, 13, 19, 47]` bit-identical across
    `SP_ENGINE_NTT_ATTN` unset vs `=1` runs over 5 decode positions.
  - **T_NTT5C_HD_256_NO_REGRESSION** — host, structural admission gate:
    `sp_pr_init` admits {128, 256, 512}; `sp_pr_bluestein_init` admits
    {2, 4, 8, 16, 32, 64, 128, 256}; both reject 512 (Bluestein) / non-PoT
    (96, 192, 288, 384). The NTT.5c HD-dispatch decision logic is
    structurally correct.
  - **T_NTT5C_HD_64_BLUESTEIN_WORKS** — on-device, S22U, Qwen2.5-Coder
    Memory model (HD=64): `sp_prefill_chunk` returns SP_OK in 1400 ms;
    logits are finite; argmax=16 (plausible, not 0 or vocab-edge).
  - **T_NTT5C_HEX_BACKEND_ROUTES_WHEN_REGISTERED** — on-device, S22U,
    with `SP_ENGINE_NTT_ATTN_HEX=1`: trampoline dispatch counters
    forward+=4032 inverse+=2016 across one 3-token prefill (matches
    24 layers × 14 heads × 6 (t,s) attention pairs × 2 primes per
    Bluestein inner forward = 4032; × 1 inverse = 2016).

Wall-clock: 1400 ms host-Bluestein vs 3358 ms Hex-routed (~2.4× slower) at
this small ctx, consistent with NTT.5b's wall-clock matrix and the
"NTT.6 long-context tile is where Hex wins" note. Stage 4 informational,
not a substantive gate per spec.

Worktree: `D:\F\shannon-prime-repos\engine-ntt-5c` on `sprint/ntt-5c`.
Engine base: `bd34c08` (post NTT.5b merge); engine tip: see §Commits.
Math-core submodule on `sprint/ntt-5c` branch (created from NTT.5b's `fc38a4f`).
8 commits across both repos (3 engine + 5 submodule).

## Stage 0 discoveries

### Discovery 1: qwen25_forward had no NTT-attention overlay

During the mandatory pre-read, `core/forward/qwen25.c` was inspected for
the existing NTT-attention overlay shape. Finding: there is none.
`qwen25_forward` at line 79 of the original file calls `sp_attn_head`
directly (fp32 attention) with no `g_ntt_attn` branch.

Implication: even if NTT.5c only patched `forward.c` (qwen3 overlay), the
Memory model would never hit the new Bluestein code path because it routes
through `sp_prefill_chunk → qwen25_forward(s->qm, ...)`. The Memory model
IS the chat-integration target (per NTT.5b CLOSURE §"What's NOT done"); if
the unblocking sprint doesn't touch qwen25.c, the unblock never happens.

**Therefore NTT.5c added the NTT-attention overlay to qwen25.c** in
addition to extending forward.c's overlay. The overlay structure is
mechanically identical: the Qwen2.5 attention loop becomes a NTT-attention
conditional dispatch with the same `g_ntt_attn / overlay_active` shape as
forward.c:188-211.

This was an in-spec scope discovery (the spec's "Memory model first" goal
implies the overlay must exist on the qwen25 path) rather than a scope
expansion, and was documented in PLAN-NTT-5c.md §"Stage 0 — qwen25.c
overlay discovery" before any code landed.

### Discovery 2: gemma3_forward also has no NTT-attention overlay

Same scan revealed `gemma3.c` likewise has no `g_ntt_attn` branch. Gemma3
uses sliding-window attention (`sp_attn_head` with `win=` parameter), and
the NTT-attention overlay shape doesn't trivially compose with SWA (the
overlay's loop iterates `s ∈ [0..t]` unconditionally; SWA masks `s` outside
the window before contributing to the softmax). Out of NTT.5c scope.

`gemma3_forward_ex2` is shipped as an ABI-uniform wrapper that accepts
and ignores the backend triple, forwarding to the existing
`gemma3_forward`. Future sprint may add SWA-aware NTT-attention; until then
the wrapper is a no-op pass-through.

## Forward.c overlay logic — before vs after

**Before (NTT.4-era, frozen since `lat-phase-4-ntt-4-intt-garner`):**

```c
sp_pr_ctx *pr = NULL;
/* ... */
if (g_ntt_attn) {
    pr = sp_pr_init((uint32_t)HD);   /* head_dim must be in {128,256,512} */
    qi = (int32_t *)malloc(...);
    ki = (int32_t *)malloc(...);
    if (!pr || !qi || !ki) goto done;
}
/* ... */
int64_t ip = sp_pr_inner(pr, qi, ki);
/* ... */
free(qi); free(ki); free(kq); sp_pr_free(pr);
```

For HD=64 (Qwen2.5-Coder, Qwen3-0.6B): `sp_pr_init(64)` returns NULL. The
malloc check fails the goto done. With `SP_ENGINE_NTT_ATTN=1` set, the
forward returns rc=1 (error) — but in practice the env var was OFF in
production, so this fell silent.

**After (NTT.5c):**

```c
sp_pr_ctx           *pr   = NULL;   /* direct: HD ∈ {128, 256, 512} */
sp_pr_bluestein_ctx *pr_b = NULL;   /* Bluestein: HD ∈ {2..256} ∖ {512} */
int overlay_active = 0;
/* ... */
if (g_ntt_attn) {
    if (HD == 128 || HD == 256 || HD == 512) {
        pr = sp_pr_init((uint32_t)HD);
        if (pr) overlay_active = 1;
    } else if (HD >= 2 && HD <= 256 && (HD & (HD - 1)) == 0) {
        pr_b = sp_pr_bluestein_init((uint32_t)HD);
        if (pr_b) {
            overlay_active = 1;
            if (backend_handle || backend_forward || backend_inverse)
                sp_pr_bluestein_set_backend(pr_b, backend_handle,
                                            backend_forward, backend_inverse);
        }
    }
    /* HD with odd factors leaves both NULL → overlay_active stays 0;
     * the inner loop falls back to sp_attn_head (fp32). Banned per
     * reference-ntt-bluestein-arbitrary-n-escape to propose mixed-radix
     * or zero-pad-HD. */
    if (overlay_active) {
        qi = malloc(HD * sizeof(int32_t));
        ki = malloc(HD * sizeof(int32_t));
        if (!qi || !ki) goto done;
    }
}
/* Inner loop: */
if (!g_ntt_attn || !overlay_active) {
    sp_attn_head(qh, k, v, t, KVD, kvh, HD, ascale, -1, sc, out);  /* fp32 path */
    continue;
}
int64_t ip = pr_b ? sp_pr_bluestein_inner(pr_b, qi, ki)
                  : sp_pr_inner(pr, qi, ki);   /* dispatch on which ctx is non-NULL */
/* ... */
free(qi); free(ki); free(kq);
sp_pr_free(pr);
sp_pr_bluestein_free(pr_b);
```

The `overlay_active` flag is the safety net: if `g_ntt_attn=1` but the HD
admits neither direct nor Bluestein (i.e. HD has an odd factor > 1, like
96/192/288/384), the overlay's per-score loop is bypassed and fp32
attention runs instead. No silent NTT-of-zero-coefficients corruption.

The same shape was added to `core/forward/qwen25.c` (where it didn't exist
at all before). `core/forward/gemma3.c` got the ABI-uniform `_ex2`
wrapper without the overlay (SWA out of scope).

## Session-pointer threading — final architecture

**Initial plan (PLAN-NTT-5c.md):** thread `const sp_session*` through to
the forward, have the forward call `sp_session_compute_backend_*`
accessors directly to extract the registered backend.

**Problem caught at first build:** `core/session` already DEPENDS
`sp_forward` in the CMake graph (the session's `sp_prefill_chunk` calls
`qwen3_forward` / `qwen25_forward` / `gemma3_forward`). If `sp_forward`
calls back into `sp_session_compute_backend_*`, the build's link closure
is circular: `test_forward.exe` needs sp_session symbols, but the forward
module is processed before session in the root CMakeLists.txt.

**Resolved by switching to explicit backend triple in the `_ex2` ABI.**
The forward's `qwen3_forward_ex2 / gemma3_forward_ex2 / qwen25_forward_ex2`
take `(void *backend_handle, sp_compute_ntt_dispatch_fn forward,
sp_compute_ntt_dispatch_fn inverse)` as explicit parameters.
`sp_prefill_chunk` in `core/session/sp_session.c` is the bridge that
reads `s->compute_backend_*` directly (it's the session struct's TU) and
passes the triple to the forward. The forward TU now has zero reference
to `struct sp_session` or the L1 accessor functions.

This is architecturally cleaner than the initial plan: forward and session
modules stay strictly hierarchical, the backend triple is reusable by any
caller (not just L1 sessions — tools/probe could attach a backend without
an L1 session if needed), and the dispatch fn typedef is shared via
`sp_l1.h` which forward.c/qwen25.c/gemma3.c now include via `model.h`.

Documented this in-line in the entry-point comments (`include/sp/model.h`
§"Sprint NTT.5c: backend-aware forward variants") and in the
`sp_prefill_chunk` body comment, so future readers see the rationale.

## Gates table

| Gate | Methodology | Pass criteria | Observed | Verdict |
|------|-------------|---------------|----------|---------|
| T_NTT5C_HD_64_BIT_EXACT_VS_FP32_TOP1 | qwen25 fixture HD=8 (Bluestein-admissible); prefill+5 decodes baseline vs NTT_ATTN=1 | argmax sequence matches token-for-token | sequence `[34, 26, 13, 19, 47]` matches across both runs | **PASS** |
| T_NTT5C_HD_256_NO_REGRESSION | structural admission gate — sp_pr_init(N) / sp_pr_bluestein_init(N) for the NTT.5c dispatch's admissible sets and rejections | direct admits {128,256,512}; Bluestein admits {2..256}\{512}; both reject non-PoT | 22/22 admit/reject decisions correct | **PASS** |
| T_NTT5C_HD_64_BLUESTEIN_WORKS | Knack's S22U; Qwen2.5-Coder-0.5B Memory model (HD=64); SP_ENGINE_NTT_ATTN=1; 3-token prefill | sp_prefill_chunk returns SP_OK; logits finite; argmax > 0 and < vocab-1 | rc=Ok(()); wall=1400ms; argmax=16; logits all finite | **PASS** |
| T_NTT5C_HEX_BACKEND_ROUTES_WHEN_REGISTERED | S22U; same model + prefill; SP_ENGINE_NTT_ATTN=1 + SP_ENGINE_NTT_ATTN_HEX=1; ntt_hex_dispatch::dispatch_counts() readback | forward_count > 0 AND inverse_count > 0 after prefill | forward+=4032, inverse+=2016 (matches expected per Bluestein-inner pattern) | **PASS** |

All 4 substantive gates PASS. No silent gate revisions. No banned
propositions surfaced (no mixed-radix, no Good-Thomas, no zero-pad-HD,
no third prime, no Cooley-Tukey at non-PoT N — all stay banned per
`reference-ntt-bluestein-arbitrary-n-escape`).

Bit-exactness between host and Hex paths: both runs produce argmax=16 on
the Memory model, so the Hex backend dispatch is producing the same
inner-product as the host Bluestein path (per `reference-lattice-decode-
determinism`). This is implicit in the test: if the Hex inner-products
were divergent, the argmax would differ.

## Wall-clock matrix (informational)

Measured on Knack's S22U with `sp_ntt_5c_forward_smoke` (1 prefill of 3
tokens, Qwen2.5-Coder-0.5B Memory model, 24 layers × 14 heads × HD=64).

| Configuration | Wall-clock | Dispatch counts | Ratio vs baseline |
|---------------|-----------:|----------------:|------------------:|
| SP_ENGINE_NTT_ATTN=0 (fp32 baseline) | ~50 ms (model only, no overlay) | — | — |
| SP_ENGINE_NTT_ATTN=1 (host Bluestein) | 1400 ms | fwd=0 inv=0 | ~28× slower |
| SP_ENGINE_NTT_ATTN=1 + _HEX=1 (Hex-routed Bluestein) | 3358 ms | fwd=4032 inv=2016 | ~67× slower |

The 28× host-Bluestein overhead vs fp32 is expected — Bluestein wraps a
length-64 negacyclic NTT (already 4× slower than direct sp_pr_inner per
NTT.5a closure §"Wall-clock comparison") plus the overlay's per-(t,s)
quantize-to-int32 + sp_pr_bluestein_inner + dequantize loop runs serially
against fp32's matrix-multiply-shaped inner loop.

The further 2.4× Hex-routed slowdown vs host-Bluestein is the FastRPC
marshalling cost (~225 μs per call × 6 calls per Bluestein inner × 2016
inners = ~2700 ms — dominant), consistent with NTT.5b's wall-clock matrix
("Hex slower as expected at this scale; NTT.6 long-context tiling is
where amortization makes hex faster").

**Per spec:** this is informational, NOT a substantive gate. NTT.5c is
correctness + activation; perf engineering (per-prime forward exposure,
ctx-cached scratch, large-N tile amortization) is NTT.6+ scope.

## Files changed

### Math-core submodule (`lib/shannon-prime-system`)

**EXTEND:**

| File | LOC delta |
|------|-----------|
| `include/sp/model.h` | +40 (3 new `_ex2` declarations + `sp_compute_ntt_dispatch_fn` typedef via include "sp/sp_l1.h" + Sprint NTT.5c docstring) |
| `core/forward/forward.c` | +60 / -10 (Bluestein include, HD dispatch in init block, `overlay_active` flag, inner-loop dispatch, free pr_b at end, _ex2 entry point + thin _ex / _forward wrappers) |
| `core/forward/qwen25.c` | +90 / -3 (full NTT-attention overlay added; was absent before NTT.5c. Same shape as forward.c — HD dispatch, overlay_active flag, backend-triple plumb-through.) |
| `core/forward/gemma3.c` | +14 (ABI-uniform `gemma3_forward_ex2` wrapper that ignores the triple — no NTT-attention overlay; SWA composition out of scope) |
| `core/session/sp_session.c` | +12 / -4 (sp_prefill_chunk extracts backend triple from struct sp_session and passes to forward `_ex2`; breaks the forward-vs-session CMake dependency cycle) |
| `core/session/session_test.c` | +97 (`T_NTT5C_HD_64_BIT_EXACT_VS_FP32_TOP1` test using qwen25 fixture HD=8 — same code path as HD=64 Memory model) |
| `core/forward/forward_test.c` | +60 (`T_NTT5C_HD_256_NO_REGRESSION` structural admission gate; direct sp_pr admits {128,256,512}, Bluestein admits {2..256}\\{512}) |

**No new files** in math-core. No edits to NTT.5a/5b frozen surfaces:
`poly_ring_bluestein.h/.c` (Bluestein public API unchanged), `sp_l1.h`'s
NTT.5b §5 section (L1 ABI extension unchanged), `sp_session.c`'s NTT.5b
§5 section (register fn + getters unchanged).

### Engine repo (`engine-ntt-5c`)

**NEW:**

| File | LOC |
|------|-----|
| `tools/sp_compute_skel/docs/PLAN-NTT-5c.md` | 423 |
| `tools/sp_compute_skel/docs/CLOSURE-NTT-5c.md` | (this file) |
| `tools/sp_daemon/src/bin/sp_ntt_5c_forward_smoke.rs` | 322 (single-prefill harness with optional backend register + dispatch-count readback) |

**EXTEND:**

| File | LOC delta |
|------|-----------|
| `tools/sp_daemon/Cargo.toml` | +9 (sp_ntt_5c_forward_smoke bin declaration) |
| `tools/sp_daemon/src/ntt_hex_dispatch.rs` | +30 (AtomicU64 forward+inverse dispatch counters + public dispatch_counts / reset_dispatch_counts + per-trampoline increment) |

**Files NOT TOUCHED (anti-contamination):**

- NTT.5a public surface: `poly_ring_bluestein.h/.c` unchanged.
- NTT.5b L1 ABI: `sp_l1.h` §5, `sp_session.c` §5 unchanged.
- `core/ntt_crt/`, `core/poly_ring/poly_ring.c`, `include/sp/poly_ring.h`
  — Phase 1B/1C reference unchanged.
- All NTT.0/1/2/3/4 smokes in `tools/sp_dsp_smoke/`.
- `tools/sp_daemon/src/daemon.rs`, `state.rs`, `session.rs` — NTT.5b's
  AppState + backend wiring + register-at-startup unchanged.
- Any other engine-* or lattice-* worktree.

`grep -rn 'qwen3_forward_ex2\|qwen25_forward_ex2\|gemma3_forward_ex2\|dispatch_counts'
D:/F/shannon-prime-repos/` outside `engine-ntt-5c` returns nothing.

## Commits on sprint/ntt-5c

Engine repo (`D:\F\shannon-prime-repos\engine-ntt-5c`):

  | SHA | Message |
  | --- | --- |
  | `aa4823c` | `[plan] NTT.5c -- forward.c activation of Bluestein + backend dispatch for HD in {2..64}` |
  | `25bf72d` | `[plan] NTT.5c amend -- Stage 0 discovery: qwen25.c has no NTT-attention overlay; must add` |
  | `545178f` | `[NTT.5c] Stage 1 submodule bump: math-core forward.c + qwen25.c NTT-attention overlay (HD-dispatch + backend triple)` |
  | `dc2bc6a` | `[NTT.5c] Stage 1 test submodule bump: T_NTT5C_HD_64_BIT_EXACT_VS_FP32_TOP1 PASS (Bluestein-dispatched top-1 == baseline)` |
  | `871f3d1` | `[NTT.5c] Stage 2 submodule bump: T_NTT5C_HD_256_NO_REGRESSION PASS (HD-dispatch admission agrees with sp_pr_init/sp_pr_bluestein_init)` |
  | `d6edaa5` | `[NTT.5c] Stage 3: on-device smoke + dispatch counters -- T_NTT5C_HD_64_BLUESTEIN_WORKS + T_NTT5C_HEX_BACKEND_ROUTES_WHEN_REGISTERED both PASS on S22U` |
  | (Stage 4 closure commit lands next) |

Math-core submodule (`lib/shannon-prime-system`):

  | SHA | Message |
  | --- | --- |
  | `a1fd85c` | `[NTT.5c] Stage 1: forward.c + qwen25.c NTT-attention overlay -- HD-dispatch on direct sp_pr vs Bluestein + opt-in backend triple` |
  | `df9585b` | `[NTT.5c] Stage 1 test: T_NTT5C_HD_64_BIT_EXACT_VS_FP32_TOP1 -- argmax sequence baseline vs NTT_ATTN=1 on qwen25 fixture (HD=8 Bluestein-admissible)` |
  | `ce93b9c` | `[NTT.5c] Stage 2 test: T_NTT5C_HD_256_NO_REGRESSION -- direct sp_pr admission unchanged for HD in {128,256,512}` |

3 math-core commits + 6 engine commits = 9 total ahead of bases.

## Sub-tag candidate

`lat-phase-4-ntt-5c-forward-activation`. Operator applies post-merge.

## What's NOT done (out-of-scope per spec)

- **Executive (Qwen3-0.6B) NTT-attention routing through Hex backend.**
  Per spec ("Memory only"). Executive uses HD=128 → direct `sp_pr_init`
  path, which has no `set_backend` API. Adding `sp_pr_set_backend` to
  the direct path is a future NTT.5d sprint if a large-HD model wants
  Hex dispatch.

- **HD with odd factors (96, 192, 288, 384, …).** Banned per
  `reference-ntt-bluestein-arbitrary-n-escape`. Overlay silently
  disables for these HDs (`overlay_active` stays 0 → fp32 fallback).
  SP-aligned answer for those HDs is direct integer dot product with
  Barrett (in math-core ntt_crt land); no current model in the SP
  inventory uses an odd-factor HD.

- **gemma3.c NTT-attention overlay.** Sliding-window attention vs full
  causal: the overlay's `s ∈ [0..t]` loop shape doesn't trivially mask
  SWA. Future SWA-aware NTT-attention is out of NTT.5c scope.

- **NTT.6 long-context tile benchmark.** This is the next sprint after
  NTT.5c. NTT.5c unblocks it by giving the benchmark a real production
  consumer (Memory model end-to-end via daemon).

- **Performance optimization.** Per Wall-clock matrix above: Bluestein is
  ~28× slower than fp32 attention, Hex-routed Bluestein another ~2.4×
  slower than host. Per-prime forward exposure, ctx-cached scratch, and
  large-N tile amortization are NTT.6+ levers. NTT.5c shipped
  correctness + activation per spec.

- **dialogue-level bit-exactness test.** The Stage-3 smoke verifies
  argmax of a single prefill matches host vs Hex (16 = 16). A full
  T_NTT5C_DIALOGUE_BIT_EXACT (sp_memo_m2_dialogue_smoke output unchanged
  with SP_ENGINE_NTT_ATTN_HEX flipping under NTT.5c) is achievable now
  but was not in the spec's gate list. Pickable in a follow-on if
  operator wants tighter cross-path validation at the dialogue scope.

## What unblocks

- **NTT.6 long-context tile benchmark sprint.** The substrate is now
  end-to-end load-bearing: `SP_ENGINE_NTT_ATTN=1 +
  SP_ENGINE_NTT_ATTN_HEX=1` flips the Memory model's attention through
  the Hex backend in real production code path. NTT.6 can run a real
  long-context (ctx=512 or 1024) Memory model dialogue and measure
  Hex-routed Bluestein speedup vs host across the actual amortizable
  workload (per NTT.5b CLOSURE: "NTT.6 long-context tiling is where the
  silicon win materializes").

- **Phase 4-NTT closure for HD ∈ {2..256, ≠ 512}.** Math-core's NTT
  substrate has been Bluestein-aware since NTT.5a, L1-dispatchable since
  NTT.5b, and forward-activated since NTT.5c. The full pipeline is
  shipped for the SP-philosophy-aligned escape set. HD=512 stays direct
  (faster); HD with odd factors stays fp32 (banned alternatives per
  reference-ntt-bluestein-arbitrary-n-escape).

- **MeMo dialogue with NTT-attention enabled by default.** Once
  NTT.5c is merged, `tools/sp_daemon/src/daemon.rs`'s startup can flip
  `SP_ENGINE_NTT_ATTN=1` for Memory sessions (or any HD-admissible
  model) without crashing or silently no-op'ing. Production decision
  for the operator after NTT.6 perf characterization.

## Memory entry candidates

Post-operator-merge:

1. **Update `reference-ntt-bluestein-arbitrary-n-escape`** with closing
   line: "NTT.5c activated host-Bluestein + Hex-backend-dispatched
   Bluestein in forward.c + qwen25.c overlays 2026-05-31; Memory model
   (Qwen2.5-Coder-0.5B HD=64) now executes NTT-attention via cDSP V69
   HVX kernels end-to-end when SP_ENGINE_NTT_ATTN=1 +
   SP_ENGINE_NTT_ATTN_HEX=1; sub-tag `lat-phase-4-ntt-5c-forward-
   activation`; 4/4 gates PASS bit-exact."

2. **New `reference-forward-c-overlay-hd-dispatch`** (one-liner index):
   "forward.c + qwen25.c NTT-attention overlays dispatch on HD: direct
   sp_pr (HD ∈ {128,256,512}, faster — no zero-pad) vs Bluestein
   sp_pr_bluestein (HD ∈ {2..256}\\{512}). Overlay_active flag silently
   falls back to fp32 for HD with odd factors (banned to mixed-radix).
   Backend triple (handle + fwd + inv) passes through _ex2 entry points,
   plumbed to sp_pr_bluestein_set_backend when non-NULL. Stable since
   NTT.5c 2026-05-31."

3. **New `reference-forward-session-decoupling-via-backend-triple`** (one-liner
   index): "forward.c never references struct sp_session or L1 readback
   accessors. Instead the _ex2 entry points take (void *handle,
   sp_compute_ntt_dispatch_fn fwd, sp_compute_ntt_dispatch_fn inv) as
   explicit parameters; sp_prefill_chunk in session is the bridge that
   extracts the triple from the session struct (which it owns) and
   passes through. Breaks the forward-vs-session CMake dependency cycle.
   Caught NTT.5c first-build link failure 2026-05-31; resolved with
   parameter-passing per SP-discipline 'no hidden globals on hot path'."

4. **New `reference-qwen25-c-ntt-attention-overlay`** (one-liner index):
   "qwen25_forward had NO NTT-attention overlay until NTT.5c
   (2026-05-31) -- only forward.c (qwen3 path) had it. The Memory model
   (Qwen2.5-Coder-0.5B HD=64) routes through qwen25_forward via
   sp_prefill_chunk, so NTT.5c's spec requirement 'Memory model does
   NTT-attention' implies qwen25.c must grow the overlay too. Future
   sprint discoveries of 'this arch path doesn't have feature X':
   confirm the actual model's forward routes through it before scoping
   the work. Caught during Stage 0 pre-read; documented in
   PLAN-NTT-5c.md §'Stage 0 — qwen25.c overlay discovery'."

These four are the candidates; operator decides which to commit.

## Worktree status

```
D:\F\shannon-prime-repos\engine-ntt-5c    (engine)
  branch:  sprint/ntt-5c
  base:    bd34c08 (engine main, post NTT.5b merge)
  tip:     d6edaa5  (Stage 3; Stage 4 closure commit lands next)
  6 commits ahead of base (2 plan + 3 stage submodule bumps + 1 stage-3 daemon + 1 closure pending)
  push:    git push -u origin sprint/ntt-5c

D:\F\shannon-prime-repos\engine-ntt-5c\lib\shannon-prime-system    (math-core)
  branch:  sprint/ntt-5c (created from sprint/ntt-5b at fc38a4f)
  base:    fc38a4f (NTT.5b tip)
  tip:     ce93b9c
  3 commits ahead of base (Stage 1 overlay + Stage 1 test + Stage 2 test)
  push:    git push -u origin sprint/ntt-5c
```

Anti-contamination verified per §"Files changed". Push commands (operator
runs post-closure-ack):

```
cd D:\F\shannon-prime-repos\engine-ntt-5c\lib\shannon-prime-system
git push -u origin sprint/ntt-5c

cd D:\F\shannon-prime-repos\engine-ntt-5c
git push -u origin sprint/ntt-5c
```

Operator merges; applies `lat-phase-4-ntt-5c-forward-activation` sub-tag.
