# PLAN-NTT-5c.md — Activate Bluestein in forward.c (math-core consumer side)

## Headline

NTT.5c flips the switch on Phase 4-NTT's actual production reach. NTT.5a
shipped host Bluestein for HD ∈ {2..256, ≠ 512}; NTT.5b shipped the
Hexagon-backend L1 dispatch substrate + Rust trampoline + Memory-session
registration on `SP_ENGINE_NTT_ATTN_HEX=1`. But math-core's
`core/forward/forward.c:115-211` NTT-attention overlay still calls
`sp_pr_init((uint32_t)HD)` which returns NULL for HD ∉ {128, 256, 512},
and still calls the host `sp_pr_inner(pr, qi, ki)` without any awareness
of a registered backend.

NTT.5c teaches the overlay two things:

  1. **Algorithm dispatch on HD.** `HD ∈ {128, 256, 512}` → direct
     `sp_pr_init` (NTT.4-era, faster — no zero-pad). `HD ∈ {2..64}` →
     `sp_pr_bluestein_init` (NTT.5a, the actual escape for Qwen2.5-Coder).
     HD with odd factors (96, 192, 384, …) → leave overlay disabled
     (silent fall-through to fp32 attention; out-of-NTT-5c-scope, banned
     to mixed-radix per `reference-ntt-bluestein-arbitrary-n-escape`).

  2. **Backend dispatch on session registration.** When the session has a
     compute backend registered (NTT.5b path: daemon set
     `SP_ENGINE_NTT_ATTN_HEX=1`), thread the
     `sp_session_compute_backend_{handle,forward,inverse}` triple through
     to the Bluestein context via `sp_pr_bluestein_set_backend`. The
     direct `sp_pr_init` path (HD ∈ {128,256,512}) has no backend-aware
     API yet — out of NTT.5c scope (would be NTT.5d if ever needed; the
     Memory model (HD=64) is the unblocking target).

After NTT.5c lands, **`SP_ENGINE_NTT_ATTN=1` + Qwen2.5-Coder-0.5B Memory
model produces working NTT-attention overlay output** (previously: silent
no-op fall-through). Adding `SP_ENGINE_NTT_ATTN_HEX=1` routes the inner
length-128 NTT calls through the cDSP V69 HVX backend via NTT.5b's
FastRPC trampoline.

This is the moment Phase 4-NTT becomes load-bearing for the PPT ARM
mission: the Memory model is the chat-integration target, HD=64 has been
the silent-fail blocker since NTT.4, and the Hex-dispatch substrate now
has a consumer end-to-end.

## Stage 0 — Mandatory pre-read citations

1. **`reference-ntt-bluestein-arbitrary-n-escape`** memory entry —
   admissible-N set is {2,4,8,16,32,64,128,256}; banned alternatives
   (mixed-radix, Good-Thomas, zero-pad-HD, third-prime). Bluestein is
   the SP-aligned escape for power-of-2 HD ≤ 256 only.

2. **`reference-sp-uses-phi-extensively`** memory entry — φ is already
   load-bearing in Phase 8 Fibonacci-Prime DHT + KV sub-sampling +
   SP_ROPE_PHI + Halton/Sobol QMC. NOT proposing any new φ framework.

3. **`tools/sp_compute_skel/docs/CLOSURE-NTT-5a.md:120-141`** — Bluestein
   admissible-N table; lines 322-327 explicitly defer forward.c wire-up
   to a future sprint (this one).

4. **`tools/sp_compute_skel/docs/CLOSURE-NTT-5b.md:36-72`** — L1 ABI
   extension signatures: `sp_session_register_compute_backend` +
   `sp_session_compute_backend_{handle,forward,inverse}` readback;
   `sp_pr_bluestein_set_backend` consumer-side setter. Lines 234-245
   explicitly carve out forward.c wire-up as "What's NOT done" → THIS
   sprint's scope.

5. **`lib/shannon-prime-system/core/forward/forward.c`** — the NTT
   overlay block to extend:
   - line 26: `#include "sp/poly_ring.h"` (need to add `poly_ring_bluestein.h`)
   - line 49: `static int g_ntt_attn = 0;` runtime gate
   - line 79: env-knob read for `SP_ENGINE_NTT_ATTN`
   - line 99: `sp_pr_ctx *pr = NULL;` (need to add `sp_pr_bluestein_ctx *pr_b = NULL;`)
   - line 115-120: init block — currently `sp_pr_init((uint32_t)HD)`
     with comment "head_dim must be in {128,256,512}"
   - line 184-211: inner-loop dispatch — currently `sp_pr_inner(pr, qi, ki)`
     at line 197 unconditionally
   - line 240: `sp_pr_free(pr);` (need to add `sp_pr_bluestein_free(pr_b);`)

6. **`lib/shannon-prime-system/include/sp/poly_ring_bluestein.h:47-104`** —
   full public API:
   - `sp_pr_bluestein_init(uint32_t N) → ctx*` (NULL for inadmissible N)
   - `sp_pr_bluestein_free(ctx)`
   - `sp_pr_bluestein_inner(ctx, q, k) → int64_t`
   - `sp_pr_bluestein_mul(ctx, a, b, out)`
   - `sp_pr_bluestein_set_backend(ctx, handle, fwd, inv)` (NTT.5b setter)

7. **`lib/shannon-prime-system/include/sp/sp_l1.h:158-222`** — L1 ABI
   extension:
   - `typedef int (*sp_compute_ntt_dispatch_fn)(void*, int, int, const uint32_t*, uint32_t*);`
   - `sp_session_register_compute_backend(s, handle, fwd, inv) → sp_status`
   - `void *sp_session_compute_backend_handle(const sp_session*)`
   - `sp_compute_ntt_dispatch_fn sp_session_compute_backend_forward(const sp_session*)`
   - `sp_compute_ntt_dispatch_fn sp_session_compute_backend_inverse(const sp_session*)`
   - L189-194 docstring: handle is opaque; per-direction NULL fn
     fallback is the API contract.

## Session-pointer threading analysis

**The problem.** `forward.c::qwen3_forward_ex(m, tokens, n_tok, logits,
kv_trees)` takes a `qwen3_model*` not an `sp_session*`. Same shape for
`gemma3_forward`, `qwen25_forward`. The chain from L1 ABI is:

```
sp_prefill_chunk(s, tokens, ...)        [core/session/sp_session.c:141]
  → qwen3_forward(s->qm, hist, ...)     [core/session/sp_session.c:157]
    → qwen3_forward_ex(m, ..., NULL)    [core/forward/forward.c:244]
```

`s` (the session pointer) is dropped before reaching `qwen3_forward_ex`.

**The decision: add `_ex2`-style session-aware entry points.** Cleanest
backwards-compatible threading without touching existing public API:

```c
/* New backend-aware entry points (additive in include/sp/model.h). */
int qwen3_forward_ex2(const qwen3_model *m, const int32_t *tokens, int n_tok,
                      float *logits, sp_kste_tree_t *kv_trees,
                      const sp_session *session);   /* NULL = host path */
int gemma3_forward_ex2(...same..., const sp_session *session);
int qwen25_forward_ex2(...same..., const sp_session *session);

/* Existing entry points become thin wrappers passing session=NULL. */
int qwen3_forward_ex(...) { return qwen3_forward_ex2(..., NULL); }
int qwen3_forward(...)    { return qwen3_forward_ex2(..., NULL, NULL); }
/* same shape for gemma3_forward / qwen25_forward. */
```

`sp_prefill_chunk` switches to call `qwen3_forward_ex2(s->qm, ..., s)`,
etc. Backend reads happen inside `qwen3_forward_ex2` via
`sp_session_compute_backend_handle(session)` / `_forward()` / `_inverse()`.

**Why this and not a thread-local?** Thread-local would work, but
explicit parameter-passing is the SP discipline (no hidden globals on
the hot path; per `reference-zero-copy-invariant` and similar). One
extra const-pointer parameter through a single call chain is cheap.

**Why not session in `sp_session.c::kv_step`?** Per `forward.c:251` doc:
"NTT-attention is prefill-only". `kv_step` (sp_decode_step backend)
intentionally does not run NTT-attention; it's the f32 reference
decode. NOT in NTT.5c scope.

**Wire-up matrix:**

| Path | Today | After NTT.5c |
|------|-------|--------------|
| `sp_prefill_chunk` → forward | host f32 OR fp32-NTT (sp_pr_inner if HD ∈ {128,256,512}) | host fp32 OR Bluestein-NTT (host or backend) for HD ∈ {2..256} |
| `sp_decode_step` (kv_step) | f32 reference | f32 reference (unchanged; NTT-attention prefill-only) |
| `qwen3_generate*` standalone | host fp32 OR fp32-NTT | host fp32 OR Bluestein-NTT (host only; standalone has no session) |

The third row covers `tools/probe`-style callers that go through
`qwen3_generate` / `qwen3_forward` directly (no L1 session). For those,
`session=NULL` → backend NULL → host Bluestein path. The new capability
(HD=64 working at all) reaches them; backend-dispatch does not.

## Stage 0 — qwen25.c overlay discovery

**Discovery during pre-read** (`core/forward/qwen25.c`): qwen25_forward
has NO NTT-attention overlay. It calls `sp_attn_head` directly at line
79 (fp32 attention). gemma3.c likewise has no NTT-attention overlay.
Only `forward.c::qwen3_forward_ex` has the `g_ntt_attn` branch.

**Implication for NTT.5c's stated mission.** The Memory model
(Qwen2.5-Coder-0.5B) goes through `qwen25_forward` via
`sp_prefill_chunk → qwen25_forward(s->qm, ...)` (sp_session.c:156).
Even if NTT.5c only patches forward.c's qwen3 overlay, the Memory
model would NEVER hit the new Bluestein code path because it routes
through qwen25_forward not qwen3_forward.

**Therefore NTT.5c MUST add the NTT-attention overlay to qwen25.c**
in addition to extending forward.c's overlay. The overlay structure is
mechanical: the Qwen2.5 attention loop at qwen25.c:77-82 becomes a
NTT-attention conditional dispatch identical in shape to forward.c:188-211.

**Out of scope:** Adding the overlay to gemma3.c. Gemma3 has sliding-
window attention via `sp_attn_head` with `win=` parameter — the NTT-
attention overlay shape doesn't trivially compose with SWA, and Gemma3
isn't the NTT.5c target model. Documented as deferred.

## Forward.c overlay logic (before vs after)

**Before (NTT.4-era; HD silently fails for {2..64}):**

```c
sp_pr_ctx *pr = NULL;
int32_t *qi = NULL, *ki = NULL;
/* ... */
if (g_ntt_attn) {
    pr = sp_pr_init((uint32_t)HD);   /* head_dim must be in {128,256,512} */
    qi = (int32_t *)malloc((size_t)HD * sizeof(int32_t));
    ki = (int32_t *)malloc((size_t)HD * sizeof(int32_t));
    if (!pr || !qi || !ki) goto done;
}
/* ... */
int64_t ip = sp_pr_inner(pr, qi, ki);
/* ... */
free(qi); free(ki); free(kq); sp_pr_free(pr);
```

**After (NTT.5c):**

```c
sp_pr_ctx           *pr   = NULL;     /* direct path: HD ∈ {128, 256, 512} */
sp_pr_bluestein_ctx *pr_b = NULL;     /* Bluestein path: HD ∈ {2..256} ∖ {512} */
int32_t *qi = NULL, *ki = NULL;
/* ... */
if (g_ntt_attn) {
    /* Algorithm dispatch on HD. Direct sp_pr_init is faster (no zero-pad)
     * for HD ∈ {128, 256, 512}; Bluestein covers all other admissible
     * power-of-2 HD ∈ {2, 4, 8, 16, 32, 64}. HD with odd factors stays
     * NULL → silent fall-through to fp32 attention (NTT.5c does NOT
     * propose mixed-radix per banned alternatives in
     * reference-ntt-bluestein-arbitrary-n-escape). */
    if (HD == 128 || HD == 256 || HD == 512) {
        pr = sp_pr_init((uint32_t)HD);
    } else if (HD >= 2 && HD <= 256 && (HD & (HD - 1)) == 0) {
        pr_b = sp_pr_bluestein_init((uint32_t)HD);
        /* If a compute backend is registered on the session, plumb it
         * through to the Bluestein context. NULL session OR no backend
         * → Bluestein keeps the host-side inner ntt_crt path. */
        if (pr_b && session) {
            void *bh = sp_session_compute_backend_handle(session);
            sp_compute_ntt_dispatch_fn bfwd = sp_session_compute_backend_forward(session);
            sp_compute_ntt_dispatch_fn binv = sp_session_compute_backend_inverse(session);
            if (bh || bfwd || binv) {   /* register iff any field non-NULL */
                sp_pr_bluestein_set_backend(pr_b, bh, bfwd, binv);
            }
        }
    }
    /* Both NULL = HD has odd factors > 1 OR allocation failed. We still
     * allocate qi/ki to keep the dispatch loop simple; the inner-loop
     * branch below picks the right path or falls back to fp32. */
    qi = (int32_t *)malloc((size_t)HD * sizeof(int32_t));
    ki = (int32_t *)malloc((size_t)HD * sizeof(int32_t));
    if (!qi || !ki) goto done;
}
/* ... */
int64_t ip;
if      (pr_b) ip = sp_pr_bluestein_inner(pr_b, qi, ki);
else if (pr)   ip = sp_pr_inner(pr, qi, ki);
else           continue;   /* HD with odd factors: fall through to next s */
/* (note: when both ctx pointers NULL the overlay is effectively a no-op;
 *  fp32 path stays at sc[s]=0.0f for these heads. We could fall back to
 *  fp32 dot here, but cleaner is to disable the overlay's per-score loop
 *  for these unsupported HD values and let the standard fp32 attention
 *  branch (g_ntt_attn=0) be the user's recourse. Document as such.) */
/* ... */
free(qi); free(ki); free(kq);
sp_pr_free(pr);
sp_pr_bluestein_free(pr_b);
```

**Refinement on the unsupported-HD case.** If both `pr` and `pr_b` are
NULL after init (HD with odd factor), we should NOT silently produce
zero attention scores — that would corrupt output. Two options:

  A. **Disable the overlay entirely for this forward call.** Set a
     local `int overlay_active = (pr != NULL || pr_b != NULL);` and
     gate the NTT-attention branch on it. When inactive, fall through
     to the standard fp32 `sp_attn_head` path for ALL heads. SAFE.

  B. **Fall back per-head.** Use fp32 dot inside the loop when both ctx
     are NULL. Marginally more efficient (only the unsupported HD heads
     pay fp32 cost), but mixes the two code paths.

Option A is simpler and matches existing pattern: `g_ntt_attn` is a
per-forward gate, so refining it to "per-forward AND HD-admissible" is
a one-line edit. Going with A.

## Files expected to change

### Math-core submodule (`lib/shannon-prime-system`)

**EXTEND:**

| File | Change | Est LOC |
|------|--------|---------|
| `include/sp/model.h` | Add `qwen3_forward_ex2` / `gemma3_forward_ex2` / `qwen25_forward_ex2` declarations (session-aware) | +30 |
| `core/forward/forward.c` | Bluestein dispatch in NTT overlay; new `qwen3_forward_ex2` impl; `qwen3_forward_ex` becomes wrapper | +60 / -8 |
| `core/forward/gemma3.c` | `gemma3_forward_ex2` impl (likely no NTT-attention in gemma3 path — confirm; just thread param through) | +15 / -2 |
| `core/forward/qwen25.c` | **CRITICAL:** qwen25_forward had NO NTT-attention overlay at all today (Stage 0 discovery — see §"Stage 0 — qwen25.c overlay discovery"). NTT.5c adds the overlay AND wires Bluestein dispatch — this is the actual unblocker for the Memory model. | +90 / -3 |
| `core/session/sp_session.c` | `sp_prefill_chunk` switches to `qwen{3,25}_forward_ex2(..., s)` per arch | +5 / -3 |
| `core/forward/CMakeLists.txt` | Already lists forward.c; no new file needed | 0 |

**No new files** in math-core.

**No edits** to NTT.5a/5b frozen surfaces (`poly_ring_bluestein.h/.c`,
`sp_l1.h`'s NTT.5b §5 section, `sp_session.c`'s NTT.5b §5 section).

### Engine repo (`engine-ntt-5c`)

**NEW:**

| File | Description | Est LOC |
|------|-------------|---------|
| `tools/sp_compute_skel/docs/PLAN-NTT-5c.md` | This file | 250 |
| `tools/sp_compute_skel/docs/CLOSURE-NTT-5c.md` | Closure | 350 |
| `tools/sp_daemon/src/bin/sp_ntt_5c_forward_smoke.rs` | Stage 3 dispatch-counter smoke (cfg(android)) | 200 |

**EXTEND:**

| File | Change | Est LOC |
|------|--------|---------|
| `tools/sp_daemon/src/ntt_hex_dispatch.rs` | Add atomic dispatch counters + readback fn for T_NTT5C_HEX_BACKEND_ROUTES_WHEN_REGISTERED | +30 |
| `tools/sp_dsp_smoke/Cargo.toml` OR `tools/sp_daemon/Cargo.toml` | Declare new smoke binary if needed | +6 |

## Substantive gates

### T_NTT5C_HD_64_BIT_EXACT_VS_FP32_TOP1 (Stage 1, host-side)

**Methodology.** Build a small standalone host harness:
- Load Qwen2.5-Coder-0.5B `.sp-model` (HD=64)
- Run `qwen25_forward_ex2(m, tokens, n_tok, logits, NULL, NULL)` with
  `SP_ENGINE_NTT_ATTN` unset → baseline tokens
- Run again with `SP_ENGINE_NTT_ATTN=1` set → NTT-attention tokens
- Compare argmax(logits[last]) at each position

Pass: argmax token IDs match across all decode positions for a fixed
prompt (`[1, 2, 3]`) and ctx ≤ 128.

If exact-argmax matches across multiple positions, NTT-attention is
behaviorally equivalent to fp32-attention at the decode level (per
`reference-lattice-decode-determinism`). Exact-logits will differ
(integer dot vs fp32 dot), but argmax invariance is what decode reads.

Harness location: `tools/probe/` already has a probe; extending it or
creating a sibling. **Decision: write a minimal host-only C test
`core/forward/forward_test_ntt_attn.c` that does the above using just
math-core APIs (no engine deps).**

### T_NTT5C_HD_256_NO_REGRESSION (Stage 2, host-side)

**Methodology.** With Gemma3-1B (HD=256 in many configs) — OR if not
available locally, with the existing T_FORWARD smoke at HD=256 path —
run the same baseline-vs-NTT-on comparison. The direct `sp_pr_init`
path (HD ∈ {128,256,512}) is unchanged by NTT.5c, so this gate verifies
"no regression at unchanged path".

Pass: argmax token sequence with `SP_ENGINE_NTT_ATTN=1` matches the
sequence from engine main @ bd34c08 with `SP_ENGINE_NTT_ATTN=1`.

If no HD=256 model is locally available, the gate degrades to a
synthetic ctx exercising the `pr != NULL && pr_b == NULL` branch via
unit test — same code path, smaller model. Documenting fallback in
closure if needed.

### T_NTT5C_HD_64_BLUESTEIN_WORKS (Stage 3, on-device)

**Methodology.** With `SP_ENGINE_NTT_ATTN=1` (no _HEX) on Knack's S22U
running the Memory-model daemon path, drive prefill_chunk on the
Qwen2.5-Coder Memory model. Verify SP_OK return + non-NULL logits.

Previously this would have silently fallen back (HD=64 → sp_pr_init
NULL → overlay no-op). Now it should produce correct logits via
Bluestein.

Pass: `sp_prefill_chunk` returns SP_OK; first logits element is finite;
argmax produces a plausible token (not 0, not vocab_size-1 stuck).

### T_NTT5C_HEX_BACKEND_ROUTES_WHEN_REGISTERED (Stage 3, on-device)

**Methodology.** With BOTH `SP_ENGINE_NTT_ATTN=1` AND
`SP_ENGINE_NTT_ATTN_HEX=1` on S22U with Memory model loaded:
- Instrument `ntt_hex_dispatch.rs` with two `AtomicU64`
  counters (one per trampoline) bumped on each invocation
- Add an exported readback fn `ComputeBackend::dispatch_counts() ->
  (u64, u64)`
- Drive the Memory model prefill once via `sp_memo_m2_dialogue_smoke`
  (or a focused harness)
- Assert both counts > 0

Pass: `forward_count > 0 AND inverse_count > 0` after run_dialogue
completes one turn.

If counts stay zero: the backend isn't reaching the Bluestein context —
indicates a wire-up failure (session pointer threading bug or
math-core's `sp_pr_bluestein_set_backend` not consulted by
`pr_blue_convolve_M`). Surface UPSTREAM, do NOT silently revise.

## Workflow discipline (per spec §workflow)

- **Stages:** plan-commit (this doc) → Stage 1 (forward.c HD-dispatch +
  T_NTT5C_HD_64_BIT_EXACT_VS_FP32_TOP1) → Stage 2 (session threading
  + T_NTT5C_HD_256_NO_REGRESSION) → Stage 3 (on-device dispatch
  verification, T_NTT5C_HD_64_BLUESTEIN_WORKS +
  T_NTT5C_HEX_BACKEND_ROUTES_WHEN_REGISTERED) → Stage 4 (closure).
- **One variable per commit** per `feedback-bundled-changeset-root-cause-ambiguity`.
- **No silent gate revisions** per `feedback-no-silent-gate-revisions`.
- **Anti-contamination strict:** edits only in `engine-ntt-5c` worktree;
  no touches to `engine-ntt-5a`/`5b` or other lattice-* worktrees.
- **Banned propositions** (per `reference-ntt-bluestein-arbitrary-n-escape`):
  no mixed-radix, no Good-Thomas, no zero-pad-HD-to-next-admissible,
  no third prime. These remain banned for NTT.5c.

## Sub-tag candidate

`lat-phase-4-ntt-5c-forward-activation`. Operator applies post-merge.

## Out-of-scope (NTT.5d or beyond)

- **Direct `sp_pr_init`/`sp_pr_inner` backend awareness.** The HD ∈
  {128, 256, 512} path stays host-only — no API exists for backend
  dispatch through the direct (non-Bluestein) inner. For the Memory
  model (HD=64) we route everything through Bluestein, which has the
  setter. For Executive (Qwen3-0.6B HD=128) we keep direct/host. If a
  large-HD model wants backend dispatch, NTT.5d adds the
  equivalent `sp_pr_set_backend` to direct `sp_pr_ctx`.

- **Executive model NTT-attention routing.** Spec carves Executive out
  ("Memory only"). Executive is HD=128; direct `sp_pr_init` works
  host-side, and there's no Bluestein call to attach backend through
  (per above).

- **HD with odd factors (96, 192, 288, 384, …).** Banned per
  `reference-ntt-bluestein-arbitrary-n-escape`. Overlay disabled
  silently for these (which means: HD with odd factors uses the
  standard fp32 attention path even when `SP_ENGINE_NTT_ATTN=1`).

- **NTT.6 long-context tile benchmark.** This is the next sprint after
  NTT.5c. NTT.5c unblocks it by giving the benchmark a real production
  consumer.

- **Performance optimization.** Per NTT.5a CLOSURE wall-clock matrix:
  Bluestein is ~3.5–4× slower than direct sp_pr_inner at N=128/256;
  Hex-backend is ~2× slower than host at small N. NTT.5c is correctness
  + activation, not perf engineering.
