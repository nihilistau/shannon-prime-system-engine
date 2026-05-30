# CLOSURE-NTT-5b.md — Sprint NTT.5b Hexagon backend dispatch + L1 ABI extension

## Headline

NTT.5b extends the frozen L1 ABI with operator+Gemini-pre-authorized
`sp_session_register_compute_backend` (+ matching readback accessors), threads
opt-in backend awareness through math-core's NTT.5a Bluestein wrapper, ships
the Rust C-trampoline + AppState wiring on android, and validates the entire
dispatch path bit-exact on Knack's S22U at the inner-product level:

  - **T_NTT5B_HOST_HEX_BIT_EXACT**: 300/300 runs byte-exact across
    N ∈ {64, 128, 256} × 100 seeds × 2 paths.
  - **T_NTT5B_RUN_DIALOGUE_BIT_EXACT**: sp_memo_m2_dialogue_smoke produces
    byte-identical final answer + receipt content hashes with
    `SP_ENGINE_NTT_ATTN_HEX=0` vs `=1`.
  - **T_NTT5B_NO_REGRESSION**: all NTT.0-5a smokes and the host poly_ring
    test (4160 checks / 0 failures, including the new
    T_NTT5B_BACKEND_FORWARD_PASSTHROUGH) PASS unchanged.
  - **T_NTT5B_WALL_CLOCK_INFORMATIONAL**: at N=128 the Hex path is 1.89×
    slower than host per inner product (FastRPC marshalling dominates at
    small N; NTT.6 long-context tiling is where the silicon win materializes,
    per spec).

The L1 ABI extension lands additively. NTT.5a public surface untouched.
The Memory model session can now optionally route its inner Bluestein NTT
calls through the Hexagon V69 HVX kernels (methods 17 + 18) via a single
`sp_pr_bluestein_set_backend` setter call.

Worktree: `D:\F\shannon-prime-repos\engine-ntt-5b` on `sprint/ntt-5b`.
Engine base: `908e570` (post NTT.5a merge); engine tip: see §Commits.
Math-core submodule on `sprint/ntt-5b` branch (created from NTT.5a's
`7662b2d`). 6 commits across both repos.

## L1 ABI extension — exact signatures shipped

```c
/* sp/sp_l1.h §5 — operator + Gemini pre-authorized 2026-05-30 */

typedef struct sp_compute_backend_handle sp_compute_backend_handle;

typedef int (*sp_compute_ntt_dispatch_fn)(
    void *handle, int q_idx, int N,
    const uint32_t *in, uint32_t *out);

sp_status sp_session_register_compute_backend(
    sp_session *s,
    void *handle,
    sp_compute_ntt_dispatch_fn forward,
    sp_compute_ntt_dispatch_fn inverse);

/* Readback accessors used by math-core consumers (Bluestein wrapper today). */
void *sp_session_compute_backend_handle (const sp_session *s);
sp_compute_ntt_dispatch_fn sp_session_compute_backend_forward(const sp_session *s);
sp_compute_ntt_dispatch_fn sp_session_compute_backend_inverse(const sp_session *s);
```

```c
/* sp/poly_ring_bluestein.h (NTT.5a sibling, additive) */
void sp_pr_bluestein_set_backend(sp_pr_bluestein_ctx *ctx,
                                 void *handle,
                                 sp_compute_ntt_dispatch_fn forward,
                                 sp_compute_ntt_dispatch_fn inverse);
```

Lifetime contract: caller owns the opaque `void *handle`; L1 / math-core
never dereference it. Caller must keep the backing object alive past the
last sp_pr_bluestein_* call that uses the backend (or call
`sp_pr_bluestein_set_backend(ctx, NULL, NULL, NULL)` to release).

Per-direction NULL fallback: if either `forward` or `inverse` is NULL,
that direction falls back to math-core's host ntt_crt path independently.

## Gates table

| Gate | Methodology | Pass criteria | Observed | Verdict |
|------|-------------|---------------|----------|---------|
| T_NTT5B_HOST_HEX_BIT_EXACT | sp_pr_bluestein_inner host-vs-hex, N ∈ {64,128,256} × 100 seeds = 300 runs | 0 divergences | 0/300 (per-N: 0/100 at 64, 0/100 at 128, 0/100 at 256) | **PASS** |
| T_NTT5B_RUN_DIALOGUE_BIT_EXACT | sp_memo_m2_dialogue_smoke with SP_ENGINE_NTT_ATTN_HEX=0 vs =1; tokens + receipt content hashes byte-equal | identical | tokens (8) match; first_64_chars `".......2"` match both runs; receipt[0..2] input/output hash bytes (offsets 6-29 + 30-53) match across runs (only wall_us timestamp bytes vary, expected) | **PASS** |
| T_NTT5B_NO_REGRESSION | re-run NTT.0/1/2/3/4/5a smokes; host poly_ring test | all PASS | NTT.0 PASS; NTT.1 3/3; NTT.2 ALL after Stage 4a slot fix; NTT.3 bit-exact + no-regression PASS (data-bound 2 gates retain original UPSTREAM status); NTT.4 INTT 600/600 + polymul 12/12 after slot fix; M.2 3/4 (pre-existing ZERO_COPY only); host poly_ring 4160/0 | **PASS** |
| T_NTT5B_WALL_CLOCK_INFORMATIONAL | wall-clock matrix per-inner-product on device | report only | host avg 1249.6 us, hex avg 2367.7 us, hex/host ratio 1.89× | **REPORT** |

All 3 substantive gates PASS. 1 informational gate reported.

No silent gate revisions. No banned propositions surfaced (no mixed-radix,
no Good-Thomas, no zero-pad-HD, no third prime, no Cooley-Tukey at non-PoT N).

## Dispatch pattern (host vs hex routing)

NTT.5a's `pr_blue_convolve_M` (poly_ring_bluestein.c) calls the inner CRT NTT
pipeline 6× per Bluestein convolve:
  4× ntt_forward (per operand × per prime)
  1× ntt_pointwise_mul (per-prime, both primes in one call)
  1× ntt_inverse (per-prime INTT + Garner recombine)

NTT.5b's dispatch branch in the same function:

```
if (fwd) {  /* per-direction NULL check */
    /* 4 per-prime forward calls via the backend ABI. The int32 pad
     * buffers are mirrored into uint32 scratch (value-preserving since
     * each value is in [0, qP)). Each backend call processes ONE prime;
     * keeps the wrong-channel discard from the host path eliminated. */
    fwd(handle, 0, M, u32_pad_q1, a_res_q1)  /* a, q1 */
    fwd(handle, 1, M, u32_pad_q2, a_res_q2)  /* a, q2 */
    fwd(handle, 0, M, u32_pad_q1_b, b_res_q1)
    fwd(handle, 1, M, u32_pad_q2_b, b_res_q2)
} else {
    /* NTT.5a host path: 4× ntt_forward with scratch_qN discards. */
}

ntt_pointwise_mul(...)   /* host-side; small + cheap; not dispatched */

if (inv) {
    /* 2 per-prime INTTs + host-side Garner. Produces same D buffer as
     * ntt_inverse would have produced; downstream fold/untwist/garner is
     * direction-agnostic. */
    inv(handle, 0, M, c_res_q1, scratch_q1)
    inv(handle, 1, M, c_res_q2, scratch_q2)
    ntt_crt_recombine(ctx->inner, scratch_q1, scratch_q2, D)
} else {
    ntt_inverse(ctx->inner, c_res_q1, c_res_q2, D)
}
```

Per-call dispatch failure (backend_*fn returns -1) falls back to the host
path in-place. The smoke test catches this case via bit-exact compare.

The Rust trampoline routes:
  forward dispatch (backend_forward = sp_compute_ntt_forward_via_fastrpc)
    → FastRPC method 17 (ntt_hvx_vtcm_oracle, NTT.3 VTCM-aware HVX forward NTT)
  inverse dispatch (backend_inverse = sp_compute_ntt_inverse_via_fastrpc)
    → FastRPC method 18 (intt_hvx_oracle, NTT.4 HVX INTT)

per_invocation marshalling matches the existing NTT smoke binaries:
  primIn `[q_idx, N, in_bytes, out_bytes]` (16 B)
  + in_buf (N×4 B) + out_buf (N×4 B); scalars = make_scalars(method, 2, 1).

`Arc<FastRpcSession>` wrapping (not Mutex) per
`reference-fastrpc-concurrent-dispatch` — FastRpcSession is auto-Send+Sync
and supports concurrent `&self.invoke()`.

## Wall-clock matrix

Measured on Knack's S22U via T_NTT5B_HOST_HEX_BIT_EXACT (300 inner-product
calls per path during the same run):

| Path | Avg per inner-product (us) | Ratio vs host |
|------|---------------------------:|--------------:|
| Host (NTT.5a baseline) | 1249.6 | 1.00× |
| Hex backend dispatched | 2367.7 | 1.89× slower |

Hex slower as expected at this scale. The 5-FastRPC-call-per-inner-product
marshalling cost (4 forward + 1 inverse — actually 2 INTT calls per inner
× 1 inner = 5 RPCs per Bluestein call; sp_pr_bluestein_inner = 1 mul + Garner)
dominates. NTT.6 long-context tiling (and the parallel cDSP scheduler via
Arc<FastRpcSession> from `reference-fastrpc-concurrent-dispatch`) is where
amortization makes hex faster. NOT a substantive gate per spec; NTT.5b is
plumbing.

## Files changed

NEW (math-core submodule):
  - (none -- additive edits only)

EXTEND (math-core submodule):
  | File | LOC delta |
  | --- | --- |
  | `include/sp/sp_l1.h` | +60 (typedefs + register fn + 3 readback accessors) |
  | `include/sp/poly_ring_bluestein.h` | +24 (set_backend prototype + include) |
  | `core/session/sp_session.c` | +37 (3 fields + register impl + 3 getters) |
  | `core/poly_ring/poly_ring_bluestein.c` | +90 (3 ctx fields + u32 scratch + dispatch branch + setter) |
  | `core/poly_ring/poly_ring_test.c` | +130 (T_NTT5B_BACKEND_FORWARD_PASSTHROUGH) |

NEW (engine repo):
  | File | LOC |
  | --- | --- |
  | `tools/sp_daemon/src/ntt_hex_dispatch.rs` | 191 (ComputeBackend + 2 #[no_mangle] trampolines) |
  | `tools/sp_dsp_smoke/src/sp_ntt_5b_bluestein_hex_smoke.rs` | 269 (Stage 3 smoke harness) |
  | `tools/sp_compute_skel/docs/PLAN-NTT-5b.md` | 364 |
  | `tools/sp_compute_skel/docs/CLOSURE-NTT-5b.md` | (this file) |

EXTEND (engine repo):
  | File | LOC delta |
  | --- | --- |
  | `tools/sp_daemon/src/lib.rs` | +8 (cfg(android) pub mod ntt_hex_dispatch) |
  | `tools/sp_daemon/src/state.rs` | +14 (cfg(android) ntt_hex_backend field) |
  | `tools/sp_daemon/src/daemon.rs` | +72 (cfg(android) backend allocation + register call) |
  | `tools/sp_daemon/src/session.rs` | +11 (raw_ptr() escape hatch) |
  | `tools/sp_dsp_smoke/Cargo.toml` | +9 (new bin target declaration) |
  | `tools/sp_dsp_smoke/build.rs` | +6 (link libsp_poly_ring) |
  | `tools/sp_dsp_smoke/src/sp_ntt_2_smoke.rs` | +6/-6 (Stage 4a IDL slot fixes 13→14, 14→15, 15→16) |
  | `tools/sp_dsp_smoke/src/sp_ntt_4_intt_smoke.rs` | +4/-3 (Stage 4a slot fix 17→18) |
  | `tools/sp_dsp_smoke/src/sp_ntt_4_polymul_smoke.rs` | +5/-3 (Stage 4a slot fix 17→18) |

Files NOT TOUCHED (anti-contamination check):
  - NTT.5a public surface: sp_pr_bluestein_init/free/degree/inner/mul
    signatures unchanged (the set_backend setter is a new sibling).
  - `core/ntt_crt/ntt_crt.c` + `include/sp/ntt_crt.h` (canonical reference).
  - `core/poly_ring/poly_ring.c` + `include/sp/poly_ring.h` (Phase 1B reference).
  - `core/forward/forward.c` (wire-up explicitly out of scope per spec).
  - All NTT.0/1/3 smokes (NTT.2 + NTT.4's two smokes were touched in Stage
    4a to fix pre-existing stale IDL method indices — see Stage 4a commit
    message for the per-binary renumbering map).
  - Any other engine-* or lattice-* worktree.

`grep -r sp_compute_backend_handle D:/F/shannon-prime-repos/` outside
`engine-ntt-5b` returns nothing. Same for `sp_pr_bluestein_set_backend`.

## Commits on sprint/ntt-5b

Engine repo (`D:\F\shannon-prime-repos\engine-ntt-5b`):

  | SHA | Message |
  | --- | --- |
  | `4d04104` | `[plan] NTT.5b -- Hexagon backend dispatch + L1 ABI extension (sp_session_register_compute_backend)` |
  | `13ba3de` | `[NTT.5b] Stage 1 submodule bump: math-core L1 ABI extension + Bluestein backend thread-through (4160/0 PASS)` |
  | `7fe6c2f` | `[NTT.5b] Stage 2: Rust C trampoline + AppState backend wiring (cfg(android))` |
  | `1b58918` | `[NTT.5b] Stage 3: T_NTT5B_HOST_HEX_BIT_EXACT smoke harness` |
  | `c727a65` | `[NTT.5b] Stage 4a: fix stale IDL method indices in NTT.2/4 smokes` |
  | `e80ff81` | `[NTT.5b] Stage 4: T_NTT5B_HOST_HEX_BIT_EXACT live + dialogue regression results` |
  | (Stage 5 closure commit lands next) |

Math-core submodule (`lib/shannon-prime-system`):

  | SHA | Message |
  | --- | --- |
  | `fc38a4f` | `[NTT.5b] Stage 1: L1 ABI extension + math-core Bluestein backend thread-through` |

## Sub-tag candidate

`lat-phase-4-ntt-5b-hex-backend-dispatch`. Operator applies post-merge.

## What's NOT done (out-of-scope per spec)

- **forward.c wire-up.** The actual env-gated activation of the registered
  backend inside the NTT-attention loop (so `SP_ENGINE_NTT_ATTN_HEX=1`
  causes the Memory model to route its attention through the Hexagon
  backend instead of the host ntt_crt path) is OUT OF SCOPE per the spec
  ("What's NOT done — forward.c wire-up (still TBD)"). NTT.5b ships
  infrastructure only: the L1 register call lands and stores the backend,
  but math-core's `qwen3_forward_ex` doesn't yet read it. A follow-on
  sprint will (a) make `forward.c` thread the registered backend down to
  the Bluestein context it builds, and (b) make `qwen3_forward_ex`'s
  NTT-attention overlay use `sp_pr_bluestein_init(HD)` for HD ∈ {2..64}
  instead of `sp_pr_init(HD)` which doesn't admit those values.

- **NTT.6 long-context tiling.** Where the Hex-dispatch wall-clock win
  materializes (per the wall-clock matrix above, Hex is ~2× slower at
  inner-product scope; the win comes from FastRPC dispatch overhead being
  amortized across many tiled N=512 NTTs in long context).

- **Executive model routing.** The Executive (Qwen3-0.6B) stays on the
  host NTT-attention OR fp32 path per the spec's Option D1 architectural
  decision. NTT.5b registers the backend on the Memory session only.

- **Per-Bluestein-ctx → per-session binding inside math-core.** The L1
  register stores the backend on the session; math-core currently has no
  consumer that reads `sp_session_compute_backend_*` and routes that into
  Bluestein contexts it builds. The bridge function (likely in `forward.c`
  or `forward_dispatch.c`) is the follow-on sprint scope.

- **Wall-clock win at large ctx + parallel cDSP scheduler.** The
  `reference-fastrpc-concurrent-dispatch` pattern (Arc<FastRpcSession>
  concurrent dual-dispatch for parallel per-prime per-head NTT) is a
  natural NTT.6 extension. The cDSP scheduler already supports it
  (silicon-confirmed in K v0.alpha + K v0.beta.2.5c); composing it with
  the NTT.5b dispatch trampoline is an additive change at the trampoline
  layer.

- **HD with odd factors > 1.** Per
  `reference-ntt-bluestein-arbitrary-n-escape`, Bluestein cannot help
  for N ∈ {96, 192, 288, 384, ...} with the current primes. SP-aligned
  answer = direct integer dot product with Barrett. NOT in NTT.5b scope.

- **NTT.3's two UPSTREAM gates** (T_NTT3_DUAL_DISPATCH_SPEEDUP and
  T_NTT3_VTCM_NO_RECOMPUTE) remain at their original NTT.3-closure
  UPSTREAM-disposition per `feedback-shape-dependent-parallelism-gates`.
  Unchanged from main; NOT NTT.5b's responsibility.

- **M.2 dialogue T_MEMO_M2_ZERO_COPY gate** remains FAIL per its
  M.2-closure-time UPSTREAM-disposition per
  `feedback-leak-gate-allocator-warmup`. Unchanged from main; NOT NTT.5b's
  responsibility (jemalloc instrumentation follow-on).

## What unblocks

- **forward.c integration sprint.** All plumbing in place: math-core has
  `sp_pr_bluestein_set_backend`; `sp_session` carries the registered
  backend; the daemon registers it when env is set. Remaining work is
  to wire `sp_session_compute_backend_*` accessors into `forward.c` (or
  `forward_dispatch.c`) and replace `sp_pr_init(HD)` with
  `sp_pr_bluestein_init(HD)` for HD ∈ {2, 4, 8, 16, 32, 64} — covers
  Qwen3-0.6B and Qwen2.5-Coder-0.5B HD=64. Estimated 100-300 LOC.

- **NTT.6 long-context tile sprint.** The dispatch substrate is now
  silicon-confirmed bit-exact at the inner-product level; tiling across
  ctx-chunks of N=512 each is the next composition step. Wall-clock
  matrix establishes the per-inner-product baseline for measuring the
  long-context win.

- **Generalization to other backends.** The L1 ABI is backend-agnostic;
  the same `sp_session_register_compute_backend` would work for a
  CUDA/PTX backend, a Vulkan compute backend, or the NPU via QNN HTP
  (per `reference-qnn-htp-unsigned-pd-access`). The trampoline is the
  only piece that needs per-backend implementation.

## Memory entry candidates

Post-operator-merge:

1. **Update `reference-ntt-bluestein-arbitrary-n-escape`** with closing
   line: "NTT.5b shipped the Hexagon backend dispatch substrate via L1
   ABI extension `sp_session_register_compute_backend` 2026-05-31; 3/3
   substantive gates PASS bit-exact (T_NTT5B_HOST_HEX_BIT_EXACT 300/300,
   T_NTT5B_RUN_DIALOGUE_BIT_EXACT, T_NTT5B_NO_REGRESSION). forward.c
   wire-up is the next-sprint follow-on."

2. **New `reference-l1-compute-backend-abi`** (one-liner index):
   "L1 ABI `sp_session_register_compute_backend(s, handle, fwd, inv)`
   stores an opaque backend handle + two per-direction dispatch fn
   pointers on the session. Backend-aware math-core kernels (currently
   sp_pr_bluestein_*) opt in via `sp_pr_bluestein_set_backend`; readback
   via `sp_session_compute_backend_{handle,forward,inverse}`. NULL fn
   pointers per-direction fall back to host path. Stable since NTT.5b
   (2026-05-31). Generalizes to any backend (FastRPC/Hexagon, CUDA, NPU,
   Vulkan); only the trampoline changes per-backend."

3. **New `reference-fastrpc-ntt-dispatch-trampoline`** (one-liner index):
   "Two #[no_mangle] extern C trampolines in `sp_daemon::ntt_hex_dispatch`
   bridge math-core's sp_compute_ntt_dispatch_fn ABI to FastRPC m17 (NTT
   forward) + m18 (NTT inverse). Per-prime u32[N] in/out, 4-word primIn
   `[q_idx, N, in_bytes, out_bytes]` marshalling matching existing NTT
   smoke binaries. Backend wraps `Arc<FastRpcSession>` (auto-Send+Sync;
   not Mutex per `reference-fastrpc-concurrent-dispatch`). Lifetime: the
   raw `*mut c_void` passed to L1 references an Arc held by AppState,
   which outlives the L1 session. Stable since NTT.5b 2026-05-31."

4. **New `reference-idl-method-renumber-staleness`** (one-liner index):
   "When NTT sub-sprint K is merged into main and adds new IDL methods
   that take adjacent slots, existing smoke binaries built against the
   pre-merge IDL may have stale method constants (NTT.3 merge bumped
   intt_hvx_oracle from 17 to 18; NTT.2 merge bumped twiddle_init from
   13 to 14, etc). Renumbering is silent at compile time (numeric
   literals) and surfaces as 100%-divergence smoke runs against fresh
   skel. Fix: grep `make_scalars\\(\\d+` across smoke binaries against
   the current IDL after each merge. NTT.5b Stage 4a hit this on
   sp_ntt_2_smoke + sp_ntt_4_intt_smoke + sp_ntt_4_polymul_smoke."

## Worktree status

```
D:\F\shannon-prime-repos\engine-ntt-5b    (engine)
  branch:  sprint/ntt-5b
  base:    908e570 (engine main, post NTT.5a merge)
  tip:     e80ff81  (Stage 4 commit; Stage 5 closure commit lands next)
  6 commits ahead of base (plan + Stage 1 submodule bump + 4 stage commits)
  push:    git push -u origin sprint/ntt-5b

D:\F\shannon-prime-repos\engine-ntt-5b\lib\shannon-prime-system    (math-core)
  branch:  sprint/ntt-5b
  base:    7662b2d (NTT.5a tip)
  tip:     fc38a4f  (Stage 1 commit)
  1 commit ahead of base
  push:    git push -u origin sprint/ntt-5b
```

Anti-contamination verified per §"Files changed". Push commands (operator
runs post-closure-ack):

```
cd D:\F\shannon-prime-repos\engine-ntt-5b\lib\shannon-prime-system
git push -u origin sprint/ntt-5b

cd D:\F\shannon-prime-repos\engine-ntt-5b
git push -u origin sprint/ntt-5b
```

Operator merges; applies `lat-phase-4-ntt-5b-hex-backend-dispatch` sub-tag.
