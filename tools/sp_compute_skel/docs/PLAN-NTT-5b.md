# PLAN-NTT-5b.md — Sprint NTT.5b: Hexagon backend dispatch for Bluestein + L1 ABI extension

## Stage 0 — Mandatory pre-read citations

The eight citations the spec requires (re-verified against source).

1. **`reference-ntt-bluestein-arbitrary-n-escape`** memory entry:
   - Bluestein with frozen primes admits only `{2,4,8,16,32,64,128,256}`; N=512 + non-power-of-2 N rejected.
   - Banned alternatives: mixed-radix, Good-Thomas, zero-padding HD, third prime (lines 48–54 of the memory entry).
   - For HD with odd factor: direct dot product with Barrett (no transform), not exotic NTT.

2. **`reference-ntt-frozen-primes-N-cap`** memory entry:
   - `v_2(q_i - 1) = 10` for both frozen primes; max 2N = 1024 → max N = 512.
   - Cited surface: `lib/shannon-prime-system/core/ntt_crt/ntt_crt.c:189`
     (NTT.5b worktree `ntt_crt.c` confirms N admissibility unchanged: 128/256/512).
   - Long context = tile, never widen N. Not in NTT.5b scope; long-context wrapper is NTT.6.

3. **`reference-sp-uses-phi-extensively`** memory entry:
   - φ already lives in Fibonacci-Prime DHT (Phase 8), Fibonacci KV sub-sampling, SP_ROPE_PHI, Halton attachments.
   - No new φ proposals in NTT.5b. Bluestein wrapping is the chirp-z escape, not a φ extension.

4. **`reference-fastrpc-concurrent-dispatch`** memory entry:
   - `FastRpcSession` is auto-Send+Sync via `libloading::Library` + bare-fn-pointers + `u64` handle. Wrap in `Arc`, NOT `Mutex`, for concurrent dispatch.
   - Cited surface: `tools/sp_daemon/src/dsp_rpc.rs:140-260` (`FastRpcSession` + `invoke(&self, ...)` is the substrate).
   - NTT.5b's Rust trampoline receives an opaque pointer that decodes back to `Arc<FastRpcSession>`.

5. **`reference-vtcm-per-stage-misalignment`** memory entry:
   - NTT.2's per-stage twiddle layout misaligns stages 2+. NTT.3 took aligned-copy scratch (slow); NTT.4 took `vmemu` (correct).
   - Since NTT.5b dispatches into existing method 17 (`ntt_hvx_vtcm_oracle` — NTT.3 path) and method 18 (`intt_hvx_oracle` — NTT.4 path), the alignment is handled INSIDE the existing skel kernels. NTT.5b does NOT introduce new HVX kernel code. Trampoline marshalling matches the byte-exact contract of those existing methods.

6. **NTT.5a closure** — `tools/sp_compute_skel/docs/CLOSURE-NTT-5a.md`:
   - Bluestein pipeline: per-prime psi-twist → zero-pad to length-M → 4× `ntt_forward` + 1× `ntt_pointwise_mul` + 1× `ntt_inverse` → fold to length-N → per-prime untwist → local Garner.
   - 4× `ntt_forward` exists because `ntt_forward` fuses both primes from one int32 input; per-prime twisted inputs differ.
   - Wall-clock at N=128: 3.49× slower than direct `sp_pr_inner` (235 µs vs 67 µs); expected, NTT.5b is plumbing.

7. **NTT.5a math-core code** — `lib/shannon-prime-system/include/sp/poly_ring_bluestein.h` + `core/poly_ring/poly_ring_bluestein.c`:
   - Re-confirmed: only THREE ntt_crt entry points called from the wrapper — `ntt_forward` (4×), `ntt_pointwise_mul` (1×), `ntt_inverse` (1×) — in `pr_blue_convolve_M` (`poly_ring_bluestein.c:385-403`).
   - Public API: `sp_pr_bluestein_init` (poly_ring_bluestein.c:279), `sp_pr_bluestein_inner` (poly_ring_bluestein.c:470), `sp_pr_bluestein_mul` (poly_ring_bluestein.c:447), `sp_pr_bluestein_free` (poly_ring_bluestein.c:329), `sp_pr_bluestein_degree` (poly_ring_bluestein.c:345).
   - Inner ctx field `inner` (poly_ring_bluestein.c:255) holds the math-core `ntt_ctx*`.

8. **NTT.3 + NTT.4 IDL surface** — `tools/sp_compute_skel/inc/sp_compute.idl`:
   - Method 13: `ntt_hvx_oracle` (per-call psi precompute; not VTCM-aware).
   - Method 14: `ntt_twiddle_init` (idempotent VTCM warm).
   - Method 17: `ntt_hvx_vtcm_oracle` (VTCM-aware HVX forward NTT) — **NTT.5b's forward endpoint**.
   - Method 18: `intt_hvx_oracle` (HVX INTT + post-pass × ninv × ipsi). **NTT.5b's inverse endpoint**.

Bonus: examined `dsp_rpc.rs:122-180` (forward NTT marshalling layout: `[q_idx, N, in_bytes, out_bytes]` primIn + in_buf + out_buf, scalars `make_scalars(method, 2, 1)`) — NTT.5b's Rust trampoline mirrors this verbatim.

## Bluestein dispatch surface analysis

From NTT.5a's `pr_blue_convolve_M` (`poly_ring_bluestein.c:385-403`), the inner pipeline calls math-core's `ntt_crt.h` API exactly six times per Bluestein convolve:

| # | call | direction | use in pipeline |
|---|------|-----------|-----------------|
| 1 | `ntt_forward(inner, a_pad_q1, a_res_q1, scratch_q2)` | fwd | a, q1-twisted input → keep q1 residue |
| 2 | `ntt_forward(inner, a_pad_q2, scratch_q1, a_res_q2)` | fwd | a, q2-twisted input → keep q2 residue |
| 3 | `ntt_forward(inner, b_pad_q1, b_res_q1, scratch_q2)` | fwd | b, q1-twisted input → keep q1 residue |
| 4 | `ntt_forward(inner, b_pad_q2, scratch_q1, a_res_q2)` | fwd | b, q2-twisted input → keep q2 residue |
| 5 | `ntt_pointwise_mul(inner, ...)` | – | per-prime pointwise multiply |
| 6 | `ntt_inverse(inner, c_res_q1, c_res_q2, D)` | inv | length-M INTT + Garner → signed int64 |

NTT.5b dispatches the four `ntt_forward` calls and the one `ntt_inverse` call through FastRPC. `ntt_pointwise_mul` stays host-side (Barrett scalar; identical to NTT.4's pattern at `sp_ntt_4_polymul_smoke.rs:178-180`) — pointwise multiply is cheap and lifting it to Hexagon would just add a dispatch round-trip.

### Per-prime decomposition

Math-core's `ntt_forward` fuses both primes from one int32 input, but Hexagon's method 17 (`ntt_hvx_vtcm_oracle`) takes a **single** `q_idx` (0 or 1) and writes a single u32[N] output. That's exactly what we need:

- Host calls (1) become Hex: `method 17, q_idx=0, in=a_pad_q1[0..M)` → a_res_q1
- Host calls (2) become Hex: `method 17, q_idx=1, in=a_pad_q2[0..M)` → a_res_q2
- (3), (4) similarly for b
- Host call (6) splits into TWO INTTs (one per prime), each into u32[M]; then host runs Garner over M signed-int64 entries

Note: Hexagon's INTT (`intt_hvx_oracle`) outputs `out_byte = N × u32 LE` in `[0, q)` (per IDL contract). The math-core `ntt_inverse` produces signed centered int64 from CRT recombination. So the Hex side INTT gives the per-prime residues; the trampoline must do the Garner recombination host-side (or call `ntt_crt_recombine`, which already exists and just reads the inner ctx's frozen primes).

**Simpler approach (chosen):** post-dispatch, write a small `pr_blue_garner_M(int M, u32* x1, u32* x2, int64_t* out)` helper IN THE WRAPPER (not modifying ntt_crt.c) that does the per-element Garner using the same algorithm as `pr_blue_garner` (already in poly_ring_bluestein.c:187) but extended to a length-M loop. This avoids `ntt_crt_recombine`'s coupling to the inner ctx and stays in additive-only territory.

Actually, even simpler: math-core's existing public `ntt_crt_recombine(ctx, x1, x2, out)` iterates `ctx->N` entries (= our inner M) and writes M int64 outputs to `out`. That's exactly what we need — the host-side fold/untwist then operates on the M-long D buffer as before. So we use `ntt_crt_recombine` directly.

## Architectural choice: parallel `_hex` API or thread a backend pointer through?

Two options for math-core wiring:

- **(A)** Add backend-aware variants `sp_pr_bluestein_inner_hex(ctx, backend, q, k)` etc. — duplicates the public surface.
- **(B)** Add a `sp_pr_bluestein_set_backend(ctx, backend)` setter — mutates context but keeps inner/mul signatures unchanged.

Choosing **(B)**. The Bluestein context already holds per-call scratch; adding a `compute_backend` pointer is one more field. When unset (default), the existing host pipeline runs unchanged; when set, `pr_blue_convolve_M` dispatches through the backend. NTT.5a's signatures stay intact; ABI is additive.

The setter is internal to the math-core's Bluestein wrapper; the L1 ABI sees it through `sp_session_register_compute_backend`, which then plumbs through to (eventually) any Bluestein contexts the session manages. **For Stage 1 the L1 register fn stores the backend on the session and the Bluestein context grows a setter**; the actual wiring of session-backend → bluestein-ctx happens at the smoke harness level (Stage 3 directly calls `sp_pr_bluestein_set_backend`). Wiring the session-internal Bluestein ctx into forward.c (so `SP_ENGINE_NTT_ATTN_HEX=1` flips on automatically) is OUT OF SCOPE for NTT.5b per the spec's "What's NOT done — forward.c wire-up (still TBD)".

## L1 ABI extension — exact signatures

Per the operator-pre-authorized signatures in the spec:

```c
/* sp/sp_l1.h — APPEND after existing surface */

/* Opaque handle pointer (Rust-side Arc<FastRpcSession> or equivalent). */
typedef struct sp_compute_backend_handle sp_compute_backend_handle;

/* Dispatch fn for one prime, one direction.
 *   handle: backend-supplied opaque pointer (registered at backend-setup time)
 *   q_idx: 0 for q_1, 1 for q_2
 *   N: transform length (must match math-core ntt_init admissible set: 128, 256, 512)
 *   in/out: N × u32 little-endian buffers in [0, q) for forward (and INTT output);
 *           forward input is N × i32 little-endian (arbitrary signed; reduced inside).
 *           Actually, to unify host + hex paths, both directions take N × u32 LE in [0, q);
 *           the wrapper does the int32 → u32-mod-q reduction host-side before dispatch.
 * Returns 0 on success, -1 on error. */
typedef int (*sp_compute_ntt_dispatch_fn)(
    void *handle, int q_idx, int N,
    const uint32_t *in, uint32_t *out);

/* Register a compute backend for this session. After registration, math-core's
 * sp_pr_bluestein_* APIs route inner NTT calls through the backend instead of
 * host ntt_crt. Pass NULL handle + NULL fn pointers to unregister.
 * Returns 0 on success, -1 on invalid arguments. */
int sp_session_register_compute_backend(
    sp_session *s,
    void *handle,
    sp_compute_ntt_dispatch_fn forward,
    sp_compute_ntt_dispatch_fn inverse);
```

Refinement: the dispatch fn takes uniform `u32` buffers for both directions. The
Bluestein wrapper converts its `int32_t *pad_qN` to `uint32_t` before dispatch by
trivially copying — `pad_qN` already holds values in `[0, qP)` per
`pr_blue_twist_input` (poly_ring_bluestein.c:354-379), so the conversion is a
no-op reinterpret. We pass a freshly-prepared `u32` view via `(const uint32_t *)pad_qN`
(both types are 4 bytes; values fit in both). This unifies the IDL-level u32 byte layout
with the host path.

### Bluestein context backend setter (internal to math-core)

```c
/* poly_ring_bluestein.h — APPEND after existing surface */

#include "sp/sp_l1.h"   /* for sp_compute_ntt_dispatch_fn */

/* Attach a compute backend for the inner NTT calls. When set, sp_pr_bluestein_inner
 * and sp_pr_bluestein_mul dispatch the 4 forward + 1 inverse inner NTT calls through
 * the supplied function pointers instead of math-core's host ntt_forward / ntt_inverse.
 * Pointwise multiply stays host-side (small + cheap).
 *
 * Setting any of forward/inverse to NULL reverts that direction to host path.
 * Setting all three (handle, forward, inverse) to NULL clears the backend entirely.
 *
 * Thread-safety: caller-serialized (same as the ctx's other accessors). */
void sp_pr_bluestein_set_backend(sp_pr_bluestein_ctx *ctx,
                                 void *handle,
                                 sp_compute_ntt_dispatch_fn forward,
                                 sp_compute_ntt_dispatch_fn inverse);
```

The poly_ring_bluestein.c implementation grows three fields in `struct sp_pr_bluestein_ctx`
(handle, forward fn ptr, inverse fn ptr), the setter, and the dispatch logic in
`pr_blue_convolve_M`. When `forward == NULL`, host path; when set, the dispatch path.

### Session backend storage

`sp_session` grows three fields (handle + forward + inverse). The
`sp_session_register_compute_backend` function stores them. NTT.5b's wiring stops here at
Stage 1 — the actual binding of session-backend → bluestein-ctx is a future hand-off.

## Opaque-handle plumbing pattern (Rust side)

```rust
// tools/sp_daemon/src/ntt_hex_dispatch.rs (new)

use std::sync::Arc;
use std::os::raw::{c_int, c_void};
use crate::dsp_rpc::{FastRpcSession, make_scalars, RemoteArg, RemoteBuf};

/// Wrapper held in AppState; the raw pointer passed across FFI is *const this.
pub struct ComputeBackend {
    pub session: Arc<FastRpcSession>,
}

/// C ABI trampoline for forward NTT dispatch.
/// handle: *const ComputeBackend (cast back inside).
#[no_mangle]
pub extern "C" fn sp_compute_ntt_forward_via_fastrpc(
    handle: *mut c_void, q_idx: c_int, n: c_int,
    in_buf: *const u32, out_buf: *mut u32,
) -> c_int {
    if handle.is_null() || in_buf.is_null() || out_buf.is_null() { return -1; }
    let backend: &ComputeBackend = unsafe { &*(handle as *const ComputeBackend) };
    let n_bytes = (n as usize) * 4;
    let mut prim_in: [u32; 4] = [q_idx as u32, n as u32, n_bytes as u32, n_bytes as u32];
    let in_slice = unsafe { std::slice::from_raw_parts(in_buf as *const u8, n_bytes) };
    let mut in_bytes: Vec<u8> = in_slice.to_vec();
    let mut out_bytes: Vec<u8> = vec![0u8; n_bytes];
    let mut args = [
        RemoteArg { buf: RemoteBuf { pv: prim_in.as_mut_ptr() as *mut c_void, nlen: 16 }},
        RemoteArg { buf: RemoteBuf { pv: in_bytes.as_mut_ptr() as *mut c_void, nlen: n_bytes }},
        RemoteArg { buf: RemoteBuf { pv: out_bytes.as_mut_ptr() as *mut c_void, nlen: n_bytes }},
    ];
    let rc = backend.session.invoke(make_scalars(17, 2, 1), &mut args);  // m17 = ntt_hvx_vtcm_oracle
    if rc.is_err() { return -1; }
    let out_slice = unsafe { std::slice::from_raw_parts_mut(out_buf as *mut u8, n_bytes) };
    out_slice.copy_from_slice(&out_bytes);
    0
}

#[no_mangle]
pub extern "C" fn sp_compute_ntt_inverse_via_fastrpc(
    handle: *mut c_void, q_idx: c_int, n: c_int,
    in_buf: *const u32, out_buf: *mut u32,
) -> c_int {
    /* same pattern but method 18 (intt_hvx_oracle) */
    /* INTT output is u32 [0, q) per-prime; caller does Garner across both primes */
    ...
}
```

The trampoline lives inside the daemon. Memory model session create time calls
`sp_session_register_compute_backend(memo_session, backend_ptr, fwd, inv)`. At
Stage 1 we plumb through the L1 ABI; the Memory-session wiring happens at Stage 2.

Lifetime: `ComputeBackend` lives in `AppState`. The raw pointer passed to L1 stays
valid as long as `AppState` outlives all sessions. Standard daemon discipline.

## Stage plan

### Stage 1 — L1 ABI extension + math-core thread-through

Math-core submodule changes:

1. `include/sp/sp_l1.h`: append `sp_compute_backend_handle` opaque typedef, `sp_compute_ntt_dispatch_fn` typedef, `sp_session_register_compute_backend` prototype.
2. `core/session/sp_session.c`: add 3 fields to struct sp_session (`backend_handle`, `backend_forward`, `backend_inverse`); implement `sp_session_register_compute_backend`. Initialize fields to NULL in `sp_session_create`.
3. `include/sp/poly_ring_bluestein.h`: append `sp_pr_bluestein_set_backend` prototype.
4. `core/poly_ring/poly_ring_bluestein.c`:
   - Add 3 fields to `struct sp_pr_bluestein_ctx` (`backend_handle`, `backend_forward`, `backend_inverse`); initialize to NULL in init.
   - Implement `sp_pr_bluestein_set_backend` setter.
   - In `pr_blue_convolve_M`: branch on backend presence. When set, replace the 4 `ntt_forward` calls with `backend_forward(handle, q_idx, M, in_u32, out_u32)`. Replace the 1 `ntt_inverse` call with TWO calls to `backend_inverse(handle, q_idx, M, in_u32, out_u32)` (one per prime), then call `ntt_crt_recombine(ctx->inner, c_res_q1, c_res_q2, D)` to recombine to signed centered int64 — same final D buffer either path.
   - When backend NOT set: existing host path, unchanged.

Verification at Stage 1: a stub dispatch fn in the math-core test that just forwards to `ntt_forward`/`ntt_inverse` proves bit-exactness host-vs-host through the new dispatch path. Add this as `T_PR_BLUE_5_BACKEND_PASSTHROUGH` in `poly_ring_test.c`.

Commit: `[NTT.5b] Stage 1: L1 ABI extension + math-core Bluestein backend thread-through`

### Stage 2 — Rust C trampoline + AppState register

Daemon side:

1. `tools/sp_daemon/src/ntt_hex_dispatch.rs` (new): `ComputeBackend` struct, two `#[no_mangle]` extern "C" trampolines (one forward, one inverse). #[cfg(target_os = "android")].
2. `tools/sp_daemon/src/lib.rs`: `pub mod ntt_hex_dispatch;` (cfg-android).
3. `tools/sp_daemon/src/state.rs`: AppState grows an optional `Option<Arc<ntt_hex_dispatch::ComputeBackend>>` field (cfg-android).
4. `tools/sp_daemon/src/daemon.rs`: at memo_session create time (around line 160), allocate `ComputeBackend { session: Arc::new(memo_dsp_fastrpc_session) }`, call `sp_session_register_compute_backend(memo_session_ptr, backend_ptr, fwd_trampoline, inv_trampoline)` if `SP_ENGINE_NTT_ATTN_HEX` env is set. Otherwise skip — session stays in host-only mode.
5. `tools/sp_daemon/build.rs`: needs no changes (bindgen picks up the new L1 fn from sp_l1.h automatically; trampoline symbols are added to the daemon binary via `#[no_mangle]`).

Note: this Stage 2 *wires the daemon-internal session* but does NOT yet bind backend→Bluestein-ctx; the explicit binding step is Stage 3's smoke harness responsibility (which calls `sp_pr_bluestein_set_backend` directly). The daemon binds via `sp_session_register_compute_backend` but that field is currently only stored — no consumer yet, since forward.c wire-up is OUT OF SCOPE.

Commit: `[NTT.5b] Stage 2: Rust C trampoline + AppState register backend`

### Stage 3 — T_NTT5B_HOST_HEX_BIT_EXACT

`tools/sp_dsp_smoke/src/bin/sp_ntt_5b_bluestein_hex_smoke.rs` (new):

Drives `sp_pr_bluestein_inner` twice per (N, seed) pair:
- Once with `set_backend(NULL, NULL, NULL)` → host path
- Once with `set_backend(&backend, fwd_trampoline, inv_trampoline)` → hex path

Compares the `int64` return value byte-exact. N ∈ {64, 128, 256} (= Bluestein-admissible
inner-product-only sweep; M ∈ {128, 256, 512}). 100 random seeds per N. Coefficients
∈ [-2^14, 2^14) per NTT.5a coefficient bit-exactness bound.

Total: 100 × 3 × 2 = 600 runs.

Pass: 0 divergences in all 600.

Pre-call: `ntt_twiddle_init(512)` once at smoke start (matches `sp_ntt_4_polymul_smoke.rs:107-119`).

Commit: `[NTT.5b] Stage 3: T_NTT5B_HOST_HEX_BIT_EXACT smoke harness (600 runs)`

### Stage 4 — T_NTT5B_RUN_DIALOGUE_BIT_EXACT + T_NTT5B_NO_REGRESSION + T_NTT5B_WALL_CLOCK_INFORMATIONAL

`tools/sp_dsp_smoke/src/bin/sp_ntt_5b_memo_attn_smoke.rs` (new):

The spec calls this `sp_ntt_5b_memo_attn_smoke.rs` (spec scope §4) — driving Memory model forward at ctx=128. The gate goal here is structural: even though forward.c wire-up is OUT OF SCOPE in NTT.5b, the harness can still drive Memory model through L1's `sp_decode_step` and confirm the existing baseline is unchanged when the L1 register call is made but no consumer reads the field. So this is more an integration regression test than the spec's "byte-exact answer" test would have been.

**Operator note on `T_NTT5B_RUN_DIALOGUE_BIT_EXACT`:** the spec promises "run_dialogue() on Knack's S22U with SP_ENGINE_NTT_ATTN_HEX=1 produces byte-identical final answer". Since forward.c wire-up is explicitly OUT OF SCOPE, the env var won't actually route anything yet — the register call lands and no consumer reads it. The test effectively verifies *no regression from the L1 register call being present*. This is documented in the closure under "What's NOT done — forward.c wire-up (still TBD)" so the gate's actual reach matches the spec's explicit "NOT done" item.

Concretely:
- Build `sp_memo_m2_dialogue_smoke` once (already exists, M.2 baseline).
- Run it on device with `SP_ENGINE_NTT_ATTN_HEX=0` (baseline).
- Run it again with `SP_ENGINE_NTT_ATTN_HEX=1` (Stage 2's register-backend path).
- Compare final answer + receipts byte-exact.

Plus `T_NTT5B_NO_REGRESSION`: existing M.2 + M.4 + chat-integration smokes still PASS unchanged. We re-run their existing test binaries.

Plus `T_NTT5B_WALL_CLOCK_INFORMATIONAL`: time both runs of M.2 dialogue. Expectation: nearly identical (no consumer of the backend means no extra FastRPC calls — measured wall-clock should be within noise of the baseline). If anyone wires forward.c in a follow-on sprint, this same harness measures actual NTT-attention overhead.

Commit: `[NTT.5b] Stage 4: dialogue bit-exact + no-regression + wall-clock smoke`

### Stage 5 — Closure

Write `tools/sp_compute_skel/docs/CLOSURE-NTT-5b.md` per spec §"Closure deliverables".

Commit: `[NTT.5b] Stage 5: closure + memory-entry candidates`

## Gates summary

| Gate | Methodology | Pass criteria | Type |
|---|---|---|---|
| T_NTT5B_HOST_HEX_BIT_EXACT | sp_pr_bluestein_inner host-vs-hex on N ∈ {64,128,256} × 100 seeds × 2 paths = 600 runs | 0 divergences | substantive |
| T_NTT5B_RUN_DIALOGUE_BIT_EXACT | sp_memo_m2_dialogue_smoke under both env settings; tokens + receipts byte-equal | identical | substantive |
| T_NTT5B_NO_REGRESSION | re-run M.2 + M.4 + chat-integration + NTT.0/1/2/3/4/5a smokes | all PASS | substantive |
| T_NTT5B_WALL_CLOCK_INFORMATIONAL | measure-and-report wall-clock; no threshold | report only | informational |

## Anti-contamination scope

Within `engine-ntt-5b` only.

**Files NEW:**
- `lib/shannon-prime-system/.../poly_ring_bluestein.h` — append set_backend prototype (additive)
- `tools/sp_daemon/src/ntt_hex_dispatch.rs` (new)
- `tools/sp_dsp_smoke/src/bin/sp_ntt_5b_bluestein_hex_smoke.rs` (new)
- `tools/sp_dsp_smoke/src/bin/sp_ntt_5b_memo_attn_smoke.rs` (new) [if structurally separate from Stage 4's runner]
- `tools/sp_compute_skel/docs/PLAN-NTT-5b.md` (this file)
- `tools/sp_compute_skel/docs/CLOSURE-NTT-5b.md` (Stage 5)

**Files EXTENDED (additive only):**
- `lib/shannon-prime-system/include/sp/sp_l1.h` — append two typedefs + one fn proto
- `lib/shannon-prime-system/include/sp/poly_ring_bluestein.h` — append one fn proto + include
- `lib/shannon-prime-system/core/poly_ring/poly_ring_bluestein.c` — 3 fields + setter + dispatch branch
- `lib/shannon-prime-system/core/session/sp_session.c` — 3 fields + register fn
- `lib/shannon-prime-system/core/poly_ring/poly_ring_test.c` — add T_PR_BLUE_5_BACKEND_PASSTHROUGH
- `tools/sp_daemon/src/state.rs` — optional ComputeBackend field (cfg-android)
- `tools/sp_daemon/src/daemon.rs` — backend allocation + register call (cfg-android)
- `tools/sp_daemon/src/lib.rs` — mod ntt_hex_dispatch (cfg-android)
- `tools/sp_dsp_smoke/Cargo.toml` — declare new bin targets

**Files NOT TOUCHED:**
- NTT.5a's existing sp_pr_bluestein_init/free/degree/inner/mul signatures (added a sibling setter only)
- ntt_crt.h, ntt_crt.c (canonical reference)
- poly_ring.h, poly_ring.c (Phase 1B reference)
- forward.c (wire-up explicitly out of scope per spec)
- Existing NTT.0–NTT.5a smokes
- Any other engine-* / lattice-* worktree

## Banned propositions (already rejected; for the record)

Per `reference-ntt-bluestein-arbitrary-n-escape:48-54`:
- Mixed-radix NTT for non-power-of-2 N — mathematically invalid for frozen primes; surface UPSTREAM.
- Good-Thomas — same.
- Zero-padding HD to next supported size — SP-philosophy regression.
- Adding a third prime — Phase-5+ architectural scope.
- Cooley-Tukey at non-power-of-2 N — primefield constraint at the algorithm level.

None of these arise in NTT.5b because Bluestein wrapping is the chosen escape and is already implemented in NTT.5a; this sprint is pure plumbing.

## What's explicitly NOT in NTT.5b

Per spec §"Closure deliverables → What's NOT done":
- forward.c wire-up (the env-gated activation of the backend within actual NTT-attention) — separate follow-on sprint.
- NTT.6 long-context (ctx ≥ 1024 tiling).
- Executive routing (Executive model stays host).
- Per-Bluestein-ctx → per-session binding (the L1 register stores the backend; nothing in math-core currently consumes session-stored backend pointers).

## Hardware

Knack's S22U FastRPC Path B Unsigned PD. `adb devices` confirmation before Stage 3+.

Stage 1 (host-only math-core) + Stage 2 (Rust-only) runnable on Windows host with cross-compile build. Stage 3 needs an aarch64-android build pushed to device. Stage 4 same.
