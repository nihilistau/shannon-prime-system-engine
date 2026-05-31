# PLAN — WIRE-HEX (sp_daemon → sp_hex_host backend dispatch)

**Sprint:** Phase 2-HX.DAEMON-WIRE — wire sp_daemon's forward to the Hexagon V69 backend
**Date:** 2026-05-31
**Worktree:** `D:\F\shannon-prime-repos\engine-wire`  (branch `sprint/wire-hex-backend`, base 9313306)

---

## Stage 0 — Reference reads (file:line citations)

### 1. Hexagon backend API surface

- `src/backends/hexagon/sp_hex_host.c:112` — **entry point**: `int gemma3_forward_hexagon(const qwen3_model *m, const int32_t *tokens, int n_tok, float *logits)`. Signature is identical to engine `gemma3_forward` (`include/sp_engine/cpu_backend.h` mirror). **WHOLE-FORWARD dispatch** (not per-matmul).
- `src/backends/hexagon/sp_hex_host.c:114` — gemma3-only (`SP_ARCH_GEMMA3` asserted; returns 1 otherwise).
- `src/backends/hexagon/sp_hex_host.c:115` — requires `m->arena` (`SP_ARENA=q8` packed arena).
- `src/backends/hexagon/sp_hex_host.c:118-135` — caches FastRPC handle + uploads ~574 MB Q8 arena ONCE keyed on `m`; subsequent calls reuse it. Embed + LM head run host-side; the cDSP runs the 26 transformer layers + final RMSNorm.
- `src/backends/hexagon/sp_hex_host.c:137-141` — wire payload: `sp_hex_forward(handle, n_layers, ..., n_tok, x, hidden)` via the qaic-generated FastRPC stub.
- `src/backends/hexagon/inc/sp_hex.idl:45-49` — FastRPC IDL `forward` method takes hidden activations [n_tok*n_embd] + a flat weight blob; returns final-normed hidden. Per-chunk = TWO FastRPC calls (upload happens once at open).
- `src/backends/hexagon/CMakeLists.txt:18-20` — `SP_ENGINE_TARGET_ANDROID` required; module is a no-op on other platforms.
- `src/backends/hexagon/CMakeLists.txt:76-87` — sp_hex_host.c + qaic stub are compiled **INTO sp_engine** (not a separate lib). `SP_ENGINE_WITH_HEXAGON=1` enables. Engine links `libcdsprpc.so` + `rpcmem.a`.
- `include/sp_engine/hexagon_backend.h:23-27` — public ABI: `gemma3_forward_hexagon(...)` + `sp_hexagon_model_release(...)`.

### 2. CPU backend dispatch pattern (the per-arch reference)

- `src/backends/cpu/cpu_forward.c:1-8` — engine **owns** its own `qwen3_forward` (NOT a wrapper over math-core's). The engine has a parallel forward implementation tree at `src/forward/`, `src/backends/cpu/`, `src/backends/cuda/`, `src/backends/vulkan/`, `src/backends/hexagon/`. The math-core `core/forward/forward.c` is the REFERENCE used by `sp_session`; the engine forwards are the production paths used by `ppl.c`.
- `src/forward/ppl.c:19` — `typedef int (*forward_fn)(const qwen3_model *, const int32_t *, int, float *);` — the canonical engine forward dispatch typedef. `gemma3_forward_hexagon` matches this signature.
- `src/forward/ppl.c:26` — default selection: `(arch == SP_ARCH_GEMMA3) ? gemma3_forward : qwen3_forward`.
- `src/forward/ppl.c:41-47` — **THE existing hexagon backend dispatch site**: `SP_BACKEND=hexagon` → `fwd = gemma3_forward_hexagon`. **Lives in ppl.c only, NOT exposed to daemon.**

### 3. Math-core sp_matmul = REFERENCE not production

- `lib/shannon-prime-system/include/sp/forward_dispatch.h:25-30` — `sp_matmul(...)` API.
- `lib/shannon-prime-system/core/forward_dispatch/forward_dispatch.c:1-13` — explicit header comment: "pure-f32 path's dot is the scalar reference sp_dot_f32 (forward_kernels) — the engine's #ifdef AVX2 reduction was a CPU-backend variant, not the L1 reference, so it is dropped." Backend dispatch is a backend concern; this is reference.

### 4. sp_session forward kernel — every sp_matmul call site

- `lib/shannon-prime-system/core/session/sp_session.c:167-170` — `sp_prefill_chunk` dispatches to **math-core's** `gemma3_forward_ex2 / qwen25_forward_ex2 / qwen3_forward_ex2` (in core/forward/forward.c, sibling lib). NOT to engine's gemma3_forward.
- `lib/shannon-prime-system/core/session/sp_session.c:243-318` — `kv_step` (qwen3 decode) calls `sp_matmul` ~7× per layer (Q/K/V/O/gate/up/down) + final head matmul.
- `lib/shannon-prime-system/core/session/sp_session.c:329-413` — `kv_step_gemma3` same pattern, gemma3 sandwich norms + GeGLU.
- `lib/shannon-prime-system/core/session/sp_session.c:416-513` — `kv_step_qwen25` same pattern, qwen2.5 biases + SwiGLU.

### 5. sp_daemon's session usage

- `tools/sp_daemon/src/session.rs:101-124` — `SpSession::create` calls `sp_session_create` (L1 ABI).
- `tools/sp_daemon/src/session.rs:155-188` — `prefill_chunk` / `decode_step` wrap `sp_prefill_chunk` / `sp_decode_step` (the math-core REFERENCE forward path).
- `tools/sp_daemon/src/dialogue_runner.rs:113-115,132-134,151-152,166-167,185-187,202-204` — every dialogue call goes through `prefill_chunk` / `decode_step` (math-core reference).
- `tools/sp_daemon/build.rs:7-25` — links **17 math-core libsp_*.a only**; does NOT link sp_engine; does NOT link sp_engine's hexagon backend.

### 6. NTT.5b precedent (the ONE existing daemon→DSP wiring)

- `lib/shannon-prime-system/include/sp/sp_l1.h:158-222` — L1 ABI extension: `sp_session_register_compute_backend` registers a (handle, forward_fn, inverse_fn) triple for NTT dispatch — opt-in, fallback to host when NULL.
- `tools/sp_daemon/src/ntt_hex_dispatch.rs:71-227` — Rust trampoline: leaks a `Box<ComputeBackend>` raw pointer; backend.session is an `Arc<FastRpcSession>` per `reference-fastrpc-concurrent-dispatch`.
- `tools/sp_daemon/src/daemon.rs:276-338` — env-gated registration (`SP_ENGINE_NTT_ATTN_HEX=1`); opens `libsp_compute_skel.so`, wraps in ComputeBackend, registers with Memory session via `sp_session_register_compute_backend`.

**Pattern to follow for WIRE-HEX:** opt-in L1 ABI hook + Rust trampoline + env-gated daemon registration + AppState owns the lifetime.

### 7. L3.FG closure — what got wired, what didn't

- `D:\F\shannon-prime-repos\shannon-prime-lattice\papers\SESSION-CLOSED-lat-2-l3-fg-cross-compile.md:9-23` — daemon's android build runs `/v1/chat` and `/v1/pouw/ledger` on-device; 12-token greedy continuation bit-stable host↔android.
- Line 28-32 — cross-compiled the **17 math-core modules only** (forward, forward_dispatch, forward_kernels, model, arena, frobenius, poly_ring, ntt_crt, vht2, weight_dtype, gguf, io_format, io_hash, session, sieve, kste, ok_arith).
- **Did NOT wire Hexagon backend** — explicitly cross-compiled the math-core reference forward to run on Cortex-X2 only. The Hexagon backend (sp_engine + sp_hex_host) was never linked into sp-daemon.

### 8. FastRPC concurrent-dispatch + Mode-D bridge alignment

- `src/backends/hexagon/sp_hex_host.c:73-77` — Mode D / Unsigned PD: `remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, ...)` before `sp_hex_open`. Matches `reference-mode-d-bridge-architecture` and `reference-qnn-htp-unsigned-pd-access-on-consumer-snapdragon-8-gen-1`.
- The Hexagon backend uses its OWN FastRPC URI + skel (`sp_hex_URI CDSP_DOMAIN`) — separate from the daemon's existing echo skel + compute skel. Per `reference-fastrpc-concurrent-dispatch` an `Arc<FastRpcSession>` is the substrate for any future cross-dispatch concurrency.

---

## Architectural finding (the meta-finding this sprint produces)

There are TWO independent forward implementations co-existing in the engine repo:

1. **math-core forward** (`lib/shannon-prime-system/core/forward/forward.c`, exposed via `sp_session` ABI in `sp_l1.h`). Used by: sp-daemon. f32 scalar reference per its own header comment.
2. **engine forward** (`src/forward/`, `src/backends/{cpu,cuda,vulkan,hexagon}/`). Used by: sp_engine.exe + `ppl.c`. Has all four backends (CPU AVX, CUDA, Vulkan, Hexagon V69 HVX).

The daemon `prefill_chunk` / `decode_step` route through path 1. The Hexagon backend lives entirely in path 2. **They share zero compiled code on the daemon's android build today.** This is the 6-month gap.

L3.FG cross-compiled path 1 for android (success). Path 2's hexagon backend is built by `scripts/build/build-hexagon.bat` into a SEPARATE android binary (`test_ppl` + `test_hex_fwd`) that the operator runs via adb. The daemon never touches it.

## Chosen wiring shape: **Shape B (full-forward backend hook at L1 ABI)**

**Justification given sp_hex_host's actual API:**

- `gemma3_forward_hexagon` is a WHOLE-FORWARD entry point with signature `(const qwen3_model *, const int32_t *tokens, int n_tok, float *logits)`. Per-matmul (Shape A) would require breaking up sp_hex_host's "build weight blob once + 1 FastRPC per chunk" amortization — devastating to perf and not what the backend exposes.
- The math-core `sp_session` reconstructs `qwen3_model *qm` via `sp_model_to_gemma3` (`sp_session.c:78-87`). The pointer is exactly what `gemma3_forward_hexagon` wants.
- Direct Rust → C dispatch (Shape C) requires linking the entire engine + hexagon backend into sp-daemon's build graph; ~3× the LOC of the Shape B hook approach with no architectural payoff.

### Concrete wiring (the changes this sprint ships)

1. **L1 ABI extension** in `lib/shannon-prime-system/include/sp/sp_l1.h`:
   - New typedef `sp_forward_dispatch_fn`:  
     `typedef int (*sp_forward_dispatch_fn)(void *handle, const struct qwen3_model_opaque *qm, const int32_t *tokens, int n_tok, float *logits);`
     - `qwen3_model_opaque` is a forward-declared struct — math-core re-emits it through; daemon side casts back to `qwen3_model*`.
   - New registration: `sp_status sp_session_register_forward_backend(sp_session *s, void *handle, sp_forward_dispatch_fn fn);`
   - Readback: `sp_forward_dispatch_fn sp_session_forward_backend(const sp_session *s);` + `void *sp_session_forward_backend_handle(const sp_session *s);`
   - Plus accessor `const void *sp_session_qwen3_model(const sp_session *s);` so the dispatcher can re-emit the existing `s->qm` pointer to the registered fn (no separate model handle owned by the daemon).

2. **sp_session.c** modifications:
   - Add fields `compute_forward_handle`, `compute_forward_fn` to struct sp_session.
   - In `sp_prefill_chunk`: if `compute_forward_fn != NULL`, call IT (with the session's `qm` + the full accumulated history). Use returned logits' last position; skip the in-tree `gemma3_forward_ex2` call.
   - In `sp_decode_step`: NOT modified — `gemma3_forward_hexagon` does NOT have a persistent KV path. Decode stays on the reference path. (Honest gap; will surface in the toks/s headline.)
   - Add `sp_session_register_forward_backend` + readback fn + `sp_session_qwen3_model`.

3. **build.rs** (tools/sp_daemon/build.rs) for android target:
   - Link the engine's hexagon backend artifacts (`sp_hex_host.o`, qaic-generated `sp_hex_stub.o`, the FastRPC libs `libcdsprpc.so` + `rpcmem.a`). This requires the engine's `scripts/build/build-hexagon.bat` to have produced these object files OR a new mini-CMake target that builds JUST sp_hex_host into a freestanding `libsp_hex_backend.a` for daemon consumption.
   - Strategy: extend `src/backends/hexagon/CMakeLists.txt` with an `add_library(sp_hex_backend STATIC ...)` target alongside the existing "compile into sp_engine" branch. Daemon picks up `libsp_hex_backend.a` from a known build-android-libs/hexagon/ path.
   - Add a thin C glue `tools/sp_daemon/c_glue/sp_daemon_hex_forward.c` that exposes a stable C symbol the Rust side `extern "C"`s — e.g. `int sp_daemon_hex_forward_dispatch(void *handle, const void *qm_opaque, const int32_t *tokens, int n_tok, float *logits)` that casts and calls `gemma3_forward_hexagon`.

4. **Rust trampoline** at `tools/sp_daemon/src/hex_forward_dispatch.rs` (android-only):
   - Same shape as `ntt_hex_dispatch.rs`: counter, `extern "C" fn` dispatch entry, opaque handle (Box-leaked).
   - The handle holds... nothing or minimal context (model handle is on the session side via `sp_session_qwen3_model`). Could just be `ptr::null_mut()` for the handle.

5. **AppState wiring** (daemon.rs, android-only):
   - Env-gated by `SP_DAEMON_BACKEND=hex` (the chosen env-var name).
   - Registers `gemma3_forward_hexagon` (via the C glue trampoline) on the **target session** (main `state.session`, not Memory) — because the target is the one most chat traffic hits.
   - Falls back to host (reference) when env unset or registration fails.

### Env-gate name: `SP_DAEMON_BACKEND=hex`

(`SP_ENGINE_BACKEND` and `SP_BACKEND` are the engine-side names per `ppl.c:30,37,44` — choosing a distinct `SP_DAEMON_BACKEND` namespace avoids cross-confusion.)

### Counter env-var for instrumentation

`SP_DAEMON_HEX_FORWARD_COUNT` is the atomic counter Rust side; readable via a new test endpoint or smoke harness.

---

## Gates methodology

- **T_WIRE_HEX_BACKEND_LINKED**: `aarch64-linux-android-objdump -t target/aarch64-linux-android/release/sp-daemon | grep gemma3_forward_hexagon` → expect non-empty.
- **T_WIRE_HEX_BACKEND_DISPATCHES**: daemon binary started with `SP_DAEMON_BACKEND=hex` on S22U; trigger one /v1/chat prefill of a Gemma3 prompt; query a new debug endpoint `/v1/debug/backend_counts` (or read counter via stderr log); expect counter > 0.
- **T_WIRE_HEX_BACKEND_BIT_EXACT_VS_REFERENCE**: on S22U, two runs of same prompt + same seed (greedy, deterministic): one with `SP_DAEMON_BACKEND=hex`, one without. Compare argmax-decoded 8-token continuation byte-exact. **Per `reference-lattice-decode-determinism`**, byte-equal expected if precondition (greedy, fixed K=1, same backend math) holds. If diverges: do NOT widen tolerance — document the divergence numbers and STOP, surface upstream.
- **T_WIRE_HEX_TOKS_MEASURED**: re-run NTT-bench (or simple prefill+decode harness with 64-token prompt + 32-token decode on Gemma3-1B Memory model proxy — IF available; ELSE use whatever model the daemon currently loads) with hex backend ON vs OFF. Report tok/s honestly. Note: decode path will be UNCHANGED (decode_step is reference path); only prefill will use hex.

---

## Risk: this sprint may surface a structural blocker

The Hexagon backend was designed for `ppl.c`'s "one full-corpus forward" use case, not the daemon's "small prefill + many decode steps" use case. Specifically:

- **No persistent KV.** `gemma3_forward_hexagon` re-runs the full forward over the accumulated history each call. The daemon's per-token decode would re-run the full forward over [hist + 1 token]. This is the same shape as the reference `sp_prefill_chunk` (which also re-runs full history), so for chat-style turn-by-turn it's not insane — but per-token decode would be catastrophically slow.
- **Per-chunk weight-blob already uploaded ONCE** (`sp_hex_host.c:118 if (g_hx.key != m) hx_build(m)`). Good — won't re-pay 574 MB upload per call.
- **574 MB Q8 arena required.** Daemon needs to be loading a Gemma3-1B model with `SP_ARENA=q8`. If the operator currently loads with a different arena setting, hex backend can't activate.

If the model loaded by the daemon ISN'T gemma3 + Q8 arena, the gate FAILS structurally and we surface UPSTREAM. The decode-path mismatch IS a real architectural gap; we document it in the closure as "Shape B's natural fit is prefill-only; decode-step bypass is a follow-up sprint."

---

## Workflow

1. **Plan-commit** (this doc) → `[plan] WIRE-HEX — sp_daemon → sp_hex_host backend dispatch wiring`
2. **Stage 1** — math-core L1 ABI extension + sp_session.c hook; host-only build check.
3. **Stage 2** — engine `libsp_hex_backend.a` CMake target; daemon build.rs linking it; Rust trampoline + AppState wiring.
4. **Stage 3** — cross-compile sp-daemon for android; push to S22U; T_WIRE_HEX_BACKEND_LINKED + T_WIRE_HEX_BACKEND_DISPATCHES + T_WIRE_HEX_BACKEND_BIT_EXACT_VS_REFERENCE on-device.
5. **Stage 4** — T_WIRE_HEX_TOKS_MEASURED on-device.
6. **Stage 5** — closure (CLOSURE-WIRE-HEX.md).

If Stage 1 or 2 reveals sp_hex_host requires modifications to expose a daemon-callable lib (e.g. sp_arena/embed_row/matmul host-side helpers also need to be linked, which would pull in the entire engine), STOP and surface UPSTREAM. No silent gate revisions; no quiet bundling.
