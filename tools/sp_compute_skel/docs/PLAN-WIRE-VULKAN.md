# PLAN — WIRE-VULKAN (sp_daemon -> vulkan_forward dispatch)

**Sprint:** Phase 2-VK.DAEMON-WIRE (WIRE-VULKAN)
**Date:** 2026-06-01
**Worktree:** `D:\F\shannon-prime-repos\engine-wire-vulkan`
**Branch:** `sprint/wire-vulkan` (base engine main @ 73f3367)
**Sub-tag candidate:** `lat-phase-2-wire-vulkan-{shipped,attempted,blocked-on-oom}` (resolved at closure)

---

## Stage 0 — Mandatory pre-read citations

The Vulkan backend is ALREADY BUILT (Phase 2-L1-PARITY closure). This sprint
threads the daemon-side wiring that was already structurally laid down for
Hexagon by WIRE-HEX + WIRE-HEX-FINISH. Symmetric template-copy with
`hex -> vulkan` rename. No kernel changes.

### A. Templates and prior art (must cite before drafting)

1. **CLOSURE-WIRE-HEX-FINISH** — `tools/sp_compute_skel/docs/CLOSURE-WIRE-HEX-FINISH.md`
   (the headline-numbers + reproduction closure shape).
2. **CLOSURE-WIRE-HEX** — `tools/sp_compute_skel/docs/CLOSURE-WIRE-HEX.md`
   (the original wiring closure, identifies all 4 gates).
3. **Rust trampoline template** — `tools/sp_daemon/src/hex_forward_dispatch.rs:36-164`
   (atomic dispatch_count + extern symbol decls + register_with_session).
4. **Daemon registration block** — `tools/sp_daemon/src/daemon.rs:340-392`
   (env-gated registration on the TARGET session; logs outcome; sets
   AppState.wire_hex_active).
5. **Cargo features** — `tools/sp_daemon/Cargo.toml:117-126`
   (`[features] default = [] wire_hex_backend = []`).
6. **Lib export** — `tools/sp_daemon/src/lib.rs:27-34`
   (`#[cfg(all(target_os = "android", feature = "wire_hex_backend"))]
   pub mod hex_forward_dispatch;`).
7. **AppState field** — `tools/sp_daemon/src/state.rs:64-71` (`wire_hex_active: bool`).
8. **Backend-counts endpoint** — `tools/sp_daemon/src/routes.rs:64-110`
   (existing `BackendCounts` struct + `v1_debug_backend_counts` handler).
9. **C glue template** — `tools/sp_daemon/c_backend/sp_daemon_hex_glue.c`
   (cast `qm_opaque` -> `qwen3_model *`; call backend entry; release hook).
10. **CMakeLists template** — `tools/sp_daemon/c_backend/CMakeLists.txt`
    (standalone static lib; no SDK-specific FastRPC stub for Vulkan).
11. **Build script template** — `tools/sp_daemon/build-android-hex-backend.bat`.
12. **build.rs link block** — `tools/sp_daemon/build.rs:132-182`
    (feature-gated link + system lib deps).

### B. Vulkan backend (DO NOT MODIFY — kernel-frozen)

13. **Entry symbols**:
    - `src/backends/vulkan/vulkan_forward.cpp:658` —
      `extern "C" int gemma3_forward_vulkan(const qwen3_model *m, const int32_t *tokens, int n_tokens, float *logits)`
    - `src/backends/vulkan/vulkan_forward.cpp:933` —
      `extern "C" int qwen3_forward_vulkan(const qwen3_model *m, const int32_t *tokens, int n_tokens, float *logits)`
      (wraps `qwen3_forward_vulkan_ex` with NULL kv_trees at `:783`).
14. **Release hook** — `include/sp_engine/vulkan_backend.h:54-55`
    `void sp_vulkan_model_release(const qwen3_model *m)` (no-op friendly).
15. **C surface header** — `include/sp_engine/vulkan_backend.h` (the full
    `extern "C"` declaration set the glue includes).
16. **Backend lifecycle** — `src/backends/vulkan/vulkan_backend.cpp:42-125`
    (`vk_ensure_instance` / `vk_ensure_device` lazy init via singleton
    `g_ctx`; the backend manages its own instance/device/queue lifetime
    keyed off the model pointer in `vulkan_forward.cpp`).
17. **Existing static lib** — `src/CMakeLists.txt:160-181` (`sp_engine_vulkan`
    STATIC: builds `vulkan_backend.cpp` + `vulkan_forward.cpp` + 12 SPIR-V
    compute shaders embedded as `.spv.h` headers via glslc). The daemon's
    standalone lib will REUSE this same source set; build target is
    `sp_vulkan_daemon_backend.{a,lib}` so it doesn't collide with the
    full-engine `sp_engine_vulkan` archive.
18. **Compute shaders (frozen)** — `src/backends/vulkan/shaders/*.comp` (12
    shaders: add, attn, attn_ntt, dequant_arena, embed_scale, gelu_mul, gemm,
    rmsnorm, rmsnorm_head, rope, round_f16, silu_mul). Each is build-time
    compiled to `*.spv.h` by glslc and `#include`d by vulkan_forward.cpp.

### C. L1 ABI surface (already shipped by WIRE-HEX)

19. **§6 forward backend hook** — `lib/shannon-prime-system/include/sp/sp_l1.h:224-309`
    (math-core submodule, pinned to `0b3b86b` per WIRE-HEX-FINISH merge into
    main). All four pieces present:
    - `typedef int (*sp_forward_dispatch_fn)(void*, const void*, const int32_t*, int, float*);`
    - `sp_status sp_session_register_forward_backend(sp_session*, void*, sp_forward_dispatch_fn);`
    - `void *sp_session_forward_backend_handle(const sp_session*);`
    - `sp_forward_dispatch_fn sp_session_forward_backend_fn(const sp_session*);`

    No math-core changes required for this sprint; the hook is general per its
    own comment ("targets the engine's per-backend full-forward entry points
    (gemma3_forward_hexagon, gemma3_forward_cuda, gemma3_forward_vulkan)" —
    sp_l1.h:230).

### D. Test failure context (out of scope)

20. **ctest log** — `ctest-vulkan-validate.log` (engine root). 27/31 PASS;
    the 4 FAIL are all `VkAllocateMemory: VkResult -2` (out of device memory):
    - **#28 M_GEMMA3_VULKAN** — `tests/test_gemma3_vulkan.c:83`,
      `Vulkan: vkAllocateMemory: VkResult -2`
    - **#29 M_QWEN3_VULKAN** — `tests/test_qwen3_vulkan.c:116`,
      `upload_weight: host OOM`
    - **#30 E_VK_5** (NTT-attention) — `tests/test_ntt_attn_vulkan.c:112`,
      `Vulkan: vkAllocateMemory: VkResult -2`
    - **#31 E_VK_6** (KSTE KV) — `tests/test_kste_kv_vulkan.c:91`,
      `Vulkan: vkAllocateMemory: VkResult -2`

    All four predate this sprint. **DO NOT TRY TO FIX.** File as separate
    `WIRE-VULKAN-OOM-BUGFIX` follow-on. The wiring is what's being validated;
    OOM is a separate device/driver/allocation-budget issue.

### E. Workflow memories — load-bearing

21. **`feedback-no-silent-gate-revisions`** — if a gate can't be met, surface
    UPSTREAM (here: BLOCKED-on-prior-OOM-bug). Do NOT massage the OOM bug to
    ship a number.
22. **`feedback-bundled-changeset-root-cause-ambiguity`** — change one
    variable at a time across stages. Static lib (Stage 1), Rust trampoline +
    daemon reg (Stage 2), runtime test (Stage 3), bit-exact + tok/s (Stages
    4-5). One commit per stage.

---

## Stage 1 — Static lib build

**Files added:**
- `tools/sp_daemon/c_backend/sp_daemon_vulkan_glue.c` — L1 forward
  dispatcher; arch-routes `m->cfg.arch` -> `gemma3_forward_vulkan` /
  `qwen3_forward_vulkan` (Vulkan backend supports both). Also exposes
  `sp_daemon_vulkan_release` -> `sp_vulkan_model_release`.
- `tools/sp_daemon/build-host-vulkan-backend.bat` — analog of
  `build-android-hex-backend.bat`; builds `sp_vulkan_daemon_backend.lib`
  for the host VS2019 + Vulkan SDK toolchain into
  `build-host-vulkan-backend/`.

**Files modified:**
- `tools/sp_daemon/c_backend/CMakeLists.txt` — add an
  `if(SP_DAEMON_BUILD_VULKAN_BACKEND)` branch that defines a separate
  `sp_vulkan_daemon_backend` STATIC target. The hex target stays unchanged.
  The Vulkan target compiles `vulkan_backend.cpp` + `vulkan_forward.cpp` +
  the 12 SPIR-V compute-shader `.spv.h` headers (built via glslc at the same
  add_custom_command pattern as `src/CMakeLists.txt:142-158`), plus the
  glue.c, into one static archive. The daemon binary link step pulls
  `Vulkan::Vulkan` (loader lib) at sp-daemon link time.

**Gate: T_WIRE_VULKAN_STATIC_LIB_BUILT**
- PASS: `build-host-vulkan-backend/sp_vulkan_daemon_backend.{a,lib}`
  exists; readable by `nm`/`lib /list`; the 4 entry symbols resolve
  inside the archive: `gemma3_forward_vulkan`, `qwen3_forward_vulkan`,
  `sp_vulkan_model_release`, `sp_daemon_vulkan_forward`.

---

## Stage 2 — Cargo feature + Rust trampoline + daemon registration

**Files added:**
- `tools/sp_daemon/src/vulkan_forward_dispatch.rs` — template-copy of
  `hex_forward_dispatch.rs` with `hex -> vulkan` rename. Drops the
  `#![cfg(target_os = "android")]` gate (Vulkan is host-side: Windows /
  Linux / macOS), keeping just the feature gate. Atomic `dispatch_count`,
  `register_with_session`, `release_for_model`.

**Files modified:**
- `tools/sp_daemon/Cargo.toml` — add `wire_vulkan_backend = []` feature.
- `tools/sp_daemon/src/lib.rs` — add
  `#[cfg(feature = "wire_vulkan_backend")] pub mod vulkan_forward_dispatch;`.
- `tools/sp_daemon/src/state.rs` — add `pub wire_vulkan_active: bool` field
  (analog of `wire_hex_active`).
- `tools/sp_daemon/src/daemon.rs` — add a Vulkan registration block
  symmetric to the WIRE-HEX block at lines 340-392. Env-gated by
  `SP_DAEMON_BACKEND=vulkan`. Logs outcome. Sets `wire_vulkan_active`.
- `tools/sp_daemon/build.rs` — feature-gated link block analogous to
  the `wire_hex_backend` block at lines 132-182. Resolves to
  `build-host-vulkan-backend/` by default; overridable via
  `SP_VULKAN_BACKEND_DIR`. Links `Vulkan::Vulkan` (the loader, e.g.
  `vulkan-1.lib` on Windows, `libvulkan.so` on Linux).
- `tools/sp_daemon/src/routes.rs` — extend `BackendCounts` struct with
  `vulkan_forward_count: u64` + `wire_vulkan_active: bool`; populate from
  `vulkan_forward_dispatch::dispatch_count()` + `state.wire_vulkan_active`.

**Gate: T_WIRE_VULKAN_DAEMON_LINKED**
- PASS: `cargo build --features wire_vulkan_backend --release` succeeds
  with VULKAN_SDK set on the host. `nm`/`dumpbin /symbols` on the produced
  `sp-daemon{.exe}` binary shows 3 symbols:
  - `gemma3_forward_vulkan` (T)
  - `sp_wire_vulkan_forward_dispatch` (T)
  - `sp_session_register_forward_backend` (T)

---

## Stage 3 — Runtime activation test

**Goal:** validate the wiring layers fire correctly.

- Run daemon with `SP_DAEMON_BACKEND=vulkan`.
- `curl -s http://127.0.0.1:8087/v1/debug/backend_counts` returns
  `{"wire_vulkan_active": true, "vulkan_forward_count": 0, ...}`.
- Send one `/v1/chat` request with a small prompt.

**Two possible outcomes:**

  - (a) Prefill succeeds -> `vulkan_forward_count` -> 1 -> PASS.
  - (b) Prefill hits the same OOM bug that fails M_GEMMA3_VULKAN +
    M_QWEN3_VULKAN ctests -> daemon logs the `vkAllocateMemory: VkResult
    -2` error via sp_last_error; counter still increments because the
    dispatcher trampoline bumps BEFORE the engine call (per
    `hex_forward_dispatch.rs:113`).

**Gate: T_WIRE_VULKAN_RUNTIME_ACTIVE**
- PASS if (a) — full runtime is up.
- PASS if (b) — counter increments (proves trampoline reached) AND the
  daemon-side log shows the engine-side OOM error. Mark as
  `BLOCKED-on-prior-OOM-bug` with the specific VkResult error logged.
- FAIL only if neither path runs (wiring broken).

---

## Stage 4 — Bit-exact verification (if Stage 3 hit path (a))

- Drive 16-token prefill + 32-step decode with reference daemon (no
  `SP_DAEMON_BACKEND` set), capture argmax sequence.
- Same prompt, same prefill, against `SP_DAEMON_BACKEND=vulkan`. Capture.
- Diff. PASS iff byte-identical.

If Stage 3 path (b): mark `T_WIRE_VULKAN_BIT_EXACT_VS_REF =
BLOCKED-on-prior-OOM-bug`. No silent gate revision.

---

## Stage 5 — Tok/s measurement + closure

If Stage 4 PASS: capture 3-rep tok/s. Headline table in closure (3 rows
per `feedback-drop-fp32-baseline-comparing-to-ourselves`: ARM ref / hex
backend / Vulkan backend if shipped, OR just ref + vulkan if hex isn't
available on the host).

If Stage 4 BLOCKED: closure documents the wiring win + the OOM blocker as
upstream, with sub-tag `lat-phase-2-wire-vulkan-blocked-on-oom`.

---

## Concrete file list (LOC estimate)

### Engine repo (engine-wire-vulkan @ branch sprint/wire-vulkan)

| File | Action | Est LOC |
|------|--------|---------|
| `tools/sp_daemon/c_backend/sp_daemon_vulkan_glue.c` | NEW | ~80 |
| `tools/sp_daemon/c_backend/CMakeLists.txt` | MODIFY | +85 |
| `tools/sp_daemon/build-host-vulkan-backend.bat` | NEW | ~60 |
| `tools/sp_daemon/Cargo.toml` | MODIFY | +10 |
| `tools/sp_daemon/src/lib.rs` | MODIFY | +8 |
| `tools/sp_daemon/src/state.rs` | MODIFY | +10 |
| `tools/sp_daemon/src/daemon.rs` | MODIFY | +45 |
| `tools/sp_daemon/src/routes.rs` | MODIFY | +18 |
| `tools/sp_daemon/src/vulkan_forward_dispatch.rs` | NEW | ~165 |
| `tools/sp_daemon/build.rs` | MODIFY | +55 |
| `tools/sp_compute_skel/docs/PLAN-WIRE-VULKAN.md` | NEW | this file |
| `tools/sp_compute_skel/docs/CLOSURE-WIRE-VULKAN.md` | NEW (Stage 5) | TBD |
| `start_wire_vulkan_daemon.sh` | NEW (host) | ~20 |

Math-core: NO changes (§6 hook already in main per WIRE-HEX-FINISH merge).

---

## Commits planned (one per stage)

1. `[plan] WIRE-VULKAN -- symmetric to WIRE-HEX-FINISH for Vulkan backend` — this file.
2. `[WIRE-VULKAN Stage 1] daemon-linkable Vulkan backend static lib + build script` — Stage 1.
3. `[WIRE-VULKAN Stage 2] Cargo feature + Rust trampoline + daemon registration + routes` — Stage 2.
4. `[WIRE-VULKAN Stage 3] runtime activation log + backend_counts` — Stage 3 evidence.
5. `[WIRE-VULKAN Stage 5] closure + sub-tag candidate` — Stage 4-5 results.

---

## What's explicitly OUT OF SCOPE

- **No kernel changes.** `vulkan_forward.cpp`, `vulkan_backend.cpp`, `vk_common.h`,
  and the 12 `*.comp` shaders are KERNEL-FROZEN per Phase 2-L1-PARITY. STOP if
  about to edit any.
- **No OOM fix.** M_GEMMA3_VULKAN / M_QWEN3_VULKAN / E_VK_5 / E_VK_6 all fail
  with `VkAllocateMemory: VkResult -2` predating this sprint. File as
  `WIRE-VULKAN-OOM-BUGFIX` follow-on. Wiring layers ship clean even if
  runtime is OOM-blocked.
- **No CUDA / no CPU.** Symmetric sprints in `engine-wire-cuda` and
  `engine-wire-cpu` worktrees per `feedback-parallel-agents-separate-worktrees`.
- **No silent gate revisions.** If Stage 3-5 are OOM-blocked, the closure
  honestly says `BLOCKED-on-prior-OOM-bug` and ships the wiring artifacts.

---

## UPSTREAM concerns (named in advance per memory discipline)

- **U-1 — vulkan_forward.cpp host-side allocator pressure.** The 4 ctest
  failures suggest something in the upload path peaks memory at the wrong
  moment. May be:
  - the per-tensor staging buffer pattern allocating before freeing,
  - a Vulkan host-visible heap budget that's small on this GPU,
  - or simply the model arena (~1 GB Q8 Gemma3-1B) plus mirrored device
    weights exceeding the device's VRAM (RTX 2060 has 6 GB).
  Diagnosis is the WIRE-VULKAN-OOM-BUGFIX sprint's job, NOT this one's.

- **U-2 — the GPU may be in use by another process.** Hyper-V / RDP /
  another daemon holding device memory could cause spurious OOM at ctest
  time. The wiring sprint cannot distinguish without dedicated profiling;
  closure documents the VkResult error string verbatim so the follow-on
  has the literal symptom.

---

## Pre-flight: nothing committed yet

```
cd D:\F\shannon-prime-repos\engine-wire-vulkan
git status
# expect: On branch sprint/wire-vulkan; clean working tree
git log --oneline -1
# expect: 73f3367 Merge sprint/v5-ffn-vtcm ...
```

Then the [plan] commit (this file) lands. Subsequent stages commit in order
per the workflow discipline.
