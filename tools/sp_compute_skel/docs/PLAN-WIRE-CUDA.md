# PLAN — WIRE-CUDA (sp_daemon -> CUDA PTX forward backend dispatch)

**Sprint:** Phase 2-CU.DAEMON-WIRE (WIRE-CUDA)
**Date:** 2026-06-01
**Worktree:** `D:\F\shannon-prime-repos\engine-wire-cuda`
**Branch:** `sprint/wire-cuda` (base: engine main @ `73f3367`)
**Sub-tag candidate:** `lat-phase-2-wire-cuda-shipped` (or `-deferred-no-hardware`)
**Host hardware:** NVIDIA RTX 2060 (sm_75, 12 GB) — CUDA tok/s measurement IS feasible
**Template:** `CLOSURE-WIRE-HEX-FINISH.md` + `CLOSURE-WIRE-HEX.md` — symmetric daemon-wiring sprint

## Goal

Mirror the WIRE-HEX-FINISH wiring for the CUDA PTX backend. `SP_DAEMON_BACKEND=cuda`
becomes a runtime selector that routes the daemon's `sp_prefill_chunk` through
the engine's `gemma3_forward_cuda` / `qwen3_forward_cuda` instead of math-core's
reference forward. NO kernel changes — the CUDA backend is already built per
Phase 2-CU.PTX-* closures.

## Stage 0 — pre-read complete (file:line citations)

1. **WIRE-HEX-FINISH closure** — `tools/sp_compute_skel/docs/CLOSURE-WIRE-HEX-FINISH.md`:
   the canonical template (gates table, headline format, per-stage discipline).

2. **WIRE-HEX closure** — `tools/sp_compute_skel/docs/CLOSURE-WIRE-HEX.md`:
   architectural rationale for Shape B (L1 §6 full-forward hook) + lib pattern.

3. **Hex trampoline pattern** — `tools/sp_daemon/src/hex_forward_dispatch.rs:1-150`:
   `#![cfg(target_os = "android")]` + atomic `WIRE_HEX_DISPATCH_COUNT` +
   `sp_wire_hex_forward_dispatch` trampoline + `register_with_session` helper.
   For CUDA we drop `#![cfg(target_os = "android")]` (CUDA is host-only) and
   gate via `#[cfg(feature = "wire_cuda_backend")]` instead.

4. **Hex glue C** — `tools/sp_daemon/c_backend/sp_daemon_hex_glue.c:1-110`:
   the L1 dispatcher signature + arch-conditional call into the engine.

5. **Hex CMakeLists** — `tools/sp_daemon/c_backend/CMakeLists.txt:1-108`:
   builds `libsp_hex_daemon_backend.a` from `sp_hex_host.c` + `sp_hex_stub.c`
   + glue. CUDA path replaces these sources with `cuda_forward.cu` +
   `cuda_backend.cu` + glue. No qaic codegen — CUDA isn't FastRPC.

6. **Daemon registration** — `tools/sp_daemon/src/daemon.rs:340-388`:
   `#[cfg(all(target_os = "android", feature = "wire_hex_backend"))]` block.
   For CUDA we use `#[cfg(feature = "wire_cuda_backend")]` (no android constraint).

7. **Cargo features** — `tools/sp_daemon/Cargo.toml:108-115`:
   `wire_hex_backend = []` (default off). Add symmetric `wire_cuda_backend = []`.

8. **build.rs wire-hex block** — `tools/sp_daemon/build.rs:117-150`:
   `CARGO_FEATURE_WIRE_HEX_BACKEND` + `SP_HEX_BACKEND_DIR` + cdsprpc/rpcmem
   linkage. CUDA equivalent: `CARGO_FEATURE_WIRE_CUDA_BACKEND` +
   `SP_CUDA_BACKEND_DIR` + `CUDA::cudart` + `CUDA::cublas` linkage.

9. **CUDA backend entry symbols** — `src/backends/cuda/cuda_forward.cu`:
   - `extern "C" int gemma3_forward_cuda(const qwen3_model *, const int32_t *, int, float *)` @ line 497
   - `extern "C" int qwen3_forward_cuda(const qwen3_model *, const int32_t *, int, float *)` @ line 724
   - `extern "C" int qwen3_forward_cuda_ex(..., sp_kste_tree_t *)` @ line 596 (KV-tree variant, not used by daemon)
   - `extern "C" void sp_cuda_model_release(const qwen3_model *)` @ line 729

10. **Engine ppl dispatch shape** — `src/forward/ppl.c:26-32`:
    Gemma3 -> `gemma3_forward_cuda`. Note: ppl only routes Gemma3 to CUDA
    today. The CUDA backend DOES export `qwen3_forward_cuda` (E_CU_5 per CU.5
    closure). For the daemon glue we will dispatch on `m->cfg.arch`:
    - `SP_ARCH_GEMMA3` -> `gemma3_forward_cuda`
    - `SP_ARCH_QWEN3`  -> `qwen3_forward_cuda` (KV-tree NULL)
    - else: return -1 with `sp_set_error("cuda: unsupported arch")`.

11. **CUDA backend public header** — `include/sp_engine/cuda_backend.h`:
    canonical extern C declarations + `sp_cuda_model_release` lifecycle hook.

12. **L1 ABI hook** — `lib/shannon-prime-system/include/sp/sp_l1.h:225-296`:
    `sp_forward_dispatch_fn` typedef + `sp_session_register_forward_backend`.
    Already shipped by WIRE-HEX; reused verbatim.

13. **CMake / build scripts** — `src/CMakeLists.txt:99-119`:
    `SP_ENGINE_WITH_CUDA` -> `sp_engine_cuda` STATIC lib from
    `cuda_backend.cu` + `cuda_forward.cu` + cudart/cublas link.
    `scripts/build/build-cuda.bat` + `scripts/env/env-cuda.bat` (CUDA 13.2 +
    VS2019 BT + `SP_CUDA_ARCH=75` for RTX 2060).

14. **ctest baseline** — `shannon-prime-system-engine/ctest-cuda-validate.log`:
    31/33 passing on this host. The 2 failures:
    - **#29 M_GEMMA3_CUDA** — `[FAIL] gemma3_forward_cuda` on f32 GGUF
      weights: `upload_weight: host OOM`. Q8 + Q4 Frobenius variants PASS
      (argmax=15/15, KL<1e-10). The failure is the f32-direct-GGUF path on
      a 1B model in 12 GB device memory; OOM during upload. Q8/Q4 fine.
    - **#33 T_FRO_4_CU** — f32 perplexity rc != 0 + Q8 PPL=0 with n_scored=0.
      Same model (Gemma3-1B), same backend, same OOM root cause.

    **OUT OF SCOPE.** Filed as `WIRE-CUDA-BUGFIX-Q8-DRIFT` follow-on. WIRE-CUDA
    Stage 4 (bit-exact gate) uses **Qwen3-0.6B** which ctest passes cleanly
    (#30 M_QWEN3_CUDA Passed 38.14s; #31 E_CU_5 Passed 3.92s).

15. **README §5.2** — `README.md:304-336`:
    "Status: built + bit-exact-validated. NOT wired into sp_daemon — symmetric
    WIRE-HEX-style sprint pending." This sprint fills that gap.

16. **`feedback-no-silent-gate-revisions`** + **`feedback-bundled-changeset-root-cause-ambiguity`**:
    one-variable-at-a-time stage discipline; surface upstream rather than tune
    fixtures.

## Wiring shape: **Shape B** (L1 ABI §6 hook, same as WIRE-HEX)

Static lib (`libsp_cuda_daemon_backend.lib`) is the boundary because:

- `cuda_forward.cu` + `cuda_backend.cu` are nvcc-compiled CUDA TUs; the daemon
  is a `cc + Rust + clang-link` target. A static lib lets cargo's rustc link
  the precompiled .obj files without nvcc on the daemon's link line.
- Symmetric to hex (whose `sp_hex_host.c` is a separate aarch64-android TU).
- Same L1 §6 contract: `sp_forward_dispatch_fn(handle, qm_opaque, tokens, n_tok, logits)`.

## Scope (what ships)

1. **Static lib `libsp_cuda_daemon_backend.lib`** (Windows .lib for MSVC; .a on
   Linux) built from:
   - `src/backends/cuda/cuda_backend.cu` (device mgmt + error mapping)
   - `src/backends/cuda/cuda_forward.cu` (whole-forward Gemma3 + Qwen3 entries)
   - `tools/sp_daemon/c_backend/sp_daemon_cuda_glue.c` (L1 §6 dispatcher +
     arch-router + lifecycle hook). NEW FILE — symmetric to
     `sp_daemon_hex_glue.c`. NO kernel name shim needed (CUDA backend doesn't
     re-export `matmul`/`embed_row`/`as_f32` — its `cuda_forward.cu` calls into
     math-core's `sp_*` via the same forward_dispatch.h that the daemon
     already links).
   - Build script `tools/sp_daemon/build-host-cuda-backend.bat`.

2. **CMakeLists at `tools/sp_daemon/c_backend/CMakeLists-cuda.txt`** — NEW.
   Mirrors the hex one but: (a) `enable_language(CUDA)` + `find_package(CUDAToolkit)`,
   (b) no qaic codegen, (c) MSVC `/MD` + nvcc `--use-local-env` per build-cuda.bat,
   (d) `CMAKE_CUDA_ARCHITECTURES=75` (RTX 2060). Separate file from `CMakeLists.txt`
   (the hex one) so the two builds don't cross-pollute.

3. **Cargo feature `wire_cuda_backend`** in `tools/sp_daemon/Cargo.toml`. Default off.

4. **Trampoline `tools/sp_daemon/src/cuda_forward_dispatch.rs`** — template-copy
   of `hex_forward_dispatch.rs` with hex->cuda rename, drop `#![cfg(target_os = "android")]`
   (CUDA host-only), atomic `WIRE_CUDA_DISPATCH_COUNT`.

5. **build.rs wire-cuda block** at end of `build.rs` mirroring wire-hex.

6. **Daemon registration** in `daemon.rs` mirroring WIRE-HEX block. Activates
   when `SP_DAEMON_BACKEND=cuda` AND `wire_cuda_backend` feature compiled.

7. **AppState** gains `wire_cuda_active: bool`. `/v1/debug/backend_counts`
   gains `cuda_forward_count` + `wire_cuda_active`.

8. **Module export** in `lib.rs`.

9. **Closure** at `tools/sp_compute_skel/docs/CLOSURE-WIRE-CUDA.md`.

## Gates

| Gate | Pass criterion |
|------|----------------|
| **T_WIRE_CUDA_STATIC_LIB_BUILT** | `libsp_cuda_daemon_backend.lib` produced. dumpbin /symbols shows `gemma3_forward_cuda`, `qwen3_forward_cuda`, `sp_daemon_cuda_forward`. |
| **T_WIRE_CUDA_DAEMON_LINKED** | `cargo build --features wire_cuda_backend --release` succeeds. `dumpbin /exports` (or `nm`) shows the 3 symbols at concrete addresses. |
| **T_WIRE_CUDA_RUNTIME_ACTIVE** | `/v1/debug/backend_counts` returns `wire_cuda_active: true` after startup with `SP_DAEMON_BACKEND=cuda`. After one prefill, `cuda_forward_count > 0`. |
| **T_WIRE_CUDA_BIT_EXACT_VS_REF** | 32-token greedy-argmax matches math-core reference on Qwen3-0.6B. (Gemma3-1B excluded due to known M_GEMMA3_CUDA OOM bug.) |
| **T_WIRE_CUDA_TOKS_MEASURED** | 3-rep tok/s on host with RTX 2060. |

## Stage discipline

- Stage 1: `c_backend/sp_daemon_cuda_glue.c` + `c_backend/CMakeLists-cuda.txt`
  + `build-host-cuda-backend.bat`. Build the static lib. Gate
  T_WIRE_CUDA_STATIC_LIB_BUILT.
- Stage 2: Cargo feature + `src/cuda_forward_dispatch.rs` + `lib.rs` export
  + `build.rs` wire-cuda block + `daemon.rs` registration + state.rs
  `wire_cuda_active` + routes.rs `cuda_forward_count`. Build the daemon binary.
  Gate T_WIRE_CUDA_DAEMON_LINKED.
- Stage 3: Launch with `SP_DAEMON_BACKEND=cuda`, hit `/v1/debug/backend_counts`,
  run one prefill, hit again. Gate T_WIRE_CUDA_RUNTIME_ACTIVE.
- Stage 4: Bit-exact via two daemon launches (cuda vs reference) on Qwen3-0.6B,
  diff 32-token output. Gate T_WIRE_CUDA_BIT_EXACT_VS_REF.
- Stage 5: 3-rep tok/s + closure. Gate T_WIRE_CUDA_TOKS_MEASURED.

## NO kernel changes

If I find myself editing `src/backends/cuda/*.cu` or `*.cuh`, STOP. Surface UPSTREAM
as `WIRE-CUDA-BUGFIX`.

## Anti-contamination

`engine-wire-cuda` only. No edits in `engine-wire-finish`, `engine-wire-cpu`,
`engine-wire-vulkan`, or `shannon-prime-system-engine`.
