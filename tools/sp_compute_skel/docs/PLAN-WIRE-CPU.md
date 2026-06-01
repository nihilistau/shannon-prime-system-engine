# PLAN — WIRE-CPU (sp_daemon -> CPU AVX-512 backend dispatch)

**Sprint:** Phase 2-CPU.DAEMON-WIRE (WIRE-CPU)
**Date:** 2026-06-01
**Worktree:** `D:\F\shannon-prime-repos\engine-wire-cpu`
**Branch:** `sprint/wire-cpu` (base 73f3367 = engine main = V5 FFN VTCM ship)
**Sub-tag candidate:** `lat-phase-2-wire-cpu-shipped`
**Status:** plan-commit (Stage 0 reads complete; pre-flight gates identified)

This plan is the symmetric port of WIRE-HEX-FINISH (closed 2026-05-31, doc
`tools/sp_compute_skel/docs/CLOSURE-WIRE-HEX-FINISH.md`) onto the CPU AVX-512
backend (built and validated under Phase 2-CPU.AVX; README §2 row 3).

The 6-month gap of "the daemon never dispatches to any backend" is closed
on Hexagon; this sprint extends that same L1 ABI §6 hook to CPU AVX-512 so
`SP_DAEMON_BACKEND=cpu` becomes a runtime selector parallel to
`SP_DAEMON_BACKEND=hex`.

---

## Stage 0 citations

### Architectural template (WIRE-HEX / WIRE-HEX-FINISH)

| Reference | File:line | Confirms |
|---|---|---|
| WIRE-HEX-FINISH closure | `tools/sp_compute_skel/docs/CLOSURE-WIRE-HEX-FINISH.md:36-46` | Gates table shape (4 gates: LIB_BUILT / DAEMON_LINKED / RUNTIME_ACTIVE / BIT_EXACT) and tok/s methodology. |
| WIRE-HEX closure (architectural shape) | `tools/sp_compute_skel/docs/CLOSURE-WIRE-HEX.md:41-75` | Static lib + Cargo feature + Rust trampoline + L1 register hook architecture. |
| Hex Rust trampoline | `tools/sp_daemon/src/hex_forward_dispatch.rs:42-150` | Template for `cpu_forward_dispatch.rs`: `WIRE_*_DISPATCH_COUNT` atomic, `extern "C" fn sp_daemon_*_forward`, `sp_wire_*_forward_dispatch` no_mangle dispatcher, `register_with_session` Rust wrapper around `sp_session_register_forward_backend`. |
| Hex C glue | `tools/sp_daemon/c_backend/sp_daemon_hex_glue.c:33-74` | Template for `sp_daemon_cpu_glue.c`: cast `qm_opaque` to `qwen3_model *`, dispatch into engine entry. |
| Hex CMake | `tools/sp_daemon/c_backend/CMakeLists.txt:74-100` | Template for the CPU static lib build. Note key fact: hex variant deliberately OMITS `cpu_overlay.c` (symbol collision with math-core's `sp_kernels_read_env` + `qwen3_q4_stats`). For WIRE-CPU we MUST include `cpu_overlay.c` (it owns the AVX path) and resolve the collision via macro-renames at compile time. |
| Hex daemon registration block | `tools/sp_daemon/src/daemon.rs:340-392` | Template for the CPU daemon registration; same shape (env-gate + feature-gate + register on TARGET session pre-Mutex-wrap + `wire_cpu_active` bool in AppState). |
| Cargo features | `tools/sp_daemon/Cargo.toml:117-126` | Template: `wire_hex_backend` line is the model for `wire_cpu_backend`. |
| build.rs link block | `tools/sp_daemon/build.rs:132-182` | Template: hex variant guards behind `CARGO_FEATURE_WIRE_HEX_BACKEND` + `target_os == "android"`; CPU variant gates behind `CARGO_FEATURE_WIRE_CPU_BACKEND` and is HOST-targeted (no Android cross-compile). |
| Routes endpoint | `tools/sp_daemon/src/routes.rs:64-110` | Template: `BackendCounts { hex_forward_count, wire_hex_active, ntt_hex_forward_count, ntt_hex_inverse_count }` -> add `cpu_forward_count` + `wire_cpu_active`. |
| AppState wire_hex_active | `tools/sp_daemon/src/state.rs:71` | Template: add `pub wire_cpu_active: bool` field. |
| `lib.rs` module export | `tools/sp_daemon/src/lib.rs:27-34` | Template: `#[cfg(all(target_os = "android", feature = "wire_hex_backend"))]` -> for CPU just `#[cfg(feature = "wire_cpu_backend")]` (no target_os gate; host-built). |

### CPU backend code (READ-ONLY; do NOT modify)

| File | Symbol | Role |
|---|---|---|
| `src/backends/cpu/cpu_forward.c:249` | `int qwen3_forward(const qwen3_model *m, const int32_t *tokens, int n_tok, float *logits)` | The Qwen3 CPU forward entry point. Collides with math-core's `qwen3_forward` (`lib/shannon-prime-system/core/forward/forward.c:300`). |
| `src/backends/cpu/cpu_gemma3.c:41` | `int gemma3_forward(const qwen3_model *m, ...)` | The Gemma3 CPU forward entry point. Collides with math-core's `gemma3_forward` (`lib/shannon-prime-system/core/forward/gemma3.c:39`). |
| `src/backends/cpu/cpu_overlay.c:30,39,44,100,182,189,196,208,243` | `sp_kernels_read_env`, `qwen3_q4_stats`, `dot_f32`, `matmul`, `rmsnorm`, `rmsnorm_head`, `rope_neox`, `kernels_attn_head`, `embed_row` | Engine-side kernel surface. `sp_kernels_read_env` + `qwen3_q4_stats` collide with math-core's same-named symbols in `lib/shannon-prime-system/core/forward_dispatch/forward_dispatch.c:30,39`. The other names (`matmul`, `embed_row` etc.) are UNIQUE to the engine. |
| `src/backends/cpu/avx512/*.c` | `sp_avx512_init`, `sp_avx512_vnni_matvec`, ... | sp_-prefixed AVX-512 primitives; tested standalone by `tests/test_avx512.c` and `tests/bench_avx512.c`. NOT currently called from `matmul_arena` (the Q8 production path is scalar in `cpu_overlay.c:81-97`; the AVX2 path in `dot_f32` fires for f32/f16 GGUF tensors only). This is an upstream gap; future WIRE-CPU-V2 sprint candidate. For tonight, AVX-512 path still compiles into the lib via the feature flags but the actual matmul kernel hot path stays scalar. We measure honest baseline anyway. |
| `include/sp_engine/hexagon_backend.h:23-27` | template header shape | Will create symmetric `include/sp_engine/cpu_backend.h` declaring `int gemma3_forward_cpu` / `int qwen3_forward_cpu` -- both symbols owned by the daemon-link lib's glue file (NOT polluting the production sp_engine target). |

### L1 ABI register hook

| File | Symbol | Role |
|---|---|---|
| `lib/shannon-prime-system/include/sp/sp_l1.h:269-296` | `sp_forward_dispatch_fn` typedef + `sp_session_register_forward_backend` + `sp_session_forward_backend_fn` | Frozen ABI §6. WIRE-HEX already validated end-to-end (closure §gates `T_WIRE_HEX_BACKEND_LINKED` PASS, `T_WIRE_HEX_BACKEND_DISPATCHES` PASS at 1 hex_forward_count). Same hook for CPU. |

### Discipline guards

- `feedback-no-silent-gate-revisions` -- if `T_WIRE_CPU_BIT_EXACT_VS_REF` divergence appears, file it; do not widen tolerance.
- `feedback-bundled-changeset-root-cause-ambiguity` -- stage commits one variable at a time (lib -> Cargo feature + trampoline -> daemon registration -> routes -> measurement).
- `reference-lattice-decode-determinism` -- byte-exact greedy-argmax across backends is the precondition we're verifying (hex variant confirmed this 2026-05-31).
- `feedback-lattice-baseline-is-prior-lattice` -- for tok/s headline, math-core scalar f32 reference IS the right baseline here per task brief ("WIRE-CPU's whole point is integer-vectorized AVX-512 vs scalar f32 ref. Both are CPU; both are host-callable. Fresh-backend wiring, not lattice-vs-lattice").

---

## Stage commits

### Stage 1 -- static lib build script + standalone CMake

**Goal:** produce `build-host-cpu-backend/libsp_cpu_daemon_backend.a` (or
`.lib` on MSVC) from `cpu_forward.c` + `cpu_gemma3.c` + `cpu_overlay.c` +
`avx512/*.c` + glue.

**Files created:**

1. `include/sp_engine/cpu_backend.h` (~30 LOC) -- the public ABI surface for
   the daemon-link variant. Declares `gemma3_forward_cpu` + `qwen3_forward_cpu`.
2. `tools/sp_daemon/c_backend/sp_daemon_cpu_glue.c` (~120 LOC) -- the L1
   §6 dispatcher (`sp_daemon_cpu_forward`) plus the macro-renamed wrappers
   that expose `gemma3_forward_cpu` / `qwen3_forward_cpu`.
3. `tools/sp_daemon/c_backend/CMakeLists-cpu.txt` (~80 LOC) -- standalone
   CMake mirroring `CMakeLists.txt` (the hex backend's). HOST target (no
   Android toolchain). Compiles engine cpu sources with
   `-Dgemma3_forward=gemma3_forward_cpu_impl
   -Dqwen3_forward=qwen3_forward_cpu_impl
   -Dsp_kernels_read_env=sp_cpu_kernels_read_env
   -Dqwen3_q4_stats=sp_cpu_q4_stats` to dodge the collision with math-core
   archives the daemon already links.
4. `tools/sp_daemon/build-host-cpu-backend.bat` (~50 LOC) -- driver bat
   (PowerShell equivalent if needed). Runs CMake host build.

**Gate:** `T_WIRE_CPU_STATIC_LIB_BUILT`. Artefact `libsp_cpu_daemon_backend.a`
(or `.lib`) exists, `nm` (Linux) / `dumpbin /symbols` (Windows MSVC) shows
both `gemma3_forward_cpu` AND `sp_daemon_cpu_forward` as defined external
symbols.

### Stage 2 -- Cargo feature + Rust trampoline + daemon registration

**Files modified/created:**

1. `tools/sp_daemon/Cargo.toml` -- add `wire_cpu_backend` feature alongside
   `wire_hex_backend` (~3 LOC).
2. `tools/sp_daemon/build.rs` -- mirror the WIRE-HEX feature-gated link
   block (~30 LOC); points at the host CPU backend lib directory; no
   FastRPC link needed (HOST target, no cDSP).
3. `tools/sp_daemon/src/lib.rs` -- export `pub mod cpu_forward_dispatch;`
   gated by `#[cfg(feature = "wire_cpu_backend")]` (no target_os gate
   because CPU backend is host-targeted) (~3 LOC).
4. `tools/sp_daemon/src/cpu_forward_dispatch.rs` (~165 LOC, new) --
   template-copy of `hex_forward_dispatch.rs` with `hex -> cpu` rename.
5. `tools/sp_daemon/src/daemon.rs` -- add a parallel registration block
   right after the existing WIRE-HEX block (around line 392). Env-gate
   on `SP_DAEMON_BACKEND=cpu` (~50 LOC).
6. `tools/sp_daemon/src/state.rs` -- add `pub wire_cpu_active: bool`
   field (~3 LOC).

**Gate:** `T_WIRE_CPU_DAEMON_LINKED`. `cargo build --features wire_cpu_backend
--release` succeeds on host. `nm` / `dumpbin` on `sp-daemon` shows three
symbols at concrete addresses: `gemma3_forward_cpu`, `sp_wire_cpu_forward_dispatch`,
`sp_session_register_forward_backend`.

### Stage 3 -- /v1/debug/backend_counts extension + runtime test

**Files modified:**

1. `tools/sp_daemon/src/routes.rs` -- extend `BackendCounts` struct with
   `cpu_forward_count` + `wire_cpu_active` fields; populate them
   identically to the hex fields (~12 LOC).

**Gate:** `T_WIRE_CPU_RUNTIME_ACTIVE`. Start daemon with
`SP_DAEMON_BACKEND=cpu` env on a CPU-AVX-capable Knack host. Curl
`/v1/debug/backend_counts`; expect `wire_cpu_active: true`. After one
`/v1/chat` prefill, expect `cpu_forward_count > 0`.

### Stage 4 -- bit-exact gate vs reference

**Procedure:**
- Start ref daemon (no env). Drive `/v1/chat` with a fixed prompt. Capture
  the first 32 greedy-argmax token IDs.
- Stop daemon. Start `SP_DAEMON_BACKEND=cpu` daemon. Drive identical prompt.
- Diff token-id sequences.

**Gate:** `T_WIRE_CPU_BIT_EXACT_VS_REF`. PASS = byte-equal. FAIL = surface
UPSTREAM (engine cpu forward + math-core forward should agree because both
are scalar-f32-equivalent for Q8 arena; the AVX2 dot_f32 only fires for
f32/f16 GGUF tensors which are absent in a Q8 arena model -- and even there
AVX2 single-precision FMA is within ULP of scalar accumulation for typical
hidden-dim sizes). Per `reference-lattice-decode-determinism` greedy + fixed
spec-decode K + same checkpoint + same context = byte-equal expected.

### Stage 5 -- tok/s + closure

**Procedure:**
- 3-rep prefill + decode tok/s, ref daemon vs CPU daemon, same model.
- Closure document at `tools/sp_compute_skel/docs/CLOSURE-WIRE-CPU.md`.
- README §2 status table: update row 2 ("CPU backend (AVX-512 + cpu_overlay)")
  from "wired no" to "wired yes (`SP_DAEMON_BACKEND=cpu`)" or "yes (build-validated;
  runtime measurement deferred)" depending on host availability.

**Gate:** `T_WIRE_CPU_TOKS_MEASURED`. Numbers reported with stddev across 3 reps.

---

## Pre-flight host availability assessment

- Knack's Windows host: VS 2019 BT + CUDA 12.4 installed per
  `reference-cuda-build-recipe`. AVX-512 silicon: Beast Canyon i9-11900KB
  (TGL-B) HAS AVX-512 + AVX-512 VNNI in silicon, per
  `reference-hyperv-cpuid-masking` (the WAITPKG bit is masked by Hyper-V,
  but standard AVX-512F/VNNI feature bits are NOT masked).
- Existing `engine-wire-finish` worktree showed successful host MSVC build
  artifacts at `D:\F\shannon-prime-repos\shannon-prime-engine\build-cuda\bin\sp-engine.exe`.
- If MSVC host CPU build trips on AVX-512 specific intrinsics, fall back to
  AVX2-only build (set `SP_ENGINE_WITH_AVX512=OFF` while keeping
  `SP_ENGINE_WITH_AVX2=ON`); the wiring contract is independent of which
  ISA tier compiles.

---

## What this sprint does NOT do

- Adds `cpu_overlay.c::matmul_arena` AVX-512 wiring (the Q8 production
  matmul path is still scalar; `sp_avx512_vnni_matvec` exists as a
  standalone primitive that's NOT called by the forward). Future
  WIRE-CPU-V2 candidate. Surfaced honestly in closure.
- Wires CUDA / Vulkan (parallel WIRE-CUDA / WIRE-VULKAN sprints in
  separate worktrees: `engine-wire-cuda`, `engine-wire-vulkan`).
- Touches kernels under `src/backends/cpu/`. If any modification appears
  necessary the sprint STOPS and surfaces upstream per task brief.

---

## Worktree status (plan-commit time)

```
$ cd D:\F\shannon-prime-repos\engine-wire-cpu
$ git status
On branch sprint/wire-cpu
nothing to commit, working tree clean

$ git log --oneline -3
73f3367 Merge sprint/v5-ffn-vtcm -- FFN VTCM PING-PONG TILE-STREAMING SHIPS LIFT...
92d6acd [Stage 6] V5 closure...
9a21a0f [Stage 5] V5 -- T_V5_DMA_PINGPONG_OBSERVED instrumentation...
```

Math-core submodule pinned at WIRE-HEX merge state (`0b3b86b`). No
math-core changes anticipated in this sprint -- the §6 hook already exists.
