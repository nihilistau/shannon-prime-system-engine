---
type: runbook
title: Build Environment — shannon-prime-system-engine
description: "Pinned toolchains, hard paths, version-locked."
tags: [runbook]
timestamp: 2026-06-02T10:54:56Z
resource: ./docs/BUILD-ENV.md
sp_status: ACTIVE
sp_gate: none
sp_commit: TBD
sp_repro: none
---

# Build Environment — shannon-prime-system-engine

Pinned toolchains, hard paths, version-locked. Set in stone for the project; not per-session decisions.

## Toolchain pins (do not bump without a project decision)

| Component | Pinned version | Path |
|-----------|----------------|------|
| **MinGW gcc (CPU primary)** | **15.2** | `C:\ProgramData\mingw64\mingw64\bin` (on PATH) |
| Visual Studio Build Tools (CUDA host) | 2019 | `C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools` |
| **VS18 BuildTools (Tier-3 MSVC-parity only)** | **18 / MSVC v14.50 (cl 19.50)** | `D:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools` |

> **VS18/2022 path saved (2026-06-02).** There is **no "VS2022"** installed on this host — vswhere shows only 2019 BT and **18 BT** (on D:, MSVC v14.50, ships `<stdatomic.h>`). VS18 is the toolchain for the **Tier-3 MSVC-parity** build only — it is NOT the CPU backend (= MinGW gcc 15.2) and NOT the CUDA host (= VS2019 BT, tightly pinned with CUDA). Pinned as `SP_PIN_VS2022_BUILDTOOLS` in `env-common.bat`, separate from the CUDA-host `SP_PIN_VS_BUILDTOOLS`.

**CPU toolchain re-pin (2026-06-02, operator-approved).** The canonical CPU
backend build uses **MinGW gcc 15.2**, not MSVC. This matches roadmap §3.7's
Tier-1 (gcc closes in-session; MSVC is Tier-3 parity). It is also *forced* by
the code: the §18 AVX512 backend (`src/backends/cpu/avx512/avx512_*.c`) uses GCC
`__attribute__((target(...)))` + `__atomic_*` builtins, and math-core
`core/sp_channel/sp_hedge.c` uses C11 `<stdatomic.h>` — none of which VS2019 BT
(v142) can compile. MSVC remains required as the **CUDA host compiler** (nvcc +
CUDA 12.4 pin) and as the Tier-3 parity target, but Tier-3 needs **VS2022**
(ships `<stdatomic.h>`; the CI `windows-msvc` job already uses it) and a port of
the AVX512 GCC-isms before it is green. Until then, VS2019 BT builds CUDA only.
| CUDA Toolkit | 12.4 | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4` |
| Vulkan SDK | ≥ 1.3.250 | `%VULKAN_SDK%` (installer sets globally) |
| Hexagon SDK | 5.4.0.x | `C:\Qualcomm\Hexagon_SDK\5.4.0.x` |
| Hexagon Tools | 8.7.06 | `$HEXAGON_SDK_ROOT\tools\HEXAGON_Tools\8.7.06` |
| CMake | ≥ 3.20 | On PATH |
| Ninja | ≥ 1.10 | On PATH |
| Git for Windows | latest | `C:\Program Files\Git` (sh.exe required for Hexagon) |

The pins live in `scripts/env/env-common.bat`. **Change them there, nowhere else.**

## Build matrix

| Backend | Env script | Build script | Output dir | Notes |
|---------|-----------|--------------|------------|-------|
| CPU     | MinGW gcc on PATH (env-cpu.bat → gcc; **TODO: script still activates MSVC, re-point to gcc**) | `scripts\build\build-cpu.bat` | `build/` (gcc) — `build-cpu/` (MSVC) is CUDA-host only | AVX2 + AVX512 compiled (GCC target attrs), runtime-selected |
| CUDA    | `scripts\env\env-cuda.bat`    | `scripts\build\build-cuda.bat`    | `build-cuda/`    | sm_86 + sm_89 fat-binary |
| Vulkan  | `scripts\env\env-vulkan.bat`  | `scripts\build\build-vulkan.bat`  | `build-vulkan/`  | compute shaders, glslc-compiled |
| Hexagon | `scripts\env\env-hexagon.bat` | `scripts\build\build-hexagon.bat` | `build-hexagon/` | host stub + V69 device .so |

All four output dirs coexist. Switching backends does not invalidate the others. The CI gate at each phase is: all four backends still build clean + smoke-test green.

## Common workflow

1. Open the workspace in VSCode: `D:\F\shannon-prime-repos\shannon-prime-lattice.code-workspace`.
2. Ctrl-Shift-B → pick `build: CPU` (or CUDA / Vulkan / Hexagon).
3. After successful build, run a smoke test via the task list (`test: CPU smoke`, etc.).
4. The build env scripts validate pinned tools at activation. If a pin doesn't resolve, the script errors out — no silent fallback.

## Common failure modes (from prior project memory)

### Hexagon

- **rpcmem alloc must equal IDL length parameter.** Over-allocating returns `AEE_EUNSUPPORTED` silently. The env var `SP_FASTRPC_STRICT_ALLOC=1` is set by `env-hexagon.bat`; the host stub must respect it.
- **qaic path differs on Windows.** Use `$HEXAGON_SDK_ROOT\ipc\fastrpc\qaic\WinNT\qaic.exe`, not the Linux `bin/qaic` path.
- **Git sh.exe required.** SDK scripts call `*.cmd` that internally use `sh.exe`. Without Git for Windows on PATH (`%SP_PIN_GIT_USR_BIN%`), SDK scripts fail with cryptic errors.
- **`hexagon_fun.cmake` requires the WinNT/qaic.exe patch.** The Phase 2-HX bring-up handles this in `cmake/toolchain-hexagon.cmake`.
- **production-locked devices need freethedsp.** Toggle `SP_FREETHEDSP=1` on device runs to opt in.

### CUDA

- **VS2019 BT + CUDA 12.4 is tightly pinned.** Newer VS / older CUDA combinations break nvcc's MSVC integration.
- **`--use-local-env` may be required** when CUDA's own MSVC discovery fails; add to nvcc flags in `CMakeLists.txt` if needed (no current need on the pinned combo).

### CPU

- Build with **MinGW gcc 15.2** (the `build/` dir), not MSVC. AVX2 is the compile floor; AVX512 paths are guarded by `__cpuid()` and dispatched at runtime so binaries run on any AVX2-capable host.
- **MSVC (VS2019 BT) cannot build the CPU backend.** `avx512_{spinor,ternlog,persist}.c` use GCC `__attribute__((target("avx512f")))` and `__atomic_*` / `__ATOMIC_*` builtins (cl.exe: `error C2059/C2143/C2065`). `core/sp_channel/sp_hedge.c` needs `<stdatomic.h>`, absent before VS2022. Tier-3 MSVC parity therefore requires VS2022 + porting the GCC intrinsics to `<intrin.h>` (`_mm512_*` + `_Interlocked*`) — tracked, not yet done. Verified clean on gcc: math-core 19/19 ctest, engine 95/96 link (2026-06-02).

> **Tier-3 MSVC-parity progress (2026-06-02, partial).** Using the VS18 toolchain above, the engine library + `sp_toks` now **compile AND link** under MSVC. Done (committed): `SP_TARGET(s)` macro in `avx512.h` (empty on MSVC, `__attribute__((target))` on GCC) applied to all per-fn target attrs (engine `db84bf3`); `avx512_persist.c` `__atomic_*`→`_Interlocked*` shim (`33c6a27`); ternlog `aligned()`→C11 `alignas`; `sp_channel` `/experimental:c11atomics` (core submodule `777a10e`). **Remaining for full MSVC parity:** (1) two micro-bench TUs still GCC-only and unported — `tests/test_avx512_persist.c` (`__ATOMIC_*`), `tests/bench_avx_spinor_sweep.c` (`stream_nt`); (2) **OPEN: the MSVC binary SEGFAULTS at runtime** — `test_kv_spinor` (E_CPU_8) which RUNS as the gcc/older binary crashes (0xC0000005) rebuilt under VS18, so MSVC AVX512 codegen / the de-GCC port has a runtime defect to debug before Tier-3 parity is real. **None of this affects the CPU backend (= MinGW gcc) or CUDA — both unchanged.**

### Vulkan

- Need subgroup ops (`VK_KHR_shader_subgroup_*`). Vulkan SDK 1.3.x has them; older SDKs miss.
- AMD / Intel GPU driver compatibility varies; CPU fallback always available.

## Environment validation (sanity check)

Open a fresh cmd.exe and run:

```cmd
"D:\F\shannon-prime-repos\shannon-prime-system-engine\scripts\env\env-cpu.bat"
```

Expected last line: `[env-cpu] MSVC <version> activated.  SP_BACKEND=cpu  SP_BUILD_DIR=...`

Repeat for env-cuda / env-vulkan / env-hexagon. Each should print activation confirmation, OR a clear ERROR with the missing tool / path.

## Per-phase build status (filled in as phases land)

| Phase | CPU | CUDA | Vulkan | Hexagon |
|-------|:---:|:----:|:------:|:-------:|
| 0     | ✓ stubs | ✓ stubs | ✓ stubs | ✓ stubs |
| 1A-F  | —   |   —  |   —    |    —    |
| 2-CPU.A-F | — | — | — | — |
| 2-CU.A-F  | — | — | — | — |
| 2-VK.A-F  | — | — | — | — |
| 2-HX.A-F  | — | — | — | — |

## Updating the pins

If a toolchain pin needs to change (e.g. CUDA 12.4 → 12.5):

1. Verify the change actually fixes a problem.
2. Update `scripts/env/env-common.bat` constants.
3. Update this doc's pin table.
4. Add a note in `papers/SESSION-STATE-lat-<phase>.md` capturing the rationale.
5. Build all four backends from scratch and confirm they pass smoke tests.

Pins are sticky on purpose. The cost of toolchain churn across a multi-backend project is higher than the benefit of being on the latest minor version.
