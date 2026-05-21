# Build Environment — shannon-prime-system-engine

Pinned toolchains, hard paths, version-locked. Set in stone for the project; not per-session decisions.

## Toolchain pins (do not bump without a project decision)

| Component | Pinned version | Path |
|-----------|----------------|------|
| Visual Studio Build Tools | 2019 | `C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools` |
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
| CPU     | `scripts\env\env-cpu.bat`     | `scripts\build\build-cpu.bat`     | `build-cpu/`     | AVX2 + AVX512 compiled, runtime-selected |
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

- `/arch:AVX2` is the compile floor; AVX512 paths are guarded by `__cpuid()` and dispatched at runtime so binaries run on any AVX2-capable host.

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
