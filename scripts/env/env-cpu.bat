@echo off
REM ─────────────────────────────────────────────────────────────────────
REM env-cpu.bat — CPU backend environment (MSVC + Ninja)
REM Activates VS2019 Build Tools x64 host/target.
REM ─────────────────────────────────────────────────────────────────────

call "%~dp0env-common.bat"

if not exist "%SP_PIN_VS_BUILDTOOLS%\VC\Auxiliary\Build\vcvars64.bat" (
    echo [env-cpu] ERROR: VS2019 Build Tools not at expected location:
    echo           %SP_PIN_VS_BUILDTOOLS%
    echo           Install or correct SP_PIN_VS_BUILDTOOLS in env-common.bat.
    exit /b 1
)

call "%SP_PIN_VS_BUILDTOOLS%\VC\Auxiliary\Build\vcvars64.bat" >nul

REM Pin AVX feature flags.  The engine has explicit AVX2 and AVX512 code paths
REM gated by CMake options; both are compiled and selected at runtime by
REM hardware probe.  We compile with /arch:AVX2 as the floor; AVX512 emitted
REM via target_compile_options on specific TUs (see src/backends/cpu/CMakeLists).
set SP_CPU_AVX2=1
set SP_CPU_AVX512=1

set SP_BACKEND=cpu
set SP_BUILD_DIR=%SP_ENGINE%\build-cpu

echo [env-cpu] MSVC %VCToolsVersion% activated.  SP_BACKEND=cpu  SP_BUILD_DIR=%SP_BUILD_DIR%
