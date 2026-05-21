@echo off
REM ---------------------------------------------------------------------
REM env-cpu.bat -- CPU backend environment (MSVC + Ninja).
REM Activates VS2019 Build Tools x64 host/target. ASCII only; goto-based
REM error handling (the VS path contains "(x86)", which breaks if-blocks).
REM ---------------------------------------------------------------------

call "%~dp0env-common.bat"

set "_VCVARS=%SP_PIN_VS_BUILDTOOLS%\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%_VCVARS%" goto :no_vcvars

call "%_VCVARS%" >nul

REM AVX feature flags. The engine has explicit AVX2 and AVX512 code paths
REM gated by CMake options; /arch:AVX2 is the floor and AVX512 is emitted on
REM specific TUs (see src/backends/cpu/CMakeLists.txt).
set SP_CPU_AVX2=1
set SP_CPU_AVX512=1

set SP_BACKEND=cpu
set SP_BUILD_DIR=%SP_ENGINE%\build-cpu

echo [env-cpu] MSVC %VCToolsVersion% activated.  SP_BACKEND=cpu  SP_BUILD_DIR=%SP_BUILD_DIR%
goto :eof

:no_vcvars
echo [env-cpu] ERROR: VS2019 Build Tools vcvars64.bat not found at:
echo           %_VCVARS%
echo           Install VS2019 Build Tools or correct SP_PIN_VS_BUILDTOOLS in env-common.bat.
exit /b 1
