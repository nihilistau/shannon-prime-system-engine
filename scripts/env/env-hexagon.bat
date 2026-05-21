@echo off
REM ─────────────────────────────────────────────────────────────────────
REM env-hexagon.bat — Hexagon backend environment
REM
REM Targets Snapdragon 8 Gen 1 V69 HTP (S22U class).  Host builds on
REM Windows; device .so deployed via FastRPC.  Critical: prepend
REM Git for Windows sh.exe so the SDK's *.cmd scripts find a POSIX shell.
REM ─────────────────────────────────────────────────────────────────────

call "%~dp0env-common.bat"

if not exist "%SP_PIN_HEXAGON_SDK%" (
    echo [env-hexagon] ERROR: Hexagon SDK not at expected location:
    echo               %SP_PIN_HEXAGON_SDK%
    echo               Install from Qualcomm Developer Network (requires login).
    exit /b 1
)

set HEXAGON_SDK_ROOT=%SP_PIN_HEXAGON_SDK%
set HEXAGON_TOOLS_VER=8.7.06

REM Prepend Git sh.exe to PATH so SDK's *.cmd scripts work.
REM (Without this, the SDK falls back to cmd.exe and silently breaks.)
if exist "%SP_PIN_GIT_USR_BIN%\sh.exe" (
    set PATH=%SP_PIN_GIT_USR_BIN%;%PATH%
) else (
    echo [env-hexagon] WARNING: Git for Windows sh.exe not at %SP_PIN_GIT_USR_BIN%
    echo               Some SDK scripts will fail.  Install Git for Windows.
)

set PATH=%HEXAGON_SDK_ROOT%\tools\HEXAGON_Tools\%HEXAGON_TOOLS_VER%\Tools\bin;%PATH%

REM FastRPC rpcmem alloc size MUST equal IDL length parameter exactly.
REM Documented failure mode: over-allocating returns AEE_EUNSUPPORTED silently.
set SP_FASTRPC_STRICT_ALLOC=1

REM freethedsp shim (opt-in via env var on device run)
set SP_FREETHEDSP=0

set SP_BACKEND=hexagon
set SP_BUILD_DIR=%SP_ENGINE%\build-hexagon
set SP_HEXAGON_TARGET=v69

if exist "%SP_PIN_VS_BUILDTOOLS%\VC\Auxiliary\Build\vcvars64.bat" (
    call "%SP_PIN_VS_BUILDTOOLS%\VC\Auxiliary\Build\vcvars64.bat" >nul
)

echo [env-hexagon] Hexagon SDK %SP_PIN_HEXAGON_SDK% activated for target %SP_HEXAGON_TARGET%.
echo [env-hexagon] SP_BUILD_DIR=%SP_BUILD_DIR%  SP_FASTRPC_STRICT_ALLOC=1  SP_FREETHEDSP=%SP_FREETHEDSP%
