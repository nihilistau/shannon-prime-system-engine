@echo off
:: ============================================================================
:: §3-HX Sprint A — build libshannonprime_echo_skel.so for V69 cDSP
:: (Path B: NO signing — Unsigned PD admission via DSPRPC_CONTROL_UNSIGNED_MODULE)
:: ============================================================================
::
:: Prerequisites (per [[reference-qualcomm-sdk-inventory]] +
::                   [[reference-hexagon-build-recipe]]):
::   - Hexagon_SDK 5.5.6.0 at C:\Qualcomm\Hexagon_SDK\5.5.6.0
::   - HEXAGON_Tools 8.7.06 at SDK\tools\HEXAGON_Tools\8.7.06\
::   - Git Bash on PATH (qaic.exe shells out to `sh`)
::   - ADB on PATH (for the push step)
::
:: Outputs:
::   build/echo.h, build/echo_stub.c, build/echo_skel.c   (qaic-generated)
::   build/libshannonprime_echo_skel.so                    (hexagon-clang)
::
:: After successful build, push to device:
::   adb push build\libshannonprime_echo_skel.so /data/local/tmp/
::
:: ============================================================================

setlocal EnableExtensions EnableDelayedExpansion

:: %~dp0 ends with a trailing backslash. Strip it so quoted -I args don't
:: have \" at the end (cmd reads that as escaped quote → swallows the next
:: arg into the same string and hexagon-clang reports "no input files").
set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
set "BUILD_DIR=%SCRIPT_DIR%\build"

if not defined HEXAGON_SDK_ROOT (
    set "HEXAGON_SDK_ROOT=C:\Qualcomm\Hexagon_SDK\5.5.6.0"
)
if not exist "%HEXAGON_SDK_ROOT%\incs\remote.h" (
    echo [ERROR] HEXAGON_SDK_ROOT does not contain incs\remote.h: %HEXAGON_SDK_ROOT%
    exit /b 1
)

set "QAIC_EXE=%HEXAGON_SDK_ROOT%\ipc\fastrpc\qaic\bin\qaic.exe"
if not exist "%QAIC_EXE%" (
    echo [ERROR] qaic.exe not found at %QAIC_EXE%
    echo         For older SDKs try %HEXAGON_SDK_ROOT%\ipc\fastrpc\qaic\WinNT\qaic.exe
    exit /b 1
)

set "HEXAGON_TOOLS_VER=8.7.06"
set "HEXAGON_CLANG=%HEXAGON_SDK_ROOT%\tools\HEXAGON_Tools\%HEXAGON_TOOLS_VER%\Tools\bin\hexagon-clang.exe"
if not exist "%HEXAGON_CLANG%" (
    echo [ERROR] hexagon-clang.exe not found at %HEXAGON_CLANG%
    echo         Update HEXAGON_TOOLS_VER if your SDK ships a different version.
    exit /b 1
)

:: Prepend Git Bash to PATH (per reference-hexagon-build-recipe)
set "GIT_BASH=%PROGRAMFILES%\Git\bin"
if exist "%GIT_BASH%\sh.exe" (
    set "PATH=%GIT_BASH%;%PROGRAMFILES%\Git\usr\bin;%PATH%"
) else (
    echo [WARN] Git sh.exe not at %GIT_BASH% — qaic may fail with cryptic error.
)

if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"
pushd "%BUILD_DIR%"

echo [echo-skel] Step 1: qaic.exe IDL -^> stub + skel + header
"%QAIC_EXE%" -mdll -I "%HEXAGON_SDK_ROOT%\incs\stddef" "%SCRIPT_DIR%\echo.idl"
if errorlevel 1 ( echo [ERROR] qaic failed && popd && exit /b 1 )

echo [echo-skel] Step 2: hexagon-clang -mv69 -^> libshannonprime_echo_skel.so
"%HEXAGON_CLANG%" ^
    -O3 -mv69 -G0 -shared -fPIC ^
    -I "%HEXAGON_SDK_ROOT%\incs" ^
    -I "%HEXAGON_SDK_ROOT%\incs\stddef" ^
    -I "%HEXAGON_SDK_ROOT%\incs\fastrpc" ^
    -I "%BUILD_DIR%" ^
    -I "%SCRIPT_DIR%" ^
    "%BUILD_DIR%\echo_skel.c" ^
    "%SCRIPT_DIR%\echo_imp.c" ^
    -o "%BUILD_DIR%\libshannonprime_echo_skel.so" ^
    -lhexagon
if errorlevel 1 ( echo [ERROR] hexagon-clang failed && popd && exit /b 1 )

echo [echo-skel] Step 3: adb push to /data/local/tmp/
adb push "%BUILD_DIR%\libshannonprime_echo_skel.so" /data/local/tmp/
if errorlevel 1 ( echo [ERROR] adb push failed (device connected?) && popd && exit /b 1 )

popd
echo [echo-skel] OK — libshannonprime_echo_skel.so built + pushed.
echo            Run smoke harness:
echo              cargo build --target aarch64-linux-android --release ^
                  --manifest-path tools\sp_dsp_smoke\Cargo.toml
echo              adb push tools\sp_dsp_smoke\target\aarch64-linux-android\release\test_dsp_rpc /data/local/tmp/
echo              adb shell "chmod +x /data/local/tmp/test_dsp_rpc"
echo              adb shell "ADSP_LIBRARY_PATH=\"/data/local/tmp;\" /data/local/tmp/test_dsp_rpc"
endlocal
