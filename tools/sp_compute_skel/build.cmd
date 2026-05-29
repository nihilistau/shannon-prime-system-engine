@echo off
:: §3-HX Sprint D HVX compute skel build — wraps SDK build_cmake.
:: Outputs: hexagon_Debug_toolv87_v69\ship\libsp_compute_skel.so
setlocal

if "%HEXAGON_SDK_ROOT%"=="" set "HEXAGON_SDK_ROOT=C:\Qualcomm\Hexagon_SDK\5.5.6.0"
set "DSP_ARCH=v69"
set "BUILD_TYPE=Release"

if "%SDK_SETUP_ENV%"=="" (
    call "%HEXAGON_SDK_ROOT%\setup_sdk_env.cmd"
    if errorlevel 1 exit /b 1
)

build_cmake hexagon DSP_ARCH=%DSP_ARCH% BUILD=%BUILD_TYPE% -gMake
if errorlevel 1 exit /b %errorlevel%

set "SKEL=hexagon_Release_toolv87_v69\ship\libsp_compute_skel.so"
if exist "%SKEL%" (
    echo [sp_compute] Pushing %SKEL% to /data/local/tmp/
    adb push "%SKEL%" /data/local/tmp/
)
endlocal
