@echo off
:: §3-HX Sprint A echo skel build — wraps SDK build_cmake (mirrors S22U pattern).
:: Outputs: hexagon_Debug_toolv87_v69\ship\libsp_echo_skel.so
setlocal

if "%HEXAGON_SDK_ROOT%"=="" set "HEXAGON_SDK_ROOT=C:\Qualcomm\Hexagon_SDK\5.5.6.0"
set "DSP_ARCH=v69"
set "BUILD_TYPE=Debug"

if "%SDK_SETUP_ENV%"=="" (
    call "%HEXAGON_SDK_ROOT%\setup_sdk_env.cmd"
    if errorlevel 1 exit /b 1
)

build_cmake hexagon DSP_ARCH=%DSP_ARCH% BUILD=%BUILD_TYPE% -gMake
if errorlevel 1 exit /b %errorlevel%

:: adb push
set "SKEL=hexagon_Debug_toolv87_v69\ship\libsp_echo_skel.so"
if exist "%SKEL%" (
    echo [sp_echo] Pushing %SKEL% to /data/local/tmp/
    adb push "%SKEL%" /data/local/tmp/
)
endlocal
