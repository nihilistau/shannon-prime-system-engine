@echo off
:: §3-HX Sprint F: build the Halide host generator + AOT-emit the Hexagon .o + .h.
:: Output: ../sp_compute_skel/halide_gen/sp_axpby_2d_halide.{o,h}
::
:: Mirrors C:\Qualcomm\HALIDE_Tools\2.4.07\Halide\Examples\standalone\simulator\
::         apps\conv3x3a32\test-conv3x3a32.cmd build chain.

setlocal EnableDelayedExpansion

if not defined HALIDE_ROOT set "HALIDE_ROOT=C:\Qualcomm\HALIDE_Tools\2.4.07\Halide"
if not defined VCVARS set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

if not exist "%HALIDE_ROOT%\lib\Halide.lib" goto :halide_missing
if not exist "%VCVARS%" goto :vcvars_missing

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
set "OUT_DIR=%SCRIPT_DIR%\..\sp_compute_skel\halide_gen"
if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

set "BUILD_DIR=%SCRIPT_DIR%\build"
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

echo [sp_halide_gen] === Stage 1: build host generator with cl.exe + Halide.lib ===
call "%VCVARS%" >nul
if errorlevel 1 goto :vcvars_setup_failed

cl.exe /EHsc /nologo /std:c++17 ^
    /I "%HALIDE_ROOT%\include" ^
    "%SCRIPT_DIR%\sp_axpby_2d_gen.cpp" ^
    "%HALIDE_ROOT%\tools\GenGen.cpp" ^
    /link /libpath:"%HALIDE_ROOT%\lib" Halide.lib ^
    /OUT:"%BUILD_DIR%\sp_axpby_2d_gen.exe"
if errorlevel 1 goto :cl_failed

echo [sp_halide_gen] === Stage 2: AOT-emit Hexagon .o + .h ===

:: Add Halide.dll to PATH so the generator exe can find it.
set "PATH=%HALIDE_ROOT%\bin;%PATH%"

"%BUILD_DIR%\sp_axpby_2d_gen.exe" ^
    -g sp_axpby_2d ^
    -f sp_axpby_2d_halide ^
    -e o,h,assembly ^
    -o "%OUT_DIR%" ^
    target=hexagon-32-noos-no_bounds_query-no_asserts-hvx_128
if errorlevel 1 goto :aot_failed

echo [sp_halide_gen] === Stage 1b: build FFN generator ===
cl.exe /EHsc /nologo /std:c++17 ^
    /I "%HALIDE_ROOT%\include" ^
    "%SCRIPT_DIR%\sp_ffn_2stage_gen.cpp" ^
    "%HALIDE_ROOT%\tools\GenGen.cpp" ^
    /link /libpath:"%HALIDE_ROOT%\lib" Halide.lib ^
    /OUT:"%BUILD_DIR%\sp_ffn_2stage_gen.exe"
if errorlevel 1 goto :cl_failed

echo [sp_halide_gen] === Stage 2b: AOT-emit FFN Hexagon .o + .h ===
"%BUILD_DIR%\sp_ffn_2stage_gen.exe" ^
    -g sp_ffn_2stage ^
    -f sp_ffn_2stage_halide ^
    -e o,h,assembly ^
    -o "%OUT_DIR%" ^
    target=hexagon-32-noos-no_bounds_query-no_asserts-hvx_128
if errorlevel 1 goto :aot_failed

echo [sp_halide_gen] === Stage 3: stage HalideRuntime.h + HalideRuntimeHexagonHost.h ===
copy /Y "%HALIDE_ROOT%\include\HalideRuntime.h" "%OUT_DIR%\HalideRuntime.h" >nul
if errorlevel 1 goto :stage_failed
copy /Y "%HALIDE_ROOT%\include\HalideRuntimeHexagonHost.h" "%OUT_DIR%\HalideRuntimeHexagonHost.h" >nul
if errorlevel 1 goto :stage_failed

echo [sp_halide_gen] === Emit summary ===
dir /b "%OUT_DIR%\sp_*_halide.*" "%OUT_DIR%\HalideRuntime*.h"
echo [sp_halide_gen] DONE
endlocal
exit /b 0

:halide_missing
echo [sp_halide_gen] ERROR: Halide.lib not found under HALIDE_ROOT
echo                  HALIDE_ROOT=%HALIDE_ROOT%
endlocal
exit /b 1

:vcvars_missing
echo [sp_halide_gen] ERROR: vcvars64.bat not found
echo                  VCVARS=%VCVARS%
echo                  set VCVARS=path\to\vcvars64.bat
endlocal
exit /b 1

:vcvars_setup_failed
echo [sp_halide_gen] ERROR: vcvars64 setup failed
endlocal
exit /b 1

:cl_failed
echo [sp_halide_gen] ERROR: cl.exe build failed
endlocal
exit /b 1

:aot_failed
echo [sp_halide_gen] ERROR: AOT emit failed
endlocal
exit /b 1

:stage_failed
echo [sp_halide_gen] ERROR: failed to stage HalideRuntime.h to OUT_DIR
endlocal
exit /b 1
