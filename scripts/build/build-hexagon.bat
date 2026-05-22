@echo off
REM ---------------------------------------------------------------------
REM build-hexagon.bat -- Phase 2-HX two-artifact build.
REM
REM Unlike CUDA (one in-tree nvcc TU that runs on the dev box), the Hexagon
REM PPL gate runs ON THE PHONE: FastRPC is intra-device IPC (Android-arm to
REM cDSP). So this script produces TWO independently-buildable artifacts:
REM   (a) AARCH64-ANDROID host: sp_engine + math-core + test_ppl, built with
REM       the NDK r25c toolchain (AVX/CUDA off, pure scalar f32). On-phone
REM       runner; push it + the GGUF + fixtures and run via adb.
REM   (b) DSP SKEL: the HVX kernel .so, built SEPARATELY by the SDK's
REM       build_cmake hexagon (hexagon-clang V69). [HX.2+ -- placeholder now.]
REM
REM Usage:  build-hexagon.bat [android^|dsp^|both]   (default android)
REM ---------------------------------------------------------------------
setlocal
REM Suppress env-hexagon's setup_sdk_env PATH-cleanup loop noise (it splits the
REM inherited spaced PATH entries into bare tokens and tries to run them; benign,
REM errorlevel reset inside env-hexagon). The redirect keeps that from leaking
REM into this shell and aborting the build.
call "%~dp0..\env\env-hexagon.bat" 1>nul 2>nul || exit /b 1

set "WHAT=%~1"
if "%WHAT%"=="" set "WHAT=android"
if /I "%WHAT%"=="android" goto :android
if /I "%WHAT%"=="dsp"     goto :dsp
if /I "%WHAT%"=="both"    goto :android
echo [build-hexagon] unknown target "%WHAT%" (android, dsp, both)
exit /b 1

:android
set "DEV=%SP_DEVICE_DIR%"
if "%DEV%"=="" set "DEV=/data/local/tmp/sp22u"
set "NDK_TC=%ANDROID_NDK_ROOT%\build\cmake\android.toolchain.cmake"
if not exist "%NDK_TC%" goto :no_ndk

REM Configure once (single logical line: caret continuations inside a
REM parenthesised if-block break cmd's parser with this host's spaced paths).
if exist "%SP_BUILD_DIR%\CMakeCache.txt" goto :android_build
cmake -S "%SP_ENGINE%" -B "%SP_BUILD_DIR%" -G %SP_GENERATOR% -DCMAKE_BUILD_TYPE=%SP_BUILD_TYPE_DEFAULT% -DCMAKE_TOOLCHAIN_FILE="%NDK_TC%" -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-31 -DSP_ENGINE_BACKEND=hexagon -DSP_ENGINE_TARGET_ANDROID=ON -DSP_ENGINE_WITH_AVX2=OFF -DSP_ENGINE_WITH_AVX512=OFF -DSP_ENGINE_WITH_CUDA=OFF -DSP_ENGINE_BUILD_TESTS=ON -DSP_GEMMA3_GGUF="%DEV%/gemma-3-1b-it-f16.gguf" -DSP_QWEN3_GGUF="%DEV%/Qwen3-0.6B-f16.gguf"
if errorlevel 1 exit /b 1

:android_build
cmake --build "%SP_BUILD_DIR%" --config %SP_BUILD_TYPE_DEFAULT% -j --target test_ppl rpcmem_probe test_hex_rt
if errorlevel 1 exit /b 1
echo ANDROID_BUILD_EXIT=0
if /I "%WHAT%"=="both" goto :dsp
goto :done

:dsp
set "DSPDIR=%SP_ENGINE%\src\backends\hexagon\dsp"
if not exist "%DSPDIR%\CMakeLists.txt" goto :dsp_missing
pushd "%DSPDIR%"
build_cmake hexagon DSP_ARCH=%SP_HEXAGON_TARGET% BUILD=%SP_BUILD_TYPE_DEFAULT% -gMake
set "DSP_RC=%ERRORLEVEL%"
popd
echo DSP_BUILD_EXIT=%DSP_RC%
goto :done

:dsp_missing
echo [build-hexagon] DSP skel not scaffolded yet (HX.2): %DSPDIR%\CMakeLists.txt
goto :done

:no_ndk
echo [build-hexagon] ERROR: NDK android.toolchain.cmake not found:
echo           %NDK_TC%
echo           Check ANDROID_NDK_ROOT (env-hexagon.bat) / SDK install.
exit /b 1

:done
endlocal
