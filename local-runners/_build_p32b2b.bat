@echo off
REM Build both the ARM shadow gate (test_gemma4_cuda) and the PPL deflection driver
REM (test_gemma4_ppl_cuda) on the VS2022/VS18 host.
call "%~dp0scripts\env\env-common.bat"
set "SP_PIN_VS_BUILDTOOLS=%SP_PIN_VS2022_BUILDTOOLS%"
set "_VCVARS=%SP_PIN_VS_BUILDTOOLS%\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%_VCVARS%" goto :novc
call "%_VCVARS%" >nul
set "CUDA_PATH=%SP_PIN_CUDA_ROOT%"
set "CUDAToolkit_ROOT=%SP_PIN_CUDA_ROOT%"
set "PATH=%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%PATH%"
set "SP_BUILD_DIR=%SP_ENGINE%\build-cuda-vs22"
if exist "%SP_BUILD_DIR%\CMakeCache.txt" goto :build
cmake -S "%SP_ENGINE%" -B "%SP_BUILD_DIR%" -G Ninja -DCMAKE_BUILD_TYPE=Release -DSP_ENGINE_BACKEND=cuda -DSP_ENGINE_WITH_CUDA=ON -DCMAKE_CUDA_COMPILER="%SP_PIN_CUDA_ROOT%/bin/nvcc.exe" -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_CUDA_FLAGS="--use-local-env" -DSP_ENGINE_BUILD_TESTS=ON
if errorlevel 1 goto :cfgfail
:build
cmake --build "%SP_BUILD_DIR%" --target test_gemma4_cuda test_gemma4_ppl_cuda
echo BUILD_EXIT=%ERRORLEVEL%
goto :eof
:novc
echo NO_VCVARS %_VCVARS%
exit /b 1
:cfgfail
echo CONFIGURE_FAIL
exit /b 1
