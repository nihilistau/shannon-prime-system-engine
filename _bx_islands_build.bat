@echo off
call "%~dp0scripts\env\env-common.bat"
set "SP_PIN_VS_BUILDTOOLS=%SP_PIN_VS2022_BUILDTOOLS%"
set "_VCVARS=%SP_PIN_VS_BUILDTOOLS%\VC\Auxiliary\Build\vcvars64.bat"
call "%_VCVARS%" >nul
set "CUDA_PATH=%SP_PIN_CUDA_ROOT%"
set "CUDAToolkit_ROOT=%SP_PIN_CUDA_ROOT%"
set "PATH=%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%PATH%"
set "SP_BUILD_DIR=%SP_ENGINE%\build-cuda-vs22"
cmake --build "%SP_BUILD_DIR%" --target test_gemma4_ppl_cuda
echo BUILD_EXIT=%ERRORLEVEL%
