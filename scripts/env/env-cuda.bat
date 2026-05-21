@echo off
REM ─────────────────────────────────────────────────────────────────────
REM env-cuda.bat — CUDA backend environment (CUDA 12.4 + VS2019 BT)
REM Pinned toolchain.  Do not bump CUDA or VS without a project decision.
REM ─────────────────────────────────────────────────────────────────────

call "%~dp0env-common.bat"

if not exist "%SP_PIN_VS_BUILDTOOLS%\VC\Auxiliary\Build\vcvars64.bat" (
    echo [env-cuda] ERROR: VS2019 Build Tools not at expected location:
    echo            %SP_PIN_VS_BUILDTOOLS%
    exit /b 1
)
if not exist "%SP_PIN_CUDA_ROOT%\bin\nvcc.exe" (
    echo [env-cuda] ERROR: CUDA %SP_PIN_CUDA_VERSION% not at expected location:
    echo            %SP_PIN_CUDA_ROOT%
    echo            Install CUDA Toolkit %SP_PIN_CUDA_VERSION% from NVIDIA.
    exit /b 1
)

call "%SP_PIN_VS_BUILDTOOLS%\VC\Auxiliary\Build\vcvars64.bat" >nul

set CUDA_PATH=%SP_PIN_CUDA_ROOT%
set CUDAToolkit_ROOT=%SP_PIN_CUDA_ROOT%
set PATH=%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%PATH%

REM Target SM architectures.  RTX 3000 series = sm_86; RTX 4000 = sm_89.
REM We compile both; selectable at runtime by cudaGetDeviceProperties.
set SP_CUDA_ARCH=86;89

set SP_BACKEND=cuda
set SP_BUILD_DIR=%SP_ENGINE%\build-cuda

nvcc --version | findstr "release" >nul && (
    for /f "tokens=5,6 delims=, " %%a in ('nvcc --version ^| findstr "release"') do (
        echo [env-cuda] nvcc release %%a%%b activated.  SP_CUDA_ARCH=%SP_CUDA_ARCH%  SP_BUILD_DIR=%SP_BUILD_DIR%
    )
) || echo [env-cuda] nvcc activated.  SP_CUDA_ARCH=%SP_CUDA_ARCH%
