@echo off
setlocal
call "%~dp0..\env\env-cuda.bat" || exit /b 1

REM --use-local-env: with VS2019 BuildTools only (no full VS in the registry),
REM nvcc's internal vcvars detection fails ("Could not set up the environment
REM for Microsoft Visual Studio"). The flag makes nvcc inherit the parent
REM vcvars64 shell env that env-cuda.bat already activated.
if not exist "%SP_BUILD_DIR%" (
    cmake -S "%SP_ENGINE%" -B "%SP_BUILD_DIR%" -G %SP_GENERATOR% ^
        -DCMAKE_BUILD_TYPE=%SP_BUILD_TYPE_DEFAULT% ^
        -DSP_ENGINE_BACKEND=cuda ^
        -DSP_ENGINE_WITH_CUDA=ON ^
        -DCMAKE_CUDA_COMPILER="%SP_PIN_CUDA_ROOT%/bin/nvcc.exe" ^
        -DCMAKE_CUDA_ARCHITECTURES="%SP_CUDA_ARCH%" ^
        -DCMAKE_CUDA_FLAGS="--use-local-env" ^
        -DSP_ENGINE_BUILD_TESTS=ON ^
        || exit /b 1
)
cmake --build "%SP_BUILD_DIR%" --config %SP_BUILD_TYPE_DEFAULT% -j
echo BUILD_EXIT=%ERRORLEVEL%
endlocal
