@echo off
REM Rebuild the daemon CUDA backend lib (carries SP_REPLAY_ALPHA in cuda_forward.cu) + the wire_cuda daemon.
call "%~dp0scripts\env\env-cuda.bat"
if errorlevel 1 ( echo ENV_FAIL & exit /b 1 )
set "ENGINE=%~dp0"
cmake -S "%ENGINE%tools\sp_daemon\c_backend_cuda" -B "%ENGINE%build-host-cuda-backend" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=cl -DCMAKE_CUDA_COMPILER="%SP_PIN_CUDA_ROOT%/bin/nvcc.exe" -DCMAKE_CUDA_ARCHITECTURES=%SP_CUDA_ARCH% -DCMAKE_CUDA_FLAGS="--use-local-env"
if errorlevel 1 ( echo CFG_FAIL & exit /b 1 )
cmake --build "%ENGINE%build-host-cuda-backend" --config Release
if errorlevel 1 ( echo LIB_FAIL & exit /b 1 )
set "LIBCLANG_PATH=C:\Program Files\LLVM\bin"
cd /d "%ENGINE%tools\sp_daemon"
cargo build --release --features wire_cuda_backend --target-dir target-wirecuda --bin sp-daemon
if errorlevel 1 ( echo CARGO_FAIL & exit /b 1 )
echo B3V9_BUILD_OK
