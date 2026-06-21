@echo off
call "D:\F\shannon-prime-repos\shannon-prime-system-engine\scripts\env\env-cuda.bat" >nul 2>&1
set "LIBCLANG_PATH=C:\Program Files\LLVM\bin"
cd /d "D:\F\shannon-prime-repos\shannon-prime-system-engine\tools\sp_daemon"
echo CURATOR_BUILD_START %DATE% %TIME% > D:\F\_curator_build.log
cargo build --release --features "wire_cuda_backend kairos" --target-dir target-wirecuda --bin sp-daemon >> D:\F\_curator_build.log 2>&1
echo CURATOR_BUILD_DONE_%errorlevel% >> D:\F\_curator_build.log