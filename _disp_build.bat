@echo off
call "D:\F\shannon-prime-repos\shannon-prime-system-engine\scripts\env\env-cuda.bat" >nul 2>&1
set "LIBCLANG_PATH=C:\Program Files\LLVM\bin"
cd /d "D:\F\shannon-prime-repos\shannon-prime-system-engine\tools\sp_daemon"
cargo build --release --features wire_cuda_backend --target-dir target-wirecuda --bin sp-daemon
echo DISP_BUILD_DONE_%errorlevel%
