@echo off
REM KAI-1c G-1b-WRAP-NULL: wrap-aware ring rewind bit-exact gate.
call "%~dp0scripts\env\env-common.bat"
set "PATH=%SP_PIN_CUDA_ROOT%\bin;%PATH%"
nvidia-smi --lock-gpu-clocks=1680 >nul 2>&1
nvidia-smi --lock-memory-clocks=7000 >nul 2>&1
set "SP_GEMMA4_SPMODEL=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-model"
set "SP_GEMMA4_SPTOK=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-tokenizer"
set "EXE=%SP_ENGINE%\build-cuda-vs22\tests\test_gemma4_cuda.exe"
set "SP_G4_KV_WRAP=1"
set "SP_G4_KV_RING_W=16"
set "SP_G4_KV_JMAX=64"
set "SP_CUDA_DECODE_INT8=1"
echo ===== [%TIME%] G-1b-WRAP-NULL W=%SP_G4_KV_RING_W% Jmax=%SP_G4_KV_JMAX% =====
"%EXE%"
echo ===== [%TIME%] WRAP_EXIT=%ERRORLEVEL% (0=GREEN 3=RED 2=err) =====
nvidia-smi --reset-gpu-clocks >nul 2>&1
nvidia-smi --reset-memory-clocks >nul 2>&1
