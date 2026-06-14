@echo off
REM KAI-1c #219: journaled-ring O(1) telemetry + D2D save-before-store tax.
call "%~dp0scripts\env\env-common.bat"
set "PATH=%SP_PIN_CUDA_ROOT%\bin;%PATH%"
nvidia-smi --lock-gpu-clocks=1680
REM NOTE: the 2060 does NOT support memory-clock lock ("not supported"); core lock is all we get.
REM Mem clock floats under DVFS => bandwidth-bound decode has ~12%% wall jitter; do not trust
REM cross-config wall-clock deltas <10%% (use within-config slopes or cudaEvent timing).
nvidia-smi --lock-memory-clocks=7000
set "SP_GEMMA4_SPMODEL=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-model"
set "SP_GEMMA4_SPTOK=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-tokenizer"
set "EXE=%SP_ENGINE%\build-cuda-vs22\tests\test_gemma4_cuda.exe"
set "SP_G4_KV_RING_TEL=1"
set "SP_G4_KV_RING_W=1024"
set "SP_CUDA_DECODE_INT8=1"
echo ===== [%TIME%] KAI-1c #219 ring telemetry W=%SP_G4_KV_RING_W% =====
"%EXE%"
echo ===== [%TIME%] RTEL_EXIT=%ERRORLEVEL% (0=CONFIRMED 3=REVIEW 2=err) =====
nvidia-smi --reset-gpu-clocks >nul 2>&1
nvidia-smi --reset-memory-clocks >nul 2>&1
