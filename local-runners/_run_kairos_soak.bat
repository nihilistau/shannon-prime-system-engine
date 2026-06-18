@echo off
REM G-KAIROS-1: >=24h unattended endurance soak on the journaled-ring metal.
REM Usage: _run_kairos_soak.bat            (full 24h)
REM        _run_kairos_soak.bat 0.05 3     (smoke: 3 loops)   args = HOURS MAXLOOPS
call "%~dp0scripts\env\env-common.bat"
set "PATH=%SP_PIN_CUDA_ROOT%\bin;%PATH%"
nvidia-smi --lock-gpu-clocks=1680
REM (2060 cannot lock memory clock; latency tripwire is consecutive-based to tolerate the DVFS jitter.)
set "SP_GEMMA4_SPMODEL=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-model"
set "SP_GEMMA4_SPTOK=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-tokenizer"
set "SP_KAIROS_TAPE=%SP_ENGINE%\tools\sp_daemon\tests\fixtures\kairos\tape_smoke.txt"
set "EXE=%SP_ENGINE%\build-cuda-vs22\tests\test_gemma4_cuda.exe"
set "SP_G4_KAIROS_SOAK=1"
set "SP_G4_KV_RING_W=1024"
set "SP_G4_KV_JMAX=160"
set "SP_CUDA_DECODE_INT8=1"
set "SP_SOAK_LOG=%SP_ENGINE%\results\kairos_soak_detail.log"
if not "%1"=="" set "SP_SOAK_HOURS=%1"
if not "%2"=="" set "SP_SOAK_MAXLOOPS=%2"
echo ===== [%DATE% %TIME%] G-KAIROS-1 soak hours=%SP_SOAK_HOURS% maxloops=%SP_SOAK_MAXLOOPS% =====
"%EXE%"
echo ===== [%DATE% %TIME%] SOAK_EXIT=%ERRORLEVEL% (0=GREEN 2=err 3=abort) =====
nvidia-smi --reset-gpu-clocks
