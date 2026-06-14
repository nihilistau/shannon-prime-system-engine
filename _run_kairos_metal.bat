@echo off
REM KAI-1c: the semantic KAIROS loop on the journaled-ring metal (commit-on-ACTION, rewind-on-NO_OP).
call "%~dp0scripts\env\env-common.bat"
set "PATH=%SP_PIN_CUDA_ROOT%\bin;%PATH%"
nvidia-smi --lock-gpu-clocks=1680
REM (2060 cannot lock memory clock; core lock is all we get. Cognition is clock-invariant anyway.)
set "SP_GEMMA4_SPMODEL=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-model"
set "SP_GEMMA4_SPTOK=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-tokenizer"
set "SP_KAIROS_TAPE=%SP_ENGINE%\tools\sp_daemon\tests\fixtures\kairos\tape_smoke.txt"
set "EXE=%SP_ENGINE%\build-cuda-vs22\tests\test_gemma4_cuda.exe"
set "SP_G4_KAIROS_METAL=1"
set "SP_G4_KV_RING_W=1024"
set "SP_G4_KV_JMAX=160"
set "SP_CUDA_DECODE_INT8=1"
echo ===== [%TIME%] KAI-1c run_kairos_metal (journaled ring W=%SP_G4_KV_RING_W%) =====
"%EXE%"
echo ===== [%TIME%] KMETAL_EXIT=%ERRORLEVEL% (0=GREEN 3=RED 2=err) =====
nvidia-smi --reset-gpu-clocks
