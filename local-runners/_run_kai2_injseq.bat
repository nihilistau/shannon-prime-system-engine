@echo off
REM KAI-3 §7.2 G-KAIROS-3-NULL: sequence-wrapper null-floor gate (gemma4_kv_inject_seq).
call "%~dp0scripts\env\env-common.bat"
set "PATH=%SP_PIN_CUDA_ROOT%\bin;%PATH%"
nvidia-smi --lock-gpu-clocks=1680
set "SP_GEMMA4_SPMODEL=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-model"
set "SP_GEMMA4_SPTOK=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-tokenizer"
set "SP_CUDA_DECODE_INT8=1"
set "SP_G4_INJ_SEQ=1"
cd /d "%SP_ENGINE%"
"%SP_ENGINE%\build-cuda-vs22\tests\test_gemma4_cuda.exe"
echo INJSEQ_GATE_EXIT=%ERRORLEVEL%
nvidia-smi --reset-gpu-clocks
