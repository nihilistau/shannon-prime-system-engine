@echo off
REM KAI-3 §7.3 G-KAIROS-3: projected-frame pivot gate on the resident 12B.
call "%~dp0scripts\env\env-common.bat"
set "PATH=%SP_PIN_CUDA_ROOT%\bin;%PATH%"
nvidia-smi --lock-gpu-clocks=1680
set "SP_GEMMA4_SPMODEL=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-model"
set "SP_GEMMA4_SPTOK=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-tokenizer"
set "SP_CUDA_DECODE_INT8=1"
set "SP_G4_KAI3=D:\F\shannon-prime-repos\_xbar\p2b\kai3\manifest.txt"
cd /d "%SP_ENGINE%"
"%SP_ENGINE%\build-cuda-vs22\tests\test_gemma4_cuda.exe"
echo KAI3_GATE_EXIT=%ERRORLEVEL%
nvidia-smi --reset-gpu-clocks
