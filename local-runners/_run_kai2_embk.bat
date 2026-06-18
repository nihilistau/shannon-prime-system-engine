@echo off
call "%~dp0scripts\env\env-common.bat"
set "PATH=%SP_PIN_CUDA_ROOT%\bin;%PATH%"
nvidia-smi --lock-gpu-clocks=1680 >nul
set "SP_GEMMA4_SPMODEL=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-model"
set "SP_GEMMA4_SPTOK=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-tokenizer"
set "SP_CUDA_DECODE_INT8=1"
set "SP_G4_KAI2_PACKET=1"
cd /d "%SP_ENGINE%"
for %%K in (4 8 16 25) do (
  echo ===== EMBK=%%K =====
  set "SP_KAI2_EMBK=%%K"
  "%SP_ENGINE%\build-cuda-vs22\tests\test_gemma4_cuda.exe" 2>&1
)
nvidia-smi --reset-gpu-clocks >nul
