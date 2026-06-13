@echo off
REM §3q Phase A dump SMOKE: tiny N to validate the hook (file created, header sane, record sizes).
call "%~dp0scripts\env\env-common.bat"
set "PATH=%SP_PIN_CUDA_ROOT%\bin;%PATH%"
set "SP_GEMMA4_SPMODEL=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-model"
set "SP_GEMMA4_SPTOK=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-tokenizer"
set "SP_PPL_TOKENS=%SP_ENGINE%\tests\fixtures\ppl\wiki.valid.g4tokens.txt"
set "SP_PPL_NCTX=128"
set "SP_PPL_CHUNKS=1"
set "SP_ARM_SHADOW=1"
set "SP_ARM_DUMP=D:\F\shannon-prime-repos\_xbar\p2b\kqdump_smoke"
if not exist "%SP_ARM_DUMP%" mkdir "%SP_ARM_DUMP%"
"%SP_ENGINE%\build-cuda-vs22\tests\test_gemma4_ppl_cuda.exe"
echo SMOKE_EXIT=%ERRORLEVEL%
dir "%SP_ARM_DUMP%"
