@echo off
REM §3q Phase A FULL dump: post-RoPE per-position (K,q) on the 8 globals over the 3 N=2048
REM wikitext-2 windows = the per-position addresser training corpus. One kq_call<N>.bin per window.
call "%~dp0scripts\env\env-common.bat"
set "PATH=%SP_PIN_CUDA_ROOT%\bin;%PATH%"
set "SP_GEMMA4_SPMODEL=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-model"
set "SP_GEMMA4_SPTOK=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-tokenizer"
set "SP_PPL_TOKENS=%SP_ENGINE%\tests\fixtures\ppl\wiki.valid.g4tokens.txt"
set "SP_PPL_NCTX=2048"
set "SP_PPL_CHUNKS=3"
set "SP_ARM_SHADOW=1"
set "SP_ARM_DUMP=D:\F\shannon-prime-repos\_xbar\p2b\kqdump3w"
if exist "%SP_ARM_DUMP%" rmdir /s /q "%SP_ARM_DUMP%"
mkdir "%SP_ARM_DUMP%"
echo ===== [%TIME%] DUMP START (N=2048 x3, 8 globals) =====
"%SP_ENGINE%\build-cuda-vs22\tests\test_gemma4_ppl_cuda.exe"
echo ===== [%TIME%] DUMP_EXIT=%ERRORLEVEL% =====
dir "%SP_ARM_DUMP%"
