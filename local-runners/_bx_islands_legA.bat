@echo off
call "%~dp0scripts\env\env-common.bat"
set "PATH=%SP_PIN_CUDA_ROOT%\bin;%PATH%"
set "SP_GEMMA4_SPMODEL=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-model"
set "SP_GEMMA4_SPTOK=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-tokenizer"
set "SP_PPL_TOKENS=%SP_ENGINE%\tests\fixtures\ppl\wiki.tiny.g4tokens.txt"
set "SP_PPL_NCTX=84"
set "SP_PPL_CHUNKS=1"
echo ===== LEG A: SP_BYTEEXACT unset (null floor) =====
set "SP_BYTEEXACT="
"%SP_ENGINE%\build-cuda-vs22\tests\test_gemma4_ppl_cuda.exe"
echo LEGA_EXIT=%ERRORLEVEL%
