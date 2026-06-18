@echo off
REM §3q ORACLE CEILING G2: exact top-B by q.K (the best any selector can do) vs FULL.
REM If oracle 8x deflection >= 2.0% => 8x is information-bounded on gemma globals,
REM no learned head recovers it, concede 4x. If < 2.0% => headroom exists, train the head.
call "%~dp0scripts\env\env-common.bat"
set "PATH=%SP_PIN_CUDA_ROOT%\bin;%PATH%"
set "SP_GEMMA4_SPMODEL=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-model"
set "SP_GEMMA4_SPTOK=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-tokenizer"
set "SP_PPL_TOKENS=%SP_ENGINE%\tests\fixtures\ppl\wiki.valid.g4tokens.txt"
set "SP_PPL_NCTX=2048"
set "SP_PPL_CHUNKS=3"
set "EXE=%SP_ENGINE%\build-cuda-vs22\tests\test_gemma4_ppl_cuda.exe"
echo ===== [%TIME%] FULL baseline (shadow OFF) =====
"%EXE%"
echo ===== [%TIME%] ORACLE 8x (exact top-256 by q.K) =====
set "SP_ARM_SHADOW=1"
set "SP_ARM_GATHER=1"
set "SP_ARM_ORACLE=1"
set "SP_ARM_B=256"
"%EXE%"
echo ===== [%TIME%] ORACLE 4x (exact top-512 by q.K) =====
set "SP_ARM_B=512"
"%EXE%"
echo ===== [%TIME%] ORACLE_EXIT=%ERRORLEVEL% =====
