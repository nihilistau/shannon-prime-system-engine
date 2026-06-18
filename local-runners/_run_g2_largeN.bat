@echo off
REM ===== Larger-N G2: sparse-recall PPL deflection at the locked N<=2k ceiling =====
REM Self-referential gate: deflection = ppl_gather / ppl_full - 1  vs LOCKED < 2.0%.
REM Corpus: wikitext-2 validation (full), 20000-token SP dump. N=2048 x 3 independent
REM windows = 3072 scored positions (~73x the earlier 42-position read).
REM B is an absolute global-key budget selected from P=n_ctx candidates -> compression = N/B:
REM   4x -> B=512 ;  8x -> B=256.   W=4 sink=2 R=32 held fixed (the frozen geom router).
call "%~dp0scripts\env\env-common.bat"
set "PATH=%SP_PIN_CUDA_ROOT%\bin;%PATH%"
set "SP_GEMMA4_SPMODEL=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-model"
set "SP_GEMMA4_SPTOK=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-tokenizer"
set "SP_PPL_TOKENS=%SP_ENGINE%\tests\fixtures\ppl\wiki.valid.g4tokens.txt"
set "SP_PPL_NCTX=2048"
set "SP_PPL_CHUNKS=3"
set "EXE=%SP_ENGINE%\build-cuda-vs22\tests\test_gemma4_ppl_cuda.exe"
echo ===== [%TIME%] FULL baseline (shadow OFF, full cache) =====
"%EXE%"
echo ===== [%TIME%] GATHER 4x (B=512 W=4 sink=2 R=32) =====
set "SP_ARM_SHADOW=1"
set "SP_ARM_GATHER=1"
set "SP_ARM_W=4"
set "SP_ARM_SINK=2"
set "SP_ARM_R=32"
set "SP_ARM_B=512"
"%EXE%"
echo ===== [%TIME%] GATHER 8x (B=256 W=4 sink=2 R=32) =====
set "SP_ARM_B=256"
"%EXE%"
echo ===== [%TIME%] ALL_EXIT=%ERRORLEVEL% =====
