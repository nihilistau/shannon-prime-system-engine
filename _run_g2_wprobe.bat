@echo off
REM ===== Larger-N G2 W-probe: can recent-global-tail anchoring tame the 8x deflection? =====
REM Same N=2048 x 3 windows, same fixture. Baseline already measured: FULL SP PPL=5.1551.
REM 8x = B=256 fixed (compression stays honest). W is the recent-window WITHIN B
REM   (global router keeps recent-W globals always); topK = B-W-sink projected keys.
REM   W=4  -> topK=250 : already measured at +4.17% (RED)
REM   W=64 -> topK=190 : THE DIRECTIVE (recent-64 globals anchored)
REM   W=128-> topK=126 : bracket (gradient)
REM deflection = ppl_gather / 5.1551 - 1  vs LOCKED < 2.0%.
call "%~dp0scripts\env\env-common.bat"
set "PATH=%SP_PIN_CUDA_ROOT%\bin;%PATH%"
set "SP_GEMMA4_SPMODEL=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-model"
set "SP_GEMMA4_SPTOK=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-tokenizer"
set "SP_PPL_TOKENS=%SP_ENGINE%\tests\fixtures\ppl\wiki.valid.g4tokens.txt"
set "SP_PPL_NCTX=2048"
set "SP_PPL_CHUNKS=3"
set "EXE=%SP_ENGINE%\build-cuda-vs22\tests\test_gemma4_ppl_cuda.exe"
set "SP_ARM_SHADOW=1"
set "SP_ARM_GATHER=1"
set "SP_ARM_SINK=2"
set "SP_ARM_R=32"
set "SP_ARM_B=256"
echo ===== [%TIME%] GATHER 8x  W=64  (B=256 sink=2 R=32 topK=190) =====
set "SP_ARM_W=64"
"%EXE%"
echo ===== [%TIME%] GATHER 8x  W=128 (B=256 sink=2 R=32 topK=126) =====
set "SP_ARM_W=128"
"%EXE%"
echo ===== [%TIME%] WPROBE_EXIT=%ERRORLEVEL% =====
