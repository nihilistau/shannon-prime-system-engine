@echo off
REM §3q Phase B G2: Learned-LSH (trained 512xr projection, M=R.Rt) vs FULL.
REM r=32 first (zero inference overhead vs v0 frozen router). PASS = deflection < 2.0% @ 8x.
REM frozen v0 was +4.17%; oracle ceiling -0.08%; LSH r=32 offline mass@256=0.893.
call "%~dp0scripts\env\env-common.bat"
set "PATH=%SP_PIN_CUDA_ROOT%\bin;%PATH%"
set "SP_GEMMA4_SPMODEL=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-model"
set "SP_GEMMA4_SPTOK=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-tokenizer"
set "SP_PPL_TOKENS=%SP_ENGINE%\tests\fixtures\ppl\wiki.valid.g4tokens.txt"
set "SP_PPL_NCTX=2048"
set "SP_PPL_CHUNKS=3"
set "EXE=%SP_ENGINE%\build-cuda-vs22\tests\test_gemma4_ppl_cuda.exe"
set "MFILE=%1"
if "%MFILE%"=="" set "MFILE=D:\F\shannon-prime-repos\_xbar\p2b\lsh_M_r32.bin"
echo ===== [%TIME%] FULL baseline =====
"%EXE%"
echo ===== [%TIME%] LSH 8x (B=256) M=%MFILE% =====
set "SP_ARM_SHADOW=1"
set "SP_ARM_GATHER=1"
set "SP_ARM_B=256"
set "SP_ARM_LSH=%MFILE%"
"%EXE%"
echo ===== [%TIME%] LSH_EXIT=%ERRORLEVEL% =====
