@echo off
REM §3q C-a: device-side top-B select (SP_ARM_DEVSEL) on the LSH r=32 router.
REM GATE: G2 8x PPL == host-select 5.1791 (selection-invariant) AND faster (no sync tax).
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
set "SP_ARM_B=256"
set "SP_ARM_LSH=D:\F\shannon-prime-repos\_xbar\p2b\lsh_M_r32.bin"
set "SP_ARM_DEVSEL=1"
echo ===== [%TIME%] LSH r=32 + DEVSEL 8x (B=256) =====
"%EXE%"
echo ===== [%TIME%] DEVSEL_EXIT=%ERRORLEVEL% =====
