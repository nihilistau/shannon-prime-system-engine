@echo off
REM §3q C-b.1: projected-key SIDECAR select (SP_ARM_LSH_R) — score RᵀK (resident r-dim)
REM instead of M-transform over full K. GATE: output-invariant == 5.1791 (proves selection
REM works off the minimal router state, the prerequisite for C-b.2 compact-slab alloc-shrink).
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
set "SP_ARM_LSH_R=D:\F\shannon-prime-repos\_xbar\p2b\lsh_R_r32_raw.bin"
set "SP_ARM_DEVSEL=1"
echo ===== [%TIME%] C-b.1 sidecar select (r=32) 8x =====
"%EXE%"
echo ===== [%TIME%] CB1_EXIT=%ERRORLEVEL% =====
