@echo off
REM C-c NIAH (G-P3-R2.b-2c-NIAH) — needle survives the O(1) compaction.
REM Usage: _run_niah_cc.bat <A|B|C> [depth] [N]
REM   A = BASELINE: globals FULL + SWA ring (fits at 16k); model must retrieve  -> HIT expected
REM   B = NEG CTRL: slab + FROZEN router (no LSH); un-selected needle absent     -> MISS expected
REM   C = GATE:     slab + LSH r=32 (8x); learned router selects needle into slab -> HIT = pass
REM SWA-isolation (needle_end <= n_prompt-W) is asserted IN the harness (aborts if violated).
call "%~dp0scripts\env\env-common.bat"
set "PATH=%SP_PIN_CUDA_ROOT%\bin;%PATH%"
nvidia-smi --lock-gpu-clocks=1680 >nul 2>&1
nvidia-smi --lock-memory-clocks=7000 >nul 2>&1
set "SP_GEMMA4_SPMODEL=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-model"
set "SP_GEMMA4_SPTOK=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-tokenizer"
set "SP_PPL_TOKENS=%SP_ENGINE%\tests\fixtures\ppl\wiki.valid.g4tokens.txt"
set "EXE=%SP_ENGINE%\build-cuda-vs22\tests\test_gemma4_cuda.exe"
set "SP_G4_NIAH=1"
set "SP_CUDA_DECODE_INT8=1"
set "SP_NIAH_N=%3"
if "%3"=="" set "SP_NIAH_N=16384"
set "SP_NIAH_DEPTH=%2"
if "%2"=="" set "SP_NIAH_DEPTH=50"
set "SP_NIAH_SECRET=837492"
set "SP_NIAH_GEN=24"
REM clear all XBAR knobs so each condition is clean
set "SP_XBAR_SWA_RING=" & set "SP_ARM_SHADOW=" & set "SP_ARM_GATHER=" & set "SP_ARM_B="
set "SP_ARM_SLAB=" & set "SP_ARM_BSLAB=" & set "SP_ARM_LSH=" & set "SP_ARM_LSH_R="
if /I "%1"=="A" (
  set "SP_XBAR_SWA_RING=1"
)
if /I "%1"=="B" (
  set "SP_XBAR_SWA_RING=1" & set "SP_ARM_SHADOW=1" & set "SP_ARM_GATHER=1" & set "SP_ARM_B=256"
  set "SP_ARM_SLAB=1" & set "SP_ARM_BSLAB=4400"
)
if /I "%1"=="C" (
  set "SP_XBAR_SWA_RING=1" & set "SP_ARM_SHADOW=1" & set "SP_ARM_GATHER=1" & set "SP_ARM_B=256"
  set "SP_ARM_SLAB=1" & set "SP_ARM_BSLAB=4400"
  set "SP_ARM_LSH=D:\F\shannon-prime-repos\_xbar\p2b\lsh_M_r32.bin"
  set "SP_ARM_LSH_R=D:\F\shannon-prime-repos\_xbar\p2b\lsh_R_r32_raw.bin"
)
echo ===== [%TIME%] C-c NIAH cond=%1 depth=%SP_NIAH_DEPTH% N=%SP_NIAH_N% =====
"%EXE%"
echo ===== [%TIME%] NIAH_EXIT=%ERRORLEVEL% (0=HIT 3=MISS 2=err) =====
nvidia-smi --reset-gpu-clocks >nul 2>&1
nvidia-smi --reset-memory-clocks >nul 2>&1
