@echo off
REM C-b.2 VRAM gate: globals -> compact slab (SP_ARM_BSLAB), SWA -> model-window ring (bit-exact).
REM Both KV terms become CONSTANT in context => nvidia-smi flat-line beneath N.
REM Usage: _run_g2_cb2_vram.bat <NCTX> <BSLAB> <CHUNKS> <logtag>
REM   e.g. _run_g2_cb2_vram.bat 2048 4400 1 smoke   (de-risk: chunk1 must == slab-only 5.4086)
REM        _run_g2_cb2_vram.bat 8192 4400 1 8k      (the cash-out: union must cap ~nh*B=4096, clip=0)
call "%~dp0scripts\env\env-common.bat"
set "PATH=%SP_PIN_CUDA_ROOT%\bin;%PATH%"
REM measurement hygiene (feedback_pin_clocks_for_tests): lock GPU clocks so timing is
REM reproducible; best-effort (needs admin; GeForce mem-clock lock may be unavailable).
REM VRAM-alloc + PPL are clock-invariant, but pin anyway for a single harness convention.
nvidia-smi --lock-gpu-clocks=1680 >nul 2>&1
nvidia-smi --lock-memory-clocks=7000 >nul 2>&1
set "SP_GEMMA4_SPMODEL=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-model"
set "SP_GEMMA4_SPTOK=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-tokenizer"
set "SP_PPL_TOKENS=%SP_ENGINE%\tests\fixtures\ppl\wiki.valid.g4tokens.txt"
set "SP_PPL_NCTX=%1"
set "SP_PPL_CHUNKS=%3"
set "EXE=%SP_ENGINE%\build-cuda-vs22\tests\test_gemma4_ppl_cuda.exe"
REM --- global sparse recall (LSH r=32) ---
set "SP_ARM_SHADOW=1"
set "SP_ARM_GATHER=1"
set "SP_ARM_B=256"
set "SP_ARM_LSH=D:\F\shannon-prime-repos\_xbar\p2b\lsh_M_r32.bin"
set "SP_ARM_LSH_R=D:\F\shannon-prime-repos\_xbar\p2b\lsh_R_r32_raw.bin"
REM --- C-b.2 compact global slab ---
set "SP_ARM_SLAB=1"
set "SP_ARM_BSLAB=%2"
REM --- SWA ring at the model's true window (no W override => bit-exact) ---
set "SP_XBAR_SWA_RING=1"
echo ===== [%TIME%] C-b.2 VRAM gate NCTX=%1 BSLAB=%2 CHUNKS=%3 (slab globals + model-SW ring) =====
"%EXE%"
echo ===== [%TIME%] VRAM_EXIT=%ERRORLEVEL% =====
nvidia-smi --reset-gpu-clocks >nul 2>&1
nvidia-smi --reset-memory-clocks >nul 2>&1
