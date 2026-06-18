@echo off
REM §3q C-b.2 step 1: validate compact-slab MECHANICS + measure the per-step UNION size.
REM SP_ARM_SLAB=1 with default Bslab=P (safe, no shrink yet): globals spill to host RAM,
REM the union is paged into compact slots [0,m), gather reads compact-slot indices.
REM GATE: output-invariant == C-b.1 sidecar 5.1676 (+0.24%). Plus: read [xbar-slab] union telemetry
REM to set the real shrink depth (SP_ARM_BSLAB) for step 2.  raw log -> tracked results/.
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
set "SP_ARM_SLAB=1"
if not "%1"=="" set "SP_ARM_BSLAB=%1"
echo ===== [%TIME%] C-b.2 slab mechanics+union (Bslab=%SP_ARM_BSLAB%, default P) 8x =====
"%EXE%"
echo ===== [%TIME%] CB2_EXIT=%ERRORLEVEL% =====
