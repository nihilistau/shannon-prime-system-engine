@echo off
REM G-P3-R2.b-2b G2 (corrected): PPL deflection on the OK_Q4B artifact (-b1), whose
REM baseline is SANE (PPL 4.6665 @ n_ctx=84, == the gold 4.68). The plain gemma4-12b
REM path is the coarse per-row QAT variant (7.4M PPL) — DO NOT use it for PPL.
REM (1) FULL baseline (report-only). (2)/(3) GATHER B=21 (~4x) and B=11 (~8x).
REM deflection = ppl_gather/ppl_full - 1 vs the LOCKED < 2.0%.
call "%~dp0scripts\env\env-common.bat"
set "PATH=%SP_PIN_CUDA_ROOT%\bin;%PATH%"
set "SP_GEMMA4_SPMODEL=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-model"
set "SP_GEMMA4_SPTOK=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-tokenizer"
set "SP_PPL_TOKENS=%SP_ENGINE%\tests\fixtures\ppl\wiki.tiny.g4tokens.txt"
set "SP_PPL_NCTX=84"
set "SP_PPL_CHUNKS=1"
echo ===== FULL (baseline, OK_Q4B -b1, expect ~4.67) =====
"%SP_ENGINE%\build-cuda-vs22\tests\test_gemma4_ppl_cuda.exe"
echo ===== GATHER B=21 (~4x N=84, W=4 sink=2 r=32) =====
set "SP_ARM_SHADOW=1"
set "SP_ARM_GATHER=1"
set "SP_ARM_W=4"
set "SP_ARM_SINK=2"
set "SP_ARM_R=32"
set "SP_ARM_B=21"
"%SP_ENGINE%\build-cuda-vs22\tests\test_gemma4_ppl_cuda.exe"
echo ===== GATHER B=11 (~8x N=84) =====
set "SP_ARM_B=11"
"%SP_ENGINE%\build-cuda-vs22\tests\test_gemma4_ppl_cuda.exe"
echo ALL_EXIT=%ERRORLEVEL%
