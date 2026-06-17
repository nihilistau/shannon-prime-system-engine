@echo off
REM C2 Step 3.0 — G-MEMO-NULL: prove the curator's cue-extraction seam is INERT WHEN OFF.
REM LEG A = baseline PPL (no observer).  LEG B = SP_ARM_SHADOW + SP_ARM_DUMP (the per-tick cue extraction).
REM Gate (curator_loop.py --null): PPL_A == PPL_B bit-for-bit  AND  dump fired  AND  empty-registry resolve = NULL.
call "%~dp0scripts\env\env-common.bat"
set "PATH=%SP_PIN_CUDA_ROOT%\bin;%PATH%"
set "SP_GEMMA4_SPMODEL=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-model"
set "SP_GEMMA4_SPTOK=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-tokenizer"
set "SP_PPL_TOKENS=%SP_ENGINE%\tests\fixtures\ppl\wiki.tiny.g4tokens.txt"
set "SP_PPL_NCTX=84"
set "SP_PPL_CHUNKS=1"
set "OUT=%SP_ENGINE%\_memo_null"
if not exist "%OUT%" mkdir "%OUT%"
set "DUMP=%OUT%\cue_dump"
if exist "%DUMP%" rmdir /s /q "%DUMP%"
mkdir "%DUMP%"

echo ===== G-MEMO-NULL LEG A: baseline (curator OFF, no observer) =====
"%SP_ENGINE%\build-cuda-vs22\tests\test_gemma4_ppl_cuda.exe" > "%OUT%\ppl_base.log" 2>&1
echo LEG_A_EXIT=%ERRORLEVEL%

echo ===== G-MEMO-NULL LEG B: cue-extraction ON (SP_ARM_SHADOW + SP_ARM_DUMP) =====
set "SP_ARM_SHADOW=1"
set "SP_ARM_DUMP=%DUMP%"
"%SP_ENGINE%\build-cuda-vs22\tests\test_gemma4_ppl_cuda.exe" > "%OUT%\ppl_dump.log" 2>&1
echo LEG_B_EXIT=%ERRORLEVEL%
set "SP_ARM_SHADOW="
set "SP_ARM_DUMP="

echo MEMO_NULL_DONE > "%OUT%\DONE.flag"
echo done.
