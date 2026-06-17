@echo off
REM G-R2-FROB-PARITY: Frobenius pi^k integer episode round-trip vs float, via SP_REPLAY deflection on 12B.
REM baseline (no replay) vs HIT float ep_wiki vs HIT frob16 vs HIT frob8 -- same NPOS=16 wiki footprint.
call "%~dp0scripts\env\env-common.bat"
set "PATH=%SP_PIN_CUDA_ROOT%\bin;%PATH%"
set "SP_GEMMA4_SPMODEL=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-model"
set "SP_GEMMA4_SPTOK=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-tokenizer"
set "SP_PPL_TOKENS=%SP_ENGINE%\tests\fixtures\ppl\wiki.tiny.g4tokens.txt"
set "SP_PPL_NCTX=84"
set "SP_PPL_CHUNKS=1"
set "SP_REPLAY_NPOS=16"
set "OUT=%SP_ENGINE%\_g_r2_frob"
if exist "%OUT%" rmdir /s /q "%OUT%"
mkdir "%OUT%"
set "EXE=%SP_ENGINE%\build-cuda-vs22\tests\test_gemma4_ppl_cuda.exe"

echo ===== baseline (no replay) =====
set "SP_REPLAY="
"%EXE%" > "%OUT%\ppl_base.log" 2>&1
echo BASE_EXIT=%ERRORLEVEL%

echo ===== HIT float ep_wiki (reference, proven +0.000%%) =====
set "SP_REPLAY=%SP_ENGINE%\_c2_ep_wiki"
"%EXE%" > "%OUT%\ppl_float.log" 2>&1
echo FLOAT_EXIT=%ERRORLEVEL%

echo ===== HIT frob int16 =====
set "SP_REPLAY=%SP_ENGINE%\_c2_ep_wiki_frob16"
"%EXE%" > "%OUT%\ppl_frob16.log" 2>&1
echo FROB16_EXIT=%ERRORLEVEL%

echo ===== HIT frob int8 =====
set "SP_REPLAY=%SP_ENGINE%\_c2_ep_wiki_frob8"
"%EXE%" > "%OUT%\ppl_frob8.log" 2>&1
echo FROB8_EXIT=%ERRORLEVEL%

set "SP_REPLAY="
echo G_R2_FROB_DONE > "%OUT%\DONE.flag"
