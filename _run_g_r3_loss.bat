@echo off
REM R3.2 G-R3-LOSS: the consolidation loss is a STEP FUNCTION (retrieve-and-verify, not lossy-gist).
REM HIT  = Ring-3 recalls the CORRECT id -> fetch+inject the matching verbatim episode (ep_wiki) -> ~0 loss.
REM MISS = Ring-3 recalls a WRONG id (capacity overflow) -> inject a foreign episode (ep_toy) -> fact lost.
REM Same NPOS=16 footprint for an apples-to-apples wrong-vs-right comparison on the wiki score.
call "%~dp0scripts\env\env-common.bat"
set "PATH=%SP_PIN_CUDA_ROOT%\bin;%PATH%"
set "SP_GEMMA4_SPMODEL=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-model"
set "SP_GEMMA4_SPTOK=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-tokenizer"
set "SP_PPL_TOKENS=%SP_ENGINE%\tests\fixtures\ppl\wiki.tiny.g4tokens.txt"
set "SP_PPL_NCTX=84"
set "SP_PPL_CHUNKS=1"
set "OUT=%SP_ENGINE%\_g_r3_loss"
if not exist "%OUT%" mkdir "%OUT%"

echo ===== baseline =====
"%SP_ENGINE%\build-cuda-vs22\tests\test_gemma4_ppl_cuda.exe" > "%OUT%\ppl_base.log" 2>&1
echo BASE_EXIT=%ERRORLEVEL%

echo ===== HIT: correct id -> verbatim ep_wiki (matched) NPOS=16 =====
set "SP_REPLAY=%SP_ENGINE%\_c2_ep_wiki"
set "SP_REPLAY_NPOS=16"
"%SP_ENGINE%\build-cuda-vs22\tests\test_gemma4_ppl_cuda.exe" > "%OUT%\ppl_hit.log" 2>&1
echo HIT_EXIT=%ERRORLEVEL%

echo ===== MISS: wrong id -> foreign ep_toy NPOS=16 =====
set "SP_REPLAY=%SP_ENGINE%\_p33_ep"
"%SP_ENGINE%\build-cuda-vs22\tests\test_gemma4_ppl_cuda.exe" > "%OUT%\ppl_miss.log" 2>&1
echo MISS_EXIT=%ERRORLEVEL%
set "SP_REPLAY="
set "SP_REPLAY_NPOS="

echo G_R3_LOSS_DONE > "%OUT%\DONE.flag"
