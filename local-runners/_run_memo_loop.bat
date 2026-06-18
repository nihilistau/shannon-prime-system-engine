@echo off
REM C2 Step 3.1 — G-MEMO-LOOP: the curator's ACCEPT / REJECT branches on 12B metal.
REM SELECT (cue->episode_id) is proven offline by G-MEMO-CUE(discrete) and transfers online (order-immune gate).
REM LEG base   = baseline PPL.
REM LEG accept = SP_REPLAY ep_wiki (the MATCHED episode) over [0,42) -> expect ~0% deflection -> PROMOTE.
REM LEG reject = SP_REPLAY ep_wiki ZEROED (corrupted recall, P3.3 collapse) -> expect >=2% -> DISCARD.
call "%~dp0scripts\env\env-common.bat"
set "PATH=%SP_PIN_CUDA_ROOT%\bin;%PATH%"
set "SP_GEMMA4_SPMODEL=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-model"
set "SP_GEMMA4_SPTOK=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-tokenizer"
set "SP_PPL_TOKENS=%SP_ENGINE%\tests\fixtures\ppl\wiki.tiny.g4tokens.txt"
set "SP_PPL_NCTX=84"
set "SP_PPL_CHUNKS=1"
set "OUT=%SP_ENGINE%\_memo_loop"
if not exist "%OUT%" mkdir "%OUT%"

echo ===== LEG base: baseline =====
"%SP_ENGINE%\build-cuda-vs22\tests\test_gemma4_ppl_cuda.exe" > "%OUT%\ppl_base.log" 2>&1
echo BASE_EXIT=%ERRORLEVEL%

echo ===== LEG accept: SP_REPLAY ep_wiki (matched) NPOS=42 =====
set "SP_REPLAY=%SP_ENGINE%\_c2_ep_wiki"
set "SP_REPLAY_NPOS=42"
"%SP_ENGINE%\build-cuda-vs22\tests\test_gemma4_ppl_cuda.exe" > "%OUT%\ppl_accept.log" 2>&1
echo ACCEPT_EXIT=%ERRORLEVEL%

echo ===== LEG reject: SP_REPLAY ep_wiki ZEROED (corrupted recall) NPOS=42 =====
set "SP_REPLAY_ZERO=1"
"%SP_ENGINE%\build-cuda-vs22\tests\test_gemma4_ppl_cuda.exe" > "%OUT%\ppl_reject.log" 2>&1
echo REJECT_EXIT=%ERRORLEVEL%
set "SP_REPLAY="
set "SP_REPLAY_NPOS="
set "SP_REPLAY_ZERO="

echo MEMO_LOOP_DONE > "%OUT%\DONE.flag"
echo done.
