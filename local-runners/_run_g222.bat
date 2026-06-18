@echo off
REM C2 #222 G-222: replay-inject into the persistent gemma4_kv_* cache + O(1) bit-exact rewind.
REM E2B fail-fast, then 12B. SP_CUDA_DECODE_INT8=1 (tied-head kv requirement).
call "%~dp0scripts\env\env-common.bat"
set "PATH=%SP_PIN_CUDA_ROOT%\bin;%PATH%"
set "SP_G4_KV_REPLAY_GATE=1"
set "SP_CUDA_DECODE_INT8=1"
set "SP_REPLAY_NPOS=8"
set "OUT=%SP_ENGINE%\_g222"
if not exist "%OUT%" mkdir "%OUT%"

echo ===== G-222 LEG E2B (15 owners/20 sharers) =====
set "SP_GEMMA4_SPMODEL=D:/F/shannon-prime-repos/models/gemma4-e2b.sp-model"
set "SP_GEMMA4_SPTOK=D:/F/shannon-prime-repos/models/gemma4-e2b.sp-tokenizer"
set "SP_REPLAY=%SP_ENGINE%\_p33_ep_e2b"
"%SP_ENGINE%\build-cuda-vs22\tests\test_gemma4_cuda.exe" > "%OUT%\g222_e2b.log" 2>&1
echo E2B_EXIT=%ERRORLEVEL%

echo ===== G-222 LEG 12B (48 owners) =====
set "SP_GEMMA4_SPMODEL=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-model"
set "SP_GEMMA4_SPTOK=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-tokenizer"
set "SP_REPLAY=%SP_ENGINE%\_p33_ep"
"%SP_ENGINE%\build-cuda-vs22\tests\test_gemma4_cuda.exe" > "%OUT%\g222_12b.log" 2>&1
echo TWELVEB_EXIT=%ERRORLEVEL%

echo G222_DONE > "%OUT%\DONE.flag"
