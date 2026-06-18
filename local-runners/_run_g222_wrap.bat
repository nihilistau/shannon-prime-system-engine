@echo off
REM C2 #222 G-222-WRAP: replay-inject into an ACTIVE SWA-ring session + O(1) bit-exact rewind via the KAI-1c journal.
REM W=16 with anchor=24 forces the replay slots to ALIAS live earlier positions (wrap-crossing) -> journal exercised.
call "%~dp0scripts\env\env-common.bat"
set "PATH=%SP_PIN_CUDA_ROOT%\bin;%PATH%"
set "SP_G4_KV_REPLAY_GATE=1"
set "SP_CUDA_DECODE_INT8=1"
set "SP_REPLAY_NPOS=8"
set "SP_G4_KV_RING_W=16"
set "SP_G4_KV_JMAX=64"
set "OUT=%SP_ENGINE%\_g222wrap"
if not exist "%OUT%" mkdir "%OUT%"

echo ===== G-222-WRAP LEG E2B =====
set "SP_GEMMA4_SPMODEL=D:/F/shannon-prime-repos/models/gemma4-e2b.sp-model"
set "SP_GEMMA4_SPTOK=D:/F/shannon-prime-repos/models/gemma4-e2b.sp-tokenizer"
set "SP_REPLAY=%SP_ENGINE%\_p33_ep_e2b"
"%SP_ENGINE%\build-cuda-vs22\tests\test_gemma4_cuda.exe" > "%OUT%\wrap_e2b.log" 2>&1
echo E2B_EXIT=%ERRORLEVEL%

echo ===== G-222-WRAP LEG 12B =====
set "SP_GEMMA4_SPMODEL=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-model"
set "SP_GEMMA4_SPTOK=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-tokenizer"
set "SP_REPLAY=%SP_ENGINE%\_p33_ep"
"%SP_ENGINE%\build-cuda-vs22\tests\test_gemma4_cuda.exe" > "%OUT%\wrap_12b.log" 2>&1
echo TWELVEB_EXIT=%ERRORLEVEL%

echo G222WRAP_DONE > "%OUT%\DONE.flag"
