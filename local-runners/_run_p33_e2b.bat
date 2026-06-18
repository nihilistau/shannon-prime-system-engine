@echo off
call "D:\F\shannon-prime-repos\shannon-prime-system-engine\scripts\env\env-common.bat"
set "PATH=%SP_PIN_CUDA_ROOT%\bin;%PATH%"
set "SP_GEMMA4_SPMODEL=D:/F/shannon-prime-repos/models/gemma4-e2b.sp-model"
set "SP_GEMMA4_SPTOK=D:/F/shannon-prime-repos/models/gemma4-e2b.sp-tokenizer"
set "SP_G4_REPLAY_GATE=1"
set "SP_REPLAY_DIR=%SP_ENGINE%\_p33_ep_e2b"
if not exist "%SP_REPLAY_DIR%" mkdir "%SP_REPLAY_DIR%"
echo [run] P3.3 SP_REPLAY gate on E2B (15 owners / 20 sharers)
"%SP_ENGINE%\build-cuda-vs22\tests\test_gemma4_cuda.exe"
echo TEST_EXIT=%ERRORLEVEL%
