@echo off
REM P3.3 SP_REPLAY injection gate (G-P3-SHARED). Fires inside test_gemma4_cuda when
REM SP_G4_REPLAY_GATE=1: ref (floor) -> WRITE episode -> replay-intact -> replay-zero.
REM PASS = intact bit-identical to ref AND zeroed diverges. The E2B-class fast fixture
REM is the tiny 4+12 token sequence {2,10,100,1000}; model = whatever SP_GEMMA4_SPMODEL.
call "%~dp0scripts\env\env-common.bat"
set "PATH=%SP_PIN_CUDA_ROOT%\bin;%PATH%"
set "SP_GEMMA4_SPMODEL=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-model"
set "SP_GEMMA4_SPTOK=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-tokenizer"
set "SP_G4_REPLAY_GATE=1"
set "SP_REPLAY_DIR=%SP_ENGINE%\_p33_ep"
if not exist "%SP_REPLAY_DIR%" mkdir "%SP_REPLAY_DIR%"
echo [run] P3.3 SP_REPLAY gate, episode dir=%SP_REPLAY_DIR%
"%SP_ENGINE%\build-cuda-vs22\tests\test_gemma4_cuda.exe"
echo TEST_EXIT=%ERRORLEVEL%
