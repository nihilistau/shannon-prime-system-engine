@echo off
call "%~dp0scripts\env\env-common.bat"
set "PATH=%SP_PIN_CUDA_ROOT%\bin;%PATH%"
set "SP_GEMMA4_SPMODEL=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-model"
set "SP_GEMMA4_SPTOK=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-tokenizer"
set "SP_PPL_TOKENS=%SP_ENGINE%\tests\fixtures\ppl\wiki.tiny.g4tokens.txt"
set "SP_PPL_NCTX=84"
set "SP_PPL_CHUNKS=1"
echo ===== P3.4 LEG A: recall-OFF baseline =====
"%SP_ENGINE%\build-cuda-vs22\tests\test_gemma4_ppl_cuda.exe"
echo ===== P3.4 LEG B: recall-ON (SP_REPLAY proven episode, NPOS=4) =====
set "SP_REPLAY=%SP_ENGINE%\_p33_ep"
set "SP_REPLAY_NPOS=4"
"%SP_ENGINE%\build-cuda-vs22\tests\test_gemma4_ppl_cuda.exe"
set "SP_REPLAY="
set "SP_REPLAY_NPOS="
echo P34_EXIT=%ERRORLEVEL%
