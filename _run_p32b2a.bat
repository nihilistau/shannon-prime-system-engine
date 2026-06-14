@echo off
REM P3.2-b-2a SWA ring-buffer shrink bit-exact gate (G-P3-R2.b-2a). With W=4 and
REM P=16 the ring wraps + evicts. The ring decode must be token-identical to the
REM full-cache decode at the same window. Look for "P3.2-b-2a SWA ring ... PASS".
call "%~dp0scripts\env\env-common.bat"
set "PATH=%SP_PIN_CUDA_ROOT%\bin;%PATH%"
set "SP_GEMMA4_SPMODEL=D:/F/shannon-prime-repos/models/gemma4-12b.sp-model"
set "SP_GEMMA4_SPTOK=D:/F/shannon-prime-repos/models/gemma4-12b.sp-tokenizer"
set "SP_XBAR_SWA_GATE=1"
echo [run] SWA_GATE=1 (W defaults to 4, scoped to the gate decodes only)
"%SP_ENGINE%\build-cuda-vs22\tests\test_gemma4_cuda.exe"
echo TEST_EXIT=%ERRORLEVEL%
