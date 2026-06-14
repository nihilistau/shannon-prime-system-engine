@echo off
REM P3.2-a shadow-spill byte-identity gate (G-P3-R2.a). The gate fires INSIDE
REM gemma4_decode_cuda whenever SP_XBAR_SPILL=dir is set: per-step owner K/V is
REM spilled to the Ring-2 stdio store, then at download: the store is read back
REM and memcmp'd vs the final live cache. Look for "G-P3-R2.a byte-identity ... PASS".
call "%~dp0scripts\env\env-common.bat"
set "PATH=%SP_PIN_CUDA_ROOT%\bin;%PATH%"
set "SP_GEMMA4_SPMODEL=D:/F/shannon-prime-repos/models/gemma4-12b.sp-model"
set "SP_GEMMA4_SPTOK=D:/F/shannon-prime-repos/models/gemma4-12b.sp-tokenizer"
set "SP_XBAR_SPILL=%SP_ENGINE%\_p32_spill"
if not exist "%SP_XBAR_SPILL%" mkdir "%SP_XBAR_SPILL%"
echo [run] SPILL=%SP_XBAR_SPILL%
"%SP_ENGINE%\build-cuda-vs22\tests\test_gemma4_cuda.exe"
echo TEST_EXIT=%ERRORLEVEL%
