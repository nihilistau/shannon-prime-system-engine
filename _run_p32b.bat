@echo off
REM P3.2-b-1 paged-read bit-exact gate (G-P3-R2.b-1). SP_XBAR_PAGE_GATE=1 runs a
REM legacy full-cache decode vs an SP_XBAR_PAGE decode (per step: spill pos ->
REM poison [0,pos] in the live cache -> page [0,pos) back off Ring-2 before
REM attention). Look for "P3.2-b-1 paged-read ... PASS" and diffs=0.
call "%~dp0scripts\env\env-common.bat"
set "PATH=%SP_PIN_CUDA_ROOT%\bin;%PATH%"
set "SP_GEMMA4_SPMODEL=D:/F/shannon-prime-repos/models/gemma4-12b.sp-model"
set "SP_GEMMA4_SPTOK=D:/F/shannon-prime-repos/models/gemma4-12b.sp-tokenizer"
set "SP_XBAR_PAGE_GATE=1"
set "SP_XBAR_PAGE_DIR=%SP_ENGINE%\_p32_page"
if not exist "%SP_XBAR_PAGE_DIR%" mkdir "%SP_XBAR_PAGE_DIR%"
echo [run] PAGE_GATE=1 DIR=%SP_XBAR_PAGE_DIR%
"%SP_ENGINE%\build-cuda-vs22\tests\test_gemma4_cuda.exe"
echo TEST_EXIT=%ERRORLEVEL%
