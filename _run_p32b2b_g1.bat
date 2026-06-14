@echo off
REM P3.2-b-2b G1: recall set served OFF RING-2 (NaN-poison rigor). SP_ARM_PAGE_GATE=1
REM runs gather-from-LIVE vs gather-from-DISK (spill + NaN-poison live globals + page the
REM recalled union off Ring-2). diffs=0 => every recalled byte served byte-exact off disk.
call "%~dp0scripts\env\env-common.bat"
set "PATH=%SP_PIN_CUDA_ROOT%\bin;%PATH%"
set "SP_GEMMA4_SPMODEL=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-model"
set "SP_GEMMA4_SPTOK=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-tokenizer"
set "SP_ARM_PAGE_GATE=1"
set "SP_ARM_PAGE_DIR=%SP_ENGINE%\_p32_armpage"
if not exist "%SP_ARM_PAGE_DIR%" mkdir "%SP_ARM_PAGE_DIR%"
echo [run] ARM_PAGE_GATE=1 DIR=%SP_ARM_PAGE_DIR%
"%SP_ENGINE%\build-cuda-vs22\tests\test_gemma4_cuda.exe"
echo TEST_EXIT=%ERRORLEVEL%
