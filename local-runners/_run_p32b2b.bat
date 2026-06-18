@echo off
REM P3.2-b-2b Phase 0/1 shadow-router parity gate. SP_ARM_GATE=1 runs a no-flag
REM baseline decode, then shadow decodes at B=0 (null floor) and B=8 (sparse) — both
REM must be byte-identical to baseline (attention untouched) AND the global projk must
REM match a fresh reprojection of the final cache (G-P3-GEOM.a). Look for "diffs=0",
REM "projk-mism=0", and the PASS lines.
call "%~dp0scripts\env\env-common.bat"
set "PATH=%SP_PIN_CUDA_ROOT%\bin;%PATH%"
set "SP_GEMMA4_SPMODEL=D:/F/shannon-prime-repos/models/gemma4-12b.sp-model"
set "SP_GEMMA4_SPTOK=D:/F/shannon-prime-repos/models/gemma4-12b.sp-tokenizer"
set "SP_ARM_GATE=1"
echo [run] ARM_GATE=1 (Phase 0/1 shadow-router oracle parity, live decode bit-exact)
"%SP_ENGINE%\build-cuda-vs22\tests\test_gemma4_cuda.exe"
echo TEST_EXIT=%ERRORLEVEL%
