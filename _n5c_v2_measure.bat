@echo off
set "ENGINE=D:\F\shannon-prime-repos\shannon-prime-system-engine"
call "%ENGINE%\scripts\env\env-cuda.bat" >nul 2>&1
cd /d "%ENGINE%"
set "EXE=%ENGINE%\build-cuda\tests\test_dg_forward_timing.exe"
set "LOG=%ENGINE%\_n5c_v2_measure.log"
set SP_DGT_NFWD=3
set SP_DGT_CANVAS=8

echo N5C_V2_MEASURE_START %DATE% %TIME% > "%LOG%"
REM --- pin SM clock (mem-clock lock unsupported on this WDDM consumer GPU; warmup carries mem settling) ---
nvidia-smi -lgc 2100,2100 >> "%LOG%" 2>&1
nvidia-smi --query-gpu=clocks.sm,clocks.mem --format=csv,noheader >> "%LOG%" 2>&1

echo. >> "%LOG%"
echo ===== n_tok=14 (decode width)  f32 (SP_DG_PACKED=0) ===== >> "%LOG%"
set SP_DGT_NTOK=14
set SP_DG_PACKED=0
"%EXE%" >> "%LOG%" 2>&1

echo. >> "%LOG%"
echo ===== n_tok=14 (decode width)  v2-packed (SP_DG_PACKED=1) ===== >> "%LOG%"
set SP_DGT_NTOK=14
set SP_DG_PACKED=1
"%EXE%" >> "%LOG%" 2>&1

echo. >> "%LOG%"
echo ===== n_tok=1852 (prefill width)  f32 (SP_DG_PACKED=0) ===== >> "%LOG%"
set SP_DGT_NTOK=1852
set SP_DG_PACKED=0
"%EXE%" >> "%LOG%" 2>&1

echo. >> "%LOG%"
echo ===== n_tok=1852 (prefill width)  v2-packed (SP_DG_PACKED=1) ===== >> "%LOG%"
set SP_DGT_NTOK=1852
set SP_DG_PACKED=1
"%EXE%" >> "%LOG%" 2>&1

nvidia-smi -rgc >> "%LOG%" 2>&1
echo N5C_V2_MEASURE_DONE %DATE% %TIME% >> "%LOG%"
