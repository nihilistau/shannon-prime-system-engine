@echo off
set "ENGINE=D:\F\shannon-prime-repos\shannon-prime-system-engine"
call "%ENGINE%\scripts\env\env-cuda.bat" >nul 2>&1
cd /d "%ENGINE%"
set "EXE=%ENGINE%\build-cuda\tests\test_dg_forward_timing.exe"
set "LOG=%ENGINE%\_n5c_wcache_gate.log"
set SP_DGT_NFWD=3
set SP_DGT_CANVAS=8
set SP_DG_PACKED=1

echo N5C_WCACHE_GATE_START %DATE% %TIME% > "%LOG%"
nvidia-smi -lgc 2100,2100 >> "%LOG%" 2>&1

echo. >> "%LOG%"
echo ===== n_tok=1852  v2 baseline (WCACHE OFF): fwd1/2/3 should be FLAT ===== >> "%LOG%"
set SP_DGT_NTOK=1852
set SP_DG_WCACHE=0
set SP_DG_TRACE=
"%EXE%" >> "%LOG%" 2>&1

echo. >> "%LOG%"
echo ===== n_tok=1852  WCACHE ON: fwd1 cold (populate), fwd2/3 warm (dense H2D eliminated) ===== >> "%LOG%"
set SP_DGT_NTOK=1852
set SP_DG_WCACHE=1
set SP_DG_TRACE=1
"%EXE%" >> "%LOG%" 2>&1

echo. >> "%LOG%"
echo ===== n_tok=14   WCACHE ON: decode-width picture ===== >> "%LOG%"
set SP_DGT_NTOK=14
set SP_DG_WCACHE=1
set SP_DG_TRACE=1
"%EXE%" >> "%LOG%" 2>&1

nvidia-smi -rgc >> "%LOG%" 2>&1
echo N5C_WCACHE_GATE_DONE %DATE% %TIME% >> "%LOG%"
