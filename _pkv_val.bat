@echo off
set "ENGINE=D:\F\shannon-prime-repos\shannon-prime-system-engine"
call "%ENGINE%\scripts\env\env-cuda.bat" >nul 2>&1
cd /d "%ENGINE%"
set SP_DJ_LIMIT=2
set SP_DJ_FLIMIT=2
set SP_DJ_STEPS=4
set SP_DJ_CANVAS=8
set SP_DJ_SEED=20260622
set SP_DG_PACKED=1
set "EXE=%ENGINE%\build-cuda\tests\test_diffjudge_denoise.exe"
echo PKV_VAL_START %DATE% %TIME% > "%ENGINE%\_pkv_val.log"
echo ================ baseline (SP_DG_PREFIXKV=0, full forward) ================ >> "%ENGINE%\_pkv_val.log"
set SP_DG_PREFIXKV=0
set "SP_DJ_OUT=%ENGINE%\tests\fixtures\chat_fullstack\G-DG-PREFIXKV-VAL-base.log"
"%EXE%" >> "%ENGINE%\_pkv_val.log" 2>&1
echo ================ prefix-KV fast (SP_DG_PREFIXKV=1, canvas-only steps 2..N) ================ >> "%ENGINE%\_pkv_val.log"
set SP_DG_PREFIXKV=1
set "SP_DJ_OUT=%ENGINE%\tests\fixtures\chat_fullstack\G-DG-PREFIXKV-VAL-fast.log"
"%EXE%" >> "%ENGINE%\_pkv_val.log" 2>&1
echo PKV_VAL_DONE %DATE% %TIME% >> "%ENGINE%\_pkv_val.log"
