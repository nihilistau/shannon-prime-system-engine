@echo off
set "ENGINE=D:\F\shannon-prime-repos\shannon-prime-system-engine"
call "%ENGINE%\scripts\env\env-cuda.bat" >nul 2>&1
cd /d "%ENGINE%"
set SP_DJ_LIMIT=2
set SP_DJ_FLIMIT=2
set SP_DJ_STEPS=4
set SP_DJ_CANVAS=8
set SP_DJ_SEED=20260622
set "EXE=%ENGINE%\build-cuda\tests\test_diffjudge_denoise.exe"
echo N5C_VAL_START %DATE% %TIME% > "%ENGINE%\_n5c_val.log"
echo ================ f32 native judge (SP_DG_PACKED=0) ================ >> "%ENGINE%\_n5c_val.log"
set SP_DG_PACKED=0
set "SP_DJ_OUT=%ENGINE%\tests\fixtures\chat_fullstack\G-DG-N5c-VAL-f32.log"
"%EXE%" >> "%ENGINE%\_n5c_val.log" 2>&1
echo ================ packed native judge (SP_DG_PACKED=1) ================ >> "%ENGINE%\_n5c_val.log"
set SP_DG_PACKED=1
set "SP_DJ_OUT=%ENGINE%\tests\fixtures\chat_fullstack\G-DG-N5c-VAL-packed.log"
"%EXE%" >> "%ENGINE%\_n5c_val.log" 2>&1
echo N5C_VAL_DONE >> "%ENGINE%\_n5c_val.log"