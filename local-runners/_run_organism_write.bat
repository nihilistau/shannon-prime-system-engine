@echo off
REM G-XBAR-ORGANISM step 1 — EAR -> Ring-2 write seam: inject a REAL audio-derived packet, serialize ep_audio.
call "%~dp0scripts\env\env-common.bat"
set "PATH=%SP_PIN_CUDA_ROOT%\bin;%PATH%"
set "SP_GEMMA4_SPMODEL=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-model"
set "SP_GEMMA4_SPTOK=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-tokenizer"
set "SP_CUDA_DECODE_INT8=1"
set "SP_G4_KAI3_WRITE=D:\F\shannon-prime-repos\_xbar\p2b\kai3\kai3_audio_packets\aud_00_ACTION.bin"
set "SP_KAI3_WRITE_OUT=%SP_ENGINE%\_ep_audio"
if exist "%SP_KAI3_WRITE_OUT%" rmdir /s /q "%SP_KAI3_WRITE_OUT%"
mkdir "%SP_KAI3_WRITE_OUT%"
"%SP_ENGINE%\build-cuda-vs22\tests\test_gemma4_cuda.exe" > "%SP_ENGINE%\_organism_write.log" 2>&1
echo WRITE_EXIT=%ERRORLEVEL% >> "%SP_ENGINE%\_organism_write.log"
echo ORG_WRITE_DONE > "%SP_KAI3_WRITE_OUT%\..\_organism_write.flag"
