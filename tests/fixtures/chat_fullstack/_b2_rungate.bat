@echo off
call scripts\env\env-cuda.bat >nul 2>&1
set "SP_CUDA_DECODE_INT8=1"
tools\sp_daemon\target-wirecuda\release\sp_wire_cuda_decode_gate.exe "D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-model" "D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-tokenizer" 32
