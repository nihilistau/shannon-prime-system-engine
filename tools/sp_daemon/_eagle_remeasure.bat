@echo off
call "%~dp0..\..\scripts\env\env-cuda.bat" >nul 2>&1
set SP_CUDA_DECODE_INT8=1
set SP_DRAFT_ASCALE=one
set SP_EAGLE_ACCEPT=1
set SP_EAGLE_K=4
set SP_EAGLE_DUMP=_eagle_flywheel
set SP_MODEL_PATH=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-model
set SP_TOKENIZER_PATH=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-tokenizer
set SP_DRAFT_GGUF=D:/F/shannon-prime-repos/shannon-prime-system-engine/tools/eagle/_eagle_data/gemma4-mtp-draft-ft.gguf
set SP_EAGLE_N=128
"%~dp0target-wirecuda\release\sp-daemon.exe"
