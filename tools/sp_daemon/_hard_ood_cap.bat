@echo off
call "%~dp0..\..\scripts\env\env-cuda.bat" >nul 2>&1
set SP_CUDA_DECODE_INT8=1
set SP_LI_CAPTURE=1
set SP_LI_LABELS=NONE,PYTHON,WEB,DB,FILE,CALC
set SP_MODEL_PATH=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-model
set SP_TOKENIZER_PATH=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-tokenizer
set SP_LI_TAPE=%~dp0..\latent_interceptor\hard_ood.txt
set SP_LI_OUT=%~dp0..\latent_interceptor\_hard_ood_data
"%~dp0target-wirecuda\release\sp-daemon.exe"
