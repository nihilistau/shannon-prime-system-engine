@echo off
call "%~dp0..\..\scripts\env\env-cuda.bat" >nul 2>&1
set SP_CUDA_DECODE_INT8=1
set SP_LI_CAPTURE=1
set SP_LI_LABELS=NO_OP,KEEP,FORGET,E2B_TOOL,ACTION
set SP_MODEL_PATH=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-model
set SP_TOKENIZER_PATH=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-tokenizer
set SP_LI_TAPE=%~dp0..\latent_interceptor\hard_train_act.txt
set SP_LI_OUT=%~dp0..\latent_interceptor\_hard_act_train_data
"%~dp0target-wirecuda\release\sp-daemon.exe"
