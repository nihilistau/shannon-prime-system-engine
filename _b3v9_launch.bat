@echo off
REM %1 = SP_REPLAY_ALPHA (memory attention attenuation; 1.0 = null floor = full-strength replay)
call "D:\F\shannon-prime-repos\shannon-prime-system-engine\scripts\env\env-cuda.bat" >nul 2>&1
set "SP_DAEMON_BACKEND=cuda"
set "SP_DAEMON_KVDECODE=1"
set "SP_CUDA_DECODE_INT8=1"
set "SP_REPLAY_ALPHA=%1"
cd /d "D:\F\shannon-prime-repos\shannon-prime-system-engine\tools\sp_daemon"
"target-wirecuda\release\sp-daemon.exe" start --model "D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-model" --tokenizer "D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-tokenizer" --port 3000
