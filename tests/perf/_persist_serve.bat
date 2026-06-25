@echo off
setlocal
set "ENGINE=D:\F\shannon-prime-repos\shannon-prime-system-engine"
set "DAEMON=%ENGINE%\tools\sp_daemon\target-wirecuda\release\sp-daemon.exe"
set "MODEL=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-model"
set "TOKENIZER=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-tokenizer"
set "PORT=3001"
call "%ENGINE%\scripts\env\env-cuda.bat" >nul 2>&1
set "SP_DAEMON_BACKEND=cuda"
set "SP_DAEMON_KVDECODE=1"
set "SP_CUDA_DECODE_INT8=1"
set "SP_DAEMON_KVDECODE_RING_W=2048"
set "SP_DAEMON_KVDECODE_PMAX=4096"
set "SP_ARENA_RELEASE=1"
set "SP_EOT_BIAS=4.0"
set "SP_PERSIST_KV=%~1"
cd /d "%ENGINE%\tools\sp_daemon"
"%DAEMON%" start --model "%MODEL%" --tokenizer "%TOKENIZER%" --port %PORT% > "%ENGINE%\_persist_serve.log" 2>&1
