@echo off
REM %1 = registry path. Ablation gate (SP_B3_DISPOSER=2), telemetry TAU=-inf (never fires).
call "D:\F\shannon-prime-repos\shannon-prime-system-engine\scripts\env\env-cuda.bat" >nul 2>&1
set "SP_DAEMON_BACKEND=cuda"
set "SP_DAEMON_KVDECODE=1"
set "SP_CUDA_DECODE_INT8=1"
set "SP_RECALL_REGISTRY=%1"
set "SP_B3_DISPOSER=2"
set "SP_REPLAY_MTARGET=42"
cd /d "D:\F\shannon-prime-repos\shannon-prime-system-engine\tools\sp_daemon"
"target-wirecuda\release\sp-daemon.exe" start --model "D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-model" --tokenizer "D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-tokenizer" --port 3000
