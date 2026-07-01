@echo off
REM Foreign-reject capture: SP_RECALL_L5 with TAU=0 (log best cos for every query, never reject)
REM + SP_B3_QDUMP to persist each query's global-Q for full offline analysis.
call "%~dp0..\..\scripts\env\env-cuda.bat" >nul 2>&1
set SP_DAEMON_BACKEND=cuda
set SP_DAEMON_KVDECODE=1
set SP_CUDA_DECODE_INT8=1
set SP_DAEMON_KVDECODE_RING_W=1024
set SP_DAEMON_KVDECODE_PMAX=20000
set SP_PERSIST_KV=1
set SP_EOT_BIAS=4.0
set SP_RECALL_REGISTRY=D:\F\shannon-prime-repos\shannon-prime-system-engine\_faithful_corpus\registry.jsonl
set SP_RECALL_L5=1
set SP_RECALL_L5_TAU=0.0
set SP_B3_QDUMP=D:\F\shannon-prime-repos\shannon-prime-system-engine\_faithful_corpus\qdump_foreign
set SP_DAEMON_LOG=D:\F\shannon-prime-repos\shannon-prime-system-engine\_l5cap.log
taskkill /F /IM sp-daemon.exe >nul 2>&1
cd /d "%~dp0"
"%~dp0target-wirecuda\release\sp-daemon.exe" start --model "D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-model" --tokenizer "D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-tokenizer" --port 3000
