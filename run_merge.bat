@echo off
REM G-STORE-MERGE (#78) — the engine serves memories the HARNESS wrote into the shared MEM-OKF
REM store. SP_MEM_OKF_STORE=<root>: at boot the engine loads full/*.md concepts, mints their L5
REM keys, and serves them per their OWN OKF policy. Empty registry => ONLY store concepts recall.
setlocal
set "ENGINE=%~dp0"
call "%ENGINE%scripts\env\env-cuda.bat" >nul 2>&1
set "DAEMON=%ENGINE%tools\sp_daemon\target-wirecuda\release\sp-daemon.exe"
set "MODEL=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-model"
set "TOKENIZER=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-tokenizer"
set "PORT=3000"
set "SP_DAEMON_BACKEND=cuda"
set "SP_DAEMON_KVDECODE=1"
set "SP_CUDA_DECODE_INT8=1"
set "SP_DAEMON_KVDECODE_RING_W=2048"
set "SP_DAEMON_KVDECODE_PMAX=4096"
set "SP_PERSIST_KV=1"
set "SP_EOT_BIAS=4.0"
set "SP_AUTO_RECALL_DEFAULT=1"
set "SP_RECALL_REGISTRY=%ENGINE%_merge_store\registry_empty.jsonl"
set "SP_MEM_OKF_STORE=%ENGINE%_merge_store"
set "SP_MEM_RECONCILE=1"
set "SP_MEM_RECONCILE_SEC=3"
set "SP_RECALL_L5=1"
set "SP_RECALL_L5_TAU=0.30"
set "SP_RECALL_L5_PROMPT=plain"
set "SP_MEM_POLICY=1"
set "SP_QKEY_MINT=1"
set "CUBLAS_WORKSPACE_CONFIG=:16:8"
set "SP_DAEMON_LOG=%ENGINE%_merge_serve.log"
echo [store-merge] engine serves harness-written memory-okf concepts
taskkill /F /IM sp-daemon.exe >nul 2>&1
"%DAEMON%" start --model "%MODEL%" --tokenizer "%TOKENIZER%" --port %PORT%
endlocal
