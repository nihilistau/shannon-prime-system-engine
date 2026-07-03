@echo off
REM G-DF-LIVE — the deployed loop LIVE: engine serves the store with SP_MEM_RECONCILE on (NO engine
REM refine), so the ONLY thing that corrects a concept is the HARNESS curator; the engine picks up
REM the frontmatter edit via reconcile-on-edit and serves the corrected policy.
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
set "SP_RECALL_L5=1"
set "SP_RECALL_L5_TAU=0.30"
set "SP_RECALL_L5_PROMPT=plain"
set "SP_QKEY_MINT=1"
set "SP_MEM_POLICY=1"
set "SP_RECALL_REGISTRY=%ENGINE%_live_corpus\registry_empty.jsonl"
set "SP_MEM_OKF_STORE=%ENGINE%_live_corpus\store"
set "SP_MEM_RECONCILE=1"
set "SP_MEM_RECONCILE_SEC=3"
set "SP_TELEMETRY=1"
set "SP_TELEMETRY_LOG=%ENGINE%_live_corpus\telemetry.jsonl"
set "CUBLAS_WORKSPACE_CONFIG=:16:8"
set "SP_DAEMON_LOG=%ENGINE%_live_serve.log"
echo [live] serving store w/ reconcile-on-edit; harness curator is the ONLY classifier
taskkill /F /IM sp-daemon.exe >nul 2>&1
"%DAEMON%" start --model "%MODEL%" --tokenizer "%TOKENIZER%" --port %PORT%
endlocal
