@echo off
REM G-LM-REFINE (LM-B3) — the idle NIGHTSHIFT model-refine CORRECTS a heuristic mis-classification.
REM Serves the _refine_corpus store with SP_MEM_CLASSIFY_REFINE=1 (idle re-classify) + SP_MEM_POLICY=1
REM (serve per policy) + SP_MEM_RECONCILE=1 (hot-reload). Run refine_gate.py setup THEN this, then
REM refine_gate.py run.
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
set "SP_RECALL_REGISTRY=%ENGINE%_refine_corpus\registry_empty.jsonl"
set "SP_MEM_OKF_STORE=%ENGINE%_refine_corpus\store"
set "SP_MEM_RECONCILE=1"
set "SP_MEM_RECONCILE_SEC=3"
if "%~1"=="on" ( set "SP_MEM_CLASSIFY_REFINE=1" )
set "SP_TELEMETRY=1"
set "SP_TELEMETRY_LOG=%ENGINE%_refine_corpus\telemetry.jsonl"
set "CUBLAS_WORKSPACE_CONFIG=:16:8"
set "SP_DAEMON_LOG=%ENGINE%_refine_serve.log"
echo [refine] serving _refine_corpus store with SP_MEM_CLASSIFY_REFINE=1 (idle model-refine)
taskkill /F /IM sp-daemon.exe >nul 2>&1
"%DAEMON%" start --model "%MODEL%" --tokenizer "%TOKENIZER%" --port %PORT%
endlocal
