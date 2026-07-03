@echo off
REM G-DF-PARITY — serve the 30 eval statements as concepts + run the 12B model_classify (refine)
REM with SP_MEM_REFINE_LOGALL so the 12B verdict for EVERY concept is logged (for the 0.5B A/B).
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
set "SP_RECALL_REGISTRY=%ENGINE%_parity_corpus\registry_empty.jsonl"
set "SP_MEM_OKF_STORE=%ENGINE%_parity_corpus\store"
set "SP_MEM_RECONCILE=1"
set "SP_MEM_RECONCILE_SEC=3"
set "SP_MEM_CLASSIFY_REFINE=1"
set "SP_MEM_REFINE_LOGALL=1"
set "CUBLAS_WORKSPACE_CONFIG=:16:8"
set "SP_DAEMON_LOG=%ENGINE%_parity_serve.log"
echo [parity] serving 30 eval concepts; 12B model_classify LOGALL on
taskkill /F /IM sp-daemon.exe >nul 2>&1
"%DAEMON%" start --model "%MODEL%" --tokenizer "%TOKENIZER%" --port %PORT%
endlocal
