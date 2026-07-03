@echo off
REM G-MEMCLASSIFY-SERVED (#73) — NIGHTSHIFT auto-classifies mem_class at capture. Arg1 phase:
REM   grow : SP_MEM_STORE + SP_MEM_CLASSIFY=1 + persist -> stores self-classify into the registry.
REM   gate : SP_MEM_POLICY=1 -> each stored memory is served per its AUTO-assigned policy.
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
set "SP_RECALL_REGISTRY=%ENGINE%_classify_corpus\registry.jsonl"
set "SP_RECALL_L5=1"
set "SP_RECALL_L5_TAU=0.30"
set "SP_RECALL_L5_PROMPT=plain"
set "SP_QKEY_MINT=1"
set "SP_MEM_CLASSIFY=1"
set "CUBLAS_WORKSPACE_CONFIG=:16:8"
if "%~1"=="grow" (
    set "SP_MEM_STORE=1"
    set "SP_NIGHTSHIFT_PERSIST=1"
)
if "%~1"=="gate" (
    set "SP_MEM_POLICY=1"
)
set "SP_DAEMON_LOG=%ENGINE%_classify_serve.log"
echo [classify] phase=%~1 (SP_MEM_CLASSIFY=1)
taskkill /F /IM sp-daemon.exe >nul 2>&1
"%DAEMON%" start --model "%MODEL%" --tokenizer "%TOKENIZER%" --port %PORT%
endlocal
