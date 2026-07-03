@echo off
REM G-SPECTEST-V3 phase 1 ? GROW the fresh corpus as live episodes (B4-SEAL path).
REM Fresh empty registry _v3_corpus\registry.jsonl; B4 + persist + L5-mint on.
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
if not exist "%ENGINE%_v3_corpus\registry.jsonl" type nul > "%ENGINE%_v3_corpus\registry.jsonl"
set "SP_AUTO_RECALL_DEFAULT=1"
set "SP_RECALL_REGISTRY=%ENGINE%_v3_corpus\registry.jsonl"
set "SP_RECALL_L5=1"
set "SP_RECALL_L5_TAU=0.30"
set "SP_RECALL_ATTR_GATE=1"
set "SP_RECALL_ATTR_TAU=0.5"
set "SP_RECALL_QONLY=1"
set "SP_RECALL_L5_PROMPT=plain"
set "CUBLAS_WORKSPACE_CONFIG=:16:8"
set "SP_QKEY_MINT=1"
set "SP_B4_NIGHTSHIFT=1"
set "SP_NIGHTSHIFT_PERSIST=1"
set "SP_DAEMON_LOG=%ENGINE%_v3_serve.log"
echo [V3-grow] fresh registry + B4 growth
taskkill /F /IM sp-daemon.exe >nul 2>&1
"%DAEMON%" start --model "%MODEL%" --tokenizer "%TOKENIZER%" --port %PORT%
endlocal

