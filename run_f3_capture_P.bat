@echo off
REM ============================================================================
REM run_f3_capture_P.bat — GEODESIC probe capture, run P (PLAIN delivery, F3 rail on).
REM G-OBEY-PROBE-OFFLINE: capture the DECIDE states of leak-prone plain-delivery
REM recall turns (balanced obey/leak labels vs the ~88%-obey systemecho A run).
REM = run_console_faithful.bat but SP_RECALL_L5_PROMPT=plain + SP_F3_CAPTURE=f3\P.
REM Attr-gate ON (canonical); the P driver sends fct paraphrases only.
REM ============================================================================
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
set "SP_RECALL_REGISTRY=%ENGINE%_faithful_corpus\registry_oneconfig.jsonl"
set "SP_RECALL_L5=1"
set "SP_RECALL_L5_TAU=0.30"
set "SP_RECALL_ATTR_GATE=1"
set "SP_RECALL_ATTR_TAU=0.5"
set "SP_RECALL_QONLY=1"
set "SP_RECALL_L5_PROMPT=plain"
set "CUBLAS_WORKSPACE_CONFIG=:16:8"

set "SP_F3_CAPTURE=%ENGINE%_faithful_corpus\f3\P"
set "SP_DAEMON_LOG=%ENGINE%_f3_serve_P.log"

echo [F3-P] probe capture serve (PLAIN delivery, F3 rail on)
taskkill /F /IM sp-daemon.exe >nul 2>&1
"%DAEMON%" start --model "%MODEL%" --tokenizer "%TOKENIZER%" --port %PORT%
endlocal
