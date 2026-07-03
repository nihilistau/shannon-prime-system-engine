@echo off
REM Zero-build probe: score the V3 episodes with the EXISTING W_c relevance head
REM (SP_B3_WC=wc_deploy.bin) instead of L5-cosine. The "B3-WC lse-mean" log line prints
REM every episode's W_c score per query -> parse offline for correct-vs-magnet rank.
REM SP_RECALL_L5 OFF so W_c is the sole selector. Registry frozen (no B4).
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
set "SP_RECALL_REGISTRY=%ENGINE%_v3_corpus\registry.jsonl"
set "SP_B3_WC=%ENGINE%_b3_wc\wc_deploy.bin"
set "CUBLAS_WORKSPACE_CONFIG=:16:8"
set "SP_DAEMON_LOG=%ENGINE%_v3_serve_wcprobe.log"
echo [V3-wcprobe] existing W_c selector, L5 off
taskkill /F /IM sp-daemon.exe >nul 2>&1
"%DAEMON%" start --model "%MODEL%" --tokenizer "%TOKENIZER%" --port %PORT%
endlocal
