@echo off
REM G-MEMPOLICY-SERVED — the served-path proof of ADR-004: the GLOBAL delivery flag is
REM SP_RECALL_L5_PROMPT=plain, but the registry entries carry mem_class=counterfact. With
REM SP_MEM_POLICY=1 the per-entry policy OVERRIDES the global flag -> systemecho delivery
REM (obey~22/0). With SP_MEM_POLICY=0 the entries fall back to the global plain (obey~11/18).
REM Arg1 = SP_MEM_POLICY value (0|1).
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
set "SP_RECALL_L5=1"
set "SP_RECALL_L5_TAU=0.30"
set "SP_RECALL_QONLY=1"
set "SP_RECALL_L5_PROMPT=plain"
set "SP_QKEY_MINT=1"
set "SP_MEM_POLICY=%~1"
set "CUBLAS_WORKSPACE_CONFIG=:16:8"
set "SP_DAEMON_LOG=%ENGINE%_v3_serve_mempolicy.log"
echo [V3-mempolicy] global=plain SP_MEM_POLICY=%~1 (entries=counterfact)
taskkill /F /IM sp-daemon.exe >nul 2>&1
"%DAEMON%" start --model "%MODEL%" --tokenizer "%TOKENIZER%" --port %PORT%
endlocal
