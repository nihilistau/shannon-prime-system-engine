@echo off
REM G-MEMPOLICY-SERVED-DECLINE — the served decline arm, POLICY-DRIVEN. The SNE registry is
REM tagged mem_class=private-secret; env SP_RECALL_ATTR_GATE is deliberately UNSET so the ONLY
REM thing that can fire the zero-inference decline is the entry's own policy (SP_MEM_POLICY=1).
REM Arg1 = SP_MEM_POLICY (0|1). policy=1 -> attr-gate-strict decline on MISMATCH; =0 -> no shield.
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
set "SP_RECALL_REGISTRY=%ENGINE%_faithful_corpus\registry_sne.jsonl"
set "SP_RECALL_L5=1"
set "SP_RECALL_L5_TAU=0.30"
set "SP_RECALL_ATTR_TAU=0.5"
set "SP_RECALL_L5_PROMPT=recite"
set "SP_MEM_POLICY=%~1"
set "CUBLAS_WORKSPACE_CONFIG=:16:8"
set "SP_DAEMON_LOG=%ENGINE%_sne_serve_mempolicy.log"
echo [SNE-mempolicy] registry=SNE(private-secret) env-attr-gate=OFF SP_MEM_POLICY=%~1
taskkill /F /IM sp-daemon.exe >nul 2>&1
"%DAEMON%" start --model "%MODEL%" --tokenizer "%TOKENIZER%" --port %PORT%
endlocal
