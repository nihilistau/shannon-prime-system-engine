@echo off
REM ============================================================================
REM run_spectest.bat — G-SPECTEST-V1 gate serve (draft→test→veto→clean-execute).
REM One-config Tier0+Tier1 with PLAIN delivery (the weak scaffold, 26/61 same-day
REM baseline) + SP_SPECTEST=1: every delivered draft is grounding-tested with the
REM stream HELD; ungrounded drafts are vetoed and replaced by the clean symbolic
REM answer from the record. GATE-ONLY (test corpus registry).
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

set "SP_SPECTEST=1"
set "SP_DAEMON_LOG=%ENGINE%_spectest_serve.log"

echo [SPECTEST] plain delivery + draft veto (SP_SPECTEST=1)
taskkill /F /IM sp-daemon.exe >nul 2>&1
"%DAEMON%" start --model "%MODEL%" --tokenizer "%TOKENIZER%" --port %PORT%
endlocal
