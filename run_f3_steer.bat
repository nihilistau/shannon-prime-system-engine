@echo off
REM ============================================================================
REM run_f3_steer.bat <alpha> — GEODESIC G-FM-STEER serve (ADR-003 §4.2).
REM One-config Tier0+Tier1 BUT delivery = PLAIN wording (the weak-scaffold
REM baseline, 40.98% receipt RUNBOOK §9) + persistent pre-head steering armed on
REM recall turns with the Tier-A mean velocity v̄ (G-FLOW-STRAIGHTNESS) at the
REM given alpha. alpha=0 (or no arg) = null floor = the plain baseline serve.
REM Gate question: does v̄ recover the systemecho scaffold's obedience (88.52%)
REM without the scaffold?
REM ============================================================================
setlocal
set "ENGINE=%~dp0"
call "%ENGINE%scripts\env\env-cuda.bat" >nul 2>&1
set "DAEMON=%ENGINE%tools\sp_daemon\target-wirecuda\release\sp-daemon.exe"
set "MODEL=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-model"
set "TOKENIZER=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-tokenizer"
set "PORT=3000"

REM ---- Tier 0 + Tier 1 (== run_console_faithful.bat) except the delivery prompt ----
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

REM ---- GEODESIC steering (vec always named; alpha from arg; 0/empty = null floor) ----
set "SP_STEER_VEC=%ENGINE%_faithful_corpus\f3\steer_vbar_f0_all81.bin"
set "SP_STEER_ALPHA=%~1"
set "SP_DAEMON_LOG=%ENGINE%_f3_steer_serve.log"

echo [F3-STEER] plain delivery + v-bar alpha=%SP_STEER_ALPHA%
taskkill /F /IM sp-daemon.exe >nul 2>&1
"%DAEMON%" start --model "%MODEL%" --tokenizer "%TOKENIZER%" --port %PORT%
endlocal
