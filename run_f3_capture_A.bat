@echo off
REM ============================================================================
REM run_f3_capture_A.bat — GEODESIC F3 capture, run A (WITH faithfulness delivery)
REM ADR-003 §5 (lattice papers/PPT-LAT-ADR-003-FLOW-TRANSPORT.md), gate G-F3-CAPTURE.
REM = run_console_faithful.bat (Tier0+Tier1, systemecho) + SP_F3_CAPTURE rail.
REM
REM *** CAPTURE-ONLY CONFIG — NEVER a canonical serve: ***
REM   SP_RECALL_ATTR_GATE=0 — the SNE zero-inference decline never decodes, so no
REM   residual exists under the shield; the A-side x1 state for SNE items IS the
REM   delivered-mismatched-fact state (confab/decline behavior is the label we
REM   record). The canonical config keeps the gate ON (RUNBOOK Tier 1).
REM ============================================================================
setlocal
set "ENGINE=%~dp0"
call "%ENGINE%scripts\env\env-cuda.bat" >nul 2>&1
set "DAEMON=%ENGINE%tools\sp_daemon\target-wirecuda\release\sp-daemon.exe"
set "MODEL=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-model"
set "TOKENIZER=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-tokenizer"
set "PORT=3000"

REM ---- Tier 0 (== run_console_faithful.bat) ----
set "SP_DAEMON_BACKEND=cuda"
set "SP_DAEMON_KVDECODE=1"
set "SP_CUDA_DECODE_INT8=1"
set "SP_DAEMON_KVDECODE_RING_W=2048"
set "SP_DAEMON_KVDECODE_PMAX=4096"
set "SP_PERSIST_KV=1"
set "SP_EOT_BIAS=4.0"

REM ---- Tier 1 minus the attr-gate (capture-only; see header) ----
set "SP_AUTO_RECALL_DEFAULT=1"
set "SP_RECALL_REGISTRY=%ENGINE%_faithful_corpus\registry_oneconfig.jsonl"
set "SP_RECALL_L5=1"
set "SP_RECALL_L5_TAU=0.30"
set "SP_RECALL_ATTR_GATE=0"
set "SP_RECALL_QONLY=1"
set "SP_RECALL_L5_PROMPT=systemecho"
set "CUBLAS_WORKSPACE_CONFIG=:16:8"

REM ---- GEODESIC F3 rail (default-off elsewhere; THE point of this launcher) ----
set "SP_F3_CAPTURE=%ENGINE%_faithful_corpus\f3\A"
set "SP_DAEMON_LOG=%ENGINE%_f3_serve_A.log"

echo [F3-A] capture serve (systemecho delivery, attr-gate OFF, SP_F3_CAPTURE=_faithful_corpus\f3\A)
taskkill /F /IM sp-daemon.exe >nul 2>&1
"%DAEMON%" start --model "%MODEL%" --tokenizer "%TOKENIZER%" --port %PORT%
endlocal
