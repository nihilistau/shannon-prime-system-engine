@echo off
REM ============================================================================
REM run_f3_capture_B.bat — GEODESIC F3 capture, run B (CLEAN: no recall, no
REM faithfulness prompt). ADR-003 §5, gate G-F3-CAPTURE.
REM Tier 0 only + the F3 rail. No registry, no L5, no attr-gate — every turn is
REM the clean parametric path; the captured state is the x0 side of each pair.
REM The harness (f3_capture_run.py B) additionally sends NO system message and
REM auto_recall=false, belt-and-braces.
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
set "CUBLAS_WORKSPACE_CONFIG=:16:8"

REM ---- GEODESIC F3 rail ----
set "SP_F3_CAPTURE=%ENGINE%_faithful_corpus\f3\B"
set "SP_DAEMON_LOG=%ENGINE%_f3_serve_B.log"

echo [F3-B] capture serve (CLEAN parametric path, SP_F3_CAPTURE=_faithful_corpus\f3\B)
taskkill /F /IM sp-daemon.exe >nul 2>&1
"%DAEMON%" start --model "%MODEL%" --tokenizer "%TOKENIZER%" --port %PORT%
endlocal
