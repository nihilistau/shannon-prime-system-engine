@echo off
REM ===========================================================================
REM B3 admission gate — launch the daemon in ABLATION mode (SP_B3_DISPOSER=2)
REM with a given SECRET + REGISTRY, telemetry TAU=-inf (never re-injects; we only
REM read the collapse log).  Mirrors _labQ*.bat.  Fire the needle's query at
REM /v1/chat; the daemon logs "B3-DISPOSER ABLATION collapse=..." per episode.
REM A genuine novel needle self-collapses catastrophically (ACCEPT); a parametric
REM leak shrugs (~0) (REJECT).
REM
REM   %1 = exact secret string (incl. leading space), e.g. " 6-SENTINEL-7993"
REM   %2 = registry jsonl path
REM ===========================================================================
call "%~dp0scripts\env\env-cuda.bat" >nul 2>&1
set "SP_DAEMON_BACKEND=cuda"
set "SP_DAEMON_KVDECODE=1"
set "SP_CUDA_DECODE_INT8=1"
set "SP_RECALL_REGISTRY=%~2"
set "SP_B3_DISPOSER=2"
set "SP_B3_SECRET=%~1"
REM TAU left at default (-inf) => telemetry only, no live re-inject.
cd /d "%~dp0tools\sp_daemon"
"target-wirecuda\release\sp-daemon.exe" start --model "D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-model" --tokenizer "D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-tokenizer" --port 3000
