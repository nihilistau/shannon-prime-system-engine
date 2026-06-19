@echo off
REM ============================================================================
REM  run_console_recall.bat -- run_console.bat + the B3-WC AUTONOMOUS LIBRARIAN.
REM
REM  Same coherent byte-exact chat as run_console.bat (SWA ring B2, served at
REM  http://127.0.0.1:3000/), but with the LEARNED-HEAD autonomous recall ARMED:
REM  every /v1/chat turn is scored against the 90-needle div registry; the W_c
REM  head (logsumexp-mean, (E+1) NULL argmax) recalls the right episode or rejects.
REM  Watch the daemon console for "B3-WC ... RECALL '<ep>'" or "NULL wins -> REJECT".
REM
REM  Try (matched):  "Which recovery code authorizes the Marlock mag-rail depot?"
REM  Try (foreign):  "What is the capital of France?"   (expect clean reject)
REM  Open the browser AFTER the daemon logs "listening":  http://127.0.0.1:3000/
REM ============================================================================
setlocal
set "ENGINE=%~dp0"
set "DAEMON=%ENGINE%tools\sp_daemon\target-wirecuda\release\sp-daemon.exe"
set "MODEL=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-model"
set "TOKENIZER=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-tokenizer"
set "PORT=3000"

call "%ENGINE%scripts\env\env-cuda.bat" >nul 2>&1
set "SP_DAEMON_BACKEND=cuda"
set "SP_DAEMON_KVDECODE=1"
set "SP_CUDA_DECODE_INT8=1"
set "SP_DAEMON_KVDECODE_RING_W=1024"
set "SP_DAEMON_KVDECODE_PMAX=20000"

REM ---- B3-WC autonomous librarian (the only delta vs run_console.bat) ----
set "SP_RECALL_REGISTRY=%ENGINE%_needle_corpus_div\registry.jsonl"
set "SP_B3_WC=%ENGINE%_b3_wc\wc_deploy.bin"
set "SP_REPLAY_MTARGET=42"
REM (NO SP_B3_DISPOSER / NO SP_B3_TAU_QK -> the legacy q.K block stays telemetry-only.)

cd /d "%ENGINE%tools\sp_daemon"
echo [recall] serving http://127.0.0.1:%PORT%/  with autonomous recall ARMED (registry=_needle_corpus_div, M=42)
echo [recall] loading the 12B (~9 GB) ... open the browser once you see "listening".
echo.
"%DAEMON%" start --model "%MODEL%" --tokenizer "%TOKENIZER%" --port %PORT%
endlocal
