@echo off
REM ============================================================================
REM  run_console_nightshift.bat -- run_console_recall.bat + B4 NIGHTSHIFT armed.
REM  Same B3-WC autonomous librarian, PLUS between-turn consolidation: every
REM  sufficiently-long user turn is captured (position-0 standalone) as a LIVE
REM  episode the W_c head can self-select on a later turn (the chat GROWS memory).
REM  Watch the console for "B4-NIGHTSHIFT: consolidated turn -> ep_live_NNN".
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
set "SP_DAEMON_KVDECODE_RING_W=2048"
set "SP_DAEMON_KVDECODE_PMAX=20000"

REM ---- B3-WC autonomous librarian ----
set "SP_RECALL_REGISTRY=%ENGINE%_needle_corpus_div\registry.jsonl"
set "SP_B3_WC=%ENGINE%_b3_wc\wc_deploy.bin"
set "SP_REPLAY_MTARGET=42"

REM ---- B4 NIGHTSHIFT (the only delta vs run_console_recall.bat) ----
set "SP_B4_NIGHTSHIFT=1"

cd /d "%ENGINE%tools\sp_daemon"
echo [nightshift] serving http://127.0.0.1:%PORT%/  recall ARMED + NIGHTSHIFT consolidation ON
echo [nightshift] loading the 12B (~9 GB) ... wait for "listening".
echo.
"%DAEMON%" start --model "%MODEL%" --tokenizer "%TOKENIZER%" --port %PORT%
endlocal
