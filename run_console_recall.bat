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
set "SP_DAEMON_KVDECODE_RING_W=2048"
set "SP_DAEMON_KVDECODE_PMAX=20000"

REM ---- B3-WC autonomous librarian (the only delta vs run_console.bat) ----
REM SP_AUTO_RECALL_DEFAULT=1 makes the librarian fire without the client having to send auto_recall
REM per request (the console DOES send it via its checkbox, but other clients / tests need this).
set "SP_AUTO_RECALL_DEFAULT=1"
set "SP_RECALL_REGISTRY=%ENGINE%_needle_corpus_div\registry.jsonl"
set "SP_B3_WC=%ENGINE%_b3_wc\wc_deploy.bin"
set "SP_REPLAY_MTARGET=42"
REM (NO SP_B3_DISPOSER / NO SP_B3_TAU_QK -> the legacy q.K block stays telemetry-only.)

REM PERSISTENT O(1) CONVERSATION KV. SAFE under the W_c librarian: the wc_score relevance ranking
REM reads the cache's global K/Q NON-COMMITTINGLY, so a NULL (reject -> clean prompt) turn leaves
REM the plain-prompt cache pristine and persist reuses it; a RECALL injects the episode for
REM synthesis -> that turn skips re-arming the reusable prefix, so the next turn full-prefills
REM clean. Byte-exact either way (G-PERSIST-KV-PARITY recall extension). Default-off; opted in here.
set "SP_PERSIST_KV=1"

REM EOT BIAS default (project_eot_coherence_fix): stop the model degenerating past a turn boundary
REM (repeated glyphs to max_tokens). Console sends eot=4 per request; this is the daemon default for
REM other clients. Per-request eot_bias overrides.
set "SP_EOT_BIAS=4.0"

REM Only ONE 12B daemon fits in 12 GB VRAM -- kill any prior sp-daemon so launches never STACK
REM (stacked daemons thrashing the GPU = the "slow af" symptom). A fresh one starts below.
taskkill /F /IM sp-daemon.exe >nul 2>&1
cd /d "%ENGINE%tools\sp_daemon"
echo [recall] serving http://127.0.0.1:%PORT%/  with autonomous recall ARMED (registry=_needle_corpus_div, M=42)
echo [recall] loading the 12B (~9 GB) ... open the browser once you see "listening".
echo.
"%DAEMON%" start --model "%MODEL%" --tokenizer "%TOKENIZER%" --port %PORT%
endlocal
