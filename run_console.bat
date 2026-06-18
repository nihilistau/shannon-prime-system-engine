@echo off
REM ============================================================================
REM  run_console.bat  --  launch the Shannon-Prime operator console end-to-end.
REM
REM  Starts the wire_cuda sp-daemon with the real Gemma-4-12B (OK_Q4B) on the
REM  RTX 2060, serving the operator console at http://127.0.0.1:3000/.
REM
REM  The daemon drives the full L0/L1/L2 stack: tokenizer (L2) -> session/clone
REM  (L2) -> CUDA forward+kvdecode backends (L1, G-WIRE-CUDA-*) -> math core (L0).
REM  Chat streams token-by-token over /v1/chat SSE.
REM
REM  Open in a browser AFTER the daemon logs "listening":  http://127.0.0.1:3000/
REM ============================================================================
setlocal
set "ENGINE=%~dp0"
set "DAEMON=%ENGINE%tools\sp_daemon\target-wirecuda\release\sp-daemon.exe"
set "MODEL=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-model"
set "TOKENIZER=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-tokenizer"
set "PORT=3000"

REM CUDA runtime DLLs on PATH (cudart/cublas) for the wire_cuda backend.
call "%ENGINE%scripts\env\env-cuda.bat" >nul 2>&1

REM Route the L1 forward + kvdecode backends to the CUDA Gemma-4 path.
set "SP_DAEMON_BACKEND=cuda"

REM #115: the 12B's tied full-vocab head is only materializable through the
REM resident persistent-KV decode cache (G-WIRE-CUDA-DECODE-GEMMA4). Without
REM these, /v1/chat prefill trips "g4 probe: FULL head needs the f32 embd".
REM KVDECODE=1 routes decode through gemma4_kv_decode_logits; INT8=1 lets
REM gemma4_kv_open build the tied head.
set "SP_DAEMON_KVDECODE=1"
set "SP_CUDA_DECODE_INT8=1"

REM CONTRACT-CHAT-FULLSTACK B2: the XBAR SWA ring on the resident chat decode is
REM DISARMED — it is RED. Armed, k_attn_decode_ring on this path produces incoherent
REM token-soup (the B2 null-parity SHA was a FALSE green: a determinism gate cannot
REM tell coherent output from garbage, so it passed bit-identical garbage). Full-cache
REM (ring unset) decode is COHERENT. Re-arm only after the ring is debugged AND a
REM coherence gate (not just a determinism SHA) passes. See CONTRACT-CHAT-FULLSTACK §5.
REM   set "SP_DAEMON_KVDECODE_RING_W=1024"
REM   set "SP_DAEMON_KVDECODE_PMAX=20000"

REM CWD must be tools\sp_daemon so the static ServeDir("frontend_mockups") resolves.
cd /d "%ENGINE%tools\sp_daemon"

echo [run_console] daemon : %DAEMON%
echo [run_console] model  : %MODEL%
echo [run_console] serving: http://127.0.0.1:%PORT%/   (console.html is also at /console.html)
echo [run_console] backend: SP_DAEMON_BACKEND=%SP_DAEMON_BACKEND%
echo.
echo [run_console] loading the 12B (~9 GB) ... open the browser once you see "listening".
echo.

"%DAEMON%" start --model "%MODEL%" --tokenizer "%TOKENIZER%" --port %PORT%

endlocal
