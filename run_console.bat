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
