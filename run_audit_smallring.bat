@echo off
REM Audit daemon, SMALL RING (RING_W=128): forces the SWA ring to WRAP under a fast
REM ~200-token prefix, so the LCP-rewind can be audited over a wrapped ring in seconds
REM (vs a ~1500-token byteexact prefill which is O(n^2) ~10min). Strictly harder wrap test.
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
set "SP_DAEMON_KVDECODE_RING_W=128"
set "SP_DAEMON_KVDECODE_PMAX=20000"
set "SP_PERSIST_KV=1"
set "SP_EOT_BIAS=4.0"
set "RUST_LOG=info"
taskkill /F /IM sp-daemon.exe >nul 2>&1
cd /d "%ENGINE%tools\sp_daemon"
"%DAEMON%" start --model "%MODEL%" --tokenizer "%TOKENIZER%" --port %PORT% > "%ENGINE%tests\perf\audit_smallring.log" 2>&1
endlocal
