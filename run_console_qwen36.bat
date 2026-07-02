@echo off
REM ============================================================================
REM run_console_qwen36.bat ? the qwen36 (Qwen3.6-35B-A3B GDN+MoE) served chat
REM (CONTRACT-QWEN36-SERVE, S3). The 337x hybrid ladder (G-MOE-GPU4-PINNED,
REM 6.073 tok/s on the 2060) served through sp-daemon /v1/chat.
REM
REM EXCLUSIVE INSTANCE: the 2060 cannot host the 12B resident cache AND the
REM 35B expert residency together ? do not run alongside run_console_faithful.
REM Port 3001 so the two launchers never collide.
REM PERF EXE: target-wirecuda-perf links the build-cpu-perf math-core libs
REM   (/openmp /arch:AVX2). The standard target-wirecuda daemon links build-cpu
REM   (NO omp/avx2) and serves the 35B at ~2 tok/s instead of ~6 (root-caused
REM   2026-07-02: CPU-only A/B hit 0.17 tok/s = the pre-OMP ladder rung). The
REM   12B one-config launcher keeps the PROVEN standard exe; do not point it at
REM   the perf build without re-running G-ONECONFIG-LIVE.
REM STATUS: DRAFT until G-QWEN36-SERVE is GREEN.
REM ============================================================================
setlocal
set "ENGINE=%~dp0"
call "%ENGINE%scripts\env\env-cuda.bat" >nul 2>&1
REM libomp lives in LLVM\bin ? without it the exe dies silently (0xC0000135,
REM receipted twice). env-cuda does not add it.
set "PATH=C:\Program Files\LLVM\bin;%PATH%"
set "DAEMON=%ENGINE%tools\sp_daemon\target-wirecuda-perf\release\sp-daemon.exe"
set "MODEL=D:/F/shannon-prime-repos/models/qwen36-35b-a3b.sp-model"
set "TOKENIZER=D:/F/shannon-prime-repos/models/qwen36-35b-a3b.sp-tokenizer"
set "PORT=3001"

REM ---- backend: SP_DAEMON_BACKEND deliberately UNSET ? the L1 wire-backend
REM      registration (prefill/kvdecode) is gemma/qwen3-session machinery and
REM      panics on the sessionless qwen36 lane. The q36 GPU hybrid boots
REM      directly via sp_q36gpu_boot (SP_Q36_GPU below); it does not need the
REM      ?6 backend hook. ----

REM ---- the qwen36 GPU hybrid (G-MOE-GPU1..GPU4): dense set resident
REM      (852.5MB) + experts under budget (26/40 layers @ 9.9GB) + pinned
REM      ping-pong streaming for the rump. SP_Q36_GPU=0 = CPU-only lane. ----
set "SP_Q36_GPU=1"
set "SP_Q36_GPU_MOE_GB=9.6"
set "SP_Q36_GPU_STREAM=1"
set "SP_Q36_PMAX=4096"

set "SP_DAEMON_LOG=%ENGINE%_qwen36_serve.log"

echo [qwen36] 35B-A3B hybrid serve on port %PORT% ? DRAFT until G-QWEN36-SERVE
taskkill /F /IM sp-daemon.exe >nul 2>&1
"%DAEMON%" start --model "%MODEL%" --tokenizer "%TOKENIZER%" --port %PORT%
endlocal

