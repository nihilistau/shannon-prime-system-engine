@echo off
REM run_console_everything_reason.bat — the WHOLE verified stack (Tier0+Tier1+B4
REM growth+persist+veto head) on the PRODUCTION registry, serving the LoRA-merged
REM reasoning model. Identical to run_console_everything.bat except MODEL and the
REM libomp.dll PATH fix for the clang/OpenMP daemon runtime.
setlocal
set "ENGINE=%~dp0"
call "%ENGINE%scripts\env\env-cuda.bat" >nul 2>&1
set "PATH=C:\Program Files\LLVM\bin;%PATH%"
set "DAEMON=%ENGINE%tools\sp_daemon\target-wirecuda\release\sp-daemon.exe"
set "MODEL=D:/F/shannon-prime-repos/models/gemma4-12b-b1-reason.sp-model"
set "TOKENIZER=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-tokenizer"
set "PORT=3000"

REM ---- Tier 0 ----
set "SP_DAEMON_BACKEND=cuda"
set "SP_DAEMON_KVDECODE=1"
set "SP_CUDA_DECODE_INT8=1"
set "SP_DAEMON_KVDECODE_RING_W=2048"
set "SP_DAEMON_KVDECODE_PMAX=4096"
set "SP_PERSIST_KV=1"
set "SP_EOT_BIAS=4.0"

REM ---- Tier 1 on the PRODUCTION registry ----
if not exist "%ENGINE%_memory_live" mkdir "%ENGINE%_memory_live"
if not exist "%ENGINE%_memory_live\registry.jsonl" type nul > "%ENGINE%_memory_live\registry.jsonl"
set "SP_AUTO_RECALL_DEFAULT=0"
set "SP_RECALL_REGISTRY=%ENGINE%_memory_live\registry.jsonl"
set "SP_RECALL_L5=1"
set "SP_RECALL_L5_TAU=0.30"
set "SP_RECALL_ATTR_GATE=1"
set "SP_RECALL_ATTR_TAU=0.5"
set "SP_RECALL_QONLY=1"
set "SP_RECALL_L5_PROMPT=plain"
set "CUBLAS_WORKSPACE_CONFIG=:16:8"

REM ---- Tier 2: live growth + the explicit store verb ----
set "SP_B4_NIGHTSHIFT=1"
set "SP_NIGHTSHIFT_PERSIST=1"
set "SP_MEM_STORE=1"
REM AUDIT-2026-07-10: question-space keys at capture (fixes statement-space
REM coin-flip selection, OKFS 5d76b8f0) + deterministic classify + per-entry
REM policy dispatch (plain global, policy delivery per entry — 21/30 0-leak,
REM OKFS 171c675e/7a779cb0).
set "SP_QKEY_MINT=1"
set "SP_MEM_CLASSIFY=1"
set "SP_MEM_POLICY=1"

REM ---- The veto (safety net) ----
set "SP_SPECTEST=1"
set "SP_SPECTEST_HEAD=%ENGINE%_faithful_corpus\f3\spectest_head_f1.bin"

set "SP_DAEMON_LOG=%ENGINE%_everything_reason_serve.log"

echo [EVERYTHING-REASON] production memory + growth + recall + attr-gate + VETO HEAD
echo   console: http://127.0.0.1:3000/
taskkill /F /IM sp-daemon.exe >nul 2>&1
"%DAEMON%" start --model "%MODEL%" --tokenizer "%TOKENIZER%" --port %PORT%
endlocal
