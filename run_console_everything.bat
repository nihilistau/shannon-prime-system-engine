@echo off
REM ============================================================================
REM run_console_everything.bat — LIVE-PLAY: the whole verified stack at once
REM (2026-07-03). = run_console_system.bat (Tier0 + Tier1 + B4 growth + persist
REM on the PRODUCTION registry _memory_live) PLUS the SPECTEST veto head as a
REM safety net on recall turns (stream held + tested before it reaches you;
REM vetoed drafts answer clean from the record).
REM
REM STATUS: LIVE-PLAY, composition UNGATED as a whole (every part individually
REM gated: G-ONECONFIG-LIVE / G-B4-GROW-RECALL-L5 / G-SPECTEST-V2; V3 = safety
REM PASS, promotion pending question-space keys). Expect: recall turns do not
REM stream token-by-token (the hold is the point); statements you assert become
REM memories; questions recall them; SNE-style mismatches decline; ungrounded
REM drafts get replaced by "From the record: ...".
REM ============================================================================
setlocal
set "ENGINE=%~dp0"
call "%ENGINE%scripts\env\env-cuda.bat" >nul 2>&1
set "DAEMON=%ENGINE%tools\sp_daemon\target-wirecuda\release\sp-daemon.exe"
set "MODEL=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-model"
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
set "SP_AUTO_RECALL_DEFAULT=1"
set "SP_RECALL_REGISTRY=%ENGINE%_memory_live\registry.jsonl"
set "SP_RECALL_L5=1"
set "SP_RECALL_L5_TAU=0.30"
set "SP_RECALL_ATTR_GATE=1"
set "SP_RECALL_ATTR_TAU=0.5"
set "SP_RECALL_QONLY=1"
REM LIVE-FIX (the "Hodor" incident): PLAIN delivery, not systemecho. systemecho
REM COMMANDS verbatim echo, so a background (off-topic) L5 match makes the model
REM parrot an irrelevant record by design. plain+head is the V2-GATED pairing
REM (52/61 obey / 2 leak): the head vetoes real leaks; off-topic deliveries get
REM answered naturally by the draft (F 2/2 robustness) and released head-primary.
set "SP_RECALL_L5_PROMPT=plain"
set "CUBLAS_WORKSPACE_CONFIG=:16:8"

REM ---- Tier 2: live growth ----
set "SP_B4_NIGHTSHIFT=1"
set "SP_NIGHTSHIFT_PERSIST=1"

REM ---- The veto (safety net; systemecho rarely trips it — that's fine) ----
set "SP_SPECTEST=1"
set "SP_SPECTEST_HEAD=%ENGINE%_faithful_corpus\f3\spectest_head_f1.bin"

set "SP_DAEMON_LOG=%ENGINE%_everything_serve.log"

echo [EVERYTHING] production memory + growth + recall + attr-gate + VETO HEAD
echo   console: http://127.0.0.1:3000/
taskkill /F /IM sp-daemon.exe >nul 2>&1
"%DAEMON%" start --model "%MODEL%" --tokenizer "%TOKENIZER%" --port %PORT%
endlocal
