@echo off
REM ============================================================================
REM run_console_reason.bat — REASON daily driver, memory-complete (AUDIT 2026-07-10).
REM Serves the LoRA-merged reasoning model with the full verified memory stack on
REM the PRODUCTION registry (_memory_live\registry.jsonl).
REM
REM AUDIT-2026-07-10 fix — the previous revision was memory-broken:
REM   * `set "SP_NIGHTSHIFT_PERSIST"` (no value) is a NO-OP in batch — persist off;
REM   * no SP_RECALL_REGISTRY — recall chain never armed (daemon loads None);
REM   * SP_RECALL_L5=0 + SP_AUTO_RECALL_DEFAULT=0 — grown memories unrecallable.
REM   Net effect: B4 captured episodes into _nightshift_live as ORPHANS (no
REM   registry row), nothing survived a restart = "memory does not persist".
REM
REM This revision wires the OKFS-sanctioned stack:
REM   SP_QKEY_MINT=1     question-space ep.l5 keys at capture — the proven fix for
REM                      statement-space coin-flip selection (OKFS 5d76b8f0/6d191b7).
REM   SP_MEM_CLASSIFY=1  deterministic mem_class at capture, episode self-governs
REM                      (OKFS 7a779cb0).
REM   SP_MEM_POLICY=1    per-entry delivery dispatch: global stays plain, classified
REM                      entries force their own delivery — 21/30 0-leak (OKFS 171c675e).
REM   SP_RECALL_L5_PROMPT=plain  Hodor binding: NEVER systemecho globally (OKFS c57745f1).
REM   SP_AUTO_RECALL_DEFAULT=1   quiet-memory mode (2026-07-03) was explicitly
REM                      "until question-space keys land" — SP_QKEY_MINT is that
REM                      landing. Console checkbox still overrides per-request.
REM Doc-update law: this change lands in RUNBOOK-ONE-CONFIG.md launcher map in the
REM same commit.
REM ============================================================================
setlocal
set "ENGINE=%~dp0"
call "%ENGINE%scripts\env\env-cuda.bat" >nul 2>&1
REM clang/OpenMP runtime (libomp.dll) for the daemon:
set "PATH=C:\Program Files\LLVM\bin;%PATH%"
set "DAEMON=%ENGINE%tools\sp_daemon\target-wirecuda\release\sp-daemon.exe"
set "MODEL=D:/F/shannon-prime-repos/models/gemma4-12b-b1-reason.sp-model"
set "TOKENIZER=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-tokenizer"
set "PORT=3000"

REM ---- Tier 0 (proven base) ----
set "SP_DAEMON_BACKEND=cuda"
set "SP_DAEMON_KVDECODE=1"
set "SP_CUDA_DECODE_INT8=1"
set "SP_DAEMON_KVDECODE_RING_W=2048"
set "SP_DAEMON_KVDECODE_PMAX=12096"
set "SP_PERSIST_KV=1"
REM HINDSIGHT: keep the O(1) conversation cache ALIVE under B4 memory growth
REM (persist was silently force-disabled whenever SP_B4_NIGHTSHIFT=1 -> every
REM turn full-re-prefilled ~1.5k tokens = the minutes-long agent turns).
set "SP_PERSIST_B4=1"
set "SP_EOT_BIAS=4.0"
set "SP_NO_REPEAT_NGRAM=3"
set "CUBLAS_WORKSPACE_CONFIG=:16:8"

REM ---- Tier 1: recall on the PRODUCTION registry ----
if not exist "%ENGINE%_memory_live" mkdir "%ENGINE%_memory_live"
if not exist "%ENGINE%_memory_live\registry.jsonl" type nul > "%ENGINE%_memory_live\registry.jsonl"
set "SP_AUTO_RECALL_DEFAULT=1"
set "SP_RECALL_REGISTRY=%ENGINE%_memory_live\registry.jsonl"
set "SP_RECALL_L5=1"
set "SP_RECALL_L5_TAU=0.30"
set "SP_RECALL_ATTR_GATE=1"
set "SP_RECALL_ATTR_TAU=0.5"
set "SP_RECALL_QONLY=1"
set "SP_RECALL_L5_PROMPT=plain"

REM ---- Tier 2: live growth + store verb + self-governing policy ----
set "SP_B4_NIGHTSHIFT=1"
set "SP_NIGHTSHIFT_PERSIST=1"
set "SP_MEM_STORE=1"
set "SP_QKEY_MINT=1"
set "SP_MEM_CLASSIFY=1"
set "SP_MEM_POLICY=1"
REM SP_SPINE=1 REMOVED (AUDIT 2026-07-10 live A/B): with SPINE on + a small
REM personal registry, a foreign question ("capital of Spain?") delivered the
REM persona record ("Knack.") — off-topic leak past attr-gate/veto. The proven
REM serve composition (run_console_everything*, G-B4-GROW-RECALL-L5) is the
REM INLINE path; keep SPINE off here until the spine composition earns its gate.

REM ---- The veto head (safety net; head-primary per Hodor fix #1) ----
set "SP_SPECTEST=1"
set "SP_SPECTEST_HEAD=%ENGINE%_faithful_corpus\f3\spectest_head_f1.bin"

set "SP_DAEMON_LOG=%ENGINE%_reason_serve.log"

echo [REASON] 12B reason model + production memory: grow/persist/recall/policy
echo   console: http://127.0.0.1:3000/
taskkill /F /IM sp-daemon.exe >nul 2>&1
"%DAEMON%" start --model "%MODEL%" --tokenizer "%TOKENIZER%" --port %PORT%
endlocal
