@echo off
REM ============================================================================
REM run_console_faithful.bat — THE canonical one-config launcher (2026-07-02)
REM Tier 0 (proven base) + Tier 1 (verified faithfulness edge: L5-cosine recall
REM + attribute-grounding gate w/ zero-inference decline).
REM Spec + flag receipts: lattice papers/RUNBOOK-ONE-CONFIG.md
REM STATUS: DRAFT — each flag individually gated GREEN; the combined stack is
REM pending its whole-run gate G-ONECONFIG-LIVE (RUNBOOK §7). Do not claim the
REM combination proven until that log exists.
REM ============================================================================
setlocal
set "ENGINE=%~dp0"
call "%ENGINE%scripts\env\env-cuda.bat" >nul 2>&1
set "DAEMON=%ENGINE%tools\sp_daemon\target-wirecuda\release\sp-daemon.exe"
set "MODEL=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-model"
set "TOKENIZER=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-tokenizer"
set "PORT=3000"

REM ---- Tier 0: proven base (CONTRACT-CHAT-FULLSTACK) ----
set "SP_DAEMON_BACKEND=cuda"
set "SP_DAEMON_KVDECODE=1"
set "SP_CUDA_DECODE_INT8=1"
set "SP_DAEMON_KVDECODE_RING_W=2048"
set "SP_DAEMON_KVDECODE_PMAX=20000"
set "SP_PERSIST_KV=1"
set "SP_EOT_BIAS=4.0"

REM ---- Tier 1: verified faithfulness edge (G-L5-RECALL-LIVE d9099cd +
REM      G-SNE-ATTRGATE-ZEROINF fc2e846). NOTE: SP_B3_WC deliberately NOT set —
REM      W_c+L5 combined is ungated (RUNBOOK §3). ----
set "SP_AUTO_RECALL_DEFAULT=1"
set "SP_RECALL_REGISTRY=%ENGINE%_faithful_corpus\registry_oneconfig.jsonl"
set "SP_RECALL_L5=1"
set "SP_RECALL_L5_TAU=0.30"
set "SP_RECALL_ATTR_GATE=1"
set "SP_RECALL_ATTR_TAU=0.5"
REM QONLY (pinned 2026-07-02, G-RECALL-QONLY-LEXICAL 188/188): conversational
REM statements skip the L5 stage (in-registry cos background >=0.9 otherwise
REM injects an irrelevant fact). Margin lever = HONEST NEGATIVE (G-L5-MARGIN-CALIB:
REM correct/background margins overlap; SNE canonical margins 0.0003-0.0007 would
REM break the SNE shield) -> SP_RECALL_L5_MARGIN stays UNSET (telemetry only).
set "SP_RECALL_QONLY=1"
REM SYSTEMECHO delivery prompt (2026-07-02 sweep winner, G-DELIVERY-SWEEP): fact as
REM SYSTEM authority + verbatim-echo priming. Full-61 = 88.52% OBEY, 0 LEAK (beats the
REM 07-01 receipt on today's stack); every correctly-selected episode obeyed — the
REM residual 7 misses are SELECTION cross-picks. Sweep: plain 40.98 < scaled 63.93 <
REM sandwich(10/16) < factecho/system(11/16) < systemecho(13/16 -> 54/61).
set "SP_RECALL_L5_PROMPT=systemecho"
REM cuBLAS determinism insurance (G-CUBLAS-PIN-CANARY 2026-07-02): no observed effect
REM at canary level, pinned as free run-to-run stability for the float GEMM surface.
set "CUBLAS_WORKSPACE_CONFIG=:16:8"
set "SP_DAEMON_LOG=%ENGINE%_oneconfig_serve.log"

echo [one-config] Tier0+Tier1 (L5 recall + attr-gate) — DRAFT until G-ONECONFIG-LIVE
taskkill /F /IM sp-daemon.exe >nul 2>&1
"%DAEMON%" start --model "%MODEL%" --tokenizer "%TOKENIZER%" --port %PORT%
endlocal
