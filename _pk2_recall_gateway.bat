@echo off
REM PK2 wave-4 live gate launcher: the agent gateway with the ADR-008 pre-turn spine armed
REM (SP_SPINE_RECALL) against an isolated TEST registry seeded by the gate script.
setlocal
set "HARNESS=D:\F\shannon-prime-repos\shannon-prime-harness"
set "SP_DAEMON_URL=http://127.0.0.1:3000"
set "SP_RECALL_REGISTRY=%TEMP%\sp_pk2_recall_live.jsonl"
set "SP_CONV_OKF_ROOT=%HARNESS%\memory-okf-conv"
set "SP_CAPS_OKF_ROOT=%HARNESS%\memory-okf-caps"
set "SP_SPINE_RECALL=1"
set "SP_SPINE_TOOLSET=1"
cd /d "%HARNESS%"
python -m harness.server.app
