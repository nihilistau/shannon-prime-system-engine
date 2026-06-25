@echo off
REM Run the live agency + consolidation scheduler alongside the sp-daemon (port 3000).
REM Consolidates the current conversation (facts -> registry, transcript -> MEM-OKF) and
REM maintains memory on a heartbeat. Start AFTER the daemon (_e2e_seed_serve.bat) is up.
setlocal
set "ENGINE=D:\F\shannon-prime-repos\shannon-prime-system-engine"
set "HARNESS=D:\F\shannon-prime-repos\shannon-prime-harness"
set "SP_DAEMON_URL=http://127.0.0.1:3000"
set "SP_RECALL_REGISTRY=%ENGINE%\_seed_corpus\registry.jsonl"
set "SP_CONV_OKF_ROOT=%HARNESS%\memory-okf-conv"
set "SP_CAPS_OKF_ROOT=%HARNESS%\memory-okf-caps"
set "SP_CURRENT_CONVO=%ENGINE%\_current_conversation.json"
set "SP_AGENCY_INTERVAL=60"
cd /d "%HARNESS%"
python run_agency.py
