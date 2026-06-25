@echo off
REM The harness AGENT GATEWAY (port 8800). The served console's chat (CHAT_URL) goes through
REM here so the model CALLS its tools (memory/system/web) in Gemma's tool_code format, instead
REM of the daemon's plain chat. Start AFTER the daemon (_e2e_seed_serve.bat). Metrics/mesh stay
REM on the daemon; only chat routes here.
setlocal
set "HARNESS=D:\F\shannon-prime-repos\shannon-prime-harness"
set "ENGINE=D:\F\shannon-prime-repos\shannon-prime-system-engine"
set "SP_DAEMON_URL=http://127.0.0.1:3000"
set "SP_RECALL_REGISTRY=%ENGINE%\_seed_corpus\registry.jsonl"
set "SP_CONV_OKF_ROOT=%HARNESS%\memory-okf-conv"
set "SP_CAPS_OKF_ROOT=%HARNESS%\memory-okf-caps"
cd /d "%HARNESS%"
python -m harness.server.app
