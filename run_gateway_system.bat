@echo off
REM ============================================================================
REM run_gateway_system.bat — the AGENT GATEWAY, fully armed (AUDIT 2026-07-10).
REM Port 8800, in front of the daemon on :3000. This is what makes the console
REM AGENTIC: ephemeral tool calling (memory/system/web/python/coding), the
REM ADR-007 spine (typed SSE: tool cards + persona chip + recall notes),
REM personality self-modify tools, and the MCP bridge (mcp_servers.json).
REM
REM The old run_gateway.bat armed NOTHING (no spine/toolset/personality/MCP)
REM and pointed memory at the stale _seed_corpus registry — that is why tool
REM calling, cards, and persona editing "stopped working": the gateway either
REM wasn't running or ran disarmed, and the console chatted straight with the
REM daemon (:3000), which streams raw model text (fabricated tool_code fences
REM visible, nothing executed).
REM
REM Start AFTER the daemon (run_console_reason.bat). Console: the UI auto-uses
REM the gateway when it is up (index.html gateway autodetect, 2026-07-10).
REM Doc-update law: RUNBOOK-ONE-CONFIG.md launcher map updated same change-set.
REM ============================================================================
setlocal
set "HARNESS=D:\F\shannon-prime-repos\shannon-prime-harness"
set "ENGINE=D:\F\shannon-prime-repos\shannon-prime-system-engine"
set "SP_DAEMON_URL=http://127.0.0.1:3000"

REM Production memory (same registry the daemon serves) — NOT _seed_corpus.
set "SP_RECALL_REGISTRY=%ENGINE%\_memory_live\registry.jsonl"
set "SP_CONV_OKF_ROOT=%HARNESS%\memory-okf-conv"
set "SP_CAPS_OKF_ROOT=%HARNESS%\memory-okf-caps"

REM ---- The agentic stack (each individually gated GREEN) ----
set "SP_SPINE_TOOLSET=1"
set "SP_SPINE_RECALL=1"
set "SP_PERSONALITY=1"
set "SP_MCP_TOOLS=1"

echo [GATEWAY] agent gateway on :8800 (spine toolset + recall + personality + MCP)
echo   operator panel: http://127.0.0.1:3000/operator.html
cd /d "%HARNESS%"
python -m harness.server.app
endlocal
