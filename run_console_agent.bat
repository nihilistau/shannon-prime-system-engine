@echo off
REM ============================================================================
REM run_console_agent.bat — THE ONE-SHOT FULL STACK (AUDIT 2026-07-10).
REM Starts the daemon (reason model + production memory, run_console_reason.bat)
REM then the armed agent gateway (run_gateway_system.bat). Chat at
REM http://127.0.0.1:3000/ — the console auto-routes chat through the gateway
REM (:8800) when it is up, so tools/cards/persona all work.
REM ============================================================================
setlocal
set "ENGINE=%~dp0"
start "sp-daemon" cmd /c "%ENGINE%run_console_reason.bat"
echo [stack] daemon starting on :3000 ... waiting 25s for model load
timeout /t 25 /nobreak >nul
start "sp-gateway" cmd /c "%ENGINE%run_gateway_system.bat"
echo [stack] gateway starting on :8800
echo [stack] console:  http://127.0.0.1:3000/
echo [stack] operator: http://127.0.0.1:3000/operator.html
endlocal
