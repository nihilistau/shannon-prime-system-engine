@echo off
REM Phase 0 placeholder — Phase 2-CPU.B fills in with real model load + logits check.
setlocal
call "%~dp0..\env\env-cpu.bat" || exit /b 1

if not exist "%SP_BUILD_DIR%\bin\sp-engine.exe" (
    echo [smoke-cpu] sp-engine.exe not built yet.  Run scripts\build\build-cpu.bat first.
    exit /b 1
)

REM Phase 0: just check the binary launches and prints --version.
"%SP_BUILD_DIR%\bin\sp-engine.exe" --version
echo SMOKE_EXIT=%ERRORLEVEL%
endlocal
