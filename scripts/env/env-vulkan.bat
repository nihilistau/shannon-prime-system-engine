@echo off
REM ─────────────────────────────────────────────────────────────────────
REM env-vulkan.bat — Vulkan backend environment (Vulkan SDK + MSVC)
REM Requires VULKAN_SDK to be set (Vulkan SDK installer does this globally).
REM ─────────────────────────────────────────────────────────────────────

call "%~dp0env-common.bat"

if "%VULKAN_SDK%"=="" (
    echo [env-vulkan] ERROR: VULKAN_SDK not set.
    echo              Install the Vulkan SDK ^>= %SP_PIN_VULKAN_MIN% from
    echo              https://vulkan.lunarg.com/ and rerun.
    exit /b 1
)
if not exist "%VULKAN_SDK%\Bin\glslc.exe" (
    echo [env-vulkan] ERROR: glslc.exe not found at %VULKAN_SDK%\Bin
    exit /b 1
)
if not exist "%SP_PIN_VS_BUILDTOOLS%\VC\Auxiliary\Build\vcvars64.bat" (
    echo [env-vulkan] ERROR: VS2019 Build Tools not at expected location.
    exit /b 1
)

call "%SP_PIN_VS_BUILDTOOLS%\VC\Auxiliary\Build\vcvars64.bat" >nul

set PATH=%VULKAN_SDK%\Bin;%PATH%
set SP_BACKEND=vulkan
set SP_BUILD_DIR=%SP_ENGINE%\build-vulkan
set SP_VULKAN_SUBGROUP_OPS=1

echo [env-vulkan] Vulkan SDK at %VULKAN_SDK% activated.
echo [env-vulkan] MSVC + glslc on PATH.  SP_BUILD_DIR=%SP_BUILD_DIR%
