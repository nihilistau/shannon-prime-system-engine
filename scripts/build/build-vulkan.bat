@echo off
setlocal
call "%~dp0..\env\env-vulkan.bat" || exit /b 1

if not exist "%SP_BUILD_DIR%" (
    cmake -S "%SP_ENGINE%" -B "%SP_BUILD_DIR%" -G %SP_GENERATOR% ^
        -DCMAKE_BUILD_TYPE=%SP_BUILD_TYPE_DEFAULT% ^
        -DSP_ENGINE_BACKEND=vulkan ^
        -DSP_ENGINE_WITH_VULKAN=ON ^
        -DSP_ENGINE_BUILD_TESTS=ON ^
        || exit /b 1
)
cmake --build "%SP_BUILD_DIR%" --config %SP_BUILD_TYPE_DEFAULT% -j
echo BUILD_EXIT=%ERRORLEVEL%
endlocal
