@echo off
setlocal
call "%~dp0..\env\env-hexagon.bat" || exit /b 1

if not exist "%SP_BUILD_DIR%" (
    cmake -S "%SP_ENGINE%" -B "%SP_BUILD_DIR%" -G %SP_GENERATOR% ^
        -DCMAKE_BUILD_TYPE=%SP_BUILD_TYPE_DEFAULT% ^
        -DSP_ENGINE_BACKEND=hexagon ^
        -DSP_ENGINE_WITH_HEXAGON=ON ^
        -DSP_HEXAGON_TARGET=%SP_HEXAGON_TARGET% ^
        -DCMAKE_TOOLCHAIN_FILE="%SP_ENGINE%\cmake\toolchain-hexagon.cmake" ^
        -DSP_ENGINE_BUILD_TESTS=ON ^
        || exit /b 1
)
cmake --build "%SP_BUILD_DIR%" --config %SP_BUILD_TYPE_DEFAULT% -j
echo BUILD_EXIT=%ERRORLEVEL%
endlocal
