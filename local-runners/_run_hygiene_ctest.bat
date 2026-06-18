@echo off
call "%~dp0scripts\env\env-common.bat"
set "PATH=%SP_PIN_CUDA_ROOT%\bin;%PATH%"
cd /d "%SP_ENGINE%\build-cuda-vs22"
ctest -R M_GEMMA4_CUDA_PPL --output-on-failure -V
echo CTEST_EXIT=%ERRORLEVEL%
