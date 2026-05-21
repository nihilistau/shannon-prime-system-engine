@echo off
REM ---------------------------------------------------------------------
REM shannon-prime-system-engine  env-common.bat
REM Shared environment for all backends. Sourced by env-cpu / env-cuda /
REM env-vulkan / env-hexagon.bat. EDIT WITH CAUTION -- paths are pinned.
REM (ASCII only, no parenthesised if-blocks: paths contain "(x86)".)
REM ---------------------------------------------------------------------

REM Repo roots
set SP_REPO_ROOT=D:\F\shannon-prime-repos
set SP_LATTICE=%SP_REPO_ROOT%\shannon-prime-lattice
set SP_SYSTEM=%SP_REPO_ROOT%\shannon-prime-system
set SP_ENGINE=%SP_REPO_ROOT%\shannon-prime-system-engine

REM Toolchain pins. Any change here is a project decision, not a session decision.
set SP_PIN_VS_BUILDTOOLS=C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools
set SP_PIN_CUDA_VERSION=12.4
set SP_PIN_CUDA_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v%SP_PIN_CUDA_VERSION%
set SP_PIN_VULKAN_MIN=1.3.250
set SP_PIN_HEXAGON_SDK=C:\Qualcomm\Hexagon_SDK\5.4.0.x
set SP_PIN_NINJA_MIN=1.10
set SP_PIN_CMAKE_MIN=3.20
set SP_PIN_GIT_USR_BIN=C:\Program Files\Git\usr\bin

REM Build directory naming convention (per backend, parallel coexistence):
REM   build-cpu  build-cpu-dbg  build-cuda  build-vulkan  build-hexagon
set SP_BUILD_TYPE_DEFAULT=Release
set SP_GENERATOR=Ninja

where cmake >nul 2>&1 || echo [env-common] WARNING: cmake not on PATH (need ^>= %SP_PIN_CMAKE_MIN%).
where ninja >nul 2>&1 || echo [env-common] WARNING: ninja not on PATH (need ^>= %SP_PIN_NINJA_MIN%).

echo [env-common] paths pinned: SP_REPO_ROOT=%SP_REPO_ROOT%
echo [env-common] toolchain pins: VS2019 BT, CUDA %SP_PIN_CUDA_VERSION%, Vulkan ^>= %SP_PIN_VULKAN_MIN%, Hexagon SDK %SP_PIN_HEXAGON_SDK%
