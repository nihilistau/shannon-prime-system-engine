# toolchain-hexagon.cmake
#
# CMake toolchain for the Hexagon backend.  Used when SP_ENGINE_BACKEND=hexagon.
# Targets Snapdragon 8 Gen 1 V69 HTP via the Hexagon SDK on a Windows host.
#
# Phase 0 placeholder.  Phase 2-HX (Hexagon backend bootstrap) fills in:
#   - host stub library (libsp_engine_hexagon_stub.dll) calling FastRPC
#   - device .so target compiled with hexagon-clang
#   - .idl interface definitions
#   - rpcmem alloc + dispatch helpers
#
# Critical constants (per project memory):
#   * SP_FASTRPC_STRICT_ALLOC=1 — rpcmem allocation size MUST equal IDL length.
#   * Prepend Git for Windows sh.exe to PATH before invoking the SDK.
#   * qaic lives at $HEXAGON_SDK_ROOT\ipc\fastrpc\qaic\WinNT\qaic.exe on
#     Windows hosts (not bin/qaic as Linux samples assume).

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR hexagonv69)

if(NOT DEFINED ENV{HEXAGON_SDK_ROOT})
    message(FATAL_ERROR
        "HEXAGON_SDK_ROOT not set. Run scripts/env/env-hexagon.bat first.")
endif()

set(HEXAGON_SDK_ROOT $ENV{HEXAGON_SDK_ROOT})
set(HEXAGON_TOOLS_VER $ENV{HEXAGON_TOOLS_VER})

# Compiler discovery is deferred to the Phase 2-HX bring-up.
# message(STATUS "Hexagon SDK: ${HEXAGON_SDK_ROOT}, tools ${HEXAGON_TOOLS_VER}")

# Placeholder marker so the build doesn't silently fall through.
message(STATUS "toolchain-hexagon.cmake: Phase 0 placeholder.  Phase 2-HX fills in.")
