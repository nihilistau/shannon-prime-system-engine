# §3-HX Sprint A echo skel — SDK CMake template per S22U workspace pattern.
# Source: C:\Qualcomm\Hexagon_IDE\S22U\CMakeLists.txt (working on Knack's
# S22U device 2026-05-29; produces a loadable libS22U_skel.so).  This file
# mirrors that pattern; only the project name and IDL file change.

cmake_minimum_required(VERSION 3.14.3)
project(sp_echo C ASM)
enable_language(ASM)

include(${HEXAGON_CMAKE_ROOT}/hexagon_fun.cmake)

set(common_incs
    ${CMAKE_CURRENT_SOURCE_DIR}/src_app/
    ${CMAKE_CURRENT_BINARY_DIR}/
    ${HEXAGON_SDK_ROOT}/incs/
    ${HEXAGON_SDK_ROOT}/incs/stddef/
    ${HEXAGON_SDK_ROOT}/rtos/qurt/
    ${HEXAGON_SDK_ROOT}/ipc/fastrpc/rpcmem/inc/
    ${HEXAGON_SDK_ROOT}/utils/examples
)

if(${OS_TYPE} MATCHES "HLOS")
    # No Android stub needed for Sprint A — our Rust dsp_rpc.rs binds dynamically
    # via libcdsprpc.so + the qaic-emitted sp_echo_URI macro from sp_echo.h.
    # If a host-side stub library is later needed for integration with the
    # daemon's C-side, add the S22U.h-style add_library(sp_echo SHARED ...) here.
else()
    # ── Hexagon dynamic skel library ─────────────────────────────────────
    set(srcs
        ${CMAKE_CURRENT_BINARY_DIR}/sp_echo_skel
        ${CMAKE_CURRENT_SOURCE_DIR}/src_dsp/sp_echo_imp
    )
    add_library(sp_echo_skel SHARED ${srcs})
    include_directories(${common_incs})
    build_idl("inc/sp_echo.idl" sp_echo_skel incs)
endif()
