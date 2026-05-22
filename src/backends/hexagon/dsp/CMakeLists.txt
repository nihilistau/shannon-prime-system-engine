# Phase 2-HX — cDSP HVX skel (Hexagon V69). Built SEPARATELY from the engine by
# the SDK's `build_cmake hexagon` (hexagon-clang V69), driven by
# scripts/build/build-hexagon.bat dsp. Produces libsp_hex_skel.so, pushed to the
# phone alongside the NDK-built host runner.
#
# Structural mirror of the SDK FastRPC example DSP branch (S22U/calculator) —
# recreated fresh, no code copied. qaic generates sp_hex_skel.c + sp_hex.h from
# ../inc/sp_hex.idl into the build dir; the skel dispatches to sp_hex_imp.c.
cmake_minimum_required(VERSION 3.14.3)
project(sp_hex_dsp C CXX ASM)
enable_language(ASM)

include(${HEXAGON_CMAKE_ROOT}/hexagon_fun.cmake)

set(SP_HEX_IDL ${CMAKE_CURRENT_SOURCE_DIR}/../inc/sp_hex.idl)

set(common_incs
    ${CMAKE_CURRENT_SOURCE_DIR}/
    ${CMAKE_CURRENT_BINARY_DIR}/
    ${HEXAGON_SDK_ROOT}/incs/
    ${HEXAGON_SDK_ROOT}/incs/stddef/
    ${HEXAGON_SDK_ROOT}/rtos/qurt/
    ${HEXAGON_SDK_ROOT}/ipc/fastrpc/rpcmem/inc/
    ${HEXAGON_SDK_ROOT}/utils/examples
    )

# Skel = qaic-generated dispatch (sp_hex_skel) + our DSP impl (sp_hex_imp).
set(srcs
    ${CMAKE_CURRENT_BINARY_DIR}/sp_hex_skel
    ${CMAKE_CURRENT_SOURCE_DIR}/sp_hex_imp
    ${HEXAGON_SDK_ROOT}/utils/examples/dsp_capabilities_utils
    )

add_library(sp_hex_skel SHARED ${srcs})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions")
include_directories(${common_incs})

# qaic: ../inc/sp_hex.idl -> sp_hex.h + sp_hex_skel.c + sp_hex_stub.c in the build dir.
build_idl(${SP_HEX_IDL} sp_hex_skel)

copy_binaries(sp_hex_skel)
