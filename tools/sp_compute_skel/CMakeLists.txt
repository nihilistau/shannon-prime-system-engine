# §3-HX Sprint D HVX compute skel — SDK CMake template per S22U workspace pattern.
# Mirrors tools/sp_echo_skel/CMakeLists.txt; only project + IDL change.
# HVX vectorization (-mhvx -mhvx-length=128B) is auto-added by the SDK
# toolchain hexagon_toolchain.cmake:218 for toolv87+.

cmake_minimum_required(VERSION 3.14.3)
project(sp_compute C ASM)
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
    # No Android stub needed for Sprint D — the smoke harness uses dsp_rpc.rs
    # to dynamic-link via libcdsprpc.so + the qaic-emitted URI macro.
else()
    set(srcs
        ${CMAKE_CURRENT_BINARY_DIR}/sp_compute_skel
        ${CMAKE_CURRENT_SOURCE_DIR}/src_dsp/sp_compute_imp
    )
    add_library(sp_compute_skel SHARED ${srcs})
    include_directories(${common_incs})
    build_idl("inc/sp_compute.idl" sp_compute_skel incs)
endif()
