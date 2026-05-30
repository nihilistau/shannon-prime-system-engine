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
        ${CMAKE_CURRENT_SOURCE_DIR}/src_dsp/sp_compute_crt_imp
        ${CMAKE_CURRENT_SOURCE_DIR}/src_dsp/sp_compute_ntt_imp
        # §4-NTT Sprint NTT.1 — HVX butterfly core (method 13).
        ${CMAKE_CURRENT_SOURCE_DIR}/src_dsp/sp_compute_ntt_hvx_imp

        # §4-NTT Sprint NTT.2 — twiddle VTCM staging
        ${CMAKE_CURRENT_SOURCE_DIR}/src_dsp/sp_compute_ntt_twiddle

        # §4-NTT Sprint NTT.3 -- dual-prime CRT dispatch (VTCM-aware HVX, method 17)
        ${CMAKE_CURRENT_SOURCE_DIR}/src_dsp/sp_compute_ntt_hvx_vtcm_imp
        # §4-NTT Sprint NTT.4 — INTT HVX skel (method 18)
        ${CMAKE_CURRENT_SOURCE_DIR}/src_dsp/sp_compute_ntt_intt_imp
    )
    add_library(sp_compute_skel SHARED ${srcs})
    include_directories(${common_incs})
    build_idl("inc/sp_compute.idl" sp_compute_skel incs)

    # §3-HX Sprint F: link the Halide AOT-emitted kernel.  The .o is built by
    # tools/sp_halide_gen/build.cmd (host MSVC + Halide.lib → Hexagon ELF .o)
    # and lives under halide_gen/.  The generated header lives next to it and is
    # included by sp_compute_imp.c via include_directories below.
    set(HALIDE_GEN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/halide_gen)
    if(EXISTS ${HALIDE_GEN_DIR}/sp_axpby_2d_halide.o)
        include_directories(${HALIDE_GEN_DIR})
        target_link_libraries(sp_compute_skel PRIVATE ${HALIDE_GEN_DIR}/sp_axpby_2d_halide.o)
        target_compile_definitions(sp_compute_skel PRIVATE SP_HAVE_HALIDE=1)
    else()
        message(WARNING "sp_compute_skel: halide_gen/sp_axpby_2d_halide.o missing — Sprint F Halide kernel disabled (run tools/sp_halide_gen/build.cmd first)")
    endif()
    if(EXISTS ${HALIDE_GEN_DIR}/sp_ffn_2stage_halide.o)
        target_link_libraries(sp_compute_skel PRIVATE ${HALIDE_GEN_DIR}/sp_ffn_2stage_halide.o)
    else()
        message(WARNING "sp_compute_skel: halide_gen/sp_ffn_2stage_halide.o missing — Sprint G FFN kernel disabled (run tools/sp_halide_gen/build.cmd first)")
    endif()
    if(EXISTS ${HALIDE_GEN_DIR}/sp_ffn_2stage_diag_halide.o)
        target_link_libraries(sp_compute_skel PRIVATE ${HALIDE_GEN_DIR}/sp_ffn_2stage_diag_halide.o)
    else()
        message(WARNING "sp_compute_skel: halide_gen/sp_ffn_2stage_diag_halide.o missing — Sprint H diagnostic kernel disabled")
    endif()
endif()
