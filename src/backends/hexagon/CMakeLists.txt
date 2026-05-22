# ── Hexagon V69 HTP backend (Phase 2-HX). ─────────────────────────────────────
# MIRRORS the CUDA block's SHAPE (a backend lib + tools behind a guard), but the
# build mechanics fundamentally diverge: FastRPC is intra-device IPC (Android-arm
# -> cDSP), so the host side cross-compiles to aarch64-android (this CMake, via
# the NDK toolchain) and the DSP-side HVX skel builds SEPARATELY with the SDK's
# build_cmake (hexagon-clang V69, under dsp/). See scripts/build/build-hexagon.bat
# and SESSION-STATE-lat-2-HX.md for the architecture.
#
# This file is only meaningful in the SP_ENGINE_TARGET_ANDROID cross build (the
# host side runs on the phone). It needs HEXAGON_SDK_ROOT in the env
# (env-hexagon.bat sets it).
#
# HX.2-prep: rpcmem_probe — a standalone capacity probe (NOT linked into the
# engine). Proved (2026-05-23, R5CT22445JA) that a single SYSTEM-heap rpcmem
# alloc succeeds to >=2000 MB, so the 574.5 MB Q8 arena uploads in ONE buffer
# (the CUDA build_weights-cache analog). HX.2+ adds the IDL stub + DSP skel here.

if(NOT SP_ENGINE_TARGET_ANDROID)
    return()
endif()

if(NOT DEFINED ENV{HEXAGON_SDK_ROOT})
    message(WARNING "hexagon backend: HEXAGON_SDK_ROOT not set — run env-hexagon.bat. "
                    "Skipping the FastRPC host-stub targets.")
    return()
endif()

set(HEXSDK $ENV{HEXAGON_SDK_ROOT})

# FastRPC host-side (= phone's ARM) deps: rpcmem static lib + the cdsprpc stub +
# the AEEStdDef / mmdefs headers. EXACT paths verified on SDK 5.5.6.0.
set(SP_HEX_RPCMEM_INC  "${HEXSDK}/ipc/fastrpc/rpcmem/inc")
set(SP_HEX_STDDEF_INC  "${HEXSDK}/incs/stddef")
set(SP_HEX_FASTRPC_INC "${HEXSDK}/incs")
set(SP_HEX_RPCMEM_LIB  "${HEXSDK}/ipc/fastrpc/rpcmem/prebuilt/android_aarch64/rpcmem.a")
set(SP_HEX_CDSPRPC_LIB "${HEXSDK}/ipc/fastrpc/remote/ship/android_aarch64/libcdsprpc.so")

# rpcmem capacity probe (HX.2-prep). Pure rpcmem + cdsprpc, no DSP/IDL.
add_executable(rpcmem_probe rpcmem_probe.c)
target_include_directories(rpcmem_probe PRIVATE
    ${SP_HEX_RPCMEM_INC} ${SP_HEX_STDDEF_INC} ${SP_HEX_FASTRPC_INC})
target_link_libraries(rpcmem_probe PRIVATE
    ${SP_HEX_RPCMEM_LIB} ${SP_HEX_CDSPRPC_LIB} log dl)
message(STATUS "sp_engine: Hexagon HX.2-prep rpcmem_probe target ON")

# ── HX.2 FastRPC round-trip (sp_hex IDL). qaic generates the host stub + header
# from inc/sp_hex.idl (same IDL the DSP skel under dsp/ uses); test_hex_rt links
# the stub + rpcmem + cdsprpc and pings the cDSP. Generated files are build
# artifacts (not committed) — only inc/sp_hex.idl is. ──
set(SP_HEX_IDL ${CMAKE_CURRENT_SOURCE_DIR}/inc/sp_hex.idl)
set(SP_HEX_GEN ${CMAKE_CURRENT_BINARY_DIR}/idl_gen)
set(SP_HEX_QAIC "${HEXSDK}/ipc/fastrpc/qaic/WinNT/qaic.exe")
file(MAKE_DIRECTORY ${SP_HEX_GEN})
add_custom_command(
    OUTPUT ${SP_HEX_GEN}/sp_hex.h ${SP_HEX_GEN}/sp_hex_stub.c ${SP_HEX_GEN}/sp_hex_skel.c
    COMMAND "${SP_HEX_QAIC}" -mdll -o ${SP_HEX_GEN}
            -I${HEXSDK}/incs/ -I${HEXSDK}/incs/stddef/ ${SP_HEX_IDL}
    DEPENDS ${SP_HEX_IDL}
    COMMENT "qaic sp_hex.idl -> host stub + header"
    VERBATIM)
add_custom_target(sp_hex_idl DEPENDS ${SP_HEX_GEN}/sp_hex.h ${SP_HEX_GEN}/sp_hex_stub.c)

add_executable(test_hex_rt sp_hex_rt.c ${SP_HEX_GEN}/sp_hex_stub.c)
add_dependencies(test_hex_rt sp_hex_idl)
target_include_directories(test_hex_rt PRIVATE
    ${SP_HEX_GEN} ${SP_HEX_RPCMEM_INC} ${SP_HEX_STDDEF_INC} ${SP_HEX_FASTRPC_INC})
target_link_libraries(test_hex_rt PRIVATE
    ${SP_HEX_RPCMEM_LIB} ${SP_HEX_CDSPRPC_LIB} log dl)
message(STATUS "sp_engine: Hexagon HX.2 test_hex_rt round-trip target ON")

# ── HX.3a forward: compile the FastRPC host stub + the layers-on-DSP orchestration
# (sp_hex_host.c) INTO sp_engine. Compiling into the engine (vs a separate lib)
# avoids a static-library cycle with sp_engine on lld (the host calls embed_row/
# matmul/sp_arena_find; ppl.c calls gemma3_forward_hexagon). Gated by
# SP_ENGINE_WITH_HEXAGON so the desktop CPU/CUDA/VK builds are untouched. ──
if(SP_ENGINE_WITH_HEXAGON)
    target_sources(sp_engine PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/sp_hex_host.c
        ${SP_HEX_GEN}/sp_hex_stub.c)
    add_dependencies(sp_engine sp_hex_idl)
    target_include_directories(sp_engine PRIVATE
        ${SP_HEX_GEN} ${CMAKE_CURRENT_SOURCE_DIR}
        ${SP_HEX_RPCMEM_INC} ${SP_HEX_STDDEF_INC} ${SP_HEX_FASTRPC_INC})
    target_compile_definitions(sp_engine PUBLIC SP_ENGINE_WITH_HEXAGON=1)
    target_link_libraries(sp_engine PUBLIC ${SP_HEX_RPCMEM_LIB} ${SP_HEX_CDSPRPC_LIB} log dl)
    message(STATUS "sp_engine: Hexagon HX.3a forward (sp_hex_host) compiled into sp_engine")
endif()
