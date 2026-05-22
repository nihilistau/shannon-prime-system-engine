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
