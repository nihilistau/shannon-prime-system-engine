# ============================================================================
# Sprint WIRE-HEX — daemon-linkable Hexagon V69 backend static library.
#
# Builds `libsp_hex_daemon_backend.a` for aarch64-android, consumed by
# sp_daemon's build.rs link step when SP_DAEMON_BACKEND=hex is intended.
# Companion to the engine's own SP_ENGINE_WITH_HEXAGON sp_engine target
# (src/backends/hexagon/CMakeLists.txt:76-87) but compiled STANDALONE so
# the daemon can link it WITHOUT pulling in the entire sp_engine library
# (which would duplicate qwen3_forward, gemma3_forward, etc. symbols
# the math-core forward path already provides via sp_session).
#
# Sources:
#   src/backends/hexagon/sp_hex_host.c   — gemma3_forward_hexagon entry
#   src/backends/cpu/cpu_overlay.c       — matmul/embed_row/as_f32/sp_kernels_read_env
#                                          (transitive deps of sp_hex_host)
#   sp_hex_stub.c                        — qaic-generated FastRPC client stub
#   sp_daemon_hex_glue.c                 — C glue exposing the L1 forward dispatcher
#                                          signature; casts to gemma3_forward_hexagon
#
# Math-core archives (sp_arena, sp_frobenius, sp_weight_dtype, sp_model,
# sp_gguf, sp_io_format) are linked by sp_daemon's build.rs ALREADY (the
# L3.FG cross-compile output). This lib stays light: it doesn't depend on
# the engine's other backends or the forward orchestration.
#
# Build target:    aarch64-linux-android (NDK r27d, API 21, arm64-v8a)
# Link surface:    libcdsprpc.so + rpcmem.a from Hexagon SDK 5.5.6.0
# Output path:     build-android-hex-backend/libsp_hex_daemon_backend.a
# ============================================================================

cmake_minimum_required(VERSION 3.20)
project(sp_hex_daemon_backend C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Engine root = three levels up from this CMakeLists (tools/sp_daemon/c_backend/).
get_filename_component(ENGINE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)

# ── Hexagon SDK paths (mirrors src/backends/hexagon/CMakeLists.txt:32-36) ──
if(NOT DEFINED ENV{HEXAGON_SDK_ROOT})
    message(FATAL_ERROR "HEXAGON_SDK_ROOT not set — source env-hexagon.bat or pass -DHEXAGON_SDK_ROOT=...")
endif()
set(HEXSDK $ENV{HEXAGON_SDK_ROOT})

set(SP_HEX_RPCMEM_INC  "${HEXSDK}/ipc/fastrpc/rpcmem/inc")
set(SP_HEX_STDDEF_INC  "${HEXSDK}/incs/stddef")
set(SP_HEX_FASTRPC_INC "${HEXSDK}/incs")
set(SP_HEX_RPCMEM_LIB  "${HEXSDK}/ipc/fastrpc/rpcmem/prebuilt/android_aarch64/rpcmem.a")
set(SP_HEX_CDSPRPC_LIB "${HEXSDK}/ipc/fastrpc/remote/ship/android_aarch64/libcdsprpc.so")

# ── qaic codegen for sp_hex.idl → host stub + header ──
set(SP_HEX_IDL  "${ENGINE_ROOT}/src/backends/hexagon/inc/sp_hex.idl")
set(SP_HEX_GEN  "${CMAKE_CURRENT_BINARY_DIR}/idl_gen")
set(SP_HEX_QAIC "${HEXSDK}/ipc/fastrpc/qaic/WinNT/qaic.exe")
file(MAKE_DIRECTORY ${SP_HEX_GEN})

add_custom_command(
    OUTPUT ${SP_HEX_GEN}/sp_hex.h ${SP_HEX_GEN}/sp_hex_stub.c ${SP_HEX_GEN}/sp_hex_skel.c
    COMMAND "${SP_HEX_QAIC}" -mdll -o ${SP_HEX_GEN}
            -I${HEXSDK}/incs/ -I${HEXSDK}/incs/stddef/ ${SP_HEX_IDL}
    DEPENDS ${SP_HEX_IDL}
    COMMENT "qaic sp_hex.idl -> host stub + header (daemon)"
    VERBATIM)
add_custom_target(sp_hex_daemon_idl
    DEPENDS ${SP_HEX_GEN}/sp_hex.h ${SP_HEX_GEN}/sp_hex_stub.c)

# ── The static library the daemon links ──
add_library(sp_hex_daemon_backend STATIC
    ${ENGINE_ROOT}/src/backends/hexagon/sp_hex_host.c
    ${ENGINE_ROOT}/src/backends/cpu/cpu_overlay.c
    ${SP_HEX_GEN}/sp_hex_stub.c
    ${CMAKE_CURRENT_SOURCE_DIR}/sp_daemon_hex_glue.c
)
add_dependencies(sp_hex_daemon_backend sp_hex_daemon_idl)

target_include_directories(sp_hex_daemon_backend PUBLIC
    ${ENGINE_ROOT}/include
    ${ENGINE_ROOT}/lib/shannon-prime-system/include
    ${ENGINE_ROOT}/src/backends/hexagon
    ${SP_HEX_GEN}
    ${SP_HEX_RPCMEM_INC}
    ${SP_HEX_STDDEF_INC}
    ${SP_HEX_FASTRPC_INC}
)

# cpu_overlay was conditionally AVX-gated in its production build; for the daemon
# backend we deliberately turn it OFF (sp_hex_host pushes the compute to the
# cDSP; the host side only needs portable C). Mirrors build-hexagon.bat:40 which
# passes -DSP_ENGINE_WITH_AVX2=OFF for the on-phone test_ppl variant.
target_compile_definitions(sp_hex_daemon_backend PRIVATE
    SP_ENGINE_WITH_HEXAGON=1
)

# The daemon links cdsprpc + rpcmem at the binary level (build.rs cargo:rustc-link-lib).
# We just need them on the include path for the stub TU; the static lib is
# unresolved-symbol clean apart from these (resolved at sp-daemon link time).

message(STATUS "sp_hex_daemon_backend: hexagon V69 backend, daemon-link variant")
message(STATUS "  sources: sp_hex_host.c + cpu_overlay.c + qaic stub + glue")
message(STATUS "  output:  ${CMAKE_CURRENT_BINARY_DIR}/libsp_hex_daemon_backend.a")
