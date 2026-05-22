# Engine core library. Grows per 2-CPU subphase:
#   2-CPU.A  loader/gguf.c           (GGUF parse)              <-- here
#   2-CPU.B  forward/*.c             (Qwen3 forward pass)
#   2-CPU.C+ backends/cpu/*.c        (Frobenius matmul, AVX2, NTT-attn, KV)
add_library(sp_engine STATIC
    common/sp_error.c
    loader/gguf.c
    forward/model.c
    forward/kernels.c
    forward/forward.c
    forward/gemma3.c
    forward/ppl.c
    forward/generate.c
    forward/arena.c
    tokenizer/tokenizer.c
)

target_include_directories(sp_engine PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Math-core primitives the forward pass / kernels consume (PUBLIC: their
# include/sp headers propagate to engine consumers and tests).
target_link_libraries(sp_engine PUBLIC
    sp_frobenius sp_ntt_crt sp_poly_ring sp_kste sp_vht2)

if(MSVC)
    target_compile_options(sp_engine PRIVATE /W4)
else()
    target_compile_options(sp_engine PRIVATE -Wall -Wextra -Wshadow)
endif()

# AVX2 / AVX512 matmul path (E_CPU_4). Compiling with the ISA enables the
# intrinsics under SP_ENGINE_AVX2 / SP_ENGINE_AVX512; the scalar reference stays
# selectable at runtime via SP_CPU_SCALAR=1 (roadmap §8.7).
if(SP_ENGINE_WITH_AVX2)
    target_compile_definitions(sp_engine PRIVATE SP_ENGINE_AVX2=1)
    if(MSVC)
        target_compile_options(sp_engine PRIVATE /arch:AVX2)
    else()
        target_compile_options(sp_engine PRIVATE -mavx2 -mfma)
    endif()
endif()
if(SP_ENGINE_WITH_AVX512)
    target_compile_definitions(sp_engine PRIVATE SP_ENGINE_AVX512=1)
    if(MSVC)
        target_compile_options(sp_engine PRIVATE /arch:AVX512)
    else()
        target_compile_options(sp_engine PRIVATE -mavx512f)
    endif()
endif()

# ── CUDA backend (Phase 2-CU). Separate static lib so the engine's /W4 +
# /arch:AVX2 host flags never reach nvcc, and vice versa. cuBLAS for the GEMM
# bulk; hand-written kernels (CU.1+) for the discrete-algebra ops. The C engine
# sees gemma3_forward_cuda via SP_ENGINE_WITH_CUDA + cuda_backend.h. ──
if(SP_ENGINE_WITH_CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_library(sp_engine_cuda STATIC
        backends/cuda/cuda_backend.cu
        backends/cuda/cuda_forward.cu
    )
    set_target_properties(sp_engine_cuda PROPERTIES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON)
    target_include_directories(sp_engine_cuda PUBLIC ${CMAKE_SOURCE_DIR}/include)
    # Math-core packed formats the CUDA kernels read directly (arena Q8/Q4 +
    # Spinor KV); CUDA runtime + cuBLAS for device mgmt and SGEMM.
    target_link_libraries(sp_engine_cuda PUBLIC
        CUDA::cudart CUDA::cublas
        sp_frobenius sp_vht2)
    # sp_engine consumers (tests) pull the CUDA lib transitively, and the C
    # sources compile the SP_BACKEND=cuda dispatch branch.
    target_link_libraries(sp_engine PUBLIC sp_engine_cuda)
    target_compile_definitions(sp_engine PUBLIC SP_ENGINE_WITH_CUDA=1)
    message(STATUS "sp_engine: CUDA backend ON (arch=${CMAKE_CUDA_ARCHITECTURES})")
endif()

# ── Hexagon V69 HTP backend (Phase 2-HX). Mirrors the CUDA block's shape, but
# the host side cross-compiles to aarch64-android (FastRPC is intra-device IPC,
# Android-arm -> cDSP) and the DSP HVX skel builds separately via the SDK. The
# subdir early-returns unless SP_ENGINE_TARGET_ANDROID. HX.2-prep: rpcmem_probe
# (proved 574.5 MB Q8 arena single-buffer upload feasible). HX.2+: IDL stub +
# DSP skel + the SP_BACKEND=hexagon dispatch. ──
add_subdirectory(backends/hexagon)
