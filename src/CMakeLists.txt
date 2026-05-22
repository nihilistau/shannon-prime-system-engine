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
    # Phase 2-FMT: .sp-model on-disk format (loader + adapter + hash primitives).
    io/sp_hash.c
    io/sp_model_load.c
    io/sp_model_adapter.c
)

target_include_directories(sp_engine PUBLIC ${CMAKE_SOURCE_DIR}/include)
# io/*.c includes "sp_hash.h" from its own directory.
target_include_directories(sp_engine PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/io)

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

# ── Vulkan compute backend (Phase 2-VK). Mirrors the CUDA block: a separate
# static lib (sp_engine_vulkan) so the engine's /W4 host flags stay isolated.
# No cuBLAS analog — the GEMM is a hand-written tiled f32 compute shader. GLSL
# compute shaders compile to SPIR-V at BUILD time via glslc into C initializer-
# list headers (foo.spv.h) that the backend #includes (no runtime .spv loading).
# The C engine sees gemma3_forward_vulkan via SP_ENGINE_WITH_VULKAN +
# vulkan_backend.h. ──
if(SP_ENGINE_WITH_VULKAN)
    find_package(Vulkan REQUIRED)
    if(NOT Vulkan_GLSLC_EXECUTABLE)
        find_program(Vulkan_GLSLC_EXECUTABLE glslc
            HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin")
    endif()
    if(NOT Vulkan_GLSLC_EXECUTABLE)
        message(FATAL_ERROR "sp_engine: glslc not found (need the Vulkan SDK on PATH)")
    endif()

    set(SP_VK_SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/backends/vulkan/shaders)
    set(SP_VK_SPV_DIR    ${CMAKE_CURRENT_BINARY_DIR}/vk_spv)
    file(MAKE_DIRECTORY ${SP_VK_SPV_DIR})
    set(SP_VK_SHADERS
        gemm embed_scale rmsnorm rmsnorm_head rope attn attn_ntt
        gelu_mul silu_mul add dequant_arena)
    set(SP_VK_SPV_HEADERS)
    foreach(s ${SP_VK_SHADERS})
        set(_src ${SP_VK_SHADER_DIR}/${s}.comp)
        set(_hdr ${SP_VK_SPV_DIR}/${s}.spv.h)
        add_custom_command(
            OUTPUT ${_hdr}
            COMMAND ${Vulkan_GLSLC_EXECUTABLE} -O --target-env=vulkan1.1
                    -mfmt=c -o ${_hdr} ${_src}
            DEPENDS ${_src}
            COMMENT "glslc ${s}.comp -> ${s}.spv.h"
            VERBATIM)
        list(APPEND SP_VK_SPV_HEADERS ${_hdr})
    endforeach()
    add_custom_target(sp_vk_shaders DEPENDS ${SP_VK_SPV_HEADERS})

    add_library(sp_engine_vulkan STATIC
        backends/vulkan/vulkan_backend.cpp
        backends/vulkan/vulkan_forward.cpp
    )
    add_dependencies(sp_engine_vulkan sp_vk_shaders)
    # vulkan_forward.cpp #includes the generated *.spv.h — make the object rebuild
    # whenever a shader changes (Ninja's initial /showIncludes scan can't see the
    # headers before they exist, so declare the dependency explicitly).
    set_source_files_properties(backends/vulkan/vulkan_forward.cpp
        PROPERTIES OBJECT_DEPENDS "${SP_VK_SPV_HEADERS}")
    target_include_directories(sp_engine_vulkan PUBLIC ${CMAKE_SOURCE_DIR}/include)
    target_include_directories(sp_engine_vulkan PRIVATE ${SP_VK_SPV_DIR})
    # Math-core packed formats the Vulkan kernels read directly (arena Q8/Q4) +
    # KSTE host encode; the Vulkan loader for device mgmt and dispatch.
    target_link_libraries(sp_engine_vulkan PUBLIC
        Vulkan::Vulkan
        sp_frobenius sp_vht2 sp_kste)
    # sp_engine consumers (tests) pull the Vulkan lib transitively, and the C
    # sources compile the SP_BACKEND=vulkan dispatch branch.
    target_link_libraries(sp_engine PUBLIC sp_engine_vulkan)
    target_compile_definitions(sp_engine PUBLIC SP_ENGINE_WITH_VULKAN=1)
    message(STATUS "sp_engine: Vulkan backend ON (glslc=${Vulkan_GLSLC_EXECUTABLE})")
endif()
