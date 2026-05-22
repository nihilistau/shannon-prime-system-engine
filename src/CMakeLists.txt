# Engine core library. Grows per 2-CPU subphase:
#   2-CPU.A  loader/gguf.c           (GGUF parse)              <-- here
#   2-CPU.B  forward/*.c             (Qwen3 forward pass)
#   2-CPU.C+ backends/cpu/*.c        (Frobenius matmul, AVX2, NTT-attn, KV)
add_library(sp_engine STATIC
    loader/gguf.c
    forward/model.c
    forward/kernels.c
    forward/forward.c
    forward/gemma3.c
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
