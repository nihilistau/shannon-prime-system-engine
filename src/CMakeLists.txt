# Engine core library. Grows per 2-CPU subphase:
#   2-CPU.A  loader/gguf.c           (GGUF parse)              <-- here
#   2-CPU.B  forward/*.c             (Qwen3 forward pass)
#   2-CPU.C+ backends/cpu/*.c        (Frobenius matmul, AVX2, NTT-attn, KV)
add_library(sp_engine STATIC
    loader/gguf.c
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
