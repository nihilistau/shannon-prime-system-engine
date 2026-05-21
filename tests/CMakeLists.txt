# Engine tests (CTest). The reference models live out-of-tree under
# d:\Files\models; override -DSP_QWEN3_GGUF=... to point elsewhere.
set(SP_QWEN3_GGUF
    "D:/Files/models/Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-f16.gguf"
    CACHE FILEPATH "Qwen3-0.6B f16 GGUF used by the engine tests")

add_executable(test_loader test_loader.c)
target_link_libraries(test_loader PRIVATE sp_engine)
target_compile_definitions(test_loader PRIVATE SP_QWEN3_GGUF="${SP_QWEN3_GGUF}")
if(MSVC)
    target_compile_options(test_loader PRIVATE /W4)
else()
    target_compile_options(test_loader PRIVATE -Wall -Wextra)
endif()

add_test(NAME E_CPU_1 COMMAND test_loader)

add_executable(test_model test_model.c)
target_link_libraries(test_model PRIVATE sp_engine)
target_compile_definitions(test_model PRIVATE SP_QWEN3_GGUF="${SP_QWEN3_GGUF}")
if(MSVC)
    target_compile_options(test_model PRIVATE /W4)
else()
    target_compile_options(test_model PRIVATE -Wall -Wextra)
endif()
add_test(NAME MODEL_BIND COMMAND test_model)

set(SP_QWEN3_REF
    "D:/Files/models/Qwen/Qwen3-0.6B-GGUF/qwen3_ref.bin"
    CACHE FILEPATH "Oracle logit dump (tools/oracle/dump_logits) for E_CPU_2")
add_executable(test_forward test_forward.c)
target_link_libraries(test_forward PRIVATE sp_engine)
target_compile_definitions(test_forward PRIVATE
    SP_QWEN3_GGUF="${SP_QWEN3_GGUF}" SP_QWEN3_REF="${SP_QWEN3_REF}")
if(MSVC)
    target_compile_options(test_forward PRIVATE /W4)
else()
    target_compile_options(test_forward PRIVATE -Wall -Wextra)
endif()
add_test(NAME E_CPU_2 COMMAND test_forward)

add_executable(test_frob test_frob.c)
target_link_libraries(test_frob PRIVATE sp_engine)
target_compile_definitions(test_frob PRIVATE
    SP_QWEN3_GGUF="${SP_QWEN3_GGUF}" SP_QWEN3_REF="${SP_QWEN3_REF}")
if(MSVC)
    target_compile_options(test_frob PRIVATE /W4)
else()
    target_compile_options(test_frob PRIVATE -Wall -Wextra)
endif()
add_test(NAME E_CPU_3 COMMAND test_frob)

add_executable(test_avx test_avx.c)
target_link_libraries(test_avx PRIVATE sp_engine)
target_compile_definitions(test_avx PRIVATE
    SP_QWEN3_GGUF="${SP_QWEN3_GGUF}" SP_QWEN3_REF="${SP_QWEN3_REF}")
if(SP_ENGINE_WITH_AVX2)
    target_compile_definitions(test_avx PRIVATE SP_ENGINE_AVX2=1)
endif()
if(MSVC)
    target_compile_options(test_avx PRIVATE /W4)
else()
    target_compile_options(test_avx PRIVATE -Wall -Wextra)
endif()
add_test(NAME E_CPU_4 COMMAND test_avx)

add_executable(test_ntt_attn test_ntt_attn.c)
target_link_libraries(test_ntt_attn PRIVATE sp_engine)
target_compile_definitions(test_ntt_attn PRIVATE
    SP_QWEN3_GGUF="${SP_QWEN3_GGUF}" SP_QWEN3_REF="${SP_QWEN3_REF}")
if(MSVC)
    target_compile_options(test_ntt_attn PRIVATE /W4)
else()
    target_compile_options(test_ntt_attn PRIVATE -Wall -Wextra)
endif()
add_test(NAME E_CPU_5 COMMAND test_ntt_attn)

add_executable(test_kste_kv test_kste_kv.c)
target_link_libraries(test_kste_kv PRIVATE sp_engine)
target_compile_definitions(test_kste_kv PRIVATE
    SP_QWEN3_GGUF="${SP_QWEN3_GGUF}" SP_QWEN3_REF="${SP_QWEN3_REF}")
if(MSVC)
    target_compile_options(test_kste_kv PRIVATE /W4)
else()
    target_compile_options(test_kste_kv PRIVATE -Wall -Wextra)
endif()
add_test(NAME E_CPU_6 COMMAND test_kste_kv)

add_executable(test_generate test_generate.c)
target_link_libraries(test_generate PRIVATE sp_engine)
target_compile_definitions(test_generate PRIVATE
    SP_QWEN3_GGUF="${SP_QWEN3_GGUF}" SP_QWEN3_REF="${SP_QWEN3_REF}")
if(MSVC)
    target_compile_options(test_generate PRIVATE /W4)
else()
    target_compile_options(test_generate PRIVATE -Wall -Wextra)
endif()
add_test(NAME GEN_QWEN3 COMMAND test_generate)

add_executable(test_tokenizer test_tokenizer.c)
target_link_libraries(test_tokenizer PRIVATE sp_engine)
target_compile_definitions(test_tokenizer PRIVATE
    SP_QWEN3_GGUF="${SP_QWEN3_GGUF}" SP_QWEN3_REF="${SP_QWEN3_REF}")
if(MSVC)
    target_compile_options(test_tokenizer PRIVATE /W4)
else()
    target_compile_options(test_tokenizer PRIVATE -Wall -Wextra)
endif()
add_test(NAME TOK_DECODE COMMAND test_tokenizer)
add_test(NAME TOK_ENCODE COMMAND test_tokenizer)

add_executable(test_q4 test_q4.c)
target_link_libraries(test_q4 PRIVATE sp_engine)
target_compile_definitions(test_q4 PRIVATE
    SP_QWEN3_GGUF="${SP_QWEN3_GGUF}" SP_QWEN3_REF="${SP_QWEN3_REF}")
if(MSVC)
    target_compile_options(test_q4 PRIVATE /W4)
else()
    target_compile_options(test_q4 PRIVATE -Wall -Wextra)
endif()
add_test(NAME E_CPU_7 COMMAND test_q4)

add_executable(test_kv_spinor test_kv_spinor.c)
target_link_libraries(test_kv_spinor PRIVATE sp_engine)
target_compile_definitions(test_kv_spinor PRIVATE
    SP_QWEN3_GGUF="${SP_QWEN3_GGUF}" SP_QWEN3_REF="${SP_QWEN3_REF}")
if(MSVC)
    target_compile_options(test_kv_spinor PRIVATE /W4)
else()
    target_compile_options(test_kv_spinor PRIVATE -Wall -Wextra)
endif()
add_test(NAME E_CPU_8 COMMAND test_kv_spinor)

add_executable(test_gen_kv test_gen_kv.c)
target_link_libraries(test_gen_kv PRIVATE sp_engine)
target_compile_definitions(test_gen_kv PRIVATE
    SP_QWEN3_GGUF="${SP_QWEN3_GGUF}" SP_QWEN3_REF="${SP_QWEN3_REF}")
if(MSVC)
    target_compile_options(test_gen_kv PRIVATE /W4)
else()
    target_compile_options(test_gen_kv PRIVATE -Wall -Wextra)
endif()
add_test(NAME GEN_KV COMMAND test_gen_kv)

add_executable(test_arena test_arena.c)
target_link_libraries(test_arena PRIVATE sp_engine)
target_compile_definitions(test_arena PRIVATE
    SP_QWEN3_GGUF="${SP_QWEN3_GGUF}" SP_QWEN3_REF="${SP_QWEN3_REF}")
if(MSVC)
    target_compile_options(test_arena PRIVATE /W4)
else()
    target_compile_options(test_arena PRIVATE -Wall -Wextra)
endif()
add_test(NAME E_CPU_9 COMMAND test_arena)
