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
