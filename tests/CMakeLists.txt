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
