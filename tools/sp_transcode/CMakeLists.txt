# sp_transcode — GGUF -> .sp-model + .sp-tokenizer transcoder (Phase 2-FMT,
# E_FMT_2/E_FMT_3). Standalone CLI; links the engine for GGUF parse + the
# Frobenius packer + the format/hash primitives.
add_executable(sp_transcode sp_transcode.c)
target_link_libraries(sp_transcode PRIVATE sp_engine)
# sp_transcode.c includes "sp_hash.h" from src/io.
target_include_directories(sp_transcode PRIVATE ${CMAKE_SOURCE_DIR}/src/io)
if(MSVC)
    target_compile_options(sp_transcode PRIVATE /W4)
else()
    target_compile_options(sp_transcode PRIVATE -Wall -Wextra)
endif()
