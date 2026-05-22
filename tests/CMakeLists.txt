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

add_executable(test_release test_release.c)
target_link_libraries(test_release PRIVATE sp_engine)
target_compile_definitions(test_release PRIVATE
    SP_QWEN3_GGUF="${SP_QWEN3_GGUF}" SP_QWEN3_REF="${SP_QWEN3_REF}")
if(MSVC)
    target_compile_options(test_release PRIVATE /W4)
else()
    target_compile_options(test_release PRIVATE -Wall -Wextra)
endif()
add_test(NAME E_CPU_10 COMMAND test_release)

add_executable(test_compose test_compose.c)
target_link_libraries(test_compose PRIVATE sp_engine)
target_compile_definitions(test_compose PRIVATE
    SP_QWEN3_GGUF="${SP_QWEN3_GGUF}" SP_QWEN3_REF="${SP_QWEN3_REF}")
if(MSVC)
    target_compile_options(test_compose PRIVATE /W4)
else()
    target_compile_options(test_compose PRIVATE -Wall -Wextra)
endif()
add_test(NAME COMPOSE COMMAND test_compose)

# ── Phase 2-FMT: .sp-model format (E_FMT_0 primitives, E_FMT_4 round-trip) ──
add_executable(test_sp_hash test_sp_hash.c)
target_link_libraries(test_sp_hash PRIVATE sp_engine)
target_include_directories(test_sp_hash PRIVATE ${CMAKE_SOURCE_DIR}/src/io)
if(MSVC)
    target_compile_options(test_sp_hash PRIVATE /W4)
else()
    target_compile_options(test_sp_hash PRIVATE -Wall -Wextra)
endif()
add_test(NAME E_FMT_0 COMMAND test_sp_hash)

# ── Gemma3-1B (2nd architecture; SP1) ──
set(SP_GEMMA3_GGUF
    "D:/Files/Models/Mine/gemma-3-1b-it/gemma-3-1b-it-f16/gemma-3-1b-it-f16.gguf"
    CACHE FILEPATH "Gemma3-1B f16 GGUF for the engine tests")
add_executable(test_gemma3_bind test_gemma3_bind.c)
target_link_libraries(test_gemma3_bind PRIVATE sp_engine)
target_compile_definitions(test_gemma3_bind PRIVATE SP_GEMMA3_GGUF="${SP_GEMMA3_GGUF}")
if(MSVC)
    target_compile_options(test_gemma3_bind PRIVATE /W4)
else()
    target_compile_options(test_gemma3_bind PRIVATE -Wall -Wextra)
endif()
add_test(NAME GEMMA3_BIND COMMAND test_gemma3_bind)

set(SP_GEMMA3_REF
    "D:/Files/Models/Mine/gemma-3-1b-it/gemma-3-1b-it-f16/gemma3_ref.bin"
    CACHE FILEPATH "Gemma3 oracle logit dump (tools/oracle/dump_logits) for M_GEMMA3_CPU")
add_executable(test_gemma3_forward test_gemma3_forward.c)
target_link_libraries(test_gemma3_forward PRIVATE sp_engine)
target_compile_definitions(test_gemma3_forward PRIVATE
    SP_GEMMA3_GGUF="${SP_GEMMA3_GGUF}" SP_GEMMA3_REF="${SP_GEMMA3_REF}")
if(MSVC)
    target_compile_options(test_gemma3_forward PRIVATE /W4)
else()
    target_compile_options(test_gemma3_forward PRIVATE -Wall -Wextra)
endif()
add_test(NAME M_GEMMA3_CPU COMMAND test_gemma3_forward)

add_executable(test_spm test_spm.c)
target_link_libraries(test_spm PRIVATE sp_engine)
target_compile_definitions(test_spm PRIVATE
    SP_GEMMA3_GGUF="${SP_GEMMA3_GGUF}"
    SP_SPM_FIXDIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/spm")
if(MSVC)
    target_compile_options(test_spm PRIVATE /W4)
else()
    target_compile_options(test_spm PRIVATE -Wall -Wextra)
endif()
add_test(NAME SPM_ENCODE COMMAND test_spm)

# E_FMT_1..4 — .sp-model format gates. One binary, four named subtests selected
# by SP_FMT_SUBTEST (the TOK_DECODE/TOK_ENCODE one-binary-many-tests pattern).
# Invokes the sp_transcode binary; writes .sp-model/.sp-tokenizer under the build
# dir. E_FMT_4 (closure gate) transcodes + runs the full forward on both paths.
add_executable(test_sp_model_roundtrip test_sp_model_roundtrip.c)
target_link_libraries(test_sp_model_roundtrip PRIVATE sp_engine)
target_include_directories(test_sp_model_roundtrip PRIVATE ${CMAKE_SOURCE_DIR}/src/io)
add_dependencies(test_sp_model_roundtrip sp_transcode)
target_compile_definitions(test_sp_model_roundtrip PRIVATE
    SP_GEMMA3_GGUF="${SP_GEMMA3_GGUF}"
    SP_QWEN3_GGUF="${SP_QWEN3_GGUF}"
    SP_TRANSCODE_BIN="$<TARGET_FILE:sp_transcode>"
    SP_FMT_OUT_DIR="${CMAKE_CURRENT_BINARY_DIR}")
if(MSVC)
    target_compile_options(test_sp_model_roundtrip PRIVATE /W4)
else()
    target_compile_options(test_sp_model_roundtrip PRIVATE -Wall -Wextra)
endif()
add_test(NAME E_FMT_1 COMMAND test_sp_model_roundtrip)
add_test(NAME E_FMT_2 COMMAND test_sp_model_roundtrip)
add_test(NAME E_FMT_3 COMMAND test_sp_model_roundtrip)
add_test(NAME E_FMT_4 COMMAND test_sp_model_roundtrip)
add_test(NAME E_FMT_4_QWEN3 COMMAND test_sp_model_roundtrip)
set_tests_properties(E_FMT_1 PROPERTIES ENVIRONMENT "SP_FMT_SUBTEST=E_FMT_1" TIMEOUT 600)
set_tests_properties(E_FMT_2 PROPERTIES ENVIRONMENT "SP_FMT_SUBTEST=E_FMT_2" TIMEOUT 600)
set_tests_properties(E_FMT_3 PROPERTIES ENVIRONMENT "SP_FMT_SUBTEST=E_FMT_3" TIMEOUT 600)
set_tests_properties(E_FMT_4 PROPERTIES ENVIRONMENT "SP_FMT_SUBTEST=E_FMT_4" TIMEOUT 1800)
set_tests_properties(E_FMT_4_QWEN3 PROPERTIES ENVIRONMENT "SP_FMT_SUBTEST=E_FMT_4_QWEN3" TIMEOUT 1800)

# T_FRO_4 — Gemma3-1B PPL gate. SLOW / phase-close: run via `ctest -R T_FRO_4`,
# not in the fast every-commit suite (a full PPL pass is minutes).
add_executable(test_ppl test_ppl.c)
target_link_libraries(test_ppl PRIVATE sp_engine)
target_compile_definitions(test_ppl PRIVATE
    SP_GEMMA3_GGUF="${SP_GEMMA3_GGUF}"
    SP_PPL_CORPUS_DEF="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/ppl/wiki.tiny.raw"
    SP_PPL_ORACLE_DEF="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/ppl/wiki.tiny.oracle_ppl.txt")
if(MSVC)
    target_compile_options(test_ppl PRIVATE /W4)
else()
    target_compile_options(test_ppl PRIVATE -Wall -Wextra)
endif()
add_test(NAME T_FRO_4 COMMAND test_ppl)
set_tests_properties(T_FRO_4 PROPERTIES LABELS "SLOW" TIMEOUT 3600)

# ── CUDA backend (Phase 2-CU). Built only when SP_ENGINE_WITH_CUDA; the test
# pulls the CUDA lib transitively through sp_engine's PUBLIC link. ──
if(SP_ENGINE_WITH_CUDA)
    add_executable(test_cuda_smoke test_cuda_smoke.c)
    target_link_libraries(test_cuda_smoke PRIVATE sp_engine)
    if(MSVC)
        target_compile_options(test_cuda_smoke PRIVATE /W4)
    else()
        target_compile_options(test_cuda_smoke PRIVATE -Wall -Wextra)
    endif()
    add_test(NAME CUDA_SMOKE COMMAND test_cuda_smoke)

    add_executable(test_gemma3_cuda test_gemma3_cuda.c)
    target_link_libraries(test_gemma3_cuda PRIVATE sp_engine)
    target_compile_definitions(test_gemma3_cuda PRIVATE SP_GEMMA3_GGUF="${SP_GEMMA3_GGUF}")
    if(MSVC)
        target_compile_options(test_gemma3_cuda PRIVATE /W4)
    else()
        target_compile_options(test_gemma3_cuda PRIVATE -Wall -Wextra)
    endif()
    add_test(NAME M_GEMMA3_CUDA COMMAND test_gemma3_cuda)
    set_tests_properties(M_GEMMA3_CUDA PROPERTIES TIMEOUT 600)

    add_executable(test_qwen3_cuda test_qwen3_cuda.c)
    target_link_libraries(test_qwen3_cuda PRIVATE sp_engine)
    target_compile_definitions(test_qwen3_cuda PRIVATE
        SP_QWEN3_GGUF="${SP_QWEN3_GGUF}" SP_QWEN3_REF="${SP_QWEN3_REF}")
    if(MSVC)
        target_compile_options(test_qwen3_cuda PRIVATE /W4)
    else()
        target_compile_options(test_qwen3_cuda PRIVATE -Wall -Wextra)
    endif()
    add_test(NAME M_QWEN3_CUDA COMMAND test_qwen3_cuda)
    set_tests_properties(M_QWEN3_CUDA PROPERTIES TIMEOUT 600)

    add_executable(test_ntt_attn_cuda test_ntt_attn_cuda.c)
    target_link_libraries(test_ntt_attn_cuda PRIVATE sp_engine)
    target_compile_definitions(test_ntt_attn_cuda PRIVATE
        SP_QWEN3_GGUF="${SP_QWEN3_GGUF}" SP_QWEN3_REF="${SP_QWEN3_REF}")
    if(MSVC)
        target_compile_options(test_ntt_attn_cuda PRIVATE /W4)
    else()
        target_compile_options(test_ntt_attn_cuda PRIVATE -Wall -Wextra)
    endif()
    add_test(NAME E_CU_5 COMMAND test_ntt_attn_cuda)
    set_tests_properties(E_CU_5 PROPERTIES TIMEOUT 600)

    add_executable(test_kste_kv_cuda test_kste_kv_cuda.c)
    target_link_libraries(test_kste_kv_cuda PRIVATE sp_engine)
    target_compile_definitions(test_kste_kv_cuda PRIVATE
        SP_QWEN3_GGUF="${SP_QWEN3_GGUF}" SP_QWEN3_REF="${SP_QWEN3_REF}")
    if(MSVC)
        target_compile_options(test_kste_kv_cuda PRIVATE /W4)
    else()
        target_compile_options(test_kste_kv_cuda PRIVATE -Wall -Wextra)
    endif()
    add_test(NAME E_CU_6 COMMAND test_kste_kv_cuda)
    set_tests_properties(E_CU_6 PROPERTIES TIMEOUT 600)

    # T_FRO_4 on CUDA: same test_ppl gate, routed through the CUDA forward via
    # SP_BACKEND=cuda. (a) cuda-f32 PPL vs the f16 oracle <=0.05%; (b) per-row Q8
    # arena drift vs cuda-f32 <=2%. SLOW / phase-close. Run inside the env-cuda
    # shell so the CUDA runtime DLLs are on PATH.
    add_test(NAME T_FRO_4_CU COMMAND test_ppl)
    set_tests_properties(T_FRO_4_CU PROPERTIES LABELS "SLOW" TIMEOUT 3600
        ENVIRONMENT "SP_BACKEND=cuda")
endif()

# ── Vulkan backend (Phase 2-VK). Built only when SP_ENGINE_WITH_VULKAN; the test
# pulls the Vulkan lib transitively through sp_engine's PUBLIC link. Mirrors the
# CUDA test set. ──
if(SP_ENGINE_WITH_VULKAN)
    add_executable(test_vulkan_smoke test_vulkan_smoke.c)
    target_link_libraries(test_vulkan_smoke PRIVATE sp_engine)
    if(MSVC)
        target_compile_options(test_vulkan_smoke PRIVATE /W4)
    else()
        target_compile_options(test_vulkan_smoke PRIVATE -Wall -Wextra)
    endif()
    add_test(NAME VULKAN_SMOKE COMMAND test_vulkan_smoke)

    add_executable(test_gemma3_vulkan test_gemma3_vulkan.c)
    target_link_libraries(test_gemma3_vulkan PRIVATE sp_engine)
    target_compile_definitions(test_gemma3_vulkan PRIVATE SP_GEMMA3_GGUF="${SP_GEMMA3_GGUF}")
    if(MSVC)
        target_compile_options(test_gemma3_vulkan PRIVATE /W4)
    else()
        target_compile_options(test_gemma3_vulkan PRIVATE -Wall -Wextra)
    endif()
    add_test(NAME M_GEMMA3_VULKAN COMMAND test_gemma3_vulkan)
    set_tests_properties(M_GEMMA3_VULKAN PROPERTIES TIMEOUT 1200)

    add_executable(test_qwen3_vulkan test_qwen3_vulkan.c)
    target_link_libraries(test_qwen3_vulkan PRIVATE sp_engine)
    target_compile_definitions(test_qwen3_vulkan PRIVATE
        SP_QWEN3_GGUF="${SP_QWEN3_GGUF}" SP_QWEN3_REF="${SP_QWEN3_REF}")
    if(MSVC)
        target_compile_options(test_qwen3_vulkan PRIVATE /W4)
    else()
        target_compile_options(test_qwen3_vulkan PRIVATE -Wall -Wextra)
    endif()
    add_test(NAME M_QWEN3_VULKAN COMMAND test_qwen3_vulkan)
    set_tests_properties(M_QWEN3_VULKAN PROPERTIES TIMEOUT 1200)

    add_executable(test_ntt_attn_vulkan test_ntt_attn_vulkan.c)
    target_link_libraries(test_ntt_attn_vulkan PRIVATE sp_engine)
    target_compile_definitions(test_ntt_attn_vulkan PRIVATE
        SP_QWEN3_GGUF="${SP_QWEN3_GGUF}" SP_QWEN3_REF="${SP_QWEN3_REF}")
    if(MSVC)
        target_compile_options(test_ntt_attn_vulkan PRIVATE /W4)
    else()
        target_compile_options(test_ntt_attn_vulkan PRIVATE -Wall -Wextra)
    endif()
    add_test(NAME E_VK_5 COMMAND test_ntt_attn_vulkan)
    set_tests_properties(E_VK_5 PROPERTIES TIMEOUT 1200)

    add_executable(test_kste_kv_vulkan test_kste_kv_vulkan.c)
    target_link_libraries(test_kste_kv_vulkan PRIVATE sp_engine)
    target_compile_definitions(test_kste_kv_vulkan PRIVATE
        SP_QWEN3_GGUF="${SP_QWEN3_GGUF}" SP_QWEN3_REF="${SP_QWEN3_REF}")
    if(MSVC)
        target_compile_options(test_kste_kv_vulkan PRIVATE /W4)
    else()
        target_compile_options(test_kste_kv_vulkan PRIVATE -Wall -Wextra)
    endif()
    add_test(NAME E_VK_6 COMMAND test_kste_kv_vulkan)
    set_tests_properties(E_VK_6 PROPERTIES TIMEOUT 1200)

    # T_FRO_4 on Vulkan: same test_ppl gate, routed through the Vulkan forward via
    # SP_BACKEND=vulkan. (a) vulkan-f32 PPL vs the f16 oracle <=0.05%; (b) per-row
    # Q8 arena drift vs vulkan-f32 <=2%. SLOW / phase-close. Run inside the
    # env-vulkan shell so the Vulkan loader is on PATH.
    add_test(NAME T_FRO_4_VK COMMAND test_ppl)
    set_tests_properties(T_FRO_4_VK PROPERTIES LABELS "SLOW" TIMEOUT 3600
        ENVIRONMENT "SP_BACKEND=vulkan")
endif()
