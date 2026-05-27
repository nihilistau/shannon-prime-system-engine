cmake_minimum_required(VERSION 3.20)
project(shannon_prime_system_engine
    VERSION 0.0.1
    DESCRIPTION "Inference engine for shannon-prime-lattice"
    LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ── Options (the per-backend build scripts under scripts/build/ pass these) ──
set(SP_ENGINE_BACKEND "cpu" CACHE STRING "Active backend: cpu|cuda|vulkan|hexagon")
option(SP_ENGINE_WITH_AVX2   "CPU backend: emit AVX2 path"   ON)
option(SP_ENGINE_WITH_AVX512 "CPU backend: emit AVX512 path" OFF)
option(SP_ENGINE_WITH_CUDA   "Build with CUDA acceleration"  OFF)
option(SP_ENGINE_WITH_VULKAN "Build with Vulkan compute acceleration" OFF)
option(SP_ENGINE_WITH_HEXAGON "Build with Hexagon V69 HTP acceleration" OFF)
option(SP_ENGINE_BUILD_TESTS "Build engine tests"            ON)
# Phase 2-HX: cross-compiling to aarch64-android for the on-phone PPL runner.
# When ON, the test tree builds only test_ppl (the T_FRO_4 gate) — the rest of
# the suite needs the Qwen3 model and is run on the desktop CPU/CUDA builds.
option(SP_ENGINE_TARGET_ANDROID "Phase 2-HX aarch64-android on-phone build" OFF)

if(SP_ENGINE_WITH_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
endif()

# ── Math core (Phase-1 primitives) via the submodule ───────────────────────
# Engine consumes sp_ok_arith / sp_ntt_crt / sp_poly_ring / sp_vht2 /
# sp_frobenius / sp_kste. Build the submodule's libs but NOT its unit tests.
set(SP_SYSTEM_BUILD_TESTS OFF CACHE BOOL "" FORCE)
if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/lib/shannon-prime-system/CMakeLists.txt)
    message(FATAL_ERROR "math-core submodule missing — run: git submodule update --init")
endif()
add_subdirectory(lib/shannon-prime-system)

# ── Engine library + backend ────────────────────────────────────────────────
add_subdirectory(src)

# ── Phase 2-CU.PTX: standalone validate / bench binaries ─────────────────────
# ptx_validate: M_PTX_1 bit-identity test for each PTX sub-phase.
# ptx_bench:    M_PTX_2 throughput benchmark (Nsight Compute target).
# Both compile against sm_75/80/86/90; on no-GPU hosts they print SKIP.
if(SP_ENGINE_WITH_CUDA)
    find_package(CUDAToolkit REQUIRED)  # idempotent — already found in src/
    foreach(_tgt ptx_validate ptx_bench)
        add_executable(${_tgt} src/backends/cuda/${_tgt}.cu)
        target_link_libraries(${_tgt} PRIVATE CUDA::cudart CUDA::cublas)
        target_include_directories(${_tgt} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src/backends/cuda
            ${CMAKE_CURRENT_SOURCE_DIR}/lib/shannon-prime-system/include
        )
        set_target_properties(${_tgt} PROPERTIES
            CUDA_ARCHITECTURES "75;80;86;90"
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
        )
    endforeach()
endif()

# ── Tools (Phase 2-FMT: the sp_transcode GGUF -> .sp-model CLI) ──────────────
add_subdirectory(tools/sp_transcode)

if(SP_ENGINE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

message(STATUS "shannon-prime-system-engine ${PROJECT_VERSION} configured "
               "(backend=${SP_ENGINE_BACKEND}, AVX2=${SP_ENGINE_WITH_AVX2}, "
               "AVX512=${SP_ENGINE_WITH_AVX512})")
