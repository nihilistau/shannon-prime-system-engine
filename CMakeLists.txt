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

option(SP_ENGINE_BUILD_TESTS "Build engine tests" ON)
option(SP_ENGINE_WITH_CUDA "Build with CUDA acceleration" OFF)

if(SP_ENGINE_WITH_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
endif()

# ──────────────────────────────────────────────────────────────────────
# Phase 0 stub.  Source layout is:
#   src/          — engine sources (GGUF loader, forward pass)
#   include/sp/   — public headers
#   tests/        — engine tests
#   lib/shannon-prime-system  — submodule pointer to math core (Phase 5+)
#
# Each phase in shannon-prime-lattice/papers/PPT-LAT-Roadmap.md names
# the deliverables; they get added to the build here.  Phase 5 lands
# the engine bootstrap; Phase 6 lands two-node sharded inference.
# ──────────────────────────────────────────────────────────────────────

# Sub-targets enabled per phase:
# add_subdirectory(lib/shannon-prime-system)
# add_subdirectory(src)
# add_subdirectory(include)

if(SP_ENGINE_BUILD_TESTS)
    enable_testing()
    # add_subdirectory(tests)  # enabled in Phase 5
endif()

message(STATUS "shannon-prime-system-engine ${PROJECT_VERSION} configured (Phase 0)")
if(SP_ENGINE_WITH_CUDA)
    message(STATUS "  CUDA acceleration: ENABLED")
endif()
