#!/usr/bin/env bash
# Build the stock-llama.cpp logit oracle (dump_logits) against the pristine
# upstream llama.cpp checkout. Run from anywhere; uses absolute paths.
#
#   bash tools/oracle/build_oracle.sh
#
# Requires the upstream llama.cpp built first (libllama.a + ggml libs):
#   cd D:/F/shannon-prime-repos/shannon-prime-lattice-llama
#   cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF \
#         -DLLAMA_CURL=OFF -DGGML_CUDA=OFF -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
#   cmake --build build --target llama -j
set -e

LL=/d/F/shannon-prime-repos/shannon-prime-lattice-llama
HERE="$(cd "$(dirname "$0")" && pwd)"

g++ -std=c++17 -O2 \
    -I "$LL/include" -I "$LL/ggml/include" \
    "$HERE/dump_logits.cpp" \
    -Wl,--start-group \
        "$LL/build/src/libllama.a" \
        "$LL/build/ggml/src/ggml.a" \
        "$LL/build/ggml/src/ggml-cpu.a" \
        "$LL/build/ggml/src/ggml-base.a" \
    -Wl,--end-group \
    -fopenmp -static -static-libgcc -static-libstdc++ \
    -o "$HERE/dump_logits.exe"

echo "built $HERE/dump_logits.exe"
