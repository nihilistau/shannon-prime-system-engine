#!/bin/bash
# Sprint WIRE-VULKAN: launch sp-daemon with the Vulkan compute backend
# wired as the prefill forward dispatcher.
#
# Symmetric to start_wire_hex_daemon.sh (cDSP V69 HVX on android), but for
# the host GPU. Activates via SP_DAEMON_BACKEND=vulkan; assumes the daemon
# was built with --features wire_vulkan_backend.
#
# Host prerequisites:
#   - Vulkan-capable GPU + Vulkan loader (vulkan-1.dll / libvulkan.so)
#   - Vulkan SDK installed (VULKAN_SDK env -> glslc + headers at build time)
#   - Built daemon at target/release/sp-daemon{,.exe}
#   - Gemma3-1B / Qwen3 .sp-model + .sp-tokenizer accessible
#
# Known prior OOM bug: ctest M_GEMMA3_VULKAN + M_QWEN3_VULKAN fail with
# vkAllocateMemory: VkResult -2 on the dev host (RTX 2060, 6 GB VRAM). The
# wiring will register cleanly; the first prefill may hit the same OOM
# (the trampoline counter still increments). See ctest-vulkan-validate.log
# + WIRE-VULKAN-OOM-BUGFIX follow-on.

set -e

MODEL="${SP_MODEL_PATH:-./gemma3-1b.sp-model}"
TOK="${SP_TOKENIZER_PATH:-./gemma3-1b.sp-tokenizer}"
PORT="${SP_HTTP_PORT:-8087}"
DAEMON="${SP_DAEMON_BIN:-./target/release/sp-daemon}"

export SP_DAEMON_BACKEND=vulkan
export SP_ARENA=q8
export RUST_LOG="${RUST_LOG:-sp_daemon=info}"

echo "WIRE-VULKAN: launching sp-daemon"
echo "  binary:  $DAEMON"
echo "  model:   $MODEL"
echo "  tok:     $TOK"
echo "  port:    $PORT"
echo "  backend: $SP_DAEMON_BACKEND"
echo "  arena:   $SP_ARENA"

exec "$DAEMON" \
    --daemon-inner \
    --model           "$MODEL" \
    --tokenizer       "$TOK" \
    --draft-model     "" \
    --draft-tokenizer "" \
    --memo-model      "" \
    --memo-tokenizer  "" \
    --pouw-ledger-path "" \
    --quic-port       0 \
    --port            "$PORT" \
    --peer            "" \
    --peers           ""
