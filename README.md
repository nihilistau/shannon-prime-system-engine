# shannon-prime-system-engine

The **engine + daemon** layer of the [shannon-prime-lattice](https://github.com/nihilistau/shannon-prime-lattice)
project: four backend implementations (CPU AVX-512, CUDA PTX, Vulkan,
Hexagon HVX) of the [shannon-prime-system](https://github.com/nihilistau/shannon-prime-system)
math-core forward path, plus `sp_daemon` — a long-lived Rust HTTP/SSE
server that wraps the frozen L1 C ABI in a chat + dialogue +
PoUW-ledger + QUIC-mesh surface that every UI binds to.

The math-core lives at `lib/shannon-prime-system/` as a Git submodule.
That submodule pin is what every engine build links against.

License: AGPL-3.0-or-later. See `LICENSE`. Commercial licensing available.

---

## Contents

1. [What this repo provides](#1-what-this-repo-provides)
2. [Current status — honest table](#2-current-status--honest-table)
3. [Quick start](#3-quick-start)
4. [Architecture](#4-architecture)
5. [The backends](#5-the-backends)
6. [`sp_daemon` Rust crates](#6-sp_daemon-rust-crates)
7. [HTTP / SSE / WebSocket API](#7-http--sse--websocket-api)
8. [Hexagon skel IDL reference](#8-hexagon-skel-idl-reference)
9. [CLI flags + environment variables](#9-cli-flags--environment-variables)
10. [Model conversion (`sp_transcode`)](#10-model-conversion-sp_transcode)
11. [Peering / QUIC mesh](#11-peering--quic-mesh)
12. [Development workflow](#12-development-workflow)
13. [Known issues / pending](#13-known-issues--pending)

---

## 1. What this repo provides

| Slot | Path | Status |
|------|------|--------|
| **Math-core submodule** | `lib/shannon-prime-system/` | linked into every backend; frozen L1 ABI |
| **CPU backend** | `src/backends/cpu/` (`cpu_forward.c`, `cpu_overlay.c`, `cpu_gemma3.c`, `avx512/`) | built |
| **CUDA backend** | `src/backends/cuda/` (`cuda_forward.cu`, `ptx_mma*.cuh`, `ptx_ntt.cuh`, `ptx_spinor.cuh`, `ptx_hash.cuh`) | built |
| **Vulkan backend** | `src/backends/vulkan/` (`vulkan_forward.cpp`, `shaders/`) | built |
| **Hexagon HVX backend (host)** | `src/backends/hexagon/sp_hex_host.c` + `sp_hex_rt.c` + `inc/` | built |
| **Hexagon cDSP skel** | `tools/sp_compute_skel/src_dsp/` (Halide-AOT FFN + HVX NTT + VTCM staging) | built |
| **`sp_daemon` HTTP/SSE server** | `tools/sp_daemon/src/{main.rs, server.rs, routes.rs, daemon.rs}` | built |
| **`sp_transcode` GGUF → `.sp-model`** | `tools/sp_transcode/sp_transcode.c` | built |
| **`sp_dsp_smoke` standalone bridge** | `tools/sp_dsp_smoke/` | built |
| **`sp_npu_spike` Snapdragon NPU spike** | `tools/sp_npu_spike/` | built (K.2-spike) |
| **`sp_halide_gen` Halide AOT compiler** | `tools/sp_halide_gen/` | built |
| **`oracle` cross-backend bit-identity oracle** | `tools/oracle/` | built |
| **PPL harness** | `src/forward/ppl.c` | built |

---

## 2. Current status — honest table

Snapshot 2026-05-31. The user has been frustrated by hours of work
that turned out to bypass the production critical path; this table
exists to prevent that. **Built** means the artefact compiles and
passes its own gates. **Wired** means the daemon routes inference
through it at runtime.

| Component | Built | Wired in `sp_daemon` | Notes |
|-----------|:-----:|:--------------------:|-------|
| Math-core reference forward | yes | **yes (default)** | byte-exact on host + aarch64-android; baseline tok/s in §4 below |
| CPU backend (AVX-512 + cpu_overlay) | yes | **yes (`SP_DAEMON_BACKEND=cpu`)** | sprint WIRE-CPU (2026-06-02): daemon registers `qwen3_forward_cpu` / `gemma3_forward_cpu` via L1 ABI §6 hook; `cpu_forward_count` increments per prefill; bit-exact vs reference (same byte stream); on i9-11900KB host wall-clock matches reference within ±1% (AVX-512 primitives present in lib, hot-path wiring is WIRE-CPU-V2 follow-on) |
| CUDA backend (PTX MMA + NTT) | yes | no | desktop target; symmetric WIRE-HEX sprint |
| Vulkan backend | yes | no | desktop target; symmetric WIRE-HEX sprint |
| Hexagon HVX backend | yes | **partially** | sprint WIRE-HEX: daemon registers `gemma3_forward_hexagon` via L1 ABI §6 hook; `hex_forward_count` increments at first prefill; **`sp_hex_forward` returns non-zero on cDSP after weight upload — cDSP skel on device needs rebuild against current IDL** |
| Polynomial-ring NTT attention (host) | yes | yes (`SP_ENGINE_NTT_ATTN=1`) | byte-exact vs scalar |
| Polynomial-ring NTT attention (Hexagon) | yes | yes (`SP_ENGINE_NTT_ATTN_HEX=1` + Memory model) | sprint NTT.5b/5c; dispatch counter `ntt_hex_forward_count` |
| Spinor-block KV cache | yes | yes (`SP_KV_SPINOR=1`) | persistent compressed KV |
| `/v1/chat` SSE endpoint | yes | yes | greedy argmax decode; stop-string matching |
| `/v1/dialogue` (M.2 MeMo: Grounding → Entity ID → Synthesis) | yes | yes | requires `--memo-model`; returns 3 base64 SpinorReceipts |
| PoUW ledger autowire | yes | yes | `--pouw-ledger-path` enables auto-append of receipts |
| QUIC mesh (`/v1/mesh/peers`, `run_garner_loop`) | yes | yes (host) | android serves empty peer_map |
| FastRPC cDSP echo (`/v1/dsp/echo`) | yes | yes (android) | 8 MB max payload |
| FastRPC cDSP model info (`/v1/dsp/model_info`) | yes | yes (android) | persistent DSP-resident model |

** tok/s baseline (S22U, math-core reference forward, ctx = 16 prefill + 32 decode):**
**These are numbers run during testing of an individual piece of the system on the phone

| Model | Wall (s) | Tokens | tok/s |
|-------|---------:|-------:|------:|
| Gemma3-1B | 18.06 | 16 | 0.89 | 
| Qwen3-0.6B | 11.21 | 16 | 1.43 | 

The HVX backend wiring is in place daemon-side (LLVM-nm on the
android binary shows `gemma3_forward_hexagon` + `sp_hex_forward` +
`sp_wire_hex_forward_dispatch` + `sp_session_register_forward_backend`
at concrete addresses; `/v1/debug/backend_counts` `wire_hex_active = true`
after startup and `hex_forward_count` increments on first prefill).
The remaining work to flip the headline tok/s number is **out of scope
for the wiring sprint** — the on-device `libsp_hex_skel.so` needs to be
rebuilt with `tools/sp_compute_skel/inc/sp_hex.idl` (a different IDL
from `sp_compute.idl`) and pushed to `/data/local/tmp/sp22u/`. Full
detail: `tools/sp_compute_skel/docs/CLOSURE-WIRE-HEX.md`.

---

## 3. Quick start

### 3.1 Build the daemon (host, Windows)

```cmd
:: One-time: set up VS 2019 BT + CUDA env (if you want CUDA backend)
call scripts\env\env-cuda.bat

:: Build math-core + engine libs + sp_daemon (CPU + Vulkan default)
scripts\build\build-cpu.bat
scripts\build\build-vulkan.bat

:: Build sp_daemon with Cargo
cd tools\sp_daemon
cargo build --release --bin sp-daemon
```

Linux equivalents are documented in `docs/BUILD-ENV.md`.

### 3.2 Transcode a GGUF model to `.sp-model`

```cmd
build-cpu\Release\sp_transcode.exe ^
    path\to\model.gguf ^
    out\model.sp-model ^
    out\model.sp-tokenizer ^
    --verify
```

`--verify` runs a round-trip dequant check (rms / max error) and rejects
the output if a Q8 row's relative error exceeds the threshold.

### 3.3 Start the daemon

```cmd
target\release\sp-daemon.exe start ^
    --model out\model.sp-model ^
    --tokenizer out\model.sp-tokenizer ^
    --port 8080
```

Daemon detaches and writes the PID to `%TEMP%\sp-daemon.pid`. Stop via
`sp-daemon stop`.

### 3.4 First chat request

```bash
curl -s -X POST http://127.0.0.1:8080/v1/chat \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Hello, what is 2+2?", "max_tokens": 32}'
```

The response is a `text/event-stream` (Server-Sent Events). Each SSE
event carries a JSON `{"delta": "...token text...", "chat_id": <u64>}`.
Stream ends with `data: [DONE]`.

### 3.5 First dialogue request (dual-model MeMo path)

Requires the daemon to be started with both a target (Executive) model
and a Memory model:

```cmd
sp-daemon start ^
    --model out\executive.sp-model ^
    --tokenizer out\executive.sp-tokenizer ^
    --memo-model out\memory.sp-model ^
    --memo-tokenizer out\memory.sp-tokenizer ^
    --pouw-ledger-path C:\sp\ledger.bin ^
    --port 8080
```

```bash
curl -s -X POST http://127.0.0.1:8080/v1/dialogue \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Who painted the Sistine Chapel?"}' | jq
```

Response shape:

```json
{
  "response": "Michelangelo painted the Sistine Chapel ceiling.",
  "receipts": [
    "<base64 of 64-byte SpinorReceipt, turn 1 Grounding>",
    "<base64 of 64-byte SpinorReceipt, turn 2 Entity ID>",
    "<base64 of 64-byte SpinorReceipt, turn 3 Synthesis>"
  ],
  "wall_ms": 412,
  "turn_us": [128000, 184000, 100000]
}
```

The three SpinorReceipts are appended to the PoUW ledger if
`--pouw-ledger-path` was set.

---

## 4. Architecture

```
                                          ┌─────────────┐
                                          │  Browser /  │
                                          │  TUI / CLI  │
                                          │  curl, etc. │
                                          └──────┬──────┘
                                                 │ HTTP/JSON
                                                 │ SSE + WebSocket
                                                 ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ sp_daemon  (axum + tokio, Rust)                                           │
│                                                                           │
│  routes.rs ─┬─ /v1/chat (SSE)         ─┬─ session::SpSession (Mutex)     │
│              ├─ /v1/dialogue (JSON)    ─┤                                 │
│              ├─ /v1/events (SSE)       ─┤  ┌─────────────────────────┐   │
│              ├─ /v1/metrics (JSON)     ─┘  │ dialogue_runner.rs      │   │
│              ├─ /v1/mesh/peers (JSON)      │ Grounding→EntityID→Synth│   │
│              ├─ /v1/receipts (JSON)        └─────────────────────────┘   │
│              ├─ /v1/pouw/ledger (SSE)                                    │
│              ├─ /v1/node/telemetry (WS)                                  │
│              ├─ /v1/abort/:id (POST)                                     │
│              ├─ /v1/dsp/echo (POST, android)                             │
│              ├─ /v1/dsp/model_info (GET, android)                        │
│              └─ /v1/debug/backend_counts (JSON)                          │
│                                                                           │
│  pouw_ledger.rs  ── Append-only SpinorReceipt ledger; canonical replay   │
│  memo_routing.rs ── KSTE-routed sparse Memory activation (M.5)           │
│  network/quic_shard.rs ── QUIC coordinator + worker + Garner CRT loop    │
│  dsp_rpc.rs (android) ── libcdsprpc.so FastRPC bridge                    │
│  ntt_hex_dispatch.rs (android) ── NTT.5b backend trampoline              │
│  hex_forward_dispatch.rs (android, feature=wire_hex_backend)              │
│        ── Sprint WIRE-HEX full-forward backend dispatcher                │
└─────────────────────────────────┬─────────────────────────────────────────┘
                                  │
                                  │ frozen L1 C ABI  (sp_l1.h)
                                  │ sp_session_create / sp_prefill_chunk /
                                  │ sp_decode_step / sp_session_clone /
                                  │ sp_session_register_forward_backend
                                  ▼
┌───────────────────────────────────────────────────────────────────────────┐
│  libshannonprime  (C — lib/shannon-prime-system/)                         │
│                                                                           │
│  core/session ── sp_session lifecycle + KV cache + clone/rewind          │
│  core/forward ── reference forward (matmul→RMSNorm→RoPE→attn→FFN)        │
│  core/ntt_crt ── dual-prime NTT-CRT primitive (Barrett)                  │
│  core/poly_ring ── R_q polynomial-ring attention                         │
│  core/poly_ring_bluestein ── arbitrary power-of-2 N via chirp-z          │
│  core/frobenius ── Q8/Q4 per-row codec                                   │
│  core/arena ── packed-weight arena                                       │
│  core/vht2 ── Spinor 63-byte block + Möbius reorder + CRC-8              │
│  core/kste ── encoder + Tier-0/Tier-1 dominance                          │
│  core/io_format ── .sp-model mmap loader                                 │
└──────────────────┬────────────────────────────────┬────────────────────┬─┘
                   │  §6 forward-backend hook        │ NTT dispatch      │
                   ▼                                  ▼ hook              │
   ┌─────────────────────────────────┐    ┌──────────────────────────┐   │
   │  Engine backends (libsp_engine) │    │  Hexagon cDSP skel       │   │
   │  src/backends/                  │    │  tools/sp_compute_skel/  │   │
   │   ├ cpu/ (AVX-512 + overlay)    │    │   ├ src_dsp/ (HVX NTT,   │   │
   │   ├ cuda/ (PTX MMA + NTT)       │    │   │   Halide FFN,        │   │
   │   ├ vulkan/ (compute shaders)   │    │   │   VTCM staging)      │   │
   │   └ hexagon/ (sp_hex_host.c)    │───▶│   └ inc/sp_compute.idl   │   │
   │       FastRPC client side        │     (FastRPC server side)        │
   └─────────────────────────────────┘    └──────────────────────────┘   │
                                                                          │
   The 4 backends + the cDSP skel all gate their output against the      │
   math-core scalar reference via T_*_BIT_EXACT tests. Math is in Z_q;   │
   floating point is plumbing.                                           ▼
```

---

## 5. The backends

Each backend lives under `src/backends/<name>/`; build flags select
which get linked. All four override slices of the math-core reference
forward and gate to it for bit-exactness.

### 5.1 CPU backend — `src/backends/cpu/`

| File | Role |
|------|------|
| `cpu_forward.c` | Whole-forward entry point (`gemma3_forward_cpu`, `qwen3_forward_cpu`) |
| `cpu_overlay.c` | Per-matmul + per-row dequant kernels honouring `SP_ENGINE_FROB`, `SP_CPU_SCALAR`, `SP_ENGINE_F16_ACT`, `SP_Q4_PROMOTE` |
| `cpu_gemma3.c` | Gemma3 arch path (sandwich pre/post norm + GeGLU FFN) |
| `cpu_generate.c` | Standalone CPU generation harness |
| `avx512/` | AVX-512 matmul + dot kernels; sub-phase Phase 2-CPU-AVX |

Build flags:

```bash
cmake -B build-cpu -G Ninja \
      -DSP_ENGINE_BACKEND=cpu \
      -DSP_ENGINE_WITH_AVX2=ON \
      -DSP_ENGINE_WITH_AVX512=ON \
      -DSP_ENGINE_BUILD_TESTS=ON
```

Or use `scripts\build\build-cpu.bat`. Status: **built**, NOT wired into
`sp_daemon` (the daemon uses the math-core reference forward by default).
Wiring is a symmetric WIRE-HEX-style sprint: extend
`sp_session_register_forward_backend` registration to call
`gemma3_forward_cpu` instead.

### 5.2 CUDA backend — `src/backends/cuda/`

| File | Role |
|------|------|
| `cuda_forward.cu` | Whole-forward entry point + cudaStream lifecycle |
| `cuda_backend.cu` | Backend init, device selection, error mapping (CUDA → `SP_ECUDA`) |
| `ptx_mma.cuh` | Inline PTX `mma.sync` for the matmul tile (Turing sm_75 + Ampere sm_80 + Hopper sm_90) |
| `ptx_mma_tile_{int4,int8}.cuh` + `ptx_mma_tile_common.cuh` | Tiled INT4 / INT8 matmul tiles |
| `ptx_mma_tile_validate.cu`, `ptx_mma_tile_bench.cu` | Bit-exact validators + perf benches |
| `ptx_ntt.cuh` | PTX NTT butterfly. **Forbidden idiom** (memory entry `reference-nvcc-paired-register-bug`): never use `mul.wide.u32` / `mad.wide.u32` — nvcc miscompiles. Decompose to separate `mul.lo.u32` + `mul.hi.u32` + `shf.r/add.cc`. |
| `ptx_spinor.cuh` | PTX Spinor-block decode kernel |
| `ptx_hash.cuh` | PTX hash chain (sm_75 has only one 32-bit ALU dispatch port shared between `lop3.b32` and `xor.b32` — silicon-capped ~1.1× on Turing; 3× reachable on Ampere; memory `reference-turing-alu-scheduler-ceiling`) |
| `ptx_validate.cu` | Cross-backend bit-identity oracle |

Build flags:

```bash
cmake -B build-cuda -G Ninja \
      -DSP_ENGINE_BACKEND=cuda \
      -DSP_ENGINE_WITH_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES="75;80;90" \
      -DCMAKE_CUDA_FLAGS="--use-local-env" \
      -DSP_ENGINE_BUILD_TESTS=ON
```

The `--use-local-env` flag is mandatory on VS2019 BuildTools (without a
full VS install nvcc's internal vcvars detection fails). See
`scripts/build/build-cuda.bat`.

Status: **built** + bit-exact-validated. **NOT wired into `sp_daemon`** —
symmetric WIRE-HEX-style sprint pending.

### 5.3 Vulkan backend — `src/backends/vulkan/`

| File | Role |
|------|------|
| `vulkan_forward.cpp` | Whole-forward entry point; command buffer + pipeline lifecycle |
| `vulkan_backend.cpp` | Backend init (instance, device, compute queue); error mapping to `SP_EVULKAN` |
| `vk_common.h` | Shared validator + scratch helpers |
| `shaders/` | GLSL → SPV compute shaders for matmul, RMSNorm, RoPE, attention, NTT |

Build flags:

```bash
cmake -B build-vulkan -G Ninja \
      -DSP_ENGINE_BACKEND=vulkan \
      -DSP_ENGINE_WITH_VULKAN=ON \
      -DSP_ENGINE_BUILD_TESTS=ON
```

Status: **built** + bit-exact-validated (closure `SESSION-CLOSED-lat-2-L1-PARITY.md`).
**NOT wired into `sp_daemon`** — symmetric WIRE-HEX-style sprint pending.

### 5.4 Hexagon HVX backend — `src/backends/hexagon/` + `tools/sp_compute_skel/`

Two pieces: **host side** runs on aarch64-android in the daemon process;
**device side** is the cDSP skel running on Hexagon V69 inside the
Qualcomm cDSP. They talk over FastRPC.

**Host side** (`src/backends/hexagon/`):

| File | Role |
|------|------|
| `sp_hex_host.c` | `gemma3_forward_hexagon` entry; weight upload; FastRPC round-trip |
| `sp_hex_rt.c` | Runtime helpers — DmaBuffer management, IDL marshalling |
| `sp_hex_layout.h` | On-device weight layout (mirrors Q8 arena) |
| `inc/sp_hex.idl` | Forward-pass IDL (separate from `sp_compute.idl` — that's the compute-primitive IDL) |
| `dsp/` | Local copy of skel sources for build cross-check |
| `echo_skel/` | Echo skel for sprint C (FastRPC smoke) |

**Device side** (`tools/sp_compute_skel/`):

| Path | Role |
|------|------|
| `inc/sp_compute.idl` | Compute-primitive IDL (axpby, scale_i16, vtcm_probe, ffn_2stage_halide, barrett_oracle, matmul_q, ntt_*) — full reference in §8 |
| `src_dsp/` | cDSP-side implementations using HVX intrinsics, Halide AOT FFN, VTCM staging |
| `halide_gen/` | Halide schedule generators |
| `hexagon_Release_toolv87_v69/` | Build artefacts |

Build (Windows host + Hexagon SDK 5.5.6.0):

```cmd
set HEXAGON_SDK_ROOT=C:\Qualcomm\Hexagon_SDK\5.5.6.0
scripts\build\build-hexagon.bat
```

For the daemon-linkable host-side static lib (`libsp_hex_daemon_backend.a`):

```cmd
tools\sp_daemon\build-android-hex-backend.bat
```

Then cross-compile sp-daemon with the WIRE-HEX feature:

```cmd
set LIBCLANG_PATH=C:\Program Files\LLVM\bin
cd tools\sp_daemon
cargo build --target aarch64-linux-android --release ^
            --features wire_hex_backend --bin sp-daemon
```

See `start_wire_hex_daemon.sh` for the on-device launcher (env vars +
adb-push sequence).

Status: **built end-to-end**. NTT primitives (forward, twiddle VTCM
staging, dual-prime CRT dispatch, INTT + Garner) are byte-exact vs
math-core. Forward-pass wiring is shipped daemon-side; the device-side
`libsp_hex_skel.so` rebuild against the current `inc/sp_hex.idl` is the
pending step (out of scope for sprint WIRE-HEX, owned by a future
HX-SKEL-REBUILD sprint with the SDK build_cmake chain).

---

## 6. `sp_daemon` Rust crates

`sp_daemon` is one Cargo package with one library + 11 binaries.

### 6.1 Library (`tools/sp_daemon/src/lib.rs`)

| Module | Purpose |
|--------|---------|
| `dialogue` | M.2 zero-copy dialogue primitives: `SpinorReceipt` (64-byte audit envelope), `DialoguePool` (pre-allocated buffers), `MODEL_ID_EXECUTIVE`/`MEMORY` constants |
| `pouw_ledger` | M.4 append-only PoUW ledger: `Ledger::open`, `append`, `read_all`, `canonical_sort`, `replay_canonical_into`; cross-device byte-identity gates |
| `memo_routing` | M.5 KSTE-routed sparse Memory activation: builds a `RoutingMask` from a K-vector via the math-core KSTE encoder |
| `network::quic_shard` | Phase 6-NET QUIC transport: `SpQuicCoordinator`, `SpQuicWorker`, `ShardBlockHeader` (64-byte wire format), `run_garner_loop` (dual-prime CRT recombine on the driver) |
| `ntt_ffi` | Bindgen output for math-core NTT-CRT primitives |
| `dsp_rpc` (android) | Dynamic libcdsprpc.so loader: `FastRpcSession`, `DmaBuffer`, `RemoteArg`, `make_scalars` — single-thread invoke per session |
| `ntt_hex_dispatch` (android) | NTT.5b backend trampoline: routes `sp_compute_ntt_dispatch_fn` calls to `ntt_hvx_vtcm_oracle` (method 17) and `intt_hvx_oracle` (method 18) over FastRPC |
| `hex_forward_dispatch` (android, feature=`wire_hex_backend`) | WIRE-HEX full-forward backend dispatcher: implements `sp_forward_dispatch_fn` for `gemma3_forward_hexagon`; bumps `dispatch_count` atomic; survives session clone |
| `ffi_l1` (android) | Bindgen output for the L1 C ABI (`sp_session_*`, `sp_prefill_chunk`, `sp_decode_step`, `sp_session_register_forward_backend`) |

### 6.2 Binaries (`tools/sp_daemon/src/bin/` and `src/main.rs`)

| Binary | Source | Purpose |
|--------|--------|---------|
| `sp-daemon` | `src/main.rs` | Main long-lived HTTP/SSE server |
| `sp-console` | `bin/sp_console.rs` | Interactive REPL against a running daemon |
| `spec_validate` | `bin/spec_validate.rs` | Phase 4-SPEC speculative-decoding validator |
| `probe` | `bin/probe.rs` | Health probe — pings `/v1/metrics` and prints output |
| `sp_memo_m1_smoke` | `bin/sp_memo_m1_smoke.rs` | M.1 dual-model load budget audit + concurrent invoke smoke (android-only) |
| `sp_memo_m2_dialogue_smoke` | `bin/sp_memo_m2_dialogue_smoke.rs` | M.2 zero-copy dialogue smoke + Spinor receipt envelope (android-only) |
| `sp_memo_m4_ledger_smoke` | `bin/sp_memo_m4_ledger_smoke.rs` | M.4 PoUW ledger smoke (append + read + replay-determinism) |
| `sp_memo_m4_canonical_replay_smoke` | `bin/sp_memo_m4_canonical_replay_smoke.rs` | mesh-canonical-order + cross-device byte-identity gates |
| `sp_memo_m5_routing_smoke` | `bin/sp_memo_m5_routing_smoke.rs` | M.5 KSTE-routed sparse activation smoke |
| `sp_chat_dialogue_smoke` | `bin/sp_chat_dialogue_smoke.rs` | `/v1/dialogue` end-to-end HTTP smoke |
| `sp_chat_ledger_autowire_smoke` | `bin/sp_chat_ledger_autowire_smoke.rs` | `/v1/dialogue` + PoUW ledger autowire smoke |
| `sp_ntt_5c_forward_smoke` | `bin/sp_ntt_5c_forward_smoke.rs` | NTT.5c forward-activation smoke (android-only) |
| `sp_ntt_bench_toks` | `bin/sp_ntt_bench_toks.rs` | NTT-bench tok/s per-cell harness (2 models × 3 configs × 3 reps) |

Most smoke binaries print `"android-only"` and exit on host builds —
their FFI surface only links against the libs that cross-compile to
aarch64-android.

### 6.3 Key Rust types

**`AppState`** (`src/state.rs`) — the axum `State<Arc<AppState>>` payload.
Holds the L1 session(s), tokenizer(s), event broadcaster, receipt store,
peer map, optional Memory model, optional ledger, optional cDSP bridge.
Drop order preserves model lifetime past sessions.

**`SpSession`** (`src/session.rs`) — Rust `Send`-not-`Sync` wrapper
around the opaque C `sp_session *`. Owns the cancel flag. Methods:
`prefill_chunk(&mut self, tokens, logits)`, `decode_step(&mut self, token, logits)`,
`clone_session(cancel_flag)`, `rewind(n)`, `position()`.

**`SpinorReceipt`** (`src/dialogue.rs`) — exact 64-byte
`#[repr(C, packed)]` struct: `turn_index u8 | model_id u8 | _pad [u8;2] |
wall_us u32 | input_hash [u8;24] | output_hash [u8;24] | n_input_tokens u32 |
n_output_tokens u8 | _reserved [u8;2] | sentinel 0xA5`. Compile-time
`size_of` assertion catches accidental padding changes.

**`DialoguePool`** (`src/dialogue.rs`) — pre-allocated buffers for the
3-turn dialogue loop. `.clear() + .push()` only inside the hot loop;
no allocator activity per turn (per `reference-zero-copy-invariant`).

**`Ledger`** (`src/pouw_ledger.rs`) — append-only file of 64-byte
receipts. Methods: `open`, `append`, `read_all`, `len`, `canonical_sort`
(stable sort on `(turn_index, input_hash[..2])`), `replay_canonical_into`
(canonical-order byte-identical replay into a new file — the cross-device
identity gate).

**`ShardBlockHeader`** (`src/network/quic_shard.rs`) — 64-byte
`#[repr(C)]` wire header: `seq_id u64 | token_pos u32 | layer_id u32 |
prime_selector u8 | _pad [u8;47]`. `prime_selector = 0` → `q_1 =
1073738753`, `1` → `q_2 = 1073732609`. `ResidueBlock` carries the
header + N residues.

---

## 7. HTTP / SSE / WebSocket API

All endpoints live under the version prefix `/v1/`. Wired in
`tools/sp_daemon/src/server.rs::build_router`.

### 7.1 Endpoint summary

| Method | Path | Purpose | Streaming |
|--------|------|---------|-----------|
| POST | `/v1/chat` | Single-shot chat → token delta stream | SSE |
| GET  | `/v1/chat/stream` | Legacy SSE stub (returns `{"status":"stub"}`) | JSON |
| POST | `/v1/dialogue` | Dual-model Grounding → Entity ID → Synthesis | JSON |
| POST | `/v1/abort/:id` | Cancel a running chat | 204 / 404 |
| GET  | `/v1/events` | Daemon-wide event stream (chat lifecycle, PoUW mints) | SSE |
| GET  | `/v1/metrics` | Tokens/sec + position + peer count | JSON |
| GET  | `/v1/receipts` | All accumulated PoUW receipts | JSON |
| GET  | `/v1/pouw/ledger` | Live KSTE receipt feed | SSE |
| GET  | `/v1/mesh/peers` | Active QUIC peers + shard assignment | JSON |
| WS   | `/v1/node/telemetry` | 1 Hz node telemetry stream | WebSocket |
| POST | `/v1/dsp/echo` | (android) FastRPC echo through cDSP | Bytes |
| GET  | `/v1/dsp/model_info` | (android) DSP-resident model metadata | JSON |
| GET  | `/v1/debug/backend_counts` | WIRE-HEX + NTT.5b dispatch counters | JSON |

Static file serving: `frontend_mockups/` is mounted as fallback under
the router (everything that doesn't match a route gets file-served).
CORS is permissive by default.

### 7.2 `POST /v1/chat`

**Request body** (one of `prompt`, `messages`, `prompt_tokens` required —
exactly one):

```json
{
  "prompt": "Why is the sky blue?",
  "max_tokens": 256,
  "stop": ["\n\n"]
}
```

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Why is the sky blue?"}
  ],
  "max_tokens": 256
}
```

```json
{
  "prompt_tokens": [2, 1037, 4, 5683],
  "max_tokens": 8,
  "stop": []
}
```

**Response** — `text/event-stream`. Each event:

```
data: {"delta":"The ","chat_id":42}

data: {"delta":"sky ","chat_id":42}

data: {"delta":"appears blue because","chat_id":42}

data: [DONE]
```

On client disconnect or `/v1/abort/:id`, an `event: cancelled` is
emitted instead of `[DONE]`. On error during prefill or decode, a
single `data: {"error":"..."}` event is sent.

**Error responses (4xx, JSON body):**
- `400` `{"error":"one of prompt / messages / prompt_tokens required"}`
- `400` `{"error":"only one of prompt / messages / prompt_tokens may be set"}`
- `400` `{"error":"chat_template_unavailable","arch_id":<id>,"hint":"use prompt or prompt_tokens"}`
- `400` `{"error":"<tokenizer error message>"}`

### 7.3 `POST /v1/dialogue`

Returns 501 if the daemon wasn't started with `--memo-model` /
`--memo-tokenizer`.

**Request body:**

```json
{"prompt": "Who painted the Sistine Chapel?"}
```

**Response (200):**

```json
{
  "response": "Michelangelo painted the Sistine Chapel ceiling.",
  "receipts": [
    "AAEAAAArAAAAVQ...   <88-char base64 of 64-byte SpinorReceipt turn 1>",
    "AAIAAAAvAAAA...     <turn 2>",
    "AAMAAAA0AAAA...     <turn 3>"
  ],
  "wall_ms": 412,
  "turn_us": [128000, 184000, 100000]
}
```

Each receipt decodes to a 64-byte `SpinorReceipt`. Byte 0 = `turn_index`,
byte 1 = `model_id` (`0xE` = Executive, `0x4D` = Memory), bytes 2-3
padding, bytes 4-7 `wall_us` LE, bytes 8-31 SHA-256 truncated input hash,
bytes 32-55 SHA-256 truncated output hash, bytes 56-59 `n_input_tokens`
LE, byte 60 `n_output_tokens`, bytes 61-62 reserved, byte 63 = `0xA5`
sentinel.

**Error responses:**
- `400` `{"error":"prompt required"}`
- `500` `{"error":"exec clone: <detail>"}`
- `500` `{"error":"memo clone: <detail>"}`
- `500` `{"error":"run_dialogue: <detail>"}`
- `501` `{"error":"memo_model_not_loaded","hint":"start sp-daemon with --memo-model ..."}`

If `--pouw-ledger-path` is set, all three receipts are best-effort
appended to the ledger BEFORE the HTTP response is built. A lock or
append failure logs a warning and the response still ships
(`tracing::warn!` rather than 5xx).

### 7.4 `POST /v1/abort/:id`

`:id` = `chat_id` from a `ChatDelta` event. Returns `204 NO_CONTENT`
if the cancel flag was flipped; `404 NOT_FOUND` if no active chat with
that id.

### 7.5 `GET /v1/events`

Long-lived SSE channel for daemon-wide events:

```
event: chat_completed
data: {"chat_id":42,"status":"done"}

event: chat_completed
data: {"chat_id":43,"status":"cancelled"}

event: mint
data: {"receipt_hex":"<304 hex chars = 152 bytes>","sig_hex":"<128 hex chars = 64 bytes>"}
```

Each new `/v1/events` subscriber gets a fresh broadcast subscription;
back-pressure-bounded (64-event channel; slow consumers drop with
`tokio_stream` lagged events).

### 7.6 `GET /v1/metrics`

```json
{
  "tokens_per_sec": 1.43,
  "ram_svm_bytes": 0,
  "peers": 0,
  "phase": "lat-phase-2-l3-tok-closed",
  "session_pos": 22
}
```

`tokens_per_sec` is lifetime tokens decoded / elapsed since daemon start.
`peers` is the count from `peer_map` (always 0 on android). `session_pos`
is the base session's current position (0 except in single-session
debug builds — chat clones the session per request, so the base stays at 0).

### 7.7 `GET /v1/receipts`

```json
{
  "receipts": [
    {"payload_hex": "<304 hex>", "sig_hex": "<128 hex>", "round": 7}
  ],
  "cursor": null
}
```

These are the PoUW receipts minted by the background sieve-mining loop
(`src/mining.rs`). Each is the 152-byte wire-format receipt plus a 64-byte
ed25519 signature over the payload.

### 7.8 `GET /v1/mesh/peers`

```json
{
  "peers": [
    {"node_id": "192.0.2.10:5000", "address": "192.0.2.10:5000",
     "shard_id": "q1", "latency_ms": 45}
  ],
  "active": 1,
  "total": 32
}
```

`shard_id` is `"q1"` if `prime_selector == 0` in the first received
block from that peer, `"q2"` if `1`. Peers in handshake state (no block
yet) show `shard_id == 255` internally (not surfaced).

### 7.9 `GET /v1/node/telemetry` (WebSocket)

Server pushes JSON every 1000 ms:

```json
{
  "node_id": "q3-beast-canyon",
  "cpu_temp_c": 58.5,
  "svm_mem_gb": 2.4,
  "dht_peers_active": 1,
  "dht_peers_total": 32,
  "pouw_frontier": 7
}
```

### 7.10 `GET /v1/pouw/ledger`

SSE feed of KSTE-formatted receipt lines as they're minted:

```
data: [KSTE] Round: 7 | Nonce: 0x4f3a... | Z_q Hash: 0xa1b2...
```

### 7.11 `POST /v1/dsp/echo` (android-only)

Routes a raw `application/octet-stream` body through the V69 cDSP echo
skel. Max body 8 MB. Returns the echoed bytes on success.

```bash
curl -X POST http://127.0.0.1:8080/v1/dsp/echo \
     -H "Content-Type: application/octet-stream" \
     --data-binary @some-blob.bin -o echo-out.bin
```

- `400` empty body
- `413` body > 8 MB
- `500` `dsp_rpc: <error>`
- `501` "cDSP session not admitted" / "v1/dsp/echo requires target_os=android"

### 7.12 `GET /v1/dsp/model_info` (android-only)

```json
{
  "n_layers": 28,
  "hidden_size": 1024,
  "n_heads": 16,
  "n_kv_heads": 8,
  "vocab_size": 151936,
  "total_dma_bytes": 731589632,
  "load_wall_ms": 4127,
  "kv_cache_bytes": 92274688
}
```

Returns 501 with body `"model not loaded"` if the DSP-resident model
load failed.

### 7.13 `GET /v1/debug/backend_counts`

```json
{
  "hex_forward_count": 1,
  "wire_hex_active": true,
  "ntt_hex_forward_count": 0,
  "ntt_hex_inverse_count": 0
}
```

- `hex_forward_count` — `gemma3_forward_hexagon` dispatcher hits since
  process start; > 0 after one prefill when `SP_DAEMON_BACKEND=hex` AND
  feature `wire_hex_backend`. Always 0 on host builds and on android
  without the feature.
- `wire_hex_active` — whether `sp_session_register_forward_backend`
  succeeded at startup. Independent of whether a prefill has run.
- `ntt_hex_forward_count` / `ntt_hex_inverse_count` — Hexagon NTT
  dispatch counters (Bluestein inner kernels via FastRPC methods 17/18).
  Always 0 when `SP_ENGINE_NTT_ATTN_HEX` is unset.

---

## 8. Hexagon skel IDL reference

The IDL at `tools/sp_compute_skel/inc/sp_compute.idl` defines the
FastRPC contract between the daemon-side trampoline and the cDSP V69
HVX skel. Each method returns `long` (0 = success, -1 = constraint
violation, other negative = AEE error code).

| qaic method | Name | Sprint | Purpose |
|:--:|------|:--:|---------|
| 1 | `axpby` | §3-HX D | Fixed-point AXPBY `y[i] = sat_i16((a·x[i] + b) >> q_bits)` — scalar pipelined |
| 2 | `scale_i16` | §3-HX D | HVX-vectorized i16 scale, the canonical HVX proof. Uses `Q6_Vh_vadd_VhRh_sat`. |
| 3 | `axpby_hvx` | §3-HX E F1 | Full axpby via HVX intrinsics — `vmpy` widening → `vadd` word → `vasr` → `vpack_VwVw_sat`. Constraints: `|a_h| ≤ 32767`, `0 ≤ q_bits ≤ 30`. |
| 4 | `scale_i16_batched` | §3-HX E F2 | Amortize FastRPC per-call overhead — batched scale_i16 |
| 5 | `vtcm_probe` | §3-HX F | `HAP_request_VTCM(size, single_page)` litmus — reports admit / deny + low-32 of VTCM addr |
| 6 | `axpby_2d_halide` | §3-HX F | Halide-AOT 2D axpby with VTCM hot-copy. `cols % 64`, `|a[c]| ≤ 32767`, `q_bits ≤ 30` |
| 7 | `ffn_2stage_halide` | §3-HX G | 2-stage matmul FFN via Halide AOT with dual-VTCM staging (external X/W1/W2/Y + internal `hidden`). Constraints: `d_in`/`h_dim`/`d_out` multiples of 128; `batch` ≥ 1 multiple of 4 |
| 8 | `ffn_2stage_diag_halide` | §3-HX H | Same as 7 but also writes post-stage-1 `hidden` to a caller buffer for matmul-1 vs matmul-2 isolation |
| 9-10 | (reserved) | — | reserved slots from earlier sprint reshuffle |
| 11 | `barrett_oracle` | §3-HX K v0.β 2.5 | N test (a, b) u32 pairs through modular multiply mod q_1 or q_2 |
| 12 | `matmul_q` | §3-HX K v0.β 2.5c | HVX `mod_q` matmul `Y[b][i] = (Σ_k X[b][k]·W[k][i]) mod q`. Constraints: `d_out % 32 == 0`, `q_idx ∈ {0,1}` |
| 13 | `ntt_oracle` | §4-NTT NTT.0 | Scalar negacyclic NTT mod q_1 or q_2. `N ∈ {128,256,512}` |
| 14 | `ntt_hvx_oracle` | §4-NTT NTT.1 | HVX-vectorized NTT butterfly. Large stages (`half ≥ 32`) HVX intrinsics; small stages scalar fallback |
| 15 | `ntt_twiddle_init` | §4-NTT NTT.2 | Precompute + pin all 6 (prime, N) twiddle tables in VTCM via `HAP_request_VTCM`. Idempotent. |
| 16 | `ntt_twiddle_status` | §4-NTT NTT.2 | Inspect one (prime, N) entry's VTCM state — base addr, size, per-sub-table offsets |
| 17 | `ntt_twiddle_dump` | §4-NTT NTT.2 | Copy one sub-table from VTCM into a caller buffer for cross-check |
| 18 | `ntt_hvx_vtcm_oracle` | §4-NTT NTT.3 | VTCM-aware HVX forward NTT. Production forward path for dual-prime CRT dispatch. Requires `ntt_twiddle_init` first. |
| 19 | `intt_hvx_oracle` | §4-NTT NTT.4 | HVX-vectorized inverse NTT mod a frozen Proth prime. Consumes `ipsi_pow`, `w_inv`, `w_inv_stages` VTCM tables. |

(Method numbers shifted slightly relative to early sprint specs due to
merge-time renumbering — both NTT.3 and NTT.4 anticipated method 17;
NTT.4 was renumbered to 18 at merge.)

Full per-method input/output buffer shapes + semantics:
`tools/sp_compute_skel/inc/sp_compute.idl`.

The separate forward-pass IDL `src/backends/hexagon/inc/sp_hex.idl`
defines the methods the WIRE-HEX path uses (`forward`, `upload_crc`,
etc.) — its skel binary `libsp_hex_skel.so` is what currently needs
rebuilding for the WIRE-HEX BIT-EXACT gate to flip.

---

## 9. CLI flags + environment variables

### 9.1 `sp-daemon start` CLI

| Flag | Env | Default | Purpose |
|------|-----|---------|---------|
| `--model` | `SP_MODEL_PATH` | (required) | Target/verifier `.sp-model` path |
| `--tokenizer` | `SP_TOKENIZER_PATH` | (required) | Matching `.sp-tokenizer` path |
| `--draft-model` | `SP_DRAFT_MODEL_PATH` | empty | Draft `.sp-model` for Phase 4-SPEC spec-decode |
| `--draft-tokenizer` | `SP_DRAFT_TOKENIZER_PATH` | empty | Draft `.sp-tokenizer` |
| `--memo-model` | `SP_MEMO_MODEL_PATH` | empty | Memory `.sp-model` for `/v1/dialogue`. Endpoint returns 501 if unset. |
| `--memo-tokenizer` | `SP_MEMO_TOKENIZER_PATH` | empty | Memory `.sp-tokenizer` |
| `--pouw-ledger-path` | `SP_POUW_LEDGER_PATH` | empty | Enable PoUW ledger autowire from `/v1/dialogue` |
| `--port` | `SP_HTTP_PORT` | `8080` | TCP port for HTTP API |
| `--quic-port` | `SP_QUIC_PORT` | `0` (disabled) | UDP port for QUIC DHT mesh |
| `--peer` | — | empty | Single QUIC peer address to dial on startup (alias for `--peers`) |
| `--peers` | `SP_PEERS` | empty | Comma-separated list of QUIC peers to dial |

### 9.2 Runtime knob env vars (read by math-core + backends per-forward)

| Env | Values | Effect |
|-----|--------|--------|
| `SP_DAEMON_BACKEND` | `hex` (else unset) | (WIRE-HEX feature build only) register `gemma3_forward_hexagon` against target session at startup |
| `SP_ENGINE_FROB` | `0..4` | Weight path: 0 = pure f32; 1 = Q8 inline; 2 = Q8 dequant; 3 = Q4 inline; 4 = Q4 mixed-precision |
| `SP_Q4_PROMOTE` | float | Q4 rows whose round-trip rel-error exceeds this get promoted to Q8 |
| `SP_ENGINE_F16_ACT` | `0`/`1` | Round matmul activations to F16 (ggml-faithful cross-validation path) |
| `SP_CPU_SCALAR` | `0`/`1` | Force scalar reduction (disable AVX vectorization) |
| `SP_KV_SPINOR` | `0`/`1` | Persistent compressed Spinor-block KV cache |
| `SP_KV_SPINOR_REF` | `0`/`1` | Parity reference: f32 cache + in-place Spinor round-trip |
| `SP_ENGINE_NTT_ATTN` | `0`/`1` | Enable polynomial-ring NTT attention overlay (prefill only) |
| `SP_ENGINE_NTT_ATTN_HEX` | `0`/`1` | (android) Route inner NTT calls through FastRPC methods 17/18 |
| `SP_ARENA` | `q8` / `q4` | Build the packed-weight arena at load (Q8 or Q4 mixed-precision) |
| `SP_ARENA_RELEASE` | `0`/`1` | Release the GGUF mapping after arena pack (~50% RAM cut) |
| `SP_ARENA_EMBED` | `0`/`1` | Include the token embedding in the arena pack |
| `ADSP_LIBRARY_PATH` | path | (android) Where FastRPC looks for `libsp_compute_skel.so` and other skels |
| `RUST_LOG` | string | `tracing-subscriber` filter (e.g. `sp_daemon=debug,axum=info`) |

Default arena precision is **8** (Q8); set `SP_ARENA=q4` for Q4 mixed.
Defaults across `SP_ENGINE_*` are 0 (off) — the unfeatured baseline is
bit-identical to a plain inference path (per
`shannon-prime-lattice/papers/PPT-LAT-Systems.md` binding rule).

---

## 10. Model conversion (`sp_transcode`)

The `sp_transcode` tool converts a GGUF model to a `.sp-model` +
`.sp-tokenizer` pair that the daemon can `mmap`-load.

### 10.1 Usage

```bash
sp_transcode <in.gguf> <out.sp-model> <out.sp-tokenizer> [--verify]
```

`--verify` runs a per-tensor round-trip dequant check (rms + max
rel-error) against the GGUF source and rejects the output if a Q8 row
exceeds the threshold.

### 10.2 Supported inputs

GGUF v3 files containing:
- Architectures: Llama-3, Qwen3, Qwen2.5, Gemma3, DeepSeek V4 (per
  `sp_arch_id` enum in `include/sp/sp_model.h`)
- Per-tensor dtypes: `GGML_T_F32`, `GGML_T_F16`, `GGML_T_Q8_0` (others
  return "unsupported src type" and abort)
- Tokenizers: SentencePiece, BPE-Llama3, BPE-GPT2, TikToken-O200K

### 10.3 Output format

Per `shannon-prime-lattice/papers/PPT-LAT-SP-MODEL-v0.md`:

- **`.sp-model`** — fixed 512-byte header + tensor table (256 B per
  entry, sorted by xxh64 of the tensor name) + data region (64 KB
  aligned, each tensor 64 B aligned). The header carries `arch_id`
  (`SP_ARCH_ID_QWEN3 = 2`, `SP_ARCH_ID_GEMMA3 = 3`, etc.) + 256 B of
  arch_struct payload memcpy'd from `sp_arch_info`.
- **Per-tensor policy:**
  - matmul weights (attn q/k/v/o, ffn gate/up/down, LM head,
    `token_embd`): dequant to f32, re-quant into `SP_DT_OK_Q8` (int8
    per-row codes) + sibling `<name>.scale` tensor
    (`SP_DT_FROBENIUS_SCALE_FP32`, one fp32 per row).
  - norms and other tensors: copied as F32 (dequant F16→F32 if needed).
- **Data-region layout:** weights and their `.scale` siblings are
  adjacent (no interposing tensor), so the loader reconstructs a
  bit-identical packed arena via a single `memcpy` per row.
- **`.sp-tokenizer`** — self-describing serialization of the GGUF
  tokenizer arrays (tokens, scores, merges) + 128-byte header carrying
  type_id, vocab size, BOS/EOS/PAD/UNK IDs, and a SHA-256 over the
  whole file. The L1 loader binds models to tokenizers by this hash
  (`SP_ETOKENIZER_HASH` if mismatched).

### 10.4 Validation

`--verify` mode reports per-tensor stats:

```
qwen3.layers.0.attn_q.weight (4096 x 4096, Q8):
    rms_err 0.000183  max_rel_err 0.0021  promoted 0/4096 to Q8
```

For Q4 mode (`SP_ARENA=q4`), rows exceeding `SP_Q4_PROMOTE` (default
`0.01`) get promoted to Q8 — the promoted count reports as
`sp_arena_promoted(arena)`.

---

## 11. Peering / QUIC mesh

The mesh is a **dual-prime CRT shard fabric** today, with the
Fibonacci-Prime DHT spec'd for Phase 8 (see
`shannon-prime-lattice/papers/PPT-LAT-Roadmap.md` §8).

### 11.1 Wire format

Each peer-to-peer message is a 64-byte `ShardBlockHeader` followed by N
× 4 bytes of u32 residue payload:

```
byte  0..8   seq_id           u64 LE   global sequence counter
byte  8..12  token_pos        u32 LE   token position in context
byte 12..16  layer_id         u32 LE   transformer layer index
byte 16      prime_selector   u8       0 = q_1 = 1073738753, 1 = q_2 = 1073732609
byte 17..64  reserved         zeros
```

Max payload: 64 + 512·4 = 2112 bytes (`N ≤ 512` per the frozen-primes
NTT cap). Streams are unidirectional QUIC streams — one per residue
block; independent delivery; no head-of-line coupling.

### 11.2 Topology

- **Coordinator** (`SpQuicCoordinator::bind`) accepts incoming connections
  and per-stream residue blocks; calls `run_garner_loop` to Garner-
  recombine paired (q_1, q_2) residues for the same `seq_id` into
  centered signed coefficients.
- **Worker** (`SpQuicWorker::connect`) dials a coordinator and sends
  `ResidueBlock`s on independent unidirectional streams.

Each peer is assigned one prime (its `shard_id`). Two-peer topology
covers both primes; the coordinator Garner-recombines into the centered
signed result and feeds it back into the forward path.

### 11.3 TLS / identity

Dev-mode TLS uses self-signed certs with a `SkipServerVerification`
verifier — acceptable for the inference-cluster smoke. Phase 5+
swaps this for ed25519 dominance identity verification against a known
lattice node registry; the integration point is documented in
`tools/sp_daemon/src/network/quic_shard.rs` (search "INTEGRATION POINT:
Replace with Phase 5 ed25519 dominance identity").

### 11.4 Receipt replay

PoUW receipts minted on one node can be replayed byte-identically on
another node via `Ledger::canonical_sort` (stable sort on
`(turn_index, input_hash[..2])`) → `Ledger::replay_canonical_into`
(write a new ledger with canonical order). Cross-device byte-identity
is the M.4 + mesh-canonical-order gate; see closure
`CLOSURE-MESH-CANONICAL-ORDER.md`.

### 11.5 Connecting peers

```cmd
sp-daemon start --model ... --tokenizer ... ^
                --port 8080 --quic-port 5000 ^
                --peers 192.0.2.10:5001,192.0.2.11:5002
```

`/v1/mesh/peers` reports the live peers. `/v1/node/telemetry`
WebSocket pushes `dht_peers_active` every second.

---

## 12. Development workflow

### 12.1 Build matrix

| Build | Command |
|-------|---------|
| CPU host (Windows) | `scripts\build\build-cpu.bat` |
| CUDA host (Windows) | `scripts\build\build-cuda.bat` |
| Vulkan host (Windows) | `scripts\build\build-vulkan.bat` |
| Hexagon host-side libs (Windows) | `scripts\build\build-hexagon.bat` |
| Hexagon cDSP skel | `scripts\build\build-hexagon.bat dsp` |
| Daemon-linkable hex backend lib | `tools\sp_daemon\build-android-hex-backend.bat` |
| Cross-compile math-core to android | `tools\sp_daemon\build-android-libs.bat` |
| `sp_daemon` cargo build (host) | `cd tools\sp_daemon && cargo build --release` |
| `sp_daemon` cargo build (android) | `cargo build --target aarch64-linux-android --release` |
| `sp_daemon` with WIRE-HEX | `cargo build --target aarch64-linux-android --release --features wire_hex_backend` |

### 12.2 Run smoke tests

```bash
# All ctest gates on a build directory
ctest --test-dir build-cpu --output-on-failure -j

# A specific smoke binary (host)
cd tools/sp_daemon && cargo run --release --bin probe

# Android-only smokes (push + run via adb)
adb push target/aarch64-linux-android/release/sp_chat_dialogue_smoke /data/local/tmp/
adb shell /data/local/tmp/sp_chat_dialogue_smoke
```

Closures from recent smokes live under
`tools/sp_compute_skel/docs/CLOSURE-*.md`. They are the audit trail.

### 12.3 Adding a new backend

The canonical pattern is sprint WIRE-HEX
(`tools/sp_compute_skel/docs/CLOSURE-WIRE-HEX.md`). Five stages:

1. **Math-core: add §6 forward-backend hook** — already shipped in
   `include/sp/sp_l1.h` (`sp_session_register_forward_backend`,
   `sp_forward_dispatch_fn` typedef, `sp_session_qwen3_model`
   accessor).
2. **Engine: build a daemon-linkable static lib** at
   `tools/sp_daemon/c_backend/lib<name>_daemon_backend.a` containing
   your `<arch>_forward_<backend>` entry point + a kernel-name shim
   that aliases `matmul`/`embed_row`/`as_f32` to the math-core
   `sp_*` variants (avoids `cpu_overlay.c` symbol collisions).
3. **Rust trampoline** at `tools/sp_daemon/src/<backend>_forward_dispatch.rs`
   implementing the §6 ABI; bump a process-static dispatch counter.
4. **AppState wiring** in `tools/sp_daemon/src/daemon.rs`: env-gate via
   `SP_DAEMON_BACKEND=<backend>`; register on the TARGET session
   pre-Mutex-wrap; surface via `AppState.<backend>_active`.
5. **Smoke**: drive `/v1/chat` + `/v1/debug/backend_counts` to confirm
   `<backend>_forward_count > 0` after one prefill.

For NTT-dispatch overlay backends (Hexagon NTT.5b, future Vulkan-NTT),
the pattern is `sp_pr_bluestein_set_backend` in
`lib/shannon-prime-system/core/poly_ring_bluestein/` (see
`tools/sp_daemon/src/ntt_hex_dispatch.rs` for the template).

### 12.4 Adding / updating an IDL method

1. Edit `tools/sp_compute_skel/inc/sp_compute.idl`. Add the new method
   at the end (renumber if a parallel sprint took your anticipated
   method id at merge time).
2. Regenerate skel stubs via qaic. The build script handles this; if
   you need to do it manually:
   ```cmd
   set HEXAGON_SDK_ROOT=C:\Qualcomm\Hexagon_SDK\5.5.6.0
   "%HEXAGON_SDK_ROOT%\tools\qaic\Ubuntu18\qaic" -mdll ^
       -o tools\sp_compute_skel\gen ^
       tools\sp_compute_skel\inc\sp_compute.idl
   ```
3. Implement the method body in `tools/sp_compute_skel/src_dsp/`.
4. Rebuild + push the skel: `scripts\build\build-hexagon.bat dsp` then
   `adb push <skel.so> /data/local/tmp/sp22u/`.
5. Add a Rust trampoline if a daemon-facing surface is desired.

### 12.5 Parallel-agent worktree discipline

Per `feedback-parallel-agents-separate-worktrees`: when dispatching 2+
agents concurrently on the same repo, **each agent operates in its own
git worktree** (`git worktree add ../wt-<sprint> main`). Otherwise
concurrent `git add` cross-contaminates: one agent's uncommitted files
get swept into another's commit. The operational fix is per-worktree
dispatch before agents start; the technical recovery (if it slips) is
to honestly disclose contamination in the closure rather than rewriting
shared history.

---

## 13. Known issues / pending

The user has been frustrated by hours of work that turned out to
bypass the production critical path. This section names what's pending
honestly.

| Issue | Workaround | Resolution |
|-------|-----------|------------|
| **WIRE-HEX BIT-EXACT gate blocked by cDSP skel mismatch** | None for the headline tok/s win | Future HX-SKEL-REBUILD sprint owns rebuilding `libsp_hex_skel.so` against current `src/backends/hexagon/inc/sp_hex.idl` and pushing to `/data/local/tmp/sp22u/` |
| **CPU AVX-512 / CUDA / Vulkan backends not wired to `sp_daemon`** | Use math-core reference path | Three symmetric WIRE-HEX-style sprints; each one is ~1 day of plumbing once the WIRE-HEX template is in hand |
| **`sp_decode_step` uses fp32 reference even with `SP_ENGINE_NTT_ATTN=1`** | Decode is the path where this matters most for tok/s; current architecture re-runs full forward on prefill backends | NTT.5e (filed, not shipped) wires decode-path NTT routing |
| **HD=128 direct path can't use Hexagon backend** | Bluestein at HD=64 covers Qwen3 / Qwen2.5-Coder; Gemma3-1B uses HD=256 (direct N=256 NTT works) | NTT.5d (filed, not shipped) wires a direct backend path at HD=128 |
| **Hexagon backend re-runs full forward per call** | Decode path stays on math-core reference; bypasses the issue | HEX-DECODE-1 sprint would add per-backend persistent-KV variant |
| **CPU backend's `cpu_overlay.c` symbol-collides with math-core's `sp_*` kernels** | The daemon-link backend lib uses a kernel-name shim (`sp_daemon_hex_glue.c`) to alias names | Same shim pattern applies to future CUDA / Vulkan daemon links |
| **TLS in QUIC mesh accepts any cert** | Dev-only; lattice clusters today are operator-controlled | Phase 5 ed25519 dominance identity verification swap |
| **Tokenizer chat-template support varies by arch** | `/v1/chat` falls back to `prompt` / `prompt_tokens` if `messages` template lookup fails | Per-arch template registration is an open task |
| **`tracing_subscriber` filter set at daemon start, not hot-reloadable** | Restart daemon to change `RUST_LOG` | `sp-daemon reload` is a no-op for v0 |
| **CRT-mesh today is two-node (one shard per peer)** | Sufficient for the dual-prime CRT bit-exact smoke | Fibonacci-Prime DHT is spec'd (`papers/PPT-LAT-Roadmap.md` §8) |

For the audit trail of what shipped when (and what didn't), the
canonical reference is the chronological closure list under
`tools/sp_compute_skel/docs/CLOSURE-*.md` plus the lattice
`papers/SESSION-CLOSED-*.md`. The most recent closures
(`CLOSURE-WIRE-HEX.md`, `CLOSURE-NTT-bench.md`, `CLOSURE-NTT-5c.md`,
`CLOSURE-LEDGER-AUTOWIRE.md`, `CLOSURE-MESH-CANONICAL-ORDER.md`) are
the up-to-date status of record.

---

## License

AGPL-3.0-or-later. See `LICENSE`. Commercial licensing available — contact
the copyright holder.
