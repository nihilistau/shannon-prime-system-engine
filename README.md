# shannon-prime-system-engine

**The inference engine of [Shannon-Prime](https://github.com/nihilistau/shannon-prime-lattice)** — accelerator backends, the sovereign weight pipeline, and the served Gemma-4-12B daemon, all built on the math core ([`shannon-prime-system`](https://github.com/nihilistau/shannon-prime-system), carried here as the `lib/shannon-prime-system` submodule).

> **Status badges (honest tiers — verify against `HISTORY.md` + the gate fixtures):**
> `served 12B chat + W_c recall` **GREEN-LIVE** · `byte-exact forward` **gated-GREEN / default-off** · `NIGHTSHIFT offline curator` **gated-GREEN on synthetic (live PENDING)** · `native diffusion judge` **UNPROVEN / in-drawer**

**License:** MIT (see `LICENSE`). **HEAD:** `6107f3e`. Public site: [Position Is Arithmetic](https://nihilistau.github.io/Position_Is_Arithmetic/).

---

## What this is

Shannon-Prime runs a **frozen Gemma-4-12B** on a **byte-exact, exact-integer** substrate (O_K = Z[(1+√−163)/2], a dual-prime CRT-NTT) on **one RTX 2060 12 GB**, with a token-free, receipted conversational-memory organism (XBAR). This repo is the **engine**: it wires the math-core forward onto four accelerator backends, ships the safetensors-direct weight transcoder, and serves the live 12B chat (with autonomous episodic recall) through the `sp_daemon` HTTP/SSE server.

### The citable headline

**Gemma-4-12B at 26.1 tok/s @ wikitext PPL 5.12 on an RTX 2060 12 GB** — the OK_Q4B sovereign artifact (public ledger `06-R10`; CUDA-graph decode EXACT 256/256, dp4a top-1 256/256). Both halves stated honestly: llama.cpp is faster (31.29 tok/s) **but on gemma-4 GGUF artifacts whose measured PPL is 192–506** — every gemma-4 GGUF measurable in June 2026 ships broken tensor data, which is exactly why the safetensors-direct path below is the only trusted one.

---

## Honest-tier capability map

| Capability | Tier | Flag / launcher | Gate · commit |
|---|---|---|---|
| **Served Gemma-4-12B chat** — coherent + byte-exact + O(1)-context on a single latent entry point | **GREEN-LIVE** | `run_console.bat` → `http://127.0.0.1:3000/` | `CONTRACT-CHAT-FULLSTACK` · engine →`7eb7231` |
| **Autonomous episodic recall** — learned latent `W_c` head; (E+1)-way argmax over `[episodes, NULL]`, replays the winner @ bounded mass `M=42` or rejects to a clean prompt | **GREEN-LIVE** | `SP_B3_WC` (`run_console_recall.bat`) | `G-CHAT-B3-WC-DEPLOY` / `-DIV2` (360/361 recall + 50/50 foreign-reject, int16==f32) · `edc8079` |
| **Byte-exact exact-integer forward** — RMSNorm/softmax/GELU/RoPE + attention as exact-integer device kernels; = exact arithmetic / cross-machine determinism / **AUDITABILITY, not compression** | **gated-GREEN / default-off** | `SP_BYTEEXACT` (unset = byte-identical null floor) | `G-BYTEEXACT-FORWARD-12B` (OFF PPL 4.6665 byte-identical / ON 4.6569 parity / run-to-run bit-identical) · `69c0588` |
| **NIGHTSHIFT offline curator** — 12B model-call `ep.secret` extractor → teacher-forced causal-ablation admit (TAU=−8) → conformant MEM-OKF emit | **gated-GREEN on SYNTHETIC; live PENDING** | `SP_NIGHTSHIFT_OFFLINE=1` (`run_console_nightshift.bat`) | `G-NIGHTSHIFT-CURATOR` criteria 1-4 (novel `8-FALCON-7729` collapse −33.59 ACCEPT / parametric `Paris` 0.00 REJECT; accepted=1 rejected=1, addr-join verified). **Criterion 5 (live B4 in-distribution) NOT yet run on real chat turns.** · `6107f3e` |
| **Native diffusion judge** (DiffusionGemma 26B-A4B reads candidate TEXTS, selects via language) | **UNPROVEN / in-drawer** | — | The **95.6% is the *external* llama.cpp PR-24423 oracle's number, NOT ours.** Our native single-forward judge was **FALSIFIED at ~25%** (`f8f76a5`); the iterative denoise loop is built (`0244800`) but the full gate is still cooking, I/O-blocked behind the design-only N5b reservoir |
| **Two-ring long-context memory (PPT-ARM)** — ±1 Rademacher recall router + byte-addressable KV offload to NVMe/Optane (CPU backend) | shipped + measured | `SP_RECALL_*` / `SP_RING2_*` | **910× resident-KV shrink @32k**, 7.57 µs/read off Optane, **8× sparsification @ +0.69% PPL**, bit-exact when off. **Honest negative kept on the board: the 32k NIAH finale MISSed** (raw router breaks the 64× selection-budget; Ring 3 is the designed fix) |
| **XBAR memory on the exact-integer O_K substrate** — Ring-3 VSA bind, Frobenius Ring-2 store, organism loop | gated-GREEN | host-tools, env-gated | `G-R3-BIND-on-O_K` (`0019b86`, 256/256 bit-identical), `G-R2-FROB`, `G-XBAR-ORGANISM-FULL` |

The **boundary thesis** runs through all of it: O_K wins on **exact arithmetic** (the container); every *structure-on-content* lever (Möbius, split-prime Dirichlet carriers, entropy-coded Frobenius codes, T2-Möbius on the real embedding) is **measured-inert and kept as an honest negative**.

> Every "on" result is a controlled delta against a byte-identical baseline. The one-shot production decode (`gemma4_decode_cuda`) is **never touched** — when every `SP_*` flag is unset the engine is at its null floor, so every closed throughput / PPL / NIAH gate stays valid.

---

## Architecture

```
                          sp_daemon  (Rust HTTP / SSE / WebSocket)
                          /v1/chat · W_c recall · NIGHTSHIFT curator · PoUW ledger · QUIC mesh
                                          │
                          frozen L1 C ABI (sp_session_register_*_backend)
                                          │
        ┌─────────────────┬───────────────┴───────────────┬─────────────────┐
        ▼                 ▼                               ▼                 ▼
   CUDA backend      CPU backend                    Vulkan backend     Hexagon HVX
  cuda_forward.cu   cpu_forward.c +                vulkan_forward.cpp   sp_hex_host.c
  (gemma4 fwd/      cpu_overlay.c                  (desktop target)     + cDSP skel
   decode, OK_Q4B   (two-ring PPT-ARM,
   dp4a, SP_XBAR/    Optane Ring-2,
   SP_BYTEEXACT,     WIRE-CPU int pipe)
   kv_decode_logits)
        │
        └──────────► lib/shannon-prime-system  (math core: decode loop, O_K / CRT-NTT,
                     exact-islands, ARM recall router, frozen L1 ABI)

   tools/sp_transcode --st  ──►  OK_Q8 / OK_Q4B .sp-model   (safetensors-direct;
                                                             the ONLY trusted weight path)
```

### Key engine surfaces

- **`src/backends/cuda/cuda_forward.cu`** — the gemma4 CUDA forward + decode (per-layer SWA/global geometry, shared-KV, AltUp/PL=0, softcap); the **OK_Q4B `k_gemv_q4b_dp4a` kernel**; the CUDA-graph decode path; the **`SP_XBAR_*` harness** (P1 KV splice/capture + P2.a residual injection + rank/score lanes); the **`SP_BYTEEXACT` exact-integer islands** (RMSNorm/softmax/GELU/RoPE + attention `k_attn_decode_win_bx`, device dual-prime, default-off = byte-identical null floor); and the additive **`gemma4_kv_decode_logits`** (the daemon's token-by-token decode entry). The persistent-KV ABI (`gemma4_kv_open/prefill/decode/rewind/inject/inject_seq/close`) lives here too.
- **`src/backends/cpu/`** — overlay dispatch into the math-core decode (two-ring PPT-ARM, Optane Ring-2, the WIRE-CPU integer pipe). `vulkan/`, `hexagon/` are built desktop/edge targets.
- **`tools/sp_transcode/`** — **`sp_transcode --st`**: the safetensors-direct pipeline, **the ONLY trusted gemma4-12B weight path** (the GGUF lane is dead — the 2026-06 gemma-4 GGUF wave shipped corrupted tensor data; gold forward PPL 4.68 vs GGUF 271–364 on identical arithmetic). Writes OK_Q8 / OK_Q4B `.sp-model`.
- **`tools/sp_daemon/`** — the universal resident daemon. Feature `wire_cuda_backend` drives the real 12B end-to-end through the L1 ABI: prefill via `sp_session_register_forward_backend` (`G-WIRE-CUDA-GEMMA4`) and token-by-token DECODE via the L1 verb `sp_session_register_kvdecode_backend` (`G-WIRE-CUDA-DECODE-GEMMA4`, 32/32 == oracle, VRAM flat O(1)).
  - `src/recall.rs` — the learned **`W_c` head** (`WcHead`/`load_wc`/`wc_score`; HD=512→r=32, relevance = logsumexp-over-positions then mean-over-heads).
  - `src/routes.rs` — the `SP_B3_WC` recall branch (the (E+1)-way NULL argmax + bounded-mass replay).
  - `src/nightshift_curator.rs` — the NIGHTSHIFT offline curator (`run_kairos_curator`; feature `kairos`, gated on `SP_NIGHTSHIFT_OFFLINE=1`; opens its own kvdecode handle, never touches the served cache).
- **`tools/sp_dsp_smoke/`** — the L2 universal Rust crate: the dual-prime Barrett / mod-q matmul / Garner CRT / NTT ladder is bit-exact-gated here, and the 4 nonlinear islands have exact-integer references in `src/sp_islands_q_ref.rs` (RoPE via deterministic fixed-point CORDIC, no libm).
- **Tests / gates** — `tests/test_gemma4_cuda.c` (env-dispatched harness modes), `tests/test_xbar_p1_cuda.c`, `tests/bench_gemv_int8.cu`, the gate bin `tools/sp_daemon/src/bin/sp_wire_cuda_decode_gate.rs`. Receipts in `tests/fixtures/`.

---

## Build

Authoritative: `docs/BUILD-ENV.md`. Two distinct toolchains (do not mix):

| Backend | Toolchain | Build dir | Notes |
|---|---|---|---|
| **CUDA** (canonical) | VS2019 BuildTools + CUDA 12.4, ninja | `build-cuda/` | sm_75 on the dev RTX 2060 (tightly pinned — newer VS / older CUDA breaks nvcc's MSVC integration) |
| **CPU** (canonical) | MinGW gcc 15.2, ninja | `build/` | MSVC cannot build the CPU tree. Standalone gcc needs `-D_POSIX_C_SOURCE=200809L -D_FILE_OFFSET_BITS=64` |

```bat
:: a CUDA gate
cd build-cuda && ninja test_gemma4_cuda && tests\test_gemma4_cuda.exe

:: the wire-cuda decode gate (daemon drives the 12B token-by-token)
cargo run --release --features wire_cuda_backend --bin sp_wire_cuda_decode_gate
```

GPU numbers need warmup + a long window + **both clocks pinned** (`-lgc` is SM-only; a weight-GEMV is memory-bound).

---

## Run the served chat

```bat
run_console.bat               :: coherent + byte-exact + O(1) 12B chat -> http://127.0.0.1:3000/
run_console_recall.bat        :: + the autonomous W_c librarian (SP_B3_WC)
run_console_nightshift.bat    :: + the offline NIGHTSHIFT curator (SP_NIGHTSHIFT_OFFLINE=1)
```

A matched query recalls its episode (e.g. `ep_n_div_000`); a foreign query ("capital of France?") rejects to NULL and answers cleanly ("Paris"). The console Send button toggles to an interrupt (POST `/v1/abort/:id`).

### Harness modes (env-var test gates)

The KAIROS / XBAR / byte-exact gates ship as **env-var-dispatched modes inside `tests/test_gemma4_cuda.c`** — each is byte-inert when its env is unset, runs the real 12B, prints its receipt, exits. A few:

| Env var | What it proves |
|---|---|
| `SP_BYTEEXACT` | exact-integer islands + attention; OFF = byte-identical null floor, ON = PPL parity + run-to-run bit-identical |
| `SP_B3_WC` | the learned W_c recall head (live selector) |
| `SP_B3_SECRET` / `SP_B3_DISPOSER=2` | the teacher-forced ablation oracle (admission gate + labeler; TAU=−8) |
| `SP_NIGHTSHIFT_OFFLINE` | the offline curator loop (extract → admit → MEM-OKF emit) |
| `SP_G4_NIAH` | a 16k-haystack needle survives the slab/ring compaction (learned router only) |
| `SP_REPLAY` / `SP_REPLAY_MTARGET` | episode replay-write + bounded injection mass |

---

## Navigation

- **`AGENTS.md`** — agent entry point + the MEM-OKF anti-rebuild pre-flight + the non-negotiables.
- **`HISTORY.md`** — the hashed tiered commit log (Tier-0 LUT; `git show <hash>` is the tier-2 store).
- **`CLAUDE.md`** — this repo's specifics for an agent picking up work.
- **Upstream truth** — lattice `papers/PPT-LAT-STATE.md` (the proven record) + `papers/STATUS-MAP-2026-06-21.md` (box-by-box honest tiers) + the active contract (`CONTRACT-CHAT-FULLSTACK`, `CONTRACT-BYTEEXACT-forward`, `CONTRACT-NIGHTSHIFT-CURATOR`).
- **Public face** — [Position Is Arithmetic](https://github.com/nihilistau/Position_Is_Arithmetic) (receipts-first papers + `LEDGER.md`).

---

## Non-negotiables (receipts-first)

- **No number without a reproducing command + a gate/commit.** Bit-exact-when-off: every `SP_*` overlay is a strict no-op by default — verify it.
- **No silent gate revision** — surface upstream. Honest negatives stay attached (the 32k NIAH MISS, the diffusion-judge falsification, the boundary-thesis inert levers).
- **Check the code + commits + `git fetch` before trusting memory.** `lib/shannon-prime-system` is a submodule of the same upstream as the standalone math-core checkout, so the two can diverge — `git fetch` and check `behind` before building/committing.
- **gated-GREEN is not GREEN-LIVE.** The byte-exact forward and the NIGHTSHIFT curator pass their gates behind a flag (null floor when unset); they are not on the served path by default.
