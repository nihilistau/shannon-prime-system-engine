# shannon-prime-system-engine — the inference engine + resident daemon + memory agency

**The engine of [Shannon-Prime](https://github.com/nihilistau/shannon-prime-lattice)** — accelerator
backends, the sovereign weight pipeline, the served Gemma-4-12B daemon, and the **model-owned memory
agency**. Built on the math core ([`shannon-prime-system`](https://github.com/nihilistau/shannon-prime-system),
carried here as the `lib/shannon-prime-system` submodule).

## What Shannon-Prime is

Shannon-Prime is a **fully local, byte-exact, auditable language-model organism**. It serves Google's
**Gemma-4-12B** (OK_Q4B quant) on a single **RTX 2060**, through **our own** inference engine, on an
**exact-integer arithmetic substrate** (`O_K = Z[(1+√−163)/2]`, dual-prime negacyclic CRT-NTT), with
a working memory it owns: it learns facts from conversation, recalls them, **forgets / supersedes /
merges** them on its own judgement, calls tools and runs code, stores whole conversations both
complete and summarized, and consolidates them on a heartbeat. Every mechanism is a flag that is a
**strict no-op when unset** (the "null floor"); every number has a reproducing command and a gate. No
cloud, no third-party inference, no telemetry.

**This repo is the engine + the resident daemon.** It wires the math-core forward onto four
accelerator backends, ships the safetensors-direct weight transcoder, serves the live 12B chat
(autonomous recall + memory agency) through `sp_daemon` (HTTP/SSE), and runs the diffusion-judge MoE.

> **Read this first:** lattice [`papers/PPT-LAT-KEYSTONE.md`](https://github.com/nihilistau/shannon-prime-lattice/blob/main/papers/PPT-LAT-KEYSTONE.md) —
> the canonical, current, complete description of the whole system. This README is this repo's
> specifics. Public site: [Position Is Arithmetic](https://nihilistau.github.io/Position_Is_Arithmetic/).
> License: MIT (`LICENSE`).

### The citable headline

**Gemma-4-12B at 26.1 tok/s @ wikitext PPL 5.12 on an RTX 2060 12 GB** — the OK_Q4B sovereign
artifact (public ledger `06-R10`; CUDA-graph decode EXACT 256/256, dp4a top-1 256/256). Stated
honestly: llama.cpp is faster (31.29 tok/s) **but on gemma-4 GGUF artifacts whose measured PPL is
192–506** — every gemma-4 GGUF measurable in June 2026 ships broken tensor data, which is exactly why
the safetensors-direct path below is the only trusted one.

## Architecture

```
                          sp_daemon  (Rust HTTP / SSE / WebSocket)
                          /v1/chat · W_c recall · FORGET/DECIDE/MERGE · NIGHTSHIFT · PoUW · QUIC mesh
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
   dg_* MoE judge,
   kv_decode_logits)
        │
        └──────────► lib/shannon-prime-system  (math core: decode loop, O_K / CRT-NTT,
                     exact-islands, ARM recall router, frozen L1 ABI)

   tools/sp_transcode --st  ──►  OK_Q8 / OK_Q4B .sp-model   (safetensors-direct;
                                                             the ONLY trusted weight path)
```

## Honest-tier capability map

`gated-GREEN` is **not** GREEN-LIVE: a default-off flag is a null floor until set.

| Capability | Tier | Flag / launcher | Gate · commit |
|---|---|---|---|
| **Served Gemma-4-12B chat** — coherent + byte-exact + O(1)-context, single latent entry, with **EOT bias** for clean stops + a default **system prompt** for in-context faithfulness | **GREEN-LIVE** | `run_console.bat` → `http://127.0.0.1:3000/` | `CONTRACT-CHAT-FULLSTACK` · `9e4b40f` / `88d924e` |
| **Autonomous episodic recall** — learned latent `W_c` head; (E+1)-way argmax over `[episodes, NULL]`, replays the winner @ bounded mass `M=42` or rejects to a clean prompt | **GREEN-LIVE** | `SP_B3_WC` (`run_console_recall.bat`) | `G-CHAT-B3-WC-DEPLOY` / `-DIV2` (360/361 recall + 50/50 reject, int16==f32) · `edc8079` |
| **Model-owned memory agency** — STORE (NIGHTSHIFT) + FORGET (`SP_FORGET`) + DECIDE/MERGE (`SP_DECIDE`: supersede a changed fact, or consolidate two complementary facts into one synthesized truth) | **GREEN-LIVE** | `SP_FORGET`, `SP_DECIDE` | `G-FORGET` / `G-DECIDE` / `G-MERGE` · `0fd52e4` |
| **Production recall/reject judge** — deterministic token-overlap (Jaccard) verifier @≈0.6 (83% recall / 95% reject on a CPU string op; beats the 26B cascade ~53%/98%, frees the GPU) | **GREEN** | (in the judge path) | `G-JUDGE-BATTERY{,-C,-D,-E}` |
| **Byte-exact exact-integer forward** — RMSNorm/softmax/GELU/RoPE + attention as exact-integer device kernels = exact arithmetic / cross-machine determinism / **AUDITABILITY, not compression** | **gated-GREEN / default-off** | `SP_BYTEEXACT` (unset = byte-identical null floor) | `G-BYTEEXACT-FORWARD-12B` (OFF PPL 4.6665 byte-identical / ON 4.6569 parity / run-to-run bit-identical) · `69c0588` |
| **NIGHTSHIFT offline curator** — 12B model-call `ep.secret` extractor → teacher-forced causal-ablation admit (TAU=−8) → conformant MEM-OKF emit | **gated-GREEN on SYNTHETIC; live PENDING** | `SP_NIGHTSHIFT_OFFLINE=1` | `G-NIGHTSHIFT-CURATOR` criteria 1-4 · `6107f3e` |
| **Diffusion-judge throughput** — MoE spillover-expert streaming (per-expert scratch-pool reuse + pinned async double-buffer prefetch, ~2x stacked, byte-exact); plus prefix-KV (~1.6x, answer-lossless) | scratch-reuse **PROVEN default-on**; async/prefix-KV byte-exact, **default-off** | `SP_DG_SCRATCHREUSE` (on) · `SP_DG_ASYNC` / `SP_DG_PREFIXKV` (off) | `e31c70d` / `2a1c830` / `5276662` |
| **Native diffusion judge** (DiffusionGemma 26B-A4B reads candidate texts, selects via language) | **UNPROVEN / in-drawer** — superseded by the deterministic verifier above | — | native single-forward falsified ~25%; the 95.6% is the *external* llama.cpp oracle's number, not ours |

The **boundary thesis** runs through all of it: O_K wins on **exact arithmetic** (the container);
every *structure-on-content* lever (Möbius, split-prime Dirichlet carriers, entropy-coded Frobenius
codes, T2-Möbius on the real embedding) is **measured-inert and kept as an honest negative**.

> Every "on" result is a controlled delta against a byte-identical baseline. When every `SP_*` flag is
> unset the engine is at its null floor, so every closed throughput / PPL / NIAH gate stays valid.

## Key engine surfaces

- **`src/backends/cuda/cuda_forward.cu`** — the gemma4 CUDA forward + decode (per-layer SWA/global
  geometry, shared-KV, AltUp/PL=0, softcap); the **OK_Q4B `k_gemv_q4b_dp4a` kernel**; the CUDA-graph
  decode path; the **`SP_XBAR_*` harness**; the **`SP_BYTEEXACT` exact-integer islands** (default-off
  = byte-identical null floor); the **diffusiongemma-26B-A4B MoE judge** (`dg_*` path); the additive
  **`gemma4_kv_decode_logits`** (the daemon's token-by-token decode entry) + the persistent-KV ABI
  (`gemma4_kv_open/prefill/decode/rewind/inject/inject_seq/ablate/close`).
- **`src/backends/cpu/`** — overlay dispatch into the math-core decode (two-ring PPT-ARM, Optane
  Ring-2, the WIRE-CPU integer pipe). `vulkan/`, `hexagon/` are desktop/edge targets.
- **`tools/sp_transcode/`** — **`sp_transcode --st`**: the safetensors-direct pipeline, **the ONLY
  trusted gemma4-12B weight path** (the GGUF lane is dead). Writes OK_Q8 / OK_Q4B `.sp-model`.
- **`tools/sp_daemon/`** — the universal resident daemon (feature `wire_cuda_backend` drives the real
  12B end-to-end through the L1 ABI):
  - `src/recall.rs` — the learned **`W_c` head** (`WcHead`/`load_wc`/`wc_score`; HD=512→r=32,
    relevance = logsumexp-over-positions then mean-over-heads).
  - `src/routes.rs` — the `/v1/chat` path: the `SP_B3_WC` recall branch ((E+1)-way NULL argmax +
    bounded-mass replay), the **memory agency** (`SP_FORGET` token-overlap drop + persisted-registry
    rewrite; `SP_DECIDE` supersede + MERGE via a side model-call framed as DETECTION + a forced
    answer prefix), the EOT bias, and the `SP_CURRENT_CONVO` consolidation hook.
  - `src/nightshift_curator.rs` — the NIGHTSHIFT offline curator (`run_kairos_curator`; opens its own
    kvdecode handle, never touches the served cache).
  - `src/kairos.rs` — the heartbeat / agency-tick control-plane stub (the model-driven realization
    lives in the harness `run_agency.py`).
- **`tools/sp_dsp_smoke/`** — the L2 universal Rust crate: the dual-prime Barrett / mod-q matmul /
  Garner CRT / NTT ladder is bit-exact-gated here; the 4 islands have exact-integer references in
  `src/sp_islands_q_ref.rs` (RoPE via fixed-point CORDIC, no libm).
- **Served console** — `index.html` (the GUI: knobs on the left, the default system prompt, the
  Send→interrupt toggle). Tests / gates: `tests/test_gemma4_cuda.c`, `tests/test_xbar_p1_cuda.c`,
  receipts in `tests/fixtures/`.

## Build

Authoritative: `docs/BUILD-ENV.md`. Two distinct toolchains (do not mix):

| Backend | Toolchain | Build dir | Notes |
|---|---|---|---|
| **CUDA** (canonical) | VS2019 BuildTools + CUDA, ninja | `build-cuda/` | sm_75 on the dev RTX 2060 (tightly pinned) |
| **CPU** (canonical) | MinGW gcc 15.2, ninja | `build/` | MSVC cannot build the CPU tree |

```bat
:: a CUDA gate
cd build-cuda && ninja test_gemma4_cuda && tests\test_gemma4_cuda.exe

:: the daemon (drives the 12B token-by-token over the L1 ABI)
cargo build --release --features wire_cuda_backend
```

GPU numbers need warmup + a long window + **both clocks pinned**. Git on these repos: **native
PowerShell, not the Linux mount** (the mount CRLF-churns large files + locks).

## Run the live organism

```bat
run_console.bat               :: coherent + byte-exact + O(1) 12B chat -> http://127.0.0.1:3000/
run_console_recall.bat        :: + the autonomous W_c librarian (SP_B3_WC)
run_console_nightshift.bat    :: + the offline NIGHTSHIFT curator (SP_NIGHTSHIFT_OFFLINE=1)
_e2e_seed_serve.bat           :: the full stack (EOT bias + recall + forget + decide + nightshift + seed registry)
```

Then, alongside the daemon, run the harness agency scheduler for the heartbeat consolidation +
maintenance (`shannon-prime-harness/run_agency.py`). A matched query recalls its episode; a foreign
query ("capital of France?") rejects to NULL and answers cleanly ("Paris"). State a fact and it is
captured; say "forget X" and it is dropped; state a contradicting/complementary fact and DECIDE
supersedes or MERGEs it. Full procedure: lattice `papers/PPT-LAT-KEYSTONE.md` §10.

## Navigation

- **The whole system, current + complete** — lattice `papers/PPT-LAT-KEYSTONE.md`.
- **`AGENTS.md`** — agent entry + the MEM-OKF anti-rebuild pre-flight + the non-negotiables.
- **`HISTORY.md`** — the hashed tiered commit log (`git show <hash>` for detail).
- **`CLAUDE.md`** — this repo's specifics for an agent picking up work.
- **Upstream truth** — lattice `papers/PPT-LAT-STATE.md` + the active contracts
  (`CONTRACT-CHAT-FULLSTACK`, `CONTRACT-BYTEEXACT-forward`, `CONTRACT-NIGHTSHIFT-CURATOR`).
- **Public face** — [Position Is Arithmetic](https://github.com/nihilistau/Position_Is_Arithmetic).

## Non-negotiables (receipts-first)

- **No number without a reproducing command + a gate/commit.** Bit-exact-when-off: every `SP_*`
  overlay is a strict no-op by default — verify it.
- **No silent gate revision** — surface upstream. Honest negatives stay attached (the 32k NIAH MISS,
  the diffusion-judge falsification, the boundary-thesis inert levers).
- **Check the code + commits + `git fetch` before trusting memory.** `lib/shannon-prime-system` can
  diverge from the standalone math-core checkout — check `behind` before building/committing.
- **gated-GREEN is not GREEN-LIVE.** The byte-exact forward and the NIGHTSHIFT curator pass their
  gates behind a flag (null floor when unset); they are not on the served path by default.
- **Served-model misbehavior is almost always ours** (template / decode / sampler / forward /
  prompt), not the weights — verify vs llama.cpp + our PPL first.
