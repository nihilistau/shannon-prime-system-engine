---
type: reference
title: shannon-prime-system-engine — the inference engine + resident daemon + memory agency
description: Repo README for the Shannon-Prime ENGINE — the sp-daemon server, CUDA backend, served Gemma-4-12B chat over the resident persistent-KV cache, memory agency, the L5-cosine + attribute-gate recall/faithfulness stack, NIGHTSHIFT curator, and KAIROS heartbeat. States repo scope vs the other four repos, the served-chat data path, exact env flags + gate names, and build/run one-liners.
tags: [shannon-prime, engine, sp-daemon, gemma4, recall, faithfulness, adr-002, cuda, l5-cosine, attr-gate]
timestamp: 2026-07-01T00:00:00Z
resource: shannon-prime-system-engine/README.md
sp_status: GREEN-LIVE
sp_gate: none
sp_commit: fc2e846
sp_repro: none
---

# shannon-prime-system-engine — the inference engine + resident daemon + memory agency

**This is THE ENGINE of [Shannon-Prime](https://github.com/nihilistau/shannon-prime-lattice).** It is
the `sp-daemon` server, the CUDA backend (`wire_cuda_backend`), the served **Gemma-4-12B** chat over a
resident persistent-KV cache, the model-owned **memory agency** (forget / decide / merge), the
**recall + faithfulness stack** (L5-cosine recall + a deterministic attribute-grounding gate),
NIGHTSHIFT curator, KAIROS heartbeat, Telepathy delegate wiring, and EAGLE/MTP draft. It consumes the
math core ([`shannon-prime-system`](https://github.com/nihilistau/shannon-prime-system), carried here as
the `lib/shannon-prime-system` submodule) through the **frozen L1 C ABI**, and is orchestrated by the
harness ([`shannon-prime-harness`](https://github.com/nihilistau/shannon-prime-harness)).

License: MIT (`LICENSE`). GitHub: `nihilistau/shannon-prime-system-engine`.

> **Read this first for the whole system:** lattice
> [`papers/PPT-LAT-KEYSTONE.md`](https://github.com/nihilistau/shannon-prime-lattice/blob/main/papers/PPT-LAT-KEYSTONE.md).
> This README is *this repo's* scope, data path, flags, and gates. Public site:
> [Position Is Arithmetic](https://nihilistau.github.io/Position_Is_Arithmetic/).

## What Shannon-Prime is

A **fully local, byte-exact, auditable language-model organism**: it serves Google's Gemma-4-12B on a
single **RTX 2060**, through our own inference engine, on an **exact-integer arithmetic substrate**
(`O_K`, dual-prime negacyclic CRT-NTT). It owns a working memory — it learns facts from conversation,
recalls them faithfully, declines when it does not know, and forgets / supersedes / merges on its own
judgement. No cloud, no third-party inference, no telemetry. **Every mechanism is a default-off env
flag = a strict no-op / byte-identical null floor when unset**; every number has a reproducing command
and a gate.

## What this repo controls vs the other four repos

| Repo | Role | This repo talks to it via |
|---|---|---|
| **shannon-prime-system-engine** *(here)* | **THE ENGINE.** `sp-daemon` server + `/v1/chat` loop, accelerator backends, served 12B chat over the resident persistent-KV cache, memory agency, recall/faithfulness stack, NIGHTSHIFT, KAIROS. | — |
| **shannon-prime-system** | The **math core** — decode loop, `O_K` / CRT-NTT, exact-islands, ARM recall router. Frozen. | the **L1 C ABI** (`sp_session_register_*_backend`), carried as `lib/shannon-prime-system` submodule |
| **shannon-prime-harness** | **Orchestration** — the agency scheduler (`run_agency.py`), corpus/curator tooling, eval drivers. | drives the daemon over HTTP + shells the gates |
| **shannon-prime-lattice** | **Docs / canon / STATE** — papers, contracts, ADRs, scoreboard, ledgers. Source of truth for status. | pointers only (no code dependency) |
| **shannon-prime** (public face) | `Position_Is_Arithmetic` — receipts-first public papers + `LEDGER.md`. | citations only |

## Architecture map

```
   shannon-prime-harness            shannon-prime-lattice (docs / canon / STATE)
   run_agency.py, gates,            PPT-LAT-KEYSTONE, VERIFIED-SCOREBOARD,
   curator/eval drivers             ADR-002, FINDINGS-LEDGER, START-HERE
        │ HTTP / shell                        ▲ pointers only
        ▼                                     │
  ┌──────────────────────────────────────────┴───────────────────────────────┐
  │  ENGINE (this repo)  —  tools/sp_daemon/  (Rust, HTTP / SSE / WebSocket)   │
  │  /v1/chat loop (routes.rs) · recall+faithfulness (recall.rs) ·             │
  │  memory agency FORGET/DECIDE/MERGE · NIGHTSHIFT curator · KAIROS heartbeat │
  │  · Telepathy delegate · EAGLE/MTP draft · sampler/tokenizer/sessions      │
  └───────────────────────────────┬───────────────────────────────────────────┘
                                   │  frozen L1 C ABI
                                   │  (sp_session_register_forward_backend,
                                   │   sp_session_register_kvdecode_backend, …)
        ┌──────────────┬──────────┴───────────┬───────────────┐
        ▼              ▼                       ▼               ▼
   CUDA backend    CPU backend            Vulkan backend   Hexagon HVX
  (wire_cuda_       (overlay dispatch      (desktop)        (edge)
   backend:          into math-core
   gemma4 fwd/       decode)
   decode, OK_Q4B
   dp4a, SP_BYTEEXACT)
        │
        └──────────► lib/shannon-prime-system  (MATH CORE, submodule:
                     decode loop, O_K / CRT-NTT, exact-islands, L1 ABI)
```

## The served-chat data path (`/v1/chat`)

The daemon holds a **resident persistent-KV cache** and drives the 12B token-by-token over the L1 ABI.
A turn flows:

```
HTTP POST /v1/chat  (tools/sp_daemon/src/server.rs → routes.rs)
   │
   ├─ apply chat template (tokenizer.rs)
   ├─ PERSISTENT O(1) KV  (SP_PERSIST_KV, default-ON): if the committed cache is a
   │     strict prefix of this turn, extend it instead of re-prefilling — flat VRAM,
   │     O(1) context growth  (routes.rs: kv:: prefill/decode/rewind)
   │
   ├─ DECIDE stage (latent, ADR-002):  choose whether to recall / decline / forget /
   │     supersede — deciders read latent, they do NOT execute:
   │        · SP_RECALL_L5      — L5-cosine paraphrase recall selector (see below)
   │        · SP_RECALL_ATTR_GATE — attribute-grounding gate → zero-inference decline
   │        · SP_FORGET / SP_DECIDE — memory agency writers
   │
   └─ EXECUTE stage (clean text): synthesize the answer in a fresh clean context
         from the DECIDE result — NEVER fusing the latent decision into generation.
```

**Governing law — ADR-002 Decide→Execute spine:** decide in latent, execute in clean text, **never
fuse**; deciders don't execute. (lattice `papers/PPT-LAT-ADR-002-DECIDE-EXECUTE-SPINE.md`.)

## Recall + faithfulness stack (CLOSED end-to-end, 2026-07-01)

The faithfulness axis is closed on the served chat. Honest tiers: **PROVEN / GREEN-LIVE** (gated *and*
on the served path), **PARKED** (built, gated, deliberately not on the hot path), **HONEST-NEGATIVE**.

| Stage | What it does | Tier | Flag(s) | Gate / commit |
|---|---|---|---|---|
| **L5-cosine recall selector** | matches the live query's global-layer-5 embedding against each episode's stored L5 query-key by cosine; on a confident match (τ=0.30) delivers the episode TEXT in-context | **GREEN-LIVE** | `SP_RECALL_L5=1` (`SP_RECALL_L5_TAU`, default 0.30) | `G-L5-RECALL-LIVE` (86.89% paraphrase LIVE) · `d9099cd` |
| **Attribute-grounding gate → zero-inference decline** | closes the zero-prior / private-data hole (SNE crucible: 80% confab, 5% secret-leak on novel entities). If the query's salient words are ABSENT from the fact (`attr_absent_ratio ≥ τ`) **and** the query carries a high-entropy entity token (`query_has_entity_token`), it emits a deterministic symbolic decline and **skips the gemma4 forward entirely** — confabulation/leak becomes mathematically impossible; paraphrase recall untouched | **GREEN-LIVE** | `SP_RECALL_ATTR_GATE=1` (`SP_RECALL_ATTR_TAU`, default 0.5) | `G-SNE-ATTRGATE-ZEROINF` (confab→0, leak→0, recall 100%) · HEAD `fc2e846` |
| **Generative recall/reject judge** | 12B reads candidate memory TEXTS and picks the one that answers the query, or NULL | **PARKED** | `SP_B3_JUDGE` | hard-foreign kill-test: **0 benefit** vs L5-direct+τ (PASSed 15/18) — parked, not on the hot path |

Relevant code: `tools/sp_daemon/src/recall.rs` (`l5_query_embed`, `cos512`, `attr_absent_ratio`,
`query_has_entity_token`), `tools/sp_daemon/src/routes.rs` (the `SP_RECALL_L5` branch + the
`symbolic_decline` synthesis-seam short-circuit).

## Other engine mechanisms (all default-off = byte-identical null floor)

| Mechanism | Tier | Flag | Gate / commit |
|---|---|---|---|
| **Persistent O(1) KV** — resident cache, flat VRAM, O(1) context growth | **PROVEN / default-ON** | `SP_PERSIST_KV` (unset ⇒ on; `=0` forces flat) | `G-PERSIST-KV` |
| **Byte-exact exact-integer forward** — RMSNorm/softmax/GELU/RoPE + attention as exact-integer device kernels = exact arithmetic / cross-machine determinism / **AUDITABILITY, not compression** | **gated-GREEN / default-off** | `SP_BYTEEXACT` | `G-BYTEEXACT-FORWARD-12B` (OFF 4.6665 byte-identical / ON 4.6569 parity / run-to-run bit-identical) · `69c0588` |
| **Model-owned memory agency** — STORE (NIGHTSHIFT) + FORGET + DECIDE/MERGE (supersede a changed fact, or consolidate two complementary facts) | **GREEN-LIVE** | `SP_FORGET`, `SP_DECIDE` | `G-FORGET` / `G-DECIDE` / `G-MERGE` |
| **NIGHTSHIFT offline curator** — 12B model-call secret-extractor → teacher-forced causal-ablation admit (TAU=−8) → conformant MEM-OKF emit | **gated-GREEN / live PENDING** | `SP_NIGHTSHIFT_OFFLINE=1` | `G-NIGHTSHIFT-CURATOR` criteria 1-4 · `6107f3e` |
| **Autonomous W_c recall head** — learned latent selector, (E+1)-way NULL argmax + bounded-mass replay | **gated-GREEN** (superseded on the hot path by L5-cosine) | `SP_B3_WC` | `G-CHAT-B3-WC-DEPLOY` / `-DIV2` · `edc8079` |

The **boundary thesis** runs through all of it: `O_K` wins on **exact arithmetic** (the container);
every *structure-on-content* lever (Möbius, split-prime Dirichlet carriers, entropy-coded Frobenius
codes, T2-Möbius on the real embedding) is **measured-inert and kept as an honest negative**.

## Build

Two distinct toolchains (do **not** mix). The CPU / math-core build uses **clang-cl** (MSVC-ABI —
`lib/shannon-prime-system/core/exact_islands/exact_islands.c` uses `__int128`, which `cl.exe` cannot
compile). Authoritative from-clean chain + every gotcha: lattice
`papers/BUILD-ENV-TOOLCHAIN.md` (gate `G-CLEAN-BUILD`).

| Backend | Toolchain | Build dir |
|---|---|---|
| **CUDA** (canonical) | VS2019 BuildTools + CUDA, ninja, sm_75 (dev RTX 2060) | `build-cuda/` |
| **CPU** (canonical) | clang-cl (MSVC-ABI) — NOT `cl.exe` | `build-cpu/` |

```bat
:: build the served daemon (drives the real 12B end-to-end over the L1 ABI)
cargo build --release --features wire_cuda_backend --target-dir target-wirecuda --bin sp-daemon

:: a CUDA gate
cd build-cuda && ninja test_gemma4_cuda && tests\test_gemma4_cuda.exe

:: the daemon decode gate (32/32 tokens == oracle, VRAM flat O(1))
cargo run --release --features wire_cuda_backend --bin sp_wire_cuda_decode_gate
```

GPU numbers need warmup + a long window + **both clocks pinned**. Git on these repos: **native
PowerShell, not the Linux mount** (the mount CRLF-churns large files + locks).

## Run the live organism

```bat
run_console.bat               :: coherent + byte-exact + O(1) 12B chat -> http://127.0.0.1:3000/
run_console_recall.bat        :: + the autonomous librarian
run_console_nightshift.bat    :: + the offline NIGHTSHIFT curator (SP_NIGHTSHIFT_OFFLINE=1)
```

The faithfulness stack is turned on with `SP_RECALL_L5=1 SP_RECALL_ATTR_GATE=1`. Alongside the daemon,
run the harness agency scheduler for the heartbeat consolidation + maintenance
(`shannon-prime-harness/run_agency.py`). State a fact and it is captured; ask an attribute the fact
does not state and it **declines in microseconds with no forward** instead of confabulating; say
"forget X" and it is dropped; state a contradicting/complementary fact and DECIDE supersedes or MERGEs
it. Full procedure: lattice `papers/PPT-LAT-KEYSTONE.md` §10.

## Key engine surfaces

- **`tools/sp_daemon/`** — the resident daemon (feature `wire_cuda_backend` drives the real 12B):
  - `src/routes.rs` — the `/v1/chat` loop: persistent O(1) KV (`SP_PERSIST_KV`), the `SP_RECALL_L5`
    branch + `symbolic_decline` synthesis-seam short-circuit, the memory agency (`SP_FORGET`,
    `SP_DECIDE`/MERGE), the EOT bias, the `SP_CURRENT_CONVO` consolidation hook.
  - `src/recall.rs` — the recall/faithfulness primitives (`l5_query_embed`, `cos512`,
    `attr_absent_ratio`, `query_has_entity_token`; plus the learned `WcHead`).
  - `src/nightshift_curator.rs` — the offline curator (opens its own kvdecode handle, never touches
    the served cache). `src/kairos.rs` / `kairos_runner.rs` — the heartbeat control-plane.
  - `src/telepathy.rs`, `src/eagle_accept.rs` — delegate wiring + EAGLE/MTP draft-accept.
- **`src/backends/cuda/cuda_forward.cu`** — the gemma4 CUDA forward + decode (per-layer SWA/global
  geometry, shared-KV, AltUp/PL=0, softcap); the **OK_Q4B `k_gemv_q4b_dp4a` kernel**; the CUDA-graph
  decode path; the **`SP_BYTEEXACT` exact-integer islands**; the additive `gemma4_kv_decode_logits`
  (the daemon's token-by-token decode entry) + the persistent-KV ABI.
- **`src/backends/cpu/`** — overlay dispatch into the math-core decode. `vulkan/`, `hexagon/` are
  desktop / edge targets.
- **`tools/sp_transcode/`** — `sp_transcode --st`: the safetensors-direct pipeline, **the ONLY trusted
  gemma4-12B weight path** (the GGUF lane is dead). Writes OK_Q8 / OK_Q4B `.sp-model`.
- **`tools/sp_dsp_smoke/`** — the L2 universal Rust crate: dual-prime Barrett / mod-q matmul / Garner
  CRT / NTT ladder, bit-exact-gated; the 4 islands' exact-integer references in `src/sp_islands_q_ref.rs`.
- **Gates / receipts** — `tests/test_gemma4_cuda.c`, `tests/test_xbar_p1_cuda.c`, `tests/fixtures/`.

## Canon / status pointers (lattice repo — source of truth)

- `papers/PPT-LAT-KEYSTONE.md` — the whole system, current + complete.
- `papers/VERIFIED-SCOREBOARD.md` — box-by-box honest tiers.
- `papers/PPT-LAT-ADR-002-DECIDE-EXECUTE-SPINE.md` — the governing decide→execute law.
- `papers/PPT-LAT-FINDINGS-LEDGER.md` — the findings ledger.
- `START-HERE.md` — orientation. `papers/PPT-LAT-STATE.md` — the PROVEN record.

## Non-negotiables (receipts-first)

- **No number without a reproducing command + a gate/commit.** Bit-exact-when-off: every `SP_*`
  overlay is a strict no-op by default — verify the null floor before claiming a delta.
- **Honest tiers, exact words.** `GREEN-LIVE` = gated AND on the served path by default. `PARKED` /
  `gated-GREEN / default-off` = passes a gate but is NOT on the hot path. `HONEST-NEGATIVE` stays
  attached (the diffusion-judge falsification, the boundary-thesis inert levers, the parked judge).
- **No silent gate revision** — surface upstream.
- **Check the code + commits + `git fetch` before trusting memory.** `lib/shannon-prime-system` can
  diverge from the standalone math-core checkout — check `behind` before building/committing.
- **Served-model misbehavior is almost always ours** (template / decode / sampler / forward / prompt),
  not the weights — verify vs llama.cpp + our PPL first.
