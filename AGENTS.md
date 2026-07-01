# AGENTS.md — shannon-prime-system-engine

Agent entry point for the **Shannon-Prime inference engine** (CUDA/CPU/Vulkan/Hexagon backends, `sp_transcode`, the served `sp_daemon`). Human-readable and agent-navigable.

> **Current state (2026-07-01): the FAITHFULNESS AXIS is CLOSED end-to-end on the served 12B chat**, under **ADR-002** (decide in latent, execute in clean text, NEVER fuse). Live recall selector = **L5-cosine** (`SP_RECALL_L5=1`, τ=0.30, 86.89% paraphrase, gate `G-L5-RECALL-LIVE`, engine `d9099cd`); the generative judge is **PARKED** (hard-foreign kill-test: 0 benefit vs L5-direct+τ); the zero-prior/private-data confab+leak hole is **CLOSED** by a deterministic **attribute-grounding gate** with a ZERO-INFERENCE symbolic decline (`SP_RECALL_ATTR_GATE=1` → confab 0 / leak 0 / recall 100%, no gemma4 forward on decline; gate `G-SNE-ATTRGATE-ZEROINF`, engine HEAD `fc2e846`). All flags default-off = byte-identical null floor; turn the closed stack on with `SP_RECALL_L5=1 SP_RECALL_ATTR_GATE=1`. NEXT: **SWARM re-elevated**. Full record in `CLAUDE.md`; status truth in lattice `papers/VERIFIED-SCOREBOARD.md` + `PPT-LAT-KEYSTONE.md` + `PPT-LAT-ADR-002-DECIDE-EXECUTE-SPINE.md`.

## Read order (do this before touching anything)

1. **`README.md`** (this repo) — the engine surfaces, the honest-tier capability map, build + run.
2. **`CLAUDE.md`** (this repo) — the short specifics + current edge.
3. **`../shannon-prime-lattice/prompt.md`** — the canonical session bootstrap (project, methodology, machine, operator).
4. **`../shannon-prime-lattice/papers/PPT-LAT-STATE.md`** — the PROVEN record. Trust it; re-prove only with concrete cause.
5. **`../shannon-prime-lattice/papers/STATUS-MAP-2026-06-21.md`** — box-by-box honest tiers (GREEN-LIVE / gated-GREEN / DESIGN).
6. **`HISTORY.md`** (this repo) — the hashed tiered commit log. The git short-hash IS the content address; `git show <hash>` is the tier-2 store.
7. **The relevant contract** — `CONTRACT-CHAT-FULLSTACK`, `CONTRACT-BYTEEXACT-forward`, `CONTRACT-NIGHTSHIFT-CURATOR` (all in lattice `papers/`).

## Anti-rebuild pre-flight (BINDING — run before building ANY subsystem)

This project has rebuilt the same subsystems 20+ times. A new file for a capability that already exists is a **defect**. Before writing code:

```bash
python ../shannon-prime-lattice/tools/okf_mem.py lookup --root ../shannon-prime-lattice/memory-okf <keyword>
```

…then `grep` the tree for the same concept. The content-addressed MEM-OKF store (LUT → summary → full) records "X already exists — don't rebuild" facts. At session end, bank durable such facts via `okf_mem.py add`.

## Build

| Backend | Toolchain | Build dir |
|---|---|---|
| CUDA (canonical) | VS2019 BuildTools + CUDA 12.4, ninja, sm_75 | `build-cuda/` |
| CPU (canonical) | MinGW gcc 15.2, ninja | `build/` |

```bat
:: CUDA gate
cd build-cuda && ninja test_gemma4_cuda && tests\test_gemma4_cuda.exe
:: daemon decode gate
cargo run --release --features wire_cuda_backend --bin sp_wire_cuda_decode_gate
:: serve the 12B chat
run_console_recall.bat   :: -> http://127.0.0.1:3000/
```

MSVC cannot build the CPU tree. GPU numbers need warmup + a long window + both clocks pinned.

## Non-negotiables (receipts-first / honest tiers)

- **No number without a reproducing command + a gate or commit.** Every `SP_*` overlay is **bit-exact-when-off** — verify the null floor before claiming the delta.
- **Honest tiers, exact words.** `GREEN-LIVE` = gated AND on the served path by default (the L5-cosine recall selector + the attribute-grounding zero-inference decline are GREEN-LIVE). `PARKED` / `gated-GREEN / default-off` = passes its gate behind a flag but is NOT on the hot path — the **generative judge is PARKED** (kill-test: 0 benefit vs L5-direct+τ, kept as an honest negative, not deleted); the byte-exact forward and the NIGHTSHIFT curator are **gated-GREEN, not GREEN-LIVE** (curator live criterion 5 **PENDING**). The native diffusion judge is **UNPROVEN** (its 95.6% is the external llama.cpp oracle's, not ours; our native single-forward was falsified ~25%).
- **No silent gate revision** — surface upstream. **Honest negatives stay attached** (the 32k NIAH MISS, the diffusion falsification, the boundary-thesis inert levers).
- **Check the code + commits + `git status`/`git fetch` BEFORE trusting memory or a summary.**
- **Drive by default.** Make the obvious call; surface only genuine forks.
- **OKF frontmatter** on every knowledge `.md` you create/touch (`type/title/description/tags/timestamp/resource` + `sp_status/sp_gate/sp_commit/sp_repro`); validate with lattice `python tools/okf_validate.py <bundle>`.

## Submodule caution (binding lesson)

`lib/shannon-prime-system` is carried as a submodule of the **same upstream** as the standalone `shannon-prime-system` checkout, so the two can diverge and the submodule pin can sit **behind** `origin/main`. **`git fetch` + check `git rev-list --count HEAD..origin/main` before building/committing.** Do not touch the submodule when working on this repo's own files. Commit + push every repo you touch, per milestone.

## Where to look

- Served daemon `/v1/chat` loop: `tools/sp_daemon/src/routes.rs` (the `SP_RECALL_L5` branch + the `symbolic_decline` synthesis-seam short-circuit; `SP_PERSIST_KV`; memory agency `SP_FORGET`/`SP_DECIDE`).
- Recall + faithfulness primitives: `tools/sp_daemon/src/recall.rs` (`l5_query_embed`, `cos512`, `attr_absent_ratio`, `query_has_entity_token`; plus the learned `WcHead`).
- Curator / heartbeat: `tools/sp_daemon/src/{nightshift_curator.rs, kairos.rs, kairos_runner.rs}`; delegate wiring: `telepathy.rs`, `eagle_accept.rs`.
- CUDA forward/decode + harnesses: `src/backends/cuda/cuda_forward.cu` (`SP_BYTEEXACT` islands, `gemma4_kv_decode_logits`, persistent-KV ABI).
- Weight pipeline: `tools/sp_transcode/sp_transcode.c` (`--st` safetensors-direct — the only trusted weight path).
- Gates / receipts: `tests/test_gemma4_cuda.c`, `tests/fixtures/`.
