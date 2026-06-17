# CLAUDE.md — shannon-prime-system-engine (the inference engine)

**This is Shannon-Prime's inference engine. The canonical session bootstrap is `D:\F\shannon-prime-repos\shannon-prime-lattice\prompt.md` — read it first** (project, current state, methodology, machine, doc map, operator). This file is the short version + this repo's specifics.

**Repo role:** the inference engine + backends, consuming the math core via the **`lib/shannon-prime-system` submodule**. Key surfaces:
- `src/backends/cuda/cuda_forward.cu` — the gemma4 CUDA forward + decode (per-layer SWA/global geometry, shared-KV, AltUp/PL=0, softcap), the **OK_Q4B `k_gemv_q4b_dp4a` kernel**, the CUDA-graph decode path, and the **`SP_XBAR_*` harness** (P1 KV splice/capture + P2.a `SP_XBAR_EMB` entry injection + rank/score lanes). The XBAR experiments + the 12B B1 artifact runs happen here.
- `src/backends/cpu/` (overlay dispatch into the math-core decode), `vulkan/`, `hexagon/`.
- `tools/sp_transcode/` — **`sp_transcode --st`**: the safetensors-direct pipeline (the ONLY trusted gemma4-12B weight path; GGUF lane is dead, see ledger 06-R8). Writes OK_Q8 / OK_Q4B `.sp-model`.
- `tests/test_gemma4_cuda.c`, `tests/test_xbar_p1_cuda.c`, `tests/bench_gemv_int8.cu`.

**Current edge (2026-06-17):** **XBAR P3 CLOSED end-to-end (P3.0→P3.4)** — P3.3 replay-write is the `SP_REPLAY` seam in `cuda_forward.cu` (inject the episode owner-K/V over prefill rows `[0,NPOS)` at BOTH `gemma4_decode_cuda` prefill stores, graph + velocity; `G-P3-SHARED` 3-leg PASS on 12B + E2B), and P3.4 recall-quality (`G-P3-PPL` +1.38% < 2% via `SP_G4_SCORE` ∘ `SP_REPLAY`, zero new engine code) are both CLOSED GREEN. **GNA "EAR" line CLOSED on physical silicon** — real speech → 12B 7/8, POT GNA-native i16 = 0.877 full recovery, GNA_HW on the Intel GNA 2.0 = 0.877 == emu == FP32; tooling in `tools/audio_port/` (`ov_gna_score.py`/`pot_gna_quantize.py`/`run_gna_hw.bat` + `GNA_HW_BRINGUP.md`). NEXT (lattice-side) = the Memo curator / Ring-3 orchestration tier above P3. The one-shot `gemma4_decode_cuda` stays byte-untouched (null floor); all new work is env-gated.

**Build:** **CUDA host = VS2019 BuildTools + CUDA, `build-cuda/` dir, ninja** (sm_75 on the dev 2060). Canonical **CPU = MinGW gcc 15.2, `build/` dir** (MSVC cannot build CPU). Authoritative doc: `docs/BUILD-ENV.md`. GPU numbers need warmup + long window + **both clocks pinned** (`-lgc` is SM-only; a weight-GEMV is memory-bound).

**Git (binding lesson):** `lib/shannon-prime-system` is a submodule of the same repo as the standalone `shannon-prime-system` checkout, so the two can diverge. `git fetch` + check `behind` before building/committing; commit + push every repo touched per milestone.

**Non-negotiables:** receipts-first (no number without a command); bit-exact / top-1-lossless gates per precision; no silent gate revision (surface upstream); **reference-first** when porting (read llama.cpp / the reference with file:line before coding); check code + commits + `git fetch` before trusting memory; verify Gemini's claims; drive by default; bakes are OS-owned + log-tailed, never poll-watched. Full detail in lattice `prompt.md`.

**Environment & credentials (2026-06-11):** compute lanes, shells/traps, storage law (incl. the models\ SHA-pairing rule) → lattice `ENVIRONMENT.md`. Current state/queue → lattice `SESSION-HANDOFF.md`. Secrets → `archive\notes_and_stuff\creds\claude-credentials.txt` (outside all repos; reference paths, never values).
