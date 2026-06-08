# CLAUDE.md — shannon-prime-system-engine (the inference engine)

**This is Shannon-Prime's inference engine. The canonical session bootstrap is `D:\F\shannon-prime-repos\shannon-prime-lattice\prompt.md` — read it first** (project, current state, methodology, machine, doc map, operator). This file is the short version + this repo's specifics.

**Repo role:** the inference engine + backends, consuming the math core via the **`lib/shannon-prime-system` submodule**. Key surfaces:
- `src/backends/cuda/cuda_forward.cu` — the gemma4 CUDA forward + decode (per-layer SWA/global geometry, shared-KV, AltUp/PL=0, softcap), the **OK_Q4B `k_gemv_q4b_dp4a` kernel**, the CUDA-graph decode path, and the **`SP_XBAR_*` harness** (P1 KV splice/capture + P2.a `SP_XBAR_EMB` entry injection + rank/score lanes). The XBAR experiments + the 12B B1 artifact runs happen here.
- `src/backends/cpu/` (overlay dispatch into the math-core decode), `vulkan/`, `hexagon/`.
- `tools/sp_transcode/` — **`sp_transcode --st`**: the safetensors-direct pipeline (the ONLY trusted gemma4-12B weight path; GGUF lane is dead, see ledger 06-R8). Writes OK_Q8 / OK_Q4B `.sp-model`.
- `tests/test_gemma4_cuda.c`, `tests/test_xbar_p1_cuda.c`, `tests/bench_gemv_int8.cu`.

**Build:** **CUDA host = VS2019 BuildTools + CUDA, `build-cuda/` dir, ninja** (sm_75 on the dev 2060). Canonical **CPU = MinGW gcc 15.2, `build/` dir** (MSVC cannot build CPU). Authoritative doc: `docs/BUILD-ENV.md`. GPU numbers need warmup + long window + **both clocks pinned** (`-lgc` is SM-only; a weight-GEMV is memory-bound).

**Git (binding lesson):** `lib/shannon-prime-system` is a submodule of the same repo as the standalone `shannon-prime-system` checkout, so the two can diverge. `git fetch` + check `behind` before building/committing; commit + push every repo touched per milestone.

**Non-negotiables:** receipts-first (no number without a command); bit-exact / top-1-lossless gates per precision; no silent gate revision (surface upstream); **reference-first** when porting (read llama.cpp / the reference with file:line before coding); check code + commits + `git fetch` before trusting memory; verify Gemini's claims; drive by default; bakes are OS-owned + log-tailed, never poll-watched. Full detail in lattice `prompt.md`.
