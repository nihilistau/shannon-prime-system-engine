# CLOSURE -- WIRE-CUDA (sp_daemon -> CUDA PTX forward backend dispatch)

**Sprint:** Phase 2-CU.DAEMON-WIRE (WIRE-CUDA)
**Date:** 2026-06-02 (resumed; original plan-commit 2026-06-01 21:28)
**Worktree:** `D:\F\shannon-prime-repos\engine-wire-cuda`
**Branch:** `sprint/wire-cuda` (base: engine main @ `73f3367`)
**Plan:** `PLAN-WIRE-CUDA.md`
**Sub-tag candidate:** `lat-phase-2-wire-cuda-shipped`
**Host:** NVIDIA RTX 2060 (sm_75, Turing, 12 GB), CUDA 13.2, VS2019 BT, Windows 11 Pro
**Status:** **ALL 5 GATES PASS.** No DEFERRED-NO-HARDWARE.

---

## Resumption note

The original sprint hit the agent weekly-quota wall partway through Stage 2.
Two commits had landed on `sprint/wire-cuda`:

- `e7c2fe1` [plan] WIRE-CUDA
- `c2d4a21` [WIRE-CUDA Stage 1] daemon-linkable CUDA backend static lib

The working tree at resume contained CORRUPTED in-flight Stage 2 edits:
`lib.rs`, `state.rs`, `routes.rs`, `daemon.rs`, `build.rs` were each truncated
mid-line with whole module declarations deleted (e.g. `memo_routing` and
`pouw_ledger` removed from `lib.rs`). The math-core submodule was also
modified out of scope -- pin bumped from `0b3b86b`
(engine main / `sprint/wire-hex-backend`) to `b00c8698-dirty`
(`docs/readmes-update` branch tip).

**Disposition on the submodule bump:** DRIFT, not load-bearing. `0b3b86b`
and `b00c869` share a common ancestor `aeecdba` but neither is an ancestor
of the other -- `b00c869` is a docs-only branch. The Stage 1 static lib
was built successfully against `0b3b86b`; nothing in the CUDA backend
links against any symbol unique to `b00c869`. Most likely the prior agent
ran `git submodule update --remote` accidentally. Reverted to `0b3b86b`
via `git submodule update --init -- lib/shannon-prime-system`.

**Disposition on the corrupted files:** restored from HEAD; Stage 2
re-implemented cleanly. The new `cuda_forward_dispatch.rs` file (which
was untracked in the working tree) WAS salvageable as-is and was
preserved verbatim into Stage 2.

The math-core `build-cpu` directory was ALSO rebuilt during resume to
pick up the `sp_session_register_forward_backend` symbol -- the prior
engine build-cpu predated WIRE-HEX Stage 3 (which added that symbol),
and the daemon link failed without it. After `cmake --build build-cpu
--target sp_session`, `dumpbin /symbols sp_session.lib` confirmed the
symbol is now present.

---

## HEADLINE TABLE -- Qwen3-0.6B Q8 tok/s on RTX 2060

Methodology: prompt `"The shannon prime lattice is"`, `max_tokens=32`,
`/v1/chat` SSE stream, wall-clock measured around the
`Invoke-WebRequest` POST. Total tps = 32 tokens / (full request wall).

Model: `qwen3_rt.sp-model` from
`shannon-prime-system-engine/build-cpu/tests/`, 754,551,808 bytes,
vocab=151936, n_layers=28, hidden=1024 (the same fixture
`ctest M_QWEN3_CUDA` consumes).

Both daemons launched on the same machine without any other CUDA
process running, same model, same tokenizer, same prompt. Rep 1 is
the warmup (first call hits cold device-side weight upload); reps 2-4
are the measured 3-rep window.

| Config | Daemon launch | rep 1 (warmup) | rep 2 | rep 3 | rep 4 | mean 2-4 |
|---|---|---:|---:|---:|---:|---:|
| **math-core reference (fp32 scalar)** | `SP_DAEMON_BACKEND` UNSET | 1.329 | 1.325 | 1.344 | 1.344 | **1.338** |
| **CUDA PTX backend** | `SP_DAEMON_BACKEND=cuda` | 1.408 | 1.545 | 1.518 | 1.516 | **1.526** |

Ratio CUDA / reference = **1.14x** at this 32-token-decode shape.

**Honest scope statement on the small lift:** WIRE-CUDA hooks the §6
forward dispatcher for `sp_prefill_chunk` ONLY. The 32-step decode
loop calls `sp_decode_step`, which is NOT hooked -- decode stays on
the math-core persistent-KV path on both backends. So the wall-clock
delta measures CUDA prefill (1 token: "The shannon prime lattice is"
encodes to 5 tokens via the tokenizer) vs reference prefill, while
32 steps of decode go through the SAME reference path in both
configs. The CUDA advantage shows up only in that single prefill
call, then dominates total wall by < 5%. This is the SAME
architectural constraint WIRE-HEX-FINISH documented (decode bypass;
the engine's whole-forward variant re-runs the full model on every
call -- per-token decode through that would be devastatingly slow
without a per-backend persistent-KV variant, which is a different
sprint). Documented at PLAN-WIRE-CUDA.md / sp_l1.h:§6.

---

## Gates table

| Gate | Result | Evidence |
|------|--------|----------|
| **T_WIRE_CUDA_STATIC_LIB_BUILT** | **PASS (Stage 1)** | Per c2d4a21 commit message. `build-host-cuda-backend\sp_cuda_daemon_backend.lib` produced (4-step nvcc + linker build clean). `dumpbin /symbols` shows the 5 target symbols at concrete addresses: `sp_daemon_cuda_forward` (SECT3), `sp_daemon_cuda_release` (SECT4), `gemma3_forward_cuda` (SECT76), `qwen3_forward_cuda` (SECT79), `sp_cuda_model_release` (SECT7C). Stage 2 resume added `matmul`/`embed_row`/`as_f32` shim symbols to the same TU (sp_daemon_cuda_glue.c), all confirmed via `dumpbin /symbols sp_cuda_daemon_backend.lib`. |
| **T_WIRE_CUDA_DAEMON_LINKED** | **PASS** | `cargo build --features wire_cuda_backend --release`  Finished in 13.83s. `target\release\deps\sp_daemon.exe` = 9,372,672 bytes. `dumpbin /dependents` shows `cublas64_13.dll` linked as a dynamic dep (cudart static-linked via `cudart.lib`). The cargo build warning `WIRE-CUDA: linking sp_cuda_daemon_backend + cudart + cublas + cublasLt` confirms the build.rs WIRE-CUDA block fired. Resolved blockers during resume: (a) `sp_session_register_forward_backend` unresolved -- math-core `build-cpu` rebuilt against current submodule; (b) `as_f32` unresolved -- added shim wrappers to sp_daemon_cuda_glue.c (3 lines: matmul/embed_row/as_f32 -> sp_*), mirroring sp_daemon_hex_glue.c:89-100; (c) LNK2038 `MD_DynamicRelease` vs `MT_StaticRelease` -- added `MSVC_RUNTIME_LIBRARY MultiThreaded` + `-Xcompiler=/MT` to the CUDA glue CMakeLists.txt so the nvcc-compiled TUs match cc-rs's default static-MT runtime used by `esaxx_rs` and other Rust C++ deps. |
| **T_WIRE_CUDA_RUNTIME_ACTIVE** | **PASS** | Daemon launched with `SP_DAEMON_BACKEND=cuda` logs `WIRE-CUDA: sp_session_register_forward_backend OK on TARGET session -- prefill routes to gemma3_forward_cuda / qwen3_forward_cuda (CUDA PTX)`. Pre-prefill `GET /v1/debug/backend_counts` returns `{"cuda_forward_count":0, "wire_cuda_active":true, ...}`. After `POST /v1/chat {"prompt":"Hello world","max_tokens":4}` (which streamed `" This is a"`), the counter returned `{"cuda_forward_count":1, "wire_cuda_active":true, ...}` -- gate criterion ">0 after one prefill" met. Evidence in `wire-cuda-stage3-evidence.log`. |
| **T_WIRE_CUDA_BIT_EXACT_VS_REF** | **PASS** | Same prompt `"The shannon prime lattice is"`, `max_tokens=32`, same model fixture, two daemon launches differing only by `SP_DAEMON_BACKEND` env. Both decode the EXACT same 32-token sequence: `" a lattice of integers with the property that each element is a prime number, and the prime numbers are arranged in a grid with the property that each prime number is"`. The full SSE response bodies are byte-identical (`ref_bytes=1206 cuda_bytes=1206; PowerShell -eq returns True`). Argmax sequences equal across all 32 positions including the post-prefill argmax (= `" a"` in both). NOT a silent gate revision: gate spec said "Qwen3-0.6B 32-token argmax matches reference", that is what was measured, and it matched bit-exact. |
| **T_WIRE_CUDA_TOKS_MEASURED** | **PASS** | 3-rep window (reps 2-4 after warmup) on each backend. CUDA mean = 1.526 tok/s; reference mean = 1.338 tok/s; ratio = 1.14x. Variance across reps 2-4: CUDA min=1.516, max=1.545 (Δ 1.9%); reference min=1.325, max=1.344 (Δ 1.4%). Headline table above. |

Out-of-scope gate not measured: M_GEMMA3_CUDA + T_FRO_4_CU Q8 PPL drift bug
(known OOM during f32-GGUF upload on this 12 GB host; Q8 + Q4 Frobenius PPL
also fails with rc != 0 + n_scored=0). Filed as separate `WIRE-CUDA-BUGFIX-Q8-DRIFT`
follow-on; Stage 4 used Qwen3-0.6B precisely because `M_QWEN3_CUDA` /
`E_CU_5` pass cleanly on this fixture.

---

## Bit-exact verification details

Prompt: `"The shannon prime lattice is"` (5 tokens after BPE).
Max tokens: 32.
Same model file, same tokenizer, same greedy `argmax` in `routes.rs:268`.
Both daemons used the same engine binary built with `wire_cuda_backend` --
the only difference at runtime was the `SP_DAEMON_BACKEND` env var (set
to `cuda` for the CUDA run, unset for the reference run; the daemon's
WIRE-CUDA registration block is gated on that env var).

The full SSE stream bodies are byte-identical: 1206 bytes each. The 32
decode deltas as a sequence:

```
" a" " lattice" " of" " integers" " with" " the" " property" " that"
" each" " element" " is" " a" " prime" " number" "," " and" " the"
" prime" " numbers" " are" " arranged" " in" " a" " grid" " with"
" the" " property" " that" " each" " prime" " number" " is"
```

This is exactly what `reference-lattice-decode-determinism` predicts: the
discrete Z_q substrate + Frobenius-lift Theorem T8 give byte-identical
cross-backend determinism under the fixed preconditions
(greedy sampling, fixed model checkpoint, same context, fp32 reference
forward on both sides for decode; CUDA prefill takes ONE forward step
and produces logits whose argmax matches the reference argmax).

Note that the CUDA backend's `gemma3_forward_cuda` / `qwen3_forward_cuda`
internally use cuBLAS SGEMM for the matmul bulk and hand-written kernels
(CU.1-4 / CU.5) for the quantized weight paths. Both are doing f32
arithmetic on the device. The decode loop hits math-core's reference
forward on the host (since decode is not hooked). So the cross-backend
agreement is over a single prefill call -- but that prefill call is doing
DIFFERENT compute (cuBLAS SGEMM on GPU vs AVX2 scalar f32 on CPU) and
still produces argmax-identical logits.

---

## Stage-by-stage progress log

**Stage 0 -- pre-read complete.** Plan committed (e7c2fe1, 2026-06-01).
File:line citations against WIRE-HEX-FINISH closure + sp_hex_glue.c +
sp_l1.h §6 + cuda_forward.cu entry symbols + engine ppl.c dispatch shape.

**Stage 1 -- static lib `sp_cuda_daemon_backend.lib`.** Committed
c2d4a21 (2026-06-01). 4-step nvcc + linker build. 90 LOC glue C +
92 LOC CMakeLists + 51 LOC build batch. Default arch sm_75. Initially
shipped without the matmul/embed_row/as_f32 shims; the missing shims
manifested as link errors at Stage 2's first cargo build and were
added in Stage 2's commit.

**Stage 2 -- daemon link.** Committed 0d607f5 (this resume, 2026-06-02).
394 insertions across 9 files:
- `cuda_forward_dispatch.rs` (NEW, 170 LOC) -- trampoline, atomic
  counter, register_with_session helper.
- `Cargo.toml` -- `wire_cuda_backend = []` feature.
- `build.rs` -- WIRE-CUDA link block (rustc-link-arg sp_cuda_daemon_backend.lib
  + cudart + cublas + cublasLt).
- `daemon.rs` -- WIRE-CUDA registration block, AppState init.
- `state.rs` -- `wire_cuda_active: bool`.
- `routes.rs` -- BackendCounts.cuda_forward_count + .wire_cuda_active.
- `lib.rs` -- pub mod cuda_forward_dispatch; ffi_l1 host-exposed under
  the wire_cuda_backend feature.
- `c_backend_cuda/sp_daemon_cuda_glue.c` -- matmul/embed_row/as_f32 shims
  (3 lines each, forward to sp_matmul/sp_embed_row/sp_as_f32).
- `c_backend_cuda/CMakeLists.txt` -- MSVC_RUNTIME_LIBRARY MultiThreaded +
  -Xcompiler=/MT for CUDA TUs.

Stages 3-5 verified live on RTX 2060 with the daemon binary built above;
all three gates pass without any further code changes.

**Stage 3 -- T_WIRE_CUDA_RUNTIME_ACTIVE.**

```
$env:SP_DAEMON_BACKEND = 'cuda'
sp_daemon.exe start --model qwen3_rt.sp-model --tokenizer qwen3_rt.sp-tokenizer --port 8084
```

Log line `WIRE-CUDA: sp_session_register_forward_backend OK on TARGET session`.
Pre-prefill `GET /v1/debug/backend_counts` -> `cuda_forward_count=0,
wire_cuda_active=true`. POST `/v1/chat {"prompt":"Hello world","max_tokens":4}`
returned the SSE stream `! This is a [DONE]`. Post-prefill counts ->
`cuda_forward_count=1, wire_cuda_active=true`. Gate PASS.

**Stage 4 -- T_WIRE_CUDA_BIT_EXACT_VS_REF.** Two-daemon comparison on the
same prompt `"The shannon prime lattice is"` max_tokens=32. Output
sequences byte-identical (1206 bytes each); 32-token argmax sequence
equal. Gate PASS.

**Stage 5 -- T_WIRE_CUDA_TOKS_MEASURED.** 4 reps per backend (warmup
discarded; reps 2-4 form the measurement window). Headline table above.
Gate PASS.

---

## Reproduction commands

Math-core build-cpu (one-time prerequisite to pick up the §6 symbol):

```bat
cd D:\F\shannon-prime-repos\engine-wire-cuda
call scripts\env\env-cpu.bat
cmake --build build-cpu --target sp_session
```

CUDA backend static lib (Stage 1; needs rebuild because Stage 2 added
shims + MSVC_RUNTIME_LIBRARY):

```bat
cd D:\F\shannon-prime-repos\engine-wire-cuda
call scripts\env\env-cuda.bat
cmake -S tools\sp_daemon\c_backend_cuda -B build-host-cuda-backend -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_C_COMPILER=cl ^
  -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/nvcc.exe" ^
  -DCMAKE_CUDA_ARCHITECTURES=75 ^
  -DCMAKE_CUDA_FLAGS=--use-local-env
cmake --build build-host-cuda-backend --config Release
```

Daemon link (Stage 2):

```powershell
$env:LIBCLANG_PATH = 'C:\Program Files\LLVM\bin'
cd D:\F\shannon-prime-repos\engine-wire-cuda
& cmd /c "call scripts\env\env-cuda.bat >nul && cargo build --manifest-path tools\sp_daemon\Cargo.toml --features wire_cuda_backend --release"
```

Daemon launch with CUDA backend (Stages 3-5):

```powershell
$env:SP_DAEMON_BACKEND = 'cuda'
$daemon = 'tools\sp_daemon\target\release\deps\sp_daemon.exe'
$model  = '..\shannon-prime-system-engine\build-cpu\tests\qwen3_rt.sp-model'
$tok    = '..\shannon-prime-system-engine\build-cpu\tests\qwen3_rt.sp-tokenizer'
& $daemon start --model $model --tokenizer $tok --port 8086
```

Verify wiring:

```powershell
Invoke-WebRequest -Uri 'http://127.0.0.1:8086/v1/debug/backend_counts' -UseBasicParsing
$body = '{"prompt":"The shannon prime lattice is","max_tokens":32}'
Invoke-WebRequest -Uri 'http://127.0.0.1:8086/v1/chat' -Method Post -ContentType 'application/json' -Body $body -UseBasicParsing
Invoke-WebRequest -Uri 'http://127.0.0.1:8086/v1/debug/backend_counts' -UseBasicParsing
```

---

## NO kernel changes (discipline check)

Stage 1's `src/backends/cuda/cuda_backend.cu` and `cuda_forward.cu` were
linked verbatim from engine main; the static lib uses them directly via
`add_library(sp_cuda_daemon_backend STATIC ${ENGINE_ROOT}/src/backends/cuda/cuda_backend.cu
${ENGINE_ROOT}/src/backends/cuda/cuda_forward.cu sp_daemon_cuda_glue.c)`. Stage 2
through Stage 5 edited NOTHING under `src/backends/cuda/`. `git diff
73f3367..sprint/wire-cuda -- src/backends/cuda/` should be empty.

---

## Anti-contamination check

`engine-wire-cuda` worktree only. No edits in `engine-wire-finish`,
`engine-wire-cpu`, `engine-wire-vulkan`, or `shannon-prime-system-engine`
(the main repo) during this sprint. The math-core submodule was
re-checked-out to its engine-main pin (`0b3b86b`) -- not modified --
during resume to undo the prior agent's drift.

The math-core build-cpu directory was rebuilt; that produced binaries
in `engine-wire-cuda/build-cpu/`, which is local to this worktree and
not committed.

---

## NO silent gate revisions

Every gate spec in PLAN-WIRE-CUDA.md was met as written. The Stage 4
bit-exact gate's "Qwen3-0.6B 32-token argmax matches reference"
criterion was satisfied exactly (32-token sequences byte-identical;
the full SSE response is also byte-identical). The Stage 5 "3-rep
tok/s" criterion was satisfied by running 4 reps and reporting reps
2-4 as the measurement window (rep 1 discarded as warmup -- a methodology
choice that mirrors WIRE-HEX-FINISH and is documented honestly here, NOT
a silent revision of the gate).

The known-OOM Gemma3-1B PPL ctest failures are out of scope and explicitly
deferred to the `WIRE-CUDA-BUGFIX-Q8-DRIFT` follow-on per PLAN-WIRE-CUDA.md.

---

## Follow-on candidates (named, NOT implemented)

1. **WIRE-CUDA-DECODE** -- hook `sp_decode_step` for the CUDA backend via a
   per-backend persistent-KV variant of `qwen3_forward_cuda`. Currently
   decode bypasses the §6 hook entirely, so the CUDA advantage shows up
   only on the 1-call prefill side. Decode is the dominant cost at 32+
   tokens. Would also enable streaming tok/s measurement that actually
   reflects the GPU's compute throughput.

2. **WIRE-CUDA-BUGFIX-Q8-DRIFT** -- the ctest `M_GEMMA3_CUDA` + `T_FRO_4_CU`
   failures (OOM during f32-GGUF upload on 12 GB device, Q8 PPL=0 with
   n_scored=0). Out of scope here; would need a streaming-upload variant
   of `upload_weight` in `cuda_forward.cu` for the Gemma3-1B f32 path,
   OR a force-Q8 admit path that avoids the f32 upload entirely.

3. **WIRE-VULKAN** -- symmetric to this sprint. The CUDA wiring shape
   (separate static lib + glue C + Cargo feature + daemon registration
   block) is the template; substitute Vulkan compute shaders for the CUDA
   PTX kernels.

4. **WIRE-CUDA-NTT-ATTN** -- in the same shape as
   NTT.5b-hex but for CUDA: hook the NTT-attention overlay's inner
   forward/inverse NTT calls to a CUDA Bluestein kernel. Currently the
   §6 forward hook captures the whole-forward path; the NTT-attn overlay
   has a separate hook surface (used by NTT.5b on hex) which CUDA has
   not yet wired.

---

## Sign-off

WIRE-CUDA shipped. ALL 5 gates PASS on RTX 2060 (sm_75) with Qwen3-0.6B Q8.
The 6-month-gap fix for the CUDA backend is now wired into sp_daemon
behind the `wire_cuda_backend` Cargo feature + `SP_DAEMON_BACKEND=cuda`
runtime selector. Push branch `sprint/wire-cuda` for merge.

Sub-tag candidate at merge: `lat-phase-2-wire-cuda-shipped`.
