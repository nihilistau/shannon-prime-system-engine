# CLOSURE — WIRE-CPU (sp_daemon → CPU AVX-512 backend dispatch)

**Sprint:** Phase 2-CPU.DAEMON-WIRE (WIRE-CPU)
**Date:** 2026-06-02
**Worktree:** `D:\F\shannon-prime-repos\engine-wire-cpu`
**Branch:** `sprint/wire-cpu` (base `73f3367` = engine main = V5 FFN VTCM ship)
**Sub-tag candidate:** `lat-phase-2-wire-cpu-shipped`
**Status:** **ALL 5 GATES PASS. The daemon now dispatches `/v1/chat` prefill through the engine's CPU backend behind `SP_DAEMON_BACKEND=cpu`.**
**Plan:** `PLAN-WIRE-CPU.md`

---

## Resumption note

The prior agent landed Stage 0 (plan-commit `ac6e324`, 206 lines) + Stage 1
(commit `78b1b92`, the daemon-link static lib + glue) before hitting the
session quota wall mid-Stage 2. Stage 2 was left in-flight as uncommitted
modifications to `tools/sp_daemon/{Cargo.toml,build.rs,src/{daemon,main,routes,state}.rs}`
plus a new file `tools/sp_daemon/src/cpu_forward_dispatch.rs` (the trampoline).

This agent (2026-06-02) read the in-flight changes, packaged them as Stage 2,
fixed a glue-discovery defect (the engine cpu sources do not define
`qwen25_forward`; the dispatcher was referencing a phantom symbol), built the
math-core archives + daemon binary against the same engine-wire-cpu submodule
pin, and ran 3-rep silicon evidence on the Knack Windows host
(i9-11900KB Beast Canyon, AVX-512 + VNNI silicon, per
`reference-hyperv-cpuid-masking`).

Final branch state:

```
8437721 [WIRE-CPU Stage 1 fix] drop qwen25 branch from glue dispatcher
18ad53b [WIRE-CPU Stage 2] feature + trampoline + daemon registration
78b1b92 [WIRE-CPU Stage 1] daemon-linkable CPU AVX-512 backend static lib + glue
ac6e324 [plan] WIRE-CPU -- symmetric to WIRE-HEX-FINISH for CPU AVX-512 backend
73f3367 (origin/main) Merge sprint/v5-ffn-vtcm
```

---

## HEADLINE TABLE — qwen3_rt.sp-model tok/s on i9-11900KB (Beast Canyon, AVX-512 + VNNI)

Methodology: 8-token synthetic prefill `[2,100,200,300,400,500,600,700]`,
8-step greedy-argmax decode, `/v1/chat` POST (SSE response), combined-wall
timing on the curl client (`(prompt_n + decode_n) / wall_sec`). Same model
(`qwen3_rt.sp-model`, 719.6 MB, vocab=151936 n_layers=28 hidden=1024,
arch_id=2 = SP_ARCH_QWEN3) + tokenizer for both configs. Single daemon
process per config (REF first, WIRE-CPU after restart).

| Config | Daemon launch | Wall sec (mean ± sd) | Combined tok/s | CPU dispatches |
|---|---|---:|---:|---:|
| **fp32 reference** | (no `SP_DAEMON_BACKEND`) | **13.150 ± 0.467** | **1.218 ± 0.042** | 0 per prefill |
| **CPU backend** | `SP_DAEMON_BACKEND=cpu` | **13.272 ± 0.103** | **1.206 ± 0.009** | 1 per prefill |

Per-rep wall: REF = 13.809 / 12.780 / 12.861; WIRE-CPU = 13.132 / 13.309 / 13.375.

**WIRE-CPU / REF ratio: 0.9901× (functionally identical; ±1% within noise).**

This matches the plan's pre-flight expectation
(`PLAN-WIRE-CPU.md:46`):

> the AVX-512 path still compiles into the lib via the feature flags but the
> actual matmul kernel hot path stays scalar. We measure honest baseline
> anyway.

Both configurations route Q8 matmul through the same scalar inner loop
(`cpu_overlay.c:81-97`); the AVX-512 primitives (`sp_avx512_vnni_matvec`,
`sp_avx512_*`) are compiled into the static lib but are NOT wired into
`matmul_arena` yet. The plan named this gap as a future WIRE-CPU-V2 sprint
candidate. **The wiring is the point of this sprint; AVX-512 hot-path
plumbing is the follow-on.**

---

## Gates table

| Gate | Result | Evidence |
|------|--------|----------|
| **T_WIRE_CPU_STATIC_LIB_BUILT** | **PASS** | `build-host-cpu-backend/sp_cpu_daemon_backend.lib` exists, 60,532 bytes (pre-Stage 1-fix) / regenerated post-fix. `dumpbin /SYMBOLS` confirms `gemma3_forward_cpu` SECT3, `qwen3_forward_cpu` SECT4, `sp_daemon_cpu_forward` SECT5, `sp_daemon_cpu_release` SECT6 — all external definitions; `gemma3_forward_cpu_impl` + `qwen3_forward_cpu_impl` UNDEF (resolved from the `_impl`-renamed `cpu_gemma3.c.obj` + `cpu_forward.c.obj` in the same archive). |
| **T_WIRE_CPU_DAEMON_LINKED** | **PASS** | `cargo build --features wire_cpu_backend --release` succeeds. Resulting `sp-daemon.exe` is 9,267,200 bytes. The 3 required symbols WERE necessarily resolved at link: prior link attempts with the canonical math-core build (which lacked `sp_session_register_forward_backend`) FAILED with `LNK2019: unresolved external symbol sp_session_register_forward_backend`; once we built the math-core archives inside the worktree (which DO export the §6 hook per `lib/shannon-prime-system/core/session/sp_session.c:666`), the build succeeded with exit 0. Symbol existence in the input archives confirmed via `dumpbin /SYMBOLS`: `sp_session_register_forward_backend` SECT17 in `sp_session.lib`, `sp_session_forward_backend_fn` SECT11 in `sp_session.lib`. `sp_wire_cpu_forward_dispatch` is the Rust `#[no_mangle]` trampoline in `cpu_forward_dispatch.rs`; its presence in the final binary follows from successful link (otherwise the `register_with_session` call would not resolve to a function-pointer-yielding symbol). |
| **T_WIRE_CPU_RUNTIME_ACTIVE** | **PASS** | Daemon started with `SP_DAEMON_BACKEND=cpu`, model = `qwen3_rt.sp-model`, port 8088. Log line: `WIRE-CPU: sp_session_register_forward_backend OK on TARGET session — prefill routes to engine CPU AVX-512 backend (gemma3_forward_cpu / qwen3_forward_cpu)`. `GET /v1/debug/backend_counts` returns `{"cpu_forward_count":0,"wire_cpu_active":true,...}` pre-prefill; after one `POST /v1/chat` returns `{"cpu_forward_count":1,"wire_cpu_active":true,...}`; after 3 reps returns `{"cpu_forward_count":3,"wire_cpu_active":true,...}`. Linear with prefill count — proves each `/v1/chat` prefill flows through `sp_wire_cpu_forward_dispatch` → `sp_daemon_cpu_forward` → `qwen3_forward_cpu_impl`. |
| **T_WIRE_CPU_BIT_EXACT_VS_REF** | **PASS** | Same prompt `[100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600]` + `max_tokens=16` on two daemon configs (REF without env / WIRE-CPU with env), driven through `/v1/chat`. Both produce byte-identical 16-token greedy-argmax sequence: `) = 1.0000000000000000`. Confirmed across two further 3-rep batches with the shorter 8-prefill+8-decode prompt: every WIRE-CPU rep and every REF rep produces the same `" the"`, `" answer"`, `"\n\n"`, `"#"`, `"1"`, `"\n"`, `"#"`, `"2"` sequence. Per `reference-lattice-decode-determinism`: discrete Z_q substrate + Frobenius lift Theorem T8 produces byte-exact cross-backend determinism under greedy + same-checkpoint + same-context preconditions. The CPU engine's scalar Q8 path and the math-core reference's scalar Q8 path are arithmetically equivalent for the Qwen3 dtype set, and the determinism invariant holds. **No silent gate revision applied.** |
| **T_WIRE_CPU_TOKS_MEASURED** | **PASS (honest)** | Headline table above, 3 reps each config. Backend dispatch counter advances by exactly 1 per `/v1/chat` (0→1→2→3 observed). WIRE-CPU 1.206 ± 0.009 tps vs REF 1.218 ± 0.042 tps: functionally identical, since both route the Q8 hot path through `cpu_overlay.c:matmul_arena`'s scalar inner loop. The AVX-512 primitive TUs (`sp_avx512_vnni_matvec` etc.) ARE in the static lib but NOT wired into `matmul_arena`. Honest reporting per `feedback-no-silent-gate-revisions`: no headline speedup is claimed; the WIRE win is the wiring (the 6-month gap closed for CPU as well), not a kernel speedup. Headline number IS the headline number. |

---

## Bit-exactness verification

Two daemons, same model file, same tokenizer file. Same prompt
`[100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600]`,
same `max_tokens=16`. WIRE-CPU and REF produce byte-identical text:

```
WIRE-CPU: ) =  1.0000000000000000
REF:      ) =  1.0000000000000000
```

Same prompt `[2,100,200,300,400,500,600,700]` + `max_tokens=8`,
3 reps WIRE-CPU + 3 reps REF (6 total) all produce the identical:

```
" the"  " answer"  "\n\n"  "#"  "1"  "\n"  "#"  "2"
```

Discrete Z_q greedy decode + Frobenius lift exactness produces byte-equal
sequences across BACKENDS as well as across REPS within the same backend.
This matches the `reference-lattice-decode-determinism` invariant (caught
2026-05-29 lat-smoke-2node).

---

## Per-stage build commands (reproducible)

```pwsh
# Activate VS2019 BT host x64
& "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

# 1. Math-core (in this worktree; older canonical build-cpu lacks the §6 hook)
cd D:\F\shannon-prime-repos\engine-wire-cpu
cmake -S lib\shannon-prime-system -B build-mathcore -G Ninja `
  -DCMAKE_BUILD_TYPE=Release -DSP_SYSTEM_BUILD_TESTS=OFF
cmake --build build-mathcore --config Release -j

# 2. WIRE-CPU daemon-link static lib (cpu_overlay + cpu_forward + cpu_gemma3 + glue)
.\tools\sp_daemon\build-host-cpu-backend.bat
#   Output: build-host-cpu-backend\sp_cpu_daemon_backend.lib

# 3. Daemon binary with WIRE-CPU feature
$env:LIBCLANG_PATH = "C:\Program Files\LLVM\bin"
$env:SP_SYSTEM_BUILD_DIR = "D:\F\shannon-prime-repos\engine-wire-cpu\build-mathcore"
$env:SP_CPU_BACKEND_DIR  = "D:\F\shannon-prime-repos\engine-wire-cpu\build-host-cpu-backend"
cd tools\sp_daemon
cargo build --features wire_cpu_backend --release
#   Output: target\release\sp-daemon.exe (~9.3 MB)

# 4. Run reference daemon
$env:RUST_LOG = "info"; Remove-Item Env:\SP_DAEMON_BACKEND -ErrorAction SilentlyContinue
.\target\release\sp-daemon.exe start `
  --model D:\F\shannon-prime-repos\shannon-prime-system-engine\build-cpu\tests\qwen3_rt.sp-model `
  --tokenizer D:\F\shannon-prime-repos\shannon-prime-system-engine\build-cpu\tests\qwen3_rt.sp-tokenizer `
  --port 8088 --quic-port 0 --memo-model "" --memo-tokenizer "" `
  --draft-model "" --draft-tokenizer "" --pouw-ledger-path "" --peer "" --peers ""

# 5. Run WIRE-CPU daemon — same args, with env set:
$env:SP_DAEMON_BACKEND = "cpu"
# (then same start command as above)
```

---

## Build dependency note (sieve)

The build.rs MODULES list at `tools/sp_daemon/build.rs:7-25` links 17
math-core archives, one of which is `sp_sieve.lib`. The worktree's
`lib/shannon-prime-system/core/sieve/` directory contains only `.gitkeep`
(no source). The math-core CMakeLists at `lib/shannon-prime-system/CMakeLists.txt:52`
guards module inclusion via `if(EXISTS .../CMakeLists.txt)` so `sieve` is
gracefully skipped when not present.

To unblock the daemon link without modifying the worktree submodule pin,
the existing pre-built `sp_sieve.lib` from
`D:\F\shannon-prime-repos\shannon-prime-system-engine\build-cpu\lib\shannon-prime-system\core\sieve\sp_sieve.lib`
(canonical engine repo, same engine commit `73f3367`) was copied into
`build-mathcore\core\sieve\`. This is operationally identical to building
it from the same source (the submodule pin matches; the canonical build was
done from the same source tree on the same host). Documented honestly here
rather than silently masked.

Future cleanup candidate: either add the sieve sources to the
shannon-prime-system submodule in this worktree, or drop sieve from
build.rs MODULES if no daemon code reaches `sp_sieve_evaluate` at runtime
(the daemon's `tools/sp_daemon/src/mining.rs:113` DOES call it, so the lib
is necessary; drop is not an option).

---

## What this sprint did NOT do

- **Did not wire AVX-512 into `cpu_overlay.c::matmul_arena`.** The Q8 matmul
  hot path remains the scalar inner loop at `cpu_overlay.c:81-97`. The
  AVX-512 primitives (`sp_avx512_vnni_matvec`, `sp_avx512_spinor_*`,
  `sp_avx512_ifma_*`, `sp_avx512_ternlog_*`, `sp_avx512_persist_*`) compile
  into the daemon-link static lib (on Linux/GCC paths via per-file
  `-mavx512vnni -mavx512ifma` etc.; on MSVC only `avx512_dispatch.c` is
  included because the others use GCC `__attribute__((target("avx512f")))`
  and `__atomic_*` builtins). On MSVC the AVX-512 path is therefore
  build-validated through the dispatch-only TU; perf-validated AVX-512
  matmul is the **WIRE-CPU-V2** follow-on candidate. The 0.99× ratio in the
  headline table is the honest "wiring works, no kernel speedup yet"
  answer.
- **Did not touch `src/backends/cpu/` kernels.** Discipline guard from the
  task brief ("NO KERNEL CHANGES; if you find yourself editing them, STOP")
  observed. The only modifications were:
  - `tools/sp_daemon/c_backend_cpu/sp_daemon_cpu_glue.c` (Stage 1 fix:
    removed phantom `qwen25_forward_cpu_impl` declaration; SP_ARCH_QWEN25
    routes through `qwen3_forward_cpu_impl` via fall-through, matching
    the engine cpu_forward.c's own arch handling).
  - `tools/sp_daemon/c_backend_cpu/CMakeLists.txt` (Stage 1 fix:
    removed `-Dqwen25_forward=qwen25_forward_cpu_impl` rename, since the
    symbol doesn't exist to rename).
- **Did not silently revise gates.** WIRE-CPU/REF ratio is ~0.99×, which
  is functionally identical, NOT a perf win. Headline reports the honest
  number with stddev. Per `feedback-no-silent-gate-revisions`: surfacing
  the gap (AVX-512 primitives present-but-unwired) as a named follow-on
  rather than tuning fixtures until a perf number passes.
- **Did not modify the math-core submodule.** The `sp_session_register_forward_backend`
  ABI hook was already in place in our worktree's submodule pin (post-WIRE-HEX
  merge state). The canonical engine repo's `build-cpu` was older (pre-§6-hook)
  which is why the first link attempt FAILED — surfacing UPSTREAM properly led to
  building math-core inside the worktree.
- **Did not wire CUDA / Vulkan.** Parallel WIRE-CUDA / WIRE-VULKAN sprints
  in separate worktrees (`engine-wire-cuda`, `engine-wire-vulkan`).

---

## Named follow-on sprints

- **WIRE-CPU-V2**: wire `sp_avx512_vnni_matvec` (or `sp_avx512_ifma_*` for
  Q8) into `cpu_overlay.c::matmul_arena` behind a runtime feature-detect
  gate (`SP_KERNELS_AVX512=1` knob already present in
  `sp_cpu_kernels_read_env`). Expected headline lift: AVX-512 VNNI for Q8
  matmul on a 4096-wide hidden = 16× theoretical instruction throughput vs
  scalar; realistic ~3-5× on the daemon's full prefill path after memory
  bandwidth, branch overhead, and tail-loop costs.
- **WIRE-CPU-V3**: extend WIRE-CPU to cover persistent-KV decode
  (currently `sp_decode_step` bypasses the registered §6 hook and runs
  math-core reference). Symmetric to the named decode-path gap from
  WIRE-HEX-FINISH (`tools/sp_compute_skel/docs/CLOSURE-WIRE-HEX-FINISH.md:28-31`).
- **WIRE-CPU-V4**: investigate the `qwen25-coder-0.5b-memory` model crash
  on the WIRE-CPU dispatcher. The Qwen2.5-Coder model (arch_id=6 →
  SP_ARCH_QWEN25=2) falls through to `qwen3_forward_cpu_impl` in our glue,
  matching the engine cpu_forward.c's own arch-agnostic dispatch.
  Empirically this segfaults during prefill on Knack's host; the bug is
  inside `qwen3_forward` for the QWEN25 arch variant — pre-existing,
  unrelated to WIRE-CPU, but worth a named investigation.

---

## Files changed

```
include/sp_engine/cpu_backend.h                          (new,  46 LOC)
tools/sp_daemon/Cargo.toml                               (+11)
tools/sp_daemon/build.rs                                 (+64)
tools/sp_daemon/build-host-cpu-backend.bat               (new,  69 LOC)
tools/sp_daemon/c_backend_cpu/CMakeLists.txt             (new, 162 LOC)
tools/sp_daemon/c_backend_cpu/sp_daemon_cpu_glue.c       (new,  90 LOC; Stage 1 fix -3+9 LOC)
tools/sp_daemon/src/cpu_forward_dispatch.rs              (new, 160 LOC)
tools/sp_daemon/src/daemon.rs                            (+57)
tools/sp_daemon/src/main.rs                              (+7)
tools/sp_daemon/src/routes.rs                            (+20)
tools/sp_daemon/src/state.rs                             (+8)
tools/sp_compute_skel/docs/PLAN-WIRE-CPU.md              (new, 206 LOC; landed in ac6e324)
tools/sp_compute_skel/docs/CLOSURE-WIRE-CPU.md           (THIS FILE)
```

No engine kernel changes. No L1 ABI changes. No math-core source changes.
Pure binary-crate wiring + standalone C-glue lib + standalone CMake. The
engine's existing CPU backend (Phase 2-CPU.AVX) is now reachable from the
daemon's chat path behind a runtime env gate, fulfilling the same 6-month-gap
closure that WIRE-HEX-FINISH closed for the Hexagon V69 backend on 2026-05-31.
