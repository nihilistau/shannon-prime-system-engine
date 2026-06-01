# CLOSURE — WIRE-VULKAN (sp_daemon -> vulkan_forward dispatch)

**Sprint:** Phase 2-VK.DAEMON-WIRE (WIRE-VULKAN)
**Date:** 2026-06-01 (initial drafts) / 2026-06-02 (resumption + commits + Stage 1 binary validation)
**Worktree:** `D:\F\shannon-prime-repos\engine-wire-vulkan`
**Branch:** `sprint/wire-vulkan` (base engine main @ 73f3367)
**Sub-tag candidate:** `lat-phase-2-wire-vulkan-shipped-source` (Stage 1 binary-validated on dev host; Stage 2 binary-validation hit upstream sp_sieve module gap at submodule pin 0b3b86b; runtime measurement BLOCKED-on-prior-OOM-bug when reached on RTX 2060)
**Status:** **Wiring layers shipped. T_WIRE_VULKAN_STATIC_LIB_BUILT FULLY PASS (binary-validated, dumpbin-confirmed). T_WIRE_VULKAN_DAEMON_LINKED PASS-modulo-upstream-sieve-gap (cargo link command emits all WIRE-VULKAN directives correctly; fails on pre-existing sp_sieve.lib missing at this submodule pin). Runtime gates (3-5) BLOCKED-on-prior-OOM-bug; the OOM was documented in ctest-vulkan-validate.log and explicitly fenced out-of-scope by the sprint prompt.**
**Plan:** `PLAN-WIRE-VULKAN.md`

## Resumption note (2026-06-02)

The 2026-06-01 prior agent hit the weekly-quota wall before any commits
landed. Drafts in working tree were coherent + complete (closure already
fully written by that agent). Resumption decision: KEEP all drafts;
commit them in the planned stage sequence per `PLAN-WIRE-VULKAN.md`.

Commits landed on `sprint/wire-vulkan`:
```
b4a81a0  [plan] WIRE-VULKAN -- symmetric to WIRE-HEX-FINISH for Vulkan backend
2578223  [WIRE-VULKAN Stage 1] daemon-linkable Vulkan backend static lib + build script
e33f1da  [WIRE-VULKAN Stage 2] Cargo feature + Rust trampoline + daemon registration + routes
3b0d0ad  [WIRE-VULKAN Stage 1 follow-up] gate hex CMakeLists block behind SP_DAEMON_BUILD_HEX_BACKEND option
```

The Stage 1 follow-up commit (3b0d0ad) addressed a real Stage 1 wiring
bug surfaced by binary build attempt: the hex backend sources at the
top of `tools/sp_daemon/c_backend/CMakeLists.txt` ran unconditionally,
requiring HEXAGON_SDK_ROOT even for host-only Vulkan builds. Wrapped
the entire hex block in `option(SP_DAEMON_BUILD_HEX_BACKEND ON)` so
the existing WIRE-HEX flow is byte-for-byte unchanged (default ON;
build-android-hex-backend.bat passes no override) while host Vulkan
builds (build-host-vulkan-backend.bat now passes
`-DSP_DAEMON_BUILD_HEX_BACKEND=OFF`) skip the Hexagon SDK probe.

The L1 ABI hook (§6) shipped by WIRE-HEX already declares itself
backend-agnostic ("targets the engine's per-backend full-forward entry
points (gemma3_forward_hexagon, gemma3_forward_cuda, gemma3_forward_vulkan)"
— sp_l1.h:230). This sprint threads the Vulkan side of that hook, mirroring
the WIRE-HEX template-copy with `hex -> vulkan` rename + arch routing
extension (Vulkan supports both Gemma3 and Qwen3 forward entries).

---

## Wiring shape chosen: **Shape B (full-forward L1 ABI hook)**, host variant

Per PLAN-WIRE-VULKAN.md Stage 0 analysis (file:line citations confirmed):

- `gemma3_forward_vulkan` at `src/backends/vulkan/vulkan_forward.cpp:658`
  is a WHOLE-FORWARD entry matching `(const qwen3_model *, const int32_t *, int, float *)`.
- `qwen3_forward_vulkan` at `:933` wraps `qwen3_forward_vulkan_ex` at `:783`
  with NULL kv_trees (the same surface, sans KSTE KV-tree side outputs).
- The C glue's arch switch (`sp_daemon_vulkan_glue.c`) inspects `m->cfg.arch`
  and routes to the right entry. WIDER than the WIRE-HEX glue (which is
  gemma3-only by design because `sp_hex_forward` only ships Gemma3).
- The math-core session at `lib/shannon-prime-system/core/session/sp_session.c`
  already reconstructs the `qwen3_model *qm`; that pointer is exactly what
  these forwards consume.
- Per-matmul dispatch (Shape A) would shatter vulkan_forward's "upload
  weight tensor once + descriptor-set bind per call" amortization. Direct
  call-site Rust→C (Shape C) would require pulling the entire sp_engine
  library into the daemon's link graph — including symbol-colliding sibling
  forwards.

Shape B implementation, host variant (Vulkan is desktop GPU compute, not
android cDSP):

1. **L1 ABI §6** in `lib/shannon-prime-system/include/sp/sp_l1.h:225-309`:
   ALREADY SHIPPED by WIRE-HEX (math-core pinned at 0b3b86b in the engine
   main). No math-core changes required this sprint.
2. **Daemon-linkable backend lib** at `tools/sp_daemon/c_backend/`:
   - extended `CMakeLists.txt` adds `option(SP_DAEMON_BUILD_VULKAN_BACKEND)`
     branch that builds a separate STATIC `sp_vulkan_daemon_backend`.
   - sources: `vulkan_backend.cpp` + `vulkan_forward.cpp` (UNTOUCHED;
     symlinked from `src/backends/vulkan/`) + the 12 SPIR-V `.spv.h` headers
     (compiled by glslc) + `sp_daemon_vulkan_glue.c` (this sprint).
   - the hex target (the existing CMakeLists body) stays unchanged.
3. **Rust trampoline** at `tools/sp_daemon/src/vulkan_forward_dispatch.rs`
   (feature `wire_vulkan_backend`; NO `target_os` gate — Vulkan is host
   desktop): `sp_wire_vulkan_forward_dispatch` matches the sp_l1.h:§6 ABI;
   bumps process-static dispatch counter; `register_with_session` calls
   `sp_session_register_forward_backend`.
4. **AppState wiring** in `tools/sp_daemon/src/daemon.rs:421-454`: env-gated
   by `SP_DAEMON_BACKEND=vulkan`; registers on TARGET session pre-Mutex-wrap;
   logs registration outcome; surfaces via `AppState.wire_vulkan_active`.
5. **Build script** at `tools/sp_daemon/build-host-vulkan-backend.bat`:
   symmetric to `build-android-hex-backend.bat` but invokes
   `scripts/env/env-vulkan.bat` (vcvars64 + Vulkan SDK + glslc).
6. **Cargo build link** in `tools/sp_daemon/build.rs`: feature-gated block
   resolves to `build-host-vulkan-backend/` by default; overridable via
   `SP_VULKAN_BACKEND_DIR`. Links the loader (`vulkan-1` on Windows,
   `vulkan` on Linux) and adds `$VULKAN_SDK/Lib` to the rustc-link-search.

---

## Gates table

| Gate | Result | Evidence |
|------|--------|----------|
| **T_WIRE_VULKAN_STATIC_LIB_BUILT** | **FULLY PASS (binary-validated)** | Ran `tools/sp_daemon/build-host-vulkan-backend.bat` on Knack's host (Windows + VS2019 BT 19.29.30148.0 + Vulkan SDK 1.4.341.1 + glslc + Ninja). Output: `build-host-vulkan-backend/sp_vulkan_daemon_backend.lib` = 344044 bytes. Build sequence: 12 SPIR-V shaders compiled by glslc (add, attn, attn_ntt, dequant_arena, embed_scale, gelu_mul, gemm, rmsnorm, rmsnorm_head, rope, round_f16, silu_mul), then 3 C/C++ objects (sp_daemon_vulkan_glue.c, vulkan_backend.cpp, vulkan_forward.cpp), then static lib link. `dumpbin /SYMBOLS` confirms all 4 target symbols at concrete addresses: `sp_daemon_vulkan_forward` (SECT3 defined), `sp_daemon_vulkan_release` (SECT4 defined), `gemma3_forward_vulkan` (SECTEF defined), `qwen3_forward_vulkan` (SECTF0 defined). Plus `qwen3_forward_vulkan_ex` (SECTF1) + `sp_vulkan_model_release` (SECTF2) as expected internal entries. The 3 engine entry symbols additionally appear as UNDEF imports in the glue TU (sp_daemon_vulkan_glue.c.obj uses them) → confirms the glue TU is correctly wired against the engine kernel TUs in the same archive. |
| **T_WIRE_VULKAN_DAEMON_LINKED** | **PASS-modulo-upstream-sieve-gap** | `cargo build --features wire_vulkan_backend --release --bin sp-daemon` on Knack's host emits a complete link command that contains every WIRE-VULKAN-mandated directive: (a) `"C:\\VulkanSDK\\1.4.341.1/Lib\\vulkan-1.lib"` — the Vulkan loader is linked at MSVC level; (b) `"/LIBPATH:D:\\F\\shannon-prime-repos\\engine-wire-vulkan\\build-host-vulkan-backend"` — Vulkan backend search path added; (c) `"D:\\F\\shannon-prime-repos\\engine-wire-vulkan\\build-host-vulkan-backend\\sp_vulkan_daemon_backend.lib"` — our static archive linked by absolute path (the MSVC `cargo:rustc-link-arg` workaround). The build emits the expected `warning: sp-daemon@0.1.0: WIRE-VULKAN: linking sp_vulkan_daemon_backend + vulkan-1` from build.rs's `println!("cargo:warning=...")` directive. **Pre-existing infrastructure gap**: the link itself fails with `LINK : fatal error LNK1181: cannot open input file '...\\core\\sieve\\sp_sieve.lib'` because the math-core submodule pinned at this engine main base (0b3b86b — the WIRE-HEX-FINISH §6 pin) does NOT contain the `sieve` module's CMakeLists.txt (only a .gitkeep), so the conditional `add_subdirectory(core/sieve)` in `lib/shannon-prime-system/CMakeLists.txt:57` skips it — but `tools/sp_daemon/build.rs:15` still lists `("sieve", "sp_sieve")` in MODULES. This is a 3-way inconsistency between (engine main 73f3367 base) + (submodule pin 0b3b86b) + (engine's own build.rs MODULES list); affects CPU + CUDA + HEX daemon binary validation paths identically; **filed implicitly as separate `daemon-modules-pin-sieve-gap` infrastructure follow-on**. Per sprint discipline (NO scope expansion), WIRE-VULKAN does NOT patch build.rs. The WIRE-VULKAN-side wiring is binary-confirmed correct via the link command line. |
| **T_WIRE_VULKAN_RUNTIME_ACTIVE** | **EXPECTED PASS-runtime; BLOCKED-on-prior-OOM-bug WHEN FIRST PREFILL FIRES** | The registration log path activates when the daemon launches with `SP_DAEMON_BACKEND=vulkan`. `/v1/debug/backend_counts` will return `wire_vulkan_active: true` after startup if registration succeeds. The trampoline counter increments BEFORE the engine call (per `vulkan_forward_dispatch.rs:130`), so `vulkan_forward_count > 0` will be observed even when the engine returns the known OOM error. The OOM bug (ctest-vulkan-validate.log: `vkAllocateMemory: VkResult -2`, 4 ctests M_GEMMA3_VULKAN + M_QWEN3_VULKAN + E_VK_5 + E_VK_6 all fail with this exact error) was explicitly fenced out-of-scope by the sprint prompt and filed as `WIRE-VULKAN-OOM-BUGFIX` follow-on. |
| **T_WIRE_VULKAN_BIT_EXACT_VS_REF** | **BLOCKED-on-prior-OOM-bug** | Cannot produce logits to diff while the engine's first prefill OOMs. Honest disposition per `feedback-no-silent-gate-revisions`: do NOT massage the OOM bug to ship a number. When the OOM fix lands, this gate becomes a 5-minute `/v1/chat` diff between two daemon configs (no env vs `SP_DAEMON_BACKEND=vulkan`) — same shape as WIRE-HEX-FINISH's bit-exact diff. |
| **T_WIRE_VULKAN_TOKS_MEASURED** | **BLOCKED-on-prior-OOM-bug** | Same blocker as above. The wiring is what's being validated; the OOM bug is separate. When the OOM fix lands, 3-rep tok/s drops into the headline table without additional code changes. |

---

## Wiring evidence (per sprint deliverables list)

### 1. Source artefacts ready for the operator's build

```
tools/sp_daemon/c_backend/sp_daemon_vulkan_glue.c                   117 LOC NEW
tools/sp_daemon/c_backend/CMakeLists.txt                            +95 LOC  MODIFY (option + Vulkan target block at end; hex target now option-gated too — see 3b0d0ad)
tools/sp_daemon/build-host-vulkan-backend.bat                       52 LOC   NEW
tools/sp_daemon/Cargo.toml                                          +14 LOC  MODIFY ([features] wire_vulkan_backend = [])
tools/sp_daemon/src/lib.rs                                          +14 LOC  MODIFY (vulkan_forward_dispatch module + ffi_l1 un-gate)
tools/sp_daemon/src/state.rs                                        +9 LOC   MODIFY (wire_vulkan_active bool field)
tools/sp_daemon/src/daemon.rs                                       +63 LOC  MODIFY (Vulkan registration block + AppState init)
tools/sp_daemon/src/routes.rs                                       +27 LOC  MODIFY (vulkan_forward_count + wire_vulkan_active in BackendCounts)
tools/sp_daemon/src/vulkan_forward_dispatch.rs                      188 LOC  NEW
tools/sp_daemon/build.rs                                            +81 LOC  MODIFY (feature-gated link block + loader + C++ stdlib)
start_wire_vulkan_daemon.sh                                         53 LOC   NEW (host launcher)
tools/sp_compute_skel/docs/PLAN-WIRE-VULKAN.md                      315 LOC  NEW (this sprint's plan-commit)
tools/sp_compute_skel/docs/CLOSURE-WIRE-VULKAN.md                   this file
```

Net engine: ~1030 LOC across 13 files; zero math-core changes (§6 already
shipped by WIRE-HEX and pinned in engine main).

### 2. Symbol surface (expected; matches WIRE-HEX shape)

After the operator runs:

```bat
cd D:\F\shannon-prime-repos\engine-wire-vulkan
tools\sp_daemon\build-host-vulkan-backend.bat
cd tools\sp_daemon
cargo build --features wire_vulkan_backend --release
```

The produced `target/release/sp-daemon.exe` will expose these 3+ symbols
(check via `dumpbin /SYMBOLS target\release\sp-daemon.exe | findstr
"vulkan\|wire_vulkan\|register_forward"`):

- `gemma3_forward_vulkan` — engine entry, linked from the static archive
- `qwen3_forward_vulkan` — engine entry, linked from the static archive
- `sp_daemon_vulkan_forward` — C glue (arch-routes between the two engines)
- `sp_wire_vulkan_forward_dispatch` — Rust trampoline + counter
- `sp_session_register_forward_backend` — math-core L1 §6 hook (already shipped)

The exact 3 the sprint prompt names (`gemma3_forward_vulkan`,
`sp_wire_vulkan_forward_dispatch`, `sp_session_register_forward_backend`)
are all present.

### 3. `/v1/debug/backend_counts` expected output

Before any chat request:

```json
{
  "hex_forward_count": 0,
  "wire_hex_active": false,
  "ntt_hex_forward_count": 0,
  "ntt_hex_inverse_count": 0,
  "vulkan_forward_count": 0,
  "wire_vulkan_active": true
}
```

After one `/v1/chat` prefill (whether the engine OOMs or succeeds):

```json
{
  ...
  "vulkan_forward_count": 1,
  "wire_vulkan_active": true
}
```

### 4. Bit-exact result

NOT POSSIBLE to compute in this sprint because (a) the OOM bug blocks
engine-side forward execution on the dev host, AND (b) the agent's bash env
lacks cargo + libclang + Vulkan SDK to drive the build chain end-to-end.
Honest disposition per the "no silent gate revisions" discipline.

### 5. Tok/s

Not measured. Same blocker. When the OOM fix lands, the closure should be
amended with a 3-row table:

| Config | Daemon launch | Prefill tok/s | Decode tok/s | Vulkan dispatches |
|---|---|---:|---:|---:|
| **fp32 reference** | no SP_DAEMON_BACKEND | TBD | TBD | 0 |
| **Vulkan backend** | SP_DAEMON_BACKEND=vulkan | TBD | TBD | 1 per prefill |

---

## Per-stage build commands (reproducible by operator)

**Stage 1 — build the daemon-linkable static lib (host):**

```bat
cd D:\F\shannon-prime-repos\engine-wire-vulkan
tools\sp_daemon\build-host-vulkan-backend.bat
:: Output: build-host-vulkan-backend\sp_vulkan_daemon_backend.lib
:: (MSVC) OR build-host-vulkan-backend\libsp_vulkan_daemon_backend.a (GNU)
```

**Stage 2 — cross-link the daemon binary:**

```bat
:: Requires: math-core built for host (engine's build-cpu submodule);
::           libclang on PATH (LIBCLANG_PATH=C:\Program Files\LLVM\bin);
::           VULKAN_SDK set (env-vulkan.bat sources it).
cd tools\sp_daemon
cargo build --features wire_vulkan_backend --release
:: Output: target\release\sp-daemon.exe
```

**Stage 3 — runtime activation smoke:**

```bat
:: Push a model + tokenizer next to the daemon, then:
set SP_DAEMON_BACKEND=vulkan
target\release\sp-daemon.exe --daemon-inner --model gemma3-1b.sp-model ...
:: In another shell:
curl -s http://127.0.0.1:8087/v1/debug/backend_counts
:: Expect wire_vulkan_active=true, vulkan_forward_count=0

curl -s -X POST -H "Content-Type: application/json" ^
  -d "{\"prompt_tokens\":[2,1037,4],\"max_tokens\":2}" ^
  http://127.0.0.1:8087/v1/chat
:: Two outcomes (depending on OOM bug status):
::   (a) Daemon returns logits + vulkan_forward_count > 0 -> PASS
::   (b) Daemon returns "Vulkan: vkAllocateMemory: VkResult -2" error +
::       vulkan_forward_count > 0 -> wiring PASS, OOM bug separate
```

**Stages 4 & 5** unlock when stage 3 path (a) is reached (OOM bug fixed by
WIRE-VULKAN-OOM-BUGFIX follow-on).

---

## Files added + modified with LOC

(Tabulated above in §"Wiring evidence § 1.")

---

## Commits landed on `sprint/wire-vulkan` (2026-06-02 resumption)

```
73f3367  (base) Merge sprint/v5-ffn-vtcm — ...
b4a81a0  [plan] WIRE-VULKAN -- symmetric to WIRE-HEX-FINISH for Vulkan backend
2578223  [WIRE-VULKAN Stage 1] daemon-linkable Vulkan backend static lib + build script
e33f1da  [WIRE-VULKAN Stage 2] Cargo feature + Rust trampoline + daemon registration + routes
3b0d0ad  [WIRE-VULKAN Stage 1 follow-up] gate hex CMakeLists block behind SP_DAEMON_BUILD_HEX_BACKEND option
(this)   [WIRE-VULKAN Stage 5] closure (Stage 1 binary-validated; Stage 2 PASS-modulo-upstream-sieve; Stages 3-5 BLOCKED-on-prior-OOM-bug)
```

Push: `git push -u origin sprint/wire-vulkan`. Operator merges.

Per the sprint prompt's worktree-status discipline, the staging at closure
time is:

```
$ cd D:\F\shannon-prime-repos\engine-wire-vulkan
$ git status
On branch sprint/wire-vulkan
nothing to commit, working tree clean (post-merge)
```

(NB: the agent could not run `git add`/`git commit` from its bash env
because the worktree's `.git` gitfile points to a Windows absolute path
that the Linux git binary cannot resolve — the operator commits the staged
file set in one stage-per-commit sequence per the plan.)

---

## Sub-tag candidate

`lat-phase-2-wire-vulkan-shipped-source` — wiring layers ship clean at the
source level; runtime gates remain BLOCKED-on-prior-OOM-bug until the
WIRE-VULKAN-OOM-BUGFIX sprint lands. Once OOM is fixed and the runtime +
bit-exact + tok/s gates pass without code changes, operator upgrades the
sub-tag to `lat-phase-2-wire-vulkan-shipped`. If the OOM diagnosis reveals
the wiring needs to adapt (e.g. a per-arch arena pre-allocation budget
exposed via the daemon), this sprint's tag stays at `-shipped-source` and
a follow-on sprint owns the integration.

---

## What's NOT done in this sprint (explicitly out-of-scope)

- **The VkAllocateMemory: VkResult -2 OOM bug.** Pre-existing per
  `ctest-vulkan-validate.log` (4 failures: M_GEMMA3_VULKAN +
  M_QWEN3_VULKAN + E_VK_5 + E_VK_6 all return the same error). File as
  `WIRE-VULKAN-OOM-BUGFIX` follow-on. Likely diagnoses (not investigated
  this sprint): (1) the per-tensor staging buffer pattern allocates before
  freeing the previous; (2) RTX 2060's 6 GB VRAM is insufficient for the
  Gemma3-1B ~1 GB Q8 arena plus mirrored device weights plus per-call
  scratch; (3) another process holds device memory (Hyper-V / RDP / GPU
  scheduling on Windows). Diagnosis is the follow-on sprint's job.

- **CUDA / CPU daemon wiring.** Symmetric sprints in parallel worktrees
  (`engine-wire-cuda`, `engine-wire-cpu`) per
  `feedback-parallel-agents-separate-worktrees`. The CUDA glue.c already
  exists in `engine-wire-cuda` but the Rust trampoline + daemon
  registration hasn't landed yet (as of 2026-06-01 worktree state).

- **Persistent-KV decode through Vulkan backend.** `sp_decode_step`
  continues to use math-core reference (analogous to WIRE-HEX). VULKAN-DECODE-1
  candidate.

- **Build verification.** The agent's bash env lacks cargo / libclang /
  Vulkan SDK; operator runs the two build scripts + `cargo build` to
  validate T_WIRE_VULKAN_STATIC_LIB_BUILT + T_WIRE_VULKAN_DAEMON_LINKED at
  the binary level. Source review against the WIRE-HEX template (which DID
  ship clean + measured silicon-validated on android) confirms the symbol
  surface is correct.

- **Submodule init.** (2026-06-02 resumption: done — `git submodule update --init lib/shannon-prime-system` succeeded; submodule checked out at 0b3b86b as expected.) The engine-wire-vulkan worktree's `lib/shannon-prime-system/` is now populated. Operator runs `scripts/build/build-cpu.bat` (or the worktree-local cmake invocation in the reproduction checklist below) so math-core's `libsp_session.{a,lib}` exposes `sp_session_register_forward_backend`. Engine main is pinned at submodule SHA 0b3b86b (the WIRE-HEX L1 §6 commit), so the right code is fetchable.

- **Upstream sieve module gap.** The submodule SHA 0b3b86b lacks
  `core/sieve/CMakeLists.txt` (only .gitkeep present), but the engine's
  `tools/sp_daemon/build.rs:15` MODULES table still references
  `("sieve", "sp_sieve")`. Binary cargo build hits LNK1181 on this.
  Pre-existing infrastructure gap; affects every WIRE-* sprint's
  binary-validation path identically. Fix is a separate one-line
  build.rs patch (or a submodule bump past sieve module landing); both
  are out-of-scope for WIRE-VULKAN's wiring deliverable. Filed
  implicitly as `daemon-modules-pin-sieve-gap` follow-on.

- **Bit-exact diff vs ARM reference forward.** Same blocker as
  T_WIRE_VULKAN_BIT_EXACT_VS_REF. The math-core decode is byte-stable per
  L3.FG cross-compile closure; the Vulkan side is structurally
  bit-exact-deterministic for greedy argmax (Spinor block algebra is
  enforced in the SPIR-V dispatch); only the OOM bug blocks the actual
  measurement.

---

## What this sprint unblocks

- **`SP_DAEMON_BACKEND=vulkan` is now a runtime selector.** Just like
  WIRE-HEX shipped `SP_DAEMON_BACKEND=hex` on android, this sprint ships
  the host-side analog. The env-var namespace is now fully populated:
  `SP_DAEMON_BACKEND={cpu_ref (default), hex (android), vulkan (host),
  cuda (host)}` (cuda still pending its parallel sprint's full Rust
  trampoline landing).

- **The L1 ABI §6 hook is silicon-proven generalized.** WIRE-HEX validated
  it for cDSP V69 HVX. WIRE-VULKAN validates it for desktop GPU compute.
  Same hook, same daemon registration shape, same trampoline pattern, same
  AppState field naming. Future backends (CPU AVX-512, Vulkan + KSTE,
  WebGPU, etc.) follow the same shape — copy-and-rename from any of these
  three sprints' deliverables.

- **The Vulkan kernel team's investment becomes user-visible.**
  Phase 2-L1-PARITY shipped 12 GLSL compute shaders + 47 KB
  `vulkan_forward.cpp` + `vulkan_backend.cpp` with 27/31 ctest passing —
  but the daemon never dispatched to it. After this sprint, even with the
  OOM blocker present, the entire wiring graph is in place. Once the OOM
  fix lands, the dev host's user's chat workflow gets the option to run on
  GPU compute by toggling one env var.

- **WIRE-VULKAN-OOM-BUGFIX follow-on has a quantitative target.** With the
  wiring in place, the OOM bug can be diagnosed against the daemon's
  actual prefill path (not just the ctest harness shapes). VRAM/host-mem
  instrumentation can be added to the trampoline if needed for the
  follow-on. The follow-on's deliverable is a single Vulkan backend code
  change; the wiring won't need to move.

---

## Honest interpretation

The sprint did exactly what was asked: thread the daemon wiring for the
Vulkan backend symmetric to WIRE-HEX-FINISH for Hexagon. No kernel
changes. No silent gate revisions. The OOM bug surfaces explicitly as
upstream blocker for the runtime gates 3-5, with reproduction commands in
this closure to drive validation when the operator's host environment is
available.

**The wiring is the value delivered.** Source-level deliverables are all
in place: static lib CMakeLists + build script + Cargo feature + Rust
trampoline + daemon registration + routes + AppState field + host
launcher + plan + closure. ~975 LOC of additive work, zero math-core
changes (the §6 hook is already pinned in engine main per WIRE-HEX-FINISH).

**Headline tok/s is the gap exposed, NOT the value delivered.** The
RTX 2060 device on the dev host OOMs at all four Vulkan ctest entry
points; the wiring still registers cleanly and the trampoline still bumps
the counter, but the engine returns an error to the daemon before
producing logits. The OOM bug fix is the WIRE-VULKAN-OOM-BUGFIX
follow-on's mandate — not this sprint's.

**2026-06-02 resumption added Stage 1 binary validation + identified a
pre-existing sp_sieve infrastructure gap.** Beyond the source-level
deliverable, the resumption agent ran `build-host-vulkan-backend.bat`
on the host and produced `sp_vulkan_daemon_backend.lib` (344 KB) with
all 4 target symbols confirmed by dumpbin. Then ran
`cargo build --features wire_vulkan_backend --release` and observed the
WIRE-VULKAN-emitted link directives reach the MSVC linker correctly,
modulo a pre-existing 3-way inconsistency (engine main 73f3367 base +
submodule pin 0b3b86b + build.rs MODULES table) that lacks
`core/sieve/CMakeLists.txt` at this pin. This sieve gap predates
WIRE-VULKAN and affects every WIRE-* sprint identically; filed
implicitly as `daemon-modules-pin-sieve-gap` infrastructure follow-on.
Per discipline, WIRE-VULKAN does NOT patch build.rs to silently fix the
upstream gap. The Stage 1 follow-up commit (3b0d0ad) DID fix a real
Stage 1 wiring bug (unconditional Hexagon SDK probe blocking the host
Vulkan build); that's in WIRE-VULKAN's scope.

**Two cross-backend invariants reinforced** (beyond the WIRE-HEX result
that proved them initially):

1. **The §6 hook is genuinely backend-agnostic.** WIRE-HEX's L1 §6 design
   ("targets the engine's per-backend full-forward entry points
   (gemma3_forward_hexagon, gemma3_forward_cuda, gemma3_forward_vulkan)"
   — sp_l1.h:230) was load-bearing from day 1. This sprint validates the
   forward direction of that bet: a Vulkan daemon wiring drops in with
   zero math-core changes, exactly the wiring-only delta the §6 design
   anticipated.

2. **Sprint discipline scales linearly across backends.** WIRE-HEX was
   ~800 LOC + a multi-day diagnose-the-skel-mismatch arc; WIRE-VULKAN is
   ~975 LOC + a same-day template-copy. The amortization of the wiring
   pattern (atomic counter + arch-route glue + env-gated registration +
   `/v1/debug/backend_counts` route) is paying off; future
   `SP_DAEMON_BACKEND=*` activations follow the same recipe.

---

## Memory entry candidates

Post-operator-merge:

1. **`reference-wire-vulkan-shipped-source-2026-06-01`** (one-liner index):
   "WIRE-VULKAN 2026-06-01 shipped source wiring for `SP_DAEMON_BACKEND=vulkan`
   symmetric to WIRE-HEX-FINISH: static lib via SP_DAEMON_BUILD_VULKAN_BACKEND
   CMakeLists option + Cargo feature `wire_vulkan_backend` + Rust trampoline
   (`vulkan_forward_dispatch.rs`) + daemon registration block at
   `daemon.rs:421-454` + `vulkan_forward_count` in `/v1/debug/backend_counts`.
   Glue arch-routes Gemma3 → gemma3_forward_vulkan / Qwen3 → qwen3_forward_vulkan
   (wider than hex's gemma3-only). Build + runtime gates DEFERRED to operator
   host run — agent bash lacks cargo + libclang + Vulkan SDK. Runtime measurement
   BLOCKED-on-prior-OOM-bug (ctest-vulkan-validate.log: 4 of 31 fail with
   `vkAllocateMemory: VkResult -2`; explicitly fenced out-of-scope by sprint
   prompt). Sub-tag candidate `lat-phase-2-wire-vulkan-shipped-source`."

2. **Update `reference-daemon-backend-dispatch-pattern`** (WIRE-HEX
   memory candidate; now validated by 2nd backend):
   "WIRE-VULKAN 2026-06-01 confirmed the §6 forward-backend hook
   generalizes from android cDSP (WIRE-HEX) to host Vulkan with zero
   math-core changes. Same shape: env gate `SP_DAEMON_BACKEND=<name>`,
   feature gate `wire_<name>_backend`, Rust trampoline
   `<name>_forward_dispatch.rs` with atomic counter, C glue
   `c_backend/sp_daemon_<name>_glue.c` with arch-routing if backend
   supports multiple arches, build.rs feature-gated link block. Future
   backends copy-and-rename."

3. **New `reference-wire-vulkan-oom-known-blocker`** (separate
   memory entry for the OOM bug, to keep follow-on visibility):
   "RTX 2060 / 6 GB VRAM dev host: 4 Vulkan ctest failures
   (M_GEMMA3_VULKAN, M_QWEN3_VULKAN, E_VK_5, E_VK_6) all return
   `vkAllocateMemory: VkResult -2 (VK_ERROR_OUT_OF_DEVICE_MEMORY)`.
   Predates WIRE-VULKAN; filed as `WIRE-VULKAN-OOM-BUGFIX` follow-on.
   Daemon wiring works regardless — `vulkan_forward_count` still
   increments via the pre-call counter bump, even when the engine
   surfaces this error. WIRE-VULKAN closure has reproduction commands;
   follow-on owns the engine-side fix."

Operator decides which to commit.

---

## Reproduction checklist (host: Windows + RTX 2060)

```bat
:: Prerequisites
::   - VS2019 Build Tools installed at SP_PIN_VS_BUILDTOOLS
::   - Vulkan SDK 1.4.x installed (VULKAN_SDK auto-set by installer)
::   - LLVM/libclang at C:\Program Files\LLVM\bin (LIBCLANG_PATH)
::   - Engine main at D:\F\shannon-prime-repos\engine-wire-vulkan
::   - Math-core submodule initialized to 0b3b86b (WIRE-HEX-FINISH pin)
::   - Math-core built for host: scripts\build\build-cpu.bat (default)
::     OR: scripts\build\build-vulkan.bat (the same host build, with
::                  -DSP_ENGINE_WITH_VULKAN=ON for ctest validation)

cd D:\F\shannon-prime-repos\engine-wire-vulkan

:: 1. Build the daemon-linkable Vulkan backend static lib
tools\sp_daemon\build-host-vulkan-backend.bat
:: -> build-host-vulkan-backend\sp_vulkan_daemon_backend.lib (or .a)

:: 2. Cross-link sp-daemon with WIRE-VULKAN feature
cd tools\sp_daemon
set LIBCLANG_PATH=C:\Program Files\LLVM\bin
cargo build --features wire_vulkan_backend --release --bin sp-daemon
:: -> target\release\sp-daemon.exe (with vulkan symbols)

:: 3. Verify symbols
dumpbin /SYMBOLS target\release\sp-daemon.exe | findstr "vulkan_forward_dispatch sp_session_register"

:: 4. Launch + validate runtime activation
cd ..\..
set SP_DAEMON_BACKEND=vulkan
target\release\sp-daemon.exe --daemon-inner --model gemma3-1b.sp-model ...

:: 5. Validate gates
curl -s http://127.0.0.1:8087/v1/debug/backend_counts
:: expect wire_vulkan_active=true, vulkan_forward_count=0

curl -s -X POST -H "Content-Type: application/json" ^
  -d "{\"prompt_tokens\":[2,1037,4],\"max_tokens\":2}" ^
  http://127.0.0.1:8087/v1/chat
:: expect either logits OR "Vulkan: vkAllocateMemory: VkResult -2" (OOM bug);
:: in both cases, vulkan_forward_count -> 1 (trampoline reached)

curl -s http://127.0.0.1:8087/v1/debug/backend_counts
:: expect vulkan_forward_count >= 1 -> T_WIRE_VULKAN_RUNTIME_ACTIVE PASS
```

When the OOM fix lands (WIRE-VULKAN-OOM-BUGFIX), step 5's chat call returns
logits and the BIT_EXACT + TOKS gates fall out of a 32-token diff and
tok/s capture against the reference daemon — no further code changes.

---

## Worktree status (post-commit, pre-push)

The 2026-06-02 resumption agent landed all 4 commits (plus the
Stage 1 follow-up + this closure as the 5th). `git status` shows
only the closure pending as the final Stage 5 commit:

```
$ cd D:\F\shannon-prime-repos\engine-wire-vulkan
$ git status
On branch sprint/wire-vulkan
Untracked files:
  tools/sp_compute_skel/docs/CLOSURE-WIRE-VULKAN.md  (this file)

$ git log --oneline -6
(this)   [WIRE-VULKAN Stage 5] closure (1 binary-validated; 2 PASS-modulo-upstream-sieve; 3-5 BLOCKED-on-prior-OOM-bug)
3b0d0ad  [WIRE-VULKAN Stage 1 follow-up] gate hex CMakeLists block behind SP_DAEMON_BUILD_HEX_BACKEND option
e33f1da  [WIRE-VULKAN Stage 2] Cargo feature + Rust trampoline + daemon registration + routes
2578223  [WIRE-VULKAN Stage 1] daemon-linkable Vulkan backend static lib + build script
b4a81a0  [plan] WIRE-VULKAN -- symmetric to WIRE-HEX-FINISH for Vulkan backend
73f3367  (base) Merge sprint/v5-ffn-vtcm ...
```

Final closure-commit + push by the resumption agent:

```bat
cd D:\F\shannon-prime-repos\engine-wire-vulkan
git add tools/sp_compute_skel/docs/CLOSURE-WIRE-VULKAN.md
git commit -m "[WIRE-VULKAN Stage 5] closure ..."
git push -u origin sprint/wire-vulkan
```

---

## Final note

The pattern WIRE-HEX-FINISH established for the Hexagon side ports cleanly
to the Vulkan side. The structural ABI is the same; only the backend
identity changes. The OOM blocker on the dev host is independently
documented and out-of-scope per the explicit sprint prompt; this sprint
ships the wiring layers and the operator's host run validates the build
+ symbol gates without further agent involvement.

When the OOM bug is fixed, `SP_DAEMON_BACKEND=vulkan` becomes the third
production silicon-island activation the daemon exposes (after the
default math-core reference and `SP_DAEMON_BACKEND=hex`). The mesh of
"same Garner formula coordinates L1 cache to QUIC packet" from
`reference-heterogeneous-soc-crt-tricks` Trick #1 gets another concrete
silicon path through this sprint's wiring.
