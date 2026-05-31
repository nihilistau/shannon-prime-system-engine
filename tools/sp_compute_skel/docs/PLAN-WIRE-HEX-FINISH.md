# PLAN — WIRE-HEX-FINISH

**Sprint:** Phase 2-HX.DAEMON-BENCH-BASELINE (WIRE-HEX-FINISH)
**Date:** 2026-05-31
**Worktree:** `D:\F\shannon-prime-repos\engine-wire-finish`
**Branch:** `sprint/wire-hex-finish` (base: `ba76c69` post-WIRE-HEX merge)
**Sub-tag candidate:** `lat-phase-2-hx-daemon-bench-baseline`
**Status (plan-commit):** Stage 0 complete; Stage 1 ready to execute

## Stage 0 — pre-read with file:line citations

1. **WIRE-HEX closure** — `tools/sp_compute_skel/docs/CLOSURE-WIRE-HEX.md:84` — exact blocker wording: *"`gemma3_forward_hexagon` returns 1 with detail `\"hexagon: sp_hex_forward failed\"` AFTER the host-side path successfully (a) opens FastRPC handle, (b) uploads 700 MB Q8 weight blob, (c) calls the FastRPC `forward` method. The cDSP-side `libsp_hex_skel.so` on `/data/local/tmp/sp22u/` appears to not match the current `inc/sp_hex.idl`"*. Same closure §"What's NOT done":174-181 names the rebuild path: `scripts/build/build-hexagon.bat dsp` + `adb push libsp_hex_skel.so /data/local/tmp/sp22u/`.

2. **NTT-bench closure methodology** — `tools/sp_compute_skel/docs/CLOSURE-NTT-bench.md:14-16` — headline table baseline: Memory fp32 prefill **2.966 tok/s** / decode **2.235 tok/s** (3-rep means, ±0.9%/±0.15%). Methodology lines 88-118: synthetic token IDs `[1..16]` prefill, 32 decode-step, greedy argmax fed back. Hex backend is **gemma3-only** by design (closure line 30 `qwen3 arch; hex backend is gemma3-only`).

3. **Skel build script** — `scripts/build/build-hexagon.bat:50-58` (`:dsp` target). Uses `build_cmake hexagon DSP_ARCH=%SP_HEXAGON_TARGET% BUILD=%SP_BUILD_TYPE_DEFAULT% -gMake` invoked from `%SP_ENGINE%\src\backends\hexagon\dsp` (CMakeLists.txt confirmed at that path). Env-hexagon.bat:22 pins `SP_PIN_HEXAGON_SDK=C:\Qualcomm\Hexagon_SDK\5.5.6.0`, line 59 pins `SP_HEXAGON_TARGET=v69`, line 60 pins `SP_HEXAGON_TOOLS_VARIANT=toolv87`. CRITICAL: env-common.bat:13 pins `SP_ENGINE=D:\F\shannon-prime-repos\shannon-prime-system-engine` — NOT our worktree. We must override `SP_ENGINE` to point at `engine-wire-finish` before invoking the build script.

4. **Hexagon backend IDL** — `src/backends/hexagon/inc/sp_hex.idl`. Four methods on `interface sp_hex : remote_handle64`:
   - `long ping(in long x, rout long y)` (line 23)
   - `long upload_crc(in sequence<uint8> data, rout long crc)` (line 30)
   - `long matmul_f32(in sequence<float> w, in long rows, in long cols, in sequence<float> x, rout sequence<float> y)` (lines 36-37)
   - `long forward(in long n_layers, in long n_embd, in long n_ff, in long head_dim, in long n_head, in long n_head_kv, in long sliding_window, in float eps, in float rope_global, in float rope_local, in long n_tok, in sequence<float> x, in sequence<uint8> weights, in sequence<float> scratch, rout sequence<float> hidden)` (lines 45-49)

   DSP implementation `src/backends/hexagon/dsp/sp_hex_imp.c` exports matching `sp_hex_open/close/ping/upload_crc/matmul_f32/forward` (line 102/111/116/139/154/251 — confirmed via grep).

5. **sp_compute_skel build** — `tools/sp_compute_skel/build.cmd:18` produces `libsp_compute_skel.so` (the NTT per-primitive skel — DIFFERENT from `libsp_hex_skel.so`). NTT-bench rebuild path. Our skel is the WIRE-HEX full-forward skel built via `scripts/build/build-hexagon.bat dsp` per (3).

6. **WIRE-HEX daemon launcher** — `start_wire_hex_daemon.sh:8-9` sets `SP_DAEMON_BACKEND=hex` + `ADSP_LIBRARY_PATH=/data/local/tmp/sp22u`. Expected device path for skel: `/data/local/tmp/sp22u/libsp_hex_skel.so` (per ADSP_LIBRARY_PATH + skel filename). The daemon binary `sp-daemon-wire-hex` is already on device (mtime 12:17 today, freshly built per WIRE-HEX closure §Repro). Companion `start_ref_daemon.sh` runs the same binary with `SP_DAEMON_BACKEND` unset for the reference baseline.

7. **Memory references**:
   - `reference-qnn-htp-unsigned-pd-access` — three operational gotchas: skel pathing via ADSP_LIBRARY_PATH (vendor's libSnpeHtp* is wrong; daemon expects our `libsp_hex_skel.so` on that path), per-execute ~1.3 ms amortized via persistent daemon.
   - `reference-mode-d-bridge-architecture` — FastRPC Path B Unsigned PD: DSPRPC_CONTROL_UNSIGNED_MODULE before remote_handle_open; IDL inherits `remote_handle64`; exact-size match with IDL Len (off-by-one = silent AEE_EUNSUPPORTED). Our IDL line 21 inherits `remote_handle64` ✓.

## Pre-state snapshot

| Item | Value |
|---|---|
| On-device skel SHA-256 (pre) | `938FED02656B079624D55277E6AB47E0DE1CC56C534558174DB779DFFC6DF9FD` |
| On-device skel size (pre) | 350608 bytes |
| On-device skel mtime (pre) | 2026-05-18 23:11 |
| Daemon binary on device | `sp-daemon-wire-hex` 9,731,920 bytes, mtime 2026-05-31 12:17 |
| Gemma3-1B model | `/data/local/tmp/gemma3-1b.sp-model` 1,003,371,008 bytes, mtime 2026-05-31 12:09 |
| Memory model | `/data/local/tmp/qwen25-coder-0.5b-memory.sp-model` 496,202,752 bytes |
| Device | R5CT22445JA (S22U) |
| ADB | `D:\Files\Android\pt-latest\platform-tools\adb.exe` |
| Hexagon SDK | `C:\Qualcomm\Hexagon_SDK\5.5.6.0` (verified present) |
| NDK (r27d) | `D:\Files\Android\android-ndk-r27d` (verified present) |
| Git for Windows sh.exe | `C:\Program Files\Git\usr\bin\sh.exe` (verified present) |

## Stage 1 — build the cDSP skel from current IDL

**Override SP_ENGINE** to `engine-wire-finish` then invoke `scripts/build/build-hexagon.bat dsp`. Expected output: `src/backends/hexagon/dsp/hexagon_Release_toolv87_v69/ship/libsp_hex_skel.so`.

**Gate T_WIRE_HEX_FINISH_SKEL_BUILT** — `libsp_hex_skel.so` exists at expected path; arch reported as Hexagon V69 via `file` or readelf.

If build fails on toolchain/SDK issues: STOP and surface UPSTREAM (no silent gate revisions per `feedback-no-silent-gate-revisions`).

## Stage 2 — push to device, verify daemon dlopen

`adb push` the new skel to `/data/local/tmp/sp22u/libsp_hex_skel.so`. Verify mtime + size differ from pre-state. Capture post-hash.

**Gate T_WIRE_HEX_FINISH_SKEL_PUSHED** — on-device stat shows new mtime/size; daemon launched via `start_wire_hex_daemon.sh` reports `wire_hex_active: true`.

## Stage 3 — bit-exact gate

Drive `/v1/chat` against the daemon launched with `SP_DAEMON_BACKEND=hex` AND a second daemon run with `start_ref_daemon.sh` (SP_DAEMON_BACKEND unset). Same fixed prompt (synthetic prompt_tokens `[2,1037,4,5,6,7,8,9,10,11,12,13,14,15,16,17]` to match NTT-bench prefill shape). Compare output token sequence.

**Gate T_WIRE_HEX_FINISH_BIT_EXACT** — identical greedy-argmax token sequence between hex and reference paths. Per `reference-lattice-decode-determinism` preconditions are met (greedy sampling, same model, same backend hardware for both runs).

If divergence: document honestly with first divergent index + both token values.

## Stage 4 — tok/s measurement (THE NUMBER)

Three configs, prefill+decode tok/s on Gemma3-1B (hex backend is gemma3-only):

| Config | Daemon launch | Expected behavior |
|---|---|---|
| **fp32 reference** | `start_ref_daemon.sh` | math-core reference forward, no NTT overlay |
| **hex backend** | `start_wire_hex_daemon.sh` | gemma3_forward_hexagon via cDSP |
| **hex backend + NTT-attention hex** | as above + `SP_ENGINE_NTT_ATTN=1 SP_ENGINE_NTT_ATTN_HEX=1` exported in shell | full-forward owned by hex backend (NTT overlay bypassed per WIRE-HEX closure note line 172 — gemma3_forward_hexagon owns entire forward) |

**NOTE:** The third config may not actually exercise the NTT-attention overlay because hex backend's `gemma3_forward_hexagon` owns the entire forward — math-core's NTT-attention overlay code path is bypassed. Per WIRE-HEX closure line 172-173: *"the prefill goes through the WIRE-HEX backend (gemma3_forward_hexagon owns the entire forward — bypassing math-core's NTT-attention overlay). Coexistence is a future decision."* This is the honest reporting per `feedback-no-silent-gate-revisions`.

Use the NTT-bench prefill+decode methodology (prefill=16, decode=32). Drive via `/v1/chat` since the daemon doesn't expose the `sp_ntt_bench_toks` harness for gemma3. If `/v1/chat` is the only channel, measure wall-clock for prefill (first response) and decode (subsequent tokens) directly from response latency.

**Gate T_WIRE_HEX_FINISH_TOKS** — 3-row table populated with honest numbers. Reference baseline: Gemma3-1B 0.89 tok/s per WIRE-HEX closure headline (single run; we'll re-measure to confirm).

## Stage 5 — closure

`CLOSURE-WIRE-HEX-FINISH.md` per spec.

## Anti-contamination

- Only worktree `engine-wire-finish`.
- Math-core submodule pinned at WIRE-HEX tip (no math-core changes this sprint).
- The skel binary is a build artifact; does NOT go in git.
- Sub-tag `lat-phase-2-hx-daemon-bench-baseline` applied by operator post-merge.
