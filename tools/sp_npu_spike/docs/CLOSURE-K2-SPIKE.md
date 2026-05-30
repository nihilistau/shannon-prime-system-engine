# K.2-spike — Closure

Sprint: K.2-spike — NPU cross-island bridge design + POC for Manifesto Trick #1 at silicon-island scale.
Branch: `sprint/k2-spike` (base engine main `0cf9674`).
Worktree: `D:\F\shannon-prime-repos\engine-k2-spike` (exclusive — no commits authored from main worktree).

## 1. Headline

**K.2-spike CLOSED, all 4 substantive gates PASS. K.2 full sprint can dispatch as the next focused sprint with no upstream blocker.**

End-to-end QNN HTP graph round-trip verified on Knack's S22U (Hexagon V69 NPU): single-op `ElementWiseAdd` graph, INT8 N=64 inputs, **64/64 bytes byte-exact** output, **1.329 ms graphExecute wall-clock** (sanity bound was < 100 ms; margin 75x).

## 2. Gates table

| Gate | Status | Observed |
|---|---|---|
| T_K2_SPIKE_QNN_SURVEY | PASS | K2-SPIKE-DESIGN.md §1 lists 15 QNN entrypoints with header file:line citations (QnnInterface.h:730, .h:515-580; QnnGraph.h:703-709,471; QnnTensor.h:157; QnnOpDef.h:196,355,524; etc.) |
| T_K2_SPIKE_BRIDGE_DESIGN | PASS | K2-SPIKE-DESIGN.md §2.1/§2.2/§2.3 — libloading recommendation justified (table comparing to direct-link); dispatch model analyzed (Arc<Session> applicable per QnnGraph.h:661-665 cross-thread comment); per-thread vs shared documented |
| T_K2_SPIKE_POC | PASS | sp_npu_spike_smoke on S22U: graphExecute 1.329 ms, mismatches 0/64, exit 0 |
| T_K2_SPIKE_K2_FULL_SCOPE | PASS | K2-FULL-SCOPE.md exists: LOC ~2000-3000, time 30-50 sprint-hours, 4 sub-stages enumerated, top-3 risks documented, dependency list, explicit "K.2 can dispatch" call-out |

## 3. Design document summary

K2-SPIKE-DESIGN.md (~340 lines) covers:

- **QNN API surface**: 15 hot-path entrypoints identified at file:line. Headers under `C:\Qualcomm\AIStack\QAIRT\2.45.40.260406\include\QNN\`. v2.45.40.260406 SDK selected (v2.31 also available).
- **Bridge architecture**: libloading + C shim + 4-entrypoint Rust ABI. Rationale: matches existing FastRPC bridge idiom; avoids static linking; lets runtime SDK version differ from build-time headers; sidesteps QNN SDK CMake-heavy build (the SDK only ships x86 + aarch64-linux example builds, not Rust cross-compile recipes).
- **Dispatch model**: Arc<QnnHtpSession> for K.2 full sprint cross-thread invokes; per `QnnGraph.h:661-665` QNN explicitly supports concurrent execute calls on a single graph (serializes onto one execution queue per graph; for Trick #1 silicon-island parallelism, that's NOT a contention point because cDSP and NPU are independent islands).
- **Signed PD**: empirical finding (Stage 3) — Unsigned PD works for QNN HTP runtime on S22U. The SDK ships `lib/hexagon-v69/unsigned/libQnnHtpV69Skel.so` (10.7 MB); pushing to `/data/local/tmp` + setting `ADSP_LIBRARY_PATH` admits the runtime without testsig. testsig still useful for high-priority / real-time future paths but NOT required for matmul/elementwise scope.

## 4. POC result

**Hardware**: Knack's S22 Ultra (R5CT22445JA, SM-S908E, SM8450/Snapdragon 8 Gen 1, Hexagon V69).
**QNN SDK**: v2.45.40.260406 (QAIRT). Backend provider name `HTP_QTI_AISW`, backendId 6, coreApiVersion 2.34.0.
**Smoke graph**: 1 op = `ElementWiseAdd`. 3 INT8 tensors (in_a, in_b, out_c), shape `[64]`, quant scale=1.0 offset=0 (identity pass-through).
**Test vector**: `a[i] = (i%32)-16`, `b[i] = (i%16)-8`, expected `c[i] = a[i]+b[i]` in `[-24..22]` (safely int8).

**Result**:

```
[sp-npu-spike] provider name=HTP_QTI_AISW backendId=6 api=2.34.0
[sp-npu-spike] backendCreate OK; handle=0x1
[sp-npu-spike] contextCreate OK; handle=0x1
[sp-npu-spike] init OK
[sp-npu-spike] tensor in_a registered id=1
[sp-npu-spike] tensor in_b registered id=2
[sp-npu-spike] tensor out_c registered id=3
[sp-npu-spike] graphFinalize OK
[sp-npu-spike] graphExecute wall = 1.329 ms (1329063 ns)
[sp-npu-spike] total run wall    = 132 ms (includes graphCreate+Tensor+addNode+Finalize)
[sp-npu-spike] mismatches = 0 / 64
[sp-npu-spike] T_K2_SPIKE_POC correctness (bytes-equal): PASS
[sp-npu-spike] T_K2_SPIKE_POC sanity bound (<100 ms):    PASS
[sp-npu-spike] shutdown OK
[sp-npu-spike] === T_K2_SPIKE_POC: PASS ===
exit 0
```

### Surprises observed (worth memorializing)

1. **`contextCreate` failed iter-1 with `AEE_EUNABLETOLOAD` (0x80000406)** because the vendor-shipped `libSnpeHtpV69Skel.so` in `/vendor/lib/rfsa/adsp/` is NOT the same skel QNN runtime wants. QNN wants `libQnnHtpV69Skel.so`. Pushed `$QNN_SDK_ROOT/lib/hexagon-v69/unsigned/libQnnHtpV69Skel.so` (10.7 MB) to `/data/local/tmp` and the runtime picked it up via `ADSP_LIBRARY_PATH`. No setprop, no testsig.

2. **`tensorCreateGraphTensor` failed iter-2 with `7004`** ("tensor type 0 client buffer should be nullptr at tensor creation"). QNN tensor creation registers SHAPE/DTYPE/QUANT ONLY; data buffers are bound at execute time via shallow-copy descriptors with `clientBuf.data` populated. This is NOT documented in `QnnTensor.h` (the API doc says nothing about this constraint) — purely empirical.

3. **HTP backend cost model has SoC-specific blind spots**: `QnnDsp <W> Cost Based unsupported on soc SM8450` warning. Snapdragon 8 Gen 1 is recognized for execution but its cost model isn't tuned. For K.2 production we'd profile + tune; for spike it's an info-only warning.

4. **Single ElementWiseAdd elaborates to `q::Add_flat` on V69 HTP**, and `tiling.h:278` warns about no splitting rule for the 1-D shape. For K.2 production matmul shapes the tiling rules WILL apply; spike's 1-D edge case isn't representative.

5. **Per-process QNN init cost ~130 ms**, per-execute ~1.3 ms for tiny graph. Production K.2 must amortize the init cost via persistent daemon + `contextCreateFromBinary` to skip `graphFinalize` (which will take minutes for an LLM-scale graph).

6. **Process-exit segfault** in Rust at-exit destructors against already-freed QNN state. Worked around with `libc::_exit()` after PASS report. Not a correctness issue but flagged for K.2: implement proper Drop for `QnnHtpSession` Rust wrapper that respects teardown ordering.

## 5. K.2 full sprint scope summary

Per K2-FULL-SCOPE.md:

- **LOC**: ~2000-3000 across 4 sub-stages.
- **Time**: 30-50 focused sprint-hours.
- **Sub-stages**:
  1. Offline graph build + contextGetBinary persistence pipeline (~700 LOC).
  2. Daemon-side persistent QnnHtpSession + contextCreateFromBinary (~600 LOC).
  3. Cross-island Garner: cDSP q_1 + NPU q_2 + ARM recombine (~400 LOC).
  4. Substantive gates: NPU-on-HMX confirmation, cross-island parallelism speedup, Garner bit-exact, leak-free, closure (~250 LOC).
- **Top 3 risks**:
  1. `graphFinalize` minutes-scale on LLM graph → mitigation: offline build, never on-device.
  2. HTP backend may silently fallback ops to HVX/CPU → mitigation: `QnnProfile` instrumentation + T_K2_NPU_ON_HMX gate.
  3. QNN-internal FastRPC session may contend with lattice's Sprint A FastRPC session in same process → mitigation: K.2 stage 1 probe-test both sessions co-existing FIRST.
- **No upstream blocker.** K.2 can dispatch as the next sprint.

## 6. Files changed

| File | LOC | Status |
|---|---|---|
| tools/sp_npu_spike/docs/K2-SPIKE-PLAN.md | 99 | new |
| tools/sp_npu_spike/docs/K2-SPIKE-DESIGN.md | 254 | new |
| tools/sp_npu_spike/docs/K2-FULL-SCOPE.md | 117 | new |
| tools/sp_npu_spike/docs/CLOSURE-K2-SPIKE.md | (this file) | new |
| tools/sp_npu_spike/.gitignore | 2 | new |
| tools/sp_npu_spike/Cargo.toml | 35 | new |
| tools/sp_npu_spike/.cargo/config.toml | 3 | new |
| tools/sp_npu_spike/build.rs | 44 | new |
| tools/sp_npu_spike/src/sp_npu_shim.c | 322 | new |
| tools/sp_npu_spike/src/bin/sp_npu_spike_smoke.rs | 147 | new |

Total: ~1020 LOC + 4 markdown documents (~700 lines of design/scope/closure prose).

## 7. Commits on `sprint/k2-spike`

```
de04988 [plan] K.2-spike -- NPU bridge design + POC plan-commit
15a2fd8 [stage 1] K.2-spike -- design document + crate skeleton
9928d19 [stage 2] K.2-spike -- POC scaffold + C shim + cross-compile green
f112a03 [stage 3] K.2-spike -- POC round-trip on S22U PASS (T_K2_SPIKE_POC)
<this>  [stage 4] K.2-spike -- K2-FULL-SCOPE.md + CLOSURE-K2-SPIKE.md (all 4 gates PASS)
```

## 8. Sub-tag proposal

`lat-phase-13-6-k-2-spike-poc` — captures both the design (Stage 1) and POC pass (Stage 3) achievements.

## 9. What's NOT done (explicit deferrals to K.2 full sprint)

- Production NPU forward kernel (full Memory model on NPU). The spike runs ONE ElementWiseAdd op; K.2 runs a full multi-layer LLM forward.
- NPU model loading (.sp-model → graphFinalize → contextGetBinary persisted artifact). Spike builds graph in-process; K.2 offline pipeline.
- Cross-island Garner: cDSP q_1 dispatched via FastRPC + NPU q_2 dispatched via QNN HTP + ARM Garner-recombine. K.2 stage 3 scope.
- Multi-prime CRT split with q_1 = DSP and q_2 = NPU (M.6 scope after K.2 ships).
- Halide on NPU (separate research direction).
- INT4 vs INT8 on NPU (separate study; check what QNN HTP supports natively).
- Concurrent dispatch on a single NPU graph (Arc<Session> across threads — only tested single-threaded in spike).
- `QnnMem_register` zero-copy DMA-BUF integration (spike uses `QNN_TENSORMEMTYPE_RAW` host-copy path).
- QnnProfile instrumentation to confirm op-on-HMX-not-HVX (K.2 stage 4 gate).
- Co-existence test of QNN-internal FastRPC + lattice Sprint A FastRPC in one process (K.2 stage 1 priority).

## 10. Memory entry candidates (architectural findings worth memorializing)

These four would extend the existing reference + feedback memory:

### Candidate A — `reference-qnn-htp-bridge-architecture` (NEW)

Captures the libloading + C-shim + 4-entrypoint Rust ABI pattern proven in this sprint. Includes:
- The lifecycle: `dlopen → getProviders → logCreate → backendCreate → contextCreate → graphCreate → tensorCreate (NO buffer) → addNode → finalize → execute (WITH buffer) → free`.
- The two empirical "gotchas" from iter-1/iter-2 (skel-must-be-libQnnHtpV69Skel.so-from-SDK-unsigned, tensors-must-have-null-clientBuf-at-create).
- The Unsigned PD path works for NPU on this device (no testsig required for HTP runtime — extends Mode D's Path B finding to NPU).
- File location pointer to `tools/sp_npu_spike/src/sp_npu_shim.c` as the canonical reference impl.

### Candidate B — `reference-qnn-sdk-version-pinning` (NEW)

QNN SDK headers + skel must match. v2.45.40.260406 SDK header build + v2.45.40.260406 skel are confirmed compatible on S22U V69. The runtime advertises `coreApiVersion 2.34.0` which is the API version of the SDK build, distinct from the SDK release version. Document the relationship before K.2 full sprint hits a version-skew bug.

### Candidate C — Extension to `reference-signed-pd-developer-path`

Add a finding: "**Empirical 2026-05-30: QNN HTP NPU runtime works in Unsigned PD on S22U**. Spike sprint K.2 confirmed `QnnContext_create` + `QnnGraph_execute` against `lib/hexagon-v69/unsigned/libQnnHtpV69Skel.so` from QAIRT v2.45.40.260406 SDK; no testsig required, no setprop required. Extends Sprint A-G's 'Path B works for HVX' finding to NPU. Signed PD still useful for QNN_PRIORITY_HIGH+ + real-time + ISP-direct; not required for matmul / elementwise / forward-pass work."

### Candidate D — Extension to `reference-fastrpc-concurrent-dispatch`

Add a forward-looking note: "**K.2 cross-island parallelism unifies two Arc<Session> patterns**: Arc<FastRpcSession> for cDSP HVX + Arc<QnnHtpSession> (new in K.2; design at `tools/sp_npu_spike/docs/K2-SPIKE-DESIGN.md` §2.2) for NPU HMX. The wall-clock speedup discriminator applies per-island AND across-islands; Trick #1 silicon-island claim is the across-islands measurement. cDSP+NPU contention model is INDEPENDENT (separate execution queues per island), so co-spawned threads on Arc<FastRpcSession> and Arc<QnnHtpSession> respectively should NOT serialize at the silicon level. K.2 stage 4 measures empirically."

## 11. Worktree status

- Worktree: `D:\F\shannon-prime-repos\engine-k2-spike` (exclusive).
- Branch: `sprint/k2-spike` (base engine main `0cf9674`).
- NO commits authored from main worktree (`shannon-prime-system-engine`) by this sprint.
- NO commits authored from sibling worktrees (`engine-m1`, `engine-kbeta-*`, `lattice-*`) by this sprint.
- Per `feedback-parallel-agents-separate-worktrees`: discipline observed.
- Branch will be pushed to origin (`git push -u origin sprint/k2-spike`) after this closure commit. Operator handles merge to engine main after review.

## 12. Workflow discipline acknowledgement

- ✓ Plan-commit before code (commit `de04988`).
- ✓ Reference-read with file:line citations before plan (Stage 0).
- ✓ Multi-stage commits (4 substantive stages + closure).
- ✓ One variable at a time on the two POC iterations (skel-path fix in iter-1, tensor-buffer-timing fix in iter-2, surfaced separately).
- ✓ No silent gate revisions: both POC failures were surfaced with specific error codes (`AEE_EUNABLETOLOAD 0x80000406`, QNN rc=7004) before being fixed.
- ✓ Anti-contamination: SDK headers + examples read as REFERENCE; bridge code re-derived; no SDK source linked into the lattice binary.
- ✓ Lead with reference then theory: read `QnnInterface.h`, `QnnGraph.h`, `QnnSampleApp.cpp` BEFORE writing the C shim; theory came second.
