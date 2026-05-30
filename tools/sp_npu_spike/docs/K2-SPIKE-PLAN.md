# K.2-spike — Plan-commit

Sprint: K.2-spike — NPU cross-island bridge design + POC for Manifesto Trick #1 at silicon-island scale.
Worktree: `D:\F\shannon-prime-repos\engine-k2-spike` (branch `sprint/k2-spike`, base `0cf9674`).

This is an INVESTIGATION sprint. Deliverable = design doc + small POC that round-trips an INT8 buffer through QNN HTP on Knack's S22U. NOT a production NPU forward kernel.

## Stage 0 — Reference reading completed (file:line citations)

1. **`reference-qualcomm-sdk-inventory`** (memory file at `spaces/.../memory/reference_qualcomm_sdk_inventory.md:33-34`).
   - AIStack path = `C:\Qualcomm\AIStack`. On disk it is actually `C:\Qualcomm\AIStack\QAIRT\2.45.40.260406\` (QAIRT = Qualcomm AI Runtime, the renamed QNN SDK; v2.45.40.260406 + v2.31.0.250130 both installed; pick v2.45 as newest).
   - QNN headers under `C:\Qualcomm\AIStack\QAIRT\2.45.40.260406\include\QNN\` — verified contains `QnnInterface.h`, `QnnBackend.h`, `QnnContext.h`, `QnnGraph.h`, `QnnDevice.h`, `QnnLog.h`, `QnnMem.h`, `QnnTensor.h`, `QnnTypes.h`, `QnnError.h`, plus `HTP/QnnHtpDevice.h`, `HTP/QnnHtpGraph.h`, etc.
   - Runtime libs on host at `lib/aarch64-android/`: `libQnnHtp.so` (top-level HTP), `libQnnHtpV69Stub.so` (V69-arch stub), `libQnnHtpPrepare.so` (graph prepare), `libQnnSystem.so` (system context), plus `libQnnHtpV69Skel.so` (DSP-side skel — already present on device).

2. **QNN SDK headers**. The API surface the bridge will wrap (cite headers):
   - **`QnnInterface.h:730`** — `QnnInterface_getProviders(QnnInterface_t***, uint32_t*)`. Entry point on every backend `.so`; returns an array of `QnnInterface_t` structs each containing a typed dispatch table of function pointers (the SDK's variant of "vtable for the backend").
   - **`QnnInterface.h:515-580`** — `QnnInterface_ImplementationV2_x_t` struct (the dispatch table): `backendCreate`, `backendFree`, `contextCreate`, `contextFree`, `graphCreate`, `graphAddNode`, `graphFinalize`, `graphExecute`, `graphExecuteAsync`, `tensorCreateGraphTensor`, `logCreate`, `deviceCreate`, `memRegister`, etc. ~50 entries.
   - **`QnnGraph.h:703-709`** — `QnnGraph_execute(graphHandle, inputs[], numInputs, outputs[], numOutputs, profileHandle, signalHandle)`. The hot-path dispatch we'll measure.
   - **`QnnGraph.h:471`** — `QnnGraph_addNode(graphHandle, opConfig)`. Op-by-op graph construction.
   - **`QnnTensor.h:157`** — `QnnTensor_createGraphTensor(graph, tensor)`.
   - **`QnnOpDef.h:196,355,524,617`** — op-name string constants. For POC we use `QNN_OP_ELEMENT_WISE_ADD` (string `"ElementWiseAdd"`) — the smallest viable op that proves data-flow round-trip.
   - **`QnnSampleApp.cpp:212-222`** — backendCreate call shape (`logHandle, backendConfig[], &backendHandle`).
   - **`QnnSampleApp.cpp:313-323`** — `contextCreate(backendHandle, deviceHandle, contextConfig[], &context)`.
   - **`main.cpp:457-528`** — full lifecycle sequence: `getProviders → backendCreate → (deviceCreate) → contextCreate → composeGraphs (addNode * N) → finalizeGraphs → executeGraphs → freeContext → freeDevice → terminateBackend`.

3. **`reference-mode-d-bridge-architecture`** (memory file). The FastRPC pattern at `tools/sp_dsp_smoke/src/dsp_rpc.rs:128-300`:
   - `FastRpcSession::new(URI)` → libloading `Library::new("libcdsprpc.so")` → resolve `remote_session_control`, `remote_handle64_{open,invoke,close}`, `rpcmem_alloc`, `rpcmem_free` → call `remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE)` → `remote_handle64_open(URI, &mut handle)`. Handle is u64.
   - **QNN bridge analog**: same libloading idiom, but the lifecycle is more layered. We open `libQnnHtp.so` once, call `QnnInterface_getProviders` to fetch a dispatch table struct, then store the resolved fn pointers behind a Rust struct. Multiple resource handles (backend, device, context, graph) rather than a single u64.
   - **Key difference**: QNN has a 4-level handle hierarchy (Backend → Device → Context → Graph) vs FastRPC's flat URI-keyed session. Each level has independent lifetime + config struct.

4. **`reference-signed-pd-developer-path`** (memory file). Signed PD requirement for NPU access:
   - HTP runs in its own protection domain on the cDSP silicon. QNN HTP backend invokes `libQnnHtpV69Skel.so` via FastRPC under the hood (the QNN .so is the host-side facade; DSP-side execution still goes through the FastRPC bridge).
   - Empirical evidence on Knack's S22U: `libQnnHtpV69Skel.so` is ALREADY DEPLOYED in `/vendor/lib/rfsa/adsp/` by Samsung (verified via `adb shell ls`). This means HTP runtime is loadable WITHOUT pushing a custom signed skel — the vendor-shipped skel is already authorized. We test this empirically in Stage 2.
   - `getprop vendor.fastrpc.process.attrs` returns empty (not `0x8` = anti-flag clear). Signed PD not force-disabled.
   - **Risk**: If `QnnBackend_create` fails on the V69 stub with `QNN_BACKEND_ERROR_NOT_SUPPORTED` or signature-related code, that's the upstream blocker per `feedback-no-silent-gate-revisions`. We SURFACE it, not stub a fake POC.

5. **`reference-fastrpc-concurrent-dispatch`** (memory file). Arc<Session> pattern; auto-Send+Sync via libloading bare-fn pointers + u64 handle:
   - **Question for K.2**: Is `QnnContext_t` + `Qnn_GraphHandle_t` similarly thread-safe across concurrent `QnnGraph_execute` calls?
   - **Per `QnnGraph.h:661-665`** — comment states "If there are simultaneous calls to QnnGraph_execute() and QnnGraph_executeAsync(), the execution will be added to the end of the queue, and QnnGraph_execute() will block while...". So QNN explicitly DOES support concurrent dispatch from multiple host threads on a single graph handle, but serializes them onto a single execution queue per graph. To get TRUE NPU parallelism we'd need either (a) two separate graphs in the same context, OR (b) `QnnGraph_executeAsync` from one thread + work on ARM in parallel.
   - **Implication for K.2 full sprint**: Trick #1 split across cDSP (q_1) + NPU (q_2) maps naturally because cDSP and NPU are SEPARATE silicon islands; the contention model is on each side independent. No single-island serialization.

6. **`reference-v69-hvx-expert-practices`** (memory file). NPU vs HVX:
   - HVX (cDSP vector pipe) = SIMD u8/u16/u32 lanes with VLIW 4-way packets, 1024-bit VRF.
   - NPU = HMX (Hexagon Matrix eXtension) tensor cores on V69+, primarily targeted at INT8 / INT16 quantized matmul + activations. **Different ISA**; you don't program it in HVX intrinsics. The QNN backend hides this — you build an op graph and let the HTP scheduler emit HMX instructions.
   - For POC we don't need to expose HMX directly; the smoke pushes an `ElementWiseAdd` op into the graph and lets QNN figure out where it runs. HTP backend executes ON the NPU silicon by default (with HVX fallback for ops the NPU doesn't support).

7. **K.beta.2.5c dual-dispatch harness** — `tools/sp_dsp_smoke/src/sp_matmul_q_dual_smoke.rs:43-216`. Pattern reference for the POC: `Arc<Session>::clone()` → spawn thread → `sess.invoke(...)` → `join`. The K.2-spike POC is single-thread (we're only proving the round-trip works); concurrent dispatch graduates to K.2 full sprint.

8. **QNN HTP example code** — `C:\Qualcomm\AIStack\QAIRT\2.45.40.260406\examples\QNN\SampleApp\SampleApp\src\` (`main.cpp`, `QnnSampleApp.cpp`, headers/utility files). `QnnSampleApp.cpp:212-509` is the canonical lifecycle. For K.2-spike POC we follow the lifecycle but with the SIMPLEST possible graph: one `ElementWiseAdd` node with two 1-D INT8 input tensors and one 1-D INT8 output tensor.

## Hardware availability check — PASS

- `adb devices` → `R5CT22445JA device` (Knack's S22U).
- `ro.product.model` = `SM-S908E`, `ro.soc.model` = `SM8450` = Snapdragon 8 Gen 1, **Hexagon V69 NPU** (NOT V79 — the prompt mentioned V79 but the S22U SoC is V69; corrected here).
- `vendor.fastrpc.process.attrs` = empty (anti-flag clear; Signed PD admission allowed).
- `/vendor/lib64/libcdsprpc.so` present.
- `/vendor/lib/rfsa/adsp/libSnpeHtpV69Skel.so` present (vendor-shipped HTP skel).
- Host has QAIRT v2.45.40.260406 installed with full V69 lib set.
- Rust target `aarch64-linux-android` installed.

## Architecture summary (chosen, justified in K2-SPIKE-DESIGN.md)

- **Bridge: libloading + thin Rust shim** (NOT direct linking). Rationale: matches the existing FastRPC bridge idiom + lets the QNN HTP runtime be located via `ADSP_LIBRARY_PATH` at runtime + sidesteps the QNN SDK's CMake-heavy build (the SDK only ships Win/Linux example builds; we want a Rust binary built via `cargo --target aarch64-linux-android`).
- **Dispatch handle: `QnnHtpSession` struct** containing the resolved fn-pointer table + Backend/Context/Graph handles. Wrap in `Arc` for K.2 full sprint (per fastrpc-concurrent-dispatch). For K.2-spike POC, single-thread is fine.
- **Memory model**: For spike, allocate input/output buffers as plain Vec<u8>; copy in/out via `Qnn_Tensor_t.dataBuffer` field. Production K.2 will use `QnnMem_register` with RPC-mem buffers (zero-copy with the underlying DMA-BUF) — out of spike scope, called out in K2-FULL-SCOPE.md.

## Stage plan (commits)

- **Stage 0 (THIS commit)**: plan-commit. Hardware-availability check + reference reads + chosen architecture + stage plan.
- **Stage 1**: Design document `K2-SPIKE-DESIGN.md` — completed BEFORE any code per workflow discipline. Cargo workspace member skeleton for `sp_npu_spike` crate.
- **Stage 2**: POC scaffold + initialization — Rust binary `sp_npu_spike_smoke.rs` that opens `libQnnHtp.so`, calls `QnnInterface_getProviders`, calls `QnnBackend_create`, calls `QnnContext_create`. Builds + cross-compiles. Push QNN libs to device via adb. Run on device; expect either (a) backend create returns 0 (proceed Stage 3) or (b) specific error code (surface UPSTREAM).
- **Stage 3**: POC round-trip — extend the binary with `QnnGraph_create`, `QnnTensor_createGraphTensor` x3 (two inputs + one output, INT8 1-D shape `[N]` with N=64), `QnnGraph_addNode` with `QNN_OP_ELEMENT_WISE_ADD`, `QnnGraph_finalize`, `QnnGraph_execute`. Inputs = known constants (e.g., `a[i]=i, b[i]=2i`). Verify output `c[i] == a[i] + b[i]` element-wise. Bound wall-clock < 100 ms (sanity). T_K2_SPIKE_POC.
- **Stage 4**: K.2 full scope estimate `K2-FULL-SCOPE.md` + closure `CLOSURE-K2-SPIKE.md`. Push branch.

## What's NOT being done in this spike (deferred to K.2 full)

- Production NPU forward kernel (Memory model on NPU)
- NPU model loading (`.sp-model` variant; QNN context binary serialization)
- Cross-island Garner across NPU + DSP (M.6 scope)
- Multi-prime CRT split where q_1 = DSP, q_2 = NPU
- Halide on NPU
- INT4 vs INT8 on NPU
- Concurrent dispatch on NPU graph(s)
- `QnnMem_register` zero-copy DMA-BUF integration

## Risk register (initial — refined in CLOSURE)

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| `QnnBackend_create` on `libQnnHtpV69Stub.so` fails on Samsung-shipped device | Medium | High | Surface UPSTREAM; document exact error code; investigate whether pushing host-shipped V69 stub overrides the vendor one. |
| QNN runtime requires deployment of files we can't push without root | Medium | Medium | Empirically test `/data/local/tmp` deployment + `LD_LIBRARY_PATH`/`ADSP_LIBRARY_PATH` workarounds first; if root needed, surface. |
| `Qnn_Tensor_t` struct layout requires C bitfield reproduction that differs across SDK versions | Low | Medium | Pin to v2.45.40.260406 only; document version dependence in design doc. |
| HTP backend silently routes to CPU/HVX fallback rather than HMX | Low | Low for spike (correctness only) | Surface as note for K.2 full sprint; not a spike gate. |

## Anti-contamination acknowledgement

- All new code lives under `tools/sp_npu_spike/` in THIS worktree (`engine-k2-spike`).
- I do not touch `shannon-prime-system-engine` (main), `engine-m1`, `engine-kbeta-*`, `lattice-*`, or any prior-cohort repo.
- QNN SDK headers + examples read as REFERENCE; code re-derived per anti-contamination rule.
