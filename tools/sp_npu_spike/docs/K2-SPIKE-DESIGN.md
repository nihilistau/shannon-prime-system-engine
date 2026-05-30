# K.2-spike — NPU Cross-Island Bridge Design

Sprint: K.2-spike (investigation, not production).
Document version: v1 (Stage 1 of 4).
Companion: `K2-SPIKE-PLAN.md` (plan-commit), `K2-FULL-SCOPE.md` (Stage 4), `CLOSURE-K2-SPIKE.md` (Stage 4).

## 1. QNN API surface summary

The QNN SDK exposes a multi-component C ABI behind a discoverable interface struct. The lattice K.2 bridge wraps roughly 12-15 hot-path entrypoints; the broader SDK surface is ~50 functions but most are out-of-spike (op-package registration, profiling internals, async-execution callbacks, etc.).

### 1.1 Headers we depend on (paths under `C:\Qualcomm\AIStack\QAIRT\2.45.40.260406\include\QNN\`)

| Header | Purpose |
|---|---|
| `QnnInterface.h` | Top-level entry: `QnnInterface_getProviders` + dispatch-table struct `QnnInterface_t` |
| `QnnTypes.h` | Common types: `Qnn_ErrorHandle_t`, `Qnn_BackendHandle_t`, `Qnn_ContextHandle_t`, `Qnn_GraphHandle_t`, `Qnn_DeviceHandle_t`, `Qnn_Tensor_t`, `Qnn_OpConfig_t`, `Qnn_DataType_t`, `Qnn_TensorType_t` |
| `QnnCommon.h` | `QNN_SUCCESS`, version macros |
| `QnnError.h` | Error code namespace constants |
| `QnnBackend.h` | `QnnBackend_Config_t`; `backendCreate`/`backendFree` declarations |
| `QnnDevice.h` | `QnnDevice_Config_t`; `deviceCreate`/`deviceFree` |
| `QnnContext.h` | `QnnContext_Config_t`; `contextCreate`/`contextFree`/`contextGetBinary` |
| `QnnGraph.h` | `QnnGraph_Config_t`; `graphCreate`/`graphAddNode`/`graphFinalize`/`graphExecute` |
| `QnnTensor.h` | `QnnTensor_createGraphTensor` (per-graph tensor registration) |
| `QnnOpDef.h` | Op-name string constants (`QNN_OP_ELEMENT_WISE_ADD`, `QNN_OP_MAT_MUL`, etc.) |
| `QnnLog.h` | `QnnLog_create` (callback-based logger; required first call before backendCreate) |
| `QnnMem.h` | `QnnMem_register` for shared/zero-copy memory (out of spike, in K.2 full scope) |
| `HTP/QnnHtpDevice.h` | HTP-specific device config (perf mode, soc model) |
| `HTP/QnnHtpGraph.h` | HTP-specific graph config (precision, optimization level) |

### 1.2 The dispatch table

`QnnInterface.h:515-580` defines `QnnInterface_ImplementationV2_x_t` as a struct of ~50 function pointers. Each backend `.so` provides one. The bridge accesses functions via `intf->v2_x.contextCreate(...)` rather than dlsym for each — only `QnnInterface_getProviders` is dlsym'd directly.

The 15 hot-path entries the bridge wraps for spike + K.2 full:

```
propertyHasCapability         // capability probe
backendCreate                 // create backend handle
backendFree                   // teardown
contextCreate                 // create context (per device)
contextFree
contextGetBinary              // serialize compiled graph (K.2 full only)
contextCreateFromBinary       // deserialize compiled graph (K.2 full only)
graphCreate                   // create empty graph in context
graphAddNode                  // add op to graph
graphFinalize                 // compile graph (NPU kernel selection happens here)
graphExecute                  // synchronous dispatch (POC + K.2 main path)
graphExecuteAsync             // asynchronous dispatch (K.2 concurrency)
tensorCreateGraphTensor       // register a tensor in a graph
logCreate                     // mandatory first call (logger handle threads through everything)
deviceCreate                  // create device handle (HTP-specific config attached)
memRegister                   // register external buffer (DMA-BUF / RPC-mem) for zero-copy
```

### 1.3 Lifecycle (canonical sequence from `SampleApp/src/main.cpp:457-528`)

```
dlopen(libQnnHtp.so)
QnnInterface_getProviders(&providers, &n)
intf = providers[0]                          // single provider per .so
intf->v2_x.logCreate(...)                    // mandatory
intf->v2_x.backendCreate(logHandle, cfg, &backend)
intf->v2_x.deviceCreate(logHandle, cfg, &device)
intf->v2_x.contextCreate(backend, device, cfg, &ctx)
intf->v2_x.graphCreate(ctx, "graph_name", cfg, &graph)
// register tensors:
intf->v2_x.tensorCreateGraphTensor(graph, &in_tensor_a)
intf->v2_x.tensorCreateGraphTensor(graph, &in_tensor_b)
intf->v2_x.tensorCreateGraphTensor(graph, &out_tensor_c)
// add ops:
intf->v2_x.graphAddNode(graph, opConfig)     // ElementWiseAdd
intf->v2_x.graphFinalize(graph, profile, NULL)
intf->v2_x.graphExecute(graph, in_tensors[], 2, out_tensors[], 1, profile, signal)
// teardown:
intf->v2_x.contextFree(ctx, NULL)
intf->v2_x.deviceFree(device)
intf->v2_x.backendFree(backend)
intf->v2_x.logFree(logHandle)
dlclose(libQnnHtp.so)
```

The K.2-spike POC implements this entire sequence at minimal complexity (single graph, single op, single execute).

## 2. Bridge architecture proposal

### 2.1 Libloading vs direct linking

**Recommendation: libloading.**

| Aspect | libloading | direct linking |
|---|---|---|
| Build system | Rust-only (`cargo --target aarch64-linux-android`) | Requires QNN SDK CMake setup + cross-compile linker discovery |
| Library version coupling | Runtime — pick which SDK version per-deployment | Build-time — SDK version baked into binary |
| Symbol resolution failure mode | `Err(libloading::Error)` returned at session-open time, recoverable | Static linker error at build time; or `dlopen` failure at process start (un-recoverable) |
| Match with existing lattice idiom | Identical to `tools/sp_dsp_smoke/src/dsp_rpc.rs:140-298` FastRPC bridge | New idiom — inconsistent with already-shipped pattern |
| Anti-contamination | Headers are read-only reference; no SDK source linked into lattice binary | Forces shipping SDK headers into the lattice tree as build deps |
| ABI risk | Forward-compatible (load v2.46 SDK against binary built knowing v2.45 ABI, as long as struct layouts match) | Binary tied to SDK build version |

The bridge re-uses Rust idioms that worked in Sprint A: open the lib once at `QnnHtpSession::new()`, resolve `QnnInterface_getProviders` via `lib.get(b"QnnInterface_getProviders")`, then EVERY OTHER function pointer is fetched from the returned dispatch struct (one dlsym per process lifetime).

### 2.2 Rust shim: `QnnHtpSession`

```rust
pub struct QnnHtpSession {
    _lib: libloading::Library,            // keeps libQnnHtp.so loaded for session lifetime
    intf: *const QnnInterface,            // dispatch table pointer (lives inside the lib)
    log_handle: Qnn_LogHandle_t,
    backend_handle: Qnn_BackendHandle_t,
    device_handle: Qnn_DeviceHandle_t,    // None if device not supported
    context_handle: Qnn_ContextHandle_t,
}
```

Constructor `QnnHtpSession::new()` performs lifecycle 1-5 (dlopen → getProviders → logCreate → backendCreate → deviceCreate → contextCreate). `Drop` performs lifecycle reverse.

A `QnnGraph` substruct wraps lifecycle 6-9 (graphCreate → addNode * N → finalize → execute). Multiple graphs can live within one context.

### 2.3 Per-thread context vs shared `Arc<Session>`

**For graph-execute concurrency: shared `Arc<QnnHtpSession>` is correct, but with a caveat.**

Evidence from `QnnGraph.h:661-665`:

> "If there are simultaneous calls to QnnGraph_execute() and QnnGraph_executeAsync(), the execution will be added to the end of the queue, and QnnGraph_execute() will block while..."

This tells us: QNN explicitly supports concurrent calls on a single graph handle from multiple host threads. They are SERIALIZED at execution per graph (single execution queue per graph) but the host-side `QnnGraph_execute` is thread-safe.

**For Trick #1 (CRT split across silicon islands)**: we don't want serialization on a single NPU graph — we want true NPU+DSP parallelism. K.2 full design:

1. cDSP q_1 dispatched via FastRPC (existing Arc<FastRpcSession> Sprint A pattern).
2. NPU q_2 dispatched via QNN HTP (Arc<QnnHtpSession>).
3. ARM Garner-recombine.

Because cDSP and NPU are SEPARATE silicon islands with INDEPENDENT execution queues, the two thread invocations don't contend. The Sprint K v0.alpha 1.935x speedup at cDSP-internal dual-context applied within ONE silicon island; K.2 measures across TWO islands which has a different ceiling (potentially better — full silicon parallelism).

**For graph composition concurrency** (multiple threads building multiple graphs in the same context): the SDK does not document graphAddNode thread-safety; we serialize graph composition behind a `Mutex` per context and only parallelize at execute time. Out of spike scope; flagged in K.2 full doc.

### 2.4 Initialization cost vs per-invoke cost

From SampleApp profiling notes + QNN SDK docs:

- **`backendCreate` + `deviceCreate`**: one-time, hundreds of milliseconds to seconds (driver init, DSP image authentication).
- **`contextCreate`**: tens of milliseconds (VTCM reservation, context-binary setup).
- **`graphAddNode` x N**: linear in N; for small graphs negligible, for full LLM (thousands of nodes) seconds.
- **`graphFinalize`**: the heavy compile step — minutes for large graphs, milliseconds for tiny graphs. This is where the NPU op-fusion + HMX kernel selection happens.
- **`graphExecute`**: hot path. Microseconds to tens of milliseconds depending on shape.
- **`contextGetBinary` + persisted-binary path**: avoids re-running graphFinalize on every process start; for production K.2 we serialize compiled graphs as part of `.sp-model` build pipeline, deserialize at daemon start.

**Implication for K.2 full sprint architecture**: graphFinalize is too heavy to do at every daemon start for an LLM-scale graph. We MUST use the persisted-binary path: build-time graphFinalize → cache `.qnn_bin` → daemon-side `contextCreateFromBinary` at startup. This is well-trodden ground in Qualcomm SDK guides (the Genie examples ship pre-finalized binaries). Spike POC bypasses this because the graph is one op.

## 3. Signed PD vs Unsigned PD for NPU access

### 3.1 Where HTP execution actually happens

The QNN HTP "backend" name suggests user-mode execution, but `libQnnHtp.so` is a host-side facade that internally invokes `libQnnHtpV69Skel.so` (or V73/V75/V79 skel for newer parts) via FastRPC on the cDSP silicon. The NPU (HMX tensor cores) are physically part of the cDSP package on V69; the QNN HTP backend programs them via a specialized DSP-side runtime.

This means the FastRPC layer + Signed PD discipline from `reference-mode-d-bridge-architecture` applies underneath QNN. We don't program the FastRPC layer directly; QNN does, but if the underlying skel signature fails, QNN backendCreate fails.

### 3.2 Empirical state on Knack's S22U

Verified at Stage 0:

- `/vendor/lib/rfsa/adsp/libSnpeHtpV69Skel.so` is present (Samsung-vendor-shipped V69 HTP skel).
- `vendor.fastrpc.process.attrs` = empty (no `0x8` anti-flag; Signed PD admission allowed).
- `testsig` NOT installed (same state as Sprint A; Mode D ran Path B / Unsigned PD).

**Hypothesis**: because Samsung's vendor skel is already signed + present in the privileged `/vendor/lib/rfsa/adsp/` path, QNN HTP backend can load against it WITHOUT a custom testsig. The vendor pre-signed all the production HTP runtime libs as part of the device firmware; we're using their signed skel + our user-mode QNN host facade. If true, this means K.2 has full Signed-PD access via the vendor path — no developer testsig needed, no signing toolchain dance.

**This will be empirically confirmed or refuted in Stage 2 of the POC**:
- If `QnnBackend_create` returns 0 (success), the vendor-skel hypothesis holds.
- If it returns a signature-related error, we fall back to either (a) push the SDK's own V69Skel + sign with developer testsig, OR (b) document the blocker + escalate to operator.

### 3.3 Path B (DSPRPC_CONTROL_UNSIGNED_MODULE) — applicable here?

QNN HTP runtime does its own FastRPC session setup internally; we do NOT set `DSPRPC_CONTROL_UNSIGNED_MODULE` ourselves. If QNN's internal FastRPC session needs that flag, the QNN backend setup handles it (and may or may not document that it does).

In the unlikely case QNN HTP requires Path B and we hit a signature-mismatch error, the workaround would be to set `vendor.fastrpc.process.attrs=0x8` via `adb shell setprop`. **Per `reference-signed-pd-developer-path`, this is the anti-flag in production**, but for spike-investigation only it's an acceptable diagnostic. We do NOT set it preemptively; only if the empirical evidence demands it.

## 4. Hardware compute units the K.2 sprint targets

V69 (Snapdragon 8 Gen 1) silicon islands:

| Unit | ISA | Role for lattice |
|---|---|---|
| ARM Cortex-X2 / A710 / A510 | AArch64 | Host marshalling, Garner CRT recombine, sieve scheduling |
| Hexagon V69 scalar (4 threads) | Hexagon | Cross-context coordination, KV cache addressing, IPC marshalling on DSP side |
| Hexagon V69 HVX (2 vector contexts) | HVX SIMD 1024-bit VRF | Mode D matmul (currently shipped), Barrett-mod-q kernel (K.beta.2.5b) |
| Hexagon V69 HMX (NPU tensor cores) | HMX MAC tile | **K.2 target** — fixed-shape INT8/INT16 matmul/conv with hardware tiling |
| Adreno 730 GPU | OpenCL/Vulkan | Not in K.2; future Phase 4 |
| ISP (image signal processor) | Fixed-function | Not in K.2; Sprint L (KSTE Tier-0) |

K.2 targets the HMX (NPU) specifically. The HMX is programmed via QNN HTP (no direct hand-written HMX intrinsics in this generation; QNN's HTP graph compiler maps ops to HMX kernels).

For the spike we don't NEED to verify HMX engagement — `ElementWiseAdd` might be CPU-fallback inside the HTP backend (small unaligned shape). For K.2 full sprint we instrument via QNN profiling to confirm the matmul ops route to HMX.

## 5. Memory model

### 5.1 Spike POC (this sprint)

- Allocate `Vec<i8>` for input A, input B, output C.
- Set `Qnn_Tensor_t.memType = QNN_TENSORMEMTYPE_RAW`.
- Set `Qnn_Tensor_t.clientBuf.data = vec.as_mut_ptr()`, `clientBuf.dataSize = vec.len()`.
- QNN driver copies host buffer → DSP-shared memory at execute time, copies back on completion.

This is the slowest path (two extra copies) but simplest. Sufficient for round-trip verification.

### 5.2 K.2 full sprint (out of spike)

- Use `QnnMem_register` to register an `rpcmem_alloc(HEAP_ID_SYSTEM, RPCMEM_TRY_MAP_STATIC, size)` buffer (same allocator as Sprint B DmaBuffer).
- Tensor uses `memType = QNN_TENSORMEMTYPE_MEMHANDLE`.
- Zero-copy DSP-side access; matches `reference-zero-copy-invariant` discipline.
- Requires `libcdsprpc.so` loaded into the same process (already true: QNN loads it transitively).

### 5.3 NPU memory budget

V69 HTP runtime reserves a fraction of VTCM (typically 2-4 MB out of 8 MB pool) for its internal staging. The remaining VTCM is available to other DSP processes. For coexistence with the Mode D matmul kernel that also uses VTCM (Sprint G recipe), the K.2 full sprint will need to coordinate VTCM partitioning — but that's well past spike scope.

## 6. Open questions for K.2 full sprint (deferred from spike)

1. Does `QnnContext_createFromBinary` reuse a single context-binary across multiple processes, or does each daemon process need its own copy? Affects multi-daemon deployment.
2. Can two graphs in the same context run concurrently (one per host thread) WITHOUT serialization at the HTP scheduler level?
3. What's the cost of `QnnGraph_finalize` on an LLM-scale graph (28 layers x Qwen3-0.6B)? Estimates from Qualcomm Genie suggest seconds to minutes.
4. Does QNN HTP support Q4 (4-bit weights) natively, or only Q8/INT16/FP16? Affects K.2 quantization parity with cDSP path.
5. Does `QnnGraph_executeAsync` actually engage NPU parallelism, or just defer the sync version onto a worker thread inside QNN?

## 7. Composition with existing lattice patterns

- **Manifesto Trick #1 silicon-island scale**: cDSP (Arc<FastRpcSession> per `reference-fastrpc-concurrent-dispatch`) + NPU (Arc<QnnHtpSession>) + ARM Garner. Per-island contention model independent. Wall-clock speedup discriminator per Sprint K v0.alpha rule applies separately to each island AND to the cross-island case.
- **Anti-contamination**: QNN SDK headers + examples read as REFERENCE; bridge code re-derived in `tools/sp_npu_spike/src/`. No SDK source linked into lattice binary.
- **Discrete substrate**: K.2 production work passes Q8 mod-q residues through NPU as INT8 tensors. The "fp NPU" framing is a deliberately avoided drift — the lattice math stays in Z_q discrete; NPU is just a different silicon dispatch.
- **Lead with reference, then theory** (per memory feedback entry): every K.2 sub-design that touches QNN cites a specific QnnXxx.h section AND a specific SampleApp source location.
