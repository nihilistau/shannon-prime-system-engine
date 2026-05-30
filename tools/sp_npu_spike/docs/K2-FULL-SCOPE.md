# K.2 Full Sprint — Scope Estimate

Sprint: K.2 (production NPU forward kernel) — NOT THIS SPRINT. Estimate produced by K.2-spike.
Document version: v1 (Stage 4 of K.2-spike).

## Headline

**K.2 can dispatch as the next focused sprint.** No blocking upstream dependency. K.2-spike has empirically resolved the three pre-K.2 unknowns (API surface, dispatch model, hardware availability), proven the round-trip end-to-end on Knack's S22U, and produced a working libloading + C shim bridge pattern that K.2 can extend.

The full K.2 sprint will land in 4 sub-stages with a forecast of ~2000-3000 LOC additions across the engine + lattice tree. Estimated focused sprint time: 30-50 sprint-hours (across multiple agent dispatches per workflow discipline).

## LOC + complexity estimate

| Sub-stage | Description | LOC (rough) | Time (sprint-hours) |
|---|---|---|---|
| K.2 stage 1 | Persisted-graph build pipeline: offline tool that constructs a multi-op QNN graph from .sp-model, runs graphFinalize, calls contextGetBinary, writes `<model>.qnn_v69_htp.bin` artifact alongside .sp-model. | 600-800 (Rust tool + shim extension) | 8-12 |
| K.2 stage 2 | Daemon-side QnnHtpSession load path: contextCreateFromBinary at startup, persistent graph handle, hot-path graphExecute. Arc<QnnHtpSession> per silicon-island. | 500-700 (Rust + shim) | 8-12 |
| K.2 stage 3 | Cross-island Garner: q_1 dispatched via existing Arc<FastRpcSession> (DSP HVX), q_2 dispatched via Arc<QnnHtpSession> (NPU HMX), ARM-side Garner recombine using sp_matmul_q_ref::garner_combine_q1_q2 (existing from K.beta.2.5c). One spawn-join cycle per matmul. | 300-500 (Rust orchestration + new smoke harness) | 6-10 |
| K.2 stage 4 | Substantive gates: cross-island parallelism speedup (wall-clock 2x target per cDSP+NPU vs sequential), Garner bit-exact, leak-free. Closure + tag. | 200-300 (gates + closure) | 6-10 |
| **Total** | | **~1600-2300 LOC** | **28-44 hours** |

Doubling for the unknowns that emerge during implementation: **2000-3000 LOC / 30-50 sprint-hours** is the honest envelope.

## Dependencies on other sprints

**Hard dependencies (K.2 cannot ship without these)**:

1. **M.0-real Memory model on cDSP** (lattice-memo-m0 lane). The K.2 cross-island split presumes a baseline cDSP path exists for q_1; otherwise there's nothing to split. M.0-real may already be in flight as Memory model on the existing Sprint J full-loader pattern. Verify status before K.2 dispatch.

2. **K.beta.2.5c CRT NTT toolkit** — DONE (engine main 0cf9674). Garner-recombine + dual-prime fixtures are in place; K.2 reuses `sp_matmul_q_ref::garner_combine_q1_q2`.

**Soft dependencies (K.2 can ship without, but better if these land first)**:

3. **Sprint J full-loader on cDSP** — DONE per `feedback-parallel-agents-separate-worktrees` history. K.2 daemon-side persistent QnnHtpSession follows the same `sp_daemon`/AppState pattern.

4. **`.sp-model` arch_struct convergence (engine vs math-core)** per `project-arch-struct-divergence`. Affects whether the K.2 offline tool needs to consume two .sp-model variants or one. Could close after K.2 if needed; not a blocker.

5. **Memory entry `reference-qnn-htp-bridge-architecture`** (this sprint produces it; first 30 sprint-hours of K.2 cite it). Drafted as part of CLOSURE-K2-SPIKE.md memory-candidates.

**No dependencies on**:
- Signed PD developer testsig setup (K.2-spike confirmed Unsigned PD works for NPU).
- Adreno/clml-sdk (separate Phase 4+ scope).
- ISP/KSTE Tier-0 (Sprint L, parallel lane).

## Risk register (top 3)

### Risk 1 — `graphFinalize` takes minutes on LLM-scale graph

**Likelihood: High.** Qualcomm Genie examples (`C:\Qualcomm\AIStack\QAIRT\2.45.40.260406\examples\Genie\configs\llama3-3b\`) all ship with pre-finalized binaries because finalize is too slow at startup. For a Memory model with 28 layers (Qwen3-0.6B class), finalize on the host build machine will likely take 10-30+ minutes per .sp-model variant.

**Impact: High** if K.2 stage 1 tries to do finalize on-device. **Low** if K.2 stage 1 builds the persisted-binary offline correctly the first time.

**Mitigation**: K.2 stage 1's deliverable is the offline tool; the daemon NEVER finalizes. CI pipeline runs the offline tool once per checkpoint, checks in the `.qnn_v69_htp.bin` artifact (~50-200 MB per model) alongside .sp-model. K.2 spec must explicitly mandate "no on-device graphFinalize on hot path".

### Risk 2 — HTP backend may auto-fallback ops to HVX or CPU silently

**Likelihood: Medium.** QNN HTP backend will run ops on whichever silicon it judges optimal; an "ElementWiseAdd" might run on the HMX OR the HVX vector unit OR even ARM CPU depending on op-config and tensor shape. The lattice's Trick #1 split assumes q_2 runs on the NPU/HMX specifically; if it actually runs on HVX, we've duplicated cDSP work and the Trick #1 parallelism evaporates.

**Impact: Medium for correctness (none — output is still right), HIGH for Trick #1 performance claims.**

**Mitigation**: K.2 instrumentation MUST include QNN profiling (`QnnProfile_create` + `QnnProfile_getEvents`) to confirm op-by-op silicon placement. T_K2_NPU_ON_HMX gate at K.2 stage 4: at least 80% of MAC ops in the q_2 forward pass attributed to HMX, NOT HVX or CPU. If gate fails, surface UPSTREAM rather than ship a Trick-#1 claim that's actually a CPU duplication.

### Risk 3 — QNN runtime per-process global state contention with FastRPC session

**Likelihood: Medium.** QNN HTP runtime internally creates a FastRPC session against `libQnnHtpV69Skel.so` and reserves DSP resources (VTCM partition, scheduler slot). The lattice already opens an INDEPENDENT FastRPC session against `libsp_compute_skel.so` for the cDSP HVX matmul kernel. Co-existence of two FastRPC sessions in one process is **not yet empirically tested**; could deadlock at session admission, contend over VTCM allocation, or serialize at the DSP scheduler.

**Impact: HIGH** if found — would force architectural pivot (e.g., separate daemon process per island, IPC over UNIX domain socket; or unify both paths through QNN).

**Mitigation**: K.2 stage 1 (lowest-risk-first) opens BOTH FastRPC sessions in the same process as a probe BEFORE building anything else. If contention surfaces, K.2 stage 0.5 is added to design the workaround. K.2-spike does NOT test this because the spike does not load the cDSP HVX kernel.

## What K.2 unlocks beyond Trick #1

- **Heterogeneous Mode D (Manifesto Trick #1 silicon-island scale)**: cDSP HVX (q_1) + NPU HMX (q_2) + ARM Garner = 2-island parallel compute with byte-exact discrete output. Sprint K v0.alpha + 2.5b + 2.5c proved cDSP-internal Trick #1; K.2 proves silicon-island Trick #1.
- **Foundation for M.6 cross-island Memory model**. K.2 + M.6 = production NPU Memory model with PoUW receipts spanning HVX + HMX.
- **Persisted-binary pattern**: same `contextCreateFromBinary` discipline applies to GPU (Adreno via clml-sdk) when that backend is investigated.
- **NPU as KSTE Tier-0/1 substrate** (Manifesto Trick #2): once NPU dispatch works, the KSTE histogram + signature paths can offload to HMX too. Sprint L scope.

## Can NPU host the FULL Memory model?

**Inconclusive from spike.** The QNN HTP runtime *can* host very large graphs (the Genie examples ship Llama3-8B running entirely on HTP), so capacity is not the issue. The open question is whether the lattice's discrete CRT-residue ops compose well with QNN's op vocabulary:

- ✓ `QNN_OP_MAT_MUL` exists (lattice Q8 matmul).
- ✓ `QNN_OP_ELEMENT_WISE_ADD` exists (verified in spike).
- ✓ Various pointwise + reduction ops exist.
- ? `QNN_OP_BARRETT_REDUCE` does NOT exist. The mod-q reduction inside our NTT kernel has no direct QNN op. We'd either (a) emit it as a chain of `QNN_OP_DIVIDE` + `QNN_OP_MULTIPLY` + `QNN_OP_SUBTRACT` (lossy — division on integers); (b) emit a QNN OpPackage with a custom HMX kernel implementing Barrett (heavy engineering); (c) keep mod-q on cDSP via FastRPC and only matmul-shape ops on NPU.

**K.2 recommendation: option (c)**. NPU hosts shape-friendly ops (matmul, layer norm, softmax, attention). Mod-q + Garner + Frobenius lift stay on cDSP. This naturally implements Trick #1 split: NPU does the dense-shape work, cDSP does the discrete-substrate work, ARM Garner-recombines.

This is exactly the cross-island shape the lattice wants. K.2 leans into it.

## "K.2 needs X before it can dispatch" — explicit call-out

**None.** K.2-spike has empirically resolved every pre-K.2 unknown:

- QNN SDK installed + accessible: VERIFIED.
- QNN HTP backend loadable on S22U: VERIFIED (vendor + SDK-shipped unsigned skel co-exist).
- Round-trip works without testsig: VERIFIED.
- Bridge architecture (libloading + C shim + Rust): VERIFIED.
- Lifecycle understood end-to-end: VERIFIED (getProviders → backendCreate → contextCreate → graphCreate → tensorCreate (no buffer) → addNode → finalize → execute (with buffer) → free).

K.2 can dispatch as the next sprint. The plan-commit for K.2 should cite this document + CLOSURE-K2-SPIKE.md + the empirical findings memory-entry candidate from CLOSURE.
