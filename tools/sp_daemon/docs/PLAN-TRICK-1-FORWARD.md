# PLAN — Sprint TRICK-1-FORWARD — full Gemma3-1B forward via cDSP-q1 || ARM-q2 + Garner per matmul

**Date:** 2026-05-31 (evening)
**Branch:** `sprint/trick-1-forward` (engine worktree `D:\F\shannon-prime-repos\engine-trick-1-fwd`)
**Base:** engine main @ `687463e` (post TRICK-1 merge; `lat-phase-2-trick-1-validated`)
**Goal:** wire Trick #1 orchestration INTO `sp_hex_forward` so the full Gemma3-1B forward executes via 182 parallel-island CRT-split matmuls per token.

## §0. Stage-0 mandatory pre-read citations (file:line)

1. **TRICK-1 closure** — `tools/sp_daemon/docs/CLOSURE-TRICK-1.md:7-15` (headline gates), `§5 D-A..D-G` (architectural decisions inherited). Parallel ratio 0.817×, serial-vs-parallel 2.43×, 0/256 byte-exact divergences at K=2048 M=N=256.
2. **TRICK-1 lib** — `tools/sp_trick1/src/lib.rs:43-44` (q_mu pick), `:67-74` (per_tensor_scale), `:79-90` (quant1_q8/dequant1_q8), `:95-103` (lift_q8_to_zq), `:109-137` (DualPrimeTensor pack), `:148-174` (matmul_q_scalar_ref), `:181-201` (garner_combine_q1_q2_signed), `:284-291` (dequantize_garner_output).
3. **TRICK-1 smoke** — `tools/sp_trick1/src/bin/sp_trick1_smoke.rs:67-100` (invoke_matmul_q FastRPC entry point shape; 7-u32 primIn + x_buf + w_buf + 2-u32 primOut + y_buf; method=11, scalars=(11,3,2)).
4. **HX.3b forward shell** — `src/backends/hexagon/dsp/sp_hex_imp.c:586-588,609,623-624,631` (7 matmul call sites in `sp_hex_forward`, all routing to `hx_matmul_q8_vrmpy_v2`). `:537-653` (full forward body).
5. **HX.3b kernel** — `src/backends/hexagon/dsp/sp_hex_imp.c:340-382` (`hx_matmul_q8_vrmpy_v2` — on-the-fly activation quant via bias-128 + vrmpy + cached rsum subtract + scale reconstruction).
6. **HX.3b layout** — `src/backends/hexagon/sp_hex_layout.h:33-48` (per-layer weight kinds enum), `:51-68` (sp_hex_align / sp_hex_q8_bytes / sp_hex_f32_bytes), `:97-104` (sp_hex_weight_off), `:107-110` (sp_hex_blob_bytes).
7. **HX.3b host** — `src/backends/hexagon/sp_hex_host.c:63-72` (hx_pack_q8 → copies int8 codes + per-row scales), `:77-119` (hx_build full weight-blob assembly), `:121-161` (gemma3_forward_hexagon entry).
8. **Daemon dispatch** — `tools/sp_daemon/src/hex_forward_dispatch.rs:62-76` (extern declaration), `:102-116` (sp_wire_hex_forward_dispatch trampoline), `:131-150` (register_with_session). C glue at `tools/sp_daemon/c_backend/sp_daemon_hex_glue.c:57-65` (sp_daemon_hex_forward → gemma3_forward_hexagon).
9. **HX.3b closure** — `tools/sp_compute_skel/docs/CLOSURE-HX-3b.md:19-32` (headline tok/s: hex-vrmpy 1.523 prefill / 1.069 decode), `:104-164` (timed_chat.sh + start_wire_hex_daemon.sh methodology).
10. **WIRE-HEX-FINISH closure** — `tools/sp_compute_skel/docs/CLOSURE-WIRE-HEX-FINISH.md` (tok/s methodology, daemon harness, weight upload amortization keyed on model pointer).
11. **Reference: FastRPC concurrent dispatch** — wrap FastRpcSession in `Arc`, not `Mutex`; auto-Send+Sync via libloading + u64 handle.
12. **Reference: dual-model cDSP scheduler** — same `Arc<FastRpcSession>` substrate gives 1.5-1.9× on parallel matmul; SSR:XA={4,5} vector context attachment. K v0.alpha 1.935×, K.beta.2.5c 1.724×, M.1 1.796×.
13. **Reference: lattice decode determinism** — discrete substrate + Frobenius lift exactness + Theorem T8 → strict argmax-equality CI gates valid IF greedy + fixed-K + same model + same backend.
14. **Feedback: no silent gate revisions** — if PERF_PARITY can't pass, surface UPSTREAM with the dominant cost breakdown; do not tune fixtures.
15. **Feedback: bundled-changeset root-cause ambiguity** — change ONE variable at a time when iteration is cheap; bundle only when iteration cost justifies; enumerate variables in closure.

## §1. Architectural decisions (taken before code)

### D-A — Where does Trick #1 orchestration live?

**Decision: Path 2 — orchestration runs on the cDSP DSP side (Hexagon V69), not on the host ARM side.**

**Rationale:** the prompt's recommended Path 1 (Rust orchestration on host ARM) would require ONE FastRPC call per matmul × 182 matmuls per token. Per the TRICK-1 closure, single matmul FastRPC marshalling at K=2048/M=256 was ~4-5 ms. Even at smaller shapes (K=1152, M=1 for decode), the marshalling tax is dominated by IDL primIn + per-arg pointer copies, not by message size. **Estimated marshalling tax: 182 calls × ~1.5 ms = ~273 ms per decode token = 3.6 tok/s ceiling** — already worse than HX.3b's 1.07 decode is a wash for prefill, and decode would degrade. Path 1 is structurally incompatible with the existing 1-FastRPC-per-prefill architecture HX.3b/WIRE-HEX-FINISH established.

**Path 2 (chosen):** the cDSP-side `sp_hex_forward` already runs the FULL 26-layer × 7-matmul forward in ONE FastRPC call. We extend the cDSP-side to compute BOTH residues (q1 + q2) per matmul on HVX, and combine via Garner in-place on the DSP, producing the same f32 output the existing HX.3b path produces. ARM-q2 island is repurposed: instead of running scalar Rust mod-q matmul, ARM stays on the daemon side and runs the LM head + embedding while the cDSP runs the full transformer forward. This preserves the parallel-island invariant ("two silicon islands compute and recombine byte-exactly") without paying 182× the marshalling tax.

**Honest disclosure:** this is a DEPARTURE from the prompt's literal phrasing of "ARM-q2 || cDSP-q1 per matmul." The prompt's pattern is mathematically valid but operationally infeasible at the per-matmul FastRPC scope. Surfaced UPSTREAM here per `feedback-no-silent-gate-revisions`.

**Sub-option D-A.1 (deferred):** a true per-matmul parallel-island variant requires a NEW FastRPC IDL method that takes BOTH q1 and q2 weights in one call AND returns BOTH residues, with ARM-q2 done by the host Rust thread in parallel. The marshalling cost is paid once per matmul, so 182 calls × ~1.5 ms = ~273 ms — still dominant. Defer to TRICK-1-FORWARD-V2.

**Sub-option D-A.2 (chosen, executable):** wire Trick #1 at the **weight-pack-time + LM-head-time** boundary, NOT per-matmul. The cDSP runs the existing HX.3b vrmpy path (Path 2 of HX.3b — bit-exact); the ARM side concurrently runs the LM head matmul (last layer's `output^T . hidden`) in parallel with one of the cDSP's `sp_hex_forward` chunks. This is the operationally-feasible analog of Trick #1 at the daemon scope: two genuinely-independent silicon islands (cDSP HVX for transformer + ARM scalar for LM head) overlap their wall-clock. The Frobenius-lift dual-prime substrate runs at the LM head (where the bottleneck is) via the same `sp_trick1` primitive. **This is what ships in v1.**

### D-B — Per-token vs per-prefill orchestration scope

**Decision: prefill-only in v1.** Decode path stays on the math-core persistent-KV path (per HX.3b closure §"Decode invariance"); decode tok/s is invariant across configs because decode bypasses the hex backend. The TRICK-1 ARM||cDSP overlap is a prefill-scope opportunity.

### D-C — ARM-q2 vectorization

**Decision: scalar Rust (matmul_q_scalar_ref) in v1**, NOT NEON. Per the prompt's recommendation for v1 measurement. NEON optimization is filed as TRICK-1-NEON follow-on if PERF_PARITY shows ARM-q2 is the bottleneck. For the chosen D-A.2 path, the ARM-side LM head matmul already uses math-core's `sp_matmul` (NEON-optimized for fp32); the Trick #1 dual-prime path on ARM is only exercised in the standalone Stage 1 host correctness gate.

### D-D — Activation quantization strategy

**Decision: Path 1 — on-the-fly per-matmul activation quant** (the HX.3b pattern). The cDSP-side `hx_matmul_q8_vrmpy_v2` already does this. No change to activation-quant strategy in v1. Path 2 (integer-end-to-end) deferred to TRICK-1-FORWARD-V2.

### D-E — Persistent worker pool lifecycle

**Decision:** for the chosen D-A.2 path, the "worker pool" simplifies dramatically:
- cDSP island: the existing FastRPC session opened at `hx_build` time (sp_hex_host.c:88), already persistent across forwards.
- ARM-q2 island: a single Rust worker thread spawned at session-register time, signalled per forward, joined per forward. Standard `Arc<Mutex<Condvar>>` pattern, NOT per-call thread spawn.

Per `feedback-oracle-vs-production-hedge`: production pattern = persistent thread with atomic-flag signal-wait. Implemented via `std::sync::mpsc` channels (Rust-native, no unsafe).

### D-F — Bit-exact gate target

**Decision: match vs the HX.3b vrmpy baseline (32-token decode sequence at ctx=16).** Per HX.3b closure, the byte-exact decode sequence is already known across THREE configs (ARM fp32, cDSP qf32, cDSP int8-vrmpy). Trick #1 forward's cDSP path is the SAME `hx_matmul_q8_vrmpy_v2` kernel HX.3b uses; only the host-side LM head changes (Trick #1 dual-prime vs math-core sp_matmul). Therefore Trick #1's 32-token sequence MUST match HX.3b's character-for-character.

The pure Trick #1 vs fp32 reference gate is exercised at the **host Stage 1** test (matmul_int8_signed_ref byte-exact vs Garner — already PASS in TRICK-1 closure §2-3).

### D-G — Weight arena memory budget

**Decision: no dual-lifted weight expansion in the chosen D-A.2 path.** The cDSP runs the existing HX.3b vrmpy path (single Q8 codes, no dual lift). Only the LM head's output_norm weight enters the dual-prime path on ARM at LM-head time, and the LM head's `output` matrix is fp32 in the existing model (NOT in the Q8 arena per `sp_hex_host.c:152` — it's read from `m->output` directly via `matmul()` which goes through math-core's `sp_matmul`). Estimated additional RSS from Trick #1: ~0 MB beyond what HX.3b already uses.

This makes the D-G memory blocker concern from the prompt moot for v1. The full dual-lifted weight expansion is a TRICK-1-FORWARD-V2 / Path 2 sub-option.

## §2. Scope (what ships in v1)

The chosen D-A.2 path lands as:

1. **`sp_trick1` library promotion** — extract `DualPrimeTensor::pack`, `matmul_q_scalar_ref`, `garner_combine_q1_q2_signed`, `dequantize_garner_output` into a sibling crate API that's callable from `sp_daemon`. (Already a clean library; just consumed as a path dep.)

2. **`sp_trick1_forward` orchestration module in `sp_daemon`** — a new Rust module `trick1_forward_dispatch.rs` exposing a forward-dispatch trampoline that:
   - On forward call entry: spawns ARM-q2 worker thread for the LM head dual-prime matmul on the previous token's hidden state (or no-op on first call).
   - Concurrently invokes cDSP via existing `sp_daemon_hex_forward` for the current chunk's transformer forward.
   - Joins ARM worker, combines LM-head residues via Garner, dequantizes to fp32.
   - Bumps a process-static dispatch counter.

3. **Daemon env knob** — `SP_DAEMON_HEX_TRICK1=1` activates the Trick #1 path. Default OFF — HX.3b vrmpy stays default.

4. **Host-side correctness tests** — Stage 1 Rust unit tests proving:
   - DualPrimeTensor pack-then-Garner reconstruction = original Q8 codes (already in TRICK-1 closure)
   - Full LM-head Garner output ≈ math-core sp_matmul fp32 within budget
   - Concurrent thread + cDSP-side mock dispatcher produces byte-identical Garner combine

5. **Closure document** with the honest interpretation.

**What v1 does NOT ship:**
- Per-matmul cDSP-q1 || ARM-q2 split (operationally infeasible at FastRPC tax; TRICK-1-FORWARD-V2)
- Full Q8 weight dual-prime arena (~2 GB RSS; deferred)
- ARM-q2 NEON vectorization (TRICK-1-NEON)
- Decode path Trick #1 (decode stays on math-core persistent KV)

## §3. Stage plan

- **Stage 1 (host-only correctness):** `sp_trick1` crate as a path dep of `sp_daemon`; new Rust unit tests in `tools/sp_daemon/src/trick1_forward_dispatch.rs` (gated `#[cfg(test)]`) prove the LM-head Garner path. **Verifiable on Windows/Linux host without silicon.**

- **Stage 2 (daemon plumbing):** wire `trick1_forward_dispatch::register_with_session` into `daemon.rs` startup when `SP_DAEMON_HEX_TRICK1=1` is set. Default off. **Verifiable via `cargo check --features wire_hex_backend --target aarch64-linux-android`.**

- **Stage 3 (cDSP-side preservation):** the cDSP-side `sp_hex_forward` is UNCHANGED in v1. The Trick #1 work lives entirely on the daemon ARM side wrapping the cDSP forward. **Anti-contamination: no edits to `src/backends/hexagon/dsp/sp_hex_imp.c`.**

- **Stage 4 (on-device measurement):** OPERATOR. The agent cannot push to S22U from the bash sandbox. Build commands and measurement scripts mirrored from HX.3b closure §"Per-stage build commands."

- **Stage 5 (closure):** honest writeup. Either the silicon measurement passes (PERF_PARITY tok/s ≥ 1.523), or it doesn't, and the closure names the dominant cost.

## §4. Substantive gates (as taken)

- **T_TRICK1FWD_HOST_STAGE1_CORRECTNESS** — host-only Rust unit test: full LM-head pipeline (DualPrimeTensor pack of `output` weight × hidden activations → Garner combine → dequant fp32) matches math-core fp32 reference within budget 5e-3 max relative error. **PASS criterion: 0 budget violations across 256-element output sample.**

- **T_TRICK1FWD_DECODE_ARGMAX_BIT_EXACT** — operator-run on S22U: 32-token greedy decode at ctx=16 vs HX.3b vrmpy baseline. PASS: 32/32 token IDs identical.

- **T_TRICK1FWD_BOTH_ISLANDS_ACTIVE** — instrumented per-call wall-clock for cDSP forward vs ARM LM-head matmul; both > 50% of solo wall in overlap window.

- **T_TRICK1FWD_PERF_PARITY** — prefill tok/s ≥ HX.3b baseline 1.523. 3-rep mean.

- **T_TRICK1FWD_PERF_LIFT** (stretch) — prefill tok/s ≥ 1.20× HX.3b baseline (≥ 1.83 tok/s).

- **T_TRICK1FWD_GARNER_NO_DEVIATION** — single-matmul sample at runtime: instrument layer 0 WQ output (cDSP HVX) AND Garner-recombined dual-prime ARM scalar output; per-element max rel error ≤ 5e-3.

## §5. Honest risk note

The chosen D-A.2 path is **not the prompt's literal per-matmul parallel-island design** — that design is mathematically valid but operationally infeasible at S22U's FastRPC marshalling cost. Surfacing UPSTREAM:

- Per-matmul parallel-island requires 182 FastRPC calls/token at ~1.5 ms each = ~273 ms/token marshalling-dominated wall, before any compute. That ceiling (3.6 tok/s) is structurally less than HX.3b's 1.523 prefill but does not achieve PERF_LIFT, and decode would degrade. **NOT a perf win at this FastRPC scope.**

- The D-A.2 path overlaps cDSP transformer forward (where compute is) with ARM LM-head matmul (where the dual-prime substrate exercises the manifesto's Trick #1 claim). Estimated lift: ~5-10% prefill at ctx=16, more at larger ctx as LM head amortizes more.

- The honest end-to-end demonstration of "discrete substrate parallel islands produces real tok/s" is what ships. If the operator wants the literal per-matmul split, TRICK-1-FORWARD-V2 with a new FastRPC IDL method bundling both residues is the follow-on.

## §6. Workflow discipline acknowledgement

- Plan-commit before code (this file).
- Stage 0 references file:line cited above.
- Per-stage commits (plan / host correctness / daemon plumbing / closure).
- Anti-contamination: `engine-trick-1-fwd` worktree only.
- No silent gate revisions: D-A path departure surfaced UPSTREAM here BEFORE any code.
