# Sprint M.1 — Memory budget audit + dual-load cDSP-internal (PLAN)

## Headline

Extend the existing dual-model AppState (`model` + `draft_model` from
Phase 4-SPEC) to load the M.0 stub Memory model (Qwen2.5-Coder-0.5B at
`D:\F\shannon-prime-repos\models\qwen25-coder-0.5b-memory.sp-model`)
concurrently with the Qwen3-0.6B Executive. Audit the per-component
memory budget on Knack's S22U. Drive concurrent forward steps on the
host-side L1 ABI (which transparently dispatches to cDSP via the shared
`Arc<FastRpcSession>` per K v0.alpha). Verify bit-identical outputs +
no leak across 1000 cycles.

## Stage 0 — Reference reading (file:line citations)

| Ref | File | Lines | Why it matters |
|---|---|---|---|
| 1 | `D:\F\shannon-prime-repos\shannon-prime-lattice\papers\PHASE-4-MEMO-M0-CHOICE.md` | 1-241 | Memory model path + sha256 + arch: `qwen25-coder-0.5b-memory.sp-model` at `D:\F\shannon-prime-repos\models\`, 473.22 MB, `812df63f..cc1126a`, Qwen2.5 24L×896H vocab=151936. M.0 closure §"Stub caveat" lines 115-137: Memory and Executive have DIFFERENT arch — fine for M.1 (budget+dispatch concerns, not weight-space). |
| 2 | `D:\F\shannon-prime-repos\shannon-prime-lattice\papers\PPT-LAT-Roadmap.md` | 5859-5870 | M.1 spec: gates `T_MEMO_BUDGET_AUDIT`, `T_MEMO_DUAL_LOAD`, `T_MEMO_DUAL_INVOKE` via `reference-fastrpc-concurrent-dispatch`. NPU dispatch deferred until K.2 ships. |
| 3 | `tools/sp_daemon/src/state.rs` | 46-58, 87-98 | EXISTING dual-model `AppState` (`model` + `draft_model` from Phase 4-SPEC `cafb349`) + android-only `dsp_session` / `dsp_model` / `kv_cache` fields. M.1 EXTENDS rather than duplicating. |
| 3a | `tools/sp_daemon/src/daemon.rs` | 97-145, 162-209 | EXISTING `run_inner` load path. Single-model load at L113; optional draft-model load at L132-145; android-only cDSP `DspModel::load` at L180-207. |
| 3b | `tools/sp_daemon/src/session.rs` | 17-44, 74-83, 101-205 | `SpModel`=Send+Sync (immutable post-load → can be shared between threads via Arc); `SpSession`=Send (NOT Sync — one per thread). `prefill_chunk` + `decode_step` + `position` are the L1 forward API; identical to probe.rs:83-100. |
| 3c | `tools/sp_daemon/src/bin/probe.rs` | 28-114 | Canonical recipe for "load .sp-model + create session + prefill[1,2,3] + decode_step(pos=4) + assert position==4". M.1 reuses this exact recipe for both models. |
| 4 | `~/memory/reference_fastrpc_concurrent_dispatch.md` | 1-184 | Arc<FastRpcSession> + dual-thread invoke = K v0.alpha 1.935× speedup, 0.9699 overlap. Wall-clock speedup is the parallelism discriminator, NOT pcycle ratio. |
| 5 | `tools/sp_dsp_smoke/src/sp_matmul_q_dual_smoke.rs` | 1-319 | Template for M.1's smoke harness: Arc<FastRpcSession> + sequential/concurrent timing + leak loop with second-half slope + functional cross-check. M.1 replaces the matmul_q invoke calls with SpSession::prefill_chunk. |
| 6 | `~/memory/feedback_leak_gate_allocator_warmup.md` | 1-110 | Leak gate metric MUST be second-half VmRSS slope (≤256 KB), NOT total delta. K v0.beta-2.5b precedent. |
| 7 | `~/memory/reference_mode_d_bridge_architecture.md` | (consulted) | FastRPC Path B Unsigned PD URI `file:///libsp_compute_skel.so?sp_compute_skel_handle_invoke&_modver=1.0&_dom=cdsp` — already in use by daemon.rs:167. M.1 does NOT change Path B → Signed PD migration in this sprint. |

## Architectural insertion point

The existing AppState ALREADY has dual-model slots from Phase 4-SPEC:

- `model: SpModel`           → reused as **Executive** (Qwen3-0.6B)
- `draft_model: Option<SpModel>` → reused as **Memory** (Qwen2.5-Coder-0.5B)
- `draft_session: Option<Mutex<SpSession>>` → reused as Memory base session

This is the cleanest reuse path: NO new AppState field, NO type churn,
NO ABI surface change. The CLI flag `--draft-model` becomes the Memory
load knob; daemon log lines for the M.1 sprint simply re-label this
slot as "Memory" in commentary, but the field name stays for Phase 4-SPEC
compatibility. (A future sprint can rename for semantic clarity once
both Phase 4-SPEC and 4-MeMo have shipped.)

**Rationale for reusing rather than adding a NEW field:**
- The M.0 closure explicitly anticipates this (line 228-230):
  "`tools/sp_daemon/src/state.rs:46-58` — existing dual-model AppState
  (`model` + `draft_model`) infrastructure from Phase 4-SPEC `cafb349`;
  M.1 will reuse semantically as Executive + Memory."
- Phase 4-SPEC (spec-decode) and Phase 4-MeMo (memory-model) both
  share the "two-models-per-process" architecture; one set of plumbing.
- The semantic relabel costs ~10 LOC of comment edits.

If during Stage 1 we discover the field reuse causes confusion or
ABI conflict, we surface UPSTREAM and propose a rename (no silent
gate revision).

## FastRpcSession sharing pattern (one Arc vs two Arcs)

**Choice: ONE `Arc<FastRpcSession>` shared between BOTH model lanes.**

Reasoning:
- K v0.alpha empirically confirmed that ONE Arc<FastRpcSession> + two
  threads engages V69 SSR:XA={4,5} dual vector contexts via the cDSP
  scheduler. The scheduler doesn't distinguish "model A vs model B" — it
  sees two HVX kernel dispatches arriving concurrently on the same
  handle, and schedules them onto two contexts.
- K.beta.2.5c shipped the same pattern for q_1 / q_2 CRT shards on the
  same handle and got 1.935× speedup at compute-bound shape.
- Adding a second FastRpcSession (= second Path-B URI open) would
  double the cDSP-side process domain overhead with NO parallelism
  benefit (the cDSP scheduler is process-domain-level, not session-
  level).

EXCEPT for the cDSP-resident `DspModel` path (android-only, full-model
DMA arena): that path uses a SEPARATE `Box::leak`'d `&'static`
FastRpcSession from `dsp_session` (daemon.rs:182), because the
DmaBuffer<'sess> arena needs static borrow. M.1 does NOT change this;
M.1's concurrent invoke runs through the L1 sp_session forward path
(host CPU + transparent cDSP dispatch for kernels), not through
DspModel.

## Smoke harness design

**ONE consolidated binary**: `tools/sp_daemon/src/bin/sp_memo_m1_smoke.rs`
(auto-discovered by Cargo; no `[[bin]]` entry needed). Auto-discovery
matches probe.rs's existing pattern.

Reasons for ONE binary not three:
- Budget audit + dual-load + concurrent invoke + no-interference all
  share the same setup (load both models, open FastRpcSession).
- Reduces APK / ADB push footprint.
- Closure JSON output is one report.
- Matches K.beta.2.5c `sp_matmul_q_dual_smoke.rs` precedent
  (one binary that runs functional + leak + speedup gates in sequence).

CLI:
```
sp_memo_m1_smoke <executive_model.spm> <executive_tok.spt> \
                 <memory_model.spm>    <memory_tok.spt>    \
                 [--cycles N]          [--report-json path]
```

Stages within the binary:

1. **Budget snapshot — pre-load.** Capture VmRSS of THIS process + read
   `/proc/meminfo` for `MemTotal`, `MemFree`, `MemAvailable`. Record
   `pre_load_vmrss_kb` + `pre_load_meminfo_*`.
2. **Open Arc<FastRpcSession>** (android-only; host build skips with
   `#[cfg(not(target_os="android"))]` stub like probe.rs).
3. **T_MEMO_DUAL_LOAD**: Load Executive first via `SpModel::load`;
   capture wall + post-Exec VmRSS; load Memory via `SpModel::load`;
   capture wall + post-Mem VmRSS. Assert: both succeed within
   30s combined; arches reported correctly.
4. **T_MEMO_BUDGET_AUDIT**: Compute deltas:
   `exec_vmrss_delta_mb = post_exec - pre_load`
   `memo_vmrss_delta_mb = post_mem  - post_exec`
   `total_vmrss_mb      = post_mem  - pre_load`
   `meminfo_pre`, `meminfo_post` snapshots.
   Estimate Android OS reservation: `MemTotal - MemAvailable_pre`
   (rough; includes zygote + system_server + other apps).
   Estimate residual headroom: `MemAvailable_post`.
   Emit JSON report + human-readable narrative.
   Pass criterion: `MemAvailable_post >= 2048 MB` (2 GB residual).
5. **Solo baselines for T_MEMO_DUAL_INVOKE**: Create one SpSession per
   model, drive `prefill_chunk([1,2,3])` solo on each, capture
   `executive_baseline_logits[0..3]` and `memory_baseline_logits[0..3]`
   (also `position()=3` per session, then `decode_step(pos=4)` for
   completeness mirroring probe.rs). Record these as the "no
   interference" reference.
6. **T_MEMO_DUAL_INVOKE**: Spawn two threads. Each thread owns its
   model's SpSession (clone from a fresh one, since SpSession is Send
   but not Sync). Each runs `prefill_chunk([1,2,3])`. Join. Assert:
   executive output bit-identical to (5)'s baseline; memory output
   bit-identical to (5)'s baseline. Also record wall-clock speedup
   vs sequential = `(seq_wall) / (concurrent_wall)` ≥ 1.1× per
   `feedback-shape-dependent-parallelism-gates` (note: 3-token prefill
   is a SHORT prompt; if speedup is below 1.1 we surface UPSTREAM as
   data-bound regime per K v0.beta-2.5b precedent and report as
   measurement, not gate fail).
7. **T_MEMO_NO_INTERFERENCE**: 1000 cycles of (6) with VmRSS samples
   at i=0, i=500, i=1000. Per-cycle: spawn-join two threads with
   fresh per-thread SpSession (cloned from base; dropped at end of
   cycle to exercise create/destroy). Track:
   `cycles_run, executive_drift_count, memory_drift_count,
    fastrpc_errors, vmrss_first_half_delta_kb,
    vmrss_second_half_delta_kb`.
   Gate: drift counts == 0, errors == 0, second-half slope ≤256 KB.
8. **Closure**: emit JSON report; print all gates PASS/FAIL; exit 0 on
   all PASS, 1 otherwise.

## Budget audit JSON schema

```json
{
  "sprint":  "M.1",
  "device":  { "model": "SM-S908E", "android": "15",
               "memtotal_kb": 11473784 },
  "pre_load": {
    "vmrss_kb":         1234,
    "memtotal_kb":      11473784,
    "memfree_kb":       712992,
    "memavailable_kb":  5510420
  },
  "executive": {
    "path":             "/data/local/tmp/qwen3_rt.sp-model",
    "size_bytes":       754551808,
    "load_wall_ms":     ...,
    "vmrss_after_kb":   ...,
    "vmrss_delta_kb":   ...,
    "arch": { "vocab": 151936, "n_layers": 28, "hidden": 1024 }
  },
  "memory": {
    "path":             "/data/local/tmp/qwen25-coder-0.5b-memory.sp-model",
    "size_bytes":       496202752,
    "load_wall_ms":     ...,
    "vmrss_after_kb":   ...,
    "vmrss_delta_kb":   ...,
    "arch": { "vocab": 151936, "n_layers": 24, "hidden": 896 }
  },
  "post_load": {
    "vmrss_kb":         ...,
    "memavailable_kb":  ...,
    "headroom_mb":      ...,
    "android_os_estimate_mb":  ...
  },
  "gates": {
    "T_MEMO_BUDGET_AUDIT":     "PASS|FAIL",
    "T_MEMO_DUAL_LOAD":        "PASS|FAIL",
    "T_MEMO_DUAL_INVOKE":      "PASS|FAIL",
    "T_MEMO_NO_INTERFERENCE":  "PASS|FAIL"
  }
}
```

## Workflow

| Stage | Commit message | Files |
|---|---|---|
| plan | `[plan] M.1 -- Memory budget audit + dual-load (cDSP-internal)` | this file |
| 1 | `[M.1] feat: Stage 1 -- sp_memo_m1_smoke binary + dual SpModel load + budget snapshot` | `tools/sp_daemon/src/bin/sp_memo_m1_smoke.rs`, `tools/sp_daemon/scripts/m1_push.ps1` |
| 2 | `[M.1] test: Stage 2 -- T_MEMO_BUDGET_AUDIT + T_MEMO_DUAL_LOAD run on S22U` | adds observed-data ingest into closure |
| 3 | `[M.1] test: Stage 3 -- T_MEMO_DUAL_INVOKE solo+concurrent on S22U` | (results into closure) |
| 4 | `[M.1] test: Stage 4 -- T_MEMO_NO_INTERFERENCE 1000-cycle + closure` | `tools/sp_compute_skel/docs/CLOSURE-M1-DUAL-LOAD.md` |

## Sub-tag

`lat-phase-4-memo-m1-dual-load`.

## Out of scope (explicit)

- M.2 dialogue loop (Grounding/Entity/Synthesis state machine).
- M.3 Frobenius-lifted TIES merge (blocked on M.0-real).
- Forward-path optimization.
- KV cache budgeting across dual context.
- NPU dispatch (K.2 / K.2-spike concurrent lane).
- AppState field rename (`draft_model` → `memory_model`); deferred
  until both Phase 4-SPEC and 4-MeMo have shipped.
- Memory-model artifact verification (sha256 / size); M.0 already
  did this and committed `lat-phase-4-memo-m0-stub`.

## Worktree discipline

- Worktree: `D:\F\shannon-prime-repos\engine-m1` (added 2026-05-30
  per `feedback-parallel-agents-separate-worktrees`).
- Branch: `sprint/memo-m1` (base `0cf9674` = K.beta.2.5c full closure).
- All commits authored from this worktree exclusively.
- Push at end: `git push -u origin sprint/memo-m1`. Operator handles
  merge to engine main after review.
- K.2-spike agent's `engine-k2-spike` worktree NOT TOUCHED.
- `shannon-prime-system-engine` main worktree NOT TOUCHED.
- Memory model + Executive model artifacts: READ-ONLY consumer.

## References

- `~/memory/reference_fastrpc_concurrent_dispatch.md`
- `~/memory/reference_mode_d_bridge_architecture.md`
- `~/memory/feedback_leak_gate_allocator_warmup.md`
- `~/memory/feedback_no_silent_gate_revisions.md`
- `~/memory/feedback_shape_dependent_parallelism_gates.md`
- `~/memory/feedback_lead_with_reference_then_theory.md`
- `~/memory/feedback_parallel_agents_separate_worktrees.md`
- `~/memory/reference_heterogeneous_soc_crt_tricks.md` Trick #1
