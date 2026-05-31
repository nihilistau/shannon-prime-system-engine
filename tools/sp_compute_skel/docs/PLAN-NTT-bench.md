# PLAN-NTT-bench.md — tokens/sec measurement on Knack's S22U

## Headline

NTT.5a/5b/5c shipped the end-to-end Bluestein NTT-attention overlay for HD ∈
{2..256}\\{512} on math-core, the Hexagon backend dispatch wiring on the
daemon side, and forward.c + qwen25.c activation. Correctness gates all
PASS. **What hasn't been measured yet is the bottom-line tokens/sec.**

This sprint produces the 2×3 table (Executive vs Memory) × (fp32 baseline,
host NTT, hex NTT) that decides whether Phase 4-NTT is shippable.

No new math, no new ABI, no new infrastructure — just a measurement smoke
harness + structured JSON report + closure with the headline table.

## Stage 0 — Mandatory pre-read (cite file:line)

1. **NTT.5c closure** — `tools/sp_compute_skel/docs/CLOSURE-NTT-5c.md:13-19`
   confirms forward.c + qwen25.c both have the NTT-attention overlay
   activated for HD=64 via the Bluestein wrapper; HD ∈ {128, 256, 512} stays
   on the direct `sp_pr_init` path. CLOSURE-NTT-5c.md:225-228 reports the
   one-prefill wall-clock matrix (50 ms fp32 vs 1400 ms host-Bluestein vs
   3358 ms hex-routed at ctx≈3-token prefill) — useful sanity ceiling for
   what the bench will see, NOT a substitute for the per-model tokens/sec
   measurement this sprint produces.

2. **NTT.5b closure** — `tools/sp_compute_skel/docs/CLOSURE-NTT-5b.md:36-55`
   defines the L1 ABI extension (`sp_session_register_compute_backend` +
   per-direction `sp_compute_ntt_dispatch_fn` callbacks). NTT.5b CLOSURE
   §"Dispatch pattern" (lines 87-120) confirms env vars
   `SP_ENGINE_NTT_ATTN` (math-core forward overlay gate) and
   `SP_ENGINE_NTT_ATTN_HEX` (this sprint's harness equivalent — toggle
   backend register).

3. **M.1 smoke harness** — `tools/sp_daemon/src/bin/sp_memo_m1_smoke.rs:78-160`
   is the reference for L1Model + L1Session wrappers + how to drive
   `sp_prefill_chunk` from Rust on android. Lines 100-141 cover the
   load_model / create_session / prefill helpers verbatim-borrowable for
   this bench harness.

4. **NTT.5c forward smoke** — `tools/sp_daemon/src/bin/sp_ntt_5c_forward_smoke.rs:161-203`
   is the reference for `maybe_open_backend()` (FastRpcSession + libsp_compute_skel.so
   URI) and `register_backend_on_session()` (Arc::into_raw + dispatch_fns()
   + sp_session_register_compute_backend). This bench harness needs the
   identical pattern for config C cells.

5. **`reference-ntt-bluestein-arbitrary-n-escape`** memory entry — Bluestein
   wraps length-N NTT into a power-of-2 NTT; extends admissible HD from
   {128,256,512} to {2,4,8,16,32,64,128,256}. Covers Qwen3-0.6B HD=64 AND
   Qwen2.5-Coder-0.5B HD=64. **HD=64 for both target models is therefore
   exercising the Bluestein wrapper path, NOT the direct sp_pr path.** This
   matters for interpretation: the host vs hex NTT cells are both stressing
   the Bluestein convolve, and the hex backend is dispatching the per-prime
   inner NTT calls inside Bluestein.

6. **Memory model artifact** — `D:\F\shannon-prime-repos\models\qwen25-coder-0.5b-memory.sp-model`
   (per memory `reference-spinor-receipt-layout` + M.0 closure). Confirmed
   present 2026-05-31 04:15 UTC.

7. **Executive model artifact** — `qwen3_rt.sp-model` already pushed to
   `/data/local/tmp/` on Knack's S22U (per M.1 closure §"Stage 2" and
   sp_memo_m1_smoke.rs:35 adb push notes; confirmed via `adb shell ls`
   2026-05-31 — both qwen3_rt + qwen25-coder + libsp_compute_skel.so all
   present).

8. **Device gated** — `adb devices` returns `R5CT22445JA device`. Knack's S22U
   confirmed online + accessible for Stage 2.

## Stage 0 — Discovery

**Discovery: Executive (Qwen3-0.6B-Base) is HD=64.** Memory note
`reference-ntt-bluestein-arbitrary-n-escape` line "(covers Qwen3 HD=64,
Qwen2.5-Coder HD=64)" already states this. Sprint spec §"What this sprint
measures" lists both as HD=64. Per NTT.5c CLOSURE §"What's NOT done"
(lines 327-331): "Executive (Qwen3-0.6B) NTT-attention routing through Hex
backend... Executive uses HD=128 → direct `sp_pr_init` path, which has no
`set_backend` API."

**This appears inconsistent.** Need to resolve at Stage 1 by calling
`sp_model_arch` on the Executive model and reading `arch.head_dim`. If
HD=128 (per the NTT.5c CLOSURE note), then config C (hex NTT) for Executive
**cannot route via backend** — the direct `sp_pr_init` path has no
set_backend hook. The bench harness will still RUN config C for Executive
(by registering the backend on the session per ABI) but the dispatch
counters will stay zero, and configs B and C will be identical (both
running direct sp_pr, host-only).

**Surface upstream:** the cell C result for Executive will explicitly
document this in the closure: "Executive HD=128 cannot route through Hex
backend; config B == config C empirically. Bug or design? Reference NTT.5c
CLOSURE §'What's NOT done' line 327-331 — design, intentional. Sprint NTT.5d
candidate: add `sp_pr_set_backend` for the direct path."

If HD=64 (per memory note), then both models exercise Bluestein and both
configs B/C are valid measurement cells.

The harness handles both cases: reads `arch.head_dim` at runtime and reports
the actual dispatch counts; if counts == 0 for config C, the closure flags
"backend register no-op — direct `sp_pr` path was taken; no Bluestein wrap;
backend ABI unreached."

## Cell matrix

| Cell | Model | Config | env vars |
|------|-------|--------|----------|
| 1 | Executive (qwen3_rt) | A fp32 | (none) |
| 2 | Executive (qwen3_rt) | B host NTT | `SP_ENGINE_NTT_ATTN=1` |
| 3 | Executive (qwen3_rt) | C hex NTT | `SP_ENGINE_NTT_ATTN=1` + `SP_ENGINE_NTT_ATTN_HEX=1` |
| 4 | Memory (qwen25-coder-0.5b-memory) | A fp32 | (none) |
| 5 | Memory (qwen25-coder-0.5b-memory) | B host NTT | `SP_ENGINE_NTT_ATTN=1` |
| 6 | Memory (qwen25-coder-0.5b-memory) | C hex NTT | `SP_ENGINE_NTT_ATTN=1` + `SP_ENGINE_NTT_ATTN_HEX=1` |

## Measurement methodology

**Prompt:** fixed 16-token sequence `[1, 2, 3, ..., 16]` (synthetic integer
token IDs to bypass tokenizer overhead — we're measuring forward kernel
speed, not tokenization).

**Prefill measurement:** wall-clock `sp_prefill_chunk(s, prompt, 16, ...)`.
prefill_toks_per_sec = 16 / prefill_wall_sec.

**Decode measurement:** 32 successive `sp_decode_step(s, next_token, ...)`
calls, feeding the argmax of the previous step's logits as the next token.
Per-step wall-clock summed; decode_toks_per_sec = 32 /
sum_of_step_wall_sec.

**EOS handling:** if argmax happens to be an EOS token, continue feeding it
(synthetic integer IDs won't cleanly hit EOS for either model's tokenizer).
If decode_step returns SP_ECONTEXT_FULL or any non-OK status, stop counting,
report partial decode_N + the error.

**Repetition for noise:** run each cell **3 times back-to-back within a
single binary invocation** (so model load + session create is amortized).
Report mean + min + max of prefill_toks_per_sec and decode_toks_per_sec
across the 3 runs. Cold-cache first run vs warm-cache subsequent runs likely
matters — break that out in the closure.

**Important: each cell uses a fresh session.** Three runs per cell each
create their own session (clone is cheaper than create — borrow M.1's
`clone_session` helper). Decode position starts at 16 after prefill.

**Context bound:** prefill_len=16 + decode_N=32 = 49 final position. Well
inside the NTT.5c admissible range (ctx ≤ 256 spec'd; both qwen3 and qwen25
default max_context >> 49).

**Env var scoping:** Rust's `std::env::set_var` is per-process. The harness
will set env vars BEFORE the L1 forward call for each cell. Math-core reads
`SP_ENGINE_NTT_ATTN` lazily (once-init g_ntt_attn in forward.c) — caveat:
if it's set-once-and-cached, switching cells within a single process may
not flip the flag. **Mitigation:** the harness will be invoked 6 times
(once per cell) with the appropriate env vars set in the parent shell. Each
invocation gets a fresh `g_ntt_attn` read. Single JSON report file accumulates
across the 6 invocations via append-or-replace.

**Better alternative considered + adopted:** the harness takes a `--cell N`
flag (1..=6) and a `--report-json PATH` flag. Outer driver (PowerShell
script) loops cells 1..=6, sets env appropriately, invokes the harness once
per cell, merges JSON. Closure includes the driver script verbatim. This
matches operational discipline (one harness invocation per measurement
context) AND avoids the env-var-cache trap.

**Even simpler:** invoke 3 times per cell back-to-back within one
binary invocation per cell (env scoped at invocation time), and have the
driver loop cells. 6 binary invocations × 3 runs each = 18 (load + 3 ×
(prefill + 32 decodes)) total. Within each invocation, the 3 runs reuse the
loaded model (cheap re-clone of session). The model+harness boot is the
expensive part (~3-5s per cell incl Executive load); harness emits a single
JSON object per invocation appended to the shared report file.

## Substantive gates

1. **T_NTT_BENCH_ALL_CELLS_COMPLETE** — 6/6 cells run to completion with
   non-NaN toks/sec for both prefill + decode across all 3 reps. Failure
   modes that count as cell-failure: prefill returns non-SP_OK; decode
   returns non-SP_OK in fewer than 8 steps (allows partial measurement);
   logits contain NaN/Inf.

2. **T_NTT_BENCH_FP32_BASELINE_CAPTURED** — cells 1 and 4 (fp32 for both
   models) run successfully; prefill + decode toks/sec reported.

3. **T_NTT_BENCH_NTT_HOST_VS_HEX_BOTH_RUN** — cells 2,3,5,6 all complete.
   Comparing the two pairs (Exec B vs Exec C, Memo B vs Memo C) tells us
   whether the Hex backend wall-clock-wins at ctx ≤ 64 on a real forward
   pass. Expected: hex slower than host at this small ctx (per NTT.5b
   wall-clock matrix: 1.89× slower per-inner-product at N=128, and the
   forward calls Bluestein many times per layer).

4. **T_NTT_BENCH_REPORT_LANDS** — `ntt_bench_toks_report.json` + closure
   markdown table both written and committed. Headline 2×3 table is the
   first thing in CLOSURE-NTT-bench.md.

If any cell errors or surfaces unexpected behavior (e.g., NTT-attention
overlay silently disabled for some HD; Executive backend register no-op
per Stage 0 discovery), surface UPSTREAM per
`feedback-no-silent-gate-revisions`. Do NOT silently exclude cells.

## Files

**NEW (this sprint adds):**

- `tools/sp_daemon/src/bin/sp_ntt_bench_toks.rs` (~300-400 LOC):
  measurement harness; takes `--cell N` (1..=6) flag, loads the appropriate
  model, runs 3 reps of (prefill 16 toks + decode 32 toks), emits a JSON
  fragment to stdout AND appends to the report file. Cell metadata
  (model name, config, env-var snapshot) embedded in the fragment.

- `tools/sp_daemon/scripts/ntt_bench_toks_run.ps1`: PowerShell driver
  that pushes the freshly-built binary, then loops cells 1..=6 setting
  env vars per cell. Captures stdout to `ntt_bench_toks_run.txt`.
  Pulls accumulated `ntt_bench_toks_report.json` back to host.

- `tools/sp_daemon/scripts/ntt_bench_toks_run.txt`: verbatim adb run
  capture (Stage 2 output).

- `tools/sp_daemon/scripts/ntt_bench_toks_report.json`: per-cell
  measurement objects.

- `tools/sp_compute_skel/docs/CLOSURE-NTT-bench.md`: closure with the
  headline table.

**EDIT:**

- `tools/sp_daemon/Cargo.toml`: add `[[bin]] name = "sp_ntt_bench_toks"`
  declaration.

**FILES NOT TOUCHED (anti-contamination):**

- All NTT.5a/5b/5c surfaces (poly_ring_bluestein.h/.c, sp_l1.h §5,
  sp_session.c §5, forward.c, qwen25.c). Bench USES, doesn't MODIFY.
- All math-core code. The math-core submodule stays pinned at NTT.5c tip
  (ce93b9c).
- Any other engine-* worktree or shannon-prime-lattice surface.
- `tools/sp_daemon/src/daemon.rs`, `state.rs`, `session.rs` — daemon
  proper is irrelevant to this bench (bench is a standalone smoke binary).

## Workflow discipline (per spec)

1. **Plan-commit first.** This file + Stage 0 citations. Commit as
   `[plan] NTT-bench — tokens/sec measurement on S22U`.

2. **Multi-file stage commits:**
   - Stage 1: smoke harness scaffold + Cargo.toml bin decl + host build
     compile-check (host stub prints "android-only"). Commit:
     `[NTT-bench] Stage 1: smoke harness scaffold + host build PASS`.
   - Stage 2: android cross-build + adb push + on-device cells 1-6 run +
     JSON report captured. Commit:
     `[NTT-bench] Stage 2: on-device measurement — 6 cells × 3 reps`.
   - Stage 3: closure with the headline table. Commit:
     `[NTT-bench] Stage 3: closure — headline tokens/sec table`.

3. **No silent gate revisions.** If config C is no-op for Executive per
   Stage 0 discovery, report explicitly in closure and surface upstream.

4. **Anti-contamination strict.** `engine-ntt-bench` only.

5. **Hardware: Knack's S22U.** `adb devices` confirmed at Stage 0; re-confirm
   before each Stage 2 invocation.

6. **HEADLINE: the toks/s table.** First thing in CLOSURE-NTT-bench.md.

## Risk register

- **Env var caching:** if math-core reads `SP_ENGINE_NTT_ATTN` once at
  process start (g_ntt_attn static init), the per-cell invocation pattern
  above is the right call. (Alternative was set/unset between cells within
  one process; rejected as fragile.)
- **Executive HD=128 vs HD=64 ambiguity:** resolved at runtime by reading
  `arch.head_dim`. Bench reports the actual value and notes when config C
  is a no-op.
- **EOS in decode loop:** synthetic integer IDs unlikely to hit EOS. If
  they do for some model, partial decode_N reported; closure documents.
- **Cold cache vs warm cache:** explicit — report rep 1 vs reps 2-3
  separately as well as mean.
- **`sp_decode_step` signature:** per `lib/shannon-prime-system/include/sp/sp_l1.h:146`:
  `sp_status sp_decode_step(sp_session *s, int32_t token, float *logits,
  size_t logits_capacity);`. Single token advance, writes that token's
  logits. Confirmed.
- **Session state across reps:** within a single invocation's 3 reps, each
  rep gets a fresh session via clone (M.1 pattern). No KV reuse between
  reps; each rep is independent.

## Sub-tag candidate

`lat-phase-4-ntt-bench-toks-baseline`. Operator applies post-merge.

## What's NOT done (in-scope-but-deferred declaration)

Per spec § "Closure deliverables" item 9:

- **NTT.6 long-context (ctx > 256 via tiling).** This bench operates at
  ctx ≤ 64 (16 prompt + 32 decode + headroom). The asymptotic O(N log N)
  win from Bluestein/Hex shows at long ctx; NTT.6 is the sprint that
  measures it.
- **Executive routing optimization.** If Executive HD=128 (per NTT.5c
  CLOSURE), config C for Executive cannot route through Hex. NTT.5d adds
  `sp_pr_set_backend` for the direct path; out of NTT-bench scope.
- **Per-layer breakdown.** Not in scope; the bench is end-to-end forward
  wall-clock, not per-layer.
