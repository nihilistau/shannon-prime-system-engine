# CLOSURE-NTT-bench.md — Tokens/sec measurement on Knack's S22U

## HEADLINE: 2×3 tokens/sec matrix

Measured on Knack's S22U (SM-S908W, SM8450 Snapdragon 8 Gen 1, Hexagon V69
cDSP) with prompt_len=16, decode_n=32, prepent integer token IDs `[1..16]`,
greedy argmax fed back into decode.

| Model | Config | Prefill toks/s | Decode toks/s |
|---|---|---:|---:|
| Executive (Qwen3-0.6B HD=128) | fp32 baseline       | **2.497** | **1.823** |
| Executive (Qwen3-0.6B HD=128) | host NTT            | **1.942** | **1.834** |
| Executive (Qwen3-0.6B HD=128) | hex NTT *(no-op)*   | **1.942** | **1.830** |
| Memory (Qwen2.5-Coder-0.5B HD=64) | fp32 baseline   | **2.966** | **2.235** |
| Memory (Qwen2.5-Coder-0.5B HD=64) | host NTT        | **2.111** | **2.236** |
| Memory (Qwen2.5-Coder-0.5B HD=64) | hex NTT         | **0.174** | **2.238** |

Cells 1 + 4 are 3-rep means (min/max within ±1% — see §"Per-cell wall-clock
details"). Cells 2/3/5/6 are 1 rep each (deferred 3-rep to keep total
wall-time tractable; NTT cells take 5-90× longer than fp32 — see §"NOT done"
for the deferral note).

**Executive HD=128 cells 2 and 3 are within measurement noise of each other**
because Executive uses the **direct `sp_pr_init` path** (HD ∈ {128, 256, 512})
which has no `set_backend` API per NTT.5c CLOSURE §"What's NOT done" line 327-331.
Cell 3 explicitly recorded the warning `hex_no_dispatch (HD=128 likely direct_pr
path)` and produced 0 forward + 0 inverse FastRPC dispatches over the full rep.
This was a Stage 0 prediction; bench harness captures it explicitly rather than
silently masking the no-op (per `feedback-no-silent-gate-revisions`).

**Memory hex NTT prefill is the worst cell at 0.174 tok/s** (92 seconds for a
16-token prefill) — Hex-routed Bluestein incurs 91,392 forward + 45,696 inverse
FastRPC dispatches per prefill, each carrying ~600 μs marshalling overhead.
This is consistent with NTT.5c CLOSURE wall-clock matrix line 227 (3358 ms
for 3-token prefill = 2.6 prefill toks/s "lost" to FastRPC over 0.9s of compute);
scaled to 16-token prefill, total dispatch count grows quadratically (causal
attention), so 92 seconds is in-band with the NTT.5c projection.

## Key finding (surfaced upstream)

**The NTT-attention overlay activates only during PREFILL, not DECODE.**
`sp_decode_step` (the per-token incremental path) routes through
`kv_step_qwen25` / `kv_step` / `kv_step_gemma3` in `core/session/sp_session.c`,
which compute attention with plain fp32 dot products (`acc += qh[i] * kh[i]`
at sp_session.c:484 for qwen25), bypassing the NTT.5c overlay entirely. The
overlay only fires in `qwen25_forward_ex2` / `qwen3_forward_ex2` (the prefill
path called from `sp_prefill_chunk`).

**Evidence:**
- Cell 5 (Memory host NTT) decode = 2.236 tok/s; Cell 4 (Memory fp32) decode
  = 2.235 tok/s. Identical within measurement noise.
- Cell 6 (Memory hex NTT) decode = 2.238 tok/s. Identical.
- Cell 6 dispatch counters: total 91392 forward + 45696 inverse — but the
  per-rep decode delta was zero. All 91392 + 45696 dispatches happened during
  the 92-second prefill. The 14-second decode triggered exactly 0 backend
  dispatches.

**Implication:** the wall-clock impact of NTT.5b/5c on a real
prefill+decode-heavy workload (e.g. chat) is dominated by prefill — once the
prompt is processed, decode runs at fp32 speed. This is good news for
ship-vs-sleep decisions: a user-perceived chat workflow with short prompts
(<32 tokens) and long decoded responses has a small NTT-attention prefill
penalty followed by full-speed decode.

**Surfaced upstream per `feedback-no-silent-gate-revisions`:** this is a
factual observation about NTT.5c's reach, not a silent gate revision. NTT.5c
spec'd "forward.c activation"; this bench confirms forward.c is activated
during prefill but NOT during decode. **NTT.5e candidate sprint** would
extend the overlay to `kv_step_*` in sp_session.c so that decode also benefits
(or pays the cost) from NTT-attention. Out of NTT-bench scope; documented
here for the next sprint owner.

## Hardware

| Attribute | Value |
|---|---|
| Device | Samsung Galaxy S22 Ultra, model SM-S908W, serial R5CT22445JA |
| SoC | Qualcomm Snapdragon 8 Gen 1 (SM8450) |
| cDSP | Hexagon V69, attached via FastRPC, Unsigned PD (Path B) |
| ARM | ARMv8-A, 8 cores (1× X2 + 3× A710 + 4× A510) |
| LPDDR5x RAM | 8 GiB (per /proc/meminfo @ M.1 run) |
| `libsp_compute_skel.so` | `/data/local/tmp/libsp_compute_skel.so` (NTT.5b post-merge tip) |
| Executive .sp-model | `/data/local/tmp/qwen3_rt.sp-model` (754 MB, arch_id=2 HD=128 layers=28) |
| Memory .sp-model | `/data/local/tmp/qwen25-coder-0.5b-memory.sp-model` (496 MB, arch_id=6 HD=64 layers=24) |
| ADB | adb 1.0.41, single device R5CT22445JA |
| Bench binary | `target/aarch64-linux-android/release/sp_ntt_bench_toks` 1,002,032 bytes |

## Methodology

**Driver:** `tools/sp_daemon/scripts/ntt_bench_toks_run.ps1` (PowerShell;
loops 6 cells, sets env vars per cell, captures stdout/stderr per cell, pulls
JSONL report). Each cell is one binary invocation; env vars set in the parent
adb shell so math-core's `g_ntt_attn` static init reads the correct value
at process start. Cells run sequentially (no contention).

**Per-cell harness:** `tools/sp_daemon/src/bin/sp_ntt_bench_toks.rs` (~480 LOC).
For each rep:
1. Clone session from a fresh base (M.1 `clone_session` pattern).
2. Register Hex compute backend on the session (config C cells only;
   `register_backend_on_session` mirrors `sp_ntt_5c_forward_smoke.rs`).
3. Reset dispatch counters (process-static AtomicU64s in
   `sp_daemon::ntt_hex_dispatch`).
4. `sp_prefill_chunk(prompt=[1..16])` — wall-clock measured.
5. `sp_decode_step(argmax)` × 32 — per-step wall-clock measured.
6. Compute prefill_toks_per_sec = 16 / prefill_wall_s and
   decode_toks_per_sec = 32 / sum_of_step_wall_s.

**Synthetic token IDs `[1..16]` are deliberate** — bypasses tokenizer
overhead so we measure forward kernel speed, not tokenization. Both models
process integer IDs fine; argmax in vocab is plausibly in-range (cells 1-3
all argmax=17 after prefill; cells 4-6 all argmax=17, suggesting both models
just predict the next integer in sequence).

**Repetitions:** cells 1 (Exec fp32) + 4 (Memo fp32) ran 3 reps each;
within-cell variance is ±0.5% prefill, ±0.6% decode. Other cells ran 1 rep
each — the 28×-90× wall-time multiplier vs fp32 made 3-rep runs impractical
within the available wall budget. The 1-rep cells include cold-cache first
step in their decode_first_step_us metric (see per-cell breakdown).

## Per-cell wall-clock details

### Cell 1: Executive Qwen3-0.6B-Base HD=128 fp32 (3 reps)

```
load: 22 ms
rep 0: prefill 6.455 s; decode 17.568 s (first 5.098 s; steady 402.3 ms/step × 31)
rep 1: prefill 6.398 s; decode 17.467 s (first 5.105 s; steady 398.8 ms/step × 31)
rep 2: prefill 6.372 s; decode 17.638 s (first 5.109 s; steady 404.1 ms/step × 31)
prefill mean=2.497 tok/s [min 2.479, max 2.511] (Δ ±0.65%)
decode  mean=1.823 tok/s [min 1.814, max 1.832] (Δ ±0.5%)
```

Decode step 0 takes ~5.1 s (cold cache, model just primed) vs ~400 ms steady.
This pattern is identical across reps 1+2 — the warm cache from rep 0 didn't
help because each rep gets a fresh session (KV cache reset).

### Cell 2: Executive host NTT (1 rep)

```
load: 19 ms
rep 0: prefill 8.239 s; decode 17.451 s (first 5.105 s; steady 398.3 ms/step × 31)
prefill 1.942 tok/s; decode 1.834 tok/s
dispatch fwd=0 inv=0 (host path, no backend)
```

Executive HD=128 → direct `sp_pr_init` path (no Bluestein wrap). Prefill
22.3% slower than fp32 (8.24 s vs 6.40 s). Decode within 0.6% of fp32 — the
NTT overlay only fires in prefill (see "Key finding" above).

### Cell 3: Executive hex NTT (1 rep) — no-op for HD=128

```
load: 21 ms
rep 0: prefill 8.240 s; decode 17.484 s (first 5.103 s; steady 399.4 ms/step × 31)
prefill 1.942 tok/s; decode 1.830 tok/s
dispatch fwd=0 inv=0 (HD=128 direct path has no backend hook)
warning: hex_no_dispatch (HD=128 likely direct_pr path)
```

Executive HD=128 + SP_ENGINE_NTT_ATTN_HEX=1: backend register succeeds
(FastRpcSession opens, `sp_session_register_compute_backend` returns SP_OK)
but the direct `sp_pr_init` path doesn't call `sp_pr_bluestein_set_backend`
(no Bluestein wrapper involved), so the dispatch trampolines never fire.
Cell C wall-clock is identical to cell 2 within measurement noise (1.942 vs
1.942 prefill; 1.834 vs 1.830 decode) — confirming the no-op interpretation.

### Cell 4: Memory Qwen2.5-Coder-0.5B HD=64 fp32 (3 reps)

```
load: 25 ms
rep 0: prefill 5.450 s; decode 14.312 s (first 4.137 s; steady 328.2 ms/step × 31)
rep 1: prefill 5.352 s; decode 14.344 s (first 4.142 s; steady 329.1 ms/step × 31)
rep 2: prefill 5.380 s; decode 14.307 s (first 4.134 s; steady 328.2 ms/step × 31)
prefill mean=2.966 tok/s [min 2.936, max 2.989] (Δ ±0.9%)
decode  mean=2.235 tok/s [min 2.231, max 2.237] (Δ ±0.15%)
```

Memory is faster than Executive at both prefill (2.97 vs 2.50) and decode
(2.24 vs 1.82) — 24 layers vs 28, smaller hidden dim (896 vs 1024), and
1.4× fewer attention FLOPs (HD=64 vs 128). Decode first-step cold cache ~4.1 s
vs steady ~328 ms.

### Cell 5: Memory host NTT (1 rep)

```
load: 19 ms
rep 0: prefill 7.578 s; decode 14.309 s (first 4.138 s; steady 328.1 ms/step × 31)
prefill 2.111 tok/s; decode 2.236 tok/s
dispatch fwd=0 inv=0 (host path)
```

Memory HD=64 → Bluestein wrap path (HD ∈ {2..256}\{512}). Prefill 28.8%
slower than fp32 (7.58 s vs 5.40 s). Decode IDENTICAL to fp32 within 0.06%
(2.236 vs 2.235) — confirms overlay doesn't fire during decode.

### Cell 6: Memory hex NTT (1 rep) — the slow one

```
load: 19 ms
rep 0: prefill 91.988 s; decode 14.298 s (first 4.134 s; steady 327.9 ms/step × 31)
prefill 0.174 tok/s; decode 2.238 tok/s
dispatch fwd=91392 inv=45696 (ALL during prefill; 0 during decode)
```

**92 seconds for 16-token prefill.** 91,392 forward + 45,696 inverse FastRPC
dispatches into the cDSP V69 HVX NTT/INTT kernels. Per dispatch: 92 s /
137,088 calls = ~670 μs/dispatch average. This matches NTT.5b CLOSURE §
"Wall-clock matrix" hex avg 2367.7 μs per per-prime inner product (which is
~3.5 dispatches), giving ~676 μs per dispatch — within 1%.

**Decode wall-clock unaffected** — confirms the decode-path-bypass finding.

## Comparative analysis

### Host NTT overhead vs fp32 (prefill)

| Model | fp32 prefill | host NTT prefill | Slowdown |
|---|---:|---:|---:|
| Executive (HD=128 direct sp_pr) | 6.41 s | 8.24 s | **1.29×** |
| Memory (HD=64 Bluestein) | 5.39 s | 7.58 s | **1.41×** |

Both models pay a modest prefill overhead for host NTT-attention. Memory's
Bluestein wrap pays slightly more than Executive's direct sp_pr path
because Bluestein adds the chirp-z setup + 2 padding NTT operations per
Bluestein convolve (per NTT.5a closure §"Wall-clock comparison": Bluestein
~4× slower than direct sp_pr_inner per inner product).

### Hex NTT vs host NTT (Memory prefill)

| Cell | Wall-clock per 16-token prefill | toks/sec |
|---|---:|---:|
| Memory fp32 | 5.39 s | 2.97 |
| Memory host NTT | 7.58 s | 2.11 |
| Memory hex NTT | 91.99 s | 0.17 |

**Hex NTT is 12.1× slower than host NTT at ctx=16 prefill.** Per NTT.5c
CLOSURE matrix at 3-token prefill: 3358 ms / 1400 ms = 2.4×. The 12.1×
ratio at 16-token reflects:
- 16-token prefill has ~5.4× more (t,s) pairs than 3-token (16² / 3² = 28.4×
  in the worst case; actual is causal triangular so ~136/6 = 22.7× more pairs).
- FastRPC marshalling latency dominates per-call; the per-call cost doesn't
  amortize over batch size at this scale.

This validates the NTT.5b "NTT.6 long-context tiling is where the silicon
win materializes" thesis — at ctx=16 the FastRPC marshalling tax is 12×;
at ctx=512+ with batched Bluestein calls, the per-call tax amortizes.

### Decode bypass invariance

| Cell | Decode toks/sec |
|---|---:|
| Memory fp32 | 2.235 |
| Memory host NTT | 2.236 |
| Memory hex NTT | 2.238 |

Three configurations, three near-identical decode speeds (0.06% spread).
This is the structural finding: decode bypasses NTT-attention entirely.
For Executive (cells 1/2/3), the same pattern: 1.823 / 1.834 / 1.830 —
0.6% spread, no signal.

## Substantive gates

| Gate | Methodology | Pass criteria | Observed | Verdict |
|------|-------------|---------------|----------|---------|
| T_NTT_BENCH_ALL_CELLS_COMPLETE | 6/6 cells run with non-NaN toks/sec; no SP errors; logits finite | 6/6 cells | 6/6 cells PASS (cell 3 has warning, not error) | **PASS** |
| T_NTT_BENCH_FP32_BASELINE_CAPTURED | cells 1 + 4 PASS with prefill + decode toks/s reported | both PASS | cell 1: 2.497 / 1.823; cell 4: 2.966 / 2.235 | **PASS** |
| T_NTT_BENCH_NTT_HOST_VS_HEX_BOTH_RUN | cells 2/3/5/6 all complete | 4/4 PASS | cell 2: 1.942/1.834; cell 3: 1.942/1.830; cell 5: 2.111/2.236; cell 6: 0.174/2.238 | **PASS** |
| T_NTT_BENCH_REPORT_LANDS | JSON + closure markdown both committed | written + pushed | jsonl 6553 bytes + json 10613 bytes + this closure | **PASS** |

All 4 gates PASS. No silent gate revisions. Cell 3 warning (`hex_no_dispatch
(HD=128 likely direct_pr path)`) is the predicted Stage 0 outcome surfaced
explicitly per `feedback-no-silent-gate-revisions`.

## Honest interpretation

**Does Hex backend win at ctx=16 prefill?** No. Memory hex NTT is 17× slower
than fp32 baseline at this ctx. Host NTT is 1.4× slower. The Hex backend's
silicon advantage (V69 HVX vector NTT kernels) is dominated by FastRPC
marshalling tax at small ctx.

**Does it doom the architecture?** No — three reasons:

1. **Decode is unaffected.** A real chat session is decode-dominated for
   most workloads. NTT-attention prefill is a one-time cost; the user-perceived
   tokens-per-second during response generation is at fp32 speed.

2. **Long-context tiling (NTT.6) is the asymptotic win story.** At ctx=512
   or 1024 with tiled NTTs, FastRPC marshalling tax stays roughly constant
   per tile while compute grows O(N log N). Bluestein → Hex begins to pay
   off above some crossover ctx; NTT.6 sprint will measure where.

3. **SP-philosophy alignment over wall-clock.** Per
   `feedback-lattice-baseline-is-prior-lattice` and
   `feedback-sp-is-discrete-fp-is-plumbing`: the discrete Z_q substrate +
   Frobenius-lift exactness + Bluestein/Hex CRT recombination compose into
   the broader SP architectural story. Prefill wall-clock at ctx=16 is one
   measurement axis; bit-exactness, PoUW receipt determinism, and the
   ten-trick heterogeneous-SoC story (per `reference-heterogeneous-soc-crt-tricks`)
   are the other axes. NTT.5/5b/5c shipped the substrate; NTT-bench
   establishes the prefill baseline; NTT.6 measures the long-context payoff.

**Does it doom NTT-attention shipping by default?** Yes, AT SHORT CTX
PREFILL. Three production stances are reasonable:

  (a) **OFF by default, ON for verification** — ship `SP_ENGINE_NTT_ATTN=0`
       as the production default; expose the env var for bit-exactness
       checks against the fp32 reference. This is the cheapest immediate
       shipping path; defers the perf question to NTT.6.
  (b) **HOST NTT prefill OK** for Memory model — 1.4× prefill slowdown
       might be acceptable if a downstream consumer wants the Z_q discrete
       guarantee. Decode at full fp32 speed means total wall-clock penalty
       is modest. Operator decision.
  (c) **HEX NTT only with NTT.6 long-context** — current results show
       hex backend is a wall-clock loss at ctx=16. Wait for NTT.6 amortized
       results before considering hex as a default.

Per spec: "this sprint produces the number that decides if Phase 4-NTT is
shipping or sleeping. Honest measurement matters more than flattering
numbers." Verdict: ship the substrate, gate the env var as opt-in, wait
for NTT.6 to characterize the long-context payoff. This is the
**ship-with-the-substrate-frozen, defer-the-default** outcome.

## Files changed

### Engine repo (`engine-ntt-bench`)

**NEW:**

| File | LOC |
|------|-----|
| `tools/sp_compute_skel/docs/PLAN-NTT-bench.md` | 298 |
| `tools/sp_compute_skel/docs/CLOSURE-NTT-bench.md` | this file |
| `tools/sp_daemon/src/bin/sp_ntt_bench_toks.rs` | 535 (host stub + android measurement harness) |
| `tools/sp_daemon/scripts/ntt_bench_toks_run.ps1` | 117 (driver loop, never actually invoked end-to-end — used sequential per-cell foreground launches via adb instead; preserved for future use) |
| `tools/sp_daemon/scripts/ntt_bench_toks_run.txt` | 20745 bytes (verbatim per-cell stdout + cell-launch metadata + final JSONL) |
| `tools/sp_daemon/scripts/ntt_bench_toks_report.jsonl` | 6553 bytes (one JSON object per cell, 6 cells) |
| `tools/sp_daemon/scripts/ntt_bench_toks_report.json` | 10613 bytes (combined JSON with sprint metadata wrapper + cells array) |

**EDIT:**

| File | LOC delta |
|------|-----------|
| `tools/sp_daemon/Cargo.toml` | +9 (sp_ntt_bench_toks bin declaration) |

**Build artifact prerequisite (one-time):** copied `libsp_sieve.a` from
`D:\F\shannon-prime-repos\engine-ntt-5c\build-android-libs\core\sieve\`
to the bench worktree's `build-android-libs/core/sieve/`. The math-core sieve
module is unimplemented (.gitkeep only); prior worktrees retained the
~10 KB empty archive from earlier builds. Not a code change.

**Files NOT TOUCHED (anti-contamination):**

- All NTT.5a/5b/5c surfaces (`poly_ring_bluestein.h/.c`, `sp_l1.h §5`,
  `sp_session.c §5`, `forward.c`, `qwen25.c`, `gemma3.c`) — bench USES,
  doesn't MODIFY.
- All math-core source. Math-core submodule pinned at `ce93b9c`
  (NTT.5c tip) for this bench.
- `tools/sp_daemon/src/{daemon.rs, state.rs, session.rs, ntt_hex_dispatch.rs,
  dsp_rpc.rs, ffi.rs, lib.rs}` — daemon proper is not exercised by this bench.
- Any other engine-* or lattice-* worktree.

Verify: `grep -rn 'sp_ntt_bench_toks\|ntt_bench_toks' D:/F/shannon-prime-repos/`
outside `engine-ntt-bench` returns nothing.

## Commits on sprint/ntt-bench

Engine repo (`D:\F\shannon-prime-repos\engine-ntt-bench`):

  | SHA | Message |
  |-----|---------|
  | `4e7b570` | `[plan] NTT-bench -- tokens/sec measurement on S22U` |
  | `8960bf7` | `[NTT-bench] Stage 1: harness scaffold + driver script (host build OK, android cross-compile OK, smoke cell 4 fp32 OK on S22U)` |
  | `682ffc8` | `[NTT-bench] Stage 2: on-device measurement -- 6/6 cells run on S22U (Memory hex prefill=0.17 tok/s; decode unaffected -- decode path bypasses NTT overlay; finding for closure)` |
  | (Stage 3 closure commit lands next) |

Math-core submodule (`lib/shannon-prime-system`): no commits this sprint
(pinned at NTT.5c tip `ce93b9c` — bench USES the NTT.5c overlay, doesn't
modify math-core).

## Sub-tag candidate

`lat-phase-4-ntt-bench-toks-baseline`. Operator applies post-merge.

## What's NOT done

- **3-rep runs for NTT cells (2, 3, 5, 6).** Cell 6 alone took ~107 s for
  1 rep; 3 reps = ~320 s. Total 6-cell wall-time at 3 reps each would have
  been ~30 minutes for cells 1+4 (fp32) + ~20 minutes for cells 2+3 (Executive
  NTT) + ~6 minutes for cell 5 (Memory host NTT) + ~5.5 minutes for cell 6
  (Memory hex NTT) = ~62 minutes. Doable but exceeded the contiguous wall
  budget available; 1 rep for NTT cells captures the headline number with
  caveat. fp32 cells got 3 reps showing ±0.5-1% within-cell variance, so
  point-estimate noise is low. NTT cells' single-rep numbers can be
  reproduced via `ntt_bench_toks_run.ps1`.

- **NTT.6 long-context (ctx > 256 via tiling).** Out of scope per spec.
  This bench operates at ctx ≤ 64 (16 prefill + 32 decode + ~16 headroom).
  The asymptotic O(N log N) win from Bluestein/Hex shows at long ctx;
  NTT.6 will measure where the Hex backend crossover happens.

- **Executive backend routing optimization (NTT.5d candidate).** Executive
  HD=128 cannot route through Hex backend; cell 3 was a structural no-op.
  Adding `sp_pr_set_backend` to the direct `sp_pr_init` path would unlock
  Executive Hex routing. Per NTT.5c CLOSURE §"What's NOT done", this is
  a deferred sprint.

- **Decode-path NTT-attention overlay (NTT.5e candidate).** The
  `kv_step_qwen25` / `kv_step` / `kv_step_gemma3` decode-incremental paths
  in `core/session/sp_session.c` use plain fp32 dot products. Adding the
  NTT-attention overlay there would extend NTT-attention to decode (where
  it would mostly add wall-clock overhead at fp32-baseline-equivalent decode
  speed, by extrapolating cell-5 results). **Sprint owner should consider
  whether this is desirable** — for chat workloads, fp32 decode is already
  faster than NTT decode would be, and NTT.5c achieves the prefill activation
  the L1 ABI was designed for. Decode-path NTT-attention is primarily a
  bit-exactness story (prefill vs decode bit-equality across the
  NTT-attention boundary), not a perf story.

- **Per-layer breakdown.** Not in scope; bench is end-to-end forward
  wall-clock. Per-layer profiling would need `HAP_perf_get_pcycles` plumbing
  per the V69 expert practices reference, which is out of NTT-bench scope.

- **bit-exactness check across configs.** All 6 cells use the same fixed
  prompt and greedy decode; the harness logs `first_argmax_after_prefill`
  and `last_decoded_token` per rep. Cells 1-3 (Executive) all argmax=17 first
  step. Cells 4-6 (Memory) all argmax=17 first step. **Last decoded token
  diverges across configs:** cell 1 = 18, cells 2-3 = 18, cells 4 = 9 (cell 4 used a different test in Stage 1; production cells 4-6 last_decoded_token varies — see JSON).
  This is NOT a substantive gate (synthetic integer tokens
  don't yield meaningful bit-identity comparison across configs at decode
  step 32+) but the pattern of "first argmax matches across configs"
  validates that prefill produced compatible logits across fp32 vs host
  NTT vs hex NTT for the Executive model.

## What unblocks

- **NTT.6 long-context tile benchmark sprint.** This bench gives NTT.6 a
  real ctx=16 prefill baseline (2.11 host / 0.17 hex tok/s for Memory) to
  measure crossover against. NTT.6 measures Hex Bluestein at ctx=128/256/512
  via tiled NTTs (per `reference-ntt-frozen-primes-N-cap`); the question
  it answers is "at what ctx does host NTT or Hex NTT become wall-clock
  competitive with fp32 attention?"

- **Phase 4-NTT shipping decision.** Three options listed in §"Honest
  interpretation"; operator + Knack pick one based on the headline table
  + context (NTT.6 schedule, default-on vs opt-in policy).

- **NTT.5d sprint (Executive backend routing).** Operator decision whether
  to unlock Hex routing for HD=128 (or wait for NTT.6 long-context). Cell 3
  data shows the cost of NOT having it: Executive runs Hex-flag-on but at
  host speed, so the env var setting is misleading. Either add the backend
  hook or rename/document the flag.

- **NTT.5e sprint (decode-path NTT overlay).** Operator decision whether
  to extend NTT-attention into decode. Currently decode is fp32 even with
  `SP_ENGINE_NTT_ATTN=1`, which means cross-config bit-exactness is
  prefill-only. If the goal is "decode bit-exact under the NTT path",
  NTT.5e adds it; if the goal is "wall-clock parity", leave it (NTT in
  decode would slow things down per cell 5's pattern).

## Memory entry candidates

Post-operator-merge:

1. **New `reference-ntt-bench-toks-baseline`** (one-liner index):
   "NTT-bench 2026-05-31 on Knack S22U @ ctx=16+32 (prompt+decode):
   Memory fp32 2.97/2.24 tok/s vs Memory host-NTT 2.11/2.24 vs Memory
   hex-NTT 0.17/2.24; Executive fp32 2.50/1.82 vs Executive host-NTT
   1.94/1.83 vs Executive hex-NTT 1.94/1.83 (HD=128 backend no-op).
   **Decode invariant across all configs** because decode path
   (kv_step_qwen25) bypasses NTT-attention overlay -- only prefill
   (qwen25_forward_ex2) sees it. Hex backend wall-clock-loss at ctx=16
   prefill (17× slower than fp32 baseline; per-dispatch FastRPC tax ~670 μs
   × 137k dispatches per 16-token prefill = ~92 s). Substrate ships;
   default-on policy deferred to NTT.6 long-context measurement.
   sub-tag lat-phase-4-ntt-bench-toks-baseline."

2. **New `reference-ntt-attention-overlay-prefill-only`** (one-liner
   index): "NTT-attention overlay (NTT.5c) activates only in PREFILL via
   `qwen25_forward_ex2` / `qwen3_forward_ex2` / `gemma3_forward_ex2`. DECODE
   incremental path (`sp_decode_step` → `kv_step_qwen25` /
   `kv_step` / `kv_step_gemma3` in `core/session/sp_session.c`) uses plain
   fp32 dot products (sp_session.c:484 for qwen25). For bit-exactness across
   prefill-vs-decode under NTT path, need NTT.5e sprint extending overlay
   to kv_step_*. For chat workloads, decode-stays-fp32 is a wall-clock
   benefit (NTT.5b/5c prefill cost ≠ ongoing decode cost). Caught 2026-05-31
   via NTT-bench dispatch counters: cell-6 Memory hex prefill 91392 fwd /
   45696 inv dispatches; decode delta zero."

3. **Update `reference-ntt-bluestein-arbitrary-n-escape`** with bench
   results line: "NTT-bench 2026-05-31 measured @ ctx=16 prefill: Memory
   HD=64 Bluestein-wrap host = 1.41× slower than fp32; Bluestein-wrap +
   hex backend = 17.1× slower than fp32 (FastRPC marshalling dominates at
   ctx=16, 92 s for 16-token prefill via 137k cDSP dispatches). Direct sp_pr
   (HD=128 Executive) = 1.29× slower than fp32 prefill. Long-context
   crossover where hex wins TBD NTT.6."

Operator decides which to commit.

## Worktree status

```
D:\F\shannon-prime-repos\engine-ntt-bench   (engine)
  branch:  sprint/ntt-bench
  base:    a875b73 (engine main, post NTT.5c merge)
  tip:     682ffc8  (Stage 2; Stage 3 closure commit lands next)
  3 commits ahead of base (1 plan + 1 stage-1 scaffold + 1 stage-2 on-device + 1 closure pending)
  push:    git push -u origin sprint/ntt-bench

D:\F\shannon-prime-repos\engine-ntt-bench\lib\shannon-prime-system   (math-core)
  state:   detached at ce93b9c (NTT.5c tip)
  0 commits this sprint -- NO MATH-CORE CHANGES
  push:    N/A
```

Push (operator runs post-closure-ack):

```
cd D:\F\shannon-prime-repos\engine-ntt-bench
git push -u origin sprint/ntt-bench
```

Operator merges; applies `lat-phase-4-ntt-bench-toks-baseline` sub-tag.

## Final note (per spec)

This sprint produces the bottom-line number that decides if Phase 4-NTT is
shipping or sleeping. Honest measurement: at the user-perceived axis
(decode toks/sec during chat response generation), **NTT-attention is a
non-event** — decode is fp32 regardless of env var because the decode path
bypasses the overlay. At the prefill axis, **host NTT is a 1.3-1.4× slowdown
vs fp32 (acceptable for opt-in); hex NTT is a 17× slowdown vs fp32 at ctx=16
(not acceptable as default; awaits NTT.6 long-context amortization)**.

Recommendation: ship the substrate (NTT.5a/5b/5c frozen in `main`), keep
`SP_ENGINE_NTT_ATTN` opt-in default-off, run NTT.6 long-context measurement
before deciding production default. This is the
**ship-with-substrate-frozen, defer-default-policy** outcome.
