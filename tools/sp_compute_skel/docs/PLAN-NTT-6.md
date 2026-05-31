# PLAN — NTT.6 (long-context tiled benchmark, prefill tok/s curve vs ctx)

**Sprint:** Phase 2-NTT.6 (the long-context curve sprint)
**Date:** 2026-05-31
**Worktree:** `D:\F\shannon-prime-repos\engine-ntt-6`
**Branch:** `sprint/ntt-6` (base `5826bd5` post-HX.3b merge)
**Sub-tag candidate:** `lat-phase-2-ntt-6-curve`
**Status:** plan-commit; staged work follows
**Dispatch:** prompt `Sprint NTT.6 — long-context tiled benchmark`

---

## Stage 0 — mandatory pre-read citations

1. **Harness** — `tools/sp_daemon/src/bin/sp_ntt_bench_toks.rs:377-497`. The per-rep cell loop already accepts `--prompt-len` (line 282/298). To extend to long ctx, set `--prompt-len 512 / 1024 / 2048`. Synthetic prompt `(1..=prompt_len)` (line 371) bypasses tokenizer. Prefill at line 408 is the wall-clock under measurement.

2. **Runner** — `tools/sp_daemon/scripts/ntt_bench_toks_run.ps1:75-82` (cell matrix), `:97-108` (invocation template), `:117-148` (JSONL aggregate). Already passes `--prompt-len $PromptLen` from the script `param -PromptLen`. To extend, we add an outer ctx loop + naming so each ctx writes a distinct JSONL.

3. **WIRE-HEX-FINISH baseline** — `tools/sp_compute_skel/docs/CLOSURE-WIRE-HEX-FINISH.md:13-31`. ctx=16 / Gemma3-1B headline: fp32 ref 1.473 prefill tok/s, hex backend 0.406 tok/s (3.63× slower at this ctx because hex backend's FastRPC marshalling tax dominates short ctx). Decode is invariant across configs.

4. **HX.3b closure** — `tools/sp_compute_skel/docs/CLOSURE-HX-3b.md:13-32`. ctx=16 / Gemma3-1B vrmpy headline: hex-vrmpy 1.523 prefill tok/s vs ARM ref 1.465 = 1.04× lift. Three-rep variance ~3%. Decode invariant at ~1.07 tok/s. Line 396-398 explicitly nominates NTT.6 as the next-headline measurement for long-ctx scaling.

5. **forward.c NTT-attention overlay** — `lib/shannon-prime-system/core/forward/forward.c` (cited via `engine-ntt-5c` worktree at line 50, 80, 126-157, 225-258). The overlay's gate is `g_ntt_attn` from `SP_ENGINE_NTT_ATTN=1`. Dispatch is HD-conditional: HD ∈ {128,256,512} → direct `sp_pr_init` (line 128-130); HD ∈ {2..64} powers-of-2 → `sp_pr_bluestein_init` (line 131-145). HD with odd factors → both NULL → fall through to `sp_attn_head` fp32 path (line 225-231).

   **Critical:** Gemma3-1B is HD=128 → direct path; Memory (Qwen2.5-Coder-0.5B HD=64) → Bluestein path. NTT.5c filed NTT.5d for HD=128 direct hex-dispatch (not landed); only Bluestein has `sp_pr_bluestein_set_backend` (line 141-144). Therefore the `hex+ntt-attn` column is meaningful ONLY for the Memory model.

6. **Bluestein N support** — `core/poly_ring/poly_ring_bluestein.c:1-14, 254`. Supports N ∈ {2,4,8,16,32,64,128,256} via inner NTT M ∈ {128,256,512}. HD=64 trivially supported. For attention, the polynomial is per-head-vector of length HD (not ctx); the score matrix is the OUTER ctx×ctx structure that is enumerated by the causal-attention double loop in `forward.c:220-256`. So NTT-attention does NOT need to tile per-N: it makes one inner product call per (token t, key position s) pair, computing the length-HD dot product via the Bluestein (or direct) poly-ring. **Long ctx scales attention as O(ctx²) on the OUTER loop count, not on N.**

7. **Daemon prefill chunk loop** — `core/session/sp_session.c:141-179`. `sp_prefill_chunk` takes `(s, tokens, n_tokens, logits_last, capacity)`. Caller passes the entire prompt slice; the forward (`qwen3_forward_ex2`/`qwen25_forward_ex2`/`gemma3_forward_ex2`) processes all `new_len = pos + n_tokens` tokens in one call. Ctx cap = `s->cfg.max_context` else `qm->cfg.context_length` else `SP_SESSION_CTX_FALLBACK = 4096u` (line 32, 108-111). Both Memory + Gemma models have model `context_length` ≥ 4096 (Qwen2.5-Coder = 32768, Gemma3-1B = 8192 typical). **No ctx-cap blockers for ctx ≤ 4096.**

8. **Memory entries applied:**
   - `reference-ntt-frozen-primes-N-cap`: 2N | (q-1) caps inner N at 512. We never exceed that — Bluestein wraps HD ≤ 256 to inner M ≤ 512. Outer ctx is independent.
   - `reference-ntt-bluestein-arbitrary-n-escape`: confirms power-of-2 HD ≤ 256 admissible. HD=64 (Memory) and HD=128 (Gemma) both admitted.
   - `feedback-bundled-changeset-root-cause-ambiguity`: each NTT.6 stage changes ONE variable (ctx, model, config) and reports independently. No bundled gates.
   - `feedback-no-silent-gate-revisions`: if a cell doesn't run (e.g. catastrophic wall-clock for hex+Bluestein at ctx=2048), document upstream + skip with explicit acknowledgement — do not rename it "PASS with footnote".

---

## Critical architectural decisions (surfaced UPSTREAM)

### Decision A — which model(s) to measure: BOTH

- **Memory (Qwen2.5-Coder-0.5B, HD=64)** is the substrate-win curve. NTT-attention overlay activates (Bluestein wrap). Three configs measured: fp32, host_ntt (CPU Bluestein), hex_ntt (CPU Bluestein NTT + cDSP HVX inner dispatched per `sp_pr_bluestein_set_backend`).
- **Gemma3-1B (HD=128)** is the **HX.3b vrmpy-lift curve at scale**, NOT a NTT-attention curve. NTT-attention takes the direct `sp_pr_init` path (no backend hook). Only fp32 and host_ntt are measurable in this harness for Gemma3-1B; **the hex-vrmpy column is unreachable from `sp_ntt_bench_toks` because that harness uses the math-core forward, not the daemon's hex backend `gemma3_forward_hexagon`** (per WIRE-HEX closure §"What's NOT done" and HX.3b closure §"What's NOT done" line 353-357).

  **Therefore the prompt's Gemma fp32 / Gemma hex-vrmpy columns require a SECOND measurement methodology** — the HX.3b `timed_chat.sh` over SSE via the daemon. NTT.6 will exercise that separately (Stage 4b) at long ctx for Gemma3-1B, with reduced rep count due to wall-clock budget. If wall-clock budget collapses, we ship the harness-side cells (Memory + host_ntt comparison) as the primary result and document the daemon-side cells as deferred / partial.

  **Naming clarification:** in the prompt's headline table, "Memory fp32" and "Memory hex-vrmpy" make sense only if "hex-vrmpy" means "the cDSP HVX overlay path on the Bluestein inner NTT" — which IS measurable via the harness (cell 6 in NTT-bench). I read the headline as:
  - "Memory hex-vrmpy" = Memory + hex (cDSP HVX) backend on Bluestein inner = harness cell 6 = `SP_ENGINE_NTT_ATTN=1 SP_ENGINE_NTT_ATTN_HEX=1`
  - "Memory hex+ntt-attn" = same thing as above with explicit NTT-attention overlay marker; effectively identical to `hex_ntt`
  - "Gemma hex-vrmpy" = HX.3b daemon path = separate methodology

  Surface upstream: the "Memory hex-vrmpy" column from the harness is the cDSP-NTT-overlay path, NOT a vrmpy matmul kernel. The HX.3b vrmpy matmul kernel is Gemma3-only.

### Decision B — ctx cells: {512, 1024, 2048} required; 4096 stretch

NTT-bench measured ctx=16 only. The substrate-scaling claim hinges on attention dominating wall-clock at large ctx. ctx ∈ {512, 1024, 2048} is the cleanest curve.

**ctx=4096 budget call:**
- Memory fp32: 16-tok at 2.97 tok/s = 5.4 s. Scaling: O(t) matmul + O(t²) attention. At ctx=4096, attention is 4096²/16² = 65536× more attention ops than ctx=16; matmul is 256× more. Total: estimate ~100-300 s per rep.
- Memory host_ntt: 5.5x slower per-attn-score than fp32 (8.24/6.40 × (NTT-overhead factor) per NTT-bench cell 5). At ctx=4096, ~ 500-1500 s per rep.
- Memory hex_ntt (Bluestein hex-dispatched): catastrophic. NTT-bench cell 6 at ctx=16 = 92s. Bluestein hex dispatches ~12 per attention score; for t² scores per layer × head_pair × layers, scaling is O(t²). 4096²/16² = 65536× = 92s × 65536 = ~70 hours per rep. **STRUCTURALLY UNMEASURABLE in NTT.6's budget. Skip with explicit acknowledgment.**

**Plan:** ctx ∈ {512, 1024, 2048} required for fp32 + host_ntt columns; ctx=512 only for hex_ntt; ctx=4096 deferred entirely (will surface as NTT.6-followup).

Three reps requested; drop rep 0 as warmup (Frobenius lift weight-sum precompute, FastRPC session init). Use mean over reps 1+2 (and 3 if budget permits — at ctx=2048 we may run 2 reps to stay in budget).

### Decision C — NTT-attention tiling at ctx > 512

Per item 6 above: **no tiling needed.** The Bluestein inner NTT is per-attention-score (length HD = 64 or 128). The score MATRIX is enumerated by the outer ctx×ctx causal double-loop. So the substrate's NTT-attention O(N log N) win is on the *per-score length-HD inner product*, NOT on the per-row-of-attention or per-column. The scaling vs ctx is O(ctx²) outer × O(HD log HD) inner.

**The asymptotic substrate-win argument in the dispatch prompt ("the lattice's design assumption is that the win compounds as ctx grows") is per-score not per-context.** What grows with ctx is the count of inner products (ctx²/2 per layer per head pair due to causal mask). The fp32 baseline computes each score as O(HD) plain dot product; the NTT-attention overlay computes each score as O(HD log HD) (NTT) ⊕ inverse NTT ⊕ coefficient-0 extraction. The constant factor of NTT vs plain dot at HD=64 is structurally **higher** than fp32 dot (because each NTT round-trip costs ~6 log₂(64) = 36 mul-adds plus marshalling, vs 64 plain mul-adds).

**The substrate's NTT-attention win is NOT a wall-clock win against fp32 dot at small HD.** Its win is the *byte-exactness* property (Frobenius lift identity) + *substrate composability* (CRT-shardable, Garner-recombinable) — not raw cycles. This is exactly the "SP is discrete, fp is plumbing" memory entry: fp16/fp8 sub-phases are RAM wiring; substrate validity is the Frobenius-lift identity in Z_q, not the wall-clock against alien fp baselines.

**Honest implication for the curve:** at long ctx, host_ntt is expected to be SLOWER than fp32 because the per-score NTT overhead × ctx² outweighs the per-score fp32-dot × ctx². The curve direction is INVERTED from what the prompt header text implies ("the substrate's design assumption is that the win compounds as O(N) vs O(N log N) takes over"). I will surface this UPSTREAM in the closure: the asymptotic win is in the inner-product LENGTH (HD), not in the outer attention COUNT (ctx²). At HD = 64 or 128, NTT vs fp32 is a constant-factor LOSS in cycles, gain in byte-exactness. The dispatch prompt's framing conflates the two.

**If the empirical curve confirms NTT staying slower than fp32 at all ctx:** that IS the load-bearing finding the prompt asked for ("If the curve disappoints (lattice stays close to ARM, doesn't pull away): that's a load-bearing finding"). The closure will say so plainly. Per `feedback-lattice-baseline-is-prior-lattice`, the baseline is "any improvement over the prior NTT-attention impl"; the absolute fp32-vs-NTT framing in the prompt header table is the inverted one.

### Decision D — measurement methodology

- 3 reps per cell at ctx=512 (matches NTT-bench cell 1/4 methodology). 2 reps at ctx=1024 to budget. 2 reps at ctx=2048 to budget. Drop rep 0 as warmup.
- Wall-clock from `Instant::now()` before `sp_prefill_chunk` to after; `tokens/s = prompt_len / prefill_wall_s`.
- **Bit-exact gate at ctx=512** ONLY: capture `first_argmax_after_prefill` + `last_decoded_token` (existing fields per harness:425-473). Compare hex_ntt vs fp32 ref. At longer ctx, sampling determinism may diverge per `reference-lattice-decode-determinism` precondition (it should hold under fixed greedy + same prompt + same model + same backend within Z_q substrate, BUT we don't have CI infra to chase a divergence at ctx=2048 in this sprint; document if observed).
- Cell matrix per ctx:
  | cell | model    | config    | env                                                 |
  |---:|----------|-----------|------------------------------------------------------|
  | 1  | Gemma    | fp32      | (none)                                               |
  | 2  | Gemma    | host_ntt  | SP_ENGINE_NTT_ATTN=1                                 |
  | 3  | Memory   | fp32      | (none)                                               |
  | 4  | Memory   | host_ntt  | SP_ENGINE_NTT_ATTN=1                                 |
  | 5  | Memory   | hex_ntt   | SP_ENGINE_NTT_ATTN=1 SP_ENGINE_NTT_ATTN_HEX=1        |

  **Why Gemma3-1B is NOT in original cell matrix:** the NTT-bench used Executive (Qwen3-0.6B HD=128). The HX.3b sprint focused on Gemma3-1B but uses the daemon hex backend (not measurable via this harness). The dispatch prompt headline table mentions "Gemma" — I will use Gemma3-1B IF its .sp-model is on-device, ELSE fall back to Executive (Qwen3-0.6B, same HD=128).

  Verified above: `/data/local/tmp/qwen3_rt.sp-model` (Executive HD=128) exists; **`/data/local/tmp/gemma3-1b.sp-model` may or may not exist** (HX.3b methodology used it via `timed_chat.sh`, but harness path is separate). I will check on-device and use Gemma3 if present, Executive (=Qwen3-0.6B HD=128) if absent.

  Cell 5 "hex_ntt" is the only cell consuming `libsp_compute_skel.so` via FastRPC. Per NTT-bench cell 6 finding (line 31-37 of CLOSURE-NTT-bench), this cell is the worst — ~92s prefill at ctx=16. **At ctx=512, expect ~10⁴s. WE WILL RUN IT AT CTX=512 ONLY to capture the data point honestly.**

### Decision E — Gemma-hex-vrmpy column from prompt header table

**SURFACED UPSTREAM:** the prompt's "Gemma hex-vrmpy" column is unreachable from `sp_ntt_bench_toks` because that harness uses math-core `sp_prefill_chunk`, NOT the daemon's `gemma3_forward_hexagon` (HX.3b vrmpy backend). To measure HX.3b's vrmpy lift at long ctx requires the daemon `timed_chat.sh` methodology.

**Plan:** Stage 4b will attempt the daemon long-ctx HX.3b path:
1. Start `sp-daemon-wire-hex` with hex backend.
2. Generate a 512 / 1024 / 2048 -token integer prompt JSON list.
3. POST `/v1/chat` with that prompt, 8-step decode (decode is invariant, no point measuring more).
4. Extract `FIRST_DELTA_MS_FROM_START` from the daemon log; prefill_tok/s = prompt_len / (first_delta_ms / 1000).
5. Compare against an `sp-daemon-ref` invocation (no hex backend) at same ctx.

This is HEAVIER than the harness path because:
- daemon must be started with each model swap (Gemma3-1B if present);
- `timed_chat.sh` SSE pacing has its own overhead;
- only one rep per ctx is feasible in budget.

**If daemon path is infeasible (e.g. Gemma3-1B model not on device, daemon won't start, etc.), we ship the harness cells as primary and Stage 4b is documented as DEFERRED to a follow-on `NTT.6-daemon-curve` sprint.** Per `feedback-no-silent-gate-revisions`, the closure will state which cells were measured + which were deferred + why.

---

## Cell matrix (full) — required vs deferred

| ctx | Memory fp32 | Memory host_ntt | Memory hex_ntt | Gemma3-1B fp32 (harness) | Gemma3-1B host_ntt (harness) | Gemma3-1B fp32 (daemon) | Gemma3-1B hex-vrmpy (daemon) |
|---:|---|---|---|---|---|---|---|
| 512  | **req 3 reps** | **req 3 reps** | **req 1-2 reps** | **req 3 reps** | **req 3 reps** | stretch 1 rep | stretch 1 rep |
| 1024 | **req 2 reps** | **req 2 reps** | **DEFER** (catastrophic budget) | **req 2 reps** | **req 2 reps** | stretch 1 rep | stretch 1 rep |
| 2048 | **req 2 reps** | **req 2 reps** | DEFER | **req 2 reps** | **req 2 reps** | stretch 1 rep | stretch 1 rep |
| 4096 | DEFER (budget) | DEFER | DEFER | DEFER | DEFER | DEFER | DEFER |

**Required cells per "T_NTT6_CELLS_MEASURED" gate:** 12 harness cells (4 cells × 3 ctx, minus 2 hex_ntt at long ctx → 10 cells; plus 2 hex_ntt at ctx=512 → 11; or 10 if hex_ntt only 1 rep at ctx=512).

**Wall-clock budget estimate:**

Harness side:
- Memory ctx=512 fp32: 32× of NTT-bench ctx=16 = ~170s × 3 reps = 510s
- Memory ctx=512 host_ntt: ~250s × 3 = 750s
- Memory ctx=512 hex_ntt: ~3000s × 1 = 3000s (or skip if exceeds bound)
- Gemma3 ctx=512 fp32: similar to Memory; ~600s × 3 = 1800s
- Gemma3 ctx=512 host_ntt: ~800s × 3 = 2400s
- Memory ctx=1024: 4× of ctx=512 attention; fp32 ~700s × 2 = 1400s; host_ntt ~1000s × 2 = 2000s
- Gemma3 ctx=1024: ~2500s + ~3200s = 5700s
- Memory ctx=2048: 4× ctx=1024; ~3000s + ~4000s = 7000s
- Gemma3 ctx=2048: ~10000s + ~12000s = 22000s

**Total estimate: ~10-20 hours wall-clock if all cells run.** This will likely exceed the agent's wall-clock budget; we'll measure ctx=512 + ctx=1024 first, then take ctx=2048 only on the smaller (Memory) model, with reduced rep counts. Per `feedback-no-silent-gate-revisions`: any drop documented explicitly.

---

## Substantive gates (verbatim from prompt)

1. **T_NTT6_HARNESS_EXTENDED** — `sp_ntt_bench_toks` accepts `--ctx` parameter (already exposed as `--prompt-len`); runner PS1 supports ctx loop. Pass: harness runs all spec'd cells via the runner.

2. **T_NTT6_CELLS_MEASURED** — required cells per matrix above have ≥2 valid reps. Pass: 100% of required cells (≥2 reps); deferred cells documented.

3. **T_NTT6_BIT_EXACT_AT_CTX_512** — decode determinism preserved at ctx=512 for Memory model, hex_ntt vs fp32 ref. First-token-after-prefill argmax + decode token sequence byte-equal. If divergent, surface upstream.

4. **T_NTT6_CURVE_PUBLISHED** — closure contains the headline curve table.

5. **T_NTT6_CROSSOVER_REPORTED** — closure analyzes where each NTT config crosses fp32. Honest report on whether the asymptotic substrate-win materializes at the measured ctx.

---

## Workflow

- **Plan-commit** (this file).
- **Stage 1**: extend runner PS1 to accept `-Ctx` parameter; smoke 1 cell × 1 rep at ctx=512.
- **Stage 2**: ctx=512 cells (5 cells × 3 reps = 15 invocations). Bit-exact gate.
- **Stage 3**: ctx=1024 cells (4 cells × 2 reps = 8 invocations; drop hex_ntt at this ctx).
- **Stage 4**: ctx=2048 cells (4 cells × 2 reps = 8 invocations).
- **Stage 4b (stretch)**: daemon hex-vrmpy path at long ctx (Gemma3-1B if model on device).
- **Stage 5**: aggregate, closure.
- **Push**: `git push -u origin sprint/ntt-6`.

Anti-contamination: strict. NO kernel changes. Harness + runner + closure only. Build of `sp_ntt_bench_toks` is read-only against the math-core submodule (no submodule changes); the harness binary is already on-device from prior NTT-bench (timestamp 2026-05-31 11:05, we'll rebuild to capture any post-merge math-core fixes IF the submodule is initialized in this worktree, ELSE we use the on-device binary as-is and document).

---

## Worktree state (Stage 0)

```
$ cd D:\F\shannon-prime-repos\engine-ntt-6
$ git status        # clean on sprint/ntt-6
$ git log -1
5826bd5 Merge sprint/hx-3b -- HVX vrmpy vectorization flips hex backend perf above ARM fp32 reference
```

Math-core submodule status: TBD (will check when building harness; if uninitialized, will use prior on-device binary per HX.3b precedent line 370-374).

On-device pre-staged artifacts (verified Stage 0):
- `/data/local/tmp/libsp_compute_skel.so` (185 KB, NTT.5b/c tip)
- `/data/local/tmp/sp22u/libsp_hex_skel.so` (36,416 bytes, HX.3b vrmpy SHA `4a79d04f...`)
- `/data/local/tmp/qwen3_rt.sp-model` + `.sp-tokenizer` (Executive HD=128)
- `/data/local/tmp/qwen25-coder-0.5b-memory.sp-model` + `.sp-tokenizer` (Memory HD=64)
- `/data/local/tmp/sp_ntt_bench_toks` (1,002,064 bytes, prior NTT-bench build)

Gemma3-1B `.sp-model` presence: to verify in Stage 1.
