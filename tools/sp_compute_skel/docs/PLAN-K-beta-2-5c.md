# PLAN — Sprint K v0.beta-2.5c (mod_q matmul + Garner recombination)
**Date:** 2026-05-30
**Branch:** `sprint/kbeta-2-5c`
**Worktree:** `D:\F\shannon-prime-repos\engine-kbeta-2-5c`
**Base:** engine main @ 0822747 (K.beta.2.5b closure merged)
**Status:** Plan-commit (Stage 0 reference reading complete)

## Stage 0 — Reference reads (file:line citations)

Per `feedback-lead-with-reference-then-theory`, this plan opens with verbatim
references that will steer the implementation.

1. **K.beta.2.5b HVX vector Barrett primitive** —
   `tools/sp_compute_skel/src_dsp/sp_compute_crt_imp.c:74-123`
   (`sp_barrett_reduce32_hvx_lane`), `:128-152` (`sp_barrett_vec_run`),
   `:164-214` (`sp_compute_barrett_oracle` IDL implementation). The matmul
   accumulator path will **reuse `sp_barrett_reduce32_hvx_lane` verbatim** —
   the silicon-confirmed primitive applies to any per-lane reduction of a
   `(va_splat) * (vb_vec)` product. This is the architecturally clean reuse
   path: we extend the building block, we do not duplicate or re-derive it.

2. **K.beta.2.5b closure diagnosis** —
   `tools/sp_compute_skel/docs/CLOSURE-K-beta-2-5b.md:65-111`
   (DUAL_DISPATCH_SPEEDUP UPSTREAM-REQUIRED §) — the exact reasoning this
   sprint exists to close: shape regime mismatch between primitive
   (1.5 ms / data-bound) and K v0.alpha matmul (17.7 ms / compute-bound).
   Lines `:113-150` (LEAK_FREE UPSTREAM-REQUIRED §) — the metric must be
   second-half slope per `feedback-leak-gate-allocator-warmup`.

3. **K.beta.2.5b SASS audit format** —
   `tools/sp_compute_skel/docs/HVX_BARRETT_SASS_GATES.md:13-41` (the
   per-intrinsic table) — every intrinsic the matmul kernel emits will be
   audited against the same per-row table format. Inner-loop divergence
   threshold: 0.

4. **K v0.alpha matmul shape regime** —
   `MEMORY.md / reference_fastrpc_concurrent_dispatch.md` (already loaded
   into context): "128×128 / B=8 / 17.7 ms / invoke / 1.935× dual-dispatch
   speedup" — our mod_q matmul targets the **same** B=8 / D_in=128 / D_out=128
   shape and the **same** per-invoke wall regime (15-20 ms) so the
   parallelism gate is meaningful per
   `feedback-shape-dependent-parallelism-gates`.

5. **`reference-hexagon-v69-32x32-widening-idiom`** — silicon-confirmed in
   K.beta.2.5b. The matmul inner loop uses the same
   `Q6_W_vmpye_VwVuh + Q6_W_vmpyoacc_WVwVh` pair via the
   `sp_barrett_reduce32_hvx_lane` building block, which already contains
   three of them (steps 1, 6, 11 in `HVX_BARRETT_SASS_GATES.md`).

6. **`feedback-shape-dependent-parallelism-gates`** — caught 2026-05-30 in
   K.beta.2.5b's data-bound regime. The fix: measure parallelism where
   wall-clock matters (matmul kernel, not Barrett primitive). This sprint
   IS the fix landing.

7. **`feedback-leak-gate-allocator-warmup`** — K.beta.2.5b observed +2100
   KB first half, +4 KB second half over 10k iter. This sprint's gate
   metric: **second-half VmRSS delta ≤ 256 KB**, NOT total delta.

8. **HVX intrinsics reference** —
   `${HEXAGON_SDK_ROOT}/tools/HEXAGON_Tools/8.7.06/Tools/lib/clang/15.0.0/include/hvx_hexagon_protos.h`:
   - `Q6_W_vmpye_VwVuh` (line 3721 / `#define` 3726) — pair-init widening,
     reused from 2.5b
   - `Q6_W_vmpyoacc_WVwVh` (line 3754 / `#define` 3759) — paired accumulator,
     reused from 2.5b
   - `Q6_V_vsplat_R` — X-element broadcast per inner-loop k
   - `Q6_Vw_vadd_VwVw` — modular-add accumulator step (already in 2.5b
     primitive at SASS row 13)
   - `Q6_Q_vcmp_gt_VuwVuw` — conditional-subtract test (already in 2.5b)
   - `Q6_V_vmux_QVV` — conditional-subtract mux (already in 2.5b)
   No new intrinsics required; matmul is built ENTIRELY from
   primitives the 2.5b SASS audit already certified.

## Sprint K v0.beta-2.5c — scope

A mod_q matmul kernel that:

1. Computes `Y[B][D_out] = X[B][D_in] · W[D_in][D_out] mod q` lane-parallel
   over HVX, where elements live in `Z_q` (per-prime invocation).
2. Reuses `sp_barrett_reduce32_hvx_lane` for per-element Barrett reduction.
3. Is dispatchable per-prime via `Arc<FastRpcSession>`; q_1 and q_2 invokes
   run concurrently on dual cDSP HVX vector contexts.
4. Outputs `r_1 mod q_1` and `r_2 mod q_2` to ARM-side `DmaBuffer`s.
5. ARM-side Garner recombination produces the 60-bit `Y` from `(r_1, r_2)`.

Target shape: **B=8, D_in=128, D_out=128**. Same as K v0.alpha matmul.
Per-invoke wall target ≈ 15-20 ms (compute density matches K v0.alpha
regime; see "Per-invoke wall estimate" below).

## Architectural choices

### Per-prime function vs parameterized function

**Chosen: one parameterized function**, dispatching q/μ from a `q_idx`
parameter inside the kernel — same pattern as `sp_barrett_vec_run` at
`sp_compute_crt_imp.c:128-152`. Rationale: keeps the SASS audit surface
single-source (no duplicate codepaths to verify per prime); reduces
maintenance risk; matches the 2.5b precedent.

### Accumulator strategy

Three candidates considered:

(A) **64-bit-per-lane accumulator using `HVX_VectorPair`** — accumulate
    raw widening products, single Barrett at end.
    - Per-k product `< q^2 < 2^60`
    - After D_in=128 accumulations: `< 128 * 2^60 = 2^67` → OVERFLOWS u64
    - Requires periodic in-pair reduction or carry handling. Adds
      complexity for no clear win.
    - REJECTED.

(B) **u32 accumulator with periodic Barrett-reduce-every-K-iters** —
    Barrett-reduce sum every K iterations to keep acc < 2^32.
    - `K * q < 2^32` → `K < 4` for q ≈ 2^30
    - K=3 safe. K=4 borderline. Means Barrett-every-3-iters in inner loop.
    - Adds bookkeeping and one extra reduce per 3 k-steps.
    - REJECTED in favor of (C) — simpler.

(C) **Per-k Barrett with always-in-[0,q) accumulator** (CHOSEN) —
    - Each iteration: widen `a*b` (60-bit), Barrett-reduce to [0, q), add
      to acc (now in [0, 2q-2)), conditional-subtract back to [0, q).
    - **No iteration-count constraint**; D_in can be arbitrary.
    - Inner-loop cost: 19 (Barrett) + 3 (vadd + vcmp + vmux) + 1 vsub
      for vmux input = ~23 intrinsics per iteration per 32-lane vector.
    - REUSES the exact silicon-confirmed `sp_barrett_reduce32_hvx_lane`
      from 2.5b — no new math; no new SASS surface that wasn't audited.

**Computed parameter K for path (B), documented for the record:**
`K_max = floor((2^32 - 1) / (q_1 - 1)) = floor(4294967295 / 1073738752) = 4`.
At K=4, max accumulator = 4 * (q-1) < 4 * 2^30 = 2^32, exactly at u32
overflow boundary — unsafe under worst-case. K=3 is the safe choice.
Path (C) elects K=1 implicitly and uses Barrett's existing two-conditional
subtract path for the modular add (one subtract suffices since
0 ≤ acc, r < q).

### Garner inverse constants (precomputed)

Modulus product: `M = q_1 * q_2 = 1152908312643096577 (60 bits exact)`

```
Q1_INV_MOD_Q2 = 894602413   (verified: (q_1 * 894602413) mod q_2 = 1)
Q2_INV_MOD_Q1 = 179131221   (verified: (q_2 * 179131221) mod q_1 = 1; reserved for future)
```

Garner formula (uses `Q1_INV_MOD_Q2` only):
```
diff = (r_2 - r_1) mod q_2       // canonical [0, q_2)
t    = (diff * Q1_INV_MOD_Q2) mod q_2
r    = r_1 + q_1 * t             // r in [0, M); fits u64
```

`r` is the unique value in `[0, M)` satisfying `r ≡ r_1 (mod q_1)` and
`r ≡ r_2 (mod q_2)`. Since `q_1 * q_2 < 2^60`, `r < 2^60`, fits u64.

### Per-invoke wall estimate

Shape: B=8, D_in=128, D_out=128. Inner loop runs `D_in × ceil(D_out/32) = 128 × 4 = 512` times per batch element. Total iterations: `B × 512 = 4096` inner-loop bodies.

Each iteration: ~23 intrinsics → ~23 packets at ~1 GHz cdsp clock = ~23 ns per iter (assuming dense VLIW packing, optimistic).

Compute floor: `4096 * 23 ns = 94 µs`. Realistic with pipeline + memory access ≈ 5-10× = 0.5-1 ms compute time.

Data movement: X (4 KB), W (64 KB), Y (4 KB) = 72 KB. FastRPC DDR→DSP marshalling at ~2 GB/s ≈ 36 µs. Negligible vs FastRPC fixed overhead (~3-5 ms per invoke per K v0.alpha observations).

**Predicted per-invoke wall: 5-15 ms.** Comparable to K v0.alpha's 17.7 ms saturating matmul. Compute-bound enough that DUAL_DISPATCH_SPEEDUP gate is meaningful.

If observed wall is < 3 ms (purely FastRPC-overhead-dominated): file finding as memory entry candidate and surface UPSTREAM-REQUIRED with shape-regime diagnosis.

If observed wall is > 30 ms: investigate likely cause (cache miss / VLIW slot starvation / unexpected SASS) and surface UPSTREAM.

## Stage execution plan

| Stage | Deliverable | Files | Commit |
|---|---|---|---|
| 0 | This plan | `PLAN-K-beta-2-5c.md` | `[plan] K v0.beta-2.5c -- mod_q matmul + Garner + DUAL_DISPATCH at compute-bound shape` |
| 1 | Scalar reference + Garner on ARM | `tools/sp_dsp_smoke/src/sp_matmul_q_ref.rs` | `[lat-3-hx-mode-k-beta-2-5c] feat: Stage 1 -- Rust scalar ref + Garner recombination + host unit test` |
| 2 | HVX mod_q matmul kernel + IDL + smoke harness | `src_dsp/sp_compute_crt_imp.c`, `inc/sp_compute.idl`, `tools/sp_dsp_smoke/src/sp_matmul_q_smoke.rs`, `Cargo.toml` | `[lat-3-hx-mode-k-beta-2-5c] feat: Stage 2 -- HVX mod_q matmul kernel + IDL method 11 + correctness smoke` |
| 3 | Dual-dispatch + leak harness | `tools/sp_dsp_smoke/src/sp_matmul_q_dual_smoke.rs`, `Cargo.toml` | `[lat-3-hx-mode-k-beta-2-5c] feat: Stage 3 -- dual-dispatch + Garner-bit-exact + LEAK_FREE harness` |
| 4 | Gate runs + SASS audit + closure | `HVX_MATMUL_Q_SASS_GATES.md`, `CLOSURE-K-beta-2-5c.md`, `tools/sp_compute_skel/docs/sp_compute_matmul_q.sass`, `tools/sp_dsp_smoke/sprint_k_beta_2_5c_*.txt` | `[lat-3-hx-mode-k-beta-2-5c] test: Stage 4 -- 4 substantive gates run + SASS audit + closure` |

**No bundling per `feedback-bundled-changeset-root-cause-ambiguity`.** Each
stage commits separately; if a gate fails, isolate by reverting only that
stage.

## File convention note

The prompt requested `tools/sp_dsp_smoke/src/bin/sp_matmul_q_smoke.rs`. The
existing repo convention is `tools/sp_dsp_smoke/src/<name>.rs` with explicit
`[[bin]]` entries in `Cargo.toml`. I follow the existing convention to
match `sp_barrett_oracle_smoke.rs`, `sp_barrett_dual_smoke.rs`, etc.

## Out of scope (filed as future sprints)

- Wiring into the actual production matmul path (separate integration
  sprint).
- Halide generator integration (Halide HVX Int(64) limit still applies).
- Multi-shape sweep (filed as K v0.beta-2.5d).

## Gates (substantive — all 4 load-bearing)

1. **T_MATMUL_Q_CORRECTNESS** — mod_q matmul output bit-exact vs Rust
   scalar reference. ≥4 seeds × both primes × B=8/D_in=128/D_out=128.
   Pass: `divergence_count = 0, max_lane_diff = 0`.
2. **T_GARNER_BIT_EXACT** — Garner-combined u64 output == 60-bit host
   reference matmul (when output < M). Pass: 0 divergences across
   the test population.
3. **T_MATMUL_DUAL_DISPATCH_SPEEDUP** — wall-clock speedup ≥ 1.5×
   (target ≥ 1.7×) via Arc<FastRpcSession> dual-thread concurrent invoke.
   Surface UPSTREAM if < 1.5× with diagnosis.
4. **T_MATMUL_LEAK_FREE** — second-half VmRSS delta ≤ 256 KB across
   10k cycles. (NOT total delta — per
   `feedback-leak-gate-allocator-warmup`.)

## Worktree discipline

- Branch: `sprint/kbeta-2-5c` on worktree `D:\F\shannon-prime-repos\engine-kbeta-2-5c`.
- No commits authored from `shannon-prime-system-engine` (main).
- No reads of `shannon-prime\`, `shannon-prime-engine\`,
  `shannon-prime-system-engine\` source trees.
- No reads of `engine-kbeta-2-5b\` (prior sprint's worktree).
- No reads or writes to `shannon-prime-lattice\` (M.0 agent's domain) or
  `lattice-memo-m0\` (M.0 agent's worktree).
- Push command at end: `git push -u origin sprint/kbeta-2-5c`.
- Merge to main: OPERATOR responsibility, NOT this agent.
