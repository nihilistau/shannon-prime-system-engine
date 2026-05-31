## PLAN — HX.3b-alpha-v2 (precompute per-row weight-sum at host pack time)

**Sprint:** Phase 2-HX.3b-alpha-v2 (the incremental lift on a known-good path)
**Date:** 2026-05-31
**Worktree:** `D:\F\shannon-prime-repos\engine-hx-3b-v2`
**Branch:** `sprint/hx-3b-alpha-v2` (base: engine main @ 5826bd5 — post HX.3b merge)
**Sub-tag candidates:** `lat-phase-2-hx-3b-alpha-v2-precomputed-wsum` (T_HX3BV2_LIFT_ACHIEVED PASS) / `lat-phase-2-hx-3b-alpha-v2-attempted` (FAIL — blocker named)
**Status:** Plan-commit. Citations + storage decision (Option A) + numerical-equivalence proof + the 7 call sites locked in before code.

---

## Stage 0 — Mandatory pre-read (cited)

| # | File | Cited line(s) | What it tells us |
|---|------|---------------|------------------|
| 1 | `tools/sp_compute_skel/docs/CLOSURE-HX-3b.md` | `:322-329` ("Per-row weight-sum precomputation at host pack time...") | Explicit follow-on: "The `wsum_b` term (Σ_i weight[j][i]) is recomputed per vrmpy call via a second vrmpy with a splat-of-1 input. Precomputing it once at host weight-pack time and storing with the per-row scale would save ~30% of vrmpy ops per matmul (the second vrmpy goes away; horizontal reduce on wsum still needed once at the boundary). Estimated additional 1.2-1.5x perf lift; deferred to HX.3b-alpha-v2 / HX.3c." This is the sprint mandate verbatim. |
| 2 | `src/backends/hexagon/dsp/sp_hex_imp.c` | `:204-251` `hx_matmul_q8_vrmpy`; `:220-234` dual-vrmpy inner loop; `:235-236` dual hsum; `:243` scalar tail `ws_b += w`; `:246` `int32_t true_dot = dot_b - 128 * ws_b` | Current implementation. The wsum-eliminating instructions specifically: line 221 `HVX_Vector acc_ws = Q6_V_vzero();`, line 222 `HVX_Vector v_ones = Q6_V_vsplat_R(0x01010101);`, line 233 `acc_ws = Q6_Vw_vrmpyacc_VwVubVb(acc_ws, v_ones, w_v);`, line 236 `int32_t ws_b = hx_hsum_w(acc_ws);` (5-step ror+vadd reduce), line 243 `ws_b += w;` (scalar tail). All of these go away; `ws_b` is replaced by a lookup `row_sum[j]`. |
| 3 | `src/backends/hexagon/dsp/sp_hex_imp.c` | `:452-454, :475, :489-490, :497` (7 call sites in `sp_hex_forward`) | Seven call sites pass `WPTR(SP_HEX_W{Q,K,V,O,GATE,UP,DOWN})` to `hx_matmul_q8_vrmpy`. With Option A (row_sum lives inside the weight blob, after scales), the call signature does NOT change — the kernel walks past codes + scales to find row_sum. Zero change to the 7 call sites. |
| 4 | `src/backends/hexagon/sp_hex_host.c` | `:54-63` `hx_pack_q8` | Pack-time function. Currently writes int8 codes (line 59 `memcpy(dst, pt->codes, ...)`) then per-row f32 scales (line 61 `memcpy(scales, ...)`). I add a third block AFTER scales: per-row int32 row_sum, computed by summing signed-int8 codes per row. |
| 5 | `src/backends/hexagon/sp_hex_layout.h` | `:54-58` `sp_hex_q8_bytes`; `:55-58` reads "codes (padded) + per-row scales", aligned 128B | Per-Q8 block byte layout. Extend `sp_hex_q8_bytes` to also reserve `out * sizeof(int32_t)` aligned to 128B after the scales. Layout becomes: `[int8 codes (sp_hex_align(out*in))][f32 row_scale[out] (sp_hex_align(out*4))][int32 row_sum[out] (sp_hex_align(out*4))]`. Single struct, single rpcmem blob, one source of truth. |
| 6 | `tools/sp_compute_skel/docs/PLAN-HX-3b.md` | `:34-67` "Upstream-surfaced architectural decision" (per `feedback-no-silent-gate-revisions`) | Discipline pattern. This v2 follows the same plan-commit-first cadence: surface arithmetic equivalence + storage decision UPSTREAM before code, then ship one variable per stage. |
| 7 | Memory entries: `feedback-bundled-changeset-root-cause-ambiguity`, `feedback-no-silent-gate-revisions`, `reference-hexagon-v69-32x32-widening-idiom` | (from MEMORY.md index) | (a) One variable per stage; bundle only when iteration is expensive. (b) NEVER silently revise gates. If T_HX3BV2_LIFT_ACHIEVED fails (<1.2x), surface UPSTREAM with the diagnostic ("inner loop bandwidth-bound", "bias correction adds back saved cycles", "FastRPC tax dominates at ctx=16"). (c) widening idiom is for u32×u32→u64 (Barrett), not relevant here; we're already on vrmpy. |

### Additional pre-read (operational)

- ADB device check on Knack's S22U: `R5CT22445JA device` — connected.
- Hexagon SDK 5.5.6.0 accessible at `C:\Qualcomm\Hexagon_SDK\5.5.6.0`.
- Build flow per HX.3b CLOSURE §"Per-stage build commands" — `build_cmake hexagon DSP_ARCH=v69 BUILD=Release` from the worktree's dsp dir.
- HX.3b baseline skel on-device: `/data/local/tmp/sp22u/libsp_hex_skel.so` SHA-256 `4a79d04fd1965750f2bdebe8ab5fb29b7a53ce3399d8bdb1826c352d8558a8ca` (36,416 bytes).
- HX.3b baseline measurements (3-rep mean, today, this device):
  - hex-vrmpy prefill 1.523 tok/s, decode 1.069 tok/s
  - ARM ref prefill 1.465 tok/s, decode 1.069 tok/s
  - 32-token decode sequence: alternating `\n`, `</b>`, `\n`, `**`, `\n`, `**`... (per CLOSURE-HX-3b.md:80-87)

---

## Architectural decisions (surfaced UPSTREAM)

### Storage decision: **Option A** (row_sum embedded in weight blob)

The packed Q8 weight block layout becomes:

```
+------------------------------------------+ <- weight block start (128B aligned)
| int8 codes, row-major [out, in]          | sp_hex_align(out*in) bytes
+------------------------------------------+ <- (128B aligned)
| f32 row_scale[out]                       | sp_hex_align(out*sizeof(float)) bytes
+------------------------------------------+ <- (128B aligned)
| int32 row_sum[out]   <-- NEW             | sp_hex_align(out*sizeof(int32_t)) bytes
+------------------------------------------+ <- (128B aligned) -- next block follows
```

**Justification for Option A over Option B (sidecar):**
- Single rpcmem blob = single FastRPC pointer pass = no marshalling overhead delta vs HX.3b.
- Block-by-block walk via `sp_hex_weight_off` already handles variable-sized blocks; the layout change is local to `sp_hex_q8_bytes`.
- Host pack code stays in `hx_pack_q8` (one function); no new allocation/lifecycle bookkeeping.
- DSP-side kernel computes `row_sum_ptr = blk + sp_hex_align(out*in) + sp_hex_align(out*sizeof(float))` — one extra address arithmetic, zero new function-call signature changes.
- Zero call-site changes in `sp_hex_forward` (the kernel pulls its own row_sum out of `blk`).

**Numerical equivalence proof (the correctness argument)**

Define:
- `act_int8[i] = round(x[i] * 127 / S_act)` clamped to [-127, +127], where `S_act = max(|x|)/127` (line 177-189 of `hx_quant_act_ub`).
- `act_ub[i] = act_int8[i] + 128` (bias-128 trick, uint8 in [1, 255]; tail bytes = 128 = signed 0).
- `w_int8[i] = signed int8 weight code` in [-127, +127] (Q8 arena codes; padded tail = 0).
- Current per-call wsum: `ws_b = sum_{i=0..in-1} w_int8[i]` (integer sum, no overflow since |sum| <= 127 * 8192 < 2^20).
- Precomputed row sum: `row_sum[j] = sum_{i=0..in-1} (int32) w_int8[i, j]` — computed at host pack time over the SAME int8 codes, with the SAME index range (the padded tail i in [actual_in, sp_hex_align(actual_in)) is zero in the Q8 arena codes per pack convention; both paths sum the same zeros there).

Claim: `row_sum[j] == ws_b` bit-for-bit, for every j and for every prefill call.

Proof:
- The DSP kernel's `ws_b` is `sum_{i in [0, nb)} vrmpy_lanes + sum_{i in [nb, in)} scalar w`, where `nb = in & ~127`. Both sub-sums are int32 accumulators of the same int8 codes the host wrote at pack time. `Q6_Vw_vrmpyacc_VwVubVb(acc_ws, v_ones, w_v)` with `v_ones = 0x01010101` per lane computes per-lane `sum_{k=0..3} 1 * w[lane*4 + k]`. The hsum then sums over 32 lanes giving exactly `sum_{i=0..127} w[i]` per block, and the outer loop accumulates across blocks. The scalar tail (line 239-244) adds the remaining bytes. The total is `sum_{i=0..in-1} w_int8[i]`.
- The pack-time `row_sum[j]` is computed in C: `int32_t s = 0; for (int i = 0; i < in; i++) s += (int32_t)(int8_t)codes[j*in + i]; row_sum[j] = s;`. Same int8 values, same index range, same accumulator type — bit-identical sum.
- Therefore, `int32_t true_dot = dot_b - 128 * row_sum[j]` is bit-identical to `int32_t true_dot = dot_b - 128 * ws_b`.
- Therefore, `float y = (float)true_dot * (S_act * scales[j] / 127.0f)` is bit-identical between v1 (HX.3b) and v2 (HX.3b-alpha-v2) paths.
- Therefore, decode greedy-argmax sequence is byte-equal between HX.3b and HX.3b-alpha-v2.

**Inner-loop simplification**

Before (HX.3b, per 128B block):
- 2 × vmem load (act_v + w_v)
- 2 × Q6_Vw_vrmpyacc_VwVubVb (dot + ws)
- ~4 vector ops per block

After (HX.3b-alpha-v2, per 128B block):
- 2 × vmem load (act_v + w_v) — unchanged
- 1 × Q6_Vw_vrmpyacc_VwVubVb (dot only)
- ~3 vector ops per block

Per-row epilogue:
- Before: 2 × hx_hsum_w (5-step ror+vadd each) + `int32_t true_dot = dot_b - 128 * ws_b`
- After: 1 × hx_hsum_w + lookup `row_sum[j]` + `int32_t true_dot = dot_b - 128 * row_sum[j]`

Predicted instruction-count delta: ≥30% reduction in vrmpy + ≥50% reduction in hsum work. Whether that translates to wall-clock depends on whether vrmpy was ALU-bound or bandwidth-bound (the two vrmpys on the same `w_v` may have shared the same load — see "honest interpretation" below).

---

## Scope (what ships)

1. **Stage 1** — host-side: add `row_sum` to packed-block layout (sp_hex_layout.h) + populate it in `hx_pack_q8` (sp_hex_host.c). Host-only build verification (no DSP change yet; DSP keeps computing wsum on-the-fly). Bit-identical output guaranteed.

2. **Stage 2** — DSP kernel: implement single-vrmpy inner loop + post-loop bias-correction lookup. Add ONE call site (WQ) updated to use new kernel; other 6 stay on the wsum-accumulator path under compile gate. Decode-determinism check via partial-swap on-device.

3. **Stage 3** — swap remaining 6 call sites; remove old wsum accumulator and old `v_ones` splat from the kernel entirely (one code path).

4. **Stage 4** — skel rebuild + push to S22U. Run T_HX3BV2_DECODE_DETERMINISM gate on-device.

5. **Stage 5** — measure tok/s (3-rep mean, same NTT-bench harness). Write closure.

6. **Bit-exact decode preserved** vs HX.3b baseline (the 32-token sequence `\n`, `</b>`, `\n`, `**`, ... per CLOSURE-HX-3b.md:80-87). If divergent, surface UPSTREAM (no silent gate revision per `feedback-no-silent-gate-revisions`).

---

## Gates

| Gate | Method | Pass criterion |
|------|--------|----------------|
| **T_HX3BV2_WSUM_PRECOMPUTED** | Read sp_hex_host.c::hx_pack_q8 + sp_hex_layout.h after Stage 1; grep for `row_sum` allocation + population. | All 7 weight tensors have row_sum populated and accessible at fixed offset within their Q8 block. |
| **T_HX3BV2_INNER_LOOP_SIMPLIFIED** | `hexagon-llvm-objdump -d libsp_hex_skel.so` before/after for `hx_matmul_q8_vrmpy`. Count vrmpy + vmem + ror + vadd inside the labeled inner-loop body. | ≥30% reduction in vector instructions in the inner-block body (dual→single vrmpy + drop wsum reduce). Report exact count. |
| **T_HX3BV2_DECODE_DETERMINISM** | Drive identical prompt `[2,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500]` + 32 decode steps through both daemon configs (HX.3b baseline vs HX.3b-alpha-v2). Extract `delta` strings from SSE log. Compare with `Compare-Object`. | Byte-equal 32-token sequence vs HX.3b baseline. **No silent gate revision** — divergence triggers UPSTREAM surface. |
| **T_HX3BV2_LIFT_ACHIEVED** | Same NTT-bench harness + `timed_chat.sh` methodology as WIRE-HEX-FINISH and HX.3b. 3-rep mean prefill tok/s. | ≥1.20× HX.3b baseline (1.523 → ≥1.828 prefill tok/s). |

**If T_HX3BV2_LIFT_ACHIEVED FAILS**, surface UPSTREAM. Possible dispositions:
- **Inner loop bandwidth-bound** — second vrmpy reused the same `w_v` register; dropping it saves ALU cycles but loads still dominate. Diagnostic: instrument `HAP_perf_get_pcycles` per inner block. v3 needs prefetch or VTCM staging.
- **Bias correction adds back saved cycles** — the `128 * row_sum[j]` post-loop multiply may be dispatched into the same scalar pipe that handled the old hsum, eating the win. v3 fuses with the existing post-reduction scale multiply.
- **FastRPC marshalling tax dominates more than expected** — ctx=16 too small for the kernel-time win to surface in wall-clock. NTT.6 long-context measures separately. v3 may be a no-op at ctx=16 but compound at ctx≥128.

In each case, the failure is itself a useful diagnostic; do NOT silently mark PASS.

---

## Workflow discipline

1. **Plan-commit first** (this file). Commit message: `[plan] HX.3b-alpha-v2 -- per-row weight-sum precompute (Option A)`.

2. **Stage commits, one variable at a time:**
   - Stage 1: layout.h + host packer (host build verify). Commit.
   - Stage 2: kernel single-vrmpy inner loop + post-loop lookup + WQ call site only. Commit.
   - Stage 3: swap remaining 6 call sites; drop old v_ones + acc_ws code. Commit.
   - Stage 4: skel rebuild + push + decode-determinism gate. Commit (skel binary not committed; document hash).
   - Stage 5: tok/s + closure. Commit.

3. **NO SILENT GATE REVISIONS** per `feedback-no-silent-gate-revisions`.

4. **Anti-contamination strict.** `engine-hx-3b-v2` only. DO NOT modify K.beta.2.5c sp_compute_crt_imp.c. DO NOT modify NTT.5a/b/c surfaces. DO NOT modify math-core submodule (left uninitialized per HX.3b precedent — sp_hex_imp.c is self-contained).

5. **Hardware: Knack's S22 Ultra (R5CT22445JA) + Hexagon SDK 5.5.6.0 on Windows host.** Confirmed `adb devices` returns the device; SDK accessible.

---

## What the closure must answer honestly

- Did v2 deliver the 1.2-1.5x lift? If yes, by how much (3-rep mean)? If not, why (bandwidth-bound diagnostic)?
- Inner-loop instruction-count delta (before/after) — exact counts from objdump.
- Was decode bit-exact? If divergent, what's the root cause? (Per the numerical-equivalence proof, divergence is structurally impossible unless the pack-time sum is off — itself a useful test).
- Wall-clock breakdown if win is partial.
- Sub-tag: `lat-phase-2-hx-3b-alpha-v2-precomputed-wsum` (only if T_HX3BV2_LIFT_ACHIEVED PASS); else `lat-phase-2-hx-3b-alpha-v2-attempted` with blocker named.

---

## Final note

This is the **incremental lift sprint** on a known-good path. Lower architectural risk than HX.3b (no domain mismatch to navigate; same vrmpy primitive; same bias-128 trick; same blob layout convention). The variable is whether the inner loop is ALU-bound enough that dropping wsum translates to wall-clock — that's the empirical question the closure answers.

If it lifts to 1.83+ tok/s: ~25% faster than ARM fp32 reference at ctx=16, and the curve gets more favorable at longer ctx.

If it doesn't lift: the closure tells the user the inner loop is bandwidth-bound, which is itself the useful diagnostic for HX.3b-alpha-v3 design (prefetch / VTCM staging / fused post-loop scale).
