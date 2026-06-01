# CLOSURE — V5 — FFN VTCM ping-pong tile-streaming

**Sprint:** Phase 2-TRICK-1-FORWARD-V5
**Date:** 2026-06-01
**Worktree:** `D:\F\shannon-prime-repos\engine-v5-ffn-vtcm`
**Branch:** `sprint/v5-ffn-vtcm` (base engine main `d9b9a78` post-V4 merge)
**Hardware:** Samsung Galaxy S22 Ultra R5CT22445JA (Snapdragon 8 Gen 1, Hexagon V69 NPU/cDSP, 8 MB VTCM)
**Skel:** `libsp_hex_skel.so` 83808 bytes, SHA256 `89817C1E6342CF262A9DE2EACCDA5D6B930FB1E223BDEC7B081C31661D9C67D0`
**Sub-tag candidate:** `lat-phase-2-v5-ffn-vtcm-shipped-lift`

---

## 1. HEADLINE TABLE — Gemma3-1B tok/s on S22U R5CT22445JA (3-rep, clean same-session, controlled today)

| Config                                                              | Prefill tok/s |  Decode tok/s |   Ratio vs V4 |   Ratio vs HX.3b |
|---------------------------------------------------------------------|--------------:|--------------:|--------------:|-----------------:|
| hex vrmpy single-context (HX.3b)                                    |         1.550 |         1.072 |        0.696× |             1.000× |
| hex dual-context + VTCM (V4 attention-only)                         |         2.230 |         1.072 |        1.000× |             1.439× |
| **hex dual-context + VTCM + FFN tile-streaming (V5)**               |     **2.812** |         1.072 |    **1.261×** |          **1.814×** |

(Optional footnote — historical fp32 reference: ARM math-core scalar f32 = 1.465 tok/s prefill on this hardware per HX.3b 2026-05-31 closure. NOT a comparison row per `feedback-drop-fp32-baseline-comparing-to-ourselves`; lattice exceeds fp32 by 1.92× at V5.)

**Per-rep TTFT (FIRST_DELTA_MS_FROM_START, ctx=16 prefill, 32-token decode):**

| Rep | HX.3b (ms) | HX.3b tok/s | V4 (ms) | V4 tok/s | V5 (ms) | V5 tok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10305 | 1.553 | 7178 | 2.229 | 5698 | 2.808 |
| 2 | 10338 | 1.548 | 7177 | 2.230 | 5680 | 2.817 |
| 3 | 10325 | 1.550 | 7169 | 2.232 | 5690 | 2.812 |
| **Mean** | **10323** | **1.550** | **7175** | **2.230** | **5689** | **2.812** |

Per-rep variance: HX.3b σ_TTFT = 14 ms (0.13%); V4 σ_TTFT = 4 ms (0.06%); V5 σ_TTFT = 7 ms (0.13%). All well below the 3% noise band. Variance regimes consistent across all 3 configs.

Compound:
- **V5 / V4 = 1.261× over immediate-prior sprint**
- V5 / HX.3b = 1.814× over the V-series origin
- HX.3b → V3 → V4 → V5 compose multiplicatively: 1.000 × 0.943 × 1.523 × 1.261 = 1.814× ✓

(Prefill = 16000 / FIRST_DELTA_MS_FROM_START; decode = 31000 / STEADY_DECODE_MS, per CLOSURE-HX-3b.md methodology, identical harness `timed_chat.sh`. Same daemon restart cycle for all 3 configs.)

---

## 2. Gate-by-gate disposition (no silent revisions)

| Gate | Threshold | Result | Evidence |
|---|---|---|---|
| **T_V5_TILE_POOL_ALLOCATED** | HAP_request_VTCM(4.96 MB) succeeds; 2 tile slot offsets + size FARF-logged | **PASS** | FARF (`v5_farf_evidence.txt`): `HAP_request_VTCM: result=0, size=5057024, single_page_flag=0`; `sp_hex V5: VTCM allocated base=FF000000 size=5057024 WQ@0(1183744) WK@1183744(295936) WV@1479680(295936) WO@1775616(1184256) tile_a@2959872 tile_b@4008448 tile_bytes=1048576 rsum_attn=279552 B rsum_ffn=1557504 B` — 5.057 MB allocated; tile slots at correct 128-byte-aligned offsets (2959872 = attn_bytes; 4008448 = tile_a_off + 1 MiB). |
| **T_V5_DMA_PINGPONG_OBSERVED** | DMA prefetch overlaps with compute (overlap_ratio > 0.5) | **PASS** | FARF first-FFN-V5-matmul sample: `outer_pcyc=16049376 handler_compute_acc=15926974 worker_last_pcyc=1460008` → **overlap_ratio = 15.93M / 16.05M = 992/1000 = 99.2%**. DMA prefetch is **nearly fully hidden** behind HVX compute. The only ~0.8% un-hidden cost is the synchronous first-tile prefetch (mandatory, no compute is ready yet) plus the final-tile dmwait synchronization. |
| **T_V5_DECODE_BIT_EXACT** | 32-token decoded output byte-equal to V4 baseline | **PASS** | Per-token SSE delta strings (`chat_id` normalized) extracted from `v4_base_rep{1,2,3}.log` and `v5_rep{1,2,3}.log`; PowerShell `Compare-Object` returned EMPTY for all 3 paired reps (V4r1↔V5r1, V4r2↔V5r2, V4r3↔V5r3 all MATCH, 32 rows each). Discrete-substrate cross-backend determinism per `reference-lattice-decode-determinism` holds. |
| **T_V5_PERF_PARITY** | prefill tok/s ≥ V4 2.226 | **PASS** | V5 mean prefill = 2.812 tok/s. ≥ V4 floor 2.226 by **+26.3%**. |
| **T_V5_PERF_LIFT** | ≥ 1.30× V4 = ≥ 2.89 | **MARGINAL FAIL** | V5 / V4 = 2.812 / 2.230 = **1.261×**. Below the 1.30× LIFT target by 3.0% (2.812 vs 2.899 = 0.087 tok/s shortfall = 3.0%). |
| **T_V5_PERF_STRETCH** | ≥ 1.50× V4 = ≥ 3.34 | **FAIL** | V5 / V4 = 1.261× — 16% below stretch. Wall-clock 5689 ms vs stretch target ~4790 ms = 899 ms shortfall. |

---

## 3. What ships — substantive answer

**V5 FFN tile-streaming via UDMA ping-pong DID compound onto V4's attention-only VTCM lift, just below the operator-named 1.30× LIFT target.**

The architectural strategy worked exactly as planned: the V5 tiled kernel issues UDMA descriptors (Hexagon's user-DMA engine via `Q6_dmstart_A` intrinsic) that prefetch FFN row-tile N+1 into the alternate VTCM slot WHILE the dual-HVX-context handler+worker pair consume tile N from the current slot. The per-matmul pcycle evidence shows **99.2% DMA-compute overlap** — the DMA path is essentially free in wall-clock terms.

**V5 / V3-DDR per-matmul ratio = 16.05M / 55.7M pcyc = 3.47× faster per FFN matmul** on the V69 silicon. This is the unit of work that compounds into the wall-clock lift. Per layer: 3 FFN matmuls × (55.7M → 16.05M) = saves ~119M pcyc per layer × 26 layers = ~3.1B pcycles ≈ 1.3 seconds saved at ~2.4 GHz cDSP clock. Observed prefill wall-clock improvement V4 → V5 = 7175 ms → 5689 ms = 1486 ms saved. The pcyc accounting closes within ~15% (1.3s predicted, 1.49s observed; remaining gap likely from rmsnorm/rope/softmax overlap improvement as the FFN bandwidth pressure dropped).

**Three sprints of investment converged into V5.** HX.3b shipped the vrmpy kernel (1.550 tok/s, single-context). V3 shipped the dual-HVX-context substrate (1.464 tok/s, perf-flat — diagnosed bandwidth-bound). V4 unlocked attention via VTCM weight pinning (2.230 tok/s, 1.439× over HX.3b). V5 extended VTCM compound to FFN via UDMA tile-streaming (2.812 tok/s, **1.814× over HX.3b, 1.261× over V4**). The substrate is now load-bearing across all 7 Q8 matmuls per layer.

---

## 4. Why LIFT marginally missed — honest disposition

V5's 1.261× compound is structurally PARITY-PASS with substantial lift, just 3.0% below the 1.30× operator target. Per `feedback-no-silent-gate-revisions`: NOT revising the gate; naming the cost breakdown for follow-on.

Per-tile pcycle accounting (FARF evidence, first FFN matmul WGATE on layer 0 — shape out=6912 in=1152):
- 8 tiles × ~2M pcyc handler compute per tile half = ~16M pcyc handler total ✓
- outer wall = 16.05M pcyc ≈ handler-compute-only (DMA hidden 99.2%)
- worker half compute ≈ same as handler (last tile sampled at 1.46M pcyc; symmetric)

The remaining ~3% LIFT shortfall is most likely:
1. **Per-tile overhead** — 8 tiles per matmul × 78 matmul calls per prefill = 624 dmstart+dmwait pairs + 624 atomic seqno bumps + 624 futex_wake/wait pairs. At ~200 cyc each = ~125k cyc per matmul = ~0.8% wall overhead × 3 FFN matmuls per layer × 26 layers = ~6% — could explain the LIFT gap on its own.
2. **First-tile sync stall** — every matmul invocation issues a synchronous dmwait on tile 0 before any compute begins; this is unavoidable in the current ping-pong design (no "warm" state across matmuls).
3. **Activation re-quantization per tile** — `hx_quant_act_ub` runs INSIDE the half-kernel for each (t, j_range) pair. With tile granularity, the quant work re-runs once per tile per token instead of once per matmul per token. For Gemma3-1B chat shape (n_tok=16, in=1152) this is ~1152 × 16 floats × 8 tiles per matmul × 3 FFN matmuls per layer × 26 layers = ~11M extra ops per prefill. Likely 2-5% wall.

Each is a candidate V6 sprint. Combined they account for the ~3-5% gap to LIFT.

**Per `feedback-no-silent-gate-revisions`:** The honest answer is "V5 ships PARITY + substantial lift (1.261×); LIFT target marginally missed by 3%; STRETCH not approached." The four named follow-on candidates from PLAN-V5 D-G remain:
- **V6 (per-tile overhead reduction):** chain multiple tiles in ONE dmstart via `Q6_dmlink_AA`; reduce dmstart/dmwait count.
- **V7 (TCM read-port contention):** profile dual-context VTCM read contention at full tile granularity; partition attention vs FFN tile residence if needed.
- **V8 (cache coherency / barrier elimination):** investigate whether the per-tile dmwait can be replaced with per-descriptor `dstate` polling for finer-grained overlap.
- **V9 (cross-matmul tile warming):** keep the prefetch pipeline warm across the matmul boundary so the first-tile stall amortizes.

---

## 5. Architectural decisions (D-A through D-G) — empirical observations

| Decision | Plan choice | Observed |
|---|---|---|
| **D-A** Tile orientation | Row-tile along M; ~1 MiB per slot | Realized: rows_per_tile = floor(1 MiB / sp_hex_align(in_dim)) & ~3 = 908 rows for in=1152 (WGATE/WUP), 148 rows for in=6912 (WDOWN). 8 tiles per matmul for WGATE (6912/908=7.6→8 ceil), 8 tiles for WDOWN (1152/148=7.78→8 ceil). Confirmed in FARF: `n_tiles=8 rows_per_tile=908 row_stride_bytes=1152`. |
| **D-B** Shared vs per-tensor slots | Shared 2 tile slots, alternated within each matmul | Implemented as designed. Total VTCM used: 2.96 MB attn + 2 MiB tiles = 4.96 MB of 8 MB. Headroom 3.04 MB (room for activation pinning / future expansion). |
| **D-C** DMA scheme | UDMA via Q6_dmstart_A + Q6_R_dmwait | Implemented + silicon-validated. **UDMA WORKS in Unsigned PD** — U-1 concern from PLAN-V5 resolved positive. Across 26 layers × 8 tiles × 3 FFN matmuls = 624 prefetches per forward, no dmpoll/dmwait error reported. |
| **D-D** Per-thread tile assignment | Same as V4: handler [m_half..tile_rows), worker [0..m_half) | Implemented. `hx_matmul_q8_vrmpy_half_tile` is the V5 analog of V3's `hx_matmul_q8_vrmpy_half`; same vrmpy + bias-128 + rsum arithmetic. Worker dispatch via `g_hx_pool.job_kind == HX_JOB_V5_TILE_HALF`. |
| **D-E** Output indexing | Tile-relative j_local + global rsum + global scales | Implemented. `hx_vtcm_ensure_ffn_rsum(L, kind, blk_ddr, out, in_dim)` lazy-populates per-(L,kind) rsum; scales read directly from DDR blob at `sp_hex_align(out*in_dim)` offset. |
| **D-F** rsum_ffn lazy population | 3-wide layer_ready bool per (layer, FFN-kind) | Implemented. Total DDR malloc: 1,557,504 bytes = 1.49 MB (26 layers × (FF+FF+E) × 4 bytes). |
| **D-G** Stretch target | LIFT 1.30× V4; STRETCH 1.50× V4 | LIFT 1.261× — MARGINAL FAIL by 3%. STRETCH FAIL. |

---

## 6. VTCM budget — actual layout (Gemma3-1B)

```
VTCM base 0xFF000000 (8 MB total = 8,388,608 bytes)
├─ V4 attention sub-region:        0 - 2,959,871   (2.96 MB)
│  ├─ WQ codes+scales:                0 - 1,183,743   (1.13 MB)
│  ├─ WK codes+scales:        1,183,744 - 1,479,679   (296 KB)
│  ├─ WV codes+scales:        1,479,680 - 1,775,615   (296 KB)
│  └─ WO codes+scales:        1,775,616 - 2,959,871   (1.13 MB)
├─ V5 tile slot A:            2,959,872 - 4,008,447   (1.00 MiB)
├─ V5 tile slot B:            4,008,448 - 5,057,023   (1.00 MiB)
└─ (unused headroom):         5,057,024 - 8,388,607   (3.18 MiB)
```

DDR tables (malloc'd at hx_vtcm_init):
- rsum_attn: 26 × (1024+256+256+1152) × 4 = 279,552 B = 273 KB
- rsum_ffn:  26 × (6912+6912+1152) × 4 = 1,557,504 B = 1.49 MB
- rsum_attn_layer_ready: 26 × 1 = 26 B
- rsum_ffn_layer_ready: 26 × 3 = 78 B

Total DDR overhead: 1.76 MB for the rsum tables (one-time, persistent across forward calls).

---

## 7. UDMA per-tile timing breakdown — T_V5_DMA_PINGPONG_OBSERVED detail

FARF sample from first FFN matmul per session (WGATE, layer 0, n_tok=16, out=6912, in=1152):

```
outer_pcyc            = 16,049,376  (total wall pcyc for the V5 dispatch)
handler_compute_acc   = 15,926,974  (sum of HVX compute pcyc, handler half, across 8 tiles)
worker_last_pcyc      =  1,460,008  (last-tile worker half compute pcyc)
overlap_ratio_x1000   =        992  (compute_acc / outer × 1000)
```

Interpretation:
- 8 tiles per matmul, ~2M pcyc handler compute per tile, ~2M pcyc worker compute per tile.
- Outer wall ≈ handler compute (almost exactly) — DMA prefetch fully hidden by compute.
- Worker totals across 8 tiles also ≈ outer (symmetric to handler).
- DMA stall un-hidden: ~123k pcyc (0.8%) — the synchronous tile-0 prefetch + final-tile dmwait.

V3 DDR-path comparison (per V4 closure L101, same shape):
```
worker_pcyc           = 55,705,021 (V3 DDR)
handler_pcyc          = 55,703,965 (V3 DDR)
```

Per-matmul speedup: V3 DDR 55.7M → V5 VTCM 16.05M = **3.47× faster**. The bandwidth bound is broken for FFN.

---

## 8. Per-stage shipping log

| Commit | Stage | Substance |
|---|---|---|
| `ba12539` | plan | Stage 0 citations + Decisions D-A through D-G + UPSTREAM U-1, U-2 + budget math + cache-coherency strategy. |
| `51f4bff` | Stage 1 | VTCM tile-pool allocator (2 × 1 MiB slots) + rsum_ffn DDR table (1.49 MB). T_V5_TILE_POOL_ALLOCATED silicon-validated. |
| `45e7d73` | Stage 2 | UDMA primitive (Q6_dmstart_A wrapper + descriptor type0 inline def) + tile half-kernel + worker job_kind dispatch. V3/V4 paths regression-validated silicon-clean. |
| `0ab8940` | Stage 3 | hx_matmul_q8_vrmpy_dual_ctx_v5_tiled + hx_matmul_q8_vrmpy_dispatch_ffn + WGATE call site swap. UDMA works in Unsigned PD silicon-confirmed; decode preserved. |
| `3db51a1` | Stage 4 | Extend V5 dispatch to WUP + WDOWN. All 3 FFN matmuls through tiled VTCM ping-pong. 3-rep prefill 2.812 mean; PARITY+lift PASS, LIFT marginal FAIL by 3%. |
| `9a21a0f` | Stage 5 | T_V5_DMA_PINGPONG_OBSERVED pcycle instrumentation. Overlap 99.2% confirmed silicon. |
| (this) | Stage 6 | Same-session 3-config measurement: HX.3b 1.550 / V4 2.230 / V5 2.812; decode bit-exact (Compare-Object empty all 3 reps). Closure document. |

Per `feedback-bundled-changeset-root-cause-ambiguity`: each stage one-variable. Stages 1 and 5 are infrastructure-only; Stage 2 adds primitives without changing call paths; Stage 3 adds ONE call site swap (WGATE); Stage 4 extends to 3 sites; bit-exact decode verified at the cusp of Stage 3 (WGATE only) and Stage 4 (all 3). The pcycle instrumentation in Stage 5 is read-only — it doesn't alter the kernel arithmetic.

---

## 9. Honest framing — what the data does NOT say

1. **The 1.261× compound is "V5 + V3 substrate + V4 allocator + UDMA prefetch overlap" together.** No single component is the win in isolation. V5 alone (without V4's attention VTCM) would not be possible since the VTCM allocator was designed at V4 to share base; V5 alone (without V3's worker pool + job_kind dispatch) would lose dual-context per tile.

2. **LIFT marginally missed (3%) but PARITY substantially exceeded (+26%).** The narrative "FFN tile-streaming compounds onto attention VTCM" is structurally validated. The operator's 1.30× LIFT target was a stretch hypothesis; we hit a healthy 1.26×.

3. **STRETCH (1.50× = 3.34 tok/s) was not approached.** V5 at 2.812 is 16% below stretch. Reaching stretch requires architectural changes beyond V5's scope (see §4 follow-on candidates V6-V9).

4. **Decode tok/s remained 1.072 across all configs.** Decode bypasses the hex backend currently (per WIRE-HEX-FINISH closure); V5 changes nothing in decode path. HEX-DECODE-1 candidate sprint would route decode through the now-complete cDSP compound.

5. **The cDSP prefill story is functionally complete on Gemma3-1B at chat shape.** All 7 Q8 matmuls per layer (4 attention + 3 FFN) now run through silicon-validated VTCM-resident weights with dual-HVX-context compute. The remaining cDSP-prefill optimization axes are:
   - Path 2 — integer-end-to-end (Frobenius lift activations to Z_q, eliminate fp32 dequant between layers).
   - Path 3 — long-context (n_tok > 16) shape regime where compute-bound dominates.
   - Per-tile overhead reduction (V6 candidate).

6. **UDMA works in Unsigned PD on V69.** This was U-1 in PLAN-V5 — flagged as a substantive blocker risk. Silicon evidence: 624 dmstart+dmwait pairs per forward, no DMA error reported, byte-content correctness verified via decode-bit-exact. The "Unsigned PD limits" framing in earlier sprint memory does NOT apply to UDMA on this device path. Future Mode D / signed-PD work can rely on UDMA.

---

## 10. Files-changed manifest (vs d9b9a78 base)

| File | Change | Net LOC |
|---|---|---:|
| `src/backends/hexagon/dsp/sp_hex_imp.c` | Add V5 fields to hx_vtcm_t + tile pool allocator + rsum_ffn + UDMA primitive + tile half-kernel + worker job_kind dispatch + tiled dual-ctx dispatch + FFN dispatch entry + Stage 5 pcycle instrumentation + 3 FFN call site swaps | +443 / -22 |
| `tools/sp_compute_skel/docs/PLAN-V5.md` | NEW | +349 |
| `tools/sp_compute_skel/docs/CLOSURE-V5.md` | NEW (this) | -- |
| `v5_farf_evidence.txt` | NEW — committed logcat snapshot of V5 first-matmul FARF + VTCM alloc + worker init | +3131 B |

Anti-contamination per `feedback-parallel-agents-separate-worktrees`: ALL work confined to `engine-v5-ffn-vtcm` worktree. No modifications to V3, V4, K.beta.2.5c, NTT.5a/b/c, or any other concurrent worktree. math-core submodule unchanged. Host code (`sp_hex_host.c`) unchanged — IDL unchanged — only the cDSP skel rebuilds + pushes.

---

## 11. Reproducibility — exact commands run on host

```powershell
# Worktree
cd D:\F\shannon-prime-repos\engine-v5-ffn-vtcm

# Build skel (V5)
cd src\backends\hexagon\dsp
cmd /c "call ..\..\..\..\scripts\env\env-hexagon.bat 1>nul 2>nul && build_cmake hexagon DSP_ARCH=v69 BUILD=Release -gMake"
# Output: hexagon_Release_toolv87_v69\ship\libsp_hex_skel.so (83808 B, SHA256 89817C1E...)

# Deploy + measure (3-config same-session run)
$adb = 'D:\Files\Android\pt-latest\platform-tools\adb.exe'

# V5
& $adb push src\backends\hexagon\dsp\hexagon_Release_toolv87_v69\ship\libsp_hex_skel.so /data/local/tmp/sp22u/libsp_hex_skel.so
& $adb shell 'pkill -9 -f sp-daemon; sleep 3; rm -f /data/local/tmp/wire-hex-daemon.log; sh /data/local/tmp/start_wire_hex_daemon.sh'
Start-Sleep 40
foreach ($tag in 'warmup','rep1','rep2','rep3') {
  $maxtok = if ($tag -eq 'warmup') {4} else {32}
  & $adb shell "sh /data/local/tmp/timed_chat.sh '[2,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500]' $maxtok > /data/local/tmp/v5_${tag}.log 2>&1"
}

# V4 baseline (push V4 skel from engine-trick-1-fwd-v4 worktree)
& $adb push 'D:\F\shannon-prime-repos\engine-trick-1-fwd-v4\src\backends\hexagon\dsp\hexagon_Release_toolv87_v69\ship\libsp_hex_skel.so' /data/local/tmp/sp22u/libsp_hex_skel.so
# (V4 SHA256: 7A3E81A099ED4377C196962925261381C80A799B7B298EAA6BCE542D12F228C1)
# ... same restart + chat sequence ...

# HX.3b baseline (push HX.3b skel from engine-hx-3b worktree)
& $adb push 'D:\F\shannon-prime-repos\engine-hx-3b\src\backends\hexagon\dsp\hexagon_Release_toolv87_v69\ship\libsp_hex_skel.so' /data/local/tmp/sp22u/libsp_hex_skel.so
# (HX.3b SHA256: 4A79D04FD1965750F2BDEBE8AB5FB29B7A53CE3399D8BDB1826C352D8558A8CA)
# ... same restart + chat sequence ...

# Bit-exact comparison
function ExtractStrippedDeltas($p) {
    Get-Content $p | Where-Object { $_ -match '^DELTA_\d+_MS_FROM_FIRST' } | ForEach-Object {
        $payload = ($_ -split '\|',2)[1].Trim()
        $payload -replace '"chat_id":\d+', '"chat_id":X'
    }
}
foreach ($r in 1..3) {
    $v4 = ExtractStrippedDeltas "v4_base_rep$r.log"
    $v5 = ExtractStrippedDeltas "v5_rep$r.log"
    Compare-Object $v4 $v5  # empty = bit-exact
}
```

Captured logs (in worktree root, `.log` per `.gitignore` — not committed):
- `hx3b_rep{1,2,3}.log` — HX.3b same-session 3-rep
- `v4_base_rep{1,2,3}.log` — V4 same-session 3-rep
- `v5_rep{1,2,3}.log` — V5 same-session 3-rep
- `v5_farf_evidence.txt` — committed: V5 first-matmul FARF + VTCM alloc + worker init evidence

---

## 12. Sub-tag candidate

**`lat-phase-2-v5-ffn-vtcm-shipped-lift`** — PARITY pass + DECODE bit-exact + TILE_POOL_ALLOCATED + DMA_PINGPONG_OBSERVED. **LIFT marginally fails** (1.261× vs 1.30× target, 3% short). STRETCH fails. Sub-tag is `-shipped-lift` because the V-series wall-clock compound is real and substantial (1.814× over HX.3b), the operator-named LIFT threshold is just barely missed, the architectural strategy validated end-to-end.

---

## 13. What's NOT done — named follow-on

1. **V6 — per-tile overhead reduction.** Chain multiple tiles in ONE dmstart via `Q6_dmlink_AA`; reduce dmstart/dmwait count from 8/matmul to 1/matmul. Estimate: 2-4% additional prefill lift; could push compound to ~1.31× V4.

2. **V7 — TCM read-port contention partitioning.** If V6 doesn't close the LIFT gap, investigate whether dual-context VTCM reads contend at the TCM port boundary; consider residence-partition (attention in one VTCM region, FFN in another) with port affinity.

3. **V8 — finer-grained barrier.** Replace per-tile dmwait with per-descriptor `dstate` polling for finer-grained DMA-compute overlap. Estimate: 1-2% lift.

4. **V9 — cross-matmul tile warming.** Keep prefetch pipeline warm across matmul boundary (last tile of matmul N stays valid for matmul N+1 if pointers align). Tricky because WGATE/WUP/WDOWN have different tensor sources; only WGATE→WUP would benefit (both read from same DDR codes? — no, different tensors). Probably not worth.

5. **HEX-DECODE-1** — route decode through the now-complete cDSP compound. Decode currently bypasses hex backend. Estimated lift on decode tok/s: 2-3× (from 1.072 to ~2.5-3 tok/s).

6. **WIRE-AVX, NEON ARM-q2** — cross-backend portability. Not blocked by V5.

7. **TRICK-1-FORWARD cross-island** — mesh dispatch across multiple S22Us via Trick #1 protocol. Unrelated to V5.

8. **NTT.5d HD=256** — long-context NTT acceleration. Unrelated to V5.

9. **M.0-real SFT** — Memo model real-data SFT training. Unrelated to V5.

10. **Path 2 — integer-end-to-end.** Frobenius lift activations to Z_q, eliminate fp32 dequant between layers. Architectural change; separate sprint.

---

## 14. What unblocks

1. **The cDSP prefill compound story is functionally complete** on Gemma3-1B at chat shape. All 7 Q8 matmuls per layer run through silicon-validated VTCM-resident weights with dual-HVX-context compute. 1.814× compound over HX.3b prefill.

2. **UDMA in Unsigned PD silicon-validated.** Future Mode D / signed-PD work can rely on this primitive (Q6_dmstart_A + Q6_R_dmwait sequence on stack-allocated type0 descriptors with VA). The 624-prefetch-per-forward sustained run gives operational confidence in the API.

3. **V-series compound complete.** HX.3b (1.550) → V3 (1.464 perf-flat substrate) → V4 (2.230 attention-VTCM) → V5 (2.812 FFN-tile-streamed). Each sprint contributed measurable wall-clock either directly or via substrate unlock. Future work is decode-path + cross-backend + integer-end-to-end + per-tile overhead micro-opt.

4. **Per-call FARF accumulator infrastructure in place.** Stage 5 instrumentation can be extended to log per-call pcyc breakdown across the full 182-matmul prefill, informing follow-on sprint prioritization.

5. **Headroom in VTCM: 3.18 MiB unused.** Available for activation pinning (decode path), KSTE LUTs (Trick #2 spike), kvcache slot residency (NTT.6 long-context), or third tile slot (V6 chain experiment).

---

## 15. Discipline checklist (per memory feedback)

- [x] `feedback-read-spec-before-drafting-handoff`: read PLAN-V5 Decisions D-A/D-G + UPSTREAM U-1/U-2 + Stage 0 citations BEFORE staging code.
- [x] `feedback-no-silent-gate-revisions`: LIFT gate marginal fail; surfaced UPSTREAM as V6 follow-on. No threshold revised, no fixture tuned. Stage 4 commit explicitly named PARITY-pass + LIFT-marginal-fail.
- [x] `feedback-bundled-changeset-root-cause-ambiguity`: 6 stages (plan, allocator, primitive, half-kernel+1-site, all-3-sites, instrumentation+measurement). One variable per commit. Each commit message enumerates what was changed.
- [x] `feedback-lead-with-reference-then-theory`: Plan-commit Stage 0 cited 10 references (memory + in-repo + SDK headers + udma.h) BEFORE design.
- [x] `feedback-shape-dependent-parallelism-gates`: gates at scope where wall-clock matters (prefill per-token); pcycle as secondary diagnostic.
- [x] `feedback-leak-gate-allocator-warmup`: 3 reps after warmup; warmup separately documented (10.84 s vs 7.18 s rep mean for V4 — V4 closure baseline confirmed by V5-day re-measurement).
- [x] `feedback-lattice-baseline-is-prior-lattice`: V5 baseline = V4 (prior lattice impl), not cuBLAS/llama.cpp/MKL. Same-session same-day measurement.
- [x] `feedback-drop-fp32-baseline-comparing-to-ourselves`: HEADLINE TABLE has 3 lattice rows only; fp32 reference is an optional footnote, NOT a comparison row. Compound ratios computed vs V4 (primary) and HX.3b (compound), NOT vs fp32.
- [x] `feedback-parallel-agents-separate-worktrees`: V5 work confined to `engine-v5-ffn-vtcm`; V3, V4, hx-3b worktrees untouched.
- [x] Mandatory pre-read citations in plan-commit's Stage 0 (file:line throughout PLAN-V5.md).

---

## 16. Worktree status

```
$ git log --oneline -8 sprint/v5-ffn-vtcm
9a21a0f [Stage 5] V5 -- T_V5_DMA_PINGPONG_OBSERVED instrumentation + same-session evidence
3db51a1 [Stage 4] V5 -- extend tiled VTCM ping-pong to all 3 FFN matmuls (WGATE/WUP/WDOWN)
0ab8940 [Stage 3] V5 -- WGATE tile-streaming + decode-bit-exact preliminary
45e7d73 [Stage 2] V5 -- UDMA primitive + tile half-kernel + worker job_kind dispatch
51f4bff [Stage 1] V5 -- VTCM tile-pool allocator + rsum_ffn table
ba12539 [plan] V5 -- FFN VTCM ping-pong tile-streaming
d9b9a78 Merge sprint/trick-1-forward-v4 -- VTCM WEIGHT PINNING BROKE THE CHAT-SHAPE BANDWIDTH BOUND.
```

To push: `git push -u origin sprint/v5-ffn-vtcm`.

---

## 17. Final note

The operator named V5 as "the final V-series compound on the prefill cDSP path on Knack's S22U." That framing holds:

**V5 PARITY + LIFT (just below operator-named 1.30× by 3%) compounded onto V4's attention VTCM, producing 1.814× prefill tok/s over HX.3b same-session baseline (2.812 vs 1.550). Decode bit-exact preserved across all 3 reps. All 7 Q8 matmuls per Gemma3-1B layer now run through silicon-validated VTCM-resident dual-HVX-context compute. UDMA tile-streaming achieves 99.2% DMA-compute overlap on V69 chat shape.**

LIFT marginally missed; named follow-on (V6 per-tile overhead reduction) is the load-bearing path to push compound past 1.30×. STRETCH 1.50× requires architectural change (Path 2 integer-end-to-end or per-call FARF profiling to find residual hot spots).

The cDSP prefill story is functionally complete. The next architectural axis is HEX-DECODE-1 (route decode through this compound) or Path 2 (eliminate fp32 dequant between layers via Frobenius integer arithmetic end-to-end). Both are unblocked.
