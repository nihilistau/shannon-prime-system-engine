# PLAN — TRICK-1-FORWARD-V4 — VTCM weight pinning to break the V69 chat-shape bandwidth bound

**Sprint:** Phase 2-TRICK-1-FORWARD-V4
**Date:** 2026-06-01
**Worktree:** `D:\F\shannon-prime-repos\engine-trick-1-fwd-v4`
**Branch:** `sprint/trick-1-forward-v4` (base engine main `db4de65` = post V3 merge)
**Predecessor:** V3 substrate (worker pool + dual-HVX-context per-matmul) silicon-validated; perf FLAT at chat shape per `reference-v69-vrmpy-chat-shape-memory-bound`. V4 attacks bandwidth via VTCM weight residency.

This plan-commit captures Stage 0 reference citations + architectural decisions A–G + UPSTREAM concerns surfaced BEFORE code per `feedback-lead-with-reference-then-theory` and `feedback-no-silent-gate-revisions`.

---

## Stage 0 — Reference citations (file:line, verbatim where load-bearing)

### S0.1  `reference-v69-vrmpy-chat-shape-memory-bound` (memory entry, ~2 days; verify-against-code)

> THIRD confirmation (2026-06-01, TRICK-1-FORWARD-V3): V3 wired dual-HVX-context per-matmul via SSR:XA={4,5} into all 7 `sp_hex_forward` call sites. Both vector contexts burn HVX pcycles concurrent (worker_pcyc=3.68M + handler_pcyc=2.66M concurrent on WQ matmul). Decode 32-token byte-equal vs HX.3b. Wall-clock: V3 1.464 vs HX.3b 1.505 prefill tok/s — 2.7% SLOWER. K v0.alpha's 1.935× ceiling was at compute-bound 128×128 / B=8; Gemma3-1B chat shape (M=1, K∈{1152, 6912}, N∈{1024, 1152, 6912}) is fundamentally different regime. Both V3 contexts contend for the same DDR/L1 weight reads — adding parallel compute parallelizes the bandwidth wait, not the compute. **THE FIX: VTCM weight pinning via `qurt_mem_l2cache_lock` (8 MB VTCM available on V69, currently unused).**

**Verification against code (per memory's own staleness note):** the V3 closure at `tools/sp_compute_skel/docs/CLOSURE-TRICK-1-FORWARD-V3.md:13-32` confirms today's 3-rep measurement of V3 1.464 vs HX.3b 1.505 prefill tok/s. The V3 wall-clock breakdown at lines 363-386 attributes +2 ms/matmul overhead to "synchronization + bandwidth contention." **Today's diagnosis stands: V4 attacks bandwidth via VTCM.**

### S0.2  `reference-v69-hvx-expert-practices` (memory entry, ~2 days; verify-against-code)

Per memory section "VTCM — Vector Tightly Coupled Memory" (lines 192-225):

> VTCM is on-chip memory directly adjacent to HVX. On V69: ~8 MB typical. Latency: approximately 1 cycle for loads vs ~30+ for DDR. Bandwidth: 256 GB/s class. **Allocation:** `HAP_compute_res_acquire_cached` with VTCM resource type, or via the Halide runtime's `halide_hexagon_set_vtcm_size` for AOT-compiled pipelines.

**CORRECTION via direct SDK check** (`C:\Qualcomm\Hexagon_SDK\5.5.6.0\incs\HAP_vtcm_mgr.h:67-150`):
The canonical V69-direct API is **`HAP_request_VTCM(unsigned int size, unsigned int single_page_flag)` / `HAP_release_VTCM(void* pVA)`** — returns a void* directly addressable from cDSP code. The memory entry's `qurt_mem_l2cache_lock` framing is a different layer (L2 pin; separate primitive). V4 uses `HAP_request_VTCM` matching the existing in-repo precedent at `tools/sp_compute_skel/src_dsp/sp_compute_ntt_twiddle.c:194` (the NTT.2 twiddle staging).

> Lattice strategy (V69, 8 MB budget): Cannot fit full K/V cache (~234 MB for Qwen3-0.6B). CAN fit per-layer streaming K/V tiles. CAN fit Frobenius per-row scales for the active layer (3072 rows × 4 bytes = 12 KB per layer; full set 336 KB fits trivially).

**For V4 weights (not K/V cache):** see Decision D-A budget table below — Gemma3-1B Q8 weights total ~670 MB; 8 MB VTCM cannot hold a single FFN tensor (each FFN ~7.6 MB), let alone a full layer (26 MB).

VTCM staging recipe (memory section "Empirical findings from Mode D Sprints", lines 308-348):

> Generator-side `set_host_alignment(128)` … Schedule `.prefetch(input, x, r, 2)` … All-buffers-in-VTCM (input, output, AND scratch), not mixing DDR + VTCM in one kernel call. … Working pattern: all I/O buffers in external VTCM (allocated via HAP_request_VTCM); hidden intermediates via `.store_in(MemoryType::VTCM)` for Halide-internal allocation. Don't mix DDR and VTCM pointers in one kernel call.

**For V4:** the "don't mix DDR+VTCM in one kernel call" finding is a Halide-codegen finding, not a hand-written intrinsics finding (V4 uses raw `Q6_Vw_vrmpyacc_VwVubVb` intrinsics, not Halide schedules). Hand-written code freely loads from either address space. Verified by the existing `sp_compute_ntt_hvx_vtcm_imp.c` mixing DDR activations with VTCM twiddles. **D-C uses mixed addressing.**

### S0.3  `reference-vtcm-per-stage-misalignment` (memory entry, ~2 days)

> HVX vector load constraint: `Q6_V_vmem` (aligned vmem) requires 128-byte-aligned source addresses on V69. Issuing aligned vmem from a misaligned address either (a) Silently rounds DOWN to the previous 128-byte boundary (reads wrong data) — observed behavior in NTT.3 Stage 1, or (b) Raises a fault on stricter configurations.

**For V4:** `HAP_request_VTCM(size, 0)` returns page-aligned (≥4 KB aligned, per SDK docs); first byte of every VTCM allocation is 128-byte aligned. **Per-tensor VTCM tiles**: ensure tile start is 128-byte aligned by allocating each as a separate `HAP_request_VTCM` call, OR pad inner sub-region offsets to multiples of 128 within a shared allocation. Plan: **separate `HAP_request_VTCM` per pinned tensor for simplicity** — fewer race conditions, each tensor's `void*` is page-aligned ergo HVX-`vmem`-safe.

### S0.4  V3 substrate (CLOSURE-TRICK-1-FORWARD-V3 + sp_hex_imp.c)

`CLOSURE-TRICK-1-FORWARD-V3.md:38-60` — substrate ready-to-use:
- cDSP-internal worker thread spawned via `qurt_thread_create` at first matmul (sp_hex_imp.c:585).
- Worker calls `qurt_hvx_lock(QURT_HVX_MODE_128B)` successfully under Unsigned PD (sp_hex_imp.c:522-530).
- Per-matmul descriptor passed via shared struct; signal-wait via atomic seqno + futex (sp_hex_imp.c:447-457 desc, 553-674 dispatch).
- Worker = rows [0, M/2); handler = rows [M/2, M); both consume same activation buffer (sp_hex_imp.c:494, 547-548).
- Output bit-exact to HX.3b single-context kernel — `Compare-Object` empty diff (closure line 68).

**V4 builds on this scaffold UNCHANGED.** The kernel function gets a new VTCM-aware variant; the worker-pool infrastructure and dispatch path stay identical.

### S0.5  Existing in-repo VTCM precedent (`sp_compute_ntt_twiddle.c`)

`tools/sp_compute_skel/src_dsp/sp_compute_ntt_twiddle.c:170-200` — `sp_tw_init_one` is the canonical V69 VTCM allocation pattern in this repo:
1. `void *p = HAP_request_VTCM(arena_size, 0u);` (line 194)
2. Failure path FARFs + returns 0 (lines 195-200).
3. Cleanup via `HAP_release_VTCM(p)` (line 293) on rollback / shutdown.

V4 mirrors this idiom for weight pinning.

### S0.6  HX.3b baseline (CLOSURE-HX-3b)

`CLOSURE-HX-3b.md:13-39` — HX.3b mean prefill 1.523 tok/s (3-rep). **V4 floor = 1.523.** V3 today re-measured HX.3b at 1.505 prefill (same rep variance band, sub-1% drift); V4 must beat the V3-day HX.3b figure (1.505) at minimum, ideally restore-or-exceed the published 1.523.

### S0.7  HX.3b-α-v2 finding (CLOSURE-HX-3b-alpha-v2)

`CLOSURE-HX-3b-alpha-v2.md` — single-vrmpy inner loop yields only 6.5% lift over dual-vrmpy because the second vrmpy occupied an unused ALU slot. **Confirms ALU is not the bottleneck.** V4's premise (attack bandwidth) is structurally correct.

### S0.8  Hexagon SDK headers verified

`C:\Qualcomm\Hexagon_SDK\5.5.6.0\incs\HAP_vtcm_mgr.h` exists. APIs `HAP_request_VTCM`, `HAP_release_VTCM`, `HAP_query_total_VTCM`, `HAP_query_avail_VTCM` available. No `qurt_mem_l2cache_lock` needed — that's a different abstraction layer.

### S0.9  Discipline references (no-silent-gate-revisions, etc.)

- `feedback-no-silent-gate-revisions`: gate FAIL surfaces UPSTREAM, NOT silently revised. V4 honors.
- `feedback-bundled-changeset-root-cause-ambiguity`: one variable per stage commit. V4 staged 1-tensor → 4-tensor → 7-tensor.
- `feedback-leak-gate-allocator-warmup`: leak metric is second-half slope, not total delta. V4 holds.

---

## Architectural decisions A–G (UPSTREAM-surfaced BEFORE code)

### D-A — VTCM budget allocation: **HYBRID — attention pinned + FFN streamed (or DDR fallback)**

#### Gemma3-1B per-layer Q8 weight budget (from `sp_hex_layout.h:43-46` + Gemma3 1B config E=1152, FF=6912, HD=256, NH=4, NKV=1, QD=1024, KVD=256):

| Tensor | Shape [out, in] | Codes (int8) | + Scales (f32) | Total per-layer | Total all 26 layers |
|---|---|---:|---:|---:|---:|
| WQ    | [1024, 1152] | 1.13 MB | 4 KB | **1.13 MB** | 29.5 MB |
| WK    | [256, 1152]  | 295 KB  | 1 KB | **296 KB**  | 7.7 MB |
| WV    | [256, 1152]  | 295 KB  | 1 KB | **296 KB**  | 7.7 MB |
| WO    | [1152, 1024] | 1.13 MB | 5 KB | **1.13 MB** | 29.5 MB |
| WGATE | [6912, 1152] | 7.59 MB | 27 KB | **7.62 MB** | 198 MB |
| WUP   | [6912, 1152] | 7.59 MB | 27 KB | **7.62 MB** | 198 MB |
| WDOWN | [1152, 6912] | 7.59 MB | 5 KB | **7.59 MB** | 197 MB |
| **PER-LAYER TOTAL** |  | | | **~25.7 MB** | **~667 MB** |
| **ATTENTION ONLY (Q,K,V,O)** | | | | **~2.85 MB** | **74 MB** |

#### VTCM available: 8 MB on V69 (`HAP_query_total_VTCM`).

#### Critical analysis:

- **Cannot pin entire model** in VTCM (667 MB needed, 8 MB available).
- **Cannot pin all of one layer** (25.7 MB needed, 8 MB available).
- **CAN pin one layer's attention weight set** (~2.85 MB ≪ 8 MB).
- **CANNOT pin one full FFN tensor** (7.6 MB doesn't fit alongside attention 2.85 MB in 8 MB).
- **CAN pin one FFN tensor alone** (7.6 MB ≤ 8 MB) — but then attention spills to DDR.

#### Choice: **D-A.HYBRID — per-layer attention residency + FFN-as-DDR-fallback (Stages 1-3); optional FFN tile-streaming (Stage 4 stretch)**

Per-layer flow:
1. Layer entry: `HAP_request_VTCM(attn_set_bytes_for_this_layer, 0)` → copy WQ + WK + WV + WO from DDR into VTCM (~2.85 MB DMA, paid once per layer).
2. Inner WQ, WK, WV, WO matmuls: V4 kernel reads weights from VTCM-resident pointers.
3. Inner WGATE, WUP, WDOWN matmuls: V4 kernel falls back to DDR-resident pointers (V3 dual-context path).
4. Layer exit: `HAP_release_VTCM(p)` — release attention region so layer L+1 can reuse the 8 MB.

**Why this hybrid:**
- **Honest about the budget**: 8 MB cannot hold even one FFN tensor + anything else.
- **Lowest-risk path**: per-layer DMA copy is `memcpy` on the cDSP side; no asynchronous DMA programming, no double-buffer race conditions, no overlap-with-compute scheduling. Deterministic.
- **Attention matmuls fully benefit**: 4 of 7 matmuls per layer see VTCM-resident weights. Each weight byte read 16× per matmul (n_tok=16); ~94% of reads now hit VTCM at 256 GB/s instead of DDR at ~10 GB/s.
- **Per-layer copy amortized**: 2.85 MB / ~10 GB/s ≈ 290 μs per layer × 26 layers = 7.5 ms total prefill DMA overhead. Vs ~10.6 s prefill = **0.07% overhead**. Negligible.

**Honest projection:**
- Attention matmul wall-time today ≈ ~(2.85/25.7) × 10.6 s ≈ ~1.2 s (proportional to byte count).
- If attention matmul lifts 2× (full bandwidth utilization at VTCM speed), saves ~600 ms of prefill.
- New prefill ≈ 10.0 s → 1.6 tok/s (vs HX.3b 1.523, +5%). **Modest lift, similar to HX.3b-α-v2's 6.5%.**
- **FFN-dominant cost (~88% of byte traffic) is NOT addressed in Stages 1-3.**

**Stage 4 (stretch goal): FFN tile-streaming.** Divide WGATE/WUP/WDOWN into K-direction tiles. Double-buffer 2 × ~1 MB tiles in VTCM. While compute consumes tile_i, prefetch tile_{i+1} via `memcpy` from DDR in parallel. The reuse-within-VTCM-window (n_tok=16 reads per byte) gives the same 16× amplification factor as attention. **High risk** (programming complexity, alignment, contention with DMA controller); deferred to V5 if not landable in time budget.

#### Alternative considered and rejected:

- **Pin per-layer FFN-only (one of WGATE/WUP/WDOWN per call cycle)** — but then attention spills back to DDR. Net byte-traffic similar to today; the gain is from "one FFN matmul lifted" vs "all 4 attention lifted." Math: FFN single matmul saved = 7.6 MB × 16 reads. Attention all-4 saved = 2.85 MB × 16 reads. Attention-all-4 is smaller byte savings but ALL FOUR matmuls lift. Stage-1 attention is the higher-confidence first move.
- **Halide-style "all-buffers-in-VTCM" pattern** — requires moving activations + outputs into VTCM too. Activations are small per-call (~n_tok × E × 4 = ~73 KB); outputs similar. Possible but the gain is incremental on top of weight pinning; deferred.
- **Pin Frobenius scales only** — scales are already L1-resident due to small footprint and per-row reuse; pinning them in VTCM gives no measurable lift.

### D-B — Activation buffer placement: **DDR (UNCHANGED)**

V3's `hx_worker_local_t::act_ub` is a per-thread 8192-byte (SP_HEX_VRMPY_MAX_IN) static buffer in BSS. It's 128-byte-aligned. **Stays in DDR** for V4 Stages 1-3. Per-call activation quant cost is ~1% of total work per HX.3b closure §"Activation quant cost"; not the bottleneck.

Stage 4 (stretch) may move activations to VTCM if cycle-budget allows.

### D-C — Cache coherency: **VTCM is cDSP-owned; no ARM cache coherency required**

VTCM is on-chip cDSP-private memory. ARM cannot directly address it; it lives in the cDSP's address space only. Therefore:
- No `DMA_BUF_IOCTL_SYNC` needed (ARM isn't reading or writing VTCM).
- No cache-flush-before-DSP / invalidate-before-ARM (memory entry `reference-v69-hvx-expert-practices` line 228-258 applies to shared DDR; VTCM bypasses).
- The DDR-resident weights ARE shared with ARM (host packed them via rpcmem); the per-layer DMA copy from DDR→VTCM is a cDSP-internal `memcpy` reading from a host-flushed DDR region; standard `qurt_mem_cache_clean(QURT_MEM_CACHE_INVALIDATE)` may apply to invalidate cDSP L1/L2 lines of the source DDR region before the copy if rpcmem hasn't already.

**Per-tensor VTCM identity tracking:** extend the existing per-matmul descriptor mechanism with a `void *blk_vtcm` field — if non-NULL, kernel uses this pointer; else falls back to the DDR `blk` pointer. The dispatch wraps the choice in `hx_matmul_q8_vrmpy_dual_ctx_vtcm`.

### D-D — Per-thread VTCM tile assignment for dual-context: **SHARED VTCM, both threads read concurrently**

Per V3 the worker thread does rows [0, M/2) and handler thread does rows [M/2, M). Both threads read the SAME activation buffer + SAME weight blob. For VTCM:

- **Option A** (split): allocate two VTCM half-tiles, copy half-of-weight to each. Doubles allocator cost. Doesn't increase total parallel bandwidth (VTCM is one physical bank).
- **Option B** (shared): one VTCM allocation containing the full per-layer attention set. Both threads read from it. **The TCM read port count question.**

The V69 HVX architecture spec is not in the SDK's public headers, but pragmatic test: NTT.5b shipped dual-thread reads from a shared VTCM twiddle arena and observed full parallelism within HVX compute (per `CLOSURE-NTT-5b.md`). Inference: V69 VTCM has dual read ports OR has enough port count that two concurrent HVX vector contexts saturate independently.

**Choice: Option B (shared VTCM).** Simpler, matches existing precedent, no measurable downside expected. If empirical T_V4_BANDWIDTH_DROP_OBSERVED fails for FFN tile-streaming due to TCM port contention, V5 splits into per-thread tiles.

### D-E — Tile sizing for FFN matrices (Stage 4 only): **ROW-MAJOR K-DIRECTION TILES**

WGATE shape [6912, 1152]: split the K dimension (1152) into tiles. With K=1152 = 9 × 128, a natural tile size = K_tile = 256 (2 × 128B HVX vectors per row, 1024 bytes per tile-row). VTCM region per buffer = 6912 rows × 256 bytes = ~1.7 MB; ping-pong = ~3.5 MB. Fits with 2.85 MB attention = ~6.3 MB total ≤ 8 MB.

Actually simpler: tile by ROWS (the M dimension, 6912 → e.g. 8 tiles of 864 rows). Tile size = 864 × 1152 ≈ 1 MB. Ping-pong = 2 MB. Plus attention 2.85 MB = ~5 MB total ≤ 8 MB.

**Choice: row-tile (M) of ~1 MB each, ping-pong (2 × 1 MB = 2 MB) for the active FFN matmul. Detail deferred to Stage 4 plan.**

### D-F — Build infrastructure

- VTCM allocation API: `HAP_request_VTCM(size, single_page_flag)` / `HAP_release_VTCM(ptr)` from `HAP_vtcm_mgr.h`.
- Include `#include "HAP_vtcm_mgr.h"` in sp_hex_imp.c.
- Existing SDK at `C:\Qualcomm\Hexagon_SDK\5.5.6.0\incs` already in CMakeLists.txt include paths.
- Skel build flow: same as HX.3b/V3: `build_cmake hexagon DSP_ARCH=v69 BUILD=Release -gMake` from `src/backends/hexagon/dsp/`.
- No host-side rebuild needed (IDL unchanged); skel-only redeploy.

### D-G — Perf-parity gate target

- **Floor (T_V4_PERF_PARITY)**: prefill tok/s ≥ 1.523 (HX.3b 3-rep mean from published closure) AND ≥ 1.505 (V3-day HX.3b same-day reference). Both must hold.
- **Lift (T_V4_PERF_LIFT)**: ≥ 1.20× HX.3b = ≥ 1.83 tok/s.
- **Stretch**: ≥ 1.50× HX.3b = ≥ 2.29 tok/s.

**Honest projection given D-A analysis:**
- Stage 1-3 (attention-only VTCM) projected lift: ~1.05× HX.3b ≈ 1.60 tok/s. **PARITY: PASS expected. LIFT: FAIL expected (below 1.83 threshold).**
- Stage 4 (FFN tile-streaming): projected lift ~1.5-1.8× HX.3b ≈ 2.3-2.7 tok/s. **LIFT: PASS expected if it ships.**

The gates remain at the prompt's stated thresholds. If T_V4_PERF_LIFT fails after Stage 1-3 with Stage 4 not shipped (V4-partial), surface UPSTREAM as V5 candidate with FFN streaming named as the load-bearing follow-on.

---

## UPSTREAM concerns (surfaced BEFORE writing code)

### Concern V4-A — VTCM single-allocation contention

The session-scope VTCM allocation contends with NTT.2's already-claimed twiddle arenas (`sp_compute_ntt_twiddle.c`). Both share the 8 MB V69 budget. Net impact at HX prefill: NTT.2's allocations are made by the `sp-daemon-wire-ntt` daemon, NOT `sp-daemon-wire-hex`. Different FastRPC sessions = different cDSP PD instances = different VTCM allocators (HAP_request_VTCM is per-PD).

**Verification:** the WIRE-HEX-FINISH daemon (`sp-daemon-wire-hex`) is what runs the V4 forward; NTT.2's daemon is `sp-daemon-wire-ntt` and isn't loaded during chat. **No contention expected.** Will FARF-log `HAP_query_total_VTCM` at sp_hex_open to verify.

### Concern V4-B — Per-layer DMA copy bandwidth interferes with handler-thread compute

The layer-entry `memcpy(vtcm_attn_buf, ddr_attn_weights, attn_set_bytes)` consumes ~290 μs of DDR-to-VTCM bandwidth. If issued while the handler thread is still running the FINAL FFN matmul of layer L (WDOWN), the DMA contends for the same DDR bus. 

**Mitigation:** issue VTCM swap at layer entry BEFORE matmul dispatch, not at layer exit. The DDR copy completes before WQ matmul starts. WDOWN of layer L-1 is fully done before this code executes (sequential C). **No contention.**

### Concern V4-C — VTCM allocation failure at session init

`HAP_request_VTCM` may fail under contention with other cDSP processes (e.g., camera HAL holding VTCM in another PD). The Unsigned PD constraint may also limit VTCM access.

**Mitigation:** fall back to DDR-resident path if `HAP_request_VTCM` returns NULL. The V3 dual-context path already handles this case (it falls back to `hx_matmul_q8_vrmpy_v2` on `rsum` cache failure; same defensive pattern applies). The kernel function `hx_matmul_q8_vrmpy_dual_ctx_vtcm` takes a `vtcm_ptr` parameter that, when NULL, runs the V3 DDR path unchanged. **No regression on VTCM-unavailable devices.**

### Concern V4-D — Closure-table truth: V4 may match V3 even with attention-only VTCM if FFN dominates entirely

If FFN matmul bandwidth is THE bottleneck (not attention matmul bandwidth), Stage 1-3 attention-only VTCM yields ≤ noise floor lift. The 4-attention-of-7-matmul share of byte traffic is ~11% (2.85 MB / 25.7 MB) per layer. Speeding up 11% of the work by 2× saves ~5.5% wall-clock. **This is within the 3% rep variance noise** observed in HX.3b/V3 measurements.

**Disposition:** Stage 1-3 is the load-bearing FEASIBILITY validation (VTCM allocation works, VTCM-resident kernel works, decode bit-exact, lift signal SEEN or UNSEEN). Stage 4 is the load-bearing PERF lift. If Stage 4 doesn't ship, V4-partial closes as "VTCM substrate validated; attention-VTCM lift below noise floor (as projected); FFN tile-streaming named as V5 critical path."

This is honest, not pessimistic. Per `feedback-no-silent-gate-revisions`: surface UPSTREAM what the data shows.

---

## Substantive gates (mirror prompt; no silent revision)

**T_V4_VTCM_ALLOCATED** — `sp_hex_open` (or first-call lazy-init equivalent) successfully calls `HAP_request_VTCM` and logs the returned pointer + size. Methodology: FARF log at session init + `HAP_query_total_VTCM` log. Pass: VTCM allocation succeeds, weights addressable from cDSP-side VTCM region for ≥ 1 layer of attention weights.

**T_V4_DUAL_CTX_VTCM_READS** — cDSP disassembly shows `vmem` reads in the dual-context kernel targeting addresses in the VTCM region. Methodology: `hexagon-llvm-objdump -d` + verify the kernel inner loop reads from a VTCM-pointer-typed argument. Pass: kernel takes a VTCM pointer argument and disasm shows it loaded via `vmem`.

**T_V4_BANDWIDTH_DROP_OBSERVED** — DDR memory-stall pcycles per matmul reduced vs HX.3b baseline. Methodology: `HAP_perf_get_pcycles` brackets per attention matmul in V4 vs HX.3b reference, compare medians. Pass: ≥ 30% pcycle reduction per attention matmul (Stage 1-3 scope) OR ≥ 30% reduction across FFN matmuls (Stage 4 scope, if shipped).

**T_V4_DECODE_BIT_EXACT** — 32-token decode matches HX.3b baseline byte-equal. Methodology: same as HX.3b/V3 — drive 16-token prompt + 32 decode steps, `Compare-Object` delta strings. Pass: byte-equal. **No silent gate revision.**

**T_V4_PERF_PARITY** — prefill tok/s ≥ 1.523 (HX.3b baseline). Methodology: 3-rep mean per HX.3b/V3 procedure. Pass: observed ≥ 1.523.

**T_V4_PERF_LIFT** — prefill tok/s ≥ 1.20× HX.3b baseline (≥ 1.83). Stretch ≥ 1.50× = 2.29 tok/s.

**Honest disposition policy** (per `feedback-no-silent-gate-revisions`):
- If Stage 1-3 ships and Stage 4 is deferred: T_V4_PERF_LIFT expected to FAIL. Surface UPSTREAM as V5 = FFN tile-streaming. T_V4_PERF_PARITY remains a hard requirement.
- If Stage 4 ships: T_V4_PERF_LIFT expected to PASS structurally.
- T_V4_DECODE_BIT_EXACT and T_V4_VTCM_ALLOCATED are non-negotiable.

---

## Scope (what ships, per-stage)

### Stage 1 — VTCM allocation infrastructure (no kernel changes)

- Add `#include "HAP_vtcm_mgr.h"` to `sp_hex_imp.c`.
- New global state `hx_vtcm_t`: cached pointer + size of the active layer's attention VTCM region; layer index of cached region (-1 = empty).
- New helper `hx_vtcm_ensure_layer(L, weights, cfg)`: if cached layer != L, free old + allocate new + memcpy attention weights from DDR into VTCM. Returns base VTCM pointer for layer L's attention set.
- New helper `hx_vtcm_release_all()`: called from `sp_hex_close`; releases any held VTCM allocation.
- Wire into `sp_hex_forward` to call `hx_vtcm_ensure_layer(L, weights, cfg)` at layer entry. **Stage 1 does NOT yet use the returned VTCM pointer in the kernel — just allocates + memcpys.**
- Gate target: T_V4_VTCM_ALLOCATED PASS. Decode bit-exact preserved (kernel still reads DDR).

### Stage 2 — VTCM-aware dual-context kernel for ONE call site (WQ)

- New kernel `hx_matmul_q8_vrmpy_dual_ctx_vtcm` accepts a `const unsigned char *vtcm_blk` parameter. If non-NULL, reads weights from VTCM; else falls back to V3's `hx_matmul_q8_vrmpy_dual_ctx` (DDR path).
- Internally: `hx_matmul_q8_vrmpy_half_vtcm` (mirror of V3's `hx_matmul_q8_vrmpy_half` but with VTCM pointer arg). The compaction of `codes` + `scales` pointers happens via the same `sp_hex_align(out*in)` offset since the VTCM copy preserves the DDR blob layout.
- Wire WQ call site to use the new kernel. Other 6 call sites untouched (V3 DDR path).
- Gate target: T_V4_DUAL_CTX_VTCM_READS via objdump on the new kernel; decode bit-exact vs HX.3b for the single-site swap.

### Stage 3 — Extend to all 4 attention matmuls (WQ/WK/WV/WO)

- Wire WK, WV, WO call sites to the VTCM kernel.
- The VTCM allocation in Stage 1 already covers all 4 attention tensors (one contiguous region per layer with internal sub-tensor offsets matching the DDR layout).
- 3-rep prefill tok/s measurement.
- Gate target: T_V4_PERF_PARITY (≥ 1.523 floor); T_V4_DECODE_BIT_EXACT byte-equal vs HX.3b.

### Stage 4 (stretch) — FFN tile-streaming via ping-pong VTCM tiles

- Conditional on Stage 1-3 + time budget. If skipped: ship V4-partial with explicit V5 named follow-on.
- Add per-FFN-matmul tile-streaming: divide M dimension into ~8 tiles of ~864 rows × 1152 codes ≈ 1 MB each.
- Ping-pong: while compute tile_i runs, memcpy DDR→VTCM for tile_{i+1}.
- Wire WGATE/WUP/WDOWN call sites.
- Gate target: T_V4_BANDWIDTH_DROP_OBSERVED (≥ 30% pcycle drop) + T_V4_PERF_LIFT (≥ 1.83 tok/s).

### Stage 5 — Skel rebuild + push to S22U + 3-rep measurement

- Build skel via `build_cmake hexagon DSP_ARCH=v69 BUILD=Release -gMake`.
- adb push to `/data/local/tmp/sp22u/libsp_hex_skel.so`.
- 3-rep prefill + decode measurement matching V3 closure §7 methodology.

### Stage 6 — Closure document

- `CLOSURE-TRICK-1-FORWARD-V4.md` with all 16 sections per prompt.
- Sub-tag candidate: `lat-phase-2-trick-1-forward-v4-shipped-lift` (LIFT+PARITY both PASS), `-shipped` (PARITY PASS, LIFT FAIL), or `-attempted` (PARITY FAIL).

---

## Files-to-be-changed manifest

| File | Status | Purpose |
|---|---|---|
| `src/backends/hexagon/dsp/sp_hex_imp.c` | EDIT | +~250-450 LOC (Stages 1-3); +~300-600 LOC if Stage 4 |
| `tools/sp_compute_skel/docs/PLAN-TRICK-1-FORWARD-V4.md` | NEW | this file |
| `tools/sp_compute_skel/docs/CLOSURE-TRICK-1-FORWARD-V4.md` | NEW | Stage 6 closure |

NO modifications to: V3 worktree files (read-only reference), K.beta.2.5c/NTT.5* (different daemon), sp_hex_host.c (no IDL/layout change), math-core submodule (empty per HX.3b precedent).

---

## Workflow

Per-stage commits with single-variable discipline:
1. `[plan] TRICK-1-FORWARD-V4 — VTCM weight pinning + Stage 0 citations + UPSTREAM A-D` (this commit)
2. `[Stage 1] TRICK-1-FORWARD-V4 — VTCM allocator + per-layer attention copy (no kernel change)`
3. `[Stage 2] TRICK-1-FORWARD-V4 — VTCM-aware kernel + WQ call site swap; T_VTCM_READS PASS`
4. `[Stage 3] TRICK-1-FORWARD-V4 — extend to WK/WV/WO; T_PERF_PARITY measurement`
5. `[Stage 4 (stretch)] TRICK-1-FORWARD-V4 — FFN tile-streaming` (conditional)
6. `[Stage 5-6] TRICK-1-FORWARD-V4 — closure: tok/s + gate disposition + V5 follow-on if applicable`

Push: `git push -u origin sprint/trick-1-forward-v4`. Operator merges.

---

## Final note

V4 is the 4th attempt at lifting V69 chat-shape prefill tok/s. V3 substrate works (dual HVX context, qurt_hvx_lock under Unsigned PD), but lift didn't materialize because chat shape is bandwidth-bound. V4 attacks the bandwidth.

**Honest framing of the projected outcome:**
- Stages 1-3 (attention-only VTCM): ~5% lift expected — same magnitude as HX.3b-α-v2's 6.5%. Crosses the PARITY threshold; below the LIFT threshold.
- Stage 4 (FFN streaming): structurally the load-bearing lift. If lands, expected 1.5-1.8× over HX.3b.

V4-partial (Stages 1-3 only) is a documented acceptable outcome with FFN streaming named as V5. V4-full (Stages 1-4) is the user's tok/s win.

Either way: honest data wins; bit-exact decode preserved; no silent gate revisions; closure tells the truth about what the bandwidth attack moved.
