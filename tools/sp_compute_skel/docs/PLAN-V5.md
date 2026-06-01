# PLAN — V5 — FFN VTCM ping-pong tile-streaming

**Sprint:** Phase 2-TRICK-1-FORWARD-V5
**Date:** 2026-06-01
**Worktree:** `D:\F\shannon-prime-repos\engine-v5-ffn-vtcm`
**Branch:** `sprint/v5-ffn-vtcm` (base engine main `d9b9a78` — post V4 merge)
**Hardware:** Samsung Galaxy S22 Ultra R5CT22445JA (Snapdragon 8 Gen 1, Hexagon V69, 8 MB VTCM)
**Compound-onto:** V4 attention-only VTCM (2.226 prefill tok/s, 1.434× over HX.3b 1.552 same-session).

---

## Stage 0 — Mandatory pre-read (file:line citations)

### S0.1 — `reference-v69-hvx-expert-practices` (memory)

VERBATIM excerpt: "VTCM 8MB on V69 — pin Frobenius scales + KSTE LUTs via qurt_mem_l2cache_lock; cache coherency (flush-before-DSP, invalidate-before-ARM) via DMA_BUF_IOCTL_SYNC; Halide schedule tile 128×4 + unroll 4 + prefetch 2-iters + 128-byte alignment."

Key claims that apply to V5:
- **VTCM latency ~1 cycle, bandwidth ~256 GB/s**; DDR latency ~30+ cycles, bandwidth ~10 GB/s. The ratio is what lets V4's 2.96 MB attention pin equalize both HVX contexts at full throughput.
- **Empirical Sprint F.1 lesson — "all-buffers-in-VTCM" is the load-bearing rule.** Mixing DDR + VTCM pointers in one HVX kernel call is brittle. V5 inherits this: when a tile is the "active" buffer for matmul-compute, the kernel reads it as VTCM-resident.
- **Cache coherency for V5:** the ARM never touches VTCM (it's cDSP-private). DDR weight bytes are already rpcmem-flushed at registration time (V4 substrate). DDR→VTCM staging (whether via memcpy in V4 or via UDMA in V5) does NOT need DMA_BUF sync. The HVX-vmem-read of VTCM after a DDR→VTCM UDMA write needs cDSP-internal ordering — that is `Q6_R_dmsyncht()` / `qurt_user_dma_dmsyncht()` discussion below.

### S0.2 — `reference-v69-vrmpy-chat-shape-memory-bound` (memory) — read all 3 historical sections

The 3-way-confirmed bandwidth-bound diagnosis at chat-shape:
- HX.3b-α-v2 single-context kernel saturated DDR/L1 for K=1152 reads at n_tok=16 small batch.
- V3 dual-context exposed the constraint: both HVX contexts contended for the same DDR controller → pcycle asymmetry 71% (V3 worker 40% slower than handler).
- V4 confirmed by inversion: VTCM-resident weights → pcycle symmetry 99% → context throughput equalized → 1.434× wall-clock lift.

WHY ping-pong is the right shape for V5:
- FFN per-tensor 7.6 MB ≫ 5 MB VTCM headroom → can't pin a full tensor.
- Row-tile streaming = load tile N while compute consumes tile N-1 (DMA + HVX run on separate hardware, can overlap).
- The cDSP UDMA engine reads DDR at the same physical DDR controller as scalar memcpy WOULD — but UDMA writes to VTCM run **concurrently** with HVX vmem reads of an already-staged tile. That is the overlap V5 buys.

### S0.3 — `reference-vtcm-per-stage-misalignment` (memory)

VERBATIM: "NTT.2's per-stage compacted twiddle arrays land at 4-byte-aligned (NOT 128-byte-aligned) byte offsets: 0, 4, 12, 28, 60, 124, 252, 508, 1020. Aligned `vmem` from stage 2+ silently reads wrong data (NTT.3 caught 600/600 divergence)."

Application to V5 — tile alignment inside VTCM:
- V5 tiles will be **128-byte aligned by construction**: tile slot base = `g_hx_vtcm.vtcm_base + ffn_tile_off_{A,B}` where the offsets are aligned to 128 by ceiling-rounding the V4 attention allocation. Each FFN row is 128-byte-aligned because the host's blob layout is (per `sp_hex_layout.h:51-52` `sp_hex_align`). Row stride in tile = `sp_hex_align(in_dim)` = same as DDR layout. So `vmem` aligned reads are CORRECT for V5; we do NOT need `vmemu`.
- DECISION (D-A-2): tile base offsets aligned to 128 explicitly; per-tile data layout is `(rows_per_tile × sp_hex_align(in_dim)) bytes` of int8 codes + `rows_per_tile × sizeof(float)` scales. Scales sub-block also 128-aligned by allocating padding.

### S0.4 — `feedback-drop-fp32-baseline-comparing-to-ourselves` (memory)

The baseline-framing rule is binding for V5 closure. Headline table = 3 lattice rows (HX.3b / V4 / V5). NO fp32 reference row in the headline. Optional footnote acceptable.

Compound ratios: `V5 / V4` (immediate prior sprint) primary; `V5 / HX.3b` secondary; fp32 not reported as a ratio.

### S0.5 — V4 closure (`tools/sp_compute_skel/docs/CLOSURE-TRICK-1-FORWARD-V4.md`)

Key cited lines:
- L47 (T_V4_VTCM_ALLOCATED): `HAP_request_VTCM: result=0, size=2959872, single_page_flag=0`; `WQ@0(1183744) WK@1183744(295936) WV@1479680(295936) WO@1775616(1184256) rsum_attn=279552 B`. **VTCM total 8388608; V4 used 2959872; headroom 5428736 bytes = 5.18 MB.**
- L101 ("FFN pcycles are 22× larger than attention (55.7M vs 2.52M VTCM, or 3.18M V3 DDR)"). FFN is the load-bearing remaining compute mass.
- L73 ("FFN matmuls are the remaining 88% of per-layer byte traffic"). V5 target.
- L159 ("Expected ~30-50% additional prefill lift compounded on V4's 1.434×; structural target = ≥ 1.50× HX.3b STRETCH gate"). LIFT 1.30× over V4 = 2.89 tok/s; STRETCH 1.50× over V4 = 3.34 tok/s.

### S0.6 — V4 implementation (`src/backends/hexagon/dsp/sp_hex_imp.c`)

Key cited lines:
- L715-750: `hx_vtcm_t` struct — V5 extends this with FFN tile-pool fields.
- L778-857: `hx_vtcm_init` — V5 grows the allocation request from `bytes_attn` only to `bytes_attn + 2*tile_slot_bytes` (single `HAP_request_VTCM` call to claim the whole working set including tile slots).
- L882-929: `hx_vtcm_ensure_layer` — V4 logic is unchanged; V5 adds NO per-layer FFN copy here (FFN streaming is tile-loop-driven inside the kernel, not layer-driven).
- L985-1034: `hx_matmul_q8_vrmpy_dual_ctx_v4` — V5 extends with a `_tiled` sibling for FFN.
- L1283-1338: forward dispatch — V5 replaces 3 V3 DDR calls (WGATE/WUP/WDOWN at L1327/L1328/L1335) with new tiled dispatch.

### S0.7 — V3 closure + worker pool (`tools/sp_compute_skel/docs/CLOSURE-TRICK-1-FORWARD-V3.md` summary)

V5 uses the V3 worker pool unchanged: handler + 1 worker, SSR:XA={4,5}, futex signal/wait. `hx_matmul_q8_vrmpy_half` is the silicon-validated half-kernel; V5's tiled variant calls a tile-aware analog `hx_matmul_q8_vrmpy_half_tile` that consumes one VTCM tile at a time but uses the same vrmpy + bias-128 + rsum arithmetic.

### S0.8 — HX.3b closure (`tools/sp_compute_skel/docs/CLOSURE-HX-3b.md` summary)

Call sites: WGATE/WUP/WDOWN, 3 per layer × 26 layers = 78 FFN matmul invocations per prefill forward at the chat-shape (n_tok=16). FFN dims: WGATE/WUP `[FF=6912, E=1152]`; WDOWN `[E=1152, FF=6912]`.

### S0.9 — Hexagon SDK DMA reference

C intrinsics from `Hexagon_SDK\5.5.6.0\tools\HEXAGON_Tools\8.7.06\Tools\target\hexagon\include\hexagon_protos.h`:
- `Q6_dmstart_A(Address Rs)` — `__builtin_HEXAGON_Y6_dmstart` — kick UDMA at descriptor chain address.
- `Q6_R_dmwait()` → Word32 — `__builtin_HEXAGON_Y6_dmwait` — blocks current thread until ALL queued descriptors complete; returns status (DM0 register).
- `Q6_R_dmpoll()` → Word32 — non-blocking status read; bits per `udma.h` (`HEXAGON_UDMA_DM0_STATUS_*`).
- `Q6_R_dmsyncht()` → Word32 — barrier; waits for outstanding DMA + TLB sync.
- `Q6_dmlink_AA(Address Rs, Address Rt)` — append descriptor chain (Rs = tail of current queue, Rt = new descriptor to enqueue).

Descriptor format (`Hexagon_SDK\5.5.6.0\tools\HEXAGON_Tools\8.7.06\Tools\libnative\include\udma.h`):
```
typedef struct hexagon_udma_descriptor_type0_s {
    void *next;              // 0 = end of chain
    unsigned length:24;       // bytes to copy (<= 16 MB)
    unsigned desctype:2;      // 0 = type0 linear
    unsigned dstcomp:1;       // 0 = no DLBC dst compression
    unsigned srccomp:1;       // 0 = no DLBC src compression
    unsigned dstbypass:1;     // 0 = use coherent path (cached); 1 = bypass cache (we want 0 for DDR→VTCM; VTCM side is uncached anyway)
    unsigned srcbypass:1;     // 0 = coherent
    unsigned order:1;         // 0 = no ordering requirement; 1 = ordered with prior descriptors
    unsigned dstate:1;        // dstate=0 incomplete, set to 1 on completion
    void *src;                // physical or virtual VA? — virtual works under user PD per qurt design
    void *dst;
} hexagon_udma_descriptor_type0_t;
```

V5 will use `type0`, linear copy, no compression, bypass off, order=0 (each tile's DMA can race independently; we don't need cross-descriptor ordering because compute waits via the per-slot `dstate` flag or the dmwait barrier before reading any slot's content).

Sync primitive: `Q6_R_dmwait()` per UDMA queue completion. Per `qurt_user_dma.h:24`: "this will stall the current thread till the instruction is complete". Behavior on V69 user PD: works in Unsigned PD (UDMA is a per-thread engine; no privilege gating beyond the standard `qurt_thread`'s memory map).

**UPSTREAM CONCERN U-1:** I have not personally exercised UDMA on Unsigned PD before. The SDK headers expose the intrinsics and the descriptor format. The V69 PRM documents the semantics. If on first silicon run the `dmstart` returns immediately with `dmpoll` showing `ERROR` (status=2), the most likely root cause is one of: (a) Unsigned PD doesn't have UDMA access (security gating I don't know about); (b) descriptor needs physical addresses, not VA, in our PD (would require `qurt_lookup_physaddr` API); (c) src/dst alignment requirement (UDMA may need ≥32B align on DDR — Gemma3 blob is 128B-aligned so this should be fine). Stage 2 of V5 is a one-tile DMA smoke test BEFORE building the ping-pong logic, exactly to surface this.

**Fallback (Stage 2.5 if U-1 hits):** synchronous tile load via scalar memcpy from DDR into VTCM tile slot, NOT via UDMA. Slower (no DMA-compute overlap), but unblocks the tile-streaming kernel + bit-exact validation of the call-site swap. We can then file V5.1 to add UDMA overlap once we resolve the gating issue. Documented as `v5-attempted` per closure spec if it lands without UDMA, but the SDK headers + memory + qurt header `qurt_user_dma_dmsyncht` are all present so I expect UDMA to work.

### S0.10 — `feedback-no-silent-gate-revisions` / `feedback-bundled-changeset-root-cause-ambiguity` / `feedback-leak-gate-allocator-warmup`

Operational discipline:
- If LIFT target misses, name the cost breakdown UPSTREAM, do NOT revise the gate.
- One variable per stage commit; bundle only when iteration cost justifies, and disclose.
- 3-rep mean, warmup separately reported; do not include first-call VTCM init overhead in steady-state perf.

---

## Architectural decisions (surfaced before code)

### D-A — Tile orientation: row-tile along M (output rows)

Gemma3-1B FFN tensor shapes (from `sp_hex_layout.h:80-88` + cfg n_ff=6912, n_embd=1152):
- WGATE: out=FF=6912 rows × in=E=1152 cols → per-row = 1152 int8 bytes = 1.125 KB.
- WUP: same shape as WGATE.
- WDOWN: out=E=1152 rows × in=FF=6912 cols → per-row = 6912 int8 bytes = 6.75 KB.

**Row-tile sizing — D-A-1:** target tile bytes ≤ 1 MB (so a pair = ≤ 2 MB fits with comfort in 5.18 MB headroom plus the scales sub-table per-tile).

For WGATE/WUP (E=1152 K dim):
- 768 rows per tile × 1152 bytes/row = 884,736 bytes ≈ 864 KB int8 codes.
- + 768 × 4 = 3,072 bytes per-row scales.
- + `sp_hex_align` padding → ~864 KB + 4 KB = ~868 KB per tile.
- 6912 / 768 = 9 tiles per matmul.

For WDOWN (FF=6912 K dim):
- 128 rows per tile × 6912 bytes/row = 884,736 bytes ≈ 864 KB int8 codes.
- + 128 × 4 = 512 bytes scales.
- 1152 / 128 = 9 tiles per matmul.

Both shapes converge on **~864 KB int8 + scales ≈ 870 KB per tile**, **9 tiles per matmul**. Symmetry simplifies the allocator: one tile slot size handles both.

**Row-tile bytes (D-A-2):** `tile_bytes = sp_hex_align(rows_per_tile * in_dim) + sp_hex_align(rows_per_tile * sizeof(float))`. Rounded up: 884736 + 3072+padding → set to a constant **`SP_HEX_V5_TILE_BYTES = 1048576` (1 MiB exact)** per slot. Each slot is then 128-byte aligned within VTCM. Two slots = 2 MiB. Plus V4's 2.96 MB attention = 4.96 MB. **VTCM total used: 4.96 MB of 8 MB budget. Headroom remaining 3.04 MB** (room for future activation pinning / decode-path / dual-tier).

Per-matmul rows-per-tile parameter is shape-driven:
- WGATE/WUP (in=1152): `rows_per_tile = (tile_int8_bytes_budget) / sp_hex_align(in)` = `884736 / 1152 = 768` rows. With 6912 rows / 768 = 9 tiles.
- WDOWN (in=6912): `884736 / 6912 = 128` rows. With 1152 / 128 = 9 tiles.

NOTE: actual int8-bytes-per-tile-row = `sp_hex_align(in)` per `sp_hex_layout.h:65-68` (the `sp_hex_q8_bytes` accounting). For in=1152: `sp_hex_align(1152) = 1152` (already 128-mult). For in=6912: `sp_hex_align(6912) = 6912` (also already 128-mult on 128-byte boundary). So the row-stride math is clean.

### D-B — Per-FFN-tensor vs shared tile slots: **shared (D-B chosen)**

Two tile slots, repurposed per FFN matmul invocation. The matmul function claims slots A+B at call entry, streams tiles into them, releases at call exit (purely a struct field reset, no allocation churn).

Memory budget stays within 5 MB headroom regardless of which FFN tensor is active. Across matmul invocations within one layer (WGATE → WUP → WDOWN), tile slot content is overwritten without harm because each matmul re-streams its own tiles.

NOT chosen: per-tensor slots (would need 6× the budget; doesn't fit; doesn't provide perf win because intra-matmul reuse is the only level that matters for streaming).

### D-C — DMA scheme: UDMA via `Q6_dmstart_A` + `Q6_R_dmwait`

Pattern per tile:
1. At matmul entry, allocate descriptor on stack (one type0 descriptor per tile).
2. Pre-prefetch tile 0 into slot A: build descriptor with src=DDR row-block-0 base, dst=VTCM slot A, length=tile_bytes_total. `Q6_dmstart_A(&desc_0)`. `Q6_R_dmwait()` — synchronous on the very first tile only.
3. For tiles 1..N-1: build prefetch descriptor for tile (i+1) targeting the OTHER slot. `Q6_dmstart_A(&desc_iplus1)` (asynchronous — kicks off the engine in the background while compute begins on tile i in the CURRENT slot).
4. Compute tile i: handler+worker dispatched via `hx_matmul_q8_vrmpy_dual_ctx_v4_tile` reading from the CURRENT slot.
5. After compute completes (worker `done` futex + handler return), `Q6_R_dmwait()` to ensure the prefetch for tile i+1 finished before we swap to it.
6. Toggle current/other slots; loop.
7. After the LAST tile compute, no more prefetches; loop exits.

Per-tile descriptor lives in the per-thread descriptor scratch buffer. The handler thread issues `Q6_dmstart_A` for each prefetch. Worker thread does NOT touch DMA — it only reads VTCM via vmem during its half-row range. Only the handler programs UDMA.

**Cache coherency for V5 (UDMA → VTCM → HVX):** UDMA writes to VTCM are visible to HVX vmem reads after `Q6_R_dmwait()` (or equivalently `qurt_user_dma_dmsyncht()`). No further flush/invalidate needed — VTCM is uncached single-port memory addressable by both UDMA engine and HVX vector unit; the engine completion semantics include the visibility ordering. (Confirmation pending Stage 2 smoke; this is the documented behavior per `qurt_user_dma.h` "ensure all posted DMA memory operations are complete".)

**UPSTREAM CONCERN U-2:** Cross-thread DMA visibility. The handler issues `dmstart`; the worker reads the same VTCM slot via vmem. Per Hexagon UDMA model, the DMA engine is per-thread-context (each scalar thread has its own DM0/DM1 status). When the handler `dmwait`s on its own thread, the data is visible to itself. Is it visible to the WORKER thread's vmem reads on the same VTCM address? VTCM is a shared physical region; the question reduces to "do scalar-thread DM0 completion semantics propagate to OTHER threads' HVX accesses?" The HVX vector contexts share L1/VTCM with the scalar threads. By the time the handler signals the worker via `qurt_futex_wake`, the dmwait already returned, so memory is committed. The futex's memory barrier is sufficient. Stage 2 smoke will verify byte-content correctness.

### D-D — Per-thread tile assignment: handler[m_half..m_tile_rows), worker[0..m_half) per tile

Same as V4 attention path: within each tile's row range `[0..tile_rows)`, split at `m_half = (tile_rows+1)/2`. Worker thread takes `[0..m_half)`; handler takes `[m_half..tile_rows)`. Both threads read the SAME tile slot (read-only weights — no cross-context contention beyond VTCM port limit, which V4 silicon-confirmed is ~symmetric for V69).

Per-tile dispatch:
```
for tile i in 0..N:
    swap current/other slots (current = even i ? A : B)
    if i+1 < N: dmstart prefetch (i+1) → other slot
    set desc.blk = current slot
    set desc.j_start = 0, j_end = m_half  (worker range)
    bump seqno, wake worker
    handler runs [m_half, tile_rows) on its half
    wait for worker done
    if i+1 < N: dmwait (ensure other slot ready)
```

Tail-tile edge case: when M not divisible by rows_per_tile (Gemma3-1B's FF=6912 / 768 = 9 exact; E=1152 / 128 = 9 exact — both clean), no tail. Defensive: allocate the last tile with `min(rows_per_tile, M - i * rows_per_tile)` rows; smaller DMA + smaller compute range.

### D-E — Output row indexing: tile-relative, fold to global at compute

Tile contains rows [`i * rows_per_tile`, `min((i+1) * rows_per_tile, M)`) of the GLOBAL output index space. Inside `hx_matmul_q8_vrmpy_half_tile` the local `j` ranges over `[j_start, j_end)` which are LOCAL row indices within the tile. Output write address `Y[t * out + (i * rows_per_tile + j_local)]`.

Scales address inside tile = `scales_tile_base + j_local`; rsum address = `rsum_global + (i * rows_per_tile + j_local)` (or tile-relative rsum if we choose to store the rsum table tile-indexed; **chosen tile-relative for D-E-2**: each tile gets its own contiguous scales+rsum mini-table prepended/appended at memcpy/UDMA time, so the kernel only sees its tile's scales+rsum slab).

Actually simpler — per `sp_hex_layout.h:65-68`, codes precede scales in DDR-resident block. **Decision D-E-2 (REVISED for simplicity):** the tile DMA copies ONLY the int8 codes sub-block (rows × in bytes). The scales table for the WHOLE matmul tensor stays in DDR — scales array is small (FF or E × 4 bytes = 27 KB or 4.6 KB), the handler reads via scalar accesses outside the inner vrmpy loop (one read per output row, NOT bandwidth-critical). Same for rsum: we compute the WHOLE-tensor rsum at first invocation per matmul (lazy, like V4's `hx_compute_rsum` on attention), store malloc'd table, reuse on subsequent layer-loop iterations. Tile kernel only consumes the int8 codes from VTCM tile + scalar scales[j_global] from DDR + scalar rsum[j_global] from DDR-resident table.

This dramatically simplifies the tile DMA — it's a single contiguous `rows_per_tile * sp_hex_align(in)` byte copy per tile.

### D-F — rsum table for FFN: per-matmul-tensor, computed lazily on first call

Different from V4 attention rsum (which was per-layer per-attention-tensor in a single contiguous `rsum_attn` table) because:
- Attention rsum was sized to fit 26 layers × (QD+2*KVD+E) ints ≈ 280 KB, all malloc'd at hx_vtcm_init.
- FFN rsum sized per-tensor is 26 layers × 3 tensors × (FF or E) ints = 26 × (6912+6912+1152) × 4 = 1.56 MB. Still small in absolute terms, fits in DDR easily.

V5 design D-F-1: extend `hx_vtcm_t` with `rsum_ffn[L * 3][FF or E ints]` malloc'd at hx_vtcm_init alongside `rsum_attn`. Per-layer-tensor flags `rsum_ffn_layer_ready[L * 3]` for lazy population. The DDR int8 codes are touched once (for rsum compute) and then per-tile-streamed (UDMA copies).

Sizing: 26 layers × (WGATE[FF=6912] + WUP[FF=6912] + WDOWN[E=1152]) × 4 bytes = 26 × 14976 × 4 = 1,557,504 bytes = 1.49 MB DDR malloc. Trivial.

### D-G — Stretch perf target

- Floor: prefill tok/s ≥ V4 2.226 (PERF_PARITY against V4).
- LIFT: ≥ 1.30× V4 = ≥ 2.89 prefill tok/s. (Operator-named target.)
- STRETCH: ≥ 1.50× V4 = ≥ 3.34 prefill tok/s. (Operator-named target. Compound = 1.50 × 1.434 = 2.15× HX.3b.)

Theoretical ceiling reasoning: V4 closure L101 documents FFN matmul pcycles = 55.7M each (V3 DDR path) — these are the wall-clock dominant. If FFN bandwidth drops at the same ratio attention saw (V4 worker pcyc 3676k → 2521k, ~31% drop), FFN pcycles could drop to ~38M. Total per-layer pcycles before V5: ~10M (attention V4) + ~167M (3 FFN V3) = ~177M. After V5: ~10M + ~114M (3 FFN V5) = ~124M. Speed-up factor = 177 / 124 = 1.43×. Compound = 1.434 (V4) × 1.43 (V5) = 2.05× HX.3b.

This puts the OPTIMISTIC outcome at ~3.18 tok/s (2.226 × 1.43). LIFT 2.89 is more conservative than the pcycle-implied projection. STRETCH 3.34 requires either:
(a) FFN tile-streaming achieves better than 31% bandwidth drop (overlap perfect → ~50% reduction), OR
(b) Some non-FFN portion (RMSNorm, attention head softmax, hx_quant_act_ub) is also part of the win.

Honestly: LIFT is reachable; STRETCH requires either lucky overlap OR additional optimization not in this sprint. V5 closure will name follow-on if STRETCH misses (consistent with V4 closure's discipline).

---

## Scope (what ships in V5)

1. **VTCM tile-pool allocator** — extend `hx_vtcm_init` to allocate `attn_bytes + 2 * tile_bytes` (4.96 MB total) in one `HAP_request_VTCM` call. New struct fields `tile_a_off`, `tile_b_off`, `tile_bytes`.

2. **Per-FFN-tensor rsum table** — extend `hx_vtcm_t` with `rsum_ffn` + `rsum_ffn_layer_ready` + `rsum_ffn_stride` fields. Lazy population at first FFN matmul call.

3. **UDMA tile-prefetch primitive** — internal helper `hx_vtcm_dma_prefetch_tile(src_ddr, dst_vtcm, bytes)` that programs a type0 descriptor + `Q6_dmstart_A`. Companion `hx_vtcm_dma_wait()` calling `Q6_R_dmwait`.

4. **Tile-aware half-kernel** `hx_matmul_q8_vrmpy_half_tile` — reads int8 codes from a VTCM tile slot, uses caller-supplied global scales + rsum table, computes a local row range with global output index mapping.

5. **Tiled dual-context dispatch** `hx_matmul_q8_vrmpy_dual_ctx_v5_tiled` — outer tile loop with ping-pong DMA + dual-context compute per tile. Falls back to V3 single-context per tile if worker-pool init fails.

6. **3 FFN call sites** in `sp_hex_forward` (WGATE/WUP/WDOWN) swapped to new dispatch.

7. **Skel rebuild + push to S22U** via existing `build-hexagon.bat dsp` flow.

8. **3-rep tok/s measurement** vs V4 baseline using `timed_chat.sh` harness; same-session daemon restart cycle.

9. **Closure document** with HEADLINE TABLE (HX.3b, V4, V5) per `feedback-drop-fp32-baseline-comparing-to-ourselves`.

---

## Substantive gates

| Gate | Threshold | Methodology |
|---|---|---|
| **T_V5_TILE_POOL_ALLOCATED** | `HAP_request_VTCM(4.96 MB)` succeeds; 2 tile slot offsets + size FARF-logged | FARF `sp_hex V5: VTCM tile_a_off=X tile_b_off=Y tile_bytes=Z` |
| **T_V5_DMA_PINGPONG_OBSERVED** | Per-tile timing shows DMA+compute total < sum-of-individual | HAP_perf_get_pcycles bracketing DMA-kick + compute-complete; sample first FFN matmul per session |
| **T_V5_DECODE_BIT_EXACT** | 32-token decode matches V4 baseline byte-equal | PowerShell `Compare-Object` on stripped SSE deltas across 3 reps |
| **T_V5_PERF_PARITY** | prefill tok/s ≥ V4 2.226 | 3-rep mean same-session |
| **T_V5_PERF_LIFT** | prefill tok/s ≥ 1.30× V4 = ≥ 2.89 | 3-rep mean |
| **T_V5_PERF_STRETCH** | prefill tok/s ≥ 1.50× V4 = ≥ 3.34 | 3-rep mean |

If T_V5_PERF_LIFT FAILS:
- (a) DMA overhead dominates → file V6 (bigger tiles or batched DMA + descriptor chaining)
- (b) TCM read-port contention with attention reads → file V7 (residence partitioning)
- (c) UDMA-write/HVX-read coherency forces stalls → file V8 (manual cache management or explicit barriers)
- (d) VTCM bandwidth saturated even with one tensor resident → file V9 (DDR L2-pin + tile pre-staging)

NO SILENT GATE REVISIONS.

---

## Workflow discipline — staged commits

1. **Plan commit (THIS)** — `[plan] V5 — FFN VTCM ping-pong tile-streaming`. Captures Stage 0 citations + D-A through D-G + U-1, U-2 + budget math + cache-coherency strategy.

2. **Stage 1 commit** — VTCM tile-pool allocator + rsum_ffn table allocation in `hx_vtcm_init`. No DMA, no kernel change. Smoke: skel builds; daemon starts; FARF shows `tile_a_off`/`tile_b_off`/`tile_bytes`. T_V5_TILE_POOL_ALLOCATED.

3. **Stage 2 commit** — UDMA primitive + single-tile smoke (one DMA copy from DDR weight into VTCM slot A, then byte-compare VTCM content to DDR source via `Q6_R_dmwait` + scalar read-back checksum). Verifies: U-1 resolved (UDMA works in Unsigned PD); U-2 noted as resolved on visibility.

4. **Stage 2.5 contingency** — if Stage 2 reveals UDMA non-functional under our PD, swap to scalar memcpy tile-load (still synchronous; no DMA overlap). Document the loss honestly; V5 still ships but with reduced expected lift; LIFT gate may miss; follow-on V5.1 to revisit UDMA.

5. **Stage 3 commit** — tile-aware half-kernel `hx_matmul_q8_vrmpy_half_tile` + tiled dual-ctx dispatch + ONE call-site swap (WGATE only, layer 0; other FFN matmuls remain V3 DDR). Run 32-token chat; verify decode bit-exact via Compare-Object to V4 baseline. T_V5_DECODE_BIT_EXACT preliminary.

6. **Stage 4 commit** — extend to all 3 FFN matmuls (WGATE/WUP/WDOWN) across all 26 layers. Run 32-token chat; full decode bit-exact verification.

7. **Stage 5 commit** — `Q6_R_dmpoll`/`HAP_perf_get_pcycles` instrumentation for T_V5_DMA_PINGPONG_OBSERVED; FARF the first-FFN-matmul-per-session sample line.

8. **Stage 6 commit** — 3-rep same-session measurement vs V4 baseline; closure document; sub-tag.

Per `feedback-bundled-changeset-root-cause-ambiguity`: each stage is one variable. Stage 3 + 4 are split because the call-site widening (1 FFN → 3 FFN × 26 layers = 78 sites) is the load-bearing scope expansion; if Stage 3 misbehaves, the diff is small.

---

## Anti-contamination

ALL V5 work confined to `engine-v5-ffn-vtcm` worktree on `sprint/v5-ffn-vtcm` branch. DO NOT modify V4 (`engine-trick-1-fwd-v4`), V3 (`engine-trick-1-fwd-v3`), K.beta.2.5c, NTT.5a/b/c, or any other concurrent worktree. math-core submodule unchanged. Host code (`sp_hex_host.c`) unchanged — only the cDSP skel (sp_hex_imp.c → libsp_hex_skel.so) rebuilds and pushes.

---

## Hardware + deployment

- Knack's S22 Ultra R5CT22445JA. ADB device accessible from Windows host.
- V4 skel currently deployed at `/data/local/tmp/sp22u/libsp_hex_skel.so`. V5 will replace it.
- Daemon `sp-daemon-wire-hex` starts via `/data/local/tmp/start_wire_hex_daemon.sh`; restart cycle: kill PID, sleep 1, run start script, sleep 25 for load.
- Measurement: `timed_chat.sh '[2,100,...,1500]' 32` with 16-token prompt + 32-token decode; harvest `FIRST_DELTA_MS_FROM_START` (prefill) + `STEADY_DECODE_MS` (decode).
- 3-rep same-session is the discipline floor; HX.3b SHA from V4 closure stable for reference.

---

## Time budget

- Stage 1 (allocator extension): ~150 LOC. ~1-2 hours.
- Stage 2 (UDMA primitive + smoke): ~100 LOC + smoke debug. ~2-3 hours if U-1 resolves cleanly; +2-4 hours if Stage 2.5 fallback.
- Stage 3 (tile half-kernel + tiled dispatch + 1 site): ~250 LOC. ~2-3 hours.
- Stage 4 (extend to all 3 FFN × 26 layers): ~50 LOC (just 3 dispatch line swaps). ~1 hour incl. test.
- Stage 5 (instrumentation): ~50 LOC. ~30 min.
- Stage 6 (measurement + closure): ~3 hours run + write.

Total estimate: **~10-14 hours** for clean UDMA path; **~14-18 hours** if Stage 2.5 contingency.

---

## What V5 ships ENABLES

- The 4 attention matmuls (V4) + 3 FFN matmuls (V5) all run through VTCM-resident weights with dual-context HVX. **Every Q8 matmul in Gemma3-1B prefill goes through the silicon-validated compound on cDSP.**
- The cDSP-prefill story is functionally complete. Remaining axes: decode path (HEX-DECODE-1 follow-on), cross-backend (NEON ARM-q2, WIRE-AVX), integer-end-to-end (Path 2 Frobenius lift activations to Z_q eliminating fp32 dequant).

---

## Closure deliverables (Stage 6)

1. HEADLINE TABLE — 3 lattice rows: HX.3b 1.552 / V4 2.226 / V5 [observed]. NO fp32 reference row.
2. Compound ratios: V5/V4 (primary), V5/HX.3b (compound).
3. Gate-by-gate disposition (6 substantive + DECODE_BIT_EXACT).
4. Architectural decisions D-A through D-G recap with empirical observations.
5. VTCM budget actual layout (attn 2.96 MB + 2 tile slots × 1 MB + rsum_ffn + headroom).
6. UDMA API + per-tile timing breakdown from T_V5_DMA_PINGPONG_OBSERVED sample.
7. Per-stage build + run commands (reproducible).
8. Skel pre/post hashes.
9. Wall-clock breakdown if win is partial; UPSTREAM follow-on (V6/V7/V8/V9) named.
10. Honest interpretation: how much V5 compounded V4; if STRETCH missed, root-cause + follow-on.
11. Files changed with LOC delta.
12. Commits on `sprint/v5-ffn-vtcm`.
13. Sub-tag: `lat-phase-2-v5-ffn-vtcm-shipped-lift` (LIFT pass) / `-shipped-stretch` (STRETCH pass) / `-shipped` (PARITY only) / `-attempted` (PARITY fail, blocker named).
14. What's NOT done — HEX-DECODE-1, WIRE-AVX, NEON ARM-q2, TRICK-1-FORWARD cross-island, NTT.5d HD=256, M.0-real SFT.
15. What unblocks — if shipped: cDSP prefill compound story functionally complete; integer-end-to-end is the next architectural axis.
16. Worktree status (push branch).

Push: `git push -u origin sprint/v5-ffn-vtcm`. Operator merges.
