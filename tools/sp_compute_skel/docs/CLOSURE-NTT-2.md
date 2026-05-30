# CLOSURE-NTT-2 — Twiddle factor VTCM staging

## Headline

Sprint NTT.2 — twiddle factor VTCM staging — **CLOSED (3 / 3 gates PASS)**
on Knack's Samsung Galaxy S22 Ultra (Snapdragon 8 Gen 1, Hexagon V69 cDSP,
Path B Unsigned PD).  All 6 (prime, N) twiddle tables now precompute
once at `ntt_twiddle_init` (qaic method 13), pin in VTCM via
`HAP_request_VTCM`, and stride-1 access is available to NTT.1 HVX
butterflies + NTT.4 INTT consumers via the new
`sp_compute_ntt_twiddle_view` skel-internal accessor.

Total VTCM use: 35,840 bytes — 1.71% of the 2 MB budget.  No
divergences (0 / 35,792 bytes) from the host-side reference.  Idempotent
init wall: 625 µs first call, 134 µs subsequent.

## Gates

| Gate                       | Result | Observed                                                                          |
|---------------------------|--------|-----------------------------------------------------------------------------------|
| T_NTT2_TWIDDLE_INIT        | PASS   | 6/6 (q_idx, N) tables present; VTCM addrs 0xff000000..0xff140000                  |
| T_NTT2_TWIDDLE_BIT_EXACT   | PASS   | 36 / 36 tables compared; 35,792 B; 0 divergences                                  |
| T_NTT2_VTCM_BUDGET         | PASS   | 35,840 B total ≤ 2,097,152 B budget (1.71% envelope use)                          |

## Init strategy decision

**Lazy on first `ntt_twiddle_init` call, idempotent across calls.**

Per PLAN-NTT-2.md Stage 0 rationale:
- Eager-at-skel-load was rejected (qaic/SDK init hooks are fragile
  relative to the Path B Unsigned-PD admission flow; init failures need
  an IDL return code).
- Lazy-on-first-NTT-invoke was rejected (silently changes timing of the
  first `ntt_oracle` call — bad for K.beta.2.5b-style leak / parallelism
  measurements).
- Explicit `ntt_twiddle_init(N)` method gives the ARM-side daemon exact
  control.  Idempotent: second call → fast path (134 µs vs 625 µs
  initial; sub-table addresses unchanged across calls per smoke
  observation).

**Confirmation on silicon:** 2nd `ntt_twiddle_init(N=512)` took 134 µs
(idempotent fast-path branch) vs the 625 µs initial allocate-and-
compute path.

## VTCM layout (silicon-observed)

V69 VTCM allocator places each `HAP_request_VTCM` arena at 0x40000-stride
(256 KB) bases:

| q_idx | N   | vtcm_addr_lo | vtcm_size | psi_pow_off | ipsi_pow_off | w_fwd_off | w_inv_off | w_fwd_stages_off | w_inv_stages_off |
|-------|-----|--------------|-----------|-------------|--------------|-----------|-----------|------------------|------------------|
| 0     | 128 | 0xff000000   | 2560      | 0           | 512          | 1024      | 1280      | 1536             | 2044             |
| 0     | 256 | 0xff040000   | 5120      | 0           | 1024         | 2048      | 2560      | 3072             | 4092             |
| 0     | 512 | 0xff080000   | 10240     | 0           | 2048         | 4096      | 5120      | 6144             | 8188             |
| 1     | 128 | 0xff0c0000   | 2560      | 0           | 512          | 1024      | 1280      | 1536             | 2044             |
| 1     | 256 | 0xff100000   | 5120      | 0           | 1024         | 2048      | 2560      | 3072             | 4092             |
| 1     | 512 | 0xff140000   | 10240     | 0           | 2048         | 4096      | 5120      | 6144             | 8188             |

Arena size = `((20N - 8 + 127) & ~127)` — 128-byte rounded.  All
sub-tables 4-byte aligned within each arena; the first sub-table
(`psi_pow`) inherits the page-aligned arena base.

## Per-stage compaction layout

For each (prime, N), the compacted per-stage forward/inverse twiddle
arrays are laid out stage-major.  Stage `s` (1 ≤ s ≤ log2(N)) has
`half_s = 2^{s-1}` entries at byte offset `4 × (2^{s-1} - 1)` from the
compacted region base.  Total compacted = `N - 1` entries per direction.

Example N=512, log2(N)=9:

| stage s | half_s | offset (entries) | offset (bytes) | size (bytes) |
|---------|--------|------------------|----------------|--------------|
| 1       | 1      | 0                | 0              | 4            |
| 2       | 2      | 1                | 4              | 8            |
| 3       | 4      | 3                | 12             | 16           |
| 4       | 8      | 7                | 28             | 32           |
| 5       | 16     | 15               | 60             | 64           |
| 6       | 32     | 31               | 124            | 128          |
| 7       | 64     | 63               | 252            | 256          |
| 8       | 128    | 127              | 508            | 512          |
| 9       | 256    | 255              | 1020           | 1024         |
| (total) | 511    |                  |                | 2044         |

The compaction makes NTT.1's HVX butterfly access stride-1 within each
stage: stage s reads `w_fwd_stages[offset_s .. offset_s + half_s)`
sequentially, not `w_fwd[k × step]` with stride > 1 as in math-core's
`ntt_core` (ntt_crt.c:241-254).

Same layout applies to N=256 (log2=8, total 255 = 1020 B) and N=128
(log2=7, total 127 = 508 B).

## Files changed

| Path                                                          | LOC delta | Note                                                                  |
|---------------------------------------------------------------|-----------|-----------------------------------------------------------------------|
| `tools/sp_compute_skel/docs/PLAN-NTT-2.md`                    | +269      | new; Stage 0 citations + init strategy + compaction layout            |
| `tools/sp_compute_skel/src_dsp/sp_compute_ntt_twiddle.c`      | +488      | new; ctx struct + 3 IDL handlers + skel-internal view accessor        |
| `tools/sp_compute_skel/inc/sp_compute.idl`                    | +66       | 3 new methods (`ntt_twiddle_init`, `ntt_twiddle_status`, `ntt_twiddle_dump`) |
| `tools/sp_compute_skel/CMakeLists.txt`                        | +2        | add `sp_compute_ntt_twiddle` to `srcs`                                |
| `tools/sp_dsp_smoke/Cargo.toml`                               | +9        | new `[[bin]] sp_ntt_2_smoke`                                          |
| `tools/sp_dsp_smoke/src/sp_ntt_2_smoke.rs`                    | +325      | new; gates harness with host-side Rust ref re-implementing prime_setup |
| `tools/sp_daemon/scripts/ntt_2_full_report.json`              | +58       | machine-parseable gate results                                        |
| `tools/sp_daemon/scripts/ntt_2_full_run.txt`                  | (verbatim) | S22U run capture                                                      |
| `tools/sp_compute_skel/docs/CLOSURE-NTT-2.md`                 | (this)    |                                                                       |

## Commits on `sprint/ntt-2`

Full chain (base `f834bff` engine main → tip after push):

```
fb80b1f  [plan] NTT.2 - twiddle VTCM staging (precomputed at init, stride-1 access for HVX butterflies)
bbee93d  [NTT.2] feat: Stage 1+2 -- twiddle VTCM staging skel C + IDL methods 14/15 + CMakeLists
b4b11a9  [NTT.2] feat: add ntt_twiddle_dump IDL method (qaic case 15)
1619a71  [NTT.2] test: Stage 3 -- T_NTT2 gates on S22U (all PASS)
<this>   [NTT.2] doc: Stage 4 -- closure (all 3 gates PASS)
```

## Sub-tag (after merge)

`lat-phase-4-ntt-2-twiddle-vtcm`

## What's NOT done (intentionally — out of NTT.2 scope)

- **HVX butterfly intrinsics** — NTT.1's lane (`sp_compute_ntt_hvx_imp.c`,
  concurrent worktree).  NTT.2 produces the VTCM-resident stride-1
  tables that NTT.1's butterflies will consume; the two sprints
  compose at the post-merge boundary.
- **Dual-prime CRT dispatch** — NTT.3 scope.  Requires NTT.1 + NTT.2
  both closed.
- **INTT (inverse transform) kernel** — NTT.4 scope.  NTT.2's
  `ipsi_pow` + `w_inv` + `w_inv_stages` tables ARE computed and stored
  in VTCM so NTT.4 can consume them directly via the existing
  `sp_compute_ntt_twiddle_view` accessor; only the kernel + IDL surface
  is new in NTT.4.
- **MeMo integration** — NTT.5 scope.
- **Engine-side glue** — `sp_engine_poly_ntt_crt` env activation +
  forward-pass wiring; out-of-tree for NTT.* sub-phase.

## What unblocks

- NTT.3 (dual-prime CRT dispatch) can dispatch once both NTT.1 and
  NTT.2 are merged.  NTT.3 consumes the VTCM-resident tables from
  both primes concurrently via the same `sp_compute_ntt_twiddle_view`
  C-internal API.
- NTT.4 (INTT) can dispatch on NTT.2 alone (does not require NTT.1's
  HVX butterfly; an INTT scalar oracle paired with NTT.2's VTCM tables
  is enough).  Operator triage.
- Engine-side `SP_ENGINE_POLY_NTT_CRT_DSP=1` env path (sp_daemon glue)
  can begin once NTT.3 lands.

## Memory entry candidates

1. **`reference-ntt-vtcm-twiddle-layout`** — proposed.  Captures the
   silicon-observed VTCM layout (256 KB-stride arena bases on V69),
   the `20N - 8` arena size formula, and the per-stage compaction
   offsets.  Single 200-char index line + a detail doc with the table
   from this closure.  File when operator triages.

2. **`reference-qaic-method-numbering-discipline`** — proposed.
   Confirms qaic assigns method indices in IDL declaration order
   (post-`open`/`close`).  When two parallel agents both add new
   methods, the merge order determines the final qaic case numbers;
   Rust smokes pick up numbers from the post-merge skel.c switch.
   NTT.2 anticipated 14/15 in the plan but qaic emitted 13/14/15
   because the dump method was added Stage 3; documented openly
   without silent revision.

3. **Update to `reference-v69-hvx-expert-practices`** — VTCM allocator
   stride observation (256 KB between independent `HAP_request_VTCM`
   calls on V69).  Useful for future multi-arena planning under the
   §11 Mode D umbrella.

## Worktree status

- **Sole worktree:** `D:\F\shannon-prime-repos\engine-ntt-2` — all
  commits on `sprint/ntt-2` originated here.
- **Anti-contamination compliance:**
  - `engine-ntt-1\` (concurrent NTT.1 worktree) — **not touched**.
  - `engine-ntt-0\` (NTT.0 closure worktree) — **not touched**.
  - `shannon-prime-system-engine\` (main worktree) — **not modified**.
  - Other `engine-*` / `lattice-*` worktrees — **not touched**.
  - `lib/shannon-prime-system` submodule — read-only reference; init
    via `git submodule update --init` was required to build math-core
    static libs for the smoke link line (per NTT.0 cross-compile
    protocol); submodule SHA unchanged.
- Method assignment risk noted in PLAN-NTT-2.md was correctly
  anticipated: PLAN expected 14/15 (assuming NTT.2 merges second);
  qaic actually emitted 13/14/15 because the dump method was added
  in NTT.2's own sequence.  No conflict with NTT.1 expected at merge:
  NTT.1's methods will fall after NTT.2's at indices 16+ (or get
  renumbered if operator merges NTT.1 first, in which case NTT.2
  smoke would need a one-line method-index fix).
