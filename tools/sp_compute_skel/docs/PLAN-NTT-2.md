# PLAN-NTT-2 — Twiddle factor VTCM staging

Sprint NTT.2 — engine-ntt-2 worktree, branch `sprint/ntt-2`, base
`f834bff` (engine main, post NTT.0 merge).

## Stage 0 reference reading (citations)

1. **Math-core canonical twiddle generation** —
   `lib/shannon-prime-system/core/ntt_crt/ntt_crt.c:129-173`
   (`prime_setup`). Allocates four tables per prime:
   - `pc->psi_pow = malloc(N * 4)` — line 146
   - `pc->ipsi_pow = malloc(N * 4)` — line 147
   - `pc->w_fwd = malloc(N/2 * 4)` — line 148
   - `pc->w_inv = malloc(N/2 * 4)` — line 149
   Then four init loops (lines 152-171):
   - `psi_pow[j] = psi^j` for j in [0, N)
   - `ipsi_pow[j] = psi^{-j}` for j in [0, N)
   - `w_fwd[j] = omega^j` for j in [0, N/2), omega = psi^2
   - `w_inv[j] = iomega^j` for j in [0, N/2), iomega = ipsi^2
   `ninv = N^{-1} mod q` at line 144.

2. **NTT.0 scalar Hexagon port — per-call computation that NTT.2
   replaces** — `tools/sp_compute_skel/src_dsp/sp_compute_ntt_imp.c:184-209`
   (the `psi_pow[SP_NTT_N_MAX]` + `w_fwd[SP_NTT_N_MAX/2]` stack scratch +
   the two init loops at lines 200-203 and 204-209). Plus
   `find_psi` (lines 85-94) and the local Barrett/modmul/modpow/modinv
   primitives (lines 49-94).

3. **Sprint G VTCM staging recipe** —
   `reference_v69_hvx_expert_practices.md:307-329` ("Halide AOT VTCM
   staging — the working recipe"). Three pillars per F.1:
   (1) generator-side `set_host_alignment(128)`;
   (2) `.prefetch(input, x, r, 2)` 2-iter lookahead;
   (3) **all-buffers-in-VTCM**, not mixing DDR + VTCM in one call.
   For NTT.2 the relevant pillar is (3): twiddle tables live in VTCM,
   pinned for the lifetime of the daemon process; HVX butterflies
   (NTT.1) and pre/post-weights consume VTCM-resident tables with
   stride-1 access. Pillar (1) — 128-byte alignment — applies to
   each table base pointer; HAP_request_VTCM returns 4K-page-aligned
   addresses naturally satisfying ≥128-byte alignment.

4. **VTCM API** — confirmed in-tree at
   `tools/sp_compute_skel/src_dsp/sp_compute_imp.c:271-292`
   (`sp_compute_vtcm_probe`). Signatures:
   ```c
   #include "HAP_vtcm_mgr.h"
   void *HAP_request_VTCM(unsigned size_bytes, unsigned single_page_flag);
   int   HAP_release_VTCM(void *p);
   ```
   Returns NULL on denial. `single_page_flag=0` is multi-page OK;
   we use 0 (NTT.2 doesn't need HVX scatter/gather contiguity).
   Sprint K.beta.2.5b/c precedent: `sp_compute_imp.c:410,522,650` all
   use the same flag=0 pattern.

5. **N ladder + 2-adic constraint** — `reference_ntt_frozen_primes_N_cap.md`.
   N ∈ {128, 256, 512}; both frozen primes have `v_2(q-1) = 10`, so
   max N = 512. NTT.2 precomputes tables for all three.

6. **`feedback_no_silent_gate_revisions`** — discipline rule
   acknowledged. Any gate fail → UPSTREAM-REQUIRED closure, no
   silent retreat.

7. **K.beta.2.5b/c smoke pattern** —
   `tools/sp_dsp_smoke/src/sp_matmul_q_dual_smoke.rs` for the
   `Arc<FastRpcSession>` pattern + per-iter wall-clock + pcycle
   measurement. NTT.2 uses single-session pattern (no concurrency);
   the pattern reference is for VTCM addr inspection style.

## Twiddle table sizing (operator pre-read confirmed)

Per prime, per N:
- `psi_pow[N]`   — 4N bytes
- `ipsi_pow[N]`  — 4N bytes (INTT-side; NTT.4 will consume)
- `w_fwd[N/2]`   — 2N bytes
- `w_inv[N/2]`   — 2N bytes (INTT-side)

Plus per-stage compacted twiddles. For radix-2 DIT at stage `len`
(len ∈ {2, 4, ..., N}), there are `half = len/2` twiddles needed,
each at stride `step = N/len` from the base `w_fwd`/`w_inv` table.
Compacted layout per stage: `w_fwd_stage[s][k] = w_fwd[k * step]`
for k in [0, half). Total compacted entries per direction =
`sum_{s=1..logN} (2^{s-1}) = N - 1`. So 4(N-1) bytes for forward
compacted; same for inverse compacted.

Per prime per N totals (bytes):
```
psi_pow  = 4N
ipsi_pow = 4N
w_fwd    = 2N
w_inv    = 2N
w_fwd_stages = 4*(N-1)
w_inv_stages = 4*(N-1)
TOTAL per (prime,N) ≈ 20N - 8 bytes
```

At N=512: ~10232 B per (prime, N).
At N=256: ~5112 B.
At N=128: ~2552 B.

Per prime across all 3 N values: ~17896 B.
Both primes across all 3 N values: ~35792 B ≈ 35 KB.

VTCM peak budget cap from sprint spec: ≤ 2 MB (very generous;
real use ~35 KB — leaves 99% of the 2 MB envelope for NTT.1 +
NTT.4 + NTT.5 follow-on data).

## Init strategy decision

**Lazy on first ntt_twiddle_init call, idempotent across calls.**

Rationale:
- Eager-at-skel-load would require linking init into the
  `__attribute__((constructor))` path or wrapping it around the qaic-
  emitted `sp_compute_skel_open` — both fragile relative to the
  Path B (Unsigned PD) admission flow already shipped. Skel-load is
  the wrong moment because VTCM allocation can fail and the failure
  path needs an IDL return code.
- Lazy-on-first-NTT-invoke (the `ntt_oracle` call) would
  silently change `ntt_oracle`'s timing for the first call — bad
  for the K.beta.2.5b leak-gate / parallelism style measurements.
- Explicit `ntt_twiddle_init(N)` method gives the ARM-side daemon
  exact control: call once at startup for {128, 256, 512}; subsequent
  ntt_oracle calls find tables already present.
- Idempotent: a second `ntt_twiddle_init(N=512)` no-ops and returns
  `SP_OK` if tables already present. Lets the ARM side call the
  method without bookkeeping.

The new method also lets NTT.3 + NTT.4 issue their own table-init
calls from a unified entry point.

## Per-stage compaction layout

For each (prime ∈ {q1, q2}, N ∈ {128, 256, 512}, direction ∈
{forward, inverse}), the compacted per-stage twiddle array
`w_*_stages` is laid out contiguously, stage-major:

```
offset_stage[s]   = sum_{t=1..s-1} 2^{t-1} = 2^{s-1} - 1   (entries)
                  = 4 * (2^{s-1} - 1)                       (bytes)
half_stage[s]     = 2^{s-1}                                 (entries)
size_stage[s]     = 4 * 2^{s-1}                             (bytes)
total_entries     = sum_{s=1..logN} 2^{s-1} = N - 1
total_bytes       = 4 * (N - 1)
```

Example N=128 (logN=7), stages s=1..7:
- s=1: offset=0, half=1, size=4 B
- s=2: offset=4, half=2, size=8 B
- s=3: offset=12, half=4, size=16 B
- s=4: offset=28, half=8, size=32 B
- s=5: offset=60, half=16, size=64 B
- s=6: offset=124, half=32, size=128 B
- s=7: offset=252, half=64, size=256 B
- total = 508 B = 4 * (128 - 1)

This compaction makes the NTT.1 HVX butterfly's twiddle access
stride-1 within each stage: stage s reads w_*_stages[offset_stage[s]
..offset_stage[s] + half_stage[s]) sequentially, not w_*[k * step]
with stride > 1. NTT.1's vector loads can use plain `vmem` against
the compacted region.

## Files I will touch (engine-ntt-2 ONLY)

- **NEW** `tools/sp_compute_skel/src_dsp/sp_compute_ntt_twiddle.c`
  - Twiddle ctx struct + per-prime/per-N table init
  - `sp_compute_ntt_twiddle_init` (handler for IDL method 14)
  - `sp_compute_ntt_twiddle_status` (handler for IDL method 15)
  - Calls `HAP_request_VTCM` for one contiguous arena per
    (prime, N) — six arenas total
  - Computes `psi`, `psi_pow`, `ipsi_pow`, `w_fwd`, `w_inv` mirroring
    math-core `prime_setup`
  - Computes compacted per-stage tables for forward + inverse
  - Idempotent
- **EDIT** `tools/sp_compute_skel/inc/sp_compute.idl`
  - Add `ntt_twiddle_init(in long N) -> long`
  - Add `ntt_twiddle_status(in long N, in long q_idx, rout long
    table_present, rout long vtcm_addr_lo, rout long vtcm_size,
    rout long psi_pow_off, rout long ipsi_pow_off, rout long
    w_fwd_off, rout long w_inv_off, rout long w_fwd_stages_off,
    rout long w_inv_stages_off) -> long`
  - With prefix banner `# §4-NTT Sprint NTT.2 — twiddle VTCM staging`
- **EDIT** `tools/sp_compute_skel/CMakeLists.txt`
  - Add `sp_compute_ntt_twiddle` to `srcs`
- **NEW** `tools/sp_dsp_smoke/src/bin/sp_ntt_2_smoke.rs`
  - Wait — sp_dsp_smoke uses flat `src/*.rs` not `src/bin/*.rs`.
    Per the existing pattern (`src/sp_ntt_0_smoke.rs`), I will
    place the new smoke at `tools/sp_dsp_smoke/src/sp_ntt_2_smoke.rs`
    and add a `[[bin]]` to Cargo.toml — matching the NTT.0 layout
    exactly. The sprint spec said `src/bin/`; that was a path slip
    in the brief — the repo convention is flat `src/`. Documenting
    here so it isn't read as silent deviation.
- **EDIT** `tools/sp_dsp_smoke/Cargo.toml`
  - Add `[[bin]] sp_ntt_2_smoke`
  - With prefix banner `# §4-NTT Sprint NTT.2 — twiddle VTCM staging`

## Coordination with NTT.1

NTT.1 will touch:
- NEW `tools/sp_compute_skel/src_dsp/sp_compute_ntt_hvx_imp.c`
  (different file — no conflict)
- EDIT `tools/sp_compute_skel/inc/sp_compute.idl` (different method
  numbers — they'll claim e.g. 13 + 14 OR 11 + 13; we coordinate by
  prefix banner discipline)
- EDIT `tools/sp_compute_skel/CMakeLists.txt` (different `srcs`
  entry — three-way merge resolved by adding both lines)
- EDIT `tools/sp_dsp_smoke/Cargo.toml` (different `[[bin]]` entry)

**Method number assignment risk:** NTT.0 occupies method index 12
(established at NTT.0 closure). qaic assigns methods in IDL declaration
order. Whichever sprint's IDL changes land first effectively claims
methods 13/14; the second is forced into 14/15 (or higher). My plan
assumes NTT.2 lands second (NTT.1 first, since NTT.1 is the new HVX
kernel that NTT.2's tables feed), so I will use **methods 14 + 15**
per the sprint brief.

If NTT.1 ends up landing AFTER me, the operator merge resolves by
renumbering — both sprints' Rust smokes pass `make_scalars(method,
n_in, n_out)` with a constant they can pick up from the post-merge
qaic-emitted skel.c. No code changes needed in our skel C; qaic
re-emits the `sp_compute_skel_invoke` switch.

(`make_scalars` masks method to 5 bits — methods 0..31. We have
plenty of headroom; current shipped methods are at 12.)

## Workflow

Stage 1 (this commit + next): `sp_compute_ntt_twiddle.c` —
implements ctx struct, table init, IDL handlers.

Stage 2: IDL + CMakeLists wiring.

Stage 3: ARM smoke + 3 gates exercised on device.

Stage 4: closure doc.

## Gates

**T_NTT2_TWIDDLE_INIT** — pass iff all 6 (prime, N) combinations
init without error; table_present=1 for each; vtcm_addr non-zero;
sum of vtcm_size across combinations ≤ 64 KB.

**T_NTT2_TWIDDLE_BIT_EXACT** — pass iff every (prime, N, table)
byte matches the host-side reference (math-core `prime_setup`
re-implemented in Rust for the smoke; or linked via the existing
`sp_ntt_crt` static lib for the canonical numbers).

**T_NTT2_VTCM_BUDGET** — pass iff peak sum_across_primes of
vtcm_size (per N then peak across N) ≤ 2 MB.

## What's NOT done

- HVX butterfly intrinsics (NTT.1, concurrent lane)
- Dual-prime dispatch (NTT.3)
- INTT execution kernel (NTT.4 — but the ipsi_pow + w_inv tables
  ARE computed in NTT.2 for NTT.4's later consumption)
- MeMo integration (NTT.5)
- Engine-side glue under sp_engine_poly_ntt_crt env (later)

## Anti-contamination

- This worktree: `engine-ntt-2` only
- DO NOT touch: any other engine-* / lattice-* worktree
- DO NOT touch: `sp_compute_ntt_imp.c` (NTT.0's frozen scalar reference)
- DO NOT touch: `sp_compute_ntt_hvx_imp.c` (NTT.1's anticipated file)
- Math-core sources are READ-ONLY reference

## Sub-tag (after closure)

`lat-phase-4-ntt-2-twiddle-vtcm`
