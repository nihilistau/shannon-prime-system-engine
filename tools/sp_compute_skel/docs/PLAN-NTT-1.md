# PLAN — Sprint NTT.1 (HVX-vectorized NTT butterfly core)

**Date:** 2026-05-30
**Branch:** `sprint/ntt-1` (engine worktree `D:\F\shannon-prime-repos\engine-ntt-1`)
**Base:** engine main @ f834bff (NTT.0 closed; ntt_oracle method 12 + scalar reference shipped)
**Concurrent sibling:** Sprint NTT.2 in `engine-ntt-2` (twiddle VTCM staging; own lane).

## Headline

Vectorize the radix-2 DIT NTT butterfly inner loop using hand-rolled HVX
intrinsics, reusing the K.beta.2.5b/c silicon-confirmed `Q6_W_vmpye_VwVuh +
Q6_W_vmpyoacc_WVwVh` 32×32→64 widening idiom for the per-lane Barrett modmul.
Large-stage path (half ≥ 32) goes through HVX; small-stage path (half < 32)
falls back to the NTT.0 scalar implementation. New IDL method
`ntt_hvx_oracle` (method 13); existing `ntt_oracle` (method 12, NTT.0
scalar) is untouched and continues to PASS its 600/600 reference gate.

## Stage 0 reference-read citations (mandatory plan opener)

1. **Math-core canonical NTT — primary reference.**
   - `lib/shannon-prime-system/core/ntt_crt/ntt_crt.c` (math-core submodule,
     verified via the `engine-ntt-0` worktree which has the submodule checked out).
   - `ntt_core` butterfly inner loop: lines **225-255** (function header
     line 229; logN-stage outer loop line 241; per-stage `len/half/step`
     computation lines 242-243; the actual butterfly `(u, v)` computation
     lines 246-251). The reference's per-stage stride is
     `step = N / len`, twiddles accessed as `wtab[widx]` with `widx +=
     step` per butterfly.
   - `forward_one` pre-weight: lines **259-272**. Step 1 pre-weights
     `out[j] = (in[j] mod q) * psi_pow[j] mod q` for j in [0, N). Step 2
     bit-reversal. Step 3 calls `ntt_core` with `wtab = pc->w_fwd`.
   - Public dual-prime API: `ntt_forward` at lines 274-278.

2. **NTT.0 scalar Hexagon port** —
   `tools/sp_compute_skel/src_dsp/sp_compute_ntt_imp.c` (whole file).
   - IDL handler `sp_compute_ntt_oracle` at lines **167-226** (method 12).
   - Per-prime scalar butterfly `ntt_forward_one_scalar` at lines
     **116-152** — byte-identical to math-core `forward_one + ntt_core`.
   - Inline Barrett primitive lines **49-54** (`barrett_reduce` mirrors
     math-core ntt_crt.c:72-78 with shifts 29 / 31, mu = floor(2^60/q)).
   - `find_psi` + twiddle precompute lines 85-94 + 191-209.
   - Frozen prime constants lines **32-35**: SP_NTT_Q1=1073738753,
     SP_MU_Q1=1073744895, SP_NTT_Q2=1073732609, SP_MU_Q2=1073751039,
     SP_NTT_Q_BITS=30. `SP_NTT_N_MAX = 512` (line 41).

3. **K.beta.2.5b/c HVX scalar Barrett primitive** —
   `tools/sp_compute_skel/src_dsp/sp_compute_crt_imp.c`.
   - `sp_barrett_reduce32_hvx_lane` at lines **74-123**. Signature:
     `static inline HVX_Vector sp_barrett_reduce32_hvx_lane(HVX_Vector
     va, HVX_Vector vb, HVX_Vector vq, HVX_Vector vq_minus_1, HVX_Vector
     vmu)`. 19 inner-loop intrinsics, silicon-confirmed SASS clean per
     `HVX_BARRETT_SASS_GATES.md`.
   - We REUSE this primitive verbatim by `#include`-ing the file's
     header surface OR by re-declaring the inline locally — see Stage 1
     architectural note.
   - Frozen primes + mu match line **24-29** (same constants as
     NTT.0 — no divergence risk).

4. **`reference-hexagon-v69-32x32-widening-idiom`** memory at
   `C:\Users\Knack\AppData\Roaming\Claude\local-agent-mode-sessions\
   3a0d5a5e-003b-461c-b139-db357a819869\26c26980-8ee7-432e-87c5-
   a2feec4b3e17\spaces\55dd71db-d563-4af9-a9ce-bc9d22ab62ff\memory\
   reference_hexagon_v69_32x32_widening_idiom.md`. Mandates the 2-op
   `Q6_W_vmpye_VwVuh + Q6_W_vmpyoacc_WVwVh` widening for any
   u32×u32→u64 inside HVX kernels. The Barrett primitive (#3) already
   uses this pattern — we get it transitively by calling the primitive
   inside each butterfly.

5. **Halide HVX Int(64) limitation** — confirmed by the absence of any
   Halide-emitted NTT in the codebase. The K.beta.2.5b/c HVX paths are
   all hand-rolled intrinsics; NTT.1 follows the same convention.
   (Memory entry not separately filed — the constraint manifests as
   the absence of any `.generator.cpp` for NTT-shaped kernels and the
   K.beta.2.5b plan §"Why hand-rolled intrinsics" rationale.)

6. **`reference-v69-hvx-expert-practices`** at
   `C:\Users\Knack\AppData\Roaming\Claude\local-agent-mode-sessions\
   3a0d5a5e-003b-461c-b139-db357a819869\26c26980-8ee7-432e-87c5-
   a2feec4b3e17\spaces\55dd71db-d563-4af9-a9ce-bc9d22ab62ff\memory\
   reference_v69_hvx_expert_practices.md`.
   - Vector context attachment: V69 SSR:XA={4,5} maps threads to
     vector contexts 0,1. We're single-thread for NTT.1 (NTT.3 owns
     dual-thread dispatch). FastRPC Unsigned PD attaches one of these
     contexts on entry.
   - .tmp/.cur load semantics — not used for NTT.1 v1 (the butterfly
     loads via vmem on HVX_Vector pointers; the compiler picks .cur
     when VRF writeback is needed and that's the only path we exercise).
   - 128-byte alignment requirement — twiddle compaction must be
     128-B aligned for HVX_Vector vmem loads. We allocate as
     `__attribute__((aligned(128)))` arrays.

7. **`reference-ntt-frozen-primes-N-cap`** —
   `C:\Users\Knack\AppData\Roaming\Claude\local-agent-mode-sessions\
   3a0d5a5e-003b-461c-b139-db357a819869\26c26980-8ee7-432e-87c5-
   a2feec4b3e17\spaces\55dd71db-d563-4af9-a9ce-bc9d22ab62ff\memory\
   reference_ntt_frozen_primes_N_cap.md`. N capped at 512 — both
   frozen primes have 2-adic valuation 10; max-N = (q-1)/(2N) needs
   v_2(q-1) ≥ logN+1, so N ∈ {128, 256, 512} only. NTT.1 inherits
   this constraint via NTT.0's IDL boundary.

8. **`feedback-no-silent-gate-revisions`** + **`feedback-bundled-changeset-root-cause-ambiguity`** —
   discipline rules. (a) If any gate fails, surface UPSTREAM-REQUIRED
   first; do NOT silently retreat to a smaller scope. (b) One variable
   per commit when iteration cost permits; bundle ONLY when iteration
   cost justifies. This sprint's 4-stage commit cadence is the discipline.

9. **K.beta.2.5b SASS audit pattern** —
   `tools/sp_compute_skel/docs/HVX_BARRETT_SASS_GATES.md` (read in
   full). Per-intrinsic table: row = (source intrinsic | expected SASS
   opcode | observed SASS opcode + address | PASS?). Audit summary
   counts inner-loop intrinsics + divergences. NTT.1's
   `HVX_NTT_SASS_GATES.md` matches this format.

## Algorithm structure for NTT.1

Same negacyclic radix-2 DIT as math-core / NTT.0; only the butterfly
inner loop and twiddle access path change. For one prime channel:

```
forward_one_hvx(in[N], out[N]):
    # Step 1 — pre-weight (SCALAR for v1; HVX vectorization is a
    #          single-vmul + vadd-q chain that's tiny vs the
    #          butterfly cost — defer to NTT.4/NTT.5 if measurement
    #          shows it dominates):
    for j in 0..N:
        out[j] = (in[j] mod q) * psi_pow[j] mod q   # scalar Barrett

    # Step 2 — bit-reversal (SCALAR; one-pass index manipulation, no
    #          arithmetic; HVX gather has alignment constraints making
    #          it less attractive than the trivial scalar swap):
    bit_reverse_in_place(out, N)

    # Step 3 — logN radix-2 DIT stages:
    for len in 2..=N step ×2:
        half = len / 2
        step = N / len
        if half >= 32:
            butterfly_stage_hvx_large(out, len, half, step, &w_fwd)
        else:
            butterfly_stage_scalar(out, len, half, step, &w_fwd)
```

For N=512 the large-stage path covers logN=9 stages where half ∈ {32,
64, 128, 256} — i.e. stages 6-9. For N=256 it covers stages 6-8. For
N=128 it covers stages 6-7. Small stages (half < 32) use the NTT.0
scalar implementation verbatim — option (i) per the prompt's
recommendation; option (ii) cross-group HVX shuffles deferred to
NTT.4 or NTT.5.

### Large-stage HVX butterfly (half ≥ 32)

For each group at offset `i` (i.e. `i ∈ [0, N) step len`):

```
for vk in 0..half step 32:                 # process 32 butterflies / iter
    HVX_Vector u_vec = vmem(&out[i + vk])
    HVX_Vector v_raw = vmem(&out[i + vk + half])
    HVX_Vector w_vec = vmem(&w_compact[vk]) # stride-1 in compacted twiddle
    HVX_Vector v_red = sp_barrett_reduce32_hvx_lane(v_raw, w_vec,
                                                   vq, vq_m1, vmu)
    # u + v mod q   (sum ∈ [0, 2q-1) → conditional sub)
    HVX_Vector sum    = vadd(u_vec, v_red)
    HVX_VectorPred gt = vcmp.gt.uw(sum, vq_m1)
    HVX_Vector u_out  = vmux(gt, vsub(sum, vq), sum)
    vmem(&out[i + vk])           = u_out

    # u - v mod q   (diff ∈ [-q+1, q-1] → conditional add)
    HVX_Vector diff   = vsub(u_vec, v_red)
    # signed-cmp: diff < 0 iff diff > (q-1) when interpreted as uw and
    # diff >= q (which never happens here) — actually since u, v ∈ [0,q)
    # and they're both u32, signed underflow shows up as a huge uw.
    # Equivalent rule: if u < v   → diff = u - v + q  (one conditional add).
    HVX_VectorPred lt = vcmp.gt.uw(v_red, u_vec)   # iff u < v lane-wise
    HVX_Vector u_lo   = vmux(lt, vadd(diff, vq), diff)
    vmem(&out[i + vk + half])    = u_lo
```

The conditional add for the modular sub path is the standard
math-core pattern (ntt_crt.c:68-70 `modsub`): `return (a >= b) ? a - b
: a + q - b`. Equivalent unsigned form: `diff = u - v` (wrap-around);
correct iff u >= v; otherwise we add q. We use the `lt` predicate to
mux between the wrapping diff and (diff + q).

**Intrinsic count per 32-lane butterfly (steady state):**

| Source op | Intrinsic | Count |
|---|---|---|
| vmem load u | `*((HVX_Vector*)p)` | 1 |
| vmem load v_raw | `*((HVX_Vector*)p)` | 1 |
| vmem load w | `*((HVX_Vector*)p)` | 1 |
| Barrett primitive (inlined) | 19 ops (per K.beta.2.5b audit) | 19 |
| sum = u + v_red | `Q6_Vw_vadd_VwVw` | 1 |
| gt_sum = vcmp.gt.uw(sum, vq_m1) | `Q6_Q_vcmp_gt_VuwVuw` | 1 |
| sum - vq | `Q6_Vw_vsub_VwVw` | 1 |
| u_out = vmux | `Q6_V_vmux_QVV` | 1 |
| diff = u - v_red | `Q6_Vw_vsub_VwVw` | 1 |
| lt = vcmp.gt.uw(v_red, u) | `Q6_Q_vcmp_gt_VuwVuw` | 1 |
| diff + vq | `Q6_Vw_vadd_VwVw` | 1 |
| u_lo = vmux | `Q6_V_vmux_QVV` | 1 |
| vmem store u_out | `*((HVX_Vector*)p)` | 1 |
| vmem store u_lo | `*((HVX_Vector*)p)` | 1 |
| **Total intrinsics per 32 butterflies** | | **31** |

Per-lane: 31/32 ≈ 1 intrinsic. Scalar reference: ~12 ops / butterfly
× 32 lanes = ~384 ops. Theoretical density ~12× over scalar (real
speedup measured at gate T_NTT1_WALL_CLOCK_WIN).

3 splats (vq, vq_m1, vmu) hoisted out of all loops — counted once per
call. Total expected intrinsics in the inner loop: **31**; total
hoisted: **3**.

### Small-stage scalar fallback (half < 32)

When half ∈ {1, 2, 4, 8, 16} — the inner k loop is too narrow to
saturate one HVX vector. Falls back to the byte-identical scalar
butterfly from NTT.0 (`ntt_forward_one_scalar`'s Step 3 loop body).
Defensible per `feedback-no-silent-gate-revisions`: this is an
explicit, documented architectural choice (option (i) in the prompt),
not a silent retreat from a spec'd HVX gate. NTT.4 or NTT.5 lifts to
cross-group HVX vectorization if measurement shows small-stages
dominate wall-clock.

### Per-stage twiddle compaction (temporary)

The full `w_fwd[N/2]` array stores `omega^j` for j ∈ [0, N/2). Stage
`len` accesses indices `0, step, 2*step, ..., (half-1)*step` —
stride-`step` reads, not stride-1. For HVX vmem loads, we need
stride-1 access.

**NTT.1 v1 strategy:** for each large stage, compact the twiddles
into a per-stage scratch array `w_compact[half]` with `w_compact[k] =
w_fwd[k*step]`. The compaction is a single scalar pass over `half`
elements (32, 64, 128, or 256 — small). Done once per stage; reused
across all `(N/len)` groups within that stage.

```c
uint32_t w_compact[256] __attribute__((aligned(128)));
for (uint32_t k = 0; k < half; k++) w_compact[k] = w_fwd[k * step];
```

This is a **temporary**: NTT.2 will lift to VTCM-resident precomputed
per-stage compacted tables (i.e. one allocation per N x prime, all
stages laid out back-to-back). NTT.1 confines the compaction to
stack scratch + the inner kernel; NTT.2 swaps the source pointer
without touching the kernel.

## Scope

1. **New file** —
   `tools/sp_compute_skel/src_dsp/sp_compute_ntt_hvx_imp.c`:
   - `sp_modadd_hvx_lane(u, v, vq, vq_m1)` — modular add, 4 intrinsics
   - `sp_modsub_hvx_lane(u, v, vq)` — modular sub, 4 intrinsics
   - `sp_ntt_butterfly_stage_hvx(out, N, len, half, step, w_compact,
     vq, vq_m1, vmu)` — large-stage butterfly with the 31-intrinsic
     inner loop
   - `sp_compute_ntt_hvx_oracle` IDL handler — same primIn layout as
     ntt_oracle (4 i32 primIn + data_in + data_out)
   - REUSES `sp_barrett_reduce32_hvx_lane` via `#include` of the
     primitive's TU OR via local declaration mirroring 2.5b.

2. **IDL** — add `ntt_hvx_oracle` (method 13) to
   `tools/sp_compute_skel/inc/sp_compute.idl`. Same primIn shape as
   method 12.

3. **CMakeLists** — add `sp_compute_ntt_hvx_imp` to `srcs`.

4. **Smoke harness** —
   `tools/sp_dsp_smoke/src/sp_ntt_1_smoke.rs`:
   - Re-runs the NTT.0 oracle path (method 12) for the
     T_NTT1_NO_REGRESSION gate.
   - Runs method 13 across the same 6 × 100 = 600 combinations and
     diffs element-wise against method 12 (T_NTT1_HVX_BIT_EXACT).
   - Times method 12 vs method 13 at all 3 N x 2 primes × 100 iters
     (T_NTT1_WALL_CLOCK_WIN).
   - Reuses the math-core FFI from NTT.0 as a tertiary check
     (HVX out vs math-core ntt_forward per-prime — must also match,
     proving HVX is correct against the canonical reference too).
   - `Cargo.toml` entry added.

5. **SASS audit** —
   `tools/sp_compute_skel/docs/HVX_NTT_SASS_GATES.md`. Per-intrinsic
   table matching K.beta.2.5b's format; disassemble via
   `hexagon-llvm-objdump.exe -d --mattr=+hvx,+hvxv69,+hvx-length128b
   libsp_compute_skel.so > docs/sp_compute_ntt_hvx_oracle.sass` then
   walk the steady-state body of `sp_ntt_butterfly_stage_hvx` (or its
   inlined site within `sp_compute_ntt_hvx_oracle`).

## SASS-expected opcode table (preview)

For the steady-state inner loop of `sp_ntt_butterfly_stage_hvx`,
expected intrinsics + opcodes (19 from Barrett primitive + 12 from
mod-add/sub framing = 31 total):

| # | Source intrinsic | Expected SASS opcode |
|---|---|---|
| 1-2 | vmem loads of u, v_raw | `Vd=vmem(Rt+#imm)` |
| 3 | vmem load of w | `Vd=vmem(Rt+#imm)` |
| 4-22 | (Barrett primitive 19 ops; see HVX_BARRETT_SASS_GATES.md) | (per K.beta.2.5b) |
| 23 | `Q6_Vw_vadd_VwVw(u, v_red)` | `Vd.w=vadd(Vu.w,Vv.w)` |
| 24 | `Q6_Q_vcmp_gt_VuwVuw(sum, vq_m1)` | `Qd=vcmp.gt(Vu.uw,Vv.uw)` |
| 25 | `Q6_Vw_vsub_VwVw(sum, vq)` | `Vd.w=vsub(Vu.w,Vv.w)` |
| 26 | `Q6_V_vmux_QVV(gt, sub_result, sum)` | `Vd=vmux(Qt,Vu,Vv)` |
| 27 | `Q6_Vw_vsub_VwVw(u, v_red)` | `Vd.w=vsub(Vu.w,Vv.w)` |
| 28 | `Q6_Q_vcmp_gt_VuwVuw(v_red, u)` | `Qd=vcmp.gt(Vu.uw,Vv.uw)` |
| 29 | `Q6_Vw_vadd_VwVw(diff, vq)` | `Vd.w=vadd(Vu.w,Vv.w)` |
| 30 | `Q6_V_vmux_QVV(lt, add_result, diff)` | `Vd=vmux(Qt,Vu,Vv)` |
| 31 | vmem store u_out, u_lo | `vmem(Rt+#imm)=Vs` |
| (hoisted) | 3 × `Q6_V_vsplat_R` for vq/vq_m1/vmu | `Vd=vsplat(Rt)` |

## Gates

- **T_NTT1_HVX_BIT_EXACT** — method 13 byte-exact vs method 12 (and
  vs math-core ntt_forward per-prime). 6 combinations × 100 random
  seeds = 600 runs. **PASS iff `divergence_count == 0`.**
- **T_NTT1_SASS_AUDIT** — every emitted HVX intrinsic in the steady-
  state body of `sp_ntt_butterfly_stage_hvx` produces the planned V69
  opcode. **PASS iff zero divergences.**
- **T_NTT1_WALL_CLOCK_WIN** — HVX path runs faster than scalar at
  N=512 (largest shape with most large-stage benefit). Measured per
  prime; report all 3 N × 2 primes matrix. Per
  `feedback-shape-dependent-parallelism-gates`, no precommitted
  threshold; **PASS iff HVX wall < scalar wall at N=512.**
- **T_NTT1_NO_REGRESSION** — method 12 (ntt_oracle, NTT.0) re-runs
  600/600 PASS unchanged. **PASS iff
  `divergence_count == 0 AND total_runs == 600`.**

## Stages + commits

1. **Stage 1 — HVX butterfly + IDL declaration.** Implement
   `sp_ntt_butterfly_stage_hvx` + `sp_modadd_hvx_lane` +
   `sp_modsub_hvx_lane` + per-stage twiddle compaction. Write IDL
   declaration. Skel handler stub returns 0 with placeholder output.
   Verify against scalar reference via host-side mock if feasible;
   otherwise correctness validated at Stage 3 on-device.
   - Commit: `[NTT.1] feat: Stage 1 -- HVX butterfly + IDL ntt_hvx_oracle (method 13)`

2. **Stage 2 — IDL routing wired.** Full `sp_compute_ntt_hvx_oracle`
   skel handler — pre-weight + bit-reversal + stage dispatch loop
   (scalar < 32, HVX ≥ 32). Add to CMakeLists. Build + push
   libsp_compute_skel.so.
   - Commit: `[NTT.1] feat: Stage 2 -- ntt_hvx_oracle handler + CMakeLists wiring`

3. **Stage 3 — ARM smoke + T_NTT1_HVX_BIT_EXACT +
   T_NTT1_WALL_CLOCK_WIN + T_NTT1_NO_REGRESSION.** Write
   `sp_ntt_1_smoke.rs`. Verify `adb devices`. Build for
   aarch64-android. Push + run on S22U. Capture verbatim output.
   - Commit: `[NTT.1] test: Stage 3 -- on-device smoke (600/600 HVX bit-exact, wall-clock, NTT.0 no-regression)`

4. **Stage 4 — SASS audit + closure.** Disassemble
   libsp_compute_skel.so; walk the `sp_compute_ntt_hvx_oracle` /
   `sp_ntt_butterfly_stage_hvx` body; complete the per-intrinsic
   table; flag any divergence. Write closure document. Push branch.
   - Commit: `[NTT.1] doc: Stage 4 -- closure + HVX NTT SASS audit (T_NTT1_SASS_AUDIT)`

## Anti-contamination commitments

- Sole worktree: `D:\F\shannon-prime-repos\engine-ntt-1`.
- DO NOT touch math-core sources (`lib/shannon-prime-system/...`) —
  read-only reference. Submodule not even checked out in this
  worktree; that's fine for our scope.
- DO NOT touch NTT.2's lane:
  - Anticipated NTT.2 file `sp_compute_ntt_twiddle.c` — never
    created or referenced in NTT.1.
  - Per-stage compaction stays INLINE inside the NTT.1 kernel; no
    separate twiddle TU.
- DO NOT modify `sp_compute_ntt_imp.c` — NTT.0's lane, frozen scalar
  reference.
- IDL + CMakeLists + Cargo.toml prefix-comment for any addition
  per coordination discipline: `// §4-NTT Sprint NTT.1 — HVX
  butterfly` or `# §4-NTT Sprint NTT.1 — HVX butterfly`.

## Hardware confirmation

Per `reference-mode-d-bridge-architecture` Knack's S22 Ultra with
FastRPC Path B Unsigned PD. Confirm `adb devices` reports the device
before Stage 3 push/run. Skel pushed via
`adb push hexagon_Release_toolv87_v69/ship/libsp_compute_skel.so
/data/local/tmp/` (already wired in `build.cmd`).

## What's NOT done (deferred)

- **Small-stage HVX (option ii)** — cross-group vectorization with
  HVX shuffles. Deferred to NTT.4 or NTT.5 follow-on.
- **VTCM-resident precomputed twiddle tables** — NTT.2 lane.
- **Dual-prime concurrent dispatch** — NTT.3 lane.
- **INTT + Garner CRT** — NTT.4 lane.
- **Tiled long-context attention** — NTT.5/NTT.6 lane.
- **Pre-weight + bit-reversal HVX** — both are small relative to the
  9-stage butterfly cost at N=512. If wall-clock measurement shows
  either dominates, file as NTT.4 follow-on.

## What unblocks

NTT.3 (dual-prime concurrent dispatch) once **both** NTT.1 and NTT.2
close: NTT.3 wraps two `Arc<FastRpcSession>` threads each calling
`ntt_hvx_oracle` with different q_idx. NTT.4 (INTT) reuses the same
butterfly kernel with `w_inv[]` swapped in for `w_fwd[]`; the kernel
is parameterized on the twiddle pointer.
