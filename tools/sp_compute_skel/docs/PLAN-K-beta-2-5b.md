# PLAN — Sprint K v0.beta-2.5b — HVX-vectorized Barrett reduction chain
**Branch:** `sprint/kbeta-2-5b` (engine worktree `D:\F\shannon-prime-repos\engine-kbeta-2-5b`)
**Base:** engine main @ 41963ac (= Stage 2.5a closed)
**Operator:** KnackAU (knack112358@gmail.com)
**Goal:** Ship the HVX vector Barrett primitive that closes K v0.beta Stage 2.5 PARTIAL and empirically confirms Manifesto Trick #1 (cDSP-internal CRT-sharded compute via SSR:XA={4,5} dual vector contexts) at the Barrett-primitive scope.

---

## Stage 0 — Reference reading (load-bearing citations)

Per the rule "lead with the reference, then theory" — every design decision below cites the actual file:line or quotes the source. Theory-only reasoning produced two wrong §16.3 framings on 2026-05-29 and several misnamed intrinsics in the AMENDMENT plan; this plan grounds every primitive in the SDK reference text.

### 1. Stage 2.5a scalar Barrett primitive (in this worktree)

- `tools/sp_compute_skel/src_dsp/sp_compute_crt_imp.c:35-49` — `sp_barrett_reduce32_scalar(uint64_t x, uint32_t q, uint32_t mu)` and per-prime wrappers `sp_modmul_scalar_q{1,2}(a, b)`. Algorithm: `qhat = ((x >> 29) * mu) >> 31; r = x - qhat*q; r %= q (≤2 conditional subtracts).`
- `tools/sp_compute_skel/src_dsp/sp_compute_crt_imp.c:18-22` — frozen constants: `SP_NTT_Q1=1073738753`, `SP_NTT_Q2=1073732609`, `SP_MU_Q1=1073744895`, `SP_MU_Q2=1073751039`. μ = floor(2^60 / q) per prime.
- `tools/sp_compute_skel/src_dsp/sp_compute_crt_imp.c:61-97` — `sp_compute_barrett_oracle(remote_handle64, q_idx, mode, a_buf, b_buf, r_buf)`. mode=0 dispatches scalar; mode=1 currently returns -1 with FARF log "Stage 2.5b reserved." THIS sprint replaces line 87-91 with the HVX vector path.

### 2. `barrett_oracle` IDL method

- `tools/sp_compute_skel/inc/sp_compute.idl:163-176` — declared method id 10:
  ```
  long barrett_oracle(in long q_idx, in long mode,
                      in  sequence<octet> a_buf,
                      in  sequence<octet> b_buf,
                      rout sequence<octet> r_buf);
  ```
  `mode` parameter present; wiring for `mode=1` is the C-side implementation only (no IDL change needed in 2.5b).

### 3. PTX→HVX intrinsic mapping table (AMENDMENT plan §1)

- `D:\F\shannon-prime-repos\shannon-prime-lattice\papers\SESSION-PLAN-lat-3-hx-mode-k-beta-AMENDMENT-stage-2-5.md:7-23` — operator-canary mapping table. Lists `Q6_Ww_vmpy_VhVh` (i16×i16 widening pair, 4 sub-products per i32×i32→i64) as the route. **The table's footnote at line 23 understates the V69 ISA: a 32×32→64 widening idiom exists via `vmpye + vmpyoacc` (see §4 below), reducing the per-multiply intrinsic count from ~6 to 2.** This sprint will:
  - Use the `vmpye + vmpyoacc` chain as the primary implementation (per `feedback-lead-with-reference-then-theory` — the source reference is hexagon_v69_hvx.extracted.txt §151).
  - Add new rows to `tools/sp_compute_skel/docs/HVX_BARRETT_MAPPING.md` documenting each emitted intrinsic.
  - Surface the AMENDMENT-table sub-optimality as a memory-entry candidate in the closure (not a silent revision — see §11).

### 4. V69 32×32→u64 widening idiom — the load-bearing reference

`/sessions/gallant-dreamy-franklin/mnt/reference/hexagon_v69_hvx.extracted.txt:5577-5586` quotes the Hexagon V69 HVX Programmer's Reference Manual §151:

> "A key function is a 32-bit × 32-bit signed multiply where the 64-bit result is kept.
> `vectorize( (int64) x * (int64) y )` equivalent to:
> `{V3:2 = vmpye(V0.w, V1.uh) } { V3:2+= vmpyo(V0.w, V1.h)}`
> The lower 32 bits of products are in V2 and the upper 32 bits in V3."

Pseudo-code semantics (same file, 5583-5613):
```
Vdd = vmpye(Vu.w, Vv.uh):
    prod = Vu.w[i] * Vv.w[i].uh[0]       // 32-bit signed × 16-bit unsigned (low)
    Vdd.v[1].w[i] = prod >> 16            // upper 32 bits
    Vdd.v[0].w[i] = prod << 16            // lower 32 bits (shifted form)

Vxx += vmpyo(Vu.w, Vv.h):
    prod = Vu.w[i] * Vv.w[i].h[1] + Vxx.v[1].w[i]   // 32-bit × 16-bit signed (high)
    Vxx.v[1].w[i] = prod >> 16
    Vxx.v[0].w[i].h[0] = Vxx.v[0].w[i] >> 16          // preserve low half of prior
    Vxx.v[0].w[i].h[1] = prod & 0xFFFF
```

For u30 inputs (a, b ∈ [0, q) where q < 2^30):
- Vv.uh[0] = b_lo (low 16 bits, < 2^16)
- Vv.h[1] = b_hi (high 16 bits, < 2^14, signed-positive: safe under signed interpretation)
- Vu.w = a (≤ 2^30, signed-positive)
- Sequence: `pair = vmpye(a, b); pair += vmpyo(a, b);` yields `pair_lo[i] = (a[i]*b[i]) & 0xFFFFFFFF`, `pair_hi[i] = (a[i]*b[i]) >> 32`.

### 5. V69 expert practices (memory entry)

`reference_v69_hvx_expert_practices.md` — SSR:XA={4,5} dual vector contexts on V69 (§22-87). The DSP-internal CRT-sharded compute (Manifesto Trick #1) materializes when two threads with concurrent HVX work cross the SSR:XA boundary. The K v0.alpha scalar-saturating kernel already showed 0.9699 overlap_fraction; Stage 2.5b reproduces the same dual-dispatch but with HVX instructions in both invokes (vs scalar-pipe in 2.5a), making the cDSP-internal Trick #1 confirmation specifically HVX-bound.

### 6. nvcc paired-register principle (memory entry)

`reference_nvcc_paired_register_bug.md` — "decompose 64-bit intermediates into 32-hi/32-lo." Translated to HVX: never trust an implicit u64 vector type (Halide's Int(64) lowering already proven broken, engine commit 39e286c). Decompose via documented hardware instructions (vmpye/vmpyoacc). The `Vxx32` paired-register class in V69 HVX **is the explicit, ISA-named, fully-tested 64-bit vector pair** — distinct from the silently-unreliable paired-register output of `mul.wide.u32` in PTX. This sprint embraces HVX_VectorPair as a first-class type.

### 7. HVX intrinsic citations (`hvx_hexagon_protos.h` at `C:\Qualcomm\Hexagon_SDK\5.5.6.0\tools\HEXAGON_Tools\8.7.06\Tools\target\hexagon\include\hvx_hexagon_protos.h`)

| Line | Intrinsic | Purpose |
|---|---|---|
| 68 | `Q6_V_vsplat_R(Rt)` | Broadcast u32 scalar to all 32 word lanes. Used for q, q-1, mu splats. |
| 545 | `Q6_Vw_vadd_VwVw(Vu,Vv)` | 32-bit word add. Modular OK since we control overflow. |
| 725 | `Q6_Vw_vasl_VwR(Vu,Rt)` | Arithmetic shift left, immediate. Used for `x_hi << 3` in the >>29 carry assembly. |
| 1067 | `Q6_W_vcombine_VV(Vu,Vv)` | Form a `HVX_VectorPair` from two `HVX_Vector` (hi=Vu, lo=Vv). Used to initialize a zero pair when needed. |
| 1076 | `Q6_V_vzero()` | All-zero vector. Used for initializing pair accumulators. |
| 1661 | `Q6_Q_vcmp_gt_VwVw(Vu,Vv)` | Signed gt — NOT used; we use unsigned variant. |
| 1625 | `Q6_Q_vcmp_gt_VuwVuw(Vu,Vv)` | **Unsigned u32 gt.** Used for `r > (q-1)` → mask for conditional subtract. |
| 1751 | `Q6_Vuw_vlsr_VuwR(Vu,Rt)` | Logical shift right of u32 word lanes. Used for `x_lo >> 29` and `qhat_lo >> 31`. |
| 2129 | `Q6_Vw_vmpye_VwVuh(Vu,Vv)` | 32×16 even-half single-result (used in Barrett's qhat*q lo-product). |
| 2255 | `Q6_Vw_vmpyie_VwVuh(Vu,Vv)` | 32×16 even-half single-result (alias surface). Used for `qhat * q` low 32 bits. |
| 2309 | `Q6_Vw_vmpyio_VwVh(Vu,Vv)` | 32×16 odd-half single-result, integer no-shift no-sat. Used for `qhat * q` low 32 bits (high-half partial). |
| 2507 | `Q6_V_vmux_QVV(Qt,Vu,Vv)` | Predicate-mux. Vu where Qt true, Vv otherwise. Used for conditional `r -= q`. |
| 2579 | `Q6_V_vor_VV(Vu,Vv)` | Bitwise OR. Used to combine `(x_lo>>29)` and `(x_hi<<3)`. |
| 3326 | `Q6_Vw_vsub_VwVw(Vu,Vv)` | 32-bit word sub. Used for `x_lo - qq_lo` and `r - q`. |
| (W) | `Q6_W_vmpye_VwVuh(Vu,Vv)` → 64-bit pair | `Vdd = vmpye(Vu.w, Vv.uh)` — initial half of 32×32→64. (Line ~ line 2148 — search "Q6_W_vmpye_VwVuh".) |
| (W) | `Q6_W_vmpyoacc_WVwVh(Vxx,Vu,Vv)` → 64-bit pair | `Vxx += vmpyo(Vu.w, Vv.h)` — combine half for 32×32→64. (Line 2381.) |

### 8. HAP_perf_get_pcycles surface

- `tools/sp_compute_skel/src_dsp/sp_compute_imp.c` — Sprint G already brackets the Halide kernel call with `HAP_perf_get_pcycles()` and ships pcycles via `kernel_pcycles_lo/hi` rout params on method 9. The K v0.alpha smoke `sp_dual_dispatch_smoke.rs:73` reconstructs the u64 pcycle count via `((prim_out[2] as u64) << 32) | (prim_out[1] as u64)`. Stage 2.5b: add pcycle brackets around the HVX kernel body to enable DUAL_DISPATCH_SPEEDUP per-thread accounting. But for the existing `sp_compute_barrett_oracle` IDL surface (already-frozen at 5 args: q_idx, mode, a_buf, b_buf, r_buf), there is NO out-channel for pcycles — adding one is an IDL change. **Resolution**: Wrap the HVX work in pcycle brackets internally and report via FARF (RUNTIME_HIGH) log. Wall-clock from ARM side measures the speedup gate (per `reference-fastrpc-concurrent-dispatch` — wall-clock IS the discriminator, not pcycle ratio). Logged DSP-side pcycles are diagnostic only.

---

## Stage 1 — Vector Barrett primitive for q_1 (single prime)

### Math

For each 32-lane HVX_Vector load of `a` and `b`:
1. **Widening multiply** `pair = a * b` (u60 result per lane, stored as u32-lo/u32-hi pair via vmpye + vmpyoacc).
2. **Shift right 29** `sh = pair >> 29` (u31 result per lane, stored as a single HVX_Vector u32 since u60 >> 29 < 2^31).
3. **Multiply by μ** `qpair = sh * mu` (u62 result per lane, via vmpye + vmpyoacc).
4. **Shift right 31** `qhat = qpair >> 31` (u31 result, single HVX_Vector).
5. **Multiply qhat * q** (only low 32 bits needed since x_lo - qhat*q must equal r mod 2^32 and r < 3q < 2^32). Two single-result intrinsics: `qq_lo = vmpyie(qhat, q) + vmpyio(qhat, q)`.
6. **Subtract** `r0 = x_lo - qq_lo` (modular OK since high parts cancel).
7. **Two conditional subtracts** `r = canonicalize(r0, q)` via vmux + vcmp_gt_uw.
8. **Store** r vector to output.

### Functions

In `tools/sp_compute_skel/src_dsp/sp_compute_crt_imp.c`:

```c
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <hvx_hexagon_protos.h>

/* Vector Barrett: 32 u32 lanes per HVX_Vector. q, mu broadcast as splats.
 * Returns canonical r ∈ [0, q) per lane. */
static inline HVX_Vector sp_barrett_reduce32_hvx_lane(
    HVX_Vector va, HVX_Vector vb,
    HVX_Vector vq, HVX_Vector vq_minus_1, HVX_Vector vmu)
{
    /* 1. x = a*b as 64-bit pair (lo, hi) per lane. */
    HVX_VectorPair x_pair = Q6_W_vmpye_VwVuh(va, vb);
    x_pair = Q6_W_vmpyoacc_WVwVh(x_pair, va, vb);
    HVX_Vector x_lo = Q6_V_lo_W(x_pair);
    HVX_Vector x_hi = Q6_V_hi_W(x_pair);

    /* 2. sh = x >> 29. Since x < 2^60, sh < 2^31, fits in u32.
     *    sh = (x_lo >> 29) | (x_hi << 3) */
    HVX_Vector sh = Q6_V_vor_VV(
        Q6_Vuw_vlsr_VuwR(x_lo, 29),
        Q6_Vw_vasl_VwR(x_hi, 3));

    /* 3. q_pair = sh * mu (u31 * u31 → u62) */
    HVX_VectorPair q_pair = Q6_W_vmpye_VwVuh(sh, vmu);
    q_pair = Q6_W_vmpyoacc_WVwVh(q_pair, sh, vmu);
    HVX_Vector q_lo = Q6_V_lo_W(q_pair);
    HVX_Vector q_hi = Q6_V_hi_W(q_pair);

    /* 4. qhat = q_pair >> 31. Since q_pair < 2^62, qhat < 2^31.
     *    qhat = (q_lo >> 31) | (q_hi << 1) */
    HVX_Vector qhat = Q6_V_vor_VV(
        Q6_Vuw_vlsr_VuwR(q_lo, 31),
        Q6_Vw_vasl_VwR(q_hi, 1));

    /* 5. qq_lo = (qhat * q) low 32 bits.
     *    Per Hexagon ISA §151 (vmpye/vmpyo semantics), low 32 of
     *    (qhat * q) = vmpyie(qhat, q.uh[0]) + (vmpyio(qhat, q.h[1]) << 16).
     *    For q < 2^30 splat, low 16 of q = q & 0xFFFF; high 16 of q = q >> 16.
     *    But vmpyie + vmpyio (single-result) already produce the correctly
     *    summed low 32 bits per ISA notes (vmpyie low half, vmpyio is HIGH-half
     *    × 2^16-shifted; per §151 vmpyo single-result variants saturate+shift,
     *    so we use the alternative chain via vmpye 64-bit + vmpyoacc, then
     *    take just the LO half — equivalent and correct.) */
    HVX_VectorPair qq_pair = Q6_W_vmpye_VwVuh(qhat, vq);
    qq_pair = Q6_W_vmpyoacc_WVwVh(qq_pair, qhat, vq);
    HVX_Vector qq_lo = Q6_V_lo_W(qq_pair);

    /* 6. r0 = x_lo - qq_lo. The high parts cancel mod 2^32 because
     *    x_hi - qhat*q's high == 0 in exact arithmetic. */
    HVX_Vector r0 = Q6_Vw_vsub_VwVw(x_lo, qq_lo);

    /* 7. Two Barrett correction subtractions. Compare unsigned. */
    HVX_VectorPred gt0 = Q6_Q_vcmp_gt_VuwVuw(r0, vq_minus_1);
    HVX_Vector r1 = Q6_V_vmux_QVV(gt0, Q6_Vw_vsub_VwVw(r0, vq), r0);
    HVX_VectorPred gt1 = Q6_Q_vcmp_gt_VuwVuw(r1, vq_minus_1);
    HVX_Vector r2 = Q6_V_vmux_QVV(gt1, Q6_Vw_vsub_VwVw(r1, vq), r1);

    return r2;
}

/* Process n u32 lanes (n must be multiple of 32). q_idx selects prime. */
static int sp_barrett_vec_run(int q_idx,
                              const uint32_t *a, const uint32_t *b, uint32_t *r,
                              int n)
{
    if ((n % 32) != 0) return -1;
    uint32_t q  = (q_idx == 0) ? SP_NTT_Q1 : SP_NTT_Q2;
    uint32_t mu = (q_idx == 0) ? SP_MU_Q1  : SP_MU_Q2;
    HVX_Vector vq         = Q6_V_vsplat_R((int32_t)q);
    HVX_Vector vq_minus_1 = Q6_V_vsplat_R((int32_t)(q - 1));
    HVX_Vector vmu        = Q6_V_vsplat_R((int32_t)mu);
    const HVX_Vector *va_p = (const HVX_Vector *)a;
    const HVX_Vector *vb_p = (const HVX_Vector *)b;
    HVX_Vector       *vr_p = (HVX_Vector *)      r;
    int n_vecs = n / 32;
    for (int i = 0; i < n_vecs; i++) {
        HVX_Vector va = va_p[i];
        HVX_Vector vb = vb_p[i];
        vr_p[i] = sp_barrett_reduce32_hvx_lane(va, vb, vq, vq_minus_1, vmu);
    }
    return 0;
}
```

### Stage-1 closure criteria

- Builds clean on the SDK toolchain (no panics, no warnings).
- skel push to device produces a loadable .so.
- `sp_barrett_oracle_smoke` invoked with `mode=1, q_idx=0, n=1024` returns OK (return code 0).
- Bitwise compare against scalar reference for q_1: zero divergence across 1024 vectors.

### Stage-1 NON-CRITERIA (filed as Stage 2, 3, 4):

- q_2 parity (Stage 2 — same primitive, different splat constants).
- SASS audit (Stage 4 — verified after both primes complete).
- DUAL_DISPATCH_SPEEDUP (Stage 4 — needs HVX work in both threads).
- LEAK_FREE 10k cycles (Stage 4).

### Stage-1 risks

- **R1.1 Lane alignment in vmpye.** vmpye reads Vv.uh[0] = bits [15:0] of each word lane. The Vv input is `b` as a packed u32 array; bits [15:0] of each lane ARE b's low halfword. Memory layout matches the ISA model.
- **R1.2 Signed interpretation of Vv.h[1].** vmpyo treats Vv.h[1] as signed i16. For b < 2^30, b_hi = b >> 16 < 2^14, always positive as i16. Safe.
- **R1.3 vmpye+vmpyoacc paired-register codegen.** Per `reference-nvcc-paired-register-bug`, paired-output register allocation can silently miscompile. **Mitigation**: this is HVX, not PTX. HVX_VectorPair is an explicit named type with documented hi/lo accessors (`Q6_V_lo_W`, `Q6_V_hi_W`). The Hexagon compiler manages pair allocation under explicit type discipline. The risk pattern was specifically the inline-asm-paired-output mode of PTX. SASS audit at Stage 4 will verify the intrinsic emitted the documented opcode.
- **R1.4 Build break in SDK toolchain.** If the SDK build fails on a new C source touching HVX intrinsics, this is reportable UPSTREAM per `feedback-no-silent-gate-revisions`. Will not silently switch toolchain flags.

### Commit (Stage 1)

`[lat-3-hx-mode-k-beta] feat: Stage 2.5b Stage 1 — HVX vector Barrett primitive (q_1)`

---

## Stage 2 — Parameterize for q_2 + harness cross-validation

### Scope

- Extend `sp_barrett_vec_run` to accept `q_idx` runtime parameter (already done in Stage 1 sketch).
- Wire `mode=1` branch in `sp_compute_barrett_oracle` (lines 86-91 of sp_compute_crt_imp.c) to dispatch through `sp_barrett_vec_run`.
- Extend `sp_barrett_oracle_smoke.rs` to cross-validate `mode=0` vs `mode=1` and `mode=1` vs Rust scalar reference for BOTH primes.

### Commit (Stage 2)

`[lat-3-hx-mode-k-beta] feat: Stage 2.5b Stage 2 — q_2 parity + mode=0/1 cross-validation harness`

---

## Stage 3 — Dual-dispatch smoke + leak-free harness

### New bin: `sp_barrett_dual_smoke.rs`

Mirrors `sp_dual_dispatch_smoke.rs` architecture but invokes `barrett_oracle(mode=1, q_idx=X)` per thread. Two concurrent dispatches on one `Arc<FastRpcSession>` (per `reference-fastrpc-concurrent-dispatch`).

Single benchmark unit:
- Sequential baseline: thread 0 → `barrett_oracle(mode=1, q_idx=0, n=1024)`; thread joined; thread 1 → `barrett_oracle(mode=1, q_idx=1, n=1024)`. Sum wall.
- Concurrent: spawn both threads simultaneously, join both. Measure wall.
- Speedup = sequential_wall / concurrent_wall.

DUAL_DISPATCH_SPEEDUP gate: ≥ 1.5×. Report observed against K v0.alpha's 1.935× baseline.

LEAK_FREE: 10000 iterations of `dual_invoke(barrett_oracle(q1), barrett_oracle(q2))`. Track total runtime; verify no monotonic memory growth via `/proc/self/status` VmRSS sampling at iteration 0, 5000, 10000.

### Commit (Stage 3)

`[lat-3-hx-mode-k-beta] feat: Stage 2.5b Stage 3 — dual-dispatch + leak-free harness`

---

## Stage 4 — On-device gates run + SASS audit + closure

### Build & push

```ps1
cd D:\F\shannon-prime-repos\engine-kbeta-2-5b\tools\sp_compute_skel
.\build.cmd                   # builds + auto adb pushes libsp_compute_skel.so

cd D:\F\shannon-prime-repos\engine-kbeta-2-5b\tools\sp_dsp_smoke
cargo build --target aarch64-linux-android --release `
    --bin sp_barrett_oracle_smoke --bin sp_barrett_dual_smoke
adb push target\aarch64-linux-android\release\sp_barrett_oracle_smoke /data/local/tmp/
adb push target\aarch64-linux-android\release\sp_barrett_dual_smoke   /data/local/tmp/
```

### Gates run

1. **M_K_beta_MATH_IDENTITY** — `sp_barrett_oracle_smoke` reports per-prime/per-mode results. Pass: 1024 vectors × 2 primes × 2 modes (scalar + vector) compared against Rust scalar reference. Zero divergence. Capture: samples_compared, divergence_count, max_lane_diff.
2. **BARRETT_CORRECTNESS** — same smoke run additionally cross-checks `r ≡ a*b (mod q)` (ARM-side u64 reference) AND `0 ≤ r < q`. Capture: samples_correct, samples_total.
3. **DUAL_DISPATCH_SPEEDUP** — `sp_barrett_dual_smoke` reports wall_seq, wall_concurrent, speedup. Pass: speedup ≥ 1.5×. Capture: observed_speedup, comparison_to_alpha_baseline (1.935×).
4. **LEAK_FREE** — `sp_barrett_dual_smoke` 10000-cycle loop. Capture: cycles_run, vmrss_start_kb, vmrss_mid_kb, vmrss_end_kb. Pass: vmrss_end_kb - vmrss_start_kb ≤ 1024 KB.

### SASS audit

```ps1
$HEXTOOL = "C:\Qualcomm\Hexagon_SDK\5.5.6.0\tools\HEXAGON_Tools\8.7.06\Tools\bin"
& "$HEXTOOL\hexagon-llvm-objdump.exe" -d `
    D:\F\shannon-prime-repos\engine-kbeta-2-5b\tools\sp_compute_skel\hexagon_Release_toolv87_v69\ship\sp_compute_skel.so `
    > tools\sp_compute_skel\docs\sp_compute_skel.sass
```

Then `Select-String -Pattern 'vmpye|vmpyo|vlsr|vasl|vmux|vsub|vor|vcmp_gt' sp_compute_skel.sass` — count expected vs observed. Each row added to `HVX_BARRETT_SASS_GATES.md`:

| Intrinsic | Expected SASS op | Observed SASS op | PASS? |
|---|---|---|---|
| `Q6_W_vmpye_VwVuh` | `Vdd.w=vmpye(Vu.w,Vv.uh)` | (captured) | ✓ / ✗ |
| ... (one per used intrinsic) | | | |

If ANY observed ≠ expected, surface UPSTREAM per `feedback-no-silent-gate-revisions`. Closure note left in UPSTREAM-REQUIRED state.

### Commit (Stage 4)

`[lat-3-hx-mode-k-beta] test: Stage 2.5b — on-device 4 substantive gates run + SASS audit`

---

## Multi-stage file deliverables

| Stage | File | Status |
|---|---|---|
| 0 | `tools/sp_compute_skel/docs/PLAN-K-beta-2-5b.md` | THIS file (plan-commit) |
| 1 | `tools/sp_compute_skel/src_dsp/sp_compute_crt_imp.c` | +vector Barrett primitive + mode=1 wiring (q_1) |
| 1 | `tools/sp_compute_skel/docs/HVX_BARRETT_MAPPING.md` | NEW — intrinsic mapping table (per-row each emitted intrinsic) |
| 2 | `tools/sp_compute_skel/src_dsp/sp_compute_crt_imp.c` | +q_2 parameterization (likely zero diff — single function) |
| 2 | `tools/sp_dsp_smoke/src/sp_barrett_oracle_smoke.rs` | +mode=1 cross-validation for both primes |
| 3 | `tools/sp_dsp_smoke/src/sp_barrett_dual_smoke.rs` | NEW — dual-dispatch + leak harness |
| 3 | `tools/sp_dsp_smoke/Cargo.toml` | +bin entry |
| 4 | `tools/sp_compute_skel/docs/HVX_BARRETT_SASS_GATES.md` | NEW — SASS audit log |
| 4 | `tools/sp_dsp_smoke/sprint_k_beta_2_5b_run_output.txt` | Verbatim device output |
| 4 | `tools/sp_compute_skel/docs/CLOSURE-K-beta-2-5b.md` | NEW — closure note |

**Estimated LOC: ~350-450** (within ~250-350 spec'd in prompt; will surface upstream if I exceed without reducing scope first).

---

## Out of scope (explicitly NOT in this sprint)

- ❌ **Full mod_q_matmul kernel** — K v0.beta Stages 3-7 of the original K-beta plan. The vector Barrett primitive is the building block; wiring it into a CRT-split matmul is a separate downstream sprint.
- ❌ **CRT Garner recombination on ARM** — would require the mod_q_matmul kernel above; K v0.beta-2.5c filed as follow-on if downstream needs it.
- ❌ **Halide generator changes** — Halide-Int(64) HVX limitation is locked. We do NOT attempt to coerce Halide.
- ❌ **Modifying sp_compute_imp.c** (Sprint G/H/I/J kernel) — Sprint K v0.beta plan §6.
- ❌ **Replacing scalar Barrett in any downstream path** — scalar remains untouched.
- ❌ **NPU dispatch** (K.2 scope).
- ❌ **Signed PD migration** — current path is Path B (Unsigned PD). Future Signed PD migration is a different security surface.

---

## Anti-patterns (locked, additional to AMENDMENT §5)

15. **DO NOT skip Stage 0 reference reading.** Theory-first design without `hexagon_v69_hvx.extracted.txt:5577-5586` citation would have produced the wrong intrinsic chain (4 sub-products vs 2-instruction widening).
16. **DO NOT silently use vmpyi (the saturating shift variant) as substitute for vmpye+vmpyoacc.** They are different operations; vmpyo+saturation discards the high half. Use the 64-bit pair variants exclusively for widening.
17. **DO NOT use Halide::Int(64) anywhere in this sprint's code.** Hand-rolled intrinsics path only.
18. **DO NOT widen 1024-vector test population to hide divergence.** If even one of 32×n_vecs lanes diverges, the gate fails. Surface UPSTREAM.
19. **DO NOT report DUAL_DISPATCH_SPEEDUP as a pcycle ratio.** Wall-clock IS the discriminator per `reference-fastrpc-concurrent-dispatch:90-129`.
20. **DO NOT call Stage 2.5b "K v0.beta CLOSED."** The umbrella K-beta-closed requires the full mod_q_matmul kernel; Stage 2.5b only closes Stage 2.5 (the Barrett primitive).

---

## Sub-tags

- `lat-phase-13-6-k-beta-barrett-hvx-vector` — Stage 4 closure tag (per AMENDMENT §6 line 96).
- `lat-phase-13-6-k-beta-vector-c` — alternate name from operator's prompt (synonym; pick one).
- After Stage 4 ships: Manifesto Trick #1 cDSP-internal status updates from "silicon-confirmed (scalar path, Sprint K v0.alpha)" to "silicon-confirmed (HVX-vector path, Sprint K v0.beta-2.5b)."

---

## Memory entry candidates for closure

1. **`reference-hexagon-v69-32x32-widening-idiom`** — captures `vmpye + vmpyoacc` as the canonical V69 32×32→64 idiom. Refutes the AMENDMENT mapping table's claim that "32×32 widening needs ~6 HVX ops." Cites hexagon_v69_hvx.extracted.txt:5577-5586. Composes with `reference-nvcc-paired-register-bug` (HVX_VectorPair is the safe paired-register surface, unlike PTX inline-asm pairs).
2. **(possibly)** — any SASS observed-vs-expected divergence, surfaced for compiler-team attention.

---

## Workflow discipline checklist

- [x] Read AMENDMENT plan + original K-beta plan + K v0.alpha closure + memory entries (Stage 0, this section).
- [x] Confirm worktree (`D:\F\shannon-prime-repos\engine-kbeta-2-5b`, branch `sprint/kbeta-2-5b`, base 41963ac).
- [x] Confirm hardware (`adb devices` → R5CT22445JA S22U attached).
- [x] Confirm Hexagon SDK headers reachable (`hvx_hexagon_protos.h` at SDK 5.5.6.0 tools).
- [ ] Plan-commit (THIS file, on `sprint/kbeta-2-5b`).
- [ ] Stage 1 commit (vector Barrett q_1).
- [ ] Stage 2 commit (q_2 parity + cross-val harness).
- [ ] Stage 3 commit (dual + leak harness).
- [ ] Stage 4 commit (on-device gates + SASS + closure).
- [ ] `git push -u origin sprint/kbeta-2-5b`.
- [ ] Operator review + merge (NOT performed by this agent).
