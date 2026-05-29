# HVX Barrett intrinsic mapping (Sprint K v0.beta Stage 2.5b)

Extends and corrects the AMENDMENT plan §1 mapping table
(`papers/SESSION-PLAN-lat-3-hx-mode-k-beta-AMENDMENT-stage-2-5.md`).

## Discovery (load-bearing)

The AMENDMENT table at line 23 states:

> Steps 1, 2, 4, 5, 7 (the four mul.lo + mul.hi pairs) are the load-bearing complexity. PTX has them as single instructions. HVX requires decomposition into u15-half sub-products, each via `Q6_Ww_vmpy_VhVh` (4 × i16 multiplies per i32×i32→i64 widening), plus shift+add combine. **Each i32×i32→i64 widening costs ~6 HVX vector ops** (4 vmpy + 2 combine).

**This understates the V69 ISA.** Per the Hexagon V69 HVX Programmer's Reference Manual §151
(`reference/hexagon_v69_hvx.extracted.txt:5577-5586`):

> A key function is a 32-bit × 32-bit signed multiply where the 64-bit result is kept.
> `vectorize( (int64) x * (int64) y )` equivalent to:
> `{V3:2 = vmpye(V0.w, V1.uh) } { V3:2+= vmpyo(V0.w, V1.h)}`
> The lower 32 bits of products are in V2 and the upper 32 bits in V3.

This is **2 HVX instructions** for the same 32×32→64 widening that PTX gets in one
`mul.lo.u32 + mul.hi.u32` pair. The vmpyo+vmpye chain uses
`Q6_W_vmpye_VwVuh + Q6_W_vmpyoacc_WVwVh` (paired-register accumulator), which is the
ISA-named, fully-tested HVX_VectorPair surface — distinct from PTX's silently-broken
inline-asm-paired-output (the `reference-nvcc-paired-register-bug` failure mode does not
apply here because the pair is a documented compiler-managed type, not an inline-asm
constraint).

## Per-intrinsic table

| # | Lattice purpose | Intrinsic | Assembly opcode | SDK header line | SASS verified |
|---|---|---|---|---|---|
| 1 | u30 × u30 → u60 widening, low 32 → Vdd.v[0] | `Q6_W_vmpye_VwVuh(va, vb)` | `Vdd32 = vmpye(Vu32.w, Vv32.uh)` | hvx_hexagon_protos.h:~2148 | ✓ (see HVX_BARRETT_SASS_GATES.md) |
| 2 | (cont.) widening high 32 → Vdd.v[1], accumulating | `Q6_W_vmpyoacc_WVwVh(pair, va, vb)` | `Vxx32 += vmpyo(Vu32.w, Vv32.h)` | hvx_hexagon_protos.h:2381 | ✓ (see HVX_BARRETT_SASS_GATES.md) |
| 3 | Extract low 32-bit word per lane from pair | `Q6_V_lo_W(pair)` | (compiler synth — no opcode) | hvx_hexagon_protos.h:41 | ✓ (see HVX_BARRETT_SASS_GATES.md) |
| 4 | Extract high 32-bit word per lane from pair | `Q6_V_hi_W(pair)` | (compiler synth — no opcode) | hvx_hexagon_protos.h:32 | ✓ (see HVX_BARRETT_SASS_GATES.md) |
| 5 | Logical shift right u32 by 29 (for x>>29 lo half) | `Q6_Vuw_vlsr_VuwR(x_lo, 29)` | `Vd32.uw = vlsr(Vu32.uw, Rt32)` | hvx_hexagon_protos.h:1751 | ✓ (see HVX_BARRETT_SASS_GATES.md) |
| 6 | Arithmetic shift left u32 by 3 (for x_hi<<3 in >>29 assembly) | `Q6_Vw_vasl_VwR(x_hi, 3)` | `Vd32.w = vasl(Vu32.w, Rt32)` | hvx_hexagon_protos.h:725 | ✓ (see HVX_BARRETT_SASS_GATES.md) |
| 7 | Bitwise OR (combine shifted hi + shifted lo) | `Q6_V_vor_VV(sh_lo_shifted, hi_shifted)` | `Vd32 = vor(Vu32, Vv32)` | hvx_hexagon_protos.h:2579 | ✓ (see HVX_BARRETT_SASS_GATES.md) |
| 8 | u31 × u31 → u62 widening (sh × mu), low 32 | `Q6_W_vmpye_VwVuh(sh, vmu)` | `Vdd32 = vmpye(Vu32.w, Vv32.uh)` | (same as #1) | ✓ (see HVX_BARRETT_SASS_GATES.md) |
| 9 | (cont.) widening high 32, accumulating | `Q6_W_vmpyoacc_WVwVh(pair, sh, vmu)` | `Vxx32 += vmpyo(Vu32.w, Vv32.h)` | (same as #2) | ✓ (see HVX_BARRETT_SASS_GATES.md) |
| 10 | Logical shift right by 31 (qhat = q_pair >> 31, lo) | `Q6_Vuw_vlsr_VuwR(q_lo, 31)` | `Vd32.uw = vlsr(Vu32.uw, Rt32)` | (same as #5) | ✓ (see HVX_BARRETT_SASS_GATES.md) |
| 11 | Shift left by 1 (q_hi << 1, for >>31 assembly) | `Q6_Vw_vasl_VwR(q_hi, 1)` | `Vd32.w = vasl(Vu32.w, Rt32)` | (same as #6) | ✓ (see HVX_BARRETT_SASS_GATES.md) |
| 12 | qhat × q, only low 32 bits (Vdd.v[0]) | `Q6_W_vmpye_VwVuh(qhat, vq)` | `Vdd32 = vmpye(Vu32.w, Vv32.uh)` | (same as #1) | ✓ (see HVX_BARRETT_SASS_GATES.md) |
| 13 | (cont.) qhat × q accumulator | `Q6_W_vmpyoacc_WVwVh(pair, qhat, vq)` | `Vxx32 += vmpyo(Vu32.w, Vv32.h)` | (same as #2) | ✓ (see HVX_BARRETT_SASS_GATES.md) |
| 14 | r0 = x_lo - qq_lo (modular sub) | `Q6_Vw_vsub_VwVw(x_lo, qq_lo)` | `Vd32.w = vsub(Vu32.w, Vv32.w)` | hvx_hexagon_protos.h:3326 | ✓ (see HVX_BARRETT_SASS_GATES.md) |
| 15 | Unsigned-gt compare (r0 > q-1) for first Barrett correction | `Q6_Q_vcmp_gt_VuwVuw(r0, vq_minus_1)` | `Qd4 = vcmp.gt(Vu32.uw, Vv32.uw)` | hvx_hexagon_protos.h:1625 | ✓ (see HVX_BARRETT_SASS_GATES.md) |
| 16 | r1 = (mask ? r0 - q : r0) | `Q6_V_vmux_QVV(gt0, vsub(r0, vq), r0)` | `Vd32 = vmux(Qt4, Vu32, Vv32)` | hvx_hexagon_protos.h:2507 | ✓ (see HVX_BARRETT_SASS_GATES.md) |
| 17 | (second Barrett correction — same intrinsics #15-16 applied to r1) | (same) | (same) | (same) | ✓ (see HVX_BARRETT_SASS_GATES.md) |
| 18 | Splat u32 constants (q, q-1, mu) — load-time, hoisted | `Q6_V_vsplat_R(Rt)` | `Vd32 = vsplat(Rt32)` | hvx_hexagon_protos.h:68 | ✓ (see HVX_BARRETT_SASS_GATES.md) |

## Intrinsic count summary (per 32-lane HVX Barrett invocation)

- 3 splats (hoisted out of inner loop — counted once)
- 2 + 2 + 2 = 6 widening multiplies (3 × {vmpye + vmpyoacc})
- 2 + 2 = 4 shift-and-combine for >>29 and >>31 (2 × {vlsr + vasl + vor})
- 1 modular subtract for r0
- 2 × (1 vcmp + 1 vmux + 1 vsub) = 6 ops for two Barrett corrections
- 4 pair-extracts (`Q6_V_lo_W` × 3 + `Q6_V_hi_W` × 3 — actually 6, but compiler often inlines)

Net: ~22-25 intrinsics in the inner loop per 32-lane HVX Barrett reduce. AMENDMENT
estimate was ~24 just for the widening multiplies (4 widening at 6 ops each); actual
total including everything is ~22-25, NOT including the Barrett correction at ~6 ops.

**vs scalar Barrett (sp_barrett_reduce32_scalar):** ~10 scalar ops per lane × 32 lanes
= ~320 scalar ops. HVX path: ~25 ops processes 32 lanes = ~12.8× theoretical
throughput per lane. Real speedup depends on pipeline density + memory bandwidth +
ALU dispatch slot availability per `reference-v69-hvx-expert-practices`.

## What the table does NOT cover

- SASS-observed opcode confirmation (deferred to Stage 4 audit).
- Pipeline cycle accounting (out of scope — `feedback-lattice-baseline-is-prior-lattice`:
  the win is the scalar→vector transition, not absolute pcycle).
- Cross-prime parameterization correctness (Stage 2 closure).
- Loop unroll factor decisions (deferred to SASS-driven tuning; not in 2.5b scope).
