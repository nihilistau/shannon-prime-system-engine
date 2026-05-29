# HVX Barrett SASS audit (Sprint K v0.beta Stage 2.5b)

Per Stage 4 of `PLAN-K-beta-2-5b.md`. Disassembled `libsp_compute_skel.so` via:

```
hexagon-llvm-objdump.exe -d --mattr=+hvx,+hvxv69,+hvx-length128b libsp_compute_skel.so
```

Function `sp_compute_barrett_oracle` at `0x6510`. `sp_barrett_reduce32_hvx_lane` and
`sp_barrett_vec_run` were INLINED into it by the compiler (expected — both marked `static inline`
+ single call-site).

## Per-intrinsic gates

Captured SASS lines from `tools/sp_compute_skel/docs/sp_compute_skel.sass` inside the
`sp_compute_barrett_oracle` body. Format: line addr in skel + parsed VLIW slot text.

| # | Intrinsic (source) | Expected SASS opcode | Observed SASS opcode | PASS? |
|---|---|---|---|---|
| 1 | `Q6_W_vmpye_VwVuh(va, vb)` | `Vdd.w=vmpye(Vu.w,Vv.uh)` | `v7:6 = vmpye(v3.w,v4.uh)` @ 0x6818 | ✓ |
| 2 | `Q6_W_vmpyoacc_WVwVh(x_pair, va, vb)` | `Vxx+=vmpyo(Vu.w,Vv.h)` | `v7:6 += vmpyo(v3.w,v4.h)` @ 0x6820 | ✓ |
| 3 | `Q6_Vuw_vlsr_VuwR(x_lo, 29)` | `Vd.uw=vlsr(Vu.uw,Rt)` | `v5.uw = vlsr(v6.uw,r9)` @ 0x6824 | ✓ |
| 4 | `Q6_Vw_vasl_VwR(x_hi, 3)` | `Vd.w=vasl(Vu.w,Rt)` | `v7.w = vasl(v7.w,r12)` @ 0x6828 | ✓ |
| 5 | `Q6_V_vor_VV(sh_lo, sh_hi)` | `Vd=vor(Vu,Vv)` | `v8 = vor(v5,v7)` @ 0x682c | ✓ |
| 6 | `Q6_W_vmpye_VwVuh(sh, vmu)` (qhat step 1) | `Vdd.w=vmpye(Vu.w,Vv.uh)` | `v5:4 = vmpye(v8.w,v1.uh)` @ 0x6830 | ✓ |
| 7 | `Q6_W_vmpyoacc_WVwVh(q_pair, sh, vmu)` | `Vxx+=vmpyo(Vu.w,Vv.h)` | `v5:4 += vmpyo(v8.w,v1.h)` @ 0x6834 | ✓ |
| 8 | `Q6_Vuw_vlsr_VuwR(q_lo, 31)` | `Vd.uw=vlsr(Vu.uw,Rt)` | `v9.uw = vlsr(v4.uw,r5)` @ 0x6838 | ✓ |
| 9 | `Q6_Vw_vasl_VwR(q_hi, 1)` | `Vd.w=vasl(Vu.w,Rt)` | `v10.w = vasl(v5.w,r13)` @ 0x683c | ✓ |
| 10 | `Q6_V_vor_VV(qhat_lo, qhat_hi)` | `Vd=vor(Vu,Vv)` | `v11 = vor(v9,v10)` @ 0x6840 | ✓ |
| 11 | `Q6_W_vmpye_VwVuh(qhat, vq)` | `Vdd.w=vmpye(Vu.w,Vv.uh)` | `v9:8 = vmpye(v11.w,v0.uh)` @ 0x6844 | ✓ |
| 12 | `Q6_W_vmpyoacc_WVwVh(qq_pair, qhat, vq)` | `Vxx+=vmpyo(Vu.w,Vv.h)` | `v9:8 += vmpyo(v11.w,v0.h)` @ 0x6848 | ✓ |
| 13 | `Q6_Vw_vsub_VwVw(x_lo, qq_lo)` | `Vd.w=vsub(Vu.w,Vv.w)` | `v6.w = vsub(v6.w,v8.w)` @ 0x684c | ✓ |
| 14 | `Q6_Q_vcmp_gt_VuwVuw(r0, vq_minus_1)` | `Qd=vcmp.gt(Vu.uw,Vv.uw)` | `q0 = vcmp.gt(v6.uw,v2.uw)` @ 0x6850 | ✓ |
| 15 | `Q6_Vw_vsub_VwVw(r0, vq)` (for vmux input) | `Vd.w=vsub(Vu.w,Vv.w)` | `v12.w = vsub(v6.w,v0.w)` @ 0x6854 | ✓ |
| 16 | `Q6_V_vmux_QVV(gt0, r0_minus_q, r0)` | `Vd=vmux(Qt,Vu,Vv)` | `v13 = vmux(q0,v12,v6)` @ 0x6858 | ✓ |
| 17 | `Q6_Q_vcmp_gt_VuwVuw(r1, vq_minus_1)` (2nd Barrett correction) | `Qd=vcmp.gt(Vu.uw,Vv.uw)` | `q1 = vcmp.gt(v13.uw,v2.uw)` @ 0x685c | ✓ |
| 18 | `Q6_Vw_vsub_VwVw(r1, vq)` (for vmux input) | `Vd.w=vsub(Vu.w,Vv.w)` | `v14.w = vsub(v13.w,v0.w)` @ 0x6860 | ✓ |
| 19 | `Q6_V_vmux_QVV(gt1, r1_minus_q, r1)` | `Vd=vmux(Qt,Vu,Vv)` | `v15 = vmux(q1,v14,v13)` @ 0x6864 | ✓ |
| 20 | `Q6_V_vsplat_R((int32_t)q)` | `Vd=vsplat(Rt)` | `v0 = vsplat(r13)` @ 0x67f8 (hoisted) | ✓ |
| 21 | `Q6_V_vsplat_R((int32_t)(q - 1u))` | `Vd=vsplat(Rt)` | `v2 = vsplat(r14)` @ 0x6818 packet | ✓ |
| 22 | `Q6_V_vsplat_R((int32_t)mu)` | `Vd=vsplat(Rt)` | `v1 = vsplat(r3)` @ 0x6804 (hoisted) | ✓ |

## Audit summary

- **Total intrinsics emitted in HVX Barrett inner loop**: 19 (3 splats hoisted out of loop).
- **Divergences from expected**: **0 (zero).**
- **Every Q6_W_vmpye/vmpyoacc pair emitted as a paired-register `Vdd`/`Vxx` instruction** — the
  V69 64-bit widening idiom (per Hexagon HVX Programmer's Reference Manual §151).
- **VLIW packet density** — visible in the disassembly that several pairs (e.g. 0x684c+0x6850
  and 0x6850+0x6854) co-execute in same packet, indicating the compiler scheduled the chain
  well. The Barrett correction `vsub + vmux + vcmp` pattern fits in single packets where
  data dependencies allow.

## Note on the SASS observation that motivates a memory entry

**The AMENDMENT plan §1 mapping table understated the V69 ISA.** The table claimed each
i32×i32→i64 widening costs ~6 HVX vector ops via `Q6_Ww_vmpy_VhVh` decomposition. The
actual SASS-confirmed cost is **2 HVX vector ops** via `Q6_W_vmpye_VwVuh + Q6_W_vmpyoacc_WVwVh`.

Memory entry candidate: **`reference-hexagon-v69-32x32-widening-idiom`** — captures the
2-op idiom + cites this SASS audit as the silicon-confirmation.

## Architectural delta

The Barrett HVX primitive emits clean, dense, V69-optimal HVX SASS — confirmed at the
opcode level. This means:

- **Manifesto Trick #1 (cDSP-internal CRT-sharded compute) is silicon-confirmed at the
  HVX-vector primitive scope** (math identity gate at Stage 2.5b — 0 divergences across
  2048 samples × 2 primes).
- The intrinsic chain (vmpye + vmpyoacc) avoids the `reference-nvcc-paired-register-bug`
  failure mode entirely because HVX_VectorPair (Vxx/Vdd) is an ISA-named explicit type,
  not an inline-asm-paired-output.
- Future Halide-upgrade follow-on (closed path for v65/v68/v69 today) could in principle
  emit this same chain via Int(64) lowering IF a Halide release adds the V69 vmpye/vmpyo
  pattern — outside this sprint's scope.
