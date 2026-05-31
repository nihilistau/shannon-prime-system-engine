//! sp-trick1 — Sprint TRICK-1 library.
//!
//! Operationalizes the manifesto's first trick (CRT-sharded compute across
//! independent silicon islands with byte-exact ARM Garner recombination) on
//! Knack's S22U. Per PLAN-TRICK-1.md §3, the NPU-as-q_2 path is upstream-blocked
//! on V69 silicon; this crate pivots to the equally architecturally-valid
//! cDSP-q1 + ARM-q2 + ARM Garner pair, where ARM Cortex-X2/A710 cores are
//! a genuinely-independent second silicon island from the Hexagon V69 cDSP.
//!
//! The mathematical building blocks (Barrett, mod-q matmul reference,
//! Garner combine, 60-bit unreduced reference) duplicate the relevant
//! functions from sibling crate `sp_dsp_smoke` — kept here rather than
//! depending on the sibling crate so this crate stays a clean architectural
//! demo without inheriting unrelated K-series module conventions.
//!
//! Cross-references back to sibling-crate equivalents:
//!   `barrett_reduce32`             == `sp_barrett_oracle::barrett_reduce32`
//!   `matmul_q_scalar_ref`          == `sp_matmul_q_ref::matmul_q_scalar_ref`
//!   `garner_combine_q1_q2_signed`  == `sp_matmul_q_ref::garner_combine_q1_q2_signed`
//!   `matmul_60bit_ref`             == `sp_matmul_q_ref::matmul_60bit_ref`
//!
//! Functions here are byte-equivalent to the sibling-crate versions — both
//! reduce the same upstream math-core ntt_crt.c arithmetic.

#![allow(dead_code)]

// ─── Frozen primes + Barrett constants (sibling: sp_barrett_oracle) ───────────

pub const SP_NTT_Q1: u32 = 1073738753;
pub const SP_NTT_Q2: u32 = 1073732609;
pub const SP_MU_Q1:  u32 = 1073744895;
pub const SP_MU_Q2:  u32 = 1073751039;

/// Garner CRT inverse: q_1^{-1} mod q_2. Python-verified
/// `(SP_NTT_Q1 * Q1_INV_MOD_Q2) mod SP_NTT_Q2 == 1`.
pub const Q1_INV_MOD_Q2: u32 = 894602413;

/// Modulus product M = q_1 * q_2 ≈ 2^60. Matches `SP_NTT_M` in
/// `lib/shannon-prime-system/include/sp/ntt_crt.h`.
pub const M_Q1Q2: u64 = (SP_NTT_Q1 as u64) * (SP_NTT_Q2 as u64);

/// Pick (q, mu) by q_idx. q_idx ∈ {0, 1}.
pub fn q_mu(q_idx: i32) -> (u32, u32) {
    if q_idx == 0 { (SP_NTT_Q1, SP_MU_Q1) } else { (SP_NTT_Q2, SP_MU_Q2) }
}

// ─── Barrett reduction (sibling: sp_barrett_oracle::barrett_reduce32) ─────────

/// Barrett reduction at k=60: r = x mod q ∈ [0, q).
/// Inputs: x ∈ [0, 2^60), q < 2^30, μ = floor(2^60 / q).
pub fn barrett_reduce32(x: u64, q: u32, mu: u32) -> u32 {
    let qhat: u64 = ((x >> 29) * (mu as u64)) >> 31;
    let r0: u64 = x.wrapping_sub(qhat.wrapping_mul(q as u64));
    let r1 = if r0 >= q as u64 { r0 - q as u64 } else { r0 };
    let r2 = if r1 >= q as u64 { r1 - q as u64 } else { r1 };
    r2 as u32
}

// ─── Frobenius dual-lift (the architectural payload) ──────────────────────────

/// Q8 symmetric code limit. Matches `SP_FROB_QMAX` from frobenius_lift.h.
pub const Q8_QMAX: i32 = 127;

/// Per-tensor scale via the symmetric absmax rule used by the Frobenius lift.
/// `sp_frob_row_scale` from `lib/shannon-prime-system/core/frobenius/frobenius_lift.c:13-21`
/// applied to the full flattened tensor.
pub fn per_tensor_scale(tensor: &[f32]) -> f32 {
    let mut m = 0.0f32;
    for &v in tensor {
        let a = v.abs();
        if a > m { m = a; }
    }
    m
}

/// Quantize a single value to its int8 code via the round-half-away-from-zero
/// rule from `sp_frob_quant1` in frobenius_lift.c:25-35. Codes are in
/// [-127, 127] (symmetric, NOT -128).
pub fn quant1_q8(v: f32, scale: f32) -> i8 {
    if scale == 0.0 { return 0; }
    let x = v / scale * 127.0;
    let r = if x >= 0.0 { (x + 0.5).floor() } else { (x - 0.5).ceil() };
    let r = r.clamp(-(Q8_QMAX as f32), Q8_QMAX as f32);
    r as i8
}

/// Dequantize a single code back to fp32 via the inline-lift rule.
pub fn dequant1_q8(code: i8, scale: f32) -> f32 {
    (code as f32) * (scale / 127.0)
}

/// Lift a Q8 code to its Z_q residue. Codes are signed; primes are u32, so
/// negative codes map to q + code. Both frozen primes are ~2^30; codes are
/// ∈ [-127, 127], so the addition is well-defined.
#[inline]
pub fn lift_q8_to_zq(code: i8, q: u32) -> u32 {
    if code >= 0 {
        code as u32
    } else {
        // code is negative; (q + code) lifts to the canonical residue
        q.wrapping_sub((-(code as i32)) as u32)
    }
}

/// Tensor packed for dual-prime Trick #1 dispatch. Holds:
///  * the raw int8 codes (for inspection / debug)
///  * the per-tensor scale (single fp32 per the per-tensor simplification, §5)
///  * two parallel u32 arrays, one per prime, each in [0, q_i)
pub struct DualPrimeTensor {
    pub codes: Vec<i8>,
    pub scale: f32,
    pub q1_residues: Vec<u32>,
    pub q2_residues: Vec<u32>,
    pub n: usize,
}

impl DualPrimeTensor {
    /// Pack an fp32 tensor (row-major or 1-D, doesn't matter — single per-tensor scale).
    pub fn pack(tensor: &[f32]) -> Self {
        let scale = per_tensor_scale(tensor);
        let codes: Vec<i8> = tensor.iter().map(|&v| quant1_q8(v, scale)).collect();
        let q1_residues: Vec<u32> = codes.iter().map(|&c| lift_q8_to_zq(c, SP_NTT_Q1)).collect();
        let q2_residues: Vec<u32> = codes.iter().map(|&c| lift_q8_to_zq(c, SP_NTT_Q2)).collect();
        DualPrimeTensor {
            n: tensor.len(),
            codes,
            scale,
            q1_residues,
            q2_residues,
        }
    }

    /// Dequantize back to fp32 via the inline-lift rule.
    pub fn dequantize(&self) -> Vec<f32> {
        self.codes.iter().map(|&c| dequant1_q8(c, self.scale)).collect()
    }
}

// ─── Scalar reference mod-q matmul (sibling: matmul_q_scalar_ref) ─────────────

/// Y[b][i] = (sum_k X[b][k] * W[k][i]) mod q
///
/// This is the same per-k Barrett + modular-add accumulation algorithm
/// the cDSP HVX kernel uses (per `sp_compute_crt_imp.c:247-293`); the only
/// difference is execution silicon (this runs on ARM Cortex-X2/A710 cores
/// via Rust scalar arithmetic). For Trick #1 ARM-q2 island this IS the
/// production path — no NEON acceleration in v1, per PLAN §D-B.
pub fn matmul_q_scalar_ref(
    q_idx: i32,
    batch: usize,
    d_in: usize,
    d_out: usize,
    x: &[u32],
    w: &[u32],
) -> Vec<u32> {
    assert_eq!(x.len(), batch * d_in);
    assert_eq!(w.len(), d_in * d_out);
    let (q, mu) = q_mu(q_idx);
    let mut y = vec![0u32; batch * d_out];
    for b in 0..batch {
        for i in 0..d_out {
            let mut acc: u32 = 0;
            for k in 0..d_in {
                let xv = x[b * d_in + k];
                let wv = w[k * d_out + i];
                let prod = barrett_reduce32(xv as u64 * wv as u64, q, mu);
                let sum = acc + prod;
                acc = if sum >= q { sum - q } else { sum };
            }
            y[b * d_out + i] = acc;
        }
    }
    y
}

// ─── Garner CRT recombination (sibling: garner_combine_q1_q2_signed) ──────────

/// Symmetric Garner returning signed centered residue in (-M/2, M/2].
/// Byte-for-byte mirror of `garner_one` in
/// `lib/shannon-prime-system/core/ntt_crt/ntt_crt.c:303-316`.
pub fn garner_combine_q1_q2_signed(r1: &[u32], r2: &[u32]) -> Vec<i64> {
    assert_eq!(r1.len(), r2.len());
    let q1 = SP_NTT_Q1;
    let q2 = SP_NTT_Q2;
    let inv = Q1_INV_MOD_Q2 as u64;
    let q2_64 = q2 as u64;
    let m = M_Q1Q2;
    let half_m: i64 = (m / 2) as i64;
    r1.iter().zip(r2.iter()).map(|(&a, &b)| {
        let diff: u64 = if b >= a {
            (b - a) as u64
        } else {
            q2_64 - ((a - b) as u64)
        };
        let t = (diff * inv) % q2_64;
        let r: u64 = (a as u64) + (q1 as u64) * t;
        let mut v: i64 = r as i64;
        if v > half_m { v -= m as i64; }
        v
    }).collect()
}

/// Unsigned-domain Garner (in [0, M)). Used only for cross-check vs the
/// signed-centered version in unit tests.
pub fn garner_combine_q1_q2_unsigned(r1: &[u32], r2: &[u32]) -> Vec<u64> {
    assert_eq!(r1.len(), r2.len());
    let q1 = SP_NTT_Q1;
    let q2 = SP_NTT_Q2;
    let inv = Q1_INV_MOD_Q2 as u64;
    let q2_64 = q2 as u64;
    r1.iter().zip(r2.iter()).map(|(&a, &b)| {
        let diff: u64 = if b >= a {
            (b - a) as u64
        } else {
            q2_64 - ((a - b) as u64)
        };
        let t = (diff * inv) % q2_64;
        (a as u64) + (q1 as u64) * t
    }).collect()
}

// ─── 60-bit unreduced reference matmul (sibling: matmul_60bit_ref) ────────────

/// SIGNED 60-bit reference matmul over int8 codes, with no modular reduction.
/// Returns i64 lanes (each the exact signed sum-of-products). Used as the
/// reference target for T_TRICK1_NUMERICAL_EQUIVALENCE (Garner-combined output
/// must match this byte-for-byte).
///
/// Codes are int8 in [-127, 127]; the maximum |sum| is K * 127² ≈ 2^25
/// (K=2048), which fits trivially in i64 AND in (-M/2, M/2].
pub fn matmul_int8_signed_ref(
    batch: usize,
    d_in: usize,
    d_out: usize,
    x_codes: &[i8],
    w_codes: &[i8],
) -> Vec<i64> {
    assert_eq!(x_codes.len(), batch * d_in);
    assert_eq!(w_codes.len(), d_in * d_out);
    let mut y = vec![0i64; batch * d_out];
    for b in 0..batch {
        for i in 0..d_out {
            let mut acc: i64 = 0;
            for k in 0..d_in {
                let xv = x_codes[b * d_in + k] as i64;
                let wv = w_codes[k * d_out + i] as i64;
                acc += xv * wv;
            }
            y[b * d_out + i] = acc;
        }
    }
    y
}

/// fp32 reference matmul on the DEQUANTIZED Q8 tensors. Used for the
/// secondary fp32-equivalence gate per PLAN §D-F (b) — expected to match
/// `dequant(garner(...))` within the documented relative-error budget,
/// not byte-for-byte.
pub fn matmul_f32_ref(
    batch: usize,
    d_in: usize,
    d_out: usize,
    x_f32: &[f32],
    w_f32: &[f32],
) -> Vec<f32> {
    assert_eq!(x_f32.len(), batch * d_in);
    assert_eq!(w_f32.len(), d_in * d_out);
    let mut y = vec![0.0f32; batch * d_out];
    for b in 0..batch {
        for i in 0..d_out {
            let mut acc: f64 = 0.0;
            for k in 0..d_in {
                acc += (x_f32[b * d_in + k] as f64) * (w_f32[k * d_out + i] as f64);
            }
            y[b * d_out + i] = acc as f32;
        }
    }
    y
}

/// Reconstruct fp32 from a Garner-combined signed integer output via the
/// dequant rule. Per PLAN §5, with per-tensor scales s_x and s_w, the
/// dequantized value is `Y_int * s_x * s_w / 127²`.
pub fn dequantize_garner_output(
    y_int: &[i64],
    s_x: f32,
    s_w: f32,
) -> Vec<f32> {
    let factor = (s_x as f64) * (s_w as f64) / (127.0_f64 * 127.0_f64);
    y_int.iter().map(|&v| (v as f64 * factor) as f32).collect()
}

// ─── Deterministic test-vector generator ──────────────────────────────────────

/// Generate a deterministic fp32 tensor of size `n` from `seed`. Values are
/// in (-1.0, 1.0) (centered, unit-bounded) so the per-tensor scale is ~1.0
/// and the int8 quantization is fully exercised.
pub fn gen_fp32_tensor(seed: u64, n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    let mut s = seed;
    for _ in 0..n {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        // Take the high 32 bits as i32, scale to (-1, 1).
        let bits = (s >> 32) as i32;
        let f = (bits as f64) / (i32::MAX as f64);
        out.push(f as f32);
    }
    out
}

// ─── Tests (host-runnable; run via `cargo test --target <host>`) ──────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify Q1_INV_MOD_Q2 satisfies the Garner inverse identity (sibling
    /// crate has the same test; we duplicate to make this crate
    /// self-validating).
    #[test]
    fn garner_inverse_identity() {
        let q1 = SP_NTT_Q1 as u64;
        let q2 = SP_NTT_Q2 as u64;
        let inv = Q1_INV_MOD_Q2 as u64;
        assert_eq!((q1 * inv) % q2, 1);
    }

    /// Q8 round-trip: a known set of values quantize-then-dequantize within
    /// the expected per-element error bound. Sanity check that the
    /// quantization formulas here match `sp_frob_quant1` / `sp_frob_dequant1`.
    #[test]
    fn q8_round_trip_bounded() {
        let scale = 1.0f32;
        // Sample at unit grid in (-1, 1). Each round-trip should produce
        // a value within (s/127)/2 of the input.
        let n = 256;
        let inputs: Vec<f32> = (0..n).map(|i| (i as f32 / (n as f32 / 2.0)) - 1.0).collect();
        let half_lsb = scale / 127.0 / 2.0;
        for &v in &inputs {
            let q = quant1_q8(v, scale);
            let v_hat = dequant1_q8(q, scale);
            // Round-half-away-from-zero rule: |v - v_hat| <= 0.5 * scale/127
            // (plus the FP rounding noise of the quantization itself).
            assert!((v - v_hat).abs() <= half_lsb + 1e-6,
                "round-trip violation: v={v} v_hat={v_hat} delta={}", v - v_hat);
        }
    }

    /// Pack an fp32 tensor into a DualPrimeTensor and verify both residue
    /// arrays are mathematically consistent: lifting a negative code into
    /// q_i must produce a value ≡ code (mod q_i).
    #[test]
    fn dual_lift_residue_consistency() {
        let tensor = vec![0.0, 0.5, -0.5, 1.0, -1.0, 0.25, -0.75];
        let dpt = DualPrimeTensor::pack(&tensor);
        assert_eq!(dpt.codes.len(), tensor.len());
        assert_eq!(dpt.q1_residues.len(), tensor.len());
        assert_eq!(dpt.q2_residues.len(), tensor.len());
        for (i, &code) in dpt.codes.iter().enumerate() {
            let r1 = dpt.q1_residues[i];
            let r2 = dpt.q2_residues[i];
            // r ≡ code (mod q)
            let expected_r1: i64 = (code as i64).rem_euclid(SP_NTT_Q1 as i64);
            let expected_r2: i64 = (code as i64).rem_euclid(SP_NTT_Q2 as i64);
            assert_eq!(r1 as i64, expected_r1, "q1 lift wrong: code={code}");
            assert_eq!(r2 as i64, expected_r2, "q2 lift wrong: code={code}");
        }
    }

    /// Stage 1 LOAD-BEARING gate: Garner combiner against `matmul_int8_signed_ref`.
    /// Pure-host int8 matmul; per-prime scalar matmul on the lifted residues;
    /// Garner-combine; compare element-wise. PASS = byte-exact.
    ///
    /// This is the T_TRICK1_GARNER_BIT_EXACT_VS_HOST_CRT gate at Stage 1
    /// (without silicon dispatch).
    #[test]
    fn stage1_garner_bit_exact_via_full_matmul() {
        let (batch, d_in, d_out) = (1usize, 2048usize, 256usize);
        let x_f32 = gen_fp32_tensor(0xDEAD_BEEF_CAFE_BABE, batch * d_in);
        let w_f32 = gen_fp32_tensor(0xFEED_FACE_BAAD_F00D, d_in * d_out);
        let x_dpt = DualPrimeTensor::pack(&x_f32);
        let w_dpt = DualPrimeTensor::pack(&w_f32);

        // Per-prime mod-q matmul on the lifted residues.
        let y_q1 = matmul_q_scalar_ref(0, batch, d_in, d_out,
                                       &x_dpt.q1_residues, &w_dpt.q1_residues);
        let y_q2 = matmul_q_scalar_ref(1, batch, d_in, d_out,
                                       &x_dpt.q2_residues, &w_dpt.q2_residues);
        let y_garner = garner_combine_q1_q2_signed(&y_q1, &y_q2);

        // Reference: int8 matmul with signed accumulation.
        let y_ref = matmul_int8_signed_ref(batch, d_in, d_out, &x_dpt.codes, &w_dpt.codes);

        assert_eq!(y_garner.len(), y_ref.len());
        let mut divergences = 0usize;
        for (i, (&g, &r)) in y_garner.iter().zip(y_ref.iter()).enumerate() {
            if g != r {
                divergences += 1;
                if divergences <= 5 {
                    eprintln!("[stage1] divergence at i={i}: garner={g} ref={r} delta={}", g - r);
                }
            }
        }
        assert_eq!(divergences, 0,
                   "Stage 1 T_TRICK1_GARNER_BIT_EXACT_VS_HOST_CRT FAIL: {divergences} / {} divergences",
                   y_ref.len());
    }

    /// Stage 1 secondary gate: dequantized Garner output matches fp32 reference
    /// matmul within the per-element relative-error budget.
    #[test]
    fn stage1_fp32_equivalence_within_budget() {
        let (batch, d_in, d_out) = (1usize, 2048usize, 256usize);
        let x_f32 = gen_fp32_tensor(0xDEAD_BEEF_CAFE_BABE, batch * d_in);
        let w_f32 = gen_fp32_tensor(0xFEED_FACE_BAAD_F00D, d_in * d_out);
        let x_dpt = DualPrimeTensor::pack(&x_f32);
        let w_dpt = DualPrimeTensor::pack(&w_f32);

        // Trick #1 path: lift → per-prime matmul → Garner → dequant.
        let y_q1 = matmul_q_scalar_ref(0, batch, d_in, d_out,
                                       &x_dpt.q1_residues, &w_dpt.q1_residues);
        let y_q2 = matmul_q_scalar_ref(1, batch, d_in, d_out,
                                       &x_dpt.q2_residues, &w_dpt.q2_residues);
        let y_garner = garner_combine_q1_q2_signed(&y_q1, &y_q2);
        let y_real_trick1 = dequantize_garner_output(&y_garner, x_dpt.scale, w_dpt.scale);

        // Reference: dequantize Q8 → fp32 matmul.
        let x_dq = x_dpt.dequantize();
        let w_dq = w_dpt.dequantize();
        let y_real_ref = matmul_f32_ref(batch, d_in, d_out, &x_dq, &w_dq);

        // Per PLAN §6: relative-error budget 5e-3.
        // (The integer path is byte-exact; the fp32 reference uses the SAME
        // dequantized codes, so the two should agree to within fp32 sum
        // ordering noise — well below the documented budget. Wide margin.)
        let budget: f64 = 5e-3;
        let mut max_relerr: f64 = 0.0;
        let mut budget_violations = 0usize;
        for (i, (&t, &r)) in y_real_trick1.iter().zip(y_real_ref.iter()).enumerate() {
            let denom = (r as f64).abs().max(1e-6);
            let relerr = ((t as f64) - (r as f64)).abs() / denom;
            if relerr > max_relerr { max_relerr = relerr; }
            if relerr > budget {
                if budget_violations < 5 {
                    eprintln!("[stage1] fp32 budget violation at i={i}: trick1={t} ref={r} relerr={relerr}");
                }
                budget_violations += 1;
            }
        }
        eprintln!("[stage1] max_relerr={max_relerr:.3e} budget={budget:.3e} violations={budget_violations} / {}",
                  y_real_ref.len());
        assert_eq!(budget_violations, 0, "Stage 1 fp32 budget gate FAIL");
    }
}
