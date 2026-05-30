//! §3-HX Sprint K v0.beta Stage 2.5c — Rust scalar reference for the mod_q
//! matmul kernel + Garner CRT recombination.
//!
//! - `matmul_q_scalar_ref` produces Y[B][D_out] = X[B][D_in] · W[D_in][D_out]
//!   mod q on the ARM side, using `barrett_reduce32` from `sp_barrett_oracle`
//!   for every intermediate reduction.  Bitwise reference against which the
//!   on-device HVX kernel is compared (T_MATMUL_Q_CORRECTNESS).
//!
//! - `garner_combine_q1_q2` recombines (r_1, r_2) into the unique r mod
//!   (q_1 * q_2) ∈ [0, M).  M = q_1 * q_2 ≈ 2^60 so r fits u64.
//!
//! - `matmul_60bit_ref` runs the SAME matmul WITHOUT modular reduction, in
//!   pure u128 arithmetic, and reports the u64 result iff it fits in M.
//!   T_GARNER_BIT_EXACT compares Garner-combined output against this.

#![allow(dead_code)]

use super::sp_barrett_oracle::{barrett_reduce32, SP_NTT_Q1, SP_NTT_Q2, SP_MU_Q1, SP_MU_Q2};

/// Garner CRT inverse: q_1^{-1} mod q_2.  Precomputed; verified
/// (q_1 * Q1_INV_MOD_Q2) mod q_2 = 1.
pub const Q1_INV_MOD_Q2: u32 = 894602413;

/// Modulus product M = q_1 * q_2.  60-bit exact value.
pub const M_Q1Q2: u64 = (SP_NTT_Q1 as u64) * (SP_NTT_Q2 as u64);

/// Pick (q, mu) by q_idx.  q_idx ∈ {0, 1}.
pub fn q_mu(q_idx: i32) -> (u32, u32) {
    if q_idx == 0 { (SP_NTT_Q1, SP_MU_Q1) } else { (SP_NTT_Q2, SP_MU_Q2) }
}

/// Scalar reference for the mod_q matmul.
///
/// Y[b][i] = ( sum_{k=0..D_in} X[b][k] * W[k][i] ) mod q
///
/// X is row-major B × D_in.  W is row-major D_in × D_out.  Y is row-major
/// B × D_out.  All elements are u32 in [0, q).  Per-step Barrett reduction
/// matches the kernel's per-k Barrett path (chosen design path C in PLAN).
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
                // prod = (xv * wv) mod q via Barrett — same code path as HVX
                let prod = barrett_reduce32(xv as u64 * wv as u64, q, mu);
                // acc + prod < 2q; one conditional subtract
                let sum = acc + prod;
                acc = if sum >= q { sum - q } else { sum };
            }
            y[b * d_out + i] = acc;
        }
    }
    y
}

/// Garner CRT recombination: given r_1 (per q_1) and r_2 (per q_2), produce
/// the unique r ∈ [0, M) with r ≡ r_1 (mod q_1) and r ≡ r_2 (mod q_2).
///
/// r = r_1 + q_1 * ((r_2 - r_1) * Q1_INV_MOD_Q2 mod q_2)
pub fn garner_combine_q1_q2(r1: &[u32], r2: &[u32]) -> Vec<u64> {
    assert_eq!(r1.len(), r2.len());
    let q1 = SP_NTT_Q1;
    let q2 = SP_NTT_Q2;
    let inv = Q1_INV_MOD_Q2 as u64;
    let q2_64 = q2 as u64;
    r1.iter().zip(r2.iter()).map(|(&a, &b)| {
        // diff = (b - a) mod q_2.  Use rem_euclid in i64 to handle wraparound.
        let diff: u64 = if b >= a {
            (b - a) as u64
        } else {
            // (b - a) is negative; canonicalize to [0, q_2)
            q2_64 - ((a - b) as u64)
        };
        // diff is already in [0, q_2); inv is in [0, q_2); product fits u64
        // since q_2 < 2^30 → q_2^2 < 2^60.
        let t = (diff * inv) % q2_64;
        (a as u64) + (q1 as u64) * t
    }).collect()
}

/// §4-NTT Sprint NTT.4 — symmetric Garner CRT producing SIGNED centered output
/// in (-M/2, M/2].
///
/// Matches math-core `garner_one` (ntt_crt.c:303-317): same unsigned recombine
/// formula `r = x1 + q1 * ((x2 - x1) * q1_inv_mod_q2 mod q2)`, then cast to
/// i64 and center: `if (v > M/2) v -= M`.
///
/// Distinct from `garner_combine_q1_q2` above (which returns u64 in [0, M));
/// K.beta.2.5c's existing gates consume the unsigned variant — DO NOT modify
/// that function.
///
/// Used by NTT.4's end-to-end polynomial multiplication smoke harness to
/// recombine per-prime INTT outputs into a single signed coefficient vector
/// matching math-core's `ntt_inverse` output.
pub fn garner_combine_q1_q2_signed(r1: &[u32], r2: &[u32]) -> Vec<i64> {
    assert_eq!(r1.len(), r2.len());
    let q1 = SP_NTT_Q1;
    let q2 = SP_NTT_Q2;
    let inv = Q1_INV_MOD_Q2 as u64;
    let q2_64 = q2 as u64;
    let m: u64 = M_Q1Q2;
    let half_m: i64 = (m / 2) as i64;
    r1.iter().zip(r2.iter()).map(|(&a, &b)| {
        let diff: u64 = if b >= a {
            (b - a) as u64
        } else {
            q2_64 - ((a - b) as u64)
        };
        let t = (diff * inv) % q2_64;
        let r: u64 = (a as u64) + (q1 as u64) * t;
        // u64 r is in [0, M); M < 2^60 so r fits i64. Center to (-M/2, M/2].
        let mut v: i64 = r as i64;
        if v > half_m { v -= m as i64; }
        v
    }).collect()
}

/// 60-bit-exact matmul WITHOUT modular reduction.  Used by T_GARNER_BIT_EXACT
/// to verify the CRT-combined output recovers the un-reduced sum-of-products.
///
/// Returns Some(y) where each lane is `sum_k X[b][k] * W[k][i]` iff every
/// lane fits in M = q_1 * q_2.  Returns None if any lane overflows (the
/// caller MUST bound inputs so this can't happen — bound provided in the
/// smoke harness, see crate docs).
pub fn matmul_60bit_ref(
    batch: usize,
    d_in: usize,
    d_out: usize,
    x: &[u32],
    w: &[u32],
) -> Option<Vec<u64>> {
    assert_eq!(x.len(), batch * d_in);
    assert_eq!(w.len(), d_in * d_out);
    let mut y = vec![0u64; batch * d_out];
    for b in 0..batch {
        for i in 0..d_out {
            let mut acc: u128 = 0;
            for k in 0..d_in {
                let xv = x[b * d_in + k] as u128;
                let wv = w[k * d_out + i] as u128;
                acc += xv * wv;
            }
            if acc >= M_Q1Q2 as u128 { return None; }
            y[b * d_out + i] = acc as u64;
        }
    }
    Some(y)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify the Garner inverse satisfies (q_1 * Q1_INV_MOD_Q2) mod q_2 = 1.
    #[test]
    fn garner_inverse_is_correct() {
        let q1 = SP_NTT_Q1 as u64;
        let q2 = SP_NTT_Q2 as u64;
        let inv = Q1_INV_MOD_Q2 as u64;
        assert_eq!((q1 * inv) % q2, 1);
    }

    /// Garner CRT identity: given r_i = u mod q_i for a known u < M, Garner
    /// must recover u.
    #[test]
    fn garner_recovers_known_residue() {
        let q1 = SP_NTT_Q1 as u64;
        let q2 = SP_NTT_Q2 as u64;
        // A handful of u values to test.
        let test_vals: Vec<u64> = vec![
            0,
            1,
            42,
            123456789,
            q1 - 1,
            q2 - 1,
            q1 * q2 / 2,
            q1 * q2 - 1,
        ];
        let r1: Vec<u32> = test_vals.iter().map(|u| (u % q1) as u32).collect();
        let r2: Vec<u32> = test_vals.iter().map(|u| (u % q2) as u32).collect();
        let combined = garner_combine_q1_q2(&r1, &r2);
        for (got, want) in combined.iter().zip(test_vals.iter()) {
            assert_eq!(got, want, "Garner mismatch: r1={} r2={}", got, want);
        }
    }

    /// matmul_q_scalar_ref produces results in [0, q) for both primes.
    #[test]
    fn matmul_q_in_canonical_range() {
        let (b, d_in, d_out) = (8usize, 128usize, 128usize);
        let mut x = vec![0u32; b * d_in];
        let mut w = vec![0u32; d_in * d_out];
        let mut seed: u64 = 0xABCDEF0123456789;
        let q = SP_NTT_Q1;
        for v in x.iter_mut() {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            *v = (seed as u32) % q;
        }
        for v in w.iter_mut() {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            *v = (seed as u32) % q;
        }
        let y = matmul_q_scalar_ref(0, b, d_in, d_out, &x, &w);
        for &v in y.iter() { assert!(v < q); }
        // q_2 path also valid range
        for v in x.iter_mut() { *v = (*v) % SP_NTT_Q2; }
        for v in w.iter_mut() { *v = (*v) % SP_NTT_Q2; }
        let y = matmul_q_scalar_ref(1, b, d_in, d_out, &x, &w);
        for &v in y.iter() { assert!(v < SP_NTT_Q2); }
    }

    /// Garner round-trip via the matmul: given X, W in [0, sqrt(M / d_in)),
    /// the unreduced matmul fits in M and Garner recombines correctly.
    #[test]
    fn garner_roundtrip_via_matmul() {
        let (b, d_in, d_out) = (4usize, 64usize, 64usize);
        // Bound elements so sum of d_in products < M.
        // Per element < sqrt(M / d_in) ≈ sqrt(2^60 / 64) = 2^27.
        let bound = 1u32 << 27;
        let mut x = vec![0u32; b * d_in];
        let mut w = vec![0u32; d_in * d_out];
        let mut seed: u64 = 0xDEADBEEFCAFEBABE;
        for v in x.iter_mut() {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            *v = (seed as u32) % bound;
        }
        for v in w.iter_mut() {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            *v = (seed as u32) % bound;
        }
        let y_q1 = matmul_q_scalar_ref(0, b, d_in, d_out, &x, &w);
        let y_q2 = matmul_q_scalar_ref(1, b, d_in, d_out, &x, &w);
        let recombined = garner_combine_q1_q2(&y_q1, &y_q2);
        let direct = matmul_60bit_ref(b, d_in, d_out, &x, &w).expect("60-bit fits");
        for i in 0..recombined.len() {
            assert_eq!(recombined[i], direct[i],
                "Garner CRT mismatch at i={}: recombined={} direct={}",
                i, recombined[i], direct[i]);
        }
    }

    /// §4-NTT Sprint NTT.4 — T_NTT4_GARNER_SIGNED_BIT_EXACT.
    ///
    /// Drive 1000 random (r_1, r_2) pairs through garner_combine_q1_q2_signed
    /// and verify the output matches math-core's garner_one centering rule:
    /// take the unsigned u64 recombine in [0, M), and if > M/2, subtract M.
    ///
    /// The unsigned step is the same as garner_combine_q1_q2 (already gated
    /// in `garner_recovers_known_residue`), so this test focuses on the
    /// centering boundary: we verify
    ///   signed == unsigned                      if unsigned <= M/2
    ///   signed == (unsigned as i64) - (M as i64) if unsigned >  M/2
    /// holds element-wise.
    #[test]
    fn garner_signed_matches_centering_rule() {
        let q1 = SP_NTT_Q1;
        let q2 = SP_NTT_Q2;
        let m = M_Q1Q2;
        let half_m: u64 = m / 2;

        let mut seed: u64 = 0xCEBA_FECA_5C0F_FEE5;
        let n_pairs = 1000usize;
        let mut r1: Vec<u32> = Vec::with_capacity(n_pairs);
        let mut r2: Vec<u32> = Vec::with_capacity(n_pairs);
        for _ in 0..n_pairs {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            r1.push((seed as u32) % q1);
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            r2.push((seed as u32) % q2);
        }

        // Force half of the pairs into the upper half (> M/2) by selecting
        // r1, r2 that recombine to a known large value. The PRNG sweep is
        // already uniform over [0, M), so statistically half land above M/2,
        // but we add a few explicit corner-case pairs to ensure coverage.
        // Boundaries: 0, M-1, M/2, M/2+1.
        for &u in &[0u64, 1, half_m, half_m + 1, m - 1] {
            r1.push((u % q1 as u64) as u32);
            r2.push((u % q2 as u64) as u32);
        }

        let unsigned = garner_combine_q1_q2(&r1, &r2);
        let signed   = garner_combine_q1_q2_signed(&r1, &r2);
        assert_eq!(unsigned.len(), signed.len());

        let half_m_i: i64 = half_m as i64;
        for i in 0..signed.len() {
            let u = unsigned[i];
            let s = signed[i];
            let expected: i64 = if u as i64 > half_m_i {
                (u as i64).wrapping_sub(m as i64)
            } else {
                u as i64
            };
            assert_eq!(s, expected,
                "centering mismatch at i={}: unsigned={} signed={} expected={} half_M={}",
                i, u, s, expected, half_m);
        }

        // Also assert: signed value is in (-M/2, M/2].
        for &s in signed.iter() {
            assert!(s > -(half_m_i + 1) && s <= half_m_i,
                "out-of-range signed value: {} (half_M={})", s, half_m);
        }

        // Range coverage diagnostic.
        let n_upper: usize = unsigned.iter().filter(|&&u| u > half_m).count();
        let n_lower: usize = unsigned.iter().filter(|&&u| u <= half_m).count();
        assert!(n_upper > 0 && n_lower > 0,
            "test did not cover both halves: upper={} lower={}", n_upper, n_lower);
    }
}
