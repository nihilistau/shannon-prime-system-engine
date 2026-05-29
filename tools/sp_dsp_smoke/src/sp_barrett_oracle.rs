//! §3-HX Sprint K v0.beta Stage 2.5a — Rust scalar Barrett reference + harness.
//!
//! Mirrors engine 63d7e2d:src/backends/cuda/ptx_ntt.cuh::barrett_reduce32_ref
//! and the C scalar in sp_compute_crt_imp.c.  Bitwise-identical math is the
//! point of T_BARRETT_SCALAR_ORACLE.

pub const SP_NTT_Q1: u32 = 1073738753;
pub const SP_NTT_Q2: u32 = 1073732609;
pub const SP_MU_Q1:  u32 = 1073744895;   // floor(2^60 / q_1)
pub const SP_MU_Q2:  u32 = 1073751039;   // floor(2^60 / q_2)

/// Barrett reduction at k=60: r = x mod q ∈ [0, q).
/// Inputs: x ∈ [0, 2^60), q < 2^30, μ = floor(2^60 / q).
pub fn barrett_reduce32(x: u64, q: u32, mu: u32) -> u32 {
    let qhat: u64 = ((x >> 29) * (mu as u64)) >> 31;
    let r0: u64 = x.wrapping_sub(qhat.wrapping_mul(q as u64));
    let r1 = if r0 >= q as u64 { r0 - q as u64 } else { r0 };
    let r2 = if r1 >= q as u64 { r1 - q as u64 } else { r1 };
    r2 as u32
}

pub fn modmul_q1(a: u32, b: u32) -> u32 {
    barrett_reduce32(a as u64 * b as u64, SP_NTT_Q1, SP_MU_Q1)
}

pub fn modmul_q2(a: u32, b: u32) -> u32 {
    barrett_reduce32(a as u64 * b as u64, SP_NTT_Q2, SP_MU_Q2)
}

/// Deterministic test-vector generator.  Returns (a_vec, b_vec) of length n,
/// covering edge cases first, then PRNG, then worst-case near (q-1, q-1).
pub fn gen_test_vectors(q: u32, seed: u64, n: usize) -> (Vec<u32>, Vec<u32>) {
    let mut a = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(n);
    // 4 edge corners
    for (av, bv) in &[(0u32, 0u32), (0, q-1), (q-1, 0), (q-1, q-1)] {
        a.push(*av); b.push(*bv);
    }
    // ~10 worst-case: both operands within 16 of q-1
    for i in 0..10 {
        a.push(q - 1 - i);
        b.push(q - 1 - (i.wrapping_mul(7) & 15));
    }
    // Fill the rest with a deterministic LCG (Numerical Recipes constants),
    // each value reduced mod q.
    let mut s = seed;
    while a.len() < n {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        let av = (s as u32) % q;
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        let bv = (s as u32) % q;
        a.push(av);
        b.push(bv);
    }
    (a, b)
}
