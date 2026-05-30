//! §3-HX Sprint K v0.beta Stage 2.5c Stage 1 — host-runnable test for the
//! scalar reference + Garner CRT.  Doubles as a unit-test harness without
//! requiring a `[lib]` target (existing crate is bin-only).
//!
//! Run on host (Windows or Linux):
//!     cargo run --bin sp_matmul_q_ref_test
//!
//! Validates four invariants:
//!   1. (q_1 * Q1_INV_MOD_Q2) mod q_2 == 1
//!   2. Garner recovers any known u < M from its (u%q_1, u%q_2) residues
//!   3. matmul_q_scalar_ref outputs lie in [0, q) for both primes
//!   4. Garner(matmul_q1, matmul_q2) == matmul_60bit_ref when inputs are
//!      bounded so the unreduced sum < M (the CRT round-trip identity)

mod sp_barrett_oracle;
mod sp_matmul_q_ref;

use sp_barrett_oracle::{SP_NTT_Q1, SP_NTT_Q2};
use sp_matmul_q_ref::{
    Q1_INV_MOD_Q2, M_Q1Q2, garner_combine_q1_q2,
    matmul_60bit_ref, matmul_q_scalar_ref,
};

fn main() {
    let mut fails = 0usize;
    eprintln!("[K-β-2.5c ref-test] === Garner inverse identity ===");
    let q1 = SP_NTT_Q1 as u64;
    let q2 = SP_NTT_Q2 as u64;
    let inv = Q1_INV_MOD_Q2 as u64;
    let lhs = (q1 * inv) % q2;
    eprintln!("[K-β-2.5c ref-test]   (q_1 * Q1_INV_MOD_Q2) mod q_2 = {} (want 1)", lhs);
    if lhs != 1 { fails += 1; }
    eprintln!("[K-β-2.5c ref-test]   M = q_1 * q_2 = {} ({} bits)", M_Q1Q2, 64 - M_Q1Q2.leading_zeros());

    eprintln!("\n[K-β-2.5c ref-test] === Garner recovers known residues ===");
    let test_vals: Vec<u64> = vec![
        0, 1, 42, 123_456_789,
        q1 - 1, q2 - 1, q1, q2,
        q1 * q2 / 2, q1 * q2 - 1,
    ];
    let r1: Vec<u32> = test_vals.iter().map(|u| (u % q1) as u32).collect();
    let r2: Vec<u32> = test_vals.iter().map(|u| (u % q2) as u32).collect();
    let combined = garner_combine_q1_q2(&r1, &r2);
    let mut roundtrip_ok = 0usize;
    for (i, (got, want)) in combined.iter().zip(test_vals.iter()).enumerate() {
        if got == want {
            roundtrip_ok += 1;
        } else {
            eprintln!("[K-β-2.5c ref-test]   FAIL @ idx={}: got={} want={}", i, got, want);
        }
    }
    eprintln!("[K-β-2.5c ref-test]   {}/{} round-trips PASS",
              roundtrip_ok, test_vals.len());
    if roundtrip_ok != test_vals.len() { fails += 1; }

    eprintln!("\n[K-β-2.5c ref-test] === matmul_q output in [0, q) ===");
    let (b, d_in, d_out) = (8usize, 128usize, 128usize);
    let mut x_q1 = vec![0u32; b * d_in];
    let mut w_q1 = vec![0u32; d_in * d_out];
    let mut seed: u64 = 0xABCDEF0123456789;
    for v in x_q1.iter_mut() {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        *v = (seed as u32) % SP_NTT_Q1;
    }
    for v in w_q1.iter_mut() {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        *v = (seed as u32) % SP_NTT_Q1;
    }
    let y1 = matmul_q_scalar_ref(0, b, d_in, d_out, &x_q1, &w_q1);
    let q1_violations = y1.iter().filter(|&&v| v >= SP_NTT_Q1).count();
    eprintln!("[K-β-2.5c ref-test]   q_1 path: {} outputs / 0 range violations", y1.len());
    if q1_violations != 0 { fails += 1; }

    let mut x_q2 = vec![0u32; b * d_in];
    let mut w_q2 = vec![0u32; d_in * d_out];
    for v in x_q2.iter_mut() {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        *v = (seed as u32) % SP_NTT_Q2;
    }
    for v in w_q2.iter_mut() {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        *v = (seed as u32) % SP_NTT_Q2;
    }
    let y2 = matmul_q_scalar_ref(1, b, d_in, d_out, &x_q2, &w_q2);
    let q2_violations = y2.iter().filter(|&&v| v >= SP_NTT_Q2).count();
    eprintln!("[K-β-2.5c ref-test]   q_2 path: {} outputs / 0 range violations", y2.len());
    if q2_violations != 0 { fails += 1; }

    eprintln!("\n[K-β-2.5c ref-test] === Garner CRT round-trip via matmul ===");
    // Bound elements so sum of d_in products < M.  Per element <= sqrt(M/d_in).
    // d_in=128: bound = floor(sqrt(M/128)) = ~floor(sqrt(9e15)) ≈ 95M.
    // Use 2^26 = 67M to leave headroom.
    let bound = 1u32 << 26;
    let (b, d_in, d_out) = (4usize, 64usize, 64usize);
    let mut x = vec![0u32; b * d_in];
    let mut w = vec![0u32; d_in * d_out];
    seed = 0xDEADBEEFCAFEBABE;
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
    let direct = matmul_60bit_ref(b, d_in, d_out, &x, &w)
        .expect("inputs bounded so 60-bit fits");
    let mut bad = 0usize;
    for i in 0..recombined.len() {
        if recombined[i] != direct[i] {
            if bad == 0 {
                eprintln!("[K-β-2.5c ref-test]   first mismatch @ i={}: recombined={} direct={}",
                          i, recombined[i], direct[i]);
            }
            bad += 1;
        }
    }
    eprintln!("[K-β-2.5c ref-test]   {} elements, {} mismatches", recombined.len(), bad);
    if bad != 0 { fails += 1; }

    if fails == 0 {
        eprintln!("\n[K-β-2.5c ref-test] ALL PASS  (Stage 1 host validation green)");
        std::process::exit(0);
    } else {
        eprintln!("\n[K-β-2.5c ref-test] {} subgate(s) FAILED", fails);
        std::process::exit(1);
    }
}
