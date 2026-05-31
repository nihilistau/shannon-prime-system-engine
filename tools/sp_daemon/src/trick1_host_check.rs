//! Sprint TRICK-1-FORWARD — host-runnable correctness module.
//!
//! Mirrors the gates that `trick1_forward_dispatch.rs` runs on-device in the
//! ARM-q2 worker, but builds + runs on Windows / Linux x86 without Hexagon.
//! This is the T_TRICK1FWD_HOST_STAGE1_CORRECTNESS gate — proves the
//! end-to-end DualPrimeTensor + Garner + dequantize pipeline is byte-exact at
//! the integer level AND within budget at the fp32 level, BEFORE any silicon
//! dispatch can mask correctness with a marshalling layer.
//!
//! Per PLAN-TRICK-1-FORWARD §3 Stage 1:
//!   • Verifiable on Windows/Linux host without silicon.
//!   • Same DualPrimeTensor/Garner code paths the on-device ARM-q2 worker uses.
//!   • If this fails on host, the on-device worker is GUARANTEED to fail too —
//!     and the failure is in our lib not the silicon.

use sp_trick1::{
    DualPrimeTensor, dequantize_garner_output, gen_fp32_tensor,
    garner_combine_q1_q2_signed, matmul_f32_ref, matmul_int8_signed_ref,
    matmul_q_scalar_ref,
};

/// Replicate the ARM-q2 worker's compute path host-side. Returns
/// (int_divergences, max_relerr_vs_fp32_ref).
///
/// `seed_x` and `seed_w` deterministically generate the fp32 input tensors;
/// the worker uses `0x5197_2110_1F11_2F33` (x) and `0x7C1B_4477_2D5A_9E04` (w)
/// at `(batch=1, d_in=1152, d_out=256)`. This function exposes the parameters
/// so unit tests can scan a few shape/seed combinations.
pub fn arm_q2_worker_check(
    batch: usize,
    d_in: usize,
    d_out: usize,
    seed_x: u64,
    seed_w: u64,
) -> (usize, f64) {
    let x_f32 = gen_fp32_tensor(seed_x, batch * d_in);
    let w_f32 = gen_fp32_tensor(seed_w, d_in * d_out);
    let x_dpt = DualPrimeTensor::pack(&x_f32);
    let w_dpt = DualPrimeTensor::pack(&w_f32);

    let y_q1 = matmul_q_scalar_ref(
        0, batch, d_in, d_out,
        &x_dpt.q1_residues, &w_dpt.q1_residues,
    );
    let y_q2 = matmul_q_scalar_ref(
        1, batch, d_in, d_out,
        &x_dpt.q2_residues, &w_dpt.q2_residues,
    );
    let y_garner = garner_combine_q1_q2_signed(&y_q1, &y_q2);
    let y_real_trick1 =
        dequantize_garner_output(&y_garner, x_dpt.scale, w_dpt.scale);

    // Integer-domain byte-exact gate.
    let y_int_ref =
        matmul_int8_signed_ref(batch, d_in, d_out, &x_dpt.codes, &w_dpt.codes);
    let int_divergences = y_garner
        .iter()
        .zip(y_int_ref.iter())
        .filter(|(a, b)| a != b)
        .count();

    // fp32 budget gate.
    let x_dq = x_dpt.dequantize();
    let w_dq = w_dpt.dequantize();
    let y_real_ref = matmul_f32_ref(batch, d_in, d_out, &x_dq, &w_dq);
    let mut max_relerr: f64 = 0.0;
    for (&t, &r) in y_real_trick1.iter().zip(y_real_ref.iter()) {
        let denom = (r as f64).abs().max(1e-6);
        let relerr = ((t as f64) - (r as f64)).abs() / denom;
        if relerr > max_relerr {
            max_relerr = relerr;
        }
    }

    (int_divergences, max_relerr)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// T_TRICK1FWD_HOST_STAGE1_CORRECTNESS — exact replica of the on-device
    /// ARM-q2 worker's fixture. Must be byte-exact at the integer level AND
    /// within 5e-3 max relerr at fp32.
    #[test]
    fn arm_q2_worker_fixture_byte_exact() {
        let (divs, max_relerr) = arm_q2_worker_check(
            1, 1152, 256,
            0x5197_2110_1F11_2F33,
            0x7C1B_4477_2D5A_9E04,
        );
        assert_eq!(
            divs, 0,
            "TRICK-1-FORWARD ARM-q2 worker integer-domain gate FAIL: {} divergences",
            divs
        );
        assert!(
            max_relerr <= 5e-3,
            "TRICK-1-FORWARD ARM-q2 worker fp32 gate FAIL: max_relerr={:.3e} > 5e-3",
            max_relerr
        );
        eprintln!(
            "TRICK-1-FORWARD ARM-q2 worker fixture: PASS \
             ({} divergences, max_relerr {:.3e})",
            divs, max_relerr
        );
    }

    /// Scan a few shapes / seeds. All must hold.
    #[test]
    fn arm_q2_worker_multiple_shapes_byte_exact() {
        // n_embd = 1152 (Gemma3-1B), head_dim = 256, FF = 6912.
        let cases: &[(usize, usize, usize, u64, u64)] = &[
            (1,  1152, 256,  0x1111, 0x2222),
            (1,  256,  256,  0x3333, 0x4444),
            (1,  1152, 1152, 0x5555, 0x6666),
            (4,  1152, 256,  0x7777, 0x8888),
            (16, 1152, 256,  0x9999, 0xAAAA),  // ctx-16 batch
        ];
        for &(b, di, dout, sx, sw) in cases {
            let (divs, max_relerr) = arm_q2_worker_check(b, di, dout, sx, sw);
            assert_eq!(
                divs, 0,
                "shape ({b},{di},{dout}) sx={sx:#x} sw={sw:#x}: {divs} int divergences"
            );
            assert!(
                max_relerr <= 5e-3,
                "shape ({b},{di},{dout}) sx={sx:#x} sw={sw:#x}: max_relerr={max_relerr:.3e}"
            );
        }
    }

    /// Edge case: a single-element matmul (d_in=d_out=1). Verifies the
    /// degenerate path doesn't trip the per-tensor scale's div-by-zero guard.
    #[test]
    fn arm_q2_worker_degenerate_shape() {
        let (divs, _) = arm_q2_worker_check(1, 1, 1, 0xCAFE, 0xBABE);
        assert_eq!(divs, 0, "degenerate (1,1,1) shape: {divs} divergences");
    }

    /// Round-trip: pack an all-zero tensor, lift, recombine. Output must be
    /// exactly 0 — this catches Garner formula bugs that produce M-cycle
    /// aliasing on the zero residue.
    #[test]
    fn all_zero_tensor_recombines_to_zero() {
        let x_f32: Vec<f32> = vec![0.0; 256];
        let w_f32: Vec<f32> = vec![0.0; 256 * 16];
        let x_dpt = DualPrimeTensor::pack(&x_f32);
        let w_dpt = DualPrimeTensor::pack(&w_f32);
        let y_q1 = matmul_q_scalar_ref(0, 1, 256, 16,
                                       &x_dpt.q1_residues, &w_dpt.q1_residues);
        let y_q2 = matmul_q_scalar_ref(1, 1, 256, 16,
                                       &x_dpt.q2_residues, &w_dpt.q2_residues);
        let y_garner = garner_combine_q1_q2_signed(&y_q1, &y_q2);
        assert!(y_garner.iter().all(|&v| v == 0),
                "all-zero tensor: Garner output not all zero: {:?}", &y_garner[..8]);
    }

    /// Stat-style: across random seeds, the ARM-q2 path's fp32 max relerr
    /// stays well below the 5e-3 budget (typical 1e-6). Establishes the
    /// safety margin per CLOSURE-TRICK-1.md §2 (sub-ULP per element times
    /// sqrt(K)).
    #[test]
    fn arm_q2_worker_safety_margin() {
        let mut max_observed: f64 = 0.0;
        for seed in 0..8u64 {
            let (divs, relerr) = arm_q2_worker_check(
                1, 1152, 256,
                seed.wrapping_mul(0x9E37_79B9_7F4A_7C15),
                seed.wrapping_mul(0xBF58_476D_1CE4_E5B9),
            );
            assert_eq!(divs, 0);
            if relerr > max_observed { max_observed = relerr; }
        }
        // Per CLOSURE-TRICK-1.md §2, K=2048 produces ~1.9e-6 max relerr.
        // At K=1152 it should be similar or smaller. Generous bound: < 1e-3.
        assert!(
            max_observed < 1e-3,
            "ARM-q2 fp32 safety margin eroded: observed {:.3e} across 8 seeds",
            max_observed
        );
        eprintln!(
            "ARM-q2 worker fp32 safety margin across 8 seeds: max_relerr {:.3e} (budget 5e-3)",
            max_observed
        );
    }
}
