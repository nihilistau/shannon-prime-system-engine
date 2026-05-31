//! Sprint TRICK-1 Stage 1 — host-runnable Garner + Frobenius dual-lift smoke.
//!
//! Pure-Rust. NO silicon dispatch. Runs on Windows / Linux x86 without
//! Hexagon. Verifies the architectural plumbing END-TO-END at the
//! integer-arithmetic level BEFORE Stage 2's cDSP dispatch lands. If this
//! binary doesn't print PASS, the Stage 2/3/4 silicon dispatch CANNOT
//! produce correct Trick #1 output — a Stage 1 failure is an upstream
//! signal that the per-prime matmul or Garner formula has a bug that has
//! nothing to do with the silicon path.
//!
//! Gates run here:
//!   T_TRICK1_GARNER_BIT_EXACT_VS_HOST_CRT (Stage 1 form):
//!     pure-Rust Trick #1 path (frob-lift + per-prime scalar matmul + Garner)
//!     equals the int8-signed-reference matmul byte-for-byte.
//!   fp32-equivalence (PLAN §D-F path b): dequantized output matches fp32
//!     reference within 5e-3 relative-error budget.
//!
//! Output: one JSON line + summary. Exit 0 on PASS; non-zero on any failure.
//!
//! Build + run:
//!   cargo run -p sp-trick1 --bin sp_trick1_host_smoke

use sp_trick1::{
    gen_fp32_tensor, garner_combine_q1_q2_signed, matmul_f32_ref,
    matmul_int8_signed_ref, matmul_q_scalar_ref, DualPrimeTensor,
    dequantize_garner_output,
};

fn main() {
    eprintln!("=== Sprint TRICK-1 Stage 1 — host-only Garner + Frobenius dual-lift ===\n");

    // Sprint-spec test fixture shape: K=2048, M=N=256, b=1.
    let (batch, d_in, d_out) = (1usize, 2048usize, 256usize);
    eprintln!("Fixture: batch={batch} d_in={d_in} d_out={d_out}");
    eprintln!("Max int8 sum range: ± {} (≈ 2^{:.1})",
              d_in * 127 * 127,
              ((d_in * 127 * 127) as f64).log2());

    // Deterministic fp32 tensors (LCG seeded).
    let x_f32 = gen_fp32_tensor(0xDEAD_BEEF_CAFE_BABE, batch * d_in);
    let w_f32 = gen_fp32_tensor(0xFEED_FACE_BAAD_F00D, d_in * d_out);

    // Frobenius dual-lift.
    let x_dpt = DualPrimeTensor::pack(&x_f32);
    let w_dpt = DualPrimeTensor::pack(&w_f32);
    eprintln!("Pack: s_x={:.6}  s_w={:.6}", x_dpt.scale, w_dpt.scale);
    eprintln!("      |x_codes|={} (rng [{} .. {}])", x_dpt.codes.len(),
              *x_dpt.codes.iter().min().unwrap_or(&0),
              *x_dpt.codes.iter().max().unwrap_or(&0));
    eprintln!("      |w_codes|={} (rng [{} .. {}])", w_dpt.codes.len(),
              *w_dpt.codes.iter().min().unwrap_or(&0),
              *w_dpt.codes.iter().max().unwrap_or(&0));

    // ─── Trick #1 path: lift → per-prime scalar matmul → Garner ────────────
    let t0 = std::time::Instant::now();
    let y_q1 = matmul_q_scalar_ref(0, batch, d_in, d_out,
                                   &x_dpt.q1_residues, &w_dpt.q1_residues);
    let t1 = std::time::Instant::now();
    let y_q2 = matmul_q_scalar_ref(1, batch, d_in, d_out,
                                   &x_dpt.q2_residues, &w_dpt.q2_residues);
    let t2 = std::time::Instant::now();
    let y_garner = garner_combine_q1_q2_signed(&y_q1, &y_q2);
    let t3 = std::time::Instant::now();
    let y_real_trick1 = dequantize_garner_output(&y_garner, x_dpt.scale, w_dpt.scale);
    let t4 = std::time::Instant::now();

    eprintln!("\nTrick #1 path (host-Rust, scalar both primes):");
    eprintln!("  mod-q1 matmul:    {:>8} ms", t1.duration_since(t0).as_millis());
    eprintln!("  mod-q2 matmul:    {:>8} ms", t2.duration_since(t1).as_millis());
    eprintln!("  Garner combine:   {:>8} ms", t3.duration_since(t2).as_millis());
    eprintln!("  Dequantize:       {:>8} ms", t4.duration_since(t3).as_millis());
    eprintln!("  total:            {:>8} ms", t4.duration_since(t0).as_millis());

    // ─── Reference 1: signed-int8 matmul ──────────────────────────────────
    let r0 = std::time::Instant::now();
    let y_int_ref = matmul_int8_signed_ref(batch, d_in, d_out, &x_dpt.codes, &w_dpt.codes);
    let r1 = std::time::Instant::now();
    eprintln!("\nReference 1: signed-int8 matmul ({:>3} ms)", r1.duration_since(r0).as_millis());

    // T_TRICK1_GARNER_BIT_EXACT_VS_HOST_CRT (Stage 1 form)
    let mut divergences = 0usize;
    let mut max_abs_diff: i64 = 0;
    for (i, (&g, &r)) in y_garner.iter().zip(y_int_ref.iter()).enumerate() {
        let d = (g - r).abs();
        if d > max_abs_diff { max_abs_diff = d; }
        if g != r {
            divergences += 1;
            if divergences <= 5 {
                eprintln!("  divergence at i={i}: garner={g} ref={r} delta={}", g - r);
            }
        }
    }
    let gate_garner_pass = divergences == 0;

    // ─── Reference 2: fp32 matmul on dequantized Q8 ───────────────────────
    let f0 = std::time::Instant::now();
    let x_dq = x_dpt.dequantize();
    let w_dq = w_dpt.dequantize();
    let y_real_ref = matmul_f32_ref(batch, d_in, d_out, &x_dq, &w_dq);
    let f1 = std::time::Instant::now();
    eprintln!("Reference 2: fp32 matmul on dequantized Q8 ({:>3} ms)", f1.duration_since(f0).as_millis());

    // fp32-equivalence within budget
    let budget: f64 = 5e-3;
    let mut max_relerr: f64 = 0.0;
    let mut budget_violations = 0usize;
    for (i, (&t, &r)) in y_real_trick1.iter().zip(y_real_ref.iter()).enumerate() {
        let denom = (r as f64).abs().max(1e-6);
        let relerr = ((t as f64) - (r as f64)).abs() / denom;
        if relerr > max_relerr { max_relerr = relerr; }
        if relerr > budget {
            budget_violations += 1;
            if budget_violations <= 5 {
                eprintln!("  fp32 budget violation at i={i}: trick1={t} ref={r} relerr={relerr:.3e}");
            }
        }
    }
    let gate_fp32_pass = budget_violations == 0;

    // ─── Summary ──────────────────────────────────────────────────────────
    eprintln!("\n=== Gates ===");
    eprintln!("T_TRICK1_GARNER_BIT_EXACT_VS_HOST_CRT  (Stage 1):  {}",
              if gate_garner_pass { "PASS" } else { "FAIL" });
    eprintln!("  divergences:  {} / {}", divergences, y_int_ref.len());
    eprintln!("  max_abs_diff: {}", max_abs_diff);
    eprintln!("fp32-equivalence within budget (PLAN §D-F b):    {}",
              if gate_fp32_pass { "PASS" } else { "FAIL" });
    eprintln!("  max_relerr:        {:.3e}", max_relerr);
    eprintln!("  budget_violations: {} / {}", budget_violations, y_real_ref.len());
    eprintln!("  budget:            {:.3e}", budget);

    // JSON one-liner for log scraping.
    let stage1_pass = gate_garner_pass && gate_fp32_pass;
    println!(
        r#"{{"stage":1,"shape":{{"batch":{batch},"d_in":{d_in},"d_out":{d_out}}},"garner_pass":{garner},"garner_divergences":{div},"garner_max_abs_diff":{max_diff},"fp32_pass":{fp32},"fp32_max_relerr":{relerr:.6e},"fp32_violations":{violations},"fp32_budget":{budget:.6e},"s_x":{sx},"s_w":{sw},"stage1_pass":{pass}}}"#,
        batch=batch, d_in=d_in, d_out=d_out,
        garner=gate_garner_pass, div=divergences, max_diff=max_abs_diff,
        fp32=gate_fp32_pass, relerr=max_relerr, violations=budget_violations, budget=budget,
        sx=x_dpt.scale, sw=w_dpt.scale, pass=stage1_pass,
    );

    eprintln!("\n=== Stage 1 overall: {} ===",
              if stage1_pass { "PASS" } else { "FAIL" });

    std::process::exit(if stage1_pass { 0 } else { 1 });
}
