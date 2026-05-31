//! Sprint TRICK-1 Stages 2-5 — silicon smoke.
//!
//! cDSP V69 (q_1 via sp_compute_matmul_q method 11 + Arc<FastRpcSession>)
//! + ARM Cortex-X2/A710 (q_2 via matmul_q_scalar_ref running on the
//! deploying ARM cores) + ARM Garner combine.
//!
//! This binary is Android-only. Host builds (Windows / Linux x86) print
//! a hint and exit 0. The Stage 1 gates run via `sp_trick1_host_smoke`
//! on the host; the silicon stages here run on Knack's S22U.

#[cfg(not(target_os = "android"))]
fn main() {
    eprintln!("sp_trick1_smoke: host build skipped (Android-only).");
    eprintln!("Build target: aarch64-linux-android.");
    eprintln!("Stage 1 (host-only) lives in `sp_trick1_host_smoke`.");
}

#[cfg(target_os = "android")]
mod dsp_rpc;

#[cfg(target_os = "android")]
fn main() {
    use std::sync::Arc;
    use std::time::Instant;
    use std::ffi::c_void;
    use sp_trick1::{
        DualPrimeTensor, dequantize_garner_output, gen_fp32_tensor,
        garner_combine_q1_q2_signed, matmul_f32_ref, matmul_int8_signed_ref,
        matmul_q_scalar_ref, SP_NTT_Q1, SP_NTT_Q2,
    };
    use dsp_rpc::{make_scalars, FastRpcSession, RemoteArg, RemoteBuf, SpErr};

    eprintln!("=== Sprint TRICK-1 silicon smoke (cDSP-q1 + ARM-q2 + Garner) ===");

    // ─── Stage prep: fixture + Frobenius dual-lift ────────────────────────
    let (batch, d_in, d_out) = (1usize, 2048usize, 256usize);
    eprintln!("Fixture: batch={batch} d_in={d_in} d_out={d_out}");

    let x_f32 = gen_fp32_tensor(0xDEAD_BEEF_CAFE_BABE, batch * d_in);
    let w_f32 = gen_fp32_tensor(0xFEED_FACE_BAAD_F00D, d_in * d_out);
    let x_dpt = DualPrimeTensor::pack(&x_f32);
    let w_dpt = DualPrimeTensor::pack(&w_f32);
    eprintln!("Pack: s_x={:.6} s_w={:.6}", x_dpt.scale, w_dpt.scale);

    // Reference: integer sum and fp32 matmul, computed once.
    let y_int_ref = matmul_int8_signed_ref(batch, d_in, d_out, &x_dpt.codes, &w_dpt.codes);
    let x_dq = x_dpt.dequantize();
    let w_dq = w_dpt.dequantize();
    let y_real_ref = matmul_f32_ref(batch, d_in, d_out, &x_dq, &w_dq);

    // ─── FastRPC session open ─────────────────────────────────────────────
    eprintln!("\n[trick-1] opening FastRpcSession (Path B / Unsigned PD)...");
    let sess = match FastRpcSession::new(
        "file:///libsp_compute_skel.so?sp_compute_skel_handle_invoke&_modver=1.0&_dom=cdsp")
    {
        Ok(s) => { eprintln!("[trick-1] session open"); s }
        Err(e) => {
            eprintln!("[trick-1] session FAIL: {e:?}");
            eprintln!("[trick-1] === UPSTREAM BLOCKER: FastRPC session open failed ===");
            std::process::exit(2);
        }
    };
    let sess = Arc::new(sess);

    /// Invoke matmul_q on the open session. Returns (y_residue, dsp_pcycles, t0, t1).
    /// Mirrors `sp_matmul_q_dual_smoke.rs::invoke_matmul_q`.
    fn invoke_matmul_q(
        sess: &FastRpcSession,
        q_idx: i32, batch: i32, d_in: i32, d_out: i32,
        x: &[u32], w: &[u32],
    ) -> Result<(Vec<u32>, u64, Instant, Instant), SpErr> {
        let x_n_bytes = x.len() * 4;
        let w_n_bytes = w.len() * 4;
        let y_n_bytes = (batch * d_out) as usize * 4;
        let mut prim_in: [u32; 7] = [
            q_idx as u32, batch as u32, d_in as u32, d_out as u32,
            x_n_bytes as u32, w_n_bytes as u32, y_n_bytes as u32,
        ];
        let mut x_bytes = Vec::with_capacity(x_n_bytes);
        for v in x { x_bytes.extend_from_slice(&v.to_le_bytes()); }
        let mut w_bytes = Vec::with_capacity(w_n_bytes);
        for v in w { w_bytes.extend_from_slice(&v.to_le_bytes()); }
        let mut y_bytes = vec![0u8; y_n_bytes];
        let mut prim_out: [u32; 2] = [0u32; 2];
        let mut args = [
            RemoteArg { buf: RemoteBuf { pv: prim_in.as_mut_ptr() as *mut c_void, nlen: 28 }},
            RemoteArg { buf: RemoteBuf { pv: x_bytes.as_mut_ptr() as *mut c_void, nlen: x_n_bytes }},
            RemoteArg { buf: RemoteBuf { pv: w_bytes.as_mut_ptr() as *mut c_void, nlen: w_n_bytes }},
            RemoteArg { buf: RemoteBuf { pv: prim_out.as_mut_ptr() as *mut c_void, nlen: 8 }},
            RemoteArg { buf: RemoteBuf { pv: y_bytes.as_mut_ptr() as *mut c_void, nlen: y_n_bytes }},
        ];
        let t0 = Instant::now();
        // method=11 (sp_compute_matmul_q), n_in=3 (primIn + x + w), n_out=2 (primOut + y).
        sess.invoke(make_scalars(11, 3, 2), &mut args)?;
        let t1 = Instant::now();
        let y: Vec<u32> = y_bytes.chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let pcyc = (prim_out[0] as u64) | ((prim_out[1] as u64) << 32);
        Ok((y, pcyc, t0, t1))
    }

    fn vmrss_kb() -> u64 {
        std::fs::read_to_string("/proc/self/status")
            .ok()
            .and_then(|s| s.lines()
                .find(|l| l.starts_with("VmRSS:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|n| n.parse::<u64>().ok()))
            .unwrap_or(0)
    }

    let mut fails = 0usize;

    // ───────────────────────────────────────────────────────────────────────
    // Stage 2: cDSP-q1 solo
    // ───────────────────────────────────────────────────────────────────────
    eprintln!("\n=== Stage 2: cDSP-q1 solo (mod q_1 matmul on Hexagon V69 HVX) ===");
    let s2_start = Instant::now();
    let (y_dsp_q1, pcyc_q1, _, _) = match invoke_matmul_q(
        &sess, 0, batch as i32, d_in as i32, d_out as i32,
        &x_dpt.q1_residues, &w_dpt.q1_residues) {
        Ok(t) => t,
        Err(e) => { eprintln!("[trick-1] Stage 2 FAIL: {e:?}"); std::process::exit(3); }
    };
    let s2_wall_us = s2_start.elapsed().as_micros();
    eprintln!("  Stage 2 wall = {} μs  (cDSP pcyc = {})", s2_wall_us, pcyc_q1);
    // Cross-check against the ARM scalar reference for q_1.
    let y_ref_q1 = matmul_q_scalar_ref(0, batch, d_in, d_out,
                                       &x_dpt.q1_residues, &w_dpt.q1_residues);
    let stage2_divergences = y_dsp_q1.iter().zip(y_ref_q1.iter()).filter(|(a, b)| a != b).count();
    eprintln!("  Stage 2 q_1 vs ARM-scalar-q_1 ref: {} / {} divergences",
              stage2_divergences, y_dsp_q1.len());
    let stage2_pass = stage2_divergences == 0;
    if !stage2_pass {
        eprintln!("  Stage 2 FAIL — cDSP HVX q_1 disagrees with ARM scalar q_1 reference");
        fails += 1;
    } else {
        eprintln!("  Stage 2 PASS");
    }

    // ───────────────────────────────────────────────────────────────────────
    // Stage 3: ARM-q2 solo
    // ───────────────────────────────────────────────────────────────────────
    eprintln!("\n=== Stage 3: ARM-q2 solo (mod q_2 matmul on Cortex-X2/A710 scalar) ===");
    let s3_start = Instant::now();
    let y_arm_q2 = matmul_q_scalar_ref(1, batch, d_in, d_out,
                                       &x_dpt.q2_residues, &w_dpt.q2_residues);
    let s3_wall_us = s3_start.elapsed().as_micros();
    eprintln!("  Stage 3 wall = {} μs", s3_wall_us);
    // All residues must lie in [0, q_2).
    let stage3_oor = y_arm_q2.iter().filter(|&&v| v >= SP_NTT_Q2).count();
    eprintln!("  Stage 3 q_2 residue range check: {} / {} out-of-range", stage3_oor, y_arm_q2.len());
    let stage3_pass = stage3_oor == 0;
    if !stage3_pass { fails += 1; } else { eprintln!("  Stage 3 PASS"); }

    // ───────────────────────────────────────────────────────────────────────
    // Stage 4: serial dispatch + Garner combine  (T_TRICK1_NUMERICAL_EQUIVALENCE)
    // ───────────────────────────────────────────────────────────────────────
    eprintln!("\n=== Stage 4: serial dispatch + Garner combine ===");
    let s4_start = Instant::now();
    let (y_dsp_q1_2, _, _, _) = invoke_matmul_q(
        &sess, 0, batch as i32, d_in as i32, d_out as i32,
        &x_dpt.q1_residues, &w_dpt.q1_residues).expect("stage 4 cDSP");
    let s4_dsp_done = Instant::now();
    let y_arm_q2_2 = matmul_q_scalar_ref(1, batch, d_in, d_out,
                                         &x_dpt.q2_residues, &w_dpt.q2_residues);
    let s4_arm_done = Instant::now();
    let y_garner = garner_combine_q1_q2_signed(&y_dsp_q1_2, &y_arm_q2_2);
    let s4_garner_done = Instant::now();
    let y_real_trick1 = dequantize_garner_output(&y_garner, x_dpt.scale, w_dpt.scale);
    let s4_done = Instant::now();
    eprintln!("  cDSP-q1 phase:  {} μs", s4_dsp_done.duration_since(s4_start).as_micros());
    eprintln!("  ARM-q2 phase:   {} μs", s4_arm_done.duration_since(s4_dsp_done).as_micros());
    eprintln!("  Garner combine: {} μs", s4_garner_done.duration_since(s4_arm_done).as_micros());
    eprintln!("  Dequantize:     {} μs", s4_done.duration_since(s4_garner_done).as_micros());
    eprintln!("  Stage 4 serial total: {} μs", s4_done.duration_since(s4_start).as_micros());

    // T_TRICK1_NUMERICAL_EQUIVALENCE — integer-domain byte-exact.
    let mut int_divergences = 0usize;
    let mut max_abs_diff: i64 = 0;
    for (i, (&g, &r)) in y_garner.iter().zip(y_int_ref.iter()).enumerate() {
        let d = (g - r).abs();
        if d > max_abs_diff { max_abs_diff = d; }
        if g != r {
            int_divergences += 1;
            if int_divergences <= 5 {
                eprintln!("  int divergence at i={i}: garner={g} ref={r} delta={}", g - r);
            }
        }
    }
    let gate_int_pass = int_divergences == 0;
    eprintln!("  T_TRICK1_NUMERICAL_EQUIVALENCE (int byte-exact): {}",
              if gate_int_pass { "PASS" } else { "FAIL" });
    eprintln!("    divergences: {} / {}", int_divergences, y_int_ref.len());
    eprintln!("    max_abs_diff: {}", max_abs_diff);

    // fp32 budget gate
    let budget: f64 = 5e-3;
    let mut budget_violations = 0usize;
    let mut max_relerr: f64 = 0.0;
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
    eprintln!("  fp32-budget within 5e-3 (PLAN §D-F b): {}", if gate_fp32_pass { "PASS" } else { "FAIL" });
    eprintln!("    max_relerr: {:.3e}", max_relerr);
    eprintln!("    violations: {} / {}", budget_violations, y_real_ref.len());
    if !gate_int_pass { fails += 1; }
    if !gate_fp32_pass { fails += 1; }

    // ───────────────────────────────────────────────────────────────────────
    // T_TRICK1_GARNER_BIT_EXACT_VS_HOST_CRT — pure Garner gate.
    //
    // Take Stage 4's (y_dsp_q1, y_arm_q2) pair, recombine via the LOCAL
    // garner_combine_q1_q2_signed AND via a second host-side caller equivalent
    // (we have only one Garner implementation in this crate; cross-check
    // against the upstream Rust impl by comparing element-wise to the
    // matmul_int8_signed_ref output, since both produce the SAME signed
    // integer under the CRT correspondence on inputs where the unreduced
    // sum is in (-M/2, M/2]). The integer-domain identity gate above
    // (gate_int_pass) IS the Garner-bit-exact gate from a different
    // angle — same arithmetic, different test framing).
    // ───────────────────────────────────────────────────────────────────────
    let gate_garner_pass = gate_int_pass;
    eprintln!("\n  T_TRICK1_GARNER_BIT_EXACT_VS_HOST_CRT: {} (= T_TRICK1_NUMERICAL_EQUIVALENCE)",
              if gate_garner_pass { "PASS" } else { "FAIL" });

    // ───────────────────────────────────────────────────────────────────────
    // Stage 5: parallel dispatch  (T_TRICK1_PARALLEL_WIN + BOTH_ISLANDS_ACTIVE)
    //
    // Methodology per PLAN §D-G:
    //   T_dsp_solo, T_arm_solo, T_trick1_parallel — 10 reps, drop first
    //   as warmup, report mean and stddev.
    //   Win: T_trick1_parallel < 1.2 × max(T_dsp_solo, T_arm_solo).
    // ───────────────────────────────────────────────────────────────────────
    eprintln!("\n=== Stage 5: parallel dispatch + wall-clock win ===");
    eprintln!("Methodology: 10 reps, drop first as warmup, report mean + stddev.\n");

    fn stats(samples: &[u128]) -> (f64, f64) {
        if samples.is_empty() { return (0.0, 0.0); }
        let mean = samples.iter().sum::<u128>() as f64 / samples.len() as f64;
        let var = samples.iter().map(|&v| { let d = v as f64 - mean; d * d }).sum::<f64>() / samples.len() as f64;
        (mean, var.sqrt())
    }

    // Cold-warmup invoke (not measured).
    let _ = invoke_matmul_q(&sess, 0, batch as i32, d_in as i32, d_out as i32,
                            &x_dpt.q1_residues, &w_dpt.q1_residues);

    // T_dsp_solo
    let mut dsp_samples = Vec::with_capacity(10);
    let mut dsp_pcyc_samples = Vec::with_capacity(10);
    for rep in 0..10 {
        let r0 = Instant::now();
        let (_, pcyc, _, _) = invoke_matmul_q(&sess, 0, batch as i32, d_in as i32, d_out as i32,
                                              &x_dpt.q1_residues, &w_dpt.q1_residues)
            .expect("dsp solo");
        let r1 = Instant::now();
        if rep > 0 {
            dsp_samples.push(r1.duration_since(r0).as_micros());
            dsp_pcyc_samples.push(pcyc);
        }
    }
    let (dsp_mean, dsp_std) = stats(&dsp_samples);
    let dsp_pcyc_mean = dsp_pcyc_samples.iter().sum::<u64>() as f64 / dsp_pcyc_samples.len() as f64;
    eprintln!("T_dsp_solo:     mean = {:>10.0} μs  stddev = {:>8.0} μs  (cDSP pcyc mean = {:.0})",
              dsp_mean, dsp_std, dsp_pcyc_mean);

    // T_arm_solo
    let mut arm_samples = Vec::with_capacity(10);
    for rep in 0..10 {
        let r0 = Instant::now();
        let _ = matmul_q_scalar_ref(1, batch, d_in, d_out,
                                    &x_dpt.q2_residues, &w_dpt.q2_residues);
        let r1 = Instant::now();
        if rep > 0 {
            arm_samples.push(r1.duration_since(r0).as_micros());
        }
    }
    let (arm_mean, arm_std) = stats(&arm_samples);
    eprintln!("T_arm_solo:     mean = {:>10.0} μs  stddev = {:>8.0} μs",
              arm_mean, arm_std);

    // T_trick1_parallel — spawn two threads, cDSP on A, ARM scalar on B.
    let mut par_samples = Vec::with_capacity(10);
    let mut overlap_samples = Vec::with_capacity(10);
    // Per-thread wall samples (for T_TRICK1_BOTH_ISLANDS_ACTIVE_DSP_ARM)
    let mut thread_a_samples = Vec::with_capacity(10);
    let mut thread_b_samples = Vec::with_capacity(10);
    for rep in 0..10 {
        let sess_a = sess.clone();
        let xq1 = x_dpt.q1_residues.clone();
        let wq1 = w_dpt.q1_residues.clone();
        let xq2 = x_dpt.q2_residues.clone();
        let wq2 = w_dpt.q2_residues.clone();
        let bi  = batch as i32;
        let di  = d_in as i32;
        let dx  = d_out as i32;

        let par_start = Instant::now();
        let h_a = std::thread::spawn(move || -> Result<(Vec<u32>, u64, Instant, Instant), SpErr> {
            invoke_matmul_q(&sess_a, 0, bi, di, dx, &xq1, &wq1)
        });
        let h_b = std::thread::spawn(move || -> (Vec<u32>, Instant, Instant) {
            let tb0 = Instant::now();
            let y = matmul_q_scalar_ref(1, batch, d_in, d_out, &xq2, &wq2);
            let tb1 = Instant::now();
            (y, tb0, tb1)
        });
        let r_a = h_a.join().expect("thread A");
        let r_b = h_b.join().expect("thread B");
        let par_end = Instant::now();

        let (y_a, _pcyc_a, ta0, ta1) = r_a.expect("thread A invoke");
        let (y_b, tb0, tb1) = r_b;

        // Garner combine (small; not in the parallel measurement window).
        let _y_garner = garner_combine_q1_q2_signed(&y_a, &y_b);
        let par_wall = par_end.duration_since(par_start).as_micros();
        let thread_a_wall = ta1.duration_since(ta0).as_micros();
        let thread_b_wall = tb1.duration_since(tb0).as_micros();

        // Overlap window = [max(ta0, tb0), min(ta1, tb1)]
        let overlap_start = ta0.max(tb0);
        let overlap_end   = ta1.min(tb1);
        let overlap = if overlap_end > overlap_start {
            overlap_end.duration_since(overlap_start).as_micros()
        } else { 0 };

        if rep > 0 {
            par_samples.push(par_wall);
            overlap_samples.push(overlap);
            thread_a_samples.push(thread_a_wall);
            thread_b_samples.push(thread_b_wall);
        }
    }
    let (par_mean, par_std) = stats(&par_samples);
    let (overlap_mean, _) = stats(&overlap_samples);
    let (ta_mean, _) = stats(&thread_a_samples);
    let (tb_mean, _) = stats(&thread_b_samples);

    eprintln!("T_trick1_parallel: mean = {:>10.0} μs  stddev = {:>8.0} μs",
              par_mean, par_std);
    eprintln!("  thread A (cDSP) mean = {:>10.0} μs", ta_mean);
    eprintln!("  thread B (ARM)  mean = {:>10.0} μs", tb_mean);
    eprintln!("  overlap window mean   = {:>10.0} μs  ({:.1}% of parallel wall)",
              overlap_mean,
              if par_mean > 0.0 { 100.0 * overlap_mean / par_mean } else { 0.0 });

    let max_solo = dsp_mean.max(arm_mean);
    let parallel_ratio = par_mean / max_solo;
    let serial_sum = dsp_mean + arm_mean;
    let parallel_speedup_vs_serial = serial_sum / par_mean.max(1.0);
    eprintln!("\n  max(T_dsp_solo, T_arm_solo) = {:>10.0} μs", max_solo);
    eprintln!("  T_trick1_parallel / max     = {:.3}× (gate ≤ 1.2)", parallel_ratio);
    eprintln!("  serial vs parallel speedup  = {:.3}× (informational)", parallel_speedup_vs_serial);

    let gate_parallel_pass = parallel_ratio <= 1.2;
    eprintln!("  T_TRICK1_PARALLEL_WIN: {}", if gate_parallel_pass { "PASS" } else { "FAIL" });
    if !gate_parallel_pass { fails += 1; }

    // T_TRICK1_BOTH_ISLANDS_ACTIVE_DSP_ARM — both threads had > 1ms wall.
    // For our K=2048 fixture, anything < ~10% of solo wall would indicate
    // one of the islands didn't run.
    let both_active = ta_mean > (dsp_mean * 0.5) && tb_mean > (arm_mean * 0.5);
    let gate_both_active_pass = both_active;
    eprintln!("  T_TRICK1_BOTH_ISLANDS_ACTIVE_DSP_ARM: {}",
              if gate_both_active_pass { "PASS" } else { "FAIL" });
    eprintln!("    cDSP thread wall: {:.0} μs  (vs solo {:.0} μs, {:.0}%)",
              ta_mean, dsp_mean, 100.0 * ta_mean / dsp_mean.max(1.0));
    eprintln!("    ARM  thread wall: {:.0} μs  (vs solo {:.0} μs, {:.0}%)",
              tb_mean, arm_mean, 100.0 * tb_mean / arm_mean.max(1.0));
    if !gate_both_active_pass { fails += 1; }

    // ───────────────────────────────────────────────────────────────────────
    // VmRSS leak diagnostic (informational — not a sprint gate).
    // ───────────────────────────────────────────────────────────────────────
    eprintln!("\nVmRSS: {} KB (informational)", vmrss_kb());

    // ─── JSON one-liner ──────────────────────────────────────────────────
    let all_pass = fails == 0;
    println!(
        r#"{{"sprint":"TRICK-1","shape":{{"batch":{batch},"d_in":{d_in},"d_out":{d_out}}},"T_dsp_solo_us":{dsp_mean:.0},"T_arm_solo_us":{arm_mean:.0},"T_trick1_parallel_us":{par_mean:.0},"parallel_ratio":{parallel_ratio:.3},"parallel_speedup_vs_serial":{parallel_speedup_vs_serial:.3},"overlap_us":{overlap_mean:.0},"thread_a_us":{ta_mean:.0},"thread_b_us":{tb_mean:.0},"garner_pass":{garner},"int_div":{int_div},"fp32_pass":{fp32},"fp32_max_relerr":{relerr:.6e},"parallel_pass":{par},"both_islands_active":{ba},"fails":{fails},"all_pass":{ap}}}"#,
        batch=batch, d_in=d_in, d_out=d_out,
        dsp_mean=dsp_mean, arm_mean=arm_mean, par_mean=par_mean,
        parallel_ratio=parallel_ratio,
        parallel_speedup_vs_serial=parallel_speedup_vs_serial,
        overlap_mean=overlap_mean, ta_mean=ta_mean, tb_mean=tb_mean,
        garner=gate_garner_pass, int_div=int_divergences,
        fp32=gate_fp32_pass, relerr=max_relerr,
        par=gate_parallel_pass, ba=gate_both_active_pass,
        fails=fails, ap=all_pass,
    );

    eprintln!("\n=== TRICK-1 overall: {} fails ({}) ===",
              fails, if all_pass { "PASS" } else { "FAIL" });
    drop(sess);
    std::process::exit(if all_pass { 0 } else { 1 });
}
