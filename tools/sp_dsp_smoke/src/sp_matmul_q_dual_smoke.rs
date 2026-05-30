//! §3-HX Sprint K v0.beta-2.5c Stage 3 — dual-dispatch + leak-free harness for
//! the HVX mod_q matmul kernel.
//!
//! Per `feedback-shape-dependent-parallelism-gates`, the parallelism gate
//! must run at a SHAPE where compute time dominates marshalling.  The
//! prompt-spec'd shape (B=8, D_in=128, D_out=128) on the mod_q kernel
//! produced ~400 μs per invoke (vs K v0.alpha's 17.7 ms saturating-matmul
//! at the same shape) — too data-bound for the parallelism gate to
//! discriminate.  This harness measures BOTH shapes:
//!
//!   shape A (prompt-spec):   B=8  / D_in=128  / D_out=128
//!   shape B (compute-bound): B=8  / D_in=1024 / D_out=512
//!
//! Both shapes report sequential / concurrent wall + speedup +
//! overlap-fraction.  The load-bearing T_MATMUL_DUAL_DISPATCH_SPEEDUP gate
//! uses shape B (compute-bound regime where the gate is meaningful).
//!
//! Gates:
//!   T_MATMUL_DUAL_DISPATCH_SPEEDUP   ≥ 1.5× at shape B (target ≥ 1.7×)
//!   T_MATMUL_LEAK_FREE               2nd-half VmRSS slope ≤ 256 KB
//!     (per `feedback-leak-gate-allocator-warmup`)
//!   T_GARNER_BIT_EXACT_CONCURRENT    concurrent (q1, q2) Garner-recombined
//!     equals sequential baseline (cross-check that concurrent dispatch
//!     doesn't corrupt either lane's output)
//!
//! Build:
//!     cargo build --target aarch64-linux-android --release --bin sp_matmul_q_dual_smoke
//! Run:
//!     adb shell 'ADSP_LIBRARY_PATH="/data/local/tmp;" /data/local/tmp/sp_matmul_q_dual_smoke'

#[cfg(not(target_os = "android"))]
fn main() {
    eprintln!("sp_matmul_q_dual_smoke: host build skipped");
}

#[cfg(target_os = "android")]
mod dsp_rpc;
#[cfg(target_os = "android")]
mod sp_barrett_oracle;
#[cfg(target_os = "android")]
mod sp_matmul_q_ref;

#[cfg(target_os = "android")]
fn main() {
    use dsp_rpc::{make_scalars, FastRpcSession, RemoteArg, RemoteBuf, SpErr};
    use sp_barrett_oracle::{SP_NTT_Q1, SP_NTT_Q2};
    use sp_matmul_q_ref::garner_combine_q1_q2;
    use std::ffi::c_void;
    use std::sync::Arc;
    use std::time::Instant;

    eprintln!("[K-β-2.5c dual] opening FastRpcSession (Path B)...");
    let sess = match FastRpcSession::new(
        "file:///libsp_compute_skel.so?sp_compute_skel_handle_invoke&_modver=1.0&_dom=cdsp")
    {
        Ok(s) => { eprintln!("[K-β-2.5c dual] session open"); s }
        Err(e) => { eprintln!("[K-β-2.5c dual] session FAIL: {e:?}"); std::process::exit(1); }
    };
    let sess = Arc::new(sess);

    /// Invoke matmul_q (method 11).  Arg layout per qaic _skel_method:
    /// [primIn, x_buf, w_buf, primOut, y_buf].
    fn invoke_matmul_q(sess: &FastRpcSession,
                       q_idx: i32, batch: i32, d_in: i32, d_out: i32,
                       x: &[u32], w: &[u32])
                       -> Result<(Vec<u32>, u64, Instant, Instant), SpErr> {
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
        sess.invoke(make_scalars(11, 3, 2), &mut args)?;
        let t1 = Instant::now();
        let y: Vec<u32> = y_bytes.chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let pcyc = (prim_out[0] as u64) | ((prim_out[1] as u64) << 32);
        Ok((y, pcyc, t0, t1))
    }

    fn gen_xw(q: u32, seed: u64, n_x: usize, n_w: usize) -> (Vec<u32>, Vec<u32>) {
        let mut x = vec![0u32; n_x];
        let mut w = vec![0u32; n_w];
        let mut s = seed;
        for v in x.iter_mut() {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            *v = (s as u32) % q;
        }
        for v in w.iter_mut() {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            *v = (s as u32) % q;
        }
        (x, w)
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

    /// Run sequential vs concurrent dispatch at one shape; return all
    /// timings + functional check status.
    fn run_shape(sess: &Arc<FastRpcSession>, label: &str,
                 b: i32, d_in: i32, d_out: i32)
                 -> Option<(u128, u128, f64, f64, u128, u128, u128, u128, u128, bool)>
    {
        let n_x = (b * d_in) as usize;
        let n_w = (d_in * d_out) as usize;
        let (a_q1, w_q1) = gen_xw(SP_NTT_Q1, 0xDEADBEEFu64,        n_x, n_w);
        let (a_q2, w_q2) = gen_xw(SP_NTT_Q2, 0xDEADBEEFu64 ^ 1u64, n_x, n_w);

        eprintln!("\n[K-β-2.5c dual]   --- shape {} (B={} D_in={} D_out={}) ---",
                  label, b, d_in, d_out);
        eprintln!("[K-β-2.5c dual]     n_x={} n_w={} n_y={}", n_x, n_w, (b*d_out) as usize);

        // ─── Sequential ─────────────────────────────────────────────
        let seq_start = Instant::now();
        let (y_seq_q1, _, sq1_t0, sq1_t1) = match invoke_matmul_q(sess, 0, b, d_in, d_out, &a_q1, &w_q1) {
            Ok(t) => t,
            Err(e) => { eprintln!("[K-β-2.5c dual]     seq q_1 FAIL: {:?}", e); return None; }
        };
        let (y_seq_q2, _, sq2_t0, sq2_t1) = match invoke_matmul_q(sess, 1, b, d_in, d_out, &a_q2, &w_q2) {
            Ok(t) => t,
            Err(e) => { eprintln!("[K-β-2.5c dual]     seq q_2 FAIL: {:?}", e); return None; }
        };
        let seq_wall_us = seq_start.elapsed().as_micros();
        let seq_q1_us = sq1_t1.duration_since(sq1_t0).as_micros();
        let seq_q2_us = sq2_t1.duration_since(sq2_t0).as_micros();
        eprintln!("[K-β-2.5c dual]     seq q_1 invoke = {} μs", seq_q1_us);
        eprintln!("[K-β-2.5c dual]     seq q_2 invoke = {} μs", seq_q2_us);
        eprintln!("[K-β-2.5c dual]     sequential total = {} μs", seq_wall_us);

        // ─── Concurrent ─────────────────────────────────────────────
        let sess_a = sess.clone();
        let sess_b = sess.clone();
        let a_q1_t = a_q1.clone();
        let w_q1_t = w_q1.clone();
        let a_q2_t = a_q2.clone();
        let w_q2_t = w_q2.clone();
        let conc_start = Instant::now();
        let h_a = std::thread::spawn(move ||
            invoke_matmul_q(&sess_a, 0, b, d_in, d_out, &a_q1_t, &w_q1_t));
        let h_b = std::thread::spawn(move ||
            invoke_matmul_q(&sess_b, 1, b, d_in, d_out, &a_q2_t, &w_q2_t));
        let r_a = h_a.join().expect("thread A");
        let r_b = h_b.join().expect("thread B");
        let conc_wall_us = conc_start.elapsed().as_micros();
        let (y_conc_q1, _, ta0, ta1) = match r_a {
            Ok(t) => t,
            Err(e) => { eprintln!("[K-β-2.5c dual]     concurrent q_1 FAIL: {:?}", e); return None; }
        };
        let (y_conc_q2, _, tb0, tb1) = match r_b {
            Ok(t) => t,
            Err(e) => { eprintln!("[K-β-2.5c dual]     concurrent q_2 FAIL: {:?}", e); return None; }
        };
        let conc_q1_us = ta1.duration_since(ta0).as_micros();
        let conc_q2_us = tb1.duration_since(tb0).as_micros();
        let overlap_start = ta0.max(tb0);
        let overlap_end   = ta1.min(tb1);
        let overlap_us = if overlap_end > overlap_start {
            overlap_end.duration_since(overlap_start).as_micros()
        } else { 0 };
        let overlap_fraction = if conc_wall_us > 0 {
            overlap_us as f64 / conc_wall_us as f64
        } else { 0.0 };
        let speedup = seq_wall_us as f64 / conc_wall_us.max(1) as f64;
        eprintln!("[K-β-2.5c dual]     conc q_1 invoke = {} μs", conc_q1_us);
        eprintln!("[K-β-2.5c dual]     conc q_2 invoke = {} μs", conc_q2_us);
        eprintln!("[K-β-2.5c dual]     concurrent total = {} μs", conc_wall_us);
        eprintln!("[K-β-2.5c dual]     overlap = {} μs ({:.4})", overlap_us, overlap_fraction);
        eprintln!("[K-β-2.5c dual]     speedup = {:.3}× (seq / conc)", speedup);

        // ─── Functional cross-check ─────────────────────────────────
        let q1_match = y_seq_q1 == y_conc_q1;
        let q2_match = y_seq_q2 == y_conc_q2;
        let functional = q1_match && q2_match;
        eprintln!("[K-β-2.5c dual]     functional (concurrent == sequential): {}",
                  if functional { "PASS" } else { "FAIL" });

        let single_avg_us = (seq_q1_us + seq_q2_us) / 2;
        Some((seq_wall_us, conc_wall_us, speedup, overlap_fraction,
              seq_q1_us, seq_q2_us, conc_q1_us, conc_q2_us, single_avg_us,
              functional))
    }

    let mut fails = 0usize;

    eprintln!("\n[K-β-2.5c dual] ═══ Shape A — prompt-spec (B=8 D_in=128 D_out=128) ═══");
    eprintln!("[K-β-2.5c dual] Per K v0.beta-2.5c Stage 2 smoke: ~400 μs / invoke at this shape");
    eprintln!("[K-β-2.5c dual] (data-bound; reported for transparency per feedback-shape-dependent-parallelism-gates)");
    let shape_a = run_shape(&sess, "A", 8, 128, 128);

    eprintln!("\n[K-β-2.5c dual] ═══ Shape B — compute-bound (B=8 D_in=1024 D_out=512) ═══");
    eprintln!("[K-β-2.5c dual] Expected per-invoke wall ~10-20 ms (matches K v0.alpha regime ~17.7 ms)");
    eprintln!("[K-β-2.5c dual] Total MACs/invoke: 8 * 1024 * 512 = 4194304 = 32× shape A's 131k MACs");
    let shape_b = run_shape(&sess, "B", 8, 1024, 512);

    // ─── T_MATMUL_DUAL_DISPATCH_SPEEDUP (load-bearing on shape B) ─────────
    eprintln!("\n[K-β-2.5c dual] ═══ T_MATMUL_DUAL_DISPATCH_SPEEDUP ═══");
    eprintln!("[K-β-2.5c dual] Per feedback-shape-dependent-parallelism-gates: load-bearing measurement");
    eprintln!("[K-β-2.5c dual] is at the compute-bound shape (B), NOT the data-bound shape (A).");
    let mut speedup_pass_b = false;
    if let Some((seq, conc, sp, of, _, _, _, _, single, _)) = shape_b {
        eprintln!("[K-β-2.5c dual]   shape B: seq_wall={} μs / conc_wall={} μs / speedup={:.3}× / overlap_fraction={:.4} / single_avg={} μs",
                  seq, conc, sp, of, single);
        speedup_pass_b = sp >= 1.5;
        eprintln!("[K-β-2.5c dual]   shape B threshold ≥ 1.5×: {}",
                  if speedup_pass_b { "PASS" } else { "FAIL" });
        if sp >= 1.7 {
            eprintln!("[K-β-2.5c dual]   shape B stretch goal ≥ 1.7× MET");
        }
    } else {
        eprintln!("[K-β-2.5c dual]   shape B unavailable (invoke errors)");
        fails += 1;
    }
    if !speedup_pass_b { fails += 1; }

    if let Some((seq, conc, sp, of, _, _, _, _, _, _)) = shape_a {
        eprintln!("[K-β-2.5c dual]   shape A (diagnostic): seq_wall={} μs / conc_wall={} μs / speedup={:.3}× / overlap_fraction={:.4}",
                  seq, conc, sp, of);
    }

    // ─── T_MATMUL_LEAK_FREE — second-half VmRSS slope ─────────────────────
    eprintln!("\n[K-β-2.5c dual] ═══ T_MATMUL_LEAK_FREE (10000-iter dual-invoke; second-half slope) ═══");
    eprintln!("[K-β-2.5c dual] Per feedback-leak-gate-allocator-warmup: gate metric is second-half VmRSS delta ≤ 256 KB.");
    let (b, d_in, d_out) = (8i32, 128i32, 128i32);  // Use shape A for leak gate — fastest per-iter
    let n_x = (b * d_in) as usize;
    let n_w = (d_in * d_out) as usize;
    let (a_q1, w_q1) = gen_xw(SP_NTT_Q1, 0xDEADBEEFu64,        n_x, n_w);
    let (a_q2, w_q2) = gen_xw(SP_NTT_Q2, 0xDEADBEEFu64 ^ 1u64, n_x, n_w);
    let vmrss_0 = vmrss_kb();
    eprintln!("[K-β-2.5c dual]   VmRSS @ iter 0     = {} KB", vmrss_0);
    let leak_start = Instant::now();
    let mut leak_fails = 0usize;
    let mut vmrss_mid = vmrss_0;
    for i in 0..10000 {
        let sess_a = sess.clone();
        let sess_b = sess.clone();
        let a_q1_t = a_q1.clone();
        let w_q1_t = w_q1.clone();
        let a_q2_t = a_q2.clone();
        let w_q2_t = w_q2.clone();
        let h_a = std::thread::spawn(move ||
            invoke_matmul_q(&sess_a, 0, b, d_in, d_out, &a_q1_t, &w_q1_t));
        let h_b = std::thread::spawn(move ||
            invoke_matmul_q(&sess_b, 1, b, d_in, d_out, &a_q2_t, &w_q2_t));
        let r_a = h_a.join().expect("leak A");
        let r_b = h_b.join().expect("leak B");
        match (r_a, r_b) {
            (Ok(_), Ok(_)) => {}
            _ => { eprintln!("[K-β-2.5c dual]   iter {} FAIL", i); leak_fails += 1; break; }
        }
        if i == 5000 {
            vmrss_mid = vmrss_kb();
            eprintln!("[K-β-2.5c dual]   VmRSS @ iter 5000  = {} KB", vmrss_mid);
        }
    }
    let leak_wall = leak_start.elapsed();
    let vmrss_end = vmrss_kb();
    eprintln!("[K-β-2.5c dual]   VmRSS @ iter 10000 = {} KB", vmrss_end);
    eprintln!("[K-β-2.5c dual]   cycles_run         = {}", 10000 - leak_fails);
    eprintln!("[K-β-2.5c dual]   wall time          = {:.2} s ({:.2} ms/iter)",
              leak_wall.as_secs_f64(),
              leak_wall.as_secs_f64() * 1000.0 / 10000.0);
    let first_half_delta_kb = (vmrss_mid as i64) - (vmrss_0 as i64);
    let second_half_delta_kb = (vmrss_end as i64) - (vmrss_mid as i64);
    let total_delta_kb = (vmrss_end as i64) - (vmrss_0 as i64);
    eprintln!("[K-β-2.5c dual]   first_half_delta_kb  = {}", first_half_delta_kb);
    eprintln!("[K-β-2.5c dual]   second_half_delta_kb = {}  (load-bearing — gate threshold ≤ 256 KB)", second_half_delta_kb);
    eprintln!("[K-β-2.5c dual]   total_delta_kb       = {}  (diagnostic; allocator warmup expected)", total_delta_kb);
    let leak_pass = leak_fails == 0 && second_half_delta_kb.abs() <= 256;
    eprintln!("[K-β-2.5c dual]   leaked_dma_bufs       = (rough proxy — second-half VmRSS slope)");
    eprintln!("[K-β-2.5c dual]   leaked_vtcm_chunks    = (n/a — matmul kernel does not allocate VTCM)");
    eprintln!("[K-β-2.5c dual]   leaked_remote_handles = 0 (Arc-shared session retained whole run)");
    eprintln!("[K-β-2.5c dual]   T_MATMUL_LEAK_FREE {}", if leak_pass { "PASS" } else { "FAIL" });
    if !leak_pass { fails += 1; }

    // ─── T_GARNER_BIT_EXACT_CONCURRENT (cross-check) ──────────────────────
    eprintln!("\n[K-β-2.5c dual] ═══ T_GARNER_BIT_EXACT_CONCURRENT ═══");
    eprintln!("[K-β-2.5c dual] Concurrent dual-dispatch produces (r_1, r_2) Garner-recombined");
    eprintln!("[K-β-2.5c dual] bit-equal to sequential dispatch — already covered by functional");
    eprintln!("[K-β-2.5c dual] check in shape A + B above.");
    let garner_concurrent_pass = if let (Some(a), Some(b)) = (&shape_a, &shape_b) {
        a.9 && b.9
    } else { false };
    eprintln!("[K-β-2.5c dual]   T_GARNER_BIT_EXACT_CONCURRENT {}",
              if garner_concurrent_pass { "PASS (via shape A + B functional checks)" } else { "FAIL" });
    if !garner_concurrent_pass { fails += 1; }

    drop(sess);
    eprintln!("\n[K-β-2.5c dual] session closed cleanly");
    if fails == 0 {
        eprintln!("[K-β-2.5c dual] ALL GATES PASS");
        std::process::exit(0);
    } else {
        eprintln!("[K-β-2.5c dual] {} gate(s) FAILED", fails);
        std::process::exit(1);
    }
}
