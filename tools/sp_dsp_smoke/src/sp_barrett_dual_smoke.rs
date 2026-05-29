//! §3-HX Sprint K v0.beta Stage 2.5b Stage 3 — dual-dispatch + leak-free gates.
//!
//! Composes the HVX vector Barrett primitive (mode=1) with the
//! Arc<FastRpcSession> concurrent invoke pattern (per
//! reference-fastrpc-concurrent-dispatch).  Two ARM threads issue
//! barrett_oracle(mode=1, q_idx=0) and barrett_oracle(mode=1, q_idx=1)
//! simultaneously on the same skel handle.  cDSP scheduler engages
//! SSR:XA={4,5} dual vector contexts (V69 expert practices §22-87) per
//! K v0.alpha empirical finding (overlap_fraction=0.9699, speedup=1.935×).
//!
//! Stage 2.5b's DUAL_DISPATCH_SPEEDUP gate is specifically HVX-bound
//! (Stage 2.5a's scalar path used only the scalar pipe; Stage 2.5b uses
//! the HVX vector pipe, so dual concurrent invokes cross the SSR:XA
//! boundary and exercise Manifesto Trick #1 at the HVX level).
//!
//! Gates:
//!   M_K_beta_DUAL_DISPATCH_SPEEDUP — wall-clock speedup ≥ 1.5× per
//!     reference-fastrpc-concurrent-dispatch (wall-clock IS the
//!     discriminator, NOT pcycle ratio).
//!   M_K_beta_LEAK_FREE — 10000 cycles of dual-invoke + drop with VmRSS
//!     stable (delta ≤ 1024 KB end-vs-start).
//!
//! Build:
//!     cargo build --target aarch64-linux-android --release --bin sp_barrett_dual_smoke
//! Run:
//!     adb shell 'ADSP_LIBRARY_PATH="/data/local/tmp;" /data/local/tmp/sp_barrett_dual_smoke'

#[cfg(not(target_os = "android"))]
fn main() {
    eprintln!("sp_barrett_dual_smoke: host build skipped");
}

#[cfg(target_os = "android")]
mod dsp_rpc;
#[cfg(target_os = "android")]
mod sp_barrett_oracle;

#[cfg(target_os = "android")]
fn main() {
    use dsp_rpc::{make_scalars, FastRpcSession, RemoteArg, RemoteBuf, SpErr};
    use sp_barrett_oracle::{gen_test_vectors, SP_NTT_Q1, SP_NTT_Q2};
    use std::ffi::c_void;
    use std::sync::Arc;
    use std::time::Instant;

    eprintln!("[K-β-2.5b dual] opening FastRpcSession (Path B)...");
    let sess = match FastRpcSession::new(
        "file:///libsp_compute_skel.so?sp_compute_skel_handle_invoke&_modver=1.0&_dom=cdsp")
    {
        Ok(s) => { eprintln!("[K-β-2.5b dual] session open"); s }
        Err(e) => { eprintln!("[K-β-2.5b dual] session FAIL: {e:?}"); std::process::exit(1); }
    };
    let sess = Arc::new(sess);

    // Shape for the HVX vector Barrett: n=1024 (32 HVX vectors per prime).
    // Larger shapes make per-invoke wall time longer than FastRPC dispatch
    // overhead so wall-clock overlap is meaningful.  Sprint K v0.alpha used
    // a 128×128 matmul that took ~17.7 ms per single invoke; the Barrett
    // primitive at n=1024 is far smaller, so we scale n up to where the
    // single-invoke wall is on the order of milliseconds.
    const N_PER_PRIME: usize = 65536;   // 65536 u32 = 2048 HVX vectors per prime

    // Pre-build the (a, b) test vectors for each prime once.
    let (a_q1, b_q1) = gen_test_vectors(SP_NTT_Q1, 0xDEADBEEFu64,        N_PER_PRIME);
    let (a_q2, b_q2) = gen_test_vectors(SP_NTT_Q2, 0xDEADBEEFu64 ^ 1u64, N_PER_PRIME);

    // Invoke helper (single thread version, takes &FastRpcSession).
    fn invoke_barrett_oracle(sess: &FastRpcSession,
                             q_idx: i32, mode: i32,
                             a: &[u32], b: &[u32]) -> Result<(Vec<u32>, Instant, Instant), SpErr> {
        assert_eq!(a.len(), b.len());
        let n = a.len();
        let n_bytes = n * 4;
        let mut prim_in: [u32; 5] = [
            q_idx as u32, mode as u32,
            n_bytes as u32, n_bytes as u32, n_bytes as u32,
        ];
        let mut a_bytes  = Vec::with_capacity(n_bytes);
        for v in a { a_bytes.extend_from_slice(&v.to_le_bytes()); }
        let mut b_bytes  = Vec::with_capacity(n_bytes);
        for v in b { b_bytes.extend_from_slice(&v.to_le_bytes()); }
        let mut r_bytes  = vec![0u8; n_bytes];
        let mut args = [
            RemoteArg { buf: RemoteBuf { pv: prim_in.as_mut_ptr() as *mut c_void, nlen: 20 }},
            RemoteArg { buf: RemoteBuf { pv: a_bytes.as_mut_ptr() as *mut c_void, nlen: n_bytes }},
            RemoteArg { buf: RemoteBuf { pv: b_bytes.as_mut_ptr() as *mut c_void, nlen: n_bytes }},
            RemoteArg { buf: RemoteBuf { pv: r_bytes.as_mut_ptr() as *mut c_void, nlen: n_bytes }},
        ];
        let t0 = Instant::now();
        sess.invoke(make_scalars(10, 3, 1), &mut args)?;
        let t1 = Instant::now();
        let r: Vec<u32> = r_bytes.chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        Ok((r, t0, t1))
    }

    // Read /proc/self/status VmRSS in KB.
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

    // ─── Sequential baseline (q_1 then q_2) ────────────────────────────────
    eprintln!("\n[K-β-2.5b dual] ═══ Sequential baseline (mode=1, n_per_prime={}) ═══", N_PER_PRIME);
    let seq_start = Instant::now();
    let r_seq_q1 = match invoke_barrett_oracle(&sess, 0, 1, &a_q1, &b_q1) {
        Ok(t) => t,
        Err(e) => { eprintln!("[K-β-2.5b dual] seq q_1 invoke FAIL: {e:?}"); std::process::exit(1); }
    };
    let r_seq_q2 = match invoke_barrett_oracle(&sess, 1, 1, &a_q2, &b_q2) {
        Ok(t) => t,
        Err(e) => { eprintln!("[K-β-2.5b dual] seq q_2 invoke FAIL: {e:?}"); std::process::exit(1); }
    };
    let seq_wall_us = seq_start.elapsed().as_micros();
    let seq_q1_invoke_us = r_seq_q1.2.duration_since(r_seq_q1.1).as_micros();
    let seq_q2_invoke_us = r_seq_q2.2.duration_since(r_seq_q2.1).as_micros();
    eprintln!("[K-β-2.5b dual]   q_1 invoke wall = {} μs", seq_q1_invoke_us);
    eprintln!("[K-β-2.5b dual]   q_2 invoke wall = {} μs", seq_q2_invoke_us);
    eprintln!("[K-β-2.5b dual]   sequential total = {} μs", seq_wall_us);

    // Save the sequential q_1 result for functional cross-check post-concurrent.
    let baseline_q1 = r_seq_q1.0.clone();
    let baseline_q2 = r_seq_q2.0.clone();

    // ─── Concurrent dual-dispatch (mode=1 both threads) ────────────────────
    eprintln!("\n[K-β-2.5b dual] ═══ Concurrent dual-dispatch (Arc<FastRpcSession>) ═══");
    let sess_a = sess.clone();
    let sess_b = sess.clone();
    let a_q1_t = a_q1.clone();
    let b_q1_t = b_q1.clone();
    let a_q2_t = a_q2.clone();
    let b_q2_t = b_q2.clone();
    let conc_start = Instant::now();
    let h_a = std::thread::spawn(move || invoke_barrett_oracle(&sess_a, 0, 1, &a_q1_t, &b_q1_t));
    let h_b = std::thread::spawn(move || invoke_barrett_oracle(&sess_b, 1, 1, &a_q2_t, &b_q2_t));
    let r_a = h_a.join().expect("thread A join");
    let r_b = h_b.join().expect("thread B join");
    let conc_wall_us = conc_start.elapsed().as_micros();
    let (r_a, ts_a_start, ts_a_end) = match r_a {
        Ok((r, s, e)) => (r, s, e),
        Err(e) => { eprintln!("[K-β-2.5b dual] concurrent q_1 thread FAIL: {e:?}"); std::process::exit(1); }
    };
    let (r_b, ts_b_start, ts_b_end) = match r_b {
        Ok((r, s, e)) => (r, s, e),
        Err(e) => { eprintln!("[K-β-2.5b dual] concurrent q_2 thread FAIL: {e:?}"); std::process::exit(1); }
    };
    let conc_q1_invoke_us = ts_a_end.duration_since(ts_a_start).as_micros();
    let conc_q2_invoke_us = ts_b_end.duration_since(ts_b_start).as_micros();
    // Overlap window: [max(starts), min(ends)] if positive.
    let overlap_start = ts_a_start.max(ts_b_start);
    let overlap_end   = ts_a_end.min(ts_b_end);
    let overlap_us = if overlap_end > overlap_start {
        overlap_end.duration_since(overlap_start).as_micros()
    } else { 0 };
    eprintln!("[K-β-2.5b dual]   q_1 thread invoke wall = {} μs", conc_q1_invoke_us);
    eprintln!("[K-β-2.5b dual]   q_2 thread invoke wall = {} μs", conc_q2_invoke_us);
    eprintln!("[K-β-2.5b dual]   concurrent total wall  = {} μs", conc_wall_us);
    eprintln!("[K-β-2.5b dual]   overlap window         = {} μs", overlap_us);

    // ─── Functional cross-check — concurrent results match sequential ──────
    eprintln!("\n[K-β-2.5b dual] ═══ Functional check (concurrent == sequential) ═══");
    let q1_match = r_a == baseline_q1;
    let q2_match = r_b == baseline_q2;
    if q1_match && q2_match {
        eprintln!("[K-β-2.5b dual]   PASS  (both prime threads produce bitwise-identical output to sequential)");
    } else {
        eprintln!("[K-β-2.5b dual]   FAIL  q1_match={q1_match} q2_match={q2_match}");
        fails += 1;
    }

    // ─── M_K_beta_DUAL_DISPATCH_SPEEDUP ────────────────────────────────────
    eprintln!("\n[K-β-2.5b dual] ═══ M_K_beta_DUAL_DISPATCH_SPEEDUP ═══");
    let speedup = seq_wall_us as f64 / conc_wall_us as f64;
    let single_avg_us = ((seq_q1_invoke_us + seq_q2_invoke_us) / 2) as f64;
    let overlap_fraction = if conc_wall_us > 0 {
        overlap_us as f64 / conc_wall_us as f64
    } else { 0.0 };
    eprintln!("[K-β-2.5b dual]   sequential_wall_us  = {}", seq_wall_us);
    eprintln!("[K-β-2.5b dual]   concurrent_wall_us  = {}", conc_wall_us);
    eprintln!("[K-β-2.5b dual]   speedup             = {:.3}× (sequential / concurrent)", speedup);
    eprintln!("[K-β-2.5b dual]   overlap_fraction    = {:.4}", overlap_fraction);
    eprintln!("[K-β-2.5b dual]   single_invoke_avg   = {} μs", single_avg_us as u128);
    eprintln!("[K-β-2.5b dual]   K v0.alpha baseline = 1.935× (engine cdaaf15, 128x128/B=8 saturating matmul)");
    let speedup_pass = speedup >= 1.5;
    eprintln!("[K-β-2.5b dual]   threshold ≥ 1.5×    : {}", if speedup_pass { "PASS" } else { "FAIL" });
    if !speedup_pass { fails += 1; }

    // ─── M_K_beta_LEAK_FREE — 10000 dual-invoke cycles ────────────────────
    eprintln!("\n[K-β-2.5b dual] ═══ M_K_beta_LEAK_FREE (10000-iter dual-invoke cycle) ═══");
    let vmrss_start = vmrss_kb();
    eprintln!("[K-β-2.5b dual]   VmRSS @ iter 0     = {} KB", vmrss_start);
    let leak_start = Instant::now();
    let mut leak_fails = 0usize;
    let mut vmrss_mid = vmrss_start;
    for i in 0..10000 {
        let sess_a = sess.clone();
        let sess_b = sess.clone();
        let a_q1_t = a_q1.clone();
        let b_q1_t = b_q1.clone();
        let a_q2_t = a_q2.clone();
        let b_q2_t = b_q2.clone();
        let h_a = std::thread::spawn(move || invoke_barrett_oracle(&sess_a, 0, 1, &a_q1_t, &b_q1_t));
        let h_b = std::thread::spawn(move || invoke_barrett_oracle(&sess_b, 1, 1, &a_q2_t, &b_q2_t));
        let r_a = h_a.join().expect("leak thread A join");
        let r_b = h_b.join().expect("leak thread B join");
        match (r_a, r_b) {
            (Ok(_), Ok(_)) => {}
            _ => { eprintln!("[K-β-2.5b dual]   iter {i} FAIL"); leak_fails += 1; break; }
        }
        if i == 5000 {
            vmrss_mid = vmrss_kb();
            eprintln!("[K-β-2.5b dual]   VmRSS @ iter 5000  = {} KB", vmrss_mid);
        }
    }
    let leak_wall = leak_start.elapsed();
    let vmrss_end = vmrss_kb();
    eprintln!("[K-β-2.5b dual]   VmRSS @ iter 10000 = {} KB", vmrss_end);
    eprintln!("[K-β-2.5b dual]   cycles_run         = {}", 10000 - leak_fails);
    eprintln!("[K-β-2.5b dual]   wall time          = {:.2} s ({:.2} ms/iter)",
              leak_wall.as_secs_f64(),
              leak_wall.as_secs_f64() * 1000.0 / 10000.0);
    let vmrss_delta_kb = vmrss_end as i64 - vmrss_start as i64;
    eprintln!("[K-β-2.5b dual]   vmrss_delta_kb     = {}", vmrss_delta_kb);
    // Leak-free criterion: less than 1 MB growth over 10k iterations.
    let leak_pass = leak_fails == 0 && vmrss_delta_kb.abs() <= 1024;
    eprintln!("[K-β-2.5b dual]   leaked_dma_bufs    = (rough proxy — VmRSS delta)");
    eprintln!("[K-β-2.5b dual]   leaked_vtcm_chunks = (n/a for Barrett primitive — no VTCM allocs)");
    eprintln!("[K-β-2.5b dual]   leaked_remote_handles = 0 (sess Arc retained for whole run; no per-iter open/close)");
    eprintln!("[K-β-2.5b dual]   {}", if leak_pass { "PASS" } else { "FAIL" });
    if !leak_pass { fails += 1; }

    drop(sess);
    eprintln!("\n[K-β-2.5b dual] session closed cleanly");
    if fails == 0 {
        eprintln!("[K-β-2.5b dual] ALL GATES PASS");
        std::process::exit(0);
    } else {
        eprintln!("[K-β-2.5b dual] {fails} gate(s) FAILED");
        std::process::exit(1);
    }
}
