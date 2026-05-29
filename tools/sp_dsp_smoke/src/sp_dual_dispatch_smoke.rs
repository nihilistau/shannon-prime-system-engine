//! §3-HX Sprint K v0.alpha — dispatch-parallelism smoke.
//!
//! Bench A (single-thread sequential): one thread calls
//! ffn_2stage_diag_halide twice back-to-back.  Sum of wall times = the
//! "no parallelism possible" lower bound.
//!
//! Bench C (Arc<FastRpcSession>, dual ARM threads): two threads each
//! call invoke concurrently on one cdsp handle.  ARM-side timestamps
//! capture per-thread start/end; overlap_fraction = overlap / wall_total.
//!
//! Decision rule:
//!   overlap_fraction ≥ 0.5 → K v0.beta dispatch authorized
//!   overlap_fraction < 0.5 → pivot to K.2 (NPU/Mode B/D)
//!
//! Build:
//!     cargo build --target aarch64-linux-android --release --bin sp_dual_dispatch_smoke
//! Run:
//!     adb shell 'ADSP_LIBRARY_PATH="/data/local/tmp;" /data/local/tmp/sp_dual_dispatch_smoke'

#[cfg(not(target_os = "android"))]
fn main() {
    eprintln!("sp_dual_dispatch_smoke: host build skipped");
}

#[cfg(target_os = "android")]
mod dsp_rpc;
#[cfg(target_os = "android")]
mod sp_dual_dispatch;

#[cfg(target_os = "android")]
fn main() {
    use dsp_rpc::FastRpcSession;
    use sp_dual_dispatch::{DualDispatch, OverlapMetrics};
    use std::sync::Arc;
    use std::time::Instant;

    // ─── Open session (one handle, shared via Arc) ──────────────────────────
    eprintln!("[K-α] opening FastRpcSession against sp_compute_skel (Path B)...");
    let sess = match FastRpcSession::new(
        "file:///libsp_compute_skel.so?sp_compute_skel_handle_invoke&_modver=1.0&_dom=cdsp")
    {
        Ok(s) => { eprintln!("[K-α] session open"); s }
        Err(e) => { eprintln!("[K-α] session FAIL: {e:?}"); std::process::exit(1); }
    };
    let sess = Arc::new(sess);
    let dispatch = DualDispatch::new(sess.clone());
    let mut fails = 0usize;

    // ─── Test shape (matches Sprint H BISECT_DIM passing config + Sprint G) ─
    const BATCH: usize = 8;
    const D_IN:  usize = 128;
    const H_DIM: usize = 128;
    const D_OUT: usize = 128;
    const B_TERM: i32  = 0;
    const Q_BITS: i32  = 14;
    let x:  Vec<i16> = (0..BATCH * D_IN).map(|i| ((i as i32 * 37 + 11) & 0x7FFF) as i16 - 16384).collect();
    let w1: Vec<i16> = (0..H_DIM * D_IN).map(|i| ((i as i32 * 41 + 7)  & 0x7F)   as i16 - 64).collect();
    let w2: Vec<i16> = vec![0i16; D_OUT * H_DIM];

    // ─── Bench A — single-thread sequential baseline (two invokes) ──────────
    eprintln!("\n[K-α] ═══ Bench A: single-thread sequential baseline ═══");
    let bench_a_start = Instant::now();
    let r_a1 = DualDispatch::invoke_once(&sess, &x, &w1, &w2,
                                          BATCH as i32, D_IN as i32, H_DIM as i32, D_OUT as i32,
                                          B_TERM, Q_BITS).expect("bench-a invoke 1");
    let r_a2 = DualDispatch::invoke_once(&sess, &x, &w1, &w2,
                                          BATCH as i32, D_IN as i32, H_DIM as i32, D_OUT as i32,
                                          B_TERM, Q_BITS).expect("bench-a invoke 2");
    let bench_a_wall = bench_a_start.elapsed();
    eprintln!("[K-α]   invoke 1: kernel_pcyc={} wall={}μs",
              r_a1.kernel_pcyc, r_a1.end.duration_since(r_a1.start).as_micros());
    eprintln!("[K-α]   invoke 2: kernel_pcyc={} wall={}μs",
              r_a2.kernel_pcyc, r_a2.end.duration_since(r_a2.start).as_micros());
    eprintln!("[K-α]   Bench-A total wall: {} ms ({} μs)",
              bench_a_wall.as_millis(), bench_a_wall.as_micros());
    if r_a1.hidden != r_a2.hidden {
        eprintln!("[K-α]   WARN: bench-a invoke 1 and 2 produce different hidden — non-determinism?");
        fails += 1;
    }
    let baseline_hidden = r_a1.hidden.clone();
    let single_wall_us: u128 = (r_a1.end.duration_since(r_a1.start).as_micros()
                              + r_a2.end.duration_since(r_a2.start).as_micros()) / 2;
    eprintln!("[K-α]   per-invoke average wall: {} μs", single_wall_us);

    // ─── Bench C — Arc<FastRpcSession>, two threads concurrent ──────────────
    eprintln!("\n[K-α] ═══ Bench C: Arc<FastRpcSession>, dual concurrent invokes ═══");
    let bench_c_start = Instant::now();
    let (r_c_a, r_c_b) = dispatch.dual_invoke(
        x.clone(), w1.clone(), w2.clone(),
        BATCH as i32, D_IN as i32, H_DIM as i32, D_OUT as i32,
        B_TERM, Q_BITS);
    let bench_c_wall = bench_c_start.elapsed();
    let r_c_a = match r_c_a { Ok(r) => r, Err(e) => { eprintln!("[K-α] Bench-C thread A FAIL: {e:?}"); std::process::exit(1); } };
    let r_c_b = match r_c_b { Ok(r) => r, Err(e) => { eprintln!("[K-α] Bench-C thread B FAIL: {e:?}"); std::process::exit(1); } };
    let m = OverlapMetrics::from(&r_c_a, &r_c_b);
    eprintln!("[K-α]   thread A: kernel_pcyc={} wall={}μs", m.kernel_pcyc_a, m.wall_a_us);
    eprintln!("[K-α]   thread B: kernel_pcyc={} wall={}μs", m.kernel_pcyc_b, m.wall_b_us);
    eprintln!("[K-α]   overlap window: {} μs", m.overlap_us);
    eprintln!("[K-α]   wall total:     {} μs (Bench-C wrapper measured {} μs)",
              m.wall_total_us, bench_c_wall.as_micros());
    eprintln!("[K-α]   overlap_fraction = {} / {} = {:.4}",
              m.overlap_us, m.wall_total_us, m.overlap_fraction);
    eprintln!("[K-α]   kernel pcyc sum: {}; max: {}; ratio max/sum: {:.4}",
              m.kernel_pcyc_sum, m.kernel_pcyc_max,
              m.kernel_pcyc_max as f64 / m.kernel_pcyc_sum as f64);
    let speedup = (2.0 * single_wall_us as f64) / m.wall_total_us as f64;
    eprintln!("[K-α]   speedup vs sequential (2× single): {:.3}×", speedup);

    // ─── M_K_alpha_FUNCTIONAL ───────────────────────────────────────────────
    eprintln!("\n[K-α] ═══ M_K_alpha_FUNCTIONAL ═══");
    let a_match = r_c_a.hidden == baseline_hidden;
    let b_match = r_c_b.hidden == baseline_hidden;
    if a_match && b_match {
        eprintln!("[K-α] M_K_alpha_FUNCTIONAL PASS  (both threads bitwise-equal Bench-A baseline)");
    } else {
        let a_idx = r_c_a.hidden.iter().zip(baseline_hidden.iter()).position(|(x,y)| x != y);
        let b_idx = r_c_b.hidden.iter().zip(baseline_hidden.iter()).position(|(x,y)| x != y);
        eprintln!("[K-α] M_K_alpha_FUNCTIONAL FAIL  a_match={a_match} b_match={b_match} a_idx={a_idx:?} b_idx={b_idx:?}");
        fails += 1;
    }

    // ─── M_K_alpha_PCYCLE_OVERLAP ───────────────────────────────────────────
    eprintln!("\n[K-α] ═══ M_K_alpha_PCYCLE_OVERLAP ═══");
    eprintln!("[K-α]   overlap_fraction = {:.4}", m.overlap_fraction);
    eprintln!("[K-α]   speedup_vs_sequential = {:.3}×", speedup);
    if m.overlap_fraction >= 0.5 {
        eprintln!("[K-α] M_K_alpha_PCYCLE_OVERLAP ≥ 0.5  → K v0.beta dispatch AUTHORIZED");
    } else {
        eprintln!("[K-α] M_K_alpha_PCYCLE_OVERLAP < 0.5  → K v0.beta NOT dispatched");
        eprintln!("[K-α]   pivot to K.2 (NPU/Mode B/D); closure documents which dispatch link serialized");
        // Diagnostic: which link is the bottleneck?
        let kernel_pcyc_in_wall_a = m.wall_a_us as u64;  // not pcyc-equivalent but informative
        let _ = kernel_pcyc_in_wall_a;
        if m.wall_a_us > (m.kernel_pcyc_a / 600) as u128 && m.wall_b_us > (m.kernel_pcyc_b / 600) as u128 {
            // Rough heuristic: pcyc / 600 ≈ μs at 600 MHz cdsp clock.  If wall is much
            // larger than that, ARM-side / FastRPC overhead dominates.
            eprintln!("[K-α]   diagnostic: ARM-side wall ≫ kernel cdsp pcyc — FastRPC marshalling/transport may serialize");
        }
        if m.kernel_pcyc_sum as u128 > 2 * m.wall_total_us {
            eprintln!("[K-α]   diagnostic: kernel pcyc sum > 2× wall — cdsp scheduler saw work but couldn't parallelize HVX");
        }
    }

    // ─── M_K_alpha_LEAK_FREE ───────────────────────────────────────────────
    eprintln!("\n[K-α] ═══ M_K_alpha_LEAK_FREE (100-iter dual_invoke cycle) ═══");
    let leak_start = Instant::now();
    let mut leak_fails = 0usize;
    for i in 0..100 {
        let (ra, rb) = dispatch.dual_invoke(
            x.clone(), w1.clone(), w2.clone(),
            BATCH as i32, D_IN as i32, H_DIM as i32, D_OUT as i32,
            B_TERM, Q_BITS);
        match (ra, rb) {
            (Ok(_), Ok(_)) => {}
            _ => { eprintln!("[K-α]   iter {i} FAIL"); leak_fails += 1; break; }
        }
    }
    let leak_elapsed = leak_start.elapsed();
    if leak_fails == 0 {
        eprintln!("[K-α]   100 iter completed in {:?} ({:.1} ms/iter avg)",
                  leak_elapsed,
                  leak_elapsed.as_secs_f64() * 1000.0 / 100.0);
        eprintln!("[K-α] M_K_alpha_LEAK_FREE PASS");
    } else {
        eprintln!("[K-α] M_K_alpha_LEAK_FREE FAIL ({leak_fails} iterations errored)");
        fails += 1;
    }

    drop(dispatch);
    drop(sess);
    eprintln!("\n[K-α] session closed cleanly");

    if fails == 0 {
        eprintln!("[K-α] ALL GATES COMPLETED — overlap_fraction = {:.4} (decision rule applied above)",
                  m.overlap_fraction);
        std::process::exit(0);
    } else {
        eprintln!("[K-α] {fails} gate(s) FAILED");
        std::process::exit(1);
    }
}
