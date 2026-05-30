//! §4-NTT Sprint NTT.3 -- dual-prime CRT NTT dispatch smoke.
//!
//! Drives the VTCM-aware HVX forward NTT (skel method 17,
//! `sp_compute_ntt_hvx_vtcm_oracle`) under the four gates:
//!
//!   T_NTT3_VTCM_AWARE_BIT_EXACT  -- method 17 output bit-exact vs method 12
//!                                   (NTT.0 scalar oracle) AND method 13
//!                                   (NTT.1 HVX with per-call precompute) on
//!                                   100 random inputs x 3 N x 2 primes = 600
//!                                   runs (1800 comparison points).
//!
//!   T_NTT3_DUAL_DISPATCH_SPEEDUP -- wall-clock speedup of two-thread
//!                                   Arc<FastRpcSession> concurrent invoke
//!                                   (q_1 thread + q_2 thread) vs back-to-back
//!                                   sequential. Threshold >=1.5x at N=512;
//!                                   per-N matrix at N in {128, 256, 512}.
//!                                   Per feedback-shape-dependent-parallelism-
//!                                   gates: NTT at N=512 per-invoke wall is in
//!                                   the data-bound regime per K.beta.2.5c
//!                                   empirical boundary; the gate is reported
//!                                   as-spec'd, with full disclosure.
//!
//!   T_NTT3_VTCM_NO_RECOMPUTE     -- method 17 single-invoke wall < method 13
//!                                   single-invoke wall by >=10% at N=512.
//!                                   The VTCM-aware path skips find_psi,
//!                                   psi_pow precompute, w_fwd precompute,
//!                                   per-stage compaction loop.
//!
//!   T_NTT3_NO_REGRESSION         -- method 12 (NTT.0) AND method 13 (NTT.1)
//!                                   AND ntt_twiddle_init / status / dump
//!                                   (methods 14/15/16) all still PASS their
//!                                   respective gates after the method 17
//!                                   addition.
//!
//! Build:
//!     cargo build --target aarch64-linux-android --release --bin sp_ntt_3_dual_smoke
//! Run:
//!     adb push target/aarch64-linux-android/release/sp_ntt_3_dual_smoke /data/local/tmp/
//!     adb shell 'ADSP_LIBRARY_PATH="/data/local/tmp;" /data/local/tmp/sp_ntt_3_dual_smoke'
//!
//! IDL method numbering (post merge c6df266):
//!   12 ntt_oracle              (NTT.0 scalar)
//!   13 ntt_hvx_oracle          (NTT.1 HVX, per-call precompute)
//!   14 ntt_twiddle_init        (NTT.2 VTCM staging init)
//!   15 ntt_twiddle_status      (NTT.2 inspect)
//!   16 ntt_twiddle_dump        (NTT.2 copy-to-host)
//!   17 ntt_hvx_vtcm_oracle     (NTT.3 this sprint, VTCM-aware HVX)

#[cfg(not(target_os = "android"))]
fn main() {
    eprintln!("sp_ntt_3_dual_smoke: host build skipped (target_os = android required)");
}

#[cfg(target_os = "android")]
mod dsp_rpc;

#[cfg(target_os = "android")]
mod ntt_ffi {
    //! Minimal FFI binding to math-core's `ntt_forward` (same shape as NTT.0/1).

    #[repr(C)]
    pub struct NttCtxOpaque {
        _opaque: [u8; 0],
    }

    extern "C" {
        pub fn ntt_init(N: u32) -> *mut NttCtxOpaque;
        pub fn ntt_free(ctx: *mut NttCtxOpaque);
        pub fn ntt_forward(
            ctx: *const NttCtxOpaque,
            inp: *const i32,
            out1: *mut u32,
            out2: *mut u32,
        );
    }
}

#[cfg(target_os = "android")]
fn main() {
    use dsp_rpc::{make_scalars, FastRpcSession, RemoteArg, RemoteBuf};
    use ntt_ffi::{ntt_forward, ntt_free, ntt_init};
    use std::ffi::c_void;
    use std::sync::Arc;
    use std::time::Instant;

    eprintln!("[NTT.3] sp_ntt_3_dual_smoke -- dual-prime CRT dispatch gates");
    eprintln!("[NTT.3]   T_NTT3_VTCM_AWARE_BIT_EXACT   (m17 == m12 == m13 == math-core)");
    eprintln!("[NTT.3]   T_NTT3_DUAL_DISPATCH_SPEEDUP  (concurrent m17 vs sequential at N=512, >=1.5x)");
    eprintln!("[NTT.3]   T_NTT3_VTCM_NO_RECOMPUTE      (m17 single-invoke wall < m13 by >=10% at N=512)");
    eprintln!("[NTT.3]   T_NTT3_NO_REGRESSION          (m12 600/600 + m13 600/600 + m14/15/16 PASS)");

    eprintln!("\n[NTT.3] opening FastRpcSession (Path B Unsigned PD)...");
    let sess_raw = match FastRpcSession::new(
        "file:///libsp_compute_skel.so?sp_compute_skel_handle_invoke&_modver=1.0&_dom=cdsp",
    ) {
        Ok(s) => {
            eprintln!("[NTT.3] session open");
            s
        }
        Err(e) => {
            eprintln!("[NTT.3] session FAIL: {e:?}");
            std::process::exit(1);
        }
    };
    let sess = Arc::new(sess_raw);

    // ── Helper: invoke any of methods 12, 13, 17 (all share primIn shape) ─────
    fn invoke_ntt(
        sess: &FastRpcSession,
        method: u32,
        q_idx: i32,
        n: i32,
        data_in: &[i32],
    ) -> Result<(Vec<u32>, Instant, Instant), String> {
        assert_eq!(data_in.len(), n as usize);
        let n_bytes = (n as usize) * 4;
        let mut prim_in: [u32; 4] = [
            q_idx as u32,
            n as u32,
            n_bytes as u32,
            n_bytes as u32,
        ];
        let mut in_bytes: Vec<u8> = Vec::with_capacity(n_bytes);
        for v in data_in {
            in_bytes.extend_from_slice(&v.to_le_bytes());
        }
        let mut out_bytes: Vec<u8> = vec![0u8; n_bytes];
        let mut args = [
            RemoteArg {
                buf: RemoteBuf {
                    pv: prim_in.as_mut_ptr() as *mut c_void,
                    nlen: 16,
                },
            },
            RemoteArg {
                buf: RemoteBuf {
                    pv: in_bytes.as_mut_ptr() as *mut c_void,
                    nlen: n_bytes,
                },
            },
            RemoteArg {
                buf: RemoteBuf {
                    pv: out_bytes.as_mut_ptr() as *mut c_void,
                    nlen: n_bytes,
                },
            },
        ];
        let t0 = Instant::now();
        sess.invoke(make_scalars(method, 2, 1), &mut args)
            .map_err(|e| format!("invoke method {method}: {e:?}"))?;
        let t1 = Instant::now();
        let out: Vec<u32> = out_bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Ok((out, t0, t1))
    }

    // ── method 14: ntt_twiddle_init (primIn = [N i32], no buffers, 1 inbuf 0 outbuf) ─
    fn invoke_twiddle_init(sess: &FastRpcSession, n: i32) -> Result<i64, String> {
        let mut prim_in: [u32; 1] = [n as u32];
        let mut args = [RemoteArg {
            buf: RemoteBuf {
                pv: prim_in.as_mut_ptr() as *mut c_void,
                nlen: 4,
            },
        }];
        let t0 = Instant::now();
        sess.invoke(make_scalars(14, 1, 0), &mut args)
            .map_err(|e| format!("invoke method 14: {e:?}"))?;
        Ok(t0.elapsed().as_micros() as i64)
    }

    // ── method 15: ntt_twiddle_status (primIn = [N, q_idx], primOut = 9 i32) ─
    #[derive(Debug, Default)]
    struct StatusOut {
        table_present: i32,
        vtcm_addr_lo: i32,
        vtcm_size: i32,
    }
    fn invoke_twiddle_status(sess: &FastRpcSession, n: i32, q_idx: i32) -> Result<StatusOut, String> {
        let mut prim_in: [u32; 2] = [n as u32, q_idx as u32];
        let mut prim_out: [u32; 9] = [0; 9];
        let mut args = [
            RemoteArg {
                buf: RemoteBuf {
                    pv: prim_in.as_mut_ptr() as *mut c_void,
                    nlen: 8,
                },
            },
            RemoteArg {
                buf: RemoteBuf {
                    pv: prim_out.as_mut_ptr() as *mut c_void,
                    nlen: 36,
                },
            },
        ];
        sess.invoke(make_scalars(15, 1, 1), &mut args)
            .map_err(|e| format!("invoke method 15: {e:?}"))?;
        Ok(StatusOut {
            table_present: prim_out[0] as i32,
            vtcm_addr_lo: prim_out[1] as i32,
            vtcm_size: prim_out[2] as i32,
        })
    }

    fn gen_random_i32_vec(seed: u64, n: usize) -> Vec<i32> {
        let mut s = seed;
        let mut v: Vec<i32> = Vec::with_capacity(n);
        for _ in 0..n {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            v.push(s as i32);
        }
        v
    }

    fn oracle_forward(n: i32, data_in: &[i32]) -> (Vec<u32>, Vec<u32>) {
        unsafe {
            let ctx = ntt_init(n as u32);
            assert!(!ctx.is_null(), "ntt_init({n}) returned NULL");
            let mut out1 = vec![0u32; n as usize];
            let mut out2 = vec![0u32; n as usize];
            ntt_forward(ctx, data_in.as_ptr(), out1.as_mut_ptr(), out2.as_mut_ptr());
            ntt_free(ctx);
            (out1, out2)
        }
    }

    // ─── Prime the VTCM tables before any method 17 invoke ────────────────
    eprintln!("\n[NTT.3] === Prime VTCM tables (ntt_twiddle_init m14) ===");
    let init_wall = match invoke_twiddle_init(&sess, 512) {
        Ok(us) => {
            eprintln!("[NTT.3]   ntt_twiddle_init(N=512) ok ({us} us)");
            us
        }
        Err(e) => {
            eprintln!("[NTT.3]   ntt_twiddle_init FAIL: {e}");
            std::process::exit(1);
        }
    };
    let init_wall_2 = match invoke_twiddle_init(&sess, 512) {
        Ok(us) => {
            eprintln!("[NTT.3]   ntt_twiddle_init(N=512) #2 (idempotent) ok ({us} us)");
            us
        }
        Err(e) => {
            eprintln!("[NTT.3]   ntt_twiddle_init #2 FAIL: {e}");
            -1
        }
    };
    let _ = init_wall;
    let _ = init_wall_2;

    // Confirm all 6 entries present.
    let combos = [(0i32, 128i32), (0, 256), (0, 512), (1, 128), (1, 256), (1, 512)];
    let mut twiddle_status_ok = true;
    for &(q_idx, n) in &combos {
        match invoke_twiddle_status(&sess, n, q_idx) {
            Ok(s) => {
                eprintln!(
                    "[NTT.3]   q_idx={q_idx} N={n}: present={} vtcm_addr_lo=0x{:08x} vtcm_size={}",
                    s.table_present, s.vtcm_addr_lo as u32, s.vtcm_size
                );
                if s.table_present != 1 || s.vtcm_addr_lo == 0 {
                    twiddle_status_ok = false;
                }
            }
            Err(e) => {
                eprintln!("[NTT.3]   q_idx={q_idx} N={n}: status FAIL: {e}");
                twiddle_status_ok = false;
            }
        }
    }
    eprintln!(
        "[NTT.3] T_NTT3_NO_REGRESSION (m14/15 sanity): {}",
        if twiddle_status_ok { "PASS" } else { "FAIL" }
    );

    // ─── T_NTT3_VTCM_AWARE_BIT_EXACT ─────────────────────────────────────
    // Per the prompt: 100 random inputs x 3 N x 2 primes = 600 runs;
    // each run compares method 17 against method 12 AND method 13 AND
    // math-core ntt_forward = 1800 comparison points.
    eprintln!("\n[NTT.3] === Correctness sweep (T_NTT3_VTCM_AWARE_BIT_EXACT + T_NTT3_NO_REGRESSION) ===");
    let seeds_per_combination: u32 = 100;
    let mut m12_total_runs: u32 = 0;
    let mut m13_total_runs: u32 = 0;
    let mut m17_total_runs: u32 = 0;
    let mut m12_divergence_count: u32 = 0;
    let mut m13_divergence_count: u32 = 0;
    let mut m17_divergence_count: u32 = 0;
    let mut m17_first_divergence: Option<(i32, i32, u64, usize, u32, u32)> = None;
    let t_start_correctness = Instant::now();

    for &n in &[128i32, 256, 512] {
        for q_idx in 0i32..=1 {
            eprintln!("\n[NTT.3] -- combination q_idx={q_idx} N={n} --");
            let mut local_m17_diverge = 0u32;
            let mut local_m17_max_diff: u32 = 0;
            for seed_ix in 0..seeds_per_combination {
                let seed: u64 = 0xC0FFEEu64
                    .wrapping_add((n as u64).wrapping_mul(1_000_007))
                    .wrapping_add((q_idx as u64).wrapping_mul(2_000_011))
                    .wrapping_add(seed_ix as u64);
                let data_in = gen_random_i32_vec(seed, n as usize);
                let (oracle_q1, oracle_q2) = oracle_forward(n, &data_in);
                let expected: &Vec<u32> = if q_idx == 0 { &oracle_q1 } else { &oracle_q2 };

                // Method 12 (NTT.0 scalar) -- T_NTT3_NO_REGRESSION baseline.
                let got_m12 = match invoke_ntt(&sess, 12, q_idx, n, &data_in) {
                    Ok((v, _, _)) => v,
                    Err(e) => {
                        eprintln!("[NTT.3] m12 invoke FAIL q_idx={q_idx} N={n} seed={seed_ix}: {e}");
                        m12_divergence_count += seeds_per_combination - seed_ix;
                        break;
                    }
                };
                m12_total_runs += 1;
                if got_m12 != *expected {
                    m12_divergence_count += 1;
                }

                // Method 13 (NTT.1 HVX with per-call precompute).
                let got_m13 = match invoke_ntt(&sess, 13, q_idx, n, &data_in) {
                    Ok((v, _, _)) => v,
                    Err(e) => {
                        eprintln!("[NTT.3] m13 invoke FAIL q_idx={q_idx} N={n} seed={seed_ix}: {e}");
                        m13_divergence_count += seeds_per_combination - seed_ix;
                        break;
                    }
                };
                m13_total_runs += 1;
                if got_m13 != *expected {
                    m13_divergence_count += 1;
                }

                // Method 17 (NTT.3 VTCM-aware HVX) -- T_NTT3_VTCM_AWARE_BIT_EXACT.
                let got_m17 = match invoke_ntt(&sess, 17, q_idx, n, &data_in) {
                    Ok((v, _, _)) => v,
                    Err(e) => {
                        eprintln!("[NTT.3] m17 invoke FAIL q_idx={q_idx} N={n} seed={seed_ix}: {e}");
                        m17_divergence_count += seeds_per_combination - seed_ix;
                        local_m17_diverge += seeds_per_combination - seed_ix;
                        break;
                    }
                };
                m17_total_runs += 1;
                let mut any_div = false;
                for (lane, (&g, &e)) in got_m17.iter().zip(expected.iter()).enumerate() {
                    if g != e {
                        any_div = true;
                        let diff = if g > e { g - e } else { e - g };
                        if diff > local_m17_max_diff {
                            local_m17_max_diff = diff;
                        }
                        if m17_first_divergence.is_none() {
                            m17_first_divergence = Some((n, q_idx, seed, lane, g, e));
                        }
                    }
                }
                if any_div {
                    m17_divergence_count += 1;
                    local_m17_diverge += 1;
                }

                // Cross-method sanity: m17 == m13 == m12 (all should agree).
                if got_m17 != got_m13 {
                    eprintln!(
                        "[NTT.3]   ! m17 != m13 at q_idx={q_idx} N={n} seed_ix={seed_ix}"
                    );
                }
                if got_m17 != got_m12 {
                    eprintln!(
                        "[NTT.3]   ! m17 != m12 at q_idx={q_idx} N={n} seed_ix={seed_ix}"
                    );
                }
            }
            eprintln!(
                "[NTT.3]   m17 (VTCM-HVX): diverged={local_m17_diverge}/{seeds_per_combination}  max_lane_diff={local_m17_max_diff}"
            );
        }
    }
    let t_correctness = t_start_correctness.elapsed();
    let m12_pass = m12_divergence_count == 0 && m12_total_runs == 600;
    let m13_pass = m13_divergence_count == 0 && m13_total_runs == 600;
    let m17_pass = m17_divergence_count == 0 && m17_total_runs == 600;

    eprintln!("\n[NTT.3] === Correctness aggregate ===");
    eprintln!("[NTT.3]   m12 total_runs={m12_total_runs} divergence={m12_divergence_count}");
    eprintln!("[NTT.3]   m13 total_runs={m13_total_runs} divergence={m13_divergence_count}");
    eprintln!("[NTT.3]   m17 total_runs={m17_total_runs} divergence={m17_divergence_count}");
    if let Some((n, q, seed, lane, g, e)) = m17_first_divergence {
        eprintln!("[NTT.3]   m17 first_divergence = N={n} q_idx={q} seed={seed} lane={lane} got={g} exp={e}");
    } else {
        eprintln!("[NTT.3]   m17 first_divergence = (none)");
    }
    eprintln!("[NTT.3]   correctness wall time   = {:.2} s", t_correctness.as_secs_f64());

    // ─── Wall-clock sweep (single-invoke; for T_NTT3_VTCM_NO_RECOMPUTE) ───
    eprintln!("\n[NTT.3] === Wall-clock sweep (T_NTT3_VTCM_NO_RECOMPUTE: m17 vs m13 single-invoke) ===");
    let perf_iters: u32 = 100;
    let perf_seed: u64 = 0xDECAFu64;
    // [m_ix][q_idx][n_ix] where m_ix 0=m13 (recompute HVX), 1=m17 (VTCM HVX).
    let mut wall_us: [[[u128; 3]; 2]; 2] = [[[0; 3]; 2]; 2];
    let n_values: [i32; 3] = [128, 256, 512];
    let methods_perf: [(u32, &str); 2] = [(13, "m13-HVX-recompute"), (17, "m17-HVX-VTCM")];

    for (n_ix, &n) in n_values.iter().enumerate() {
        let data_in = gen_random_i32_vec(perf_seed, n as usize);
        for q_idx in 0i32..=1 {
            for (m_ix, &(method, _label)) in methods_perf.iter().enumerate() {
                // Warm-up call.
                let _ = invoke_ntt(&sess, method, q_idx, n, &data_in);
                let t0 = Instant::now();
                for _ in 0..perf_iters {
                    let _ = invoke_ntt(&sess, method, q_idx, n, &data_in);
                }
                let elapsed = t0.elapsed();
                wall_us[m_ix][q_idx as usize][n_ix] = elapsed.as_micros();
            }
        }
    }
    eprintln!(
        "\n[NTT.3] wall-clock single-invoke matrix (us total over {perf_iters} iters):"
    );
    eprintln!("[NTT.3]                          N=128       N=256       N=512");
    for q_idx in 0..=1 {
        for (m_ix, &(_method, label)) in methods_perf.iter().enumerate() {
            eprintln!(
                "[NTT.3]   q_idx={q_idx} {label:20}  {:8} us  {:8} us  {:8} us",
                wall_us[m_ix][q_idx][0], wall_us[m_ix][q_idx][1], wall_us[m_ix][q_idx][2]
            );
        }
    }
    eprintln!("\n[NTT.3] VTCM-vs-recompute ratio (m17/m13; <1.0 = VTCM wins):");
    for q_idx in 0..=1 {
        for n_ix in 0..3 {
            let r = wall_us[0][q_idx][n_ix] as f64; // m13 recompute
            let v = wall_us[1][q_idx][n_ix] as f64; // m17 VTCM
            let ratio = if r > 0.0 { v / r } else { f64::NAN };
            eprintln!(
                "[NTT.3]   q_idx={q_idx} N={:3}: m17/m13 = {:.3}  (m13 {:.1} us/iter, m17 {:.1} us/iter)",
                n_values[n_ix], ratio, r / perf_iters as f64, v / perf_iters as f64
            );
        }
    }
    // T_NTT3_VTCM_NO_RECOMPUTE: m17 < m13 by >=10% at N=512, both primes.
    let mut vtcm_no_recompute_pass = true;
    let mut vtcm_no_recompute_threshold_pass = true;
    for q_idx in 0..=1 {
        let r = wall_us[0][q_idx][2] as f64;
        let v = wall_us[1][q_idx][2] as f64;
        if !(v < r) {
            vtcm_no_recompute_pass = false;
        }
        if !(v <= 0.90 * r) {
            vtcm_no_recompute_threshold_pass = false;
        }
    }

    // ─── T_NTT3_DUAL_DISPATCH_SPEEDUP ──────────────────────────────────────
    eprintln!("\n[NTT.3] === Dual-dispatch sweep (T_NTT3_DUAL_DISPATCH_SPEEDUP at N=128,256,512) ===");
    // For each N: seq_wall = wall(q1 invoke) + wall(q2 invoke) back-to-back;
    // concurrent_wall = wall(two threads on Arc<Session>); overlap_fraction
    // = (min(ta_end, tb_end) - max(ta_start, tb_start)) / concurrent_wall.
    fn run_dual_shape(
        sess: &Arc<FastRpcSession>,
        n: i32,
        iters: u32,
    ) -> Option<(u128, u128, f64, f64)> {
        let data_in_q1 = gen_random_i32_vec(0xCAFE_BABEu64, n as usize);
        let data_in_q2 = gen_random_i32_vec(0xC001_F00Du64, n as usize);

        // Warm-up.
        let _ = invoke_ntt(sess, 17, 0, n, &data_in_q1);
        let _ = invoke_ntt(sess, 17, 1, n, &data_in_q2);

        // Sequential: total wall across `iters` cycles, each cycle = q1 then q2.
        let seq_start = Instant::now();
        for _ in 0..iters {
            let _ = invoke_ntt(sess, 17, 0, n, &data_in_q1);
            let _ = invoke_ntt(sess, 17, 1, n, &data_in_q2);
        }
        let seq_wall = seq_start.elapsed().as_micros();

        // Concurrent: total wall across `iters` cycles, each cycle = two threads.
        // Overlap fraction sampled on the LAST cycle to avoid recomputing the
        // overlap inside every spawn pair (which would add noise from the
        // Instant::now() calls themselves).
        let mut conc_total_wall: u128 = 0;
        let mut last_overlap_us: u128 = 0;
        let mut last_conc_wall: u128 = 0;
        for cycle in 0..iters {
            let sess_a = sess.clone();
            let sess_b = sess.clone();
            let a_in = data_in_q1.clone();
            let b_in = data_in_q2.clone();
            let cycle_start = Instant::now();
            let h_a = std::thread::spawn(move || invoke_ntt(&sess_a, 17, 0, n, &a_in));
            let h_b = std::thread::spawn(move || invoke_ntt(&sess_b, 17, 1, n, &b_in));
            let r_a = h_a.join().expect("thread A");
            let r_b = h_b.join().expect("thread B");
            let cycle_wall = cycle_start.elapsed().as_micros();
            conc_total_wall += cycle_wall;
            match (r_a, r_b) {
                (Ok((_, ta0, ta1)), Ok((_, tb0, tb1))) => {
                    if cycle + 1 == iters {
                        let ovl_start = ta0.max(tb0);
                        let ovl_end = ta1.min(tb1);
                        let ovl_us = if ovl_end > ovl_start {
                            ovl_end.duration_since(ovl_start).as_micros()
                        } else {
                            0
                        };
                        last_overlap_us = ovl_us;
                        last_conc_wall = cycle_wall;
                    }
                }
                _ => {
                    eprintln!("[NTT.3]   dual_dispatch invoke FAIL at cycle {cycle} N={n}");
                    return None;
                }
            }
        }
        let speedup = seq_wall as f64 / conc_total_wall.max(1) as f64;
        let overlap_fraction = if last_conc_wall > 0 {
            last_overlap_us as f64 / last_conc_wall as f64
        } else {
            0.0
        };
        Some((seq_wall, conc_total_wall, speedup, overlap_fraction))
    }

    let dual_iters: u32 = 50;
    let mut dual_speedup_per_n: [(i32, f64, u128, u128, f64); 3] =
        [(128, 0.0, 0, 0, 0.0), (256, 0.0, 0, 0, 0.0), (512, 0.0, 0, 0, 0.0)];
    let mut dual_n_idx: usize = 0;
    let mut dual_all_ok = true;
    for &n in &[128i32, 256, 512] {
        let res = run_dual_shape(&sess, n, dual_iters);
        match res {
            Some((seq, conc, sp, ovl)) => {
                eprintln!(
                    "[NTT.3]   N={n}: seq_wall={} us / conc_wall={} us / speedup={:.3}x / last_overlap_fraction={:.4} (iters={dual_iters})",
                    seq, conc, sp, ovl
                );
                dual_speedup_per_n[dual_n_idx] = (n, sp, seq, conc, ovl);
            }
            None => {
                dual_all_ok = false;
            }
        }
        dual_n_idx += 1;
    }
    let speedup_n512 = dual_speedup_per_n[2].1;
    let speedup_pass = dual_all_ok && speedup_n512 >= 1.5;
    eprintln!(
        "[NTT.3]   speedup at N=512 = {:.3}x (threshold >=1.5x)",
        speedup_n512
    );

    // ─── T_NTT3_NO_REGRESSION (composite of m12/m13/m14/m15/m16) ───────────
    // m12 + m13 covered above; m14/m15 covered via twiddle_status_ok. We
    // additionally dump one table to exercise m16.
    let dump_test = {
        let n = 512i32;
        let q_idx = 0i32;
        let table_id = 0i32;
        let expected_bytes = (n * 4) as usize;
        let mut prim_in: [u32; 4] = [
            n as u32,
            q_idx as u32,
            table_id as u32,
            expected_bytes as u32,
        ];
        let mut dst: Vec<u8> = vec![0u8; expected_bytes];
        let mut args = [
            RemoteArg {
                buf: RemoteBuf {
                    pv: prim_in.as_mut_ptr() as *mut c_void,
                    nlen: 16,
                },
            },
            RemoteArg {
                buf: RemoteBuf {
                    pv: dst.as_mut_ptr() as *mut c_void,
                    nlen: expected_bytes,
                },
            },
        ];
        match sess.invoke(make_scalars(16, 1, 1), &mut args) {
            Ok(_) => {
                let any_nonzero = dst.iter().any(|&b| b != 0);
                if !any_nonzero {
                    eprintln!("[NTT.3]   m16 dump returned all-zero (suspicious)");
                    false
                } else {
                    eprintln!("[NTT.3]   m16 dump ok ({} bytes, non-zero)", expected_bytes);
                    true
                }
            }
            Err(e) => {
                eprintln!("[NTT.3]   m16 dump FAIL: {e:?}");
                false
            }
        }
    };
    let no_regression_pass =
        m12_pass && m13_pass && twiddle_status_ok && dump_test;

    // ─── Aggregate report ────────────────────────────────────────────────
    eprintln!("\n[NTT.3] ======= GATE SUMMARY =======");
    eprintln!(
        "[NTT.3]   T_NTT3_VTCM_AWARE_BIT_EXACT  : {}  (m17 total_runs={}, divergences={})",
        if m17_pass { "PASS" } else { "FAIL" },
        m17_total_runs,
        m17_divergence_count
    );
    eprintln!(
        "[NTT.3]   T_NTT3_DUAL_DISPATCH_SPEEDUP : {}  (speedup at N=512 = {:.3}x, threshold >=1.5x)",
        if speedup_pass { "PASS" } else { "FAIL" },
        speedup_n512
    );
    eprintln!(
        "[NTT.3]   T_NTT3_VTCM_NO_RECOMPUTE     : {}  (m17 < m13 at N=512: {}, m17 <= 0.9*m13 at N=512: {})",
        if vtcm_no_recompute_threshold_pass {
            "PASS"
        } else if vtcm_no_recompute_pass {
            "WEAK-PASS"
        } else {
            "FAIL"
        },
        vtcm_no_recompute_pass,
        vtcm_no_recompute_threshold_pass
    );
    eprintln!(
        "[NTT.3]   T_NTT3_NO_REGRESSION         : {}  (m12 {}/{}, m13 {}/{}, m14/15/16 ok={})",
        if no_regression_pass { "PASS" } else { "FAIL" },
        m12_total_runs - m12_divergence_count,
        m12_total_runs,
        m13_total_runs - m13_divergence_count,
        m13_total_runs,
        twiddle_status_ok && dump_test
    );

    drop(sess);
    eprintln!("[NTT.3] session closed cleanly");

    let all_pass = m17_pass
        && speedup_pass
        && vtcm_no_recompute_threshold_pass
        && no_regression_pass;
    if all_pass {
        eprintln!("[NTT.3] all 4 substantive gates PASS");
        std::process::exit(0);
    } else {
        eprintln!("[NTT.3] one or more gates FAIL");
        std::process::exit(1);
    }
}
