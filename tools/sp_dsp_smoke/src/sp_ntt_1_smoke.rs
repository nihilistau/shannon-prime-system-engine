//! §4-NTT Sprint NTT.1 — HVX butterfly smoke harness.
//!
//! Drives:
//!   - `sp_compute_ntt_oracle`     (skel method 12, NTT.0 scalar reference)
//!   - `sp_compute_ntt_hvx_oracle` (skel method 13, NTT.1 HVX vectorized)
//!
//! Three gates:
//!
//!   T_NTT1_HVX_BIT_EXACT — method 13 output element-wise == method 12 output
//!                          AND == math-core ntt_forward per-prime channel.
//!                          6 combinations × 100 random seeds = 600 runs.
//!                          Pass iff divergence_count == 0.
//!
//!   T_NTT1_WALL_CLOCK_WIN — wall-clock measurement: method 12 vs method 13
//!                           at all 3 N × 2 primes × 100 iters per shape.
//!                           Per feedback-shape-dependent-parallelism-gates,
//!                           no precommitted threshold; report all matrix
//!                           entries. PASS iff method 13 wall < method 12
//!                           wall at N=512 (largest shape, most HVX benefit).
//!
//!   T_NTT1_NO_REGRESSION — method 12 (NTT.0 ntt_oracle) re-runs 600/600 PASS
//!                          unchanged vs math-core. Pass iff
//!                          divergence_count == 0 AND total_runs == 600.
//!
//! Build:
//!     cargo build --target aarch64-linux-android --release --bin sp_ntt_1_smoke
//! Run:
//!     adb push target/aarch64-linux-android/release/sp_ntt_1_smoke /data/local/tmp/
//!     adb shell 'ADSP_LIBRARY_PATH="/data/local/tmp;" /data/local/tmp/sp_ntt_1_smoke'

#[cfg(not(target_os = "android"))]
fn main() {
    eprintln!("sp_ntt_1_smoke: host build skipped (target_os = android required)");
}

#[cfg(target_os = "android")]
mod dsp_rpc;

#[cfg(target_os = "android")]
mod ntt_ffi {
    //! Minimal FFI binding to math-core's `ntt_forward` (same as NTT.0).
    use std::os::raw::c_void;

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

    #[allow(dead_code)]
    fn _hint(_: *mut c_void) {}
}

#[cfg(target_os = "android")]
fn main() {
    use dsp_rpc::{make_scalars, FastRpcSession, RemoteArg, RemoteBuf};
    use ntt_ffi::{ntt_init, ntt_free, ntt_forward};
    use std::ffi::c_void;
    use std::time::Instant;

    eprintln!("[NTT.1] sp_ntt_1_smoke -- HVX butterfly gates");
    eprintln!("[NTT.1]   T_NTT1_HVX_BIT_EXACT   (method 13 == method 12 == math-core)");
    eprintln!("[NTT.1]   T_NTT1_WALL_CLOCK_WIN  (method 13 wall < method 12 wall at N=512)");
    eprintln!("[NTT.1]   T_NTT1_NO_REGRESSION   (method 12 still 600/600 PASS vs math-core)");

    eprintln!("\n[NTT.1] opening FastRpcSession (Path B Unsigned PD)...");
    let sess = match FastRpcSession::new(
        "file:///libsp_compute_skel.so?sp_compute_skel_handle_invoke&_modver=1.0&_dom=cdsp",
    ) {
        Ok(s) => { eprintln!("[NTT.1] session open"); s }
        Err(e) => { eprintln!("[NTT.1] session FAIL: {e:?}"); std::process::exit(1); }
    };

    // Method dispatcher: takes method id (12 or 13). primIn layout for both
    // methods is identical: [q_idx i32, N i32, data_inLen i32, data_outLen i32].
    // sp_compute_skel.c::sp_compute_skel_invoke routes case 12 -> ntt_oracle,
    // case 13 -> ntt_hvx_oracle. Both have 2 INBUFS (primIn + data_in) and
    // 1 OUTBUFS (data_out) -> scalars = MAKEX(method, 2, 1, 0, 0).
    fn invoke_ntt(
        sess: &FastRpcSession,
        method: u32,
        q_idx: i32,
        n: i32,
        data_in: &[i32],
    ) -> Result<Vec<u32>, String> {
        assert_eq!(data_in.len(), n as usize);
        let n_bytes = (n as usize) * 4;
        let mut prim_in: [u32; 4] = [
            q_idx as u32,
            n as u32,
            n_bytes as u32,
            n_bytes as u32,
        ];
        let mut in_bytes: Vec<u8> = Vec::with_capacity(n_bytes);
        for v in data_in { in_bytes.extend_from_slice(&v.to_le_bytes()); }
        let mut out_bytes: Vec<u8> = vec![0u8; n_bytes];
        let mut args = [
            RemoteArg { buf: RemoteBuf { pv: prim_in.as_mut_ptr() as *mut c_void, nlen: 16 }},
            RemoteArg { buf: RemoteBuf { pv: in_bytes.as_mut_ptr() as *mut c_void, nlen: n_bytes }},
            RemoteArg { buf: RemoteBuf { pv: out_bytes.as_mut_ptr() as *mut c_void, nlen: n_bytes }},
        ];
        sess.invoke(make_scalars(method, 2, 1), &mut args)
            .map_err(|e| format!("invoke method {method}: {e:?}"))?;
        let out: Vec<u32> = out_bytes.chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        Ok(out)
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

    // ── Gate counters ──────────────────────────────────────────────────────
    let mut hvx_combinations_tested: u32 = 0;
    let mut scalar_combinations_tested: u32 = 0;
    let seeds_per_combination: u32 = 100;
    let mut hvx_total_runs: u32 = 0;
    let mut scalar_total_runs: u32 = 0;
    let mut hvx_divergence_count: u32 = 0;
    let mut scalar_divergence_count: u32 = 0;
    let mut hvx_max_diff_per_prime: [u32; 2] = [0, 0];
    let mut hvx_max_diff_per_n: [(i32, u32); 3] = [(128, 0), (256, 0), (512, 0)];
    let mut hvx_first_divergence: Option<(i32, i32, u64, usize, u32, u32)> = None;
    let mut scalar_first_divergence: Option<(i32, i32, u64, usize, u32, u32)> = None;

    let t_start_correctness = Instant::now();

    // ── Correctness sweep: methods 12 + 13 against math-core, AND each other ──
    eprintln!("\n[NTT.1] === Correctness sweep (T_NTT1_HVX_BIT_EXACT + T_NTT1_NO_REGRESSION) ===");
    for &n in &[128i32, 256, 512] {
        for q_idx in 0i32..=1 {
            hvx_combinations_tested += 1;
            scalar_combinations_tested += 1;
            eprintln!("\n[NTT.1] -- combination q_idx={q_idx}  N={n} --");
            let mut local_hvx_diverge = 0u32;
            let mut local_scalar_diverge = 0u32;
            let mut local_hvx_max_diff: u32 = 0;
            for seed_ix in 0..seeds_per_combination {
                let seed: u64 = 0xC0FFEEu64
                    .wrapping_add((n as u64).wrapping_mul(1_000_007))
                    .wrapping_add((q_idx as u64).wrapping_mul(2_000_011))
                    .wrapping_add(seed_ix as u64);
                let data_in = gen_random_i32_vec(seed, n as usize);
                let (oracle_q1, oracle_q2) = oracle_forward(n, &data_in);
                let expected: &Vec<u32> = if q_idx == 0 { &oracle_q1 } else { &oracle_q2 };

                // Method 12 (scalar) — T_NTT1_NO_REGRESSION.
                let got_scalar = match invoke_ntt(&sess, 12, q_idx, n, &data_in) {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("[NTT.1] method 12 invoke FAIL q_idx={q_idx} N={n} seed={seed_ix}: {e}");
                        scalar_divergence_count += seeds_per_combination - seed_ix;
                        local_scalar_diverge += seeds_per_combination - seed_ix;
                        break;
                    }
                };
                scalar_total_runs += 1;
                let scalar_div = got_scalar.iter().zip(expected.iter())
                    .any(|(g, e)| g != e);
                if scalar_div {
                    scalar_divergence_count += 1;
                    local_scalar_diverge += 1;
                    if scalar_first_divergence.is_none() {
                        for (lane, (&g, &e)) in got_scalar.iter().zip(expected.iter()).enumerate() {
                            if g != e {
                                scalar_first_divergence = Some((n, q_idx, seed, lane, g, e));
                                break;
                            }
                        }
                    }
                }

                // Method 13 (HVX) — T_NTT1_HVX_BIT_EXACT.
                let got_hvx = match invoke_ntt(&sess, 13, q_idx, n, &data_in) {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("[NTT.1] method 13 invoke FAIL q_idx={q_idx} N={n} seed={seed_ix}: {e}");
                        hvx_divergence_count += seeds_per_combination - seed_ix;
                        local_hvx_diverge += seeds_per_combination - seed_ix;
                        break;
                    }
                };
                hvx_total_runs += 1;
                let mut any_hvx_div = false;
                for (lane, (&g, &e)) in got_hvx.iter().zip(expected.iter()).enumerate() {
                    if g != e {
                        any_hvx_div = true;
                        let diff = if g > e { g - e } else { e - g };
                        if diff > local_hvx_max_diff { local_hvx_max_diff = diff; }
                        if hvx_first_divergence.is_none() {
                            hvx_first_divergence = Some((n, q_idx, seed, lane, g, e));
                        }
                    }
                }
                if any_hvx_div {
                    hvx_divergence_count += 1;
                    local_hvx_diverge += 1;
                }

                // Cross-check: methods 12 and 13 should produce the SAME output.
                // If either matches the math-core oracle, this is implied; but
                // we record a hard divergence count here too in case there's a
                // subtle skel-routing issue.
                if got_scalar != got_hvx {
                    // already counted via the per-method divergences; just
                    // surface a one-line note (rare; shouldn't happen if both
                    // gates pass).
                    eprintln!("[NTT.1]   ! method 12 != method 13 at q_idx={q_idx} N={n} seed_ix={seed_ix}");
                }
            }
            if local_hvx_max_diff > hvx_max_diff_per_prime[q_idx as usize] {
                hvx_max_diff_per_prime[q_idx as usize] = local_hvx_max_diff;
            }
            for slot in hvx_max_diff_per_n.iter_mut() {
                if slot.0 == n && local_hvx_max_diff > slot.1 { slot.1 = local_hvx_max_diff; }
            }
            eprintln!("[NTT.1]   m12 (scalar): diverged={local_scalar_diverge}/{seeds_per_combination}");
            eprintln!("[NTT.1]   m13 (HVX):    diverged={local_hvx_diverge}/{seeds_per_combination}  max_lane_diff={local_hvx_max_diff}");
        }
    }
    let t_correctness = t_start_correctness.elapsed();

    // ── Wall-clock sweep: 100 iters per (method, q_idx, N) shape. ──────────
    eprintln!("\n[NTT.1] === Wall-clock sweep (T_NTT1_WALL_CLOCK_WIN) ===");
    let perf_iters: u32 = 100;
    let perf_seed: u64 = 0xDECAFu64;

    // Matrix: method -> q_idx -> N -> wall_ms_total. Hard-coded shape for
    // simple println; arr[method_ix][q_idx][n_ix].
    let mut wall_us: [[[u128; 3]; 2]; 2] = [[[0; 3]; 2]; 2];   // [m_ix][q_idx][n_ix]
    let n_values: [i32; 3] = [128, 256, 512];
    let methods: [(u32, &str); 2] = [(12, "m12-scalar"), (13, "m13-HVX")];

    for (n_ix, &n) in n_values.iter().enumerate() {
        // Build a fixed input vec ONCE per N (avoid timing the input prep).
        let data_in = gen_random_i32_vec(perf_seed, n as usize);
        for q_idx in 0i32..=1 {
            for (m_ix, &(method, _label)) in methods.iter().enumerate() {
                // Warm-up (1 call -- account for find_psi + twiddle precompute).
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

    eprintln!("\n[NTT.1] wall-clock matrix (us total over {perf_iters} iters):");
    eprintln!("[NTT.1]     N=128                N=256                N=512");
    for q_idx in 0..=1 {
        for (m_ix, &(_method, label)) in methods.iter().enumerate() {
            eprintln!("[NTT.1]   q_idx={q_idx} {label:14}  {:8} us  {:8} us  {:8} us",
                wall_us[m_ix][q_idx][0],
                wall_us[m_ix][q_idx][1],
                wall_us[m_ix][q_idx][2]);
        }
    }
    eprintln!("\n[NTT.1] HVX-vs-scalar ratio (m13 / m12; <1.0 = HVX wins):");
    for q_idx in 0..=1 {
        for n_ix in 0..3 {
            let s = wall_us[0][q_idx][n_ix] as f64;
            let h = wall_us[1][q_idx][n_ix] as f64;
            let ratio = if s > 0.0 { h / s } else { f64::NAN };
            eprintln!("[NTT.1]   q_idx={q_idx} N={:3}: m13/m12 = {:.3}  (scalar {:.1} us / iter, HVX {:.1} us / iter)",
                n_values[n_ix], ratio,
                s / perf_iters as f64,
                h / perf_iters as f64);
        }
    }

    // T_NTT1_WALL_CLOCK_WIN: HVX wall < scalar wall at N=512, both primes.
    let n512_ix = 2usize;
    let hvx_win_q0 = wall_us[1][0][n512_ix] < wall_us[0][0][n512_ix];
    let hvx_win_q1 = wall_us[1][1][n512_ix] < wall_us[0][1][n512_ix];
    let wall_pass = hvx_win_q0 && hvx_win_q1;

    // ── Aggregate report ───────────────────────────────────────────────────
    eprintln!("\n[NTT.1] === T_NTT1_HVX_BIT_EXACT aggregate ===");
    eprintln!("[NTT.1]   hvx_combinations_tested  = {hvx_combinations_tested}");
    eprintln!("[NTT.1]   seeds_per_combination    = {seeds_per_combination}");
    eprintln!("[NTT.1]   hvx_total_runs           = {hvx_total_runs}");
    eprintln!("[NTT.1]   hvx_divergence_count     = {hvx_divergence_count}");
    eprintln!("[NTT.1]   hvx_max_diff_per_prime   = {{q_1: {}, q_2: {}}}",
              hvx_max_diff_per_prime[0], hvx_max_diff_per_prime[1]);
    eprintln!("[NTT.1]   hvx_max_diff_per_N       = {{128: {}, 256: {}, 512: {}}}",
              hvx_max_diff_per_n[0].1, hvx_max_diff_per_n[1].1, hvx_max_diff_per_n[2].1);
    if let Some((n, q, seed, lane, g, e)) = hvx_first_divergence {
        eprintln!("[NTT.1]   hvx_first_divergence     = N={n} q_idx={q} seed={seed} lane={lane} got={g} exp={e}");
    } else {
        eprintln!("[NTT.1]   hvx_first_divergence     = (none)");
    }

    eprintln!("\n[NTT.1] === T_NTT1_NO_REGRESSION aggregate ===");
    eprintln!("[NTT.1]   scalar_combinations_tested = {scalar_combinations_tested}");
    eprintln!("[NTT.1]   scalar_total_runs          = {scalar_total_runs}");
    eprintln!("[NTT.1]   scalar_divergence_count    = {scalar_divergence_count}");
    if let Some((n, q, seed, lane, g, e)) = scalar_first_divergence {
        eprintln!("[NTT.1]   scalar_first_divergence    = N={n} q_idx={q} seed={seed} lane={lane} got={g} exp={e}");
    } else {
        eprintln!("[NTT.1]   scalar_first_divergence    = (none)");
    }

    eprintln!("\n[NTT.1]   correctness_wall_time    = {:.2} s", t_correctness.as_secs_f64());

    drop(sess);
    eprintln!("[NTT.1] session closed cleanly");

    let hvx_pass = hvx_divergence_count == 0 && hvx_total_runs == 600;
    let scalar_pass = scalar_divergence_count == 0 && scalar_total_runs == 600;

    eprintln!("\n[NTT.1] ======= GATE SUMMARY =======");
    eprintln!("[NTT.1]   T_NTT1_HVX_BIT_EXACT  : {}", if hvx_pass { "PASS" } else { "FAIL" });
    eprintln!("[NTT.1]   T_NTT1_NO_REGRESSION  : {}", if scalar_pass { "PASS" } else { "FAIL" });
    eprintln!("[NTT.1]   T_NTT1_WALL_CLOCK_WIN : {}  (q0_win={hvx_win_q0}, q1_win={hvx_win_q1} at N=512)",
              if wall_pass { "PASS" } else { "FAIL" });

    if hvx_pass && scalar_pass && wall_pass {
        eprintln!("[NTT.1] all 3 substantive gates PASS");
        std::process::exit(0);
    } else {
        eprintln!("[NTT.1] one or more gates FAIL");
        std::process::exit(1);
    }
}
