//! Sprint NTT.0 T_NTT0_SCALAR_BIT_EXACT smoke harness.
//!
//! Drives `sp_compute_ntt_oracle` (skel method 12) for all 6 combinations of
//! (q_idx, N) ∈ {0, 1} × {128, 256, 512}, 100 random seeds per combination.
//! Each output is compared element-wise to math-core's `ntt_forward` (called
//! host-side via the static-lib link added in build.rs).
//!
//! Pass criterion: 0 divergences across 600 runs (6 combinations × 100 seeds).
//!
//! Build:
//!     cargo build --target aarch64-linux-android --release --bin sp_ntt_0_smoke
//! Run:
//!     adb push target/aarch64-linux-android/release/sp_ntt_0_smoke /data/local/tmp/
//!     adb shell 'ADSP_LIBRARY_PATH="/data/local/tmp;" /data/local/tmp/sp_ntt_0_smoke'

#[cfg(not(target_os = "android"))]
fn main() {
    eprintln!("sp_ntt_0_smoke: host build skipped (target_os = android required)");
}

#[cfg(target_os = "android")]
mod dsp_rpc;

#[cfg(target_os = "android")]
mod ntt_ffi {
    //! Minimal FFI binding to math-core's `ntt_forward` (lib/shannon-prime-system
    //! /include/sp/ntt_crt.h).  We only need init + forward + free for the oracle.
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

    // Silences "unused" if rustc's dead-code analyser reaches here.
    #[allow(dead_code)]
    fn _hint(_: *mut c_void) {}
}

#[cfg(target_os = "android")]
fn main() {
    use dsp_rpc::{make_scalars, FastRpcSession, RemoteArg, RemoteBuf};
    use ntt_ffi::{ntt_init, ntt_free, ntt_forward};
    use std::ffi::c_void;
    use std::process::ExitCode;
    use std::time::Instant;

    eprintln!("[NTT.0] sp_ntt_0_smoke — T_NTT0_SCALAR_BIT_EXACT");
    eprintln!("[NTT.0]   oracle = math-core ntt_forward (per-prime channel)");
    eprintln!("[NTT.0]   subject = sp_compute_ntt_oracle (skel method 12)");
    eprintln!("[NTT.0]   N ∈ {{128, 256, 512}}  ×  q_idx ∈ {{0, 1}}  ×  100 seeds = 600 runs");

    eprintln!("\n[NTT.0] opening FastRpcSession (Path B Unsigned PD)...");
    let sess = match FastRpcSession::new(
        "file:///libsp_compute_skel.so?sp_compute_skel_handle_invoke&_modver=1.0&_dom=cdsp",
    ) {
        Ok(s) => { eprintln!("[NTT.0] session open"); s }
        Err(e) => { eprintln!("[NTT.0] session FAIL: {e:?}"); std::process::exit(1); }
    };

    // Invoke ntt_oracle (method 12).  See sp_compute_skel.c::_skel_method
    // signature: 2 INBUFS (primIn + data_in), 1 OUTBUFS (data_out).
    //
    // primIn = [q_idx(i32), N(i32), data_inLen(i32), data_outLen(i32)] (16 B).
    fn invoke_ntt_oracle(
        sess: &FastRpcSession,
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
        // scalars: method=12, inbufs=2, outbufs=1, inhandles=0, outhandles=0
        sess.invoke(make_scalars(12, 2, 1), &mut args)
            .map_err(|e| format!("invoke method 12: {e:?}"))?;
        let out: Vec<u32> = out_bytes.chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        Ok(out)
    }

    // Deterministic LCG (Numerical Recipes constants).  Same constants as
    // sp_barrett_oracle::gen_test_vectors so the smoke is reproducible.
    fn gen_random_i32_vec(seed: u64, n: usize) -> Vec<i32> {
        let mut s = seed;
        let mut v: Vec<i32> = Vec::with_capacity(n);
        for _ in 0..n {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            v.push(s as i32);   // arbitrary signed i32, no range clamp
        }
        v
    }

    // Oracle: call math-core ntt_forward, return (out1 mod q1, out2 mod q2).
    // We pick the relevant residue channel via q_idx.
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

    // Aggregate report counters, per gate-spec.
    let mut combinations_tested: u32 = 0;
    let seeds_per_combination: u32 = 100;
    let mut total_runs: u32 = 0;
    let mut divergence_count: u32 = 0;
    // max_lane_diff(_per_prime, _per_N) -- absolute u32 lane diff.
    let mut max_diff_per_prime: [u32; 2] = [0, 0];
    let mut max_diff_per_n: [(i32, u32); 3] = [(128, 0), (256, 0), (512, 0)];
    let mut first_divergence: Option<(i32, i32, u64, usize, u32, u32)> = None;

    let t_start = Instant::now();

    for &n in &[128i32, 256, 512] {
        for q_idx in 0i32..=1 {
            combinations_tested += 1;
            eprintln!("\n[NTT.0] === combination q_idx={q_idx}  N={n} ===");
            let mut local_diverge = 0u32;
            let mut local_max_diff: u32 = 0;
            for seed_ix in 0..seeds_per_combination {
                let seed: u64 = 0xC0FFEEu64
                    .wrapping_add((n as u64).wrapping_mul(1_000_007))
                    .wrapping_add((q_idx as u64).wrapping_mul(2_000_011))
                    .wrapping_add(seed_ix as u64);
                let data_in = gen_random_i32_vec(seed, n as usize);
                let (oracle_q1, oracle_q2) = oracle_forward(n, &data_in);
                let expected: &Vec<u32> = if q_idx == 0 { &oracle_q1 } else { &oracle_q2 };

                let got = match invoke_ntt_oracle(&sess, q_idx, n, &data_in) {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("[NTT.0] invoke FAIL q_idx={q_idx} N={n} seed={seed_ix}: {e}");
                        divergence_count += seeds_per_combination - seed_ix;
                        local_diverge += seeds_per_combination - seed_ix;
                        break;
                    }
                };
                total_runs += 1;

                // Element-wise compare; track first divergence + max lane diff.
                for (lane, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
                    if g != e {
                        let diff = if g > e { g - e } else { e - g };
                        if diff > local_max_diff { local_max_diff = diff; }
                        if first_divergence.is_none() {
                            first_divergence = Some((n, q_idx, seed, lane, g, e));
                        }
                    }
                }
                let any_div = got.iter().zip(expected.iter()).any(|(g, e)| g != e);
                if any_div {
                    divergence_count += 1;
                    local_diverge += 1;
                }
            }
            // Promote local maxes to per-prime + per-N aggregate.
            if local_max_diff > max_diff_per_prime[q_idx as usize] {
                max_diff_per_prime[q_idx as usize] = local_max_diff;
            }
            for slot in max_diff_per_n.iter_mut() {
                if slot.0 == n && local_max_diff > slot.1 { slot.1 = local_max_diff; }
            }
            eprintln!("[NTT.0]   q_idx={q_idx} N={n}: diverged={local_diverge}/{seeds_per_combination}  max_lane_diff={local_max_diff}");
        }
    }

    let elapsed = t_start.elapsed();
    eprintln!("\n[NTT.0] ═══ T_NTT0_SCALAR_BIT_EXACT — aggregate report ═══");
    eprintln!("[NTT.0]   combinations_tested      = {combinations_tested}");
    eprintln!("[NTT.0]   seeds_per_combination    = {seeds_per_combination}");
    eprintln!("[NTT.0]   total_runs               = {total_runs}");
    eprintln!("[NTT.0]   divergence_count         = {divergence_count}");
    eprintln!("[NTT.0]   max_diff_per_prime       = {{q_1: {}, q_2: {}}}",
              max_diff_per_prime[0], max_diff_per_prime[1]);
    eprintln!("[NTT.0]   max_diff_per_N           = {{128: {}, 256: {}, 512: {}}}",
              max_diff_per_n[0].1, max_diff_per_n[1].1, max_diff_per_n[2].1);
    if let Some((n, q, seed, lane, g, e)) = first_divergence {
        eprintln!("[NTT.0]   first_divergence         = N={n} q_idx={q} seed={seed} lane={lane} got={g} exp={e}");
    } else {
        eprintln!("[NTT.0]   first_divergence         = (none)");
    }
    eprintln!("[NTT.0]   wall_time                = {:.2} s", elapsed.as_secs_f64());

    drop(sess);
    eprintln!("\n[NTT.0] session closed cleanly");

    let pass = divergence_count == 0 && total_runs == 600;
    if pass {
        eprintln!("[NTT.0] T_NTT0_SCALAR_BIT_EXACT PASS");
        std::process::exit(0);
    } else {
        eprintln!("[NTT.0] T_NTT0_SCALAR_BIT_EXACT FAIL  (div={divergence_count} runs={total_runs}/600)");
        std::process::exit(1);
    }
    #[allow(unreachable_code)]
    let _: ExitCode = ExitCode::SUCCESS;
}
