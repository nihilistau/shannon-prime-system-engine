//! §4-NTT Sprint NTT.4 Stage 3 — End-to-end polynomial multiplication smoke.
//!
//! Drives the COMPLETE round-trip:
//!   1. Random a, b ∈ Z[x]/(x^N + 1) (i32 inputs; mod-reduced inside).
//!   2. Forward NTT of each via method 13 (ntt_hvx_oracle) for q_idx=0,1
//!      → a_q1, a_q2, b_q1, b_q2.
//!   3. Pointwise multiply per-prime ARM-side (Barrett scalar reference).
//!   4. INTT via method 17 (intt_hvx_oracle) for each prime
//!      → c_q1, c_q2 (each u32[N] in [0, q_i)).
//!   5. ARM-side garner_combine_q1_q2_signed → c_out: Vec<i64> in (-M/2, M/2].
//!   6. Reference: math-core ntt_init + ntt_forward(a) + ntt_forward(b)
//!      + ntt_pointwise_mul + ntt_inverse → c_ref: Vec<i64>.
//!   7. Element-wise compare c_out vs c_ref; 0 divergences expected.
//!
//! T_NTT4_POLY_MUL_EXACT — N ∈ {128, 256, 512} × 4 seeds = 12 runs.
//! Pass: 0 divergences across all runs.
//!
//! Wall-clock (informational): forward + pointwise + INTT + Garner at N=512
//! over 100 iters.
//!
//! Build:
//!     cargo build --target aarch64-linux-android --release --bin sp_ntt_4_polymul_smoke
//! Run:
//!     adb push target/aarch64-linux-android/release/sp_ntt_4_polymul_smoke /data/local/tmp/
//!     adb shell 'ADSP_LIBRARY_PATH="/data/local/tmp;" /data/local/tmp/sp_ntt_4_polymul_smoke'

#[cfg(not(target_os = "android"))]
fn main() {
    eprintln!("sp_ntt_4_polymul_smoke: host build skipped (target_os = android required)");
}

#[cfg(target_os = "android")]
mod dsp_rpc;

#[cfg(target_os = "android")]
mod ntt_ffi {
    //! FFI binding to math-core's sp_ntt_crt static library — public
    //! functions: ntt_init, ntt_free, ntt_forward, ntt_pointwise_mul,
    //! ntt_inverse.

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
        pub fn ntt_pointwise_mul(
            ctx: *const NttCtxOpaque,
            a1: *const u32, a2: *const u32,
            b1: *const u32, b2: *const u32,
            out1: *mut u32, out2: *mut u32,
        );
        pub fn ntt_inverse(
            ctx: *const NttCtxOpaque,
            in1: *const u32,
            in2: *const u32,
            out: *mut i64,
        );
    }
}

#[cfg(target_os = "android")]
#[path = "sp_barrett_oracle.rs"]
mod sp_barrett_oracle;

#[cfg(target_os = "android")]
#[path = "sp_matmul_q_ref.rs"]
mod sp_matmul_q_ref;

#[cfg(target_os = "android")]
fn main() {
    use dsp_rpc::{make_scalars, FastRpcSession, RemoteArg, RemoteBuf};
    use ntt_ffi::{ntt_init, ntt_free, ntt_forward, ntt_pointwise_mul, ntt_inverse};
    use sp_matmul_q_ref::garner_combine_q1_q2_signed;
    use std::ffi::c_void;
    use std::time::Instant;

    // ── Frozen prime parameters ──
    const SP_NTT_Q1: u32 = 1_073_738_753;
    const SP_NTT_Q2: u32 = 1_073_732_609;
    const SP_MU_Q1: u64  = 1_073_744_895;
    const SP_MU_Q2: u64  = 1_073_751_039;
    const SP_NTT_Q_BITS: u32 = 30;

    eprintln!("[NTT.4] sp_ntt_4_polymul_smoke -- end-to-end polynomial multiplication");
    eprintln!("[NTT.4]   T_NTT4_POLY_MUL_EXACT (round-trip == math-core ntt_inverse)");

    // ── FastRPC session open ──
    eprintln!("\n[NTT.4] opening FastRpcSession (Path B Unsigned PD)...");
    let sess = match FastRpcSession::new(
        "file:///libsp_compute_skel.so?sp_compute_skel_handle_invoke&_modver=1.0&_dom=cdsp",
    ) {
        Ok(s) => { eprintln!("[NTT.4] session open"); s }
        Err(e) => { eprintln!("[NTT.4] session FAIL: {e:?}"); std::process::exit(1); }
    };

    // ── method 14: ntt_twiddle_init ──
    eprintln!("[NTT.4] priming VTCM twiddles via ntt_twiddle_init(N=512)...");
    {
        let mut prim_in: [u32; 1] = [512u32];
        let mut args = [
            RemoteArg { buf: RemoteBuf { pv: prim_in.as_mut_ptr() as *mut c_void, nlen: 4 }},
        ];
        match sess.invoke(make_scalars(14, 1, 0), &mut args) {
            Ok(_) => eprintln!("[NTT.4] ntt_twiddle_init ok"),
            Err(e) => {
                eprintln!("[NTT.4] ntt_twiddle_init FAIL: {e:?}");
                std::process::exit(1);
            }
        }
    }

    // ── method 13: forward NTT ──
    fn invoke_forward(
        sess: &FastRpcSession, q_idx: i32, n: i32, data_in: &[i32],
    ) -> Result<Vec<u32>, String> {
        let n_bytes = (n as usize) * 4;
        let mut prim_in: [u32; 4] = [
            q_idx as u32, n as u32, n_bytes as u32, n_bytes as u32,
        ];
        let mut in_bytes: Vec<u8> = Vec::with_capacity(n_bytes);
        for v in data_in { in_bytes.extend_from_slice(&v.to_le_bytes()); }
        let mut out_bytes: Vec<u8> = vec![0u8; n_bytes];
        let mut args = [
            RemoteArg { buf: RemoteBuf { pv: prim_in.as_mut_ptr() as *mut c_void, nlen: 16 }},
            RemoteArg { buf: RemoteBuf { pv: in_bytes.as_mut_ptr() as *mut c_void, nlen: n_bytes }},
            RemoteArg { buf: RemoteBuf { pv: out_bytes.as_mut_ptr() as *mut c_void, nlen: n_bytes }},
        ];
        sess.invoke(make_scalars(13, 2, 1), &mut args)
            .map_err(|e| format!("invoke ntt_hvx_oracle: {e:?}"))?;
        Ok(out_bytes.chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }

    // ── method 17 (this worktree): INTT ──
    // Anticipated method 18 post-NTT.3-merge; this worktree's IDL slot is 17.
    const INTT_METHOD: u32 = 17;
    fn invoke_intt(
        sess: &FastRpcSession, q_idx: i32, n: i32, data_in: &[u32],
    ) -> Result<Vec<u32>, String> {
        let n_bytes = (n as usize) * 4;
        let mut prim_in: [u32; 4] = [
            q_idx as u32, n as u32, n_bytes as u32, n_bytes as u32,
        ];
        let mut in_bytes: Vec<u8> = Vec::with_capacity(n_bytes);
        for v in data_in { in_bytes.extend_from_slice(&v.to_le_bytes()); }
        let mut out_bytes: Vec<u8> = vec![0u8; n_bytes];
        let mut args = [
            RemoteArg { buf: RemoteBuf { pv: prim_in.as_mut_ptr() as *mut c_void, nlen: 16 }},
            RemoteArg { buf: RemoteBuf { pv: in_bytes.as_mut_ptr() as *mut c_void, nlen: n_bytes }},
            RemoteArg { buf: RemoteBuf { pv: out_bytes.as_mut_ptr() as *mut c_void, nlen: n_bytes }},
        ];
        sess.invoke(make_scalars(INTT_METHOD, 2, 1), &mut args)
            .map_err(|e| format!("invoke intt_hvx_oracle: {e:?}"))?;
        Ok(out_bytes.chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }

    // ── ARM-side Barrett scalar pointwise multiply per-prime ──
    fn barrett_reduce(x: u64, q: u64, mu: u64) -> u64 {
        let qhat = (((x >> (SP_NTT_Q_BITS - 1)) as u128 * mu as u128) >> (SP_NTT_Q_BITS + 1)) as u64;
        let mut r = x.wrapping_sub(qhat.wrapping_mul(q));
        if r >= q { r -= q; }
        if r >= q { r -= q; }
        r
    }
    fn modmul(a: u32, b: u32, q: u32, mu: u64) -> u32 {
        barrett_reduce((a as u64) * (b as u64), q as u64, mu) as u32
    }
    fn pointwise_mul_q(a: &[u32], b: &[u32], q: u32, mu: u64) -> Vec<u32> {
        a.iter().zip(b.iter()).map(|(&x, &y)| modmul(x, y, q, mu)).collect()
    }

    // ── PRNG ──
    fn gen_random_i32_vec(seed: u64, n: usize) -> Vec<i32> {
        let mut s = seed;
        let mut v: Vec<i32> = Vec::with_capacity(n);
        for _ in 0..n {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            v.push(s as i32);
        }
        v
    }

    // ── host-side reference: math-core ntt_init + ntt_forward(a,b) +
    //    ntt_pointwise_mul + ntt_inverse -> Vec<i64> ──
    fn host_polymul(n: i32, a: &[i32], b: &[i32]) -> Vec<i64> {
        unsafe {
            let ctx = ntt_init(n as u32);
            assert!(!ctx.is_null(), "ntt_init({n}) returned NULL");
            let mut a1 = vec![0u32; n as usize];
            let mut a2 = vec![0u32; n as usize];
            let mut b1 = vec![0u32; n as usize];
            let mut b2 = vec![0u32; n as usize];
            ntt_forward(ctx, a.as_ptr(), a1.as_mut_ptr(), a2.as_mut_ptr());
            ntt_forward(ctx, b.as_ptr(), b1.as_mut_ptr(), b2.as_mut_ptr());
            let mut c1 = vec![0u32; n as usize];
            let mut c2 = vec![0u32; n as usize];
            ntt_pointwise_mul(ctx, a1.as_ptr(), a2.as_ptr(),
                                   b1.as_ptr(), b2.as_ptr(),
                                   c1.as_mut_ptr(), c2.as_mut_ptr());
            let mut out = vec![0i64; n as usize];
            ntt_inverse(ctx, c1.as_ptr(), c2.as_ptr(), out.as_mut_ptr());
            ntt_free(ctx);
            out
        }
    }

    // ── Driver: per-N + per-seed end-to-end ──
    let seeds: [u64; 4] = [
        0x5EE0_1234_5678_9ABC,
        0xDEEF_FACE_C001_BEAD,
        0x1234_5678_9ABC_DEF0,
        0xCAFE_BABE_FEED_F00D,
    ];

    let mut combinations_tested: u32 = 0;
    let mut total_runs: u32 = 0;
    let mut divergence_count: u32 = 0;
    let mut first_divergence: Option<(i32, usize, usize, i64, i64)> = None;
    let mut per_n_results: Vec<(i32, u32, u32)> = Vec::new(); // (N, seeds, divergences)

    let t_start = Instant::now();

    for &n in &[128i32, 256, 512] {
        let mut n_seeds_tested: u32 = 0;
        let mut n_divergences: u32 = 0;
        for (seed_ix, &seed) in seeds.iter().enumerate() {
            combinations_tested += 1;
            n_seeds_tested += 1;
            eprintln!("\n[NTT.4] -- N={n}  seed_ix={seed_ix} (seed={seed:#x}) --");
            let a = gen_random_i32_vec(seed, n as usize);
            let b = gen_random_i32_vec(seed.wrapping_add(1), n as usize);

            // Forward NTT on device for both primes, both polynomials.
            let a_q1 = match invoke_forward(&sess, 0, n, &a) {
                Ok(v) => v, Err(e) => { eprintln!("[NTT.4] FAIL: {e}"); divergence_count += 1; n_divergences += 1; total_runs += 1; continue; }
            };
            let a_q2 = match invoke_forward(&sess, 1, n, &a) {
                Ok(v) => v, Err(e) => { eprintln!("[NTT.4] FAIL: {e}"); divergence_count += 1; n_divergences += 1; total_runs += 1; continue; }
            };
            let b_q1 = match invoke_forward(&sess, 0, n, &b) {
                Ok(v) => v, Err(e) => { eprintln!("[NTT.4] FAIL: {e}"); divergence_count += 1; n_divergences += 1; total_runs += 1; continue; }
            };
            let b_q2 = match invoke_forward(&sess, 1, n, &b) {
                Ok(v) => v, Err(e) => { eprintln!("[NTT.4] FAIL: {e}"); divergence_count += 1; n_divergences += 1; total_runs += 1; continue; }
            };

            // Pointwise multiply per-prime ARM-side.
            let c_q1 = pointwise_mul_q(&a_q1, &b_q1, SP_NTT_Q1, SP_MU_Q1);
            let c_q2 = pointwise_mul_q(&a_q2, &b_q2, SP_NTT_Q2, SP_MU_Q2);

            // INTT on device for both primes.
            let r_q1 = match invoke_intt(&sess, 0, n, &c_q1) {
                Ok(v) => v, Err(e) => { eprintln!("[NTT.4] FAIL: {e}"); divergence_count += 1; n_divergences += 1; total_runs += 1; continue; }
            };
            let r_q2 = match invoke_intt(&sess, 1, n, &c_q2) {
                Ok(v) => v, Err(e) => { eprintln!("[NTT.4] FAIL: {e}"); divergence_count += 1; n_divergences += 1; total_runs += 1; continue; }
            };

            // ARM-side signed Garner.
            let dev_out: Vec<i64> = garner_combine_q1_q2_signed(&r_q1, &r_q2);

            // Host reference via math-core.
            let ref_out: Vec<i64> = host_polymul(n, &a, &b);

            // Element-wise compare.
            total_runs += 1;
            let mut local_diverge = 0u32;
            for j in 0..(n as usize) {
                if dev_out[j] != ref_out[j] {
                    if local_diverge == 0 {
                        if first_divergence.is_none() {
                            first_divergence = Some((n, seed_ix, j, dev_out[j], ref_out[j]));
                        }
                        divergence_count += 1;
                        n_divergences += 1;
                    }
                    local_diverge += 1;
                }
            }
            if local_diverge > 0 {
                eprintln!("[NTT.4]   ! N={n} seed_ix={seed_ix}: {local_diverge}/{n} coefficient divergences");
            } else {
                eprintln!("[NTT.4]   N={n} seed_ix={seed_ix} OK (byte-exact, N={} coefficients)", n);
            }
        }
        per_n_results.push((n, n_seeds_tested, n_divergences));
    }

    let elapsed_correctness = t_start.elapsed().as_secs_f64();

    eprintln!("\n[NTT.4] -- T_NTT4_POLY_MUL_EXACT SUMMARY --");
    eprintln!("[NTT.4] combinations_tested : {combinations_tested}");
    eprintln!("[NTT.4] total_runs          : {total_runs}");
    eprintln!("[NTT.4] divergence_count    : {divergence_count}");
    for (n, s, d) in per_n_results.iter() {
        eprintln!("[NTT.4]   N={n}: seeds_tested={s} divergences={d}");
    }
    eprintln!("[NTT.4] elapsed             : {elapsed_correctness:.2} s");
    if let Some((n, sx, j, d, r)) = first_divergence {
        eprintln!("[NTT.4] first_divergence    : N={n} seed_ix={sx} j={j} dev={d} ref={r}");
    }

    let correctness_pass = divergence_count == 0 && total_runs == combinations_tested;

    // ── Wall-clock benchmark (informational; N=512, 100 iters) ──
    eprintln!("\n[NTT.4] -- Wall-clock benchmark (informational, N=512, 100 iters) --");
    {
        let n: i32 = 512;
        let seed: u64 = 0x12345678_9ABCDEF0;
        let a = gen_random_i32_vec(seed, n as usize);
        let b = gen_random_i32_vec(seed.wrapping_add(0xDEAD), n as usize);

        let t0 = Instant::now();
        let mut iters_completed = 0u32;
        for _ in 0..100u32 {
            let a_q1 = invoke_forward(&sess, 0, n, &a).expect("benchmark fwd a_q1");
            let a_q2 = invoke_forward(&sess, 1, n, &a).expect("benchmark fwd a_q2");
            let b_q1 = invoke_forward(&sess, 0, n, &b).expect("benchmark fwd b_q1");
            let b_q2 = invoke_forward(&sess, 1, n, &b).expect("benchmark fwd b_q2");
            let c_q1 = pointwise_mul_q(&a_q1, &b_q1, SP_NTT_Q1, SP_MU_Q1);
            let c_q2 = pointwise_mul_q(&a_q2, &b_q2, SP_NTT_Q2, SP_MU_Q2);
            let r_q1 = invoke_intt(&sess, 0, n, &c_q1).expect("benchmark intt r_q1");
            let r_q2 = invoke_intt(&sess, 1, n, &c_q2).expect("benchmark intt r_q2");
            let _out = garner_combine_q1_q2_signed(&r_q1, &r_q2);
            iters_completed += 1;
        }
        let elapsed_bench = t0.elapsed();
        let per_iter_us = (elapsed_bench.as_micros() as f64) / (iters_completed as f64);
        eprintln!("[NTT.4] N=512 polymul round-trip: total={:?} iters={iters_completed} per_iter={per_iter_us:.1} us",
                  elapsed_bench);
        eprintln!("[NTT.4]   (per iter includes 4 forward NTTs + 2 pointwise + 2 INTTs + Garner)");
    }

    if correctness_pass {
        eprintln!("\n[NTT.4] T_NTT4_POLY_MUL_EXACT PASS  ({total_runs}/{total_runs} runs byte-exact)");
        std::process::exit(0);
    } else {
        eprintln!("\n[NTT.4] T_NTT4_POLY_MUL_EXACT FAIL  ({divergence_count}/{total_runs} divergences)");
        std::process::exit(1);
    }
}
