//! §3-HX Sprint K v0.beta Stage 2.5c Stage 2 — mod_q matmul correctness smoke.
//!
//! Drives the on-device HVX mod_q matmul kernel (IDL method 11) against
//! the Rust scalar reference (T_MATMUL_Q_CORRECTNESS) and the unreduced
//! 60-bit matmul through Garner recombination (T_GARNER_BIT_EXACT).
//!
//! Shape: B=8 / D_in=128 / D_out=128.  Same regime as K v0.alpha matmul
//! (per reference-fastrpc-concurrent-dispatch — 17.7 ms / invoke baseline).
//!
//! Gates:
//!   T_MATMUL_Q_CORRECTNESS — for each of 4 seeds × both primes, kernel
//!     output == Rust scalar reference.  Pass: 0 divergences.
//!   T_GARNER_BIT_EXACT — for each seed, Garner(matmul_q1, matmul_q2)
//!     equals matmul_60bit_ref (inputs bounded so unreduced sum < M).
//!
//! Reports per-invoke wall time and DSP-side pcycles as secondary
//! diagnostics — needed to confirm the compute-bound regime at this shape
//! per feedback-shape-dependent-parallelism-gates.
//!
//! Build:
//!     cargo build --target aarch64-linux-android --release --bin sp_matmul_q_smoke
//! Run:
//!     adb shell 'ADSP_LIBRARY_PATH="/data/local/tmp;" /data/local/tmp/sp_matmul_q_smoke'

#[cfg(not(target_os = "android"))]
fn main() {
    eprintln!("sp_matmul_q_smoke: host build skipped");
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
    use sp_matmul_q_ref::{garner_combine_q1_q2, matmul_60bit_ref, matmul_q_scalar_ref};
    use std::ffi::c_void;
    use std::time::Instant;

    eprintln!("[K-β-2.5c] opening FastRpcSession (Path B)...");
    let sess = match FastRpcSession::new(
        "file:///libsp_compute_skel.so?sp_compute_skel_handle_invoke&_modver=1.0&_dom=cdsp")
    {
        Ok(s) => { eprintln!("[K-β-2.5c] session open"); s }
        Err(e) => { eprintln!("[K-β-2.5c] session FAIL: {e:?}"); std::process::exit(1); }
    };

    /// Invoke IDL method 11 (matmul_q).
    ///
    /// qaic-emitted arg layout (verified via diag method 9 / ffn_2stage_halide
    /// method 7 — primOut sits BEFORE rout buffers in the args array):
    ///   primIn[7] = [q_idx, batch, d_in, d_out, x_bufLen, w_bufLen, y_bufLen]  (28 B)
    ///   pra[0]    = primIn  (in)
    ///   pra[1]    = x_buf   (in)
    ///   pra[2]    = w_buf   (in)
    ///   pra[3]    = primOut (rout, packed [kernel_pcycles_lo, kernel_pcycles_hi] = 8 B)
    ///   pra[4]    = y_buf   (rout)
    ///   scalars   = make_scalars(11, 3, 2) — 3 in args + 2 out args = 5 total
    ///
    /// In-arg count includes the primIn buf (counts as 1 in arg).
    /// Out-arg count includes the primOut buf (counts as 1 out arg).
    fn invoke_matmul_q(sess: &FastRpcSession,
                       q_idx: i32, batch: i32, d_in: i32, d_out: i32,
                       x: &[u32], w: &[u32])
                       -> Result<(Vec<u32>, u64, Instant, Instant), SpErr> {
        assert_eq!(x.len(), (batch * d_in) as usize);
        assert_eq!(w.len(), (d_in * d_out) as usize);
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

    /// Deterministic test-vector generator for X, W in [0, q).
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

    /// Bounded X, W for the Garner round-trip check — elements < 2^26 so
    /// the unreduced 60-bit matmul stays in [0, M).
    fn gen_xw_bounded(seed: u64, n_x: usize, n_w: usize) -> (Vec<u32>, Vec<u32>) {
        let bound = 1u32 << 26;
        let mut x = vec![0u32; n_x];
        let mut w = vec![0u32; n_w];
        let mut s = seed;
        for v in x.iter_mut() {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            *v = (s as u32) % bound;
        }
        for v in w.iter_mut() {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            *v = (s as u32) % bound;
        }
        (x, w)
    }

    let (b, d_in, d_out) = (8i32, 128i32, 128i32);
    let n_x = (b * d_in) as usize;
    let n_w = (d_in * d_out) as usize;
    let mut fails = 0usize;
    let mut total_invoke_us: u128 = 0;
    let mut n_invokes: usize = 0;
    let seeds: [u64; 4] = [0xDEADBEEF_u64, 0xCAFEBABE_u64, 0xFEEDFACE_u64, 0xBAADF00D_u64];

    eprintln!("\n[K-β-2.5c] ═══ T_MATMUL_Q_CORRECTNESS ═══");
    eprintln!("[K-β-2.5c]   shape: B={} D_in={} D_out={}  ({} seeds × 2 primes)", b, d_in, d_out, seeds.len());
    let mut div_q1 = 0usize;
    let mut div_q2 = 0usize;
    let mut max_diff: i64 = 0;
    for &seed in &seeds {
        for (q_idx, q) in &[(0i32, SP_NTT_Q1), (1i32, SP_NTT_Q2)] {
            let (x, w) = gen_xw(*q, seed, n_x, n_w);
            let exp = matmul_q_scalar_ref(*q_idx, b as usize, d_in as usize, d_out as usize, &x, &w);
            match invoke_matmul_q(&sess, *q_idx, b, d_in, d_out, &x, &w) {
                Ok((got, pcyc, t0, t1)) => {
                    let wall = t1.duration_since(t0).as_micros();
                    total_invoke_us += wall;
                    n_invokes += 1;
                    let mut local_div = 0usize;
                    let mut local_max: i64 = 0;
                    let mut first: Option<(usize, u32, u32)> = None;
                    for (i, (g, e)) in got.iter().zip(exp.iter()).enumerate() {
                        if g != e {
                            local_div += 1;
                            let d = (*g as i64) - (*e as i64);
                            if d.abs() > local_max { local_max = d.abs(); }
                            if first.is_none() { first = Some((i, *g, *e)); }
                        }
                    }
                    if local_max > max_diff { max_diff = local_max; }
                    if *q_idx == 0 { div_q1 += local_div; } else { div_q2 += local_div; }
                    if local_div == 0 {
                        eprintln!("[K-β-2.5c]   seed=0x{:016x} q_idx={} PASS  pcyc={} wall={} μs y[0]={} y[last]={}",
                                  seed, q_idx, pcyc, wall, got[0], got[got.len()-1]);
                    } else {
                        eprintln!("[K-β-2.5c]   seed=0x{:016x} q_idx={} FAIL  diverge={}  pcyc={} wall={} μs",
                                  seed, q_idx, local_div, pcyc, wall);
                        if let Some((i, g, e)) = first {
                            eprintln!("[K-β-2.5c]     first diff @ i={}: got={} exp={}", i, g, e);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("[K-β-2.5c]   seed=0x{:016x} q_idx={} INVOKE FAIL: {:?}", seed, q_idx, e);
                    fails += 1;
                }
            }
        }
    }
    let total_q_samples = (n_x as usize / b as usize) * 0  // unused
                          + (b as usize) * (d_out as usize) * seeds.len();
    eprintln!("[K-β-2.5c]   q_1 divergences        = {}", div_q1);
    eprintln!("[K-β-2.5c]   q_2 divergences        = {}", div_q2);
    eprintln!("[K-β-2.5c]   max_lane_diff (signed) = {}", max_diff);
    eprintln!("[K-β-2.5c]   samples_compared (per prime) = {}", total_q_samples);
    let correctness_pass = div_q1 == 0 && div_q2 == 0;
    eprintln!("[K-β-2.5c]   T_MATMUL_Q_CORRECTNESS {}", if correctness_pass { "PASS" } else { "FAIL" });
    if !correctness_pass { fails += 1; }

    eprintln!("\n[K-β-2.5c] ═══ T_GARNER_BIT_EXACT ═══");
    eprintln!("[K-β-2.5c]   inputs bounded < 2^26 so unreduced sum < M = q_1*q_2 ≈ 2^60");
    let mut garner_div = 0usize;
    for &seed in &seeds {
        let (x, w) = gen_xw_bounded(seed, n_x, n_w);
        let y_q1 = match invoke_matmul_q(&sess, 0, b, d_in, d_out, &x, &w) {
            Ok((g, _, _, _)) => g,
            Err(e) => { eprintln!("[K-β-2.5c]   seed=0x{:016x} q_1 invoke FAIL: {:?}", seed, e);
                        fails += 1; continue; }
        };
        let y_q2 = match invoke_matmul_q(&sess, 1, b, d_in, d_out, &x, &w) {
            Ok((g, _, _, _)) => g,
            Err(e) => { eprintln!("[K-β-2.5c]   seed=0x{:016x} q_2 invoke FAIL: {:?}", seed, e);
                        fails += 1; continue; }
        };
        let recombined = garner_combine_q1_q2(&y_q1, &y_q2);
        let direct = match matmul_60bit_ref(b as usize, d_in as usize, d_out as usize, &x, &w) {
            Some(d) => d,
            None => { eprintln!("[K-β-2.5c]   seed=0x{:016x} unbounded — input bound too loose", seed);
                      fails += 1; continue; }
        };
        let mut local = 0usize;
        let mut first: Option<(usize, u64, u64)> = None;
        for (i, (g, e)) in recombined.iter().zip(direct.iter()).enumerate() {
            if g != e {
                local += 1;
                if first.is_none() { first = Some((i, *g, *e)); }
            }
        }
        if local == 0 {
            eprintln!("[K-β-2.5c]   seed=0x{:016x} PASS  ({} elements)", seed, recombined.len());
        } else {
            eprintln!("[K-β-2.5c]   seed=0x{:016x} FAIL  diverge={}", seed, local);
            if let Some((i, g, e)) = first {
                eprintln!("[K-β-2.5c]     first diff @ i={}: garner={} direct60bit={}", i, g, e);
            }
        }
        garner_div += local;
    }
    eprintln!("[K-β-2.5c]   total divergences      = {}", garner_div);
    let garner_pass = garner_div == 0;
    eprintln!("[K-β-2.5c]   T_GARNER_BIT_EXACT {}", if garner_pass { "PASS" } else { "FAIL" });
    if !garner_pass { fails += 1; }

    eprintln!("\n[K-β-2.5c] ═══ Per-invoke timing summary ═══");
    if n_invokes > 0 {
        eprintln!("[K-β-2.5c]   n_invokes              = {}", n_invokes);
        eprintln!("[K-β-2.5c]   total invoke wall      = {} μs", total_invoke_us);
        eprintln!("[K-β-2.5c]   avg per-invoke wall    = {} μs", total_invoke_us / n_invokes as u128);
        eprintln!("[K-β-2.5c]   K v0.alpha matmul ref  = ~17700 μs (saturating matmul, same B/D_in/D_out)");
    }

    drop(sess);
    eprintln!("\n[K-β-2.5c] session closed cleanly");
    if fails == 0 {
        eprintln!("[K-β-2.5c] Stage 2 correctness + Garner gates ALL PASS");
        std::process::exit(0);
    } else {
        eprintln!("[K-β-2.5c] {} subgate(s) FAILED", fails);
        std::process::exit(1);
    }
}
