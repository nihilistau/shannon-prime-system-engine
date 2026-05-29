//! §3-HX Sprint K v0.beta Stage 2.5a + Stage 2.5b — T_BARRETT_*_ORACLE bin.
//!
//! Stage 2.5a (scalar):
//!     Drives ~1024 test vectors per prime through the skel-side scalar Barrett
//!     (sp_compute_barrett_oracle, method 10, mode=0) and compares each result
//!     bitwise against the Rust scalar reference.
//!
//! Stage 2.5b (HVX vector — this commit):
//!     Same test population AND a per-prime mode=1 invocation.  Each result is
//!     bitwise-compared against (a) the Rust scalar reference (M_K_beta_MATH_IDENTITY)
//!     and (b) the same kernel run in mode=0 on the same buffers (cross-mode
//!     identity).  Also verifies M_K_beta_BARRETT_CORRECTNESS: each output r
//!     satisfies r == (a*b) % q (ARM-side i128 reference) AND 0 ≤ r < q.
//!
//! Build:
//!     cargo build --target aarch64-linux-android --release --bin sp_barrett_oracle_smoke
//! Run:
//!     adb shell 'ADSP_LIBRARY_PATH="/data/local/tmp;" /data/local/tmp/sp_barrett_oracle_smoke'

#[cfg(not(target_os = "android"))]
fn main() {
    eprintln!("sp_barrett_oracle_smoke: host build skipped");
}

#[cfg(target_os = "android")]
mod dsp_rpc;
#[cfg(target_os = "android")]
mod sp_barrett_oracle;

#[cfg(target_os = "android")]
fn main() {
    use dsp_rpc::{make_scalars, FastRpcSession, RemoteArg, RemoteBuf, SpErr};
    use sp_barrett_oracle::{barrett_reduce32, gen_test_vectors,
                             SP_NTT_Q1, SP_NTT_Q2, SP_MU_Q1, SP_MU_Q2};
    use std::ffi::c_void;

    // Open session against the (now-extended) Sprint G compute skel.
    eprintln!("[K-β-2.5b] opening FastRpcSession (Path B)...");
    let sess = match FastRpcSession::new(
        "file:///libsp_compute_skel.so?sp_compute_skel_handle_invoke&_modver=1.0&_dom=cdsp")
    {
        Ok(s) => { eprintln!("[K-β-2.5b] session open"); s }
        Err(e) => { eprintln!("[K-β-2.5b] session FAIL: {e:?}"); std::process::exit(1); }
    };

    // qaic-emitted shape for method 10 = sp_compute_barrett_oracle:
    //   primIn[5] = [q_idx, mode, a_bufLen, b_bufLen, r_bufLen]   (20 B)
    //   pra[0]    = primIn (in)
    //   pra[1]    = a_buf  (in,  n × u32 LE)
    //   pra[2]    = b_buf  (in)
    //   pra[3]    = r_buf  (out, n × u32 LE)
    //   scalars   = MAKEX(0, 10, 3, 1, 0, 0)
    fn invoke_barrett_oracle(sess: &FastRpcSession,
                             q_idx: i32, mode: i32,
                             a: &[u32], b: &[u32]) -> Result<Vec<u32>, SpErr> {
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
        sess.invoke(make_scalars(10, 3, 1), &mut args)?;
        let r: Vec<u32> = r_bytes.chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        Ok(r)
    }

    let mut fails = 0usize;

    // Stage 2.5a gates — scalar Barrett path.
    eprintln!("\n[K-β-2.5b] ═══ Stage 2.5a — scalar Barrett (mode=0) ═══");
    for (label, q_idx, q, mu) in [
        ("T_BARRETT_SCALAR_ORACLE_q1", 0i32, SP_NTT_Q1, SP_MU_Q1),
        ("T_BARRETT_SCALAR_ORACLE_q2", 1,    SP_NTT_Q2, SP_MU_Q2),
    ] {
        let n = 1024usize;
        let (a, b) = gen_test_vectors(q, 0xDEADBEEFu64.wrapping_add(q_idx as u64), n);
        let exp: Vec<u32> = a.iter().zip(b.iter())
            .map(|(&av, &bv)| barrett_reduce32(av as u64 * bv as u64, q, mu))
            .collect();
        match invoke_barrett_oracle(&sess, q_idx, 0, &a, &b) {
            Ok(got) => {
                let bad = got.iter().zip(exp.iter()).enumerate()
                    .find(|(_, (g, e))| g != e);
                if let Some((i, (&g, &e))) = bad {
                    eprintln!("[K-β-2.5b] {label} FAIL @ idx={i}  a={} b={} got={} exp={}",
                              a[i], b[i], g, e);
                    fails += 1;
                } else {
                    eprintln!("[K-β-2.5b] {label} PASS  ({n} vectors / q={} mu={})", q, mu);
                    eprintln!("[K-β-2.5b]   sample: a[0]={} b[0]={} r[0]={}; a[5]={} b[5]={} r[5]={}",
                              a[0], b[0], got[0], a[5], b[5], got[5]);
                }
            }
            Err(e) => { eprintln!("[K-β-2.5b] {label} FAIL invoke: {e:?}"); fails += 1; }
        }
    }

    // Stage 2.5b gates — HVX vector Barrett path.
    eprintln!("\n[K-β-2.5b] ═══ Stage 2.5b — HVX vector Barrett (mode=1) ═══");
    eprintln!("[K-β-2.5b]   M_K_beta_MATH_IDENTITY     = vector == Rust scalar reference");
    eprintln!("[K-β-2.5b]   M_K_beta_BARRETT_CORRECTNESS = vector result satisfies r≡a·b (mod q) AND 0≤r<q");
    eprintln!("[K-β-2.5b]   plus cross-mode invariant   = mode=0 == mode=1");
    let mut math_identity_samples = 0usize;
    let mut math_identity_diverge = 0usize;
    let mut math_identity_max_diff: i64 = 0;
    let mut barrett_correct = 0usize;
    let mut barrett_total = 0usize;
    let mut cross_mode_diverge = 0usize;
    for (label, q_idx, q, mu) in [
        ("T_BARRETT_VECTOR_ORACLE_q1", 0i32, SP_NTT_Q1, SP_MU_Q1),
        ("T_BARRETT_VECTOR_ORACLE_q2", 1,    SP_NTT_Q2, SP_MU_Q2),
    ] {
        // n must be multiple of 32 (HVX 32-lane vectors); use 1024 = 32 vectors.
        let n = 1024usize;
        let (a, b) = gen_test_vectors(q, 0xDEADBEEFu64.wrapping_add(q_idx as u64), n);
        let exp: Vec<u32> = a.iter().zip(b.iter())
            .map(|(&av, &bv)| barrett_reduce32(av as u64 * bv as u64, q, mu))
            .collect();
        let scalar_got = match invoke_barrett_oracle(&sess, q_idx, 0, &a, &b) {
            Ok(got) => got,
            Err(e) => { eprintln!("[K-β-2.5b] {label} FAIL mode=0 baseline invoke: {e:?}");
                        fails += 1; continue; }
        };
        match invoke_barrett_oracle(&sess, q_idx, 1, &a, &b) {
            Ok(got) => {
                // Three checks: vs Rust ref, vs scalar skel, vs Barrett-invariant.
                let mut local_diverge = 0usize;
                let mut local_correct = 0usize;
                let mut local_cross_diverge = 0usize;
                let mut first_div: Option<(usize, u32, u32)> = None;
                for (i, ((&g, &e), &s)) in got.iter().zip(exp.iter())
                                                  .zip(scalar_got.iter()).enumerate()
                {
                    math_identity_samples += 1;
                    barrett_total += 1;
                    if g != e {
                        local_diverge += 1;
                        let diff = (g as i64) - (e as i64);
                        if diff.abs() > math_identity_max_diff { math_identity_max_diff = diff.abs(); }
                        if first_div.is_none() { first_div = Some((i, g, e)); }
                    }
                    if g != s {
                        local_cross_diverge += 1;
                    }
                    // BARRETT_CORRECTNESS: g == (a*b) mod q AND g < q.
                    let exp_mod = ((a[i] as u64) * (b[i] as u64)) % (q as u64);
                    if (g as u64) == exp_mod && g < q { local_correct += 1; }
                }
                math_identity_diverge += local_diverge;
                cross_mode_diverge += local_cross_diverge;
                barrett_correct += local_correct;
                if local_diverge == 0 && local_cross_diverge == 0 && local_correct == n {
                    eprintln!("[K-β-2.5b] {label} PASS  ({n} vectors / q={} mu={})", q, mu);
                    eprintln!("[K-β-2.5b]   sample: a[0]={} b[0]={} r[0]={}; a[5]={} b[5]={} r[5]={}",
                              a[0], b[0], got[0], a[5], b[5], got[5]);
                } else {
                    eprintln!("[K-β-2.5b] {label} FAIL  diverge_vs_rust={} diverge_vs_scalar={} correct={}/{}",
                              local_diverge, local_cross_diverge, local_correct, n);
                    if let Some((i, g, e)) = first_div {
                        eprintln!("[K-β-2.5b]   first divergence @ idx={i}  a={} b={} got={} exp={}",
                                  a[i], b[i], g, e);
                    }
                    fails += 1;
                }
            }
            Err(e) => { eprintln!("[K-β-2.5b] {label} FAIL invoke: {e:?}"); fails += 1; }
        }
    }

    eprintln!("\n[K-β-2.5b] ═══ Aggregate substantive-gate report ═══");
    eprintln!("[K-β-2.5b] M_K_beta_MATH_IDENTITY:");
    eprintln!("[K-β-2.5b]   samples_compared    = {}", math_identity_samples);
    eprintln!("[K-β-2.5b]   divergence_count    = {}", math_identity_diverge);
    eprintln!("[K-β-2.5b]   max_lane_diff       = {}", math_identity_max_diff);
    let identity_pass = math_identity_diverge == 0
        && math_identity_samples == 2 * 1024;
    eprintln!("[K-β-2.5b]   {}", if identity_pass { "PASS" } else { "FAIL" });

    eprintln!("[K-β-2.5b] BARRETT_CORRECTNESS:");
    eprintln!("[K-β-2.5b]   samples_correct     = {}", barrett_correct);
    eprintln!("[K-β-2.5b]   samples_total       = {}", barrett_total);
    let correctness_pass = barrett_correct == barrett_total && barrett_total == 2 * 1024;
    eprintln!("[K-β-2.5b]   {}", if correctness_pass { "PASS" } else { "FAIL" });

    eprintln!("[K-β-2.5b] cross-mode invariant (mode=0 == mode=1):");
    eprintln!("[K-β-2.5b]   divergences         = {}", cross_mode_diverge);
    let cross_pass = cross_mode_diverge == 0;
    eprintln!("[K-β-2.5b]   {}", if cross_pass { "PASS" } else { "FAIL" });

    drop(sess);
    eprintln!("\n[K-β-2.5b] session closed cleanly");
    if fails == 0 && identity_pass && correctness_pass && cross_pass {
        eprintln!("[K-β-2.5b] T_BARRETT_*_ORACLE — Stage 2.5a + 2.5b primitive gates PASS");
        std::process::exit(0);
    } else {
        eprintln!("[K-β-2.5b] FAIL  (subgates: fails={fails} identity={identity_pass} correctness={correctness_pass} cross={cross_pass})");
        std::process::exit(1);
    }
}
