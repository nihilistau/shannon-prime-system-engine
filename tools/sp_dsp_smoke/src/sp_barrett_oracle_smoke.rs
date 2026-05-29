//! §3-HX Sprint K v0.beta Stage 2.5a — T_BARRETT_SCALAR_ORACLE bin.
//!
//! Drives ~1024 test vectors per prime through the skel-side scalar Barrett
//! (sp_compute_barrett_oracle, method 10, mode=0) and compares each result
//! bitwise against the Rust scalar reference.
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
    eprintln!("[K-β-2.5a] opening FastRpcSession (Path B)...");
    let sess = match FastRpcSession::new(
        "file:///libsp_compute_skel.so?sp_compute_skel_handle_invoke&_modver=1.0&_dom=cdsp")
    {
        Ok(s) => { eprintln!("[K-β-2.5a] session open"); s }
        Err(e) => { eprintln!("[K-β-2.5a] session FAIL: {e:?}"); std::process::exit(1); }
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

    for (label, q_idx, q, mu) in [
        ("T_BARRETT_SCALAR_ORACLE_q1", 0i32, SP_NTT_Q1, SP_MU_Q1),
        ("T_BARRETT_SCALAR_ORACLE_q2", 1,    SP_NTT_Q2, SP_MU_Q2),
    ] {
        let n = 1024usize;
        let (a, b) = gen_test_vectors(q, 0xDEADBEEFu64.wrapping_add(q_idx as u64), n);
        // Rust reference: compute expected per-element.
        let exp: Vec<u32> = a.iter().zip(b.iter())
            .map(|(&av, &bv)| barrett_reduce32(av as u64 * bv as u64, q, mu))
            .collect();
        match invoke_barrett_oracle(&sess, q_idx, 0, &a, &b) {
            Ok(got) => {
                let bad = got.iter().zip(exp.iter()).enumerate()
                    .find(|(_, (g, e))| g != e);
                if let Some((i, (&g, &e))) = bad {
                    eprintln!("[K-β-2.5a] {label} FAIL @ idx={i}  a={} b={} got={} exp={}",
                              a[i], b[i], g, e);
                    fails += 1;
                } else {
                    eprintln!("[K-β-2.5a] {label} PASS  ({n} vectors / q={} mu={})", q, mu);
                    eprintln!("[K-β-2.5a]   sample: a[0]={} b[0]={} r[0]={}; a[5]={} b[5]={} r[5]={}",
                              a[0], b[0], got[0], a[5], b[5], got[5]);
                }
            }
            Err(e) => { eprintln!("[K-β-2.5a] {label} FAIL invoke: {e:?}"); fails += 1; }
        }
    }

    // Stage 2.5b probe (will return -1 from the skel until HVX vector lands).
    match invoke_barrett_oracle(&sess, 0, 1, &[100u32], &[200u32]) {
        Ok(_)  => eprintln!("[K-β-2.5a]   note: mode=1 (HVX) unexpectedly returned Ok — Stage 2.5b lands?"),
        Err(e) => eprintln!("[K-β-2.5a]   note: mode=1 (HVX, Stage 2.5b) returns Err as expected: {e:?}"),
    }

    drop(sess);
    eprintln!("[K-β-2.5a] session closed cleanly");
    if fails == 0 {
        eprintln!("[K-β-2.5a] T_BARRETT_SCALAR_ORACLE_C_SCALAR ALL GATES PASS");
        std::process::exit(0);
    } else {
        eprintln!("[K-β-2.5a] {fails} gate(s) FAILED");
        std::process::exit(1);
    }
}
