//! §3-HX Sprint D — HVX-vectorized i16 scale smoke (sp_compute_skel).
//!
//! Tests `sp_compute_scale_i16(n, a_h, x, y)` which computes
//! `y[i] = saturate_i16(x[i] + a_h)` using HVX intrinsics
//! (`Q6_Vh_vadd_VhVh_sat`, verified in SASS via hexagon-llvm-objdump).
//!
//! Gates:
//!   T_HVX_SCALE_BITWISE_{64, 1024, 65536} - vs scalar Rust reference
//!   T_HVX_SCALE_SATURATE - input that overflows i16 clamps to ±32767
//!   T_HVX_SCALE_VS_SCALAR - 1000-iter 64 KB wall comparison
//!
//! Build:
//!   cargo build --target aarch64-linux-android --release --bin test_hvx

#[cfg(not(target_os = "android"))]
fn main() {
    eprintln!("test_hvx: host build skipped");
}

#[cfg(target_os = "android")]
mod dsp_rpc;

#[cfg(target_os = "android")]
fn main() {
    use dsp_rpc::{make_scalars, FastRpcSession, RemoteArg, RemoteBuf, SpErr};
    use std::ffi::c_void;
    use std::time::Instant;

    // sp_compute_URI from qaic-generated sp_compute.h:
    //   "file:///libsp_compute_skel.so?sp_compute_skel_handle_invoke&_modver=1.0"
    // plus &_dom=cdsp for the V69 cDSP domain.
    let skel_uri = "file:///libsp_compute_skel.so?sp_compute_skel_handle_invoke&_modver=1.0&_dom=cdsp";

    eprintln!("[hvx] opening FastRpcSession against sp_compute_skel (Path B)...");
    let sess = match FastRpcSession::new(skel_uri) {
        Ok(s) => s,
        Err(e) => { eprintln!("[hvx] FAIL: {e:?}"); std::process::exit(1); }
    };
    eprintln!("[hvx] session open");

    // Scalar reference for sp_compute_scale_i16.
    fn scale_ref(x: &[i16], a_h: i16) -> Vec<i16> {
        x.iter().map(|&v| (v as i32 + a_h as i32).clamp(-32768, 32767) as i16).collect()
    }

    /// Invoke sp_compute_scale_i16 via FastRPC.
    /// Method 3 (open=0, close=1, axpby=2, scale_i16=3).
    /// Per qaic stub (sp_compute_stub.c:329-349):
    ///   primIn[4] = [n:u32, a_h:u32, x_bufLen:u32, y_bufLen:u32]   (16 B)
    ///   pra[0]   = primIn buffer
    ///   pra[1]   = x_buf (i16 LE bytes)
    ///   pra[2]   = y_buf (i16 LE bytes)
    ///   scalars  = MAKEX(0, mid=3, n_in=2, n_out=1, 0, 0)
    fn invoke_scale(sess: &FastRpcSession, x: &[i16], a_h: i16) -> Result<Vec<i16>, SpErr> {
        let n = x.len();
        let n_bytes = n * 2;
        let mut prim_in: [u32; 4] = [n as u32, a_h as u32, n_bytes as u32, n_bytes as u32];

        // Copy x to a u8 buffer (FastRPC marshalling uses byte-arg).
        let mut x_bytes = Vec::with_capacity(n_bytes);
        for v in x { x_bytes.extend_from_slice(&v.to_le_bytes()); }
        let mut y_bytes = vec![0u8; n_bytes];

        let mut args = [
            RemoteArg { buf: RemoteBuf { pv: prim_in.as_mut_ptr() as *mut c_void, nlen: 16 }},
            RemoteArg { buf: RemoteBuf { pv: x_bytes.as_mut_ptr() as *mut c_void, nlen: n_bytes }},
            RemoteArg { buf: RemoteBuf { pv: y_bytes.as_mut_ptr() as *mut c_void, nlen: n_bytes }},
        ];
        sess.invoke(make_scalars(3, 2, 1), &mut args)?;

        Ok(y_bytes.chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect())
    }

    let mut fails = 0;

    for (name, n, a_h) in [
        ("T_HVX_SCALE_BITWISE_64",    64usize,   100i16),
        ("T_HVX_SCALE_BITWISE_1024",  1024,      -200),
        ("T_HVX_SCALE_BITWISE_65536", 65536,     1234),
    ] {
        let x: Vec<i16> = (0..n).map(|i| ((i as i32 * 37 + 11) & 0x7FFF) as i16 - 16384).collect();
        let exp = scale_ref(&x, a_h);
        match invoke_scale(&sess, &x, a_h) {
            Ok(got) if got == exp => eprintln!("[hvx] {name} (n={n}, a_h={a_h}) PASS"),
            Ok(got) => {
                let idx = got.iter().zip(exp.iter()).position(|(a, b)| a != b);
                eprintln!("[hvx] {name} FAIL: diverge at {idx:?}  got={:?} exp={:?}",
                          idx.and_then(|i| got.get(i)), idx.and_then(|i| exp.get(i)));
                fails += 1;
            }
            Err(e) => { eprintln!("[hvx] {name} FAIL invoke: {e:?}"); fails += 1; }
        }
    }

    // T_HVX_SCALE_SATURATE: 30000 + 30000 should saturate to 32767
    {
        let x: Vec<i16> = vec![30000; 128];
        let a_h: i16 = 30000;
        let exp = scale_ref(&x, a_h);  // all 32767
        match invoke_scale(&sess, &x, a_h) {
            Ok(got) if got == exp && got.iter().all(|&v| v == 32767) => {
                eprintln!("[hvx] T_HVX_SCALE_SATURATE (clamp +) PASS");
            }
            Ok(got) => { eprintln!("[hvx] T_HVX_SCALE_SATURATE FAIL: got[0]={}", got[0]); fails += 1; }
            Err(e) => { eprintln!("[hvx] T_HVX_SCALE_SATURATE FAIL invoke: {e:?}"); fails += 1; }
        }
        // negative saturation: -30000 + -10000 → -32768
        let x: Vec<i16> = vec![-30000; 128];
        let a_h: i16 = -10000;
        let exp = scale_ref(&x, a_h);
        match invoke_scale(&sess, &x, a_h) {
            Ok(got) if got == exp && got.iter().all(|&v| v == -32768) => {
                eprintln!("[hvx] T_HVX_SCALE_SATURATE (clamp -) PASS");
            }
            Ok(got) => { eprintln!("[hvx] T_HVX_SCALE_SATURATE - FAIL: got[0]={}", got[0]); fails += 1; }
            Err(e) => { eprintln!("[hvx] T_HVX_SCALE_SATURATE - FAIL invoke: {e:?}"); fails += 1; }
        }
    }

    // T_HVX_SCALE_VS_SCALAR: 1000-iter 64 KB i16 elements wall.
    {
        const N: usize = 64 * 1024 / 2;  // 32 K elements = 64 KB i16
        const ITERS: u32 = 1000;
        let x: Vec<i16> = (0..N).map(|i| ((i as i32 * 31 + 7) & 0x7FFF) as i16 - 16384).collect();

        let t0 = Instant::now();
        for _ in 0..ITERS {
            let _ = invoke_scale(&sess, &x, 100).expect("invoke");
        }
        let dsp_wall = t0.elapsed();

        // Host scalar reference loop time, ignoring i/o
        let t0 = Instant::now();
        for _ in 0..ITERS {
            let _ = scale_ref(&x, 100);
        }
        let host_wall = t0.elapsed();

        eprintln!("[hvx] T_HVX_SCALE_VS_SCALAR (64 KB × {ITERS} iter): dsp_hvx {dsp_wall:?}  host_scalar {host_wall:?}");
        // No specific gate — finding.  The DSP path pays FastRPC round-trip
        // per call (~ms) which dominates for small payloads; the win is
        // measured per-byte at larger sizes + Sprint E batching.
    }

    drop(sess);
    eprintln!("[hvx] session closed cleanly");

    if fails == 0 {
        eprintln!("[hvx] ALL GATES PASS");
        std::process::exit(0);
    } else {
        eprintln!("[hvx] {fails} gate(s) FAILED");
        std::process::exit(1);
    }
}
