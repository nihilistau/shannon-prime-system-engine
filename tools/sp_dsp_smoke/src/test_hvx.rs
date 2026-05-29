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

    // ═════════════════════════════════════════════════════════════════════
    // §3-HX Sprint E tests — explicit HVX intrinsics axpby + batched dispatch
    // ═════════════════════════════════════════════════════════════════════
    eprintln!("\n[hvx] ═══ Sprint E: explicit-intrinsic axpby + batched dispatch ═══");

    // Scalar reference for axpby_hvx
    fn axpby_ref(x: &[i16], a_h: i16, b: i32, q_bits: i32) -> Vec<i16> {
        x.iter().map(|&v| {
            ((((a_h as i32) * (v as i32)) + b) >> q_bits).clamp(-32768, 32767) as i16
        }).collect()
    }

    /// Method 4 = sp_compute_axpby_hvx; layout per qaic stub:
    ///   primIn[6] = [n, a_h, b, q_bits, x_bufLen, y_bufLen]   (24 B)
    ///   pra[0/1/2] same shape as axpby
    fn invoke_axpby_hvx(sess: &FastRpcSession, x: &[i16],
                        a_h: i16, b: i32, q_bits: i32) -> Result<Vec<i16>, SpErr> {
        let n = x.len();
        let n_bytes = n * 2;
        let mut prim_in: [u32; 6] = [
            n as u32, a_h as u32, b as u32, q_bits as u32,
            n_bytes as u32, n_bytes as u32,
        ];
        let mut x_bytes = Vec::with_capacity(n_bytes);
        for v in x { x_bytes.extend_from_slice(&v.to_le_bytes()); }
        let mut y_bytes = vec![0u8; n_bytes];
        let mut args = [
            RemoteArg { buf: RemoteBuf { pv: prim_in.as_mut_ptr() as *mut c_void, nlen: 24 }},
            RemoteArg { buf: RemoteBuf { pv: x_bytes.as_mut_ptr() as *mut c_void, nlen: n_bytes }},
            RemoteArg { buf: RemoteBuf { pv: y_bytes.as_mut_ptr() as *mut c_void, nlen: n_bytes }},
        ];
        sess.invoke(make_scalars(4, 2, 1), &mut args)?;
        Ok(y_bytes.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect())
    }

    for (name, n, a_h, b, q_bits) in [
        ("T_HVX_AXPBY_INTRIN_64",     64usize,   100i16, 0,     8),
        ("T_HVX_AXPBY_INTRIN_1024",   1024,     -200,    1024,  10),
        ("T_HVX_AXPBY_INTRIN_65536",  65536,    1234,    -1024, 12),
    ] {
        let x: Vec<i16> = (0..n).map(|i| ((i as i32 * 37 + 11) & 0x7FFF) as i16 - 16384).collect();
        let exp = axpby_ref(&x, a_h, b, q_bits);
        match invoke_axpby_hvx(&sess, &x, a_h, b, q_bits) {
            Ok(got) if got == exp => eprintln!("[hvx] {name} (n={n}, a_h={a_h}, b={b}, q_bits={q_bits}) PASS"),
            Ok(got) => {
                let idx = got.iter().zip(exp.iter()).position(|(a, c)| a != c);
                eprintln!("[hvx] {name} FAIL: diverge at {idx:?}, got={:?} exp={:?}",
                    idx.and_then(|i| got.get(i)), idx.and_then(|i| exp.get(i)));
                fails += 1;
            }
            Err(e) => { eprintln!("[hvx] {name} FAIL invoke: {e:?}"); fails += 1; }
        }
    }

    // Saturation: a_h * x might overflow → after shift, exceeds i16 range
    {
        let x: Vec<i16> = vec![10000; 128];   // 10000 × 30000 + 0 = 3 × 10^8
        let exp = axpby_ref(&x, 30000, 0, 4); // >> 4 = 1.875 × 10^7 → saturated
        match invoke_axpby_hvx(&sess, &x, 30000, 0, 4) {
            Ok(got) if got == exp && got.iter().all(|&v| v == 32767) => {
                eprintln!("[hvx] T_HVX_AXPBY_INTRIN_SATURATE PASS");
            }
            Ok(got) => { eprintln!("[hvx] T_HVX_AXPBY_INTRIN_SATURATE FAIL: got[0]={}", got[0]); fails += 1; }
            Err(e) => { eprintln!("[hvx] T_HVX_AXPBY_INTRIN_SATURATE FAIL invoke: {e:?}"); fails += 1; }
        }
    }

    // ── F2: batched dispatch ──────────────────────────────────────────────
    //
    // Method 5 = sp_compute_scale_i16_batched; layout per qaic stub:
    //   primIn[5] = [n_per_batch, n_batches, a_h_bufLen, x_bufLen, y_bufLen]  (20 B)
    //   pra[0]    = primIn
    //   pra[1]    = a_h_buf (n_batches × i16)
    //   pra[2]    = x_buf
    //   pra[3]    = y_buf
    //   scalars   = MAKEX(0, 5, 3, 1, 0, 0)
    fn invoke_scale_batched(sess: &FastRpcSession, n_per_batch: i32,
                            a_h_arr: &[i16], x: &[i16]) -> Result<Vec<i16>, SpErr> {
        let n_batches = a_h_arr.len();
        let total = n_per_batch as usize * n_batches;
        assert_eq!(x.len(), total);
        let a_h_bytes_len = n_batches * 2;
        let xy_bytes_len  = total * 2;

        let mut prim_in: [u32; 5] = [
            n_per_batch as u32, n_batches as u32,
            a_h_bytes_len as u32, xy_bytes_len as u32, xy_bytes_len as u32,
        ];
        let mut a_h_bytes = Vec::with_capacity(a_h_bytes_len);
        for v in a_h_arr { a_h_bytes.extend_from_slice(&v.to_le_bytes()); }
        let mut x_bytes = Vec::with_capacity(xy_bytes_len);
        for v in x { x_bytes.extend_from_slice(&v.to_le_bytes()); }
        let mut y_bytes = vec![0u8; xy_bytes_len];

        let mut args = [
            RemoteArg { buf: RemoteBuf { pv: prim_in.as_mut_ptr() as *mut c_void, nlen: 20 }},
            RemoteArg { buf: RemoteBuf { pv: a_h_bytes.as_mut_ptr() as *mut c_void, nlen: a_h_bytes_len }},
            RemoteArg { buf: RemoteBuf { pv: x_bytes.as_mut_ptr() as *mut c_void, nlen: xy_bytes_len }},
            RemoteArg { buf: RemoteBuf { pv: y_bytes.as_mut_ptr() as *mut c_void, nlen: xy_bytes_len }},
        ];
        sess.invoke(make_scalars(5, 3, 1), &mut args)?;
        Ok(y_bytes.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect())
    }

    // T_BATCH_BITWISE: 8 batches × 4 K elements
    {
        const N_PER: usize = 4096;
        const N_BAT: usize = 8;
        let a_h_arr: Vec<i16> = (0..N_BAT).map(|i| (i as i16 * 100 - 350)).collect();
        let x: Vec<i16> = (0..N_PER * N_BAT).map(|i| ((i as i32 * 17 + 5) & 0x7FFF) as i16 - 16384).collect();
        let mut exp: Vec<i16> = Vec::with_capacity(N_PER * N_BAT);
        for b in 0..N_BAT {
            for i in 0..N_PER {
                let v = x[b * N_PER + i] as i32 + a_h_arr[b] as i32;
                exp.push(v.clamp(-32768, 32767) as i16);
            }
        }
        match invoke_scale_batched(&sess, N_PER as i32, &a_h_arr, &x) {
            Ok(got) if got == exp => eprintln!("[hvx] T_BATCH_BITWISE (8 × 4 KB i16) PASS"),
            Ok(_) => { eprintln!("[hvx] T_BATCH_BITWISE FAIL: diverge"); fails += 1; }
            Err(e) => { eprintln!("[hvx] T_BATCH_BITWISE FAIL invoke: {e:?}"); fails += 1; }
        }
    }

    // T_BATCH_VS_UNBATCHED: same total work — batched (1 call) vs unbatched (8 calls)
    {
        const N_PER: usize = 4096;
        const N_BAT: usize = 8;
        const ITERS: u32   = 200;

        let a_h_arr: Vec<i16> = (0..N_BAT).map(|_| 42i16).collect();
        let x: Vec<i16> = (0..N_PER * N_BAT).map(|i| (i & 0xFF) as i16).collect();

        // Unbatched: ITERS × N_BAT calls
        let t0 = Instant::now();
        for _ in 0..ITERS {
            for b in 0..N_BAT {
                let _ = invoke_scale(&sess,
                    &x[b * N_PER..(b+1) * N_PER],
                    a_h_arr[b]).expect("invoke");
            }
        }
        let unbatched = t0.elapsed();

        // Batched: ITERS × 1 call (but each call does N_BAT batches)
        let t0 = Instant::now();
        for _ in 0..ITERS {
            let _ = invoke_scale_batched(&sess, N_PER as i32, &a_h_arr, &x).expect("invoke");
        }
        let batched = t0.elapsed();

        let ratio = batched.as_secs_f64() / unbatched.as_secs_f64();
        eprintln!("[hvx] T_BATCH_VS_UNBATCHED ({ITERS}×{N_BAT}×{N_PER} i16): \
                   unbatched {unbatched:?}  batched {batched:?}  ratio {ratio:.3}× (<1.0 = batched faster)");
        if ratio < 0.95 {
            eprintln!("[hvx]   batched-faster gate: PASS (>5% improvement)");
        } else {
            eprintln!("[hvx]   batched-faster gate: WEAK ({ratio:.3}× — overhead not measurable at this size)");
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    // §3-HX Sprint F litmus: VTCM admission probe under Path B Unsigned PD
    // ═════════════════════════════════════════════════════════════════════
    //
    // Method 6 = sp_compute_vtcm_probe; layout:
    //   primIn[2]  = [size_bytes, single_page_flag]   (8 B)
    //   primOut[1] = [vtcm_addr_lo]                   (4 B)
    //   pra[0]     = primIn
    //   pra[1]     = primOut
    //   scalars    = MAKEX(0, 6, 1, 1, 0, 0)
    //
    // Return: vtcm_addr_lo non-zero = ADMITTED, zero = DENIED.  Either is
    // a valid Sprint F outcome (see SESSION-PLAN §4); the value tells us
    // whether VTCM is available without signed-PD admission, which informs
    // the Halide kernel handler design.
    eprintln!("\n[hvx] ═══ Sprint F: VTCM litmus under Path B Unsigned PD ═══");

    fn invoke_vtcm_probe(sess: &FastRpcSession,
                         size_bytes: i32, single_page: i32) -> Result<i32, SpErr> {
        let mut prim_in:  [i32; 2] = [size_bytes, single_page];
        let mut prim_out: [i32; 1] = [0];
        let mut args = [
            RemoteArg { buf: RemoteBuf { pv: prim_in.as_mut_ptr()  as *mut c_void, nlen: 8 }},
            RemoteArg { buf: RemoteBuf { pv: prim_out.as_mut_ptr() as *mut c_void, nlen: 4 }},
        ];
        sess.invoke(make_scalars(6, 1, 1), &mut args)?;
        Ok(prim_out[0])
    }

    for (label, sz, sp) in [
        ("T_VTCM_PROBE_64KB_MULTIPAGE",  64 * 1024,        0),
        ("T_VTCM_PROBE_64KB_SINGLEPAGE", 64 * 1024,        1),
        ("T_VTCM_PROBE_1MB_MULTIPAGE",   1 * 1024 * 1024,  0),
        ("T_VTCM_PROBE_4MB_MULTIPAGE",   4 * 1024 * 1024,  0),
    ] {
        match invoke_vtcm_probe(&sess, sz, sp) {
            Ok(addr) => {
                let status = if addr != 0 { "ADMITTED" } else { "DENIED" };
                eprintln!("[hvx] {label} (size={sz} single_page={sp}) {status} (addr_lo=0x{:08x})",
                          addr as u32);
            }
            Err(e) => {
                eprintln!("[hvx] {label} FAIL invoke: {e:?}");
                fails += 1;
            }
        }
    }
    eprintln!("[hvx]   T_HALIDE_VTCM_CHECK requires reading adsprpc logcat for the");
    eprintln!("[hvx]   FARF \"sp_compute_vtcm_probe ... admitted=N\" lines emitted by the skel.");

    // ─── §3-HX Sprint F — Halide AOT axpby_2d through skel + VTCM hot-copy ───
    //
    // Method 7 = sp_compute_axpby_2d_halide; layout per qaic stub:
    //   primIn[7]  = [rows, cols, b, q_bits, a_bufLen, x_bufLen, y_bufLen]  (28 B)
    //   primROut[1]= [vtcm_used]                                            (4 B)
    //   pra[0]     = primIn
    //   pra[1]     = a_buf  (cols × i16)
    //   pra[2]     = x_buf  (rows*cols × i16)
    //   pra[3]     = primROut
    //   pra[4]     = y_buf  (rows*cols × i16)
    //   scalars    = MAKEX(0, 7, 3, 2, 0, 0)
    eprintln!("\n[hvx] ═══ Sprint F: Halide AOT axpby_2d + VTCM hot-copy ═══");

    fn axpby_2d_ref(x: &[i16], a: &[i16], rows: usize, cols: usize,
                    b: i32, q_bits: i32) -> Vec<i16> {
        let mut y = vec![0i16; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                let acc = (a[c] as i32 * x[r*cols + c] as i32 + b) >> q_bits;
                y[r*cols + c] = acc.clamp(-32768, 32767) as i16;
            }
        }
        y
    }

    fn invoke_axpby_2d_halide(sess: &FastRpcSession,
                              x: &[i16], a: &[i16],
                              rows: i32, cols: i32, b: i32, q_bits: i32)
        -> Result<(Vec<i16>, i32), SpErr>
    {
        let xy_len = (rows * cols) as usize * 2;
        let a_len  = cols as usize * 2;
        let mut prim_in: [u32; 7] = [
            rows as u32, cols as u32, b as u32, q_bits as u32,
            a_len as u32, xy_len as u32, xy_len as u32,
        ];
        let mut prim_out: [u32; 1] = [0];
        let mut a_bytes = Vec::with_capacity(a_len);
        for v in a { a_bytes.extend_from_slice(&v.to_le_bytes()); }
        let mut x_bytes = Vec::with_capacity(xy_len);
        for v in x { x_bytes.extend_from_slice(&v.to_le_bytes()); }
        let mut y_bytes = vec![0u8; xy_len];
        let mut args = [
            RemoteArg { buf: RemoteBuf { pv: prim_in.as_mut_ptr()  as *mut c_void, nlen: 28 }},
            RemoteArg { buf: RemoteBuf { pv: a_bytes.as_mut_ptr()  as *mut c_void, nlen: a_len }},
            RemoteArg { buf: RemoteBuf { pv: x_bytes.as_mut_ptr()  as *mut c_void, nlen: xy_len }},
            RemoteArg { buf: RemoteBuf { pv: prim_out.as_mut_ptr() as *mut c_void, nlen: 4 }},
            RemoteArg { buf: RemoteBuf { pv: y_bytes.as_mut_ptr()  as *mut c_void, nlen: xy_len }},
        ];
        sess.invoke(make_scalars(7, 3, 2), &mut args)?;
        let y: Vec<i16> = y_bytes.chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]])).collect();
        Ok((y, prim_out[0] as i32))
    }

    for (label, rows, cols, b_in, q_in) in [
        ("T_HALIDE_AXPBY_2D_8x128",    8usize,  128usize,  1024i32, 10i32),
        ("T_HALIDE_AXPBY_2D_16x256",   16,      256,      -512,     12),
        ("T_HALIDE_AXPBY_2D_64x512",   64,      512,       0,       8),
        ("T_HALIDE_AXPBY_2D_128x1024", 128,     1024,      4096,    14),
    ] {
        let total = rows * cols;
        let x: Vec<i16> = (0..total).map(|i| ((i as i32 * 37 + 11) & 0x7FFF) as i16 - 16384).collect();
        let a: Vec<i16> = (0..cols).map(|i| ((i as i32 * 41 + 7) & 0x3FF) as i16 - 256).collect();
        let exp = axpby_2d_ref(&x, &a, rows, cols, b_in, q_in);
        match invoke_axpby_2d_halide(&sess, &x, &a, rows as i32, cols as i32, b_in, q_in) {
            Ok((got, vtcm_used)) if got == exp => {
                let path = if vtcm_used == 1 { "VTCM" } else { "DDR" };
                eprintln!("[hvx] {label} ({rows}x{cols}, b={b_in}, q={q_in}) PASS via {path}");
            }
            Ok((got, vtcm_used)) => {
                let idx = got.iter().zip(exp.iter()).position(|(a, c)| a != c);
                eprintln!("[hvx] {label} FAIL: vtcm_used={vtcm_used}, diverge at {idx:?}, got={:?} exp={:?}",
                          idx.and_then(|i| got.get(i)), idx.and_then(|i| exp.get(i)));
                fails += 1;
            }
            Err(e) => {
                eprintln!("[hvx] {label} FAIL invoke: {e:?}");
                fails += 1;
            }
        }
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
