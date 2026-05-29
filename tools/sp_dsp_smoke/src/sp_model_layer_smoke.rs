//! §3-HX Sprint I — single-layer FFN W_gate matmul smoke.
//!
//! Loads ONE 128×128 Q8 tile from a real Qwen3-0.6B `.sp-model` (blk.0
//! ffn_gate.weight + scale companion), pushes through the existing
//! Sprint G dual-VTCM Halide matmul kernel (Sprint H diag method,
//! `sp_compute_ffn_2stage_diag_halide`), verifies bit-identity against
//! the inline scalar reference at q_bits=14.
//!
//! Build:
//!     cargo build --target aarch64-linux-android --release --bin sp_model_layer_smoke
//!
//! Push + run:
//!     adb push qwen3_rt.sp-model /data/local/tmp/
//!     adb push sp_model_layer_smoke /data/local/tmp/
//!     adb shell chmod +x /data/local/tmp/sp_model_layer_smoke
//!     adb shell 'ADSP_LIBRARY_PATH="/data/local/tmp;" /data/local/tmp/sp_model_layer_smoke'
//!
//! Default model path: /data/local/tmp/qwen3_rt.sp-model
//! Override: `sp_model_layer_smoke <path>`

#[cfg(not(target_os = "android"))]
fn main() {
    eprintln!("sp_model_layer_smoke: host build skipped");
}

#[cfg(target_os = "android")]
mod dsp_rpc;
#[cfg(target_os = "android")]
mod sp_model_layer;

#[cfg(target_os = "android")]
fn main() {
    use dsp_rpc::{make_scalars, FastRpcSession, RemoteArg, RemoteBuf, SpErr};
    use sp_model_layer::{read_header, read_tensor_table, find_tensor,
                          read_layer_w_gate_tile, SP_MODEL_MAGIC_LE, SP_HEADER_SIZE,
                          SP_DT_OK_Q8, SP_DT_FROBENIUS_SCALE_FP32};
    use std::ffi::c_void;
    use std::time::Instant;

    let model_path = std::env::args().nth(1)
        .unwrap_or_else(|| "/data/local/tmp/qwen3_rt.sp-model".to_string());
    let mut fails = 0usize;

    // ─── Open session (Sprint G compute skel) ───────────────────────────────
    eprintln!("[I] opening FastRpcSession against sp_compute_skel (Path B)...");
    let sess = match FastRpcSession::new(
        "file:///libsp_compute_skel.so?sp_compute_skel_handle_invoke&_modver=1.0&_dom=cdsp")
    {
        Ok(s) => { eprintln!("[I] session open"); s }
        Err(e) => { eprintln!("[I] session FAIL: {e:?}"); std::process::exit(1); }
    };

    // ─── T_MODEL_HEADER_PARSE ───────────────────────────────────────────────
    let mut model_file = match std::fs::File::open(&model_path) {
        Ok(f) => f,
        Err(e) => { eprintln!("[I] cannot open '{model_path}': {e:?}"); std::process::exit(1); }
    };
    let hdr = match read_header(&mut model_file) {
        Ok(h) => h,
        Err(e) => { eprintln!("[I] T_MODEL_HEADER_PARSE FAIL: {e:?}"); std::process::exit(1); }
    };
    let header_ok =
        hdr.magic == SP_MODEL_MAGIC_LE &&
        hdr.version_major == 0 &&
        hdr.header_size == SP_HEADER_SIZE as u32 &&
        hdr.tensor_table_offset == 512 &&
        hdr.tensor_data_offset % 65536 == 0 &&
        hdr.tensor_count > 0;
    if header_ok {
        eprintln!("[I] T_MODEL_HEADER_PARSE PASS  (arch_id={} tensor_count={} data_offset=0x{:x} file_size={})",
                  hdr.arch_id, hdr.tensor_count, hdr.tensor_data_offset, hdr.file_size);
    } else {
        eprintln!("[I] T_MODEL_HEADER_PARSE FAIL: {hdr:?}");
        fails += 1;
    }

    // ─── Find W_gate tensor + scale companion ───────────────────────────────
    let table = match read_tensor_table(&mut model_file, &hdr) {
        Ok(t) => t,
        Err(e) => { eprintln!("[I] tensor table read FAIL: {e:?}"); std::process::exit(1); }
    };
    let w_name = "blk.0.ffn_gate.weight";
    let s_name = "blk.0.ffn_gate.weight.scale";
    let w_entry = match find_tensor(&table, w_name) {
        Some(e) => e,
        None => {
            eprintln!("[I] tensor '{w_name}' not found.  Dumping first 8 tensor names:");
            for (i, e) in table.iter().take(8).enumerate() {
                eprintln!("[I]   [{i}] {} dtype={}", e.name, e.dtype_id);
            }
            std::process::exit(1);
        }
    };
    let s_entry = match find_tensor(&table, s_name) {
        Some(e) => e,
        None => { eprintln!("[I] scale '{s_name}' not found"); std::process::exit(1); }
    };
    eprintln!("[I]   W_gate: dtype={} dims=[{},{}] size={} offset=0x{:x}",
              w_entry.dtype_id, w_entry.dims[0], w_entry.dims[1],
              w_entry.size_bytes, w_entry.offset_in_data);
    eprintln!("[I]   scale:  dtype={} dims=[{},{}] size={} offset=0x{:x}",
              s_entry.dtype_id, s_entry.dims[0], s_entry.dims[1],
              s_entry.size_bytes, s_entry.offset_in_data);
    if w_entry.dtype_id != SP_DT_OK_Q8 || s_entry.dtype_id != SP_DT_FROBENIUS_SCALE_FP32 {
        eprintln!("[I] dtype mismatch (expected Q8 weight + FP32 scale)");
        fails += 1;
    }

    // ─── T_DMA_TILE_LOAD ────────────────────────────────────────────────────
    const TILE_ROWS: usize = 128;
    const TILE_COLS: usize = 128;
    const Q_BITS: i32 = 14;
    const B_TERM: i32 = 0;
    const FP_SCALE: f32 = 64.0;
    let w_tile = match read_layer_w_gate_tile(
        &mut model_file, &hdr, w_entry, s_entry, 0, (TILE_ROWS, TILE_COLS), FP_SCALE)
    {
        Ok(t) => t,
        Err(e) => { eprintln!("[I] T_DMA_TILE_LOAD FAIL load: {e:?}"); std::process::exit(1); }
    };
    if w_tile.len() != TILE_ROWS * TILE_COLS {
        eprintln!("[I] T_DMA_TILE_LOAD FAIL: tile len {} expected {}", w_tile.len(), TILE_ROWS*TILE_COLS);
        fails += 1;
    } else {
        eprintln!("[I] T_DMA_TILE_LOAD PASS  ({}×{} = {} i16; w_tile[0..4]={:?})",
                  TILE_ROWS, TILE_COLS, w_tile.len(), &w_tile[..4]);
    }

    // ─── Scalar reference (matches test_hvx.rs:674-702 byte-for-byte) ───────
    fn ffn_2stage_ref_with_hidden(
        x: &[i16], w1: &[i16], w2: &[i16],
        batch: usize, d_in: usize, h_dim: usize, d_out: usize,
        b_term: i32, q_bits: i32,
    ) -> (Vec<i16>, Vec<i16>) {
        let mut y = vec![0i16; batch * d_out];
        let mut hidden = vec![0i16; batch * h_dim];
        for b in 0..batch {
            for h in 0..h_dim {
                let mut acc: i32 = 0;
                for d in 0..d_in {
                    let prod = (x[b*d_in + d] as i32) * (w1[h*d_in + d] as i32);
                    acc = acc.saturating_add(prod);
                }
                let s = (acc.saturating_add(b_term) >> q_bits).clamp(0, 32767);
                hidden[b*h_dim + h] = s as i16;
            }
        }
        for b in 0..batch {
            for c in 0..d_out {
                let mut acc: i32 = 0;
                for h in 0..h_dim {
                    let prod = (hidden[b*h_dim + h] as i32) * (w2[c*h_dim + h] as i32);
                    acc = acc.saturating_add(prod);
                }
                let s = (acc.saturating_add(b_term) >> q_bits).clamp(-32768, 32767);
                y[b*d_out + c] = s as i16;
            }
        }
        (y, hidden)
    }

    // ─── Halide diag invoker (mirrors test_hvx.rs:630-668 byte-for-byte) ────
    fn invoke_ffn_diag(
        sess: &FastRpcSession,
        x: &[i16], w1: &[i16], w2: &[i16],
        batch: i32, d_in: i32, h_dim: i32, d_out: i32,
        b_term: i32, q_bits: i32,
    ) -> Result<(Vec<i16>, Vec<i16>, i32, u64), SpErr> {
        let n_x  = (batch * d_in)  as usize * 2;
        let n_w1 = (h_dim * d_in)  as usize * 2;
        let n_w2 = (d_out * h_dim) as usize * 2;
        let n_y  = (batch * d_out) as usize * 2;
        let n_h  = (batch * h_dim) as usize * 2;
        let mut prim_in: [u32; 11] = [
            batch as u32, d_in as u32, h_dim as u32, d_out as u32,
            b_term as u32, q_bits as u32,
            n_x as u32, n_w1 as u32, n_w2 as u32, n_y as u32, n_h as u32,
        ];
        let mut prim_out: [u32; 3] = [0, 0, 0];
        let mut x_bytes  = Vec::with_capacity(n_x);  for v in x  { x_bytes.extend_from_slice(&v.to_le_bytes());  }
        let mut w1_bytes = Vec::with_capacity(n_w1); for v in w1 { w1_bytes.extend_from_slice(&v.to_le_bytes()); }
        let mut w2_bytes = Vec::with_capacity(n_w2); for v in w2 { w2_bytes.extend_from_slice(&v.to_le_bytes()); }
        let mut y_bytes  = vec![0u8; n_y];
        let mut h_bytes  = vec![0u8; n_h];
        let mut args = [
            RemoteArg { buf: RemoteBuf { pv: prim_in.as_mut_ptr()  as *mut c_void, nlen: 44 }},
            RemoteArg { buf: RemoteBuf { pv: x_bytes.as_mut_ptr()  as *mut c_void, nlen: n_x }},
            RemoteArg { buf: RemoteBuf { pv: w1_bytes.as_mut_ptr() as *mut c_void, nlen: n_w1 }},
            RemoteArg { buf: RemoteBuf { pv: w2_bytes.as_mut_ptr() as *mut c_void, nlen: n_w2 }},
            RemoteArg { buf: RemoteBuf { pv: prim_out.as_mut_ptr() as *mut c_void, nlen: 12 }},
            RemoteArg { buf: RemoteBuf { pv: y_bytes.as_mut_ptr()  as *mut c_void, nlen: n_y }},
            RemoteArg { buf: RemoteBuf { pv: h_bytes.as_mut_ptr()  as *mut c_void, nlen: n_h }},
        ];
        sess.invoke(make_scalars(9, 4, 3), &mut args)?;
        let y: Vec<i16> = y_bytes.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect();
        let hi: Vec<i16> = h_bytes.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect();
        let pcyc = ((prim_out[2] as u64) << 32) | (prim_out[1] as u64);
        Ok((y, hi, prim_out[0] as i32, pcyc))
    }

    // ─── T_LAYER_MATMUL_BITWISE ─────────────────────────────────────────────
    const BATCH: usize = 4;
    const D_IN:  usize = TILE_COLS;
    const H_DIM: usize = TILE_ROWS;
    const D_OUT: usize = 128;
    let w2_zeros: Vec<i16> = vec![0i16; D_OUT * H_DIM];

    let patterns: Vec<(&str, Vec<i16>)> = vec![
        ("sentinel",
         vec![0x1234i16; BATCH * D_IN]),
        ("pseudorandom",
         (0..BATCH * D_IN).map(|i| ((i as i32 * 37 + 11) & 0x7FFF) as i16 - 16384).collect()),
        ("all-ones",
         vec![1i16; BATCH * D_IN]),
    ];

    eprintln!("\n[I] ═══ T_LAYER_MATMUL_BITWISE (B={BATCH} D_in={D_IN} H={H_DIM} D_out={D_OUT} q={Q_BITS} b={B_TERM}) ═══");
    for (name, x) in &patterns {
        let (_exp_y, exp_h) = ffn_2stage_ref_with_hidden(
            x, &w_tile, &w2_zeros, BATCH, D_IN, H_DIM, D_OUT, B_TERM, Q_BITS);
        match invoke_ffn_diag(&sess, x, &w_tile, &w2_zeros,
                              BATCH as i32, D_IN as i32, H_DIM as i32, D_OUT as i32,
                              B_TERM, Q_BITS) {
            Ok((_got_y, got_h, vtcm, pcyc)) => {
                if got_h == exp_h {
                    eprintln!("[I]   pattern={:<13} PASS via {} (pcyc={}, hidden[0..4]={:?})",
                              name, if vtcm==1 {"VTCM"} else {"DDR"}, pcyc, &got_h[..4]);
                } else {
                    let idx = got_h.iter().zip(exp_h.iter()).position(|(a, c)| a != c);
                    eprintln!("[I]   pattern={:<13} FAIL: diverge at {idx:?}; got={:?} exp={:?}",
                              name,
                              idx.and_then(|i| got_h.get(i..(i+4).min(got_h.len()))),
                              idx.and_then(|i| exp_h.get(i..(i+4).min(exp_h.len()))));
                    fails += 1;
                }
            }
            Err(e) => { eprintln!("[I]   pattern={:<13} FAIL invoke: {e:?}", name); fails += 1; }
        }
    }

    // ─── T_LAYER_NO_HEAP_LEAK ───────────────────────────────────────────────
    // 100-iter cycle.  Re-uses the pseudorandom pattern to avoid re-paying
    // pattern-generation cost.
    eprintln!("\n[I] ═══ T_LAYER_NO_HEAP_LEAK (100 iter, pseudorandom pattern) ═══");
    let x_random = &patterns[1].1;
    let t0 = Instant::now();
    let mut leak_fail = 0;
    for i in 0..100 {
        match invoke_ffn_diag(&sess, x_random, &w_tile, &w2_zeros,
                              BATCH as i32, D_IN as i32, H_DIM as i32, D_OUT as i32,
                              B_TERM, Q_BITS) {
            Ok(_) => {}
            Err(e) => { eprintln!("[I]   iter {i} FAIL: {e:?}"); leak_fail += 1; break; }
        }
    }
    let elapsed = t0.elapsed();
    if leak_fail == 0 {
        eprintln!("[I]   100 iter completed in {elapsed:?}  ({:.1} ms/iter avg)",
                  elapsed.as_secs_f64() * 1000.0 / 100.0);
        eprintln!("[I] T_LAYER_NO_HEAP_LEAK PASS");
    } else {
        eprintln!("[I] T_LAYER_NO_HEAP_LEAK FAIL ({leak_fail} iterations errored)");
        fails += 1;
    }

    drop(sess);
    eprintln!("[I] session closed cleanly");

    if fails == 0 {
        eprintln!("[I] ALL GATES PASS");
        std::process::exit(0);
    } else {
        eprintln!("[I] {fails} gate(s) FAILED");
        std::process::exit(1);
    }
}
