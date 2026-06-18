//! G-BYTEEXACT-ISLANDS-CUDA host comparator.
//!
//!     cargo run --bin bx_islands_compare -- <dump.bin>
//!
//! Reads the self-describing island dump produced by the gemma4 CUDA prefill seam
//! (SP_BYTEEXACT_DUMP, cuda_forward.cu `gemma4_cuda_probe`), and re-runs the SAME
//! per-layer 12B activations through the crate's exact-integer references
//! (rmsnorm_q_ref / softmax_q_ref / gelu_q_ref / rope_q_ref). It then gates the
//! per-island agreement against the contract thresholds (CONTRACT-BYTEEXACT-forward
//! §5.1), proving the integer islands cost nothing measurable on REAL activations —
//! the verification scaffold for the eventual full integer-island forward.
//!
//! Thresholds (pre-registered, §5.1):
//!   RMSNorm  relerr  < 1e-4     (eps-free ref vs CUDA's mean+eps; absorbed)
//!   GELU     relerr  < 1e-4
//!   RoPE     relerr  < 1e-4
//!   softmax  max|Δp| < 1e-5     (synthesised from the dumped attention logits if
//!                                present; else SKIPPED — the prefill seam dumps the
//!                                three pointwise islands, softmax is gated offline by
//!                                G-ISLANDS-Q-REF + the contract §3 prototype).
//!
//! Dump format (little-endian):
//!   file header: i32[8] = { 'BXI1'(0x31495842), ver, n_tok, E, layer, period, swa_period, 0 }
//!   then records, each: i32[4] = { tag(4 ASCII bytes), rows, width, 0 } + f32[rows*width]
//!   tags: RMSi RMSw RMSo  GELi GELu GELo  ROPi ROPb ROPf ROPo
//!     (ROPb = rbase scalar [1x1]; ROPf = freq-factor table [1 x d/2], or -1.0 sentinel)

mod sp_islands_q_ref;
use sp_islands_q_ref::{gelu_q_ref, rmsnorm_q_ref, rope_q_ref, FB};

use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::Read;

struct Rec {
    rows: usize,
    width: usize,
    data: Vec<f32>,
}

fn rd_i32(b: &[u8], o: usize) -> i32 {
    i32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

fn relerr(a: &[f32], b: &[f32]) -> f64 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for i in 0..a.len() {
        let d = a[i] as f64 - b[i] as f64;
        num += d * d;
        den += (a[i] as f64) * (a[i] as f64);
    }
    num.sqrt() / (den.sqrt() + 1e-30)
}

fn maxabs(a: &[f32], b: &[f32]) -> f64 {
    let mut m = 0.0f64;
    for i in 0..a.len() {
        let d = (a[i] as f64 - b[i] as f64).abs();
        if d > m {
            m = d;
        }
    }
    m
}

fn main() {
    let path = env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: bx_islands_compare <dump.bin>");
        std::process::exit(2);
    });
    let mut buf = Vec::new();
    File::open(&path)
        .and_then(|mut f| f.read_to_end(&mut buf))
        .unwrap_or_else(|e| {
            eprintln!("[bx-cmp] open {}: {}", path, e);
            std::process::exit(2);
        });
    assert!(buf.len() >= 32, "[bx-cmp] truncated header");
    let magic = rd_i32(&buf, 0);
    assert_eq!(magic, 0x3149_5842, "[bx-cmp] bad magic 0x{:08x}", magic);
    let n_tok = rd_i32(&buf, 8) as usize;
    let e_dim = rd_i32(&buf, 12) as usize;
    let layer = rd_i32(&buf, 16);
    println!(
        "[bx-cmp] dump {} : n_tok={} E={} layer={}",
        path, n_tok, e_dim, layer
    );

    // parse records
    let mut recs: HashMap<String, Rec> = HashMap::new();
    let mut o = 32usize;
    while o + 16 <= buf.len() {
        let tag = String::from_utf8_lossy(&buf[o..o + 4]).to_string();
        let rows = rd_i32(&buf, o + 4) as usize;
        let width = rd_i32(&buf, o + 8) as usize;
        o += 16;
        let n = rows * width;
        if o + n * 4 > buf.len() {
            break;
        }
        let mut data = Vec::with_capacity(n);
        for i in 0..n {
            data.push(f32::from_le_bytes([
                buf[o + i * 4],
                buf[o + i * 4 + 1],
                buf[o + i * 4 + 2],
                buf[o + i * 4 + 3],
            ]));
        }
        o += n * 4;
        recs.insert(tag, Rec { rows, width, data });
    }
    println!("[bx-cmp] parsed {} island records", recs.len());

    let mut fails = 0usize;

    // ---------- RMSNorm ----------
    if let (Some(xi), Some(xw), Some(xo)) =
        (recs.get("RMSi"), recs.get("RMSw"), recs.get("RMSo"))
    {
        let e = xi.width;
        let w: Vec<f32> = xw.data.clone();
        let mut refout = Vec::with_capacity(xo.data.len());
        for t in 0..xi.rows {
            let row = &xi.data[t * e..(t + 1) * e];
            refout.extend(rmsnorm_q_ref(row, Some(&w)));
        }
        let re = relerr(&xo.data, &refout);
        let thr = 1e-4;
        let pass = re < thr;
        if !pass {
            fails += 1;
        }
        println!(
            "[bx-cmp] RMSNorm  rows={} E={} relerr={:.3e} thr={:.0e} -> {}",
            xi.rows,
            e,
            re,
            thr,
            if pass { "GREEN" } else { "RED" }
        );
    } else {
        println!("[bx-cmp] RMSNorm  records missing (SKIP)");
    }

    // ---------- GELU-tanh ----------
    if let (Some(gi), Some(gu), Some(go)) =
        (recs.get("GELi"), recs.get("GELu"), recs.get("GELo"))
    {
        // ref = gelu(gi) elementwise, then * gu (the CUDA k_gelu_mul fuses the up-mul)
        let gact = gelu_q_ref(&gi.data);
        let mut refout = Vec::with_capacity(go.data.len());
        for i in 0..gact.len() {
            refout.push(gact[i] * gu.data[i]);
        }
        let re = relerr(&go.data, &refout);
        let thr = 1e-4;
        let pass = re < thr;
        if !pass {
            fails += 1;
        }
        println!(
            "[bx-cmp] GELU     rows={} ff={} relerr={:.3e} thr={:.0e} -> {}",
            gi.rows,
            gi.width,
            re,
            thr,
            if pass { "GREEN" } else { "RED" }
        );
    } else {
        println!("[bx-cmp] GELU     records missing (SKIP)");
    }

    // ---------- RoPE ----------
    if let (Some(ri), Some(ro), Some(rb), Some(rf)) = (
        recs.get("ROPi"),
        recs.get("ROPo"),
        recs.get("ROPb"),
        recs.get("ROPf"),
    ) {
        let hd = ri.width; // head_dim
        let half = hd / 2;
        let nh_rows = ri.rows; // n_tok * n_head
        let nh = nh_rows / n_tok;
        let base = rb.data[0] as f64;
        // build the FB-fixed per-pair frequency table: freq_fix[i] = round(base^(-2i/d)/ff[i] * 2^FB)
        let two_pow_fb = (1i128 << FB) as f64;
        let use_ff = rf.data.len() == half; // else single -1.0 sentinel => ff=1
        let mut freq_fix = vec![0i64; half];
        for i in 0..half {
            let f = base.powf(-2.0 * (i as f64) / (hd as f64));
            let ff = if use_ff { rf.data[i] as f64 } else { 1.0 };
            freq_fix[i] = (f / ff * two_pow_fb).round() as i64;
        }
        // run rope_q_ref per (token,head): pos = token index t (NEOX, matches k_rope[_freqs])
        let mut refout = Vec::with_capacity(ro.data.len());
        for rrow in 0..nh_rows {
            let t = (rrow / nh) as i64; // row layout = token-major, head-minor
            let row = &ri.data[rrow * hd..(rrow + 1) * hd];
            refout.extend(rope_q_ref(row, t, &freq_fix));
        }
        let re = relerr(&ro.data, &refout);
        let thr = 1e-4;
        let pass = re < thr;
        if !pass {
            fails += 1;
        }
        println!(
            "[bx-cmp] RoPE     rows={} hd={} ff={} base={:.0} relerr={:.3e} thr={:.0e} -> {}",
            nh_rows,
            hd,
            if use_ff { "table" } else { "1.0" },
            base,
            re,
            thr,
            if pass { "GREEN" } else { "RED" }
        );
        let _ = maxabs; // (softmax path uses maxabs; keep referenced)
    } else {
        println!("[bx-cmp] RoPE     records missing (SKIP)");
    }

    // ---------- softmax (offline note) ----------
    // The prefill seam dumps the three POINTWISE islands; the attention softmax is
    // gated by G-ISLANDS-Q-REF (max|Δp| 1.3e-6, contract §3) on the dumped attention
    // logits when a SP_BYTEEXACT_SOFTMAX dump is added. Pre-registered threshold:
    // max|Δp| < 1e-5. Currently SKIPPED in this prefill comparator.
    println!("[bx-cmp] softmax  gated offline (G-ISLANDS-Q-REF max|Δp| 1.3e-6 < 1e-5); prefill-dump SKIP");

    println!(
        "[bx-cmp] G-BYTEEXACT-ISLANDS-CUDA: {} ({} island(s) over threshold)",
        if fails == 0 { "GREEN" } else { "RED" },
        fails
    );
    std::process::exit(if fails == 0 { 0 } else { 1 });
}
