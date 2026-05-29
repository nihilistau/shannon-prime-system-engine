//! §3-HX Sprint K v0.alpha — dispatch-parallelism dispatcher.
//!
//! `Arc<FastRpcSession>` shared between two ARM threads, each calling
//! `sp_compute_ffn_2stage_diag_halide` (method 9) concurrently on one
//! cdsp handle.  Measures wall-clock overlap to decide whether the
//! FastRPC + cdsp dual-vector-context stack actually parallelizes —
//! the load-bearing premise for K v0.beta's CRT-split kernel.
//!
//! Critical: invoke is synchronous (returns when cdsp work completes).
//! Mutex<FastRpcSession> would serialize at the lock-hold level by
//! construction; Arc allows both threads to call invoke(&self) at the
//! same time, leaving the parallelism question to FastRPC + cdsp.

use crate::dsp_rpc::{make_scalars, FastRpcSession, RemoteArg, RemoteBuf, SpErr};
use std::ffi::c_void;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

/// Per-invoke result returned from a single dispatch call.
pub struct InvokeResult {
    pub hidden: Vec<i16>,
    pub start: Instant,
    pub end: Instant,
    pub kernel_pcyc: u64,
}

pub struct DualDispatch {
    pub sess: Arc<FastRpcSession>,
}

impl DualDispatch {
    pub fn new(sess: Arc<FastRpcSession>) -> Self {
        DualDispatch { sess }
    }

    /// Single-thread invoke of Sprint H diag method 9.  Captures ARM-side
    /// timestamps + kernel pcycles for the overlap calculation.
    pub fn invoke_once(
        sess: &FastRpcSession,
        x: &[i16], w1: &[i16], w2: &[i16],
        batch: i32, d_in: i32, h_dim: i32, d_out: i32,
        b_term: i32, q_bits: i32,
    ) -> Result<InvokeResult, SpErr> {
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
        let start = Instant::now();
        sess.invoke(make_scalars(9, 4, 3), &mut args)?;
        let end = Instant::now();
        let kernel_pcyc = ((prim_out[2] as u64) << 32) | (prim_out[1] as u64);
        let hidden: Vec<i16> = h_bytes.chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]])).collect();
        Ok(InvokeResult { hidden, start, end, kernel_pcyc })
    }

    /// Dual-thread concurrent invoke.  Both threads call invoke on the
    /// same Arc<FastRpcSession> — concurrency at the FastRPC layer.
    /// Returns (result_a, result_b) for the dispatcher to analyze.
    pub fn dual_invoke(
        &self,
        x: Vec<i16>, w1: Vec<i16>, w2: Vec<i16>,
        batch: i32, d_in: i32, h_dim: i32, d_out: i32,
        b_term: i32, q_bits: i32,
    ) -> (Result<InvokeResult, SpErr>, Result<InvokeResult, SpErr>) {
        let sess_a = self.sess.clone();
        let sess_b = self.sess.clone();
        let (xa, w1a, w2a) = (x.clone(), w1.clone(), w2.clone());
        let (xb, w1b, w2b) = (x, w1, w2);
        let h_a = thread::spawn(move || {
            Self::invoke_once(&sess_a, &xa, &w1a, &w2a,
                              batch, d_in, h_dim, d_out, b_term, q_bits)
        });
        let h_b = thread::spawn(move || {
            Self::invoke_once(&sess_b, &xb, &w1b, &w2b,
                              batch, d_in, h_dim, d_out, b_term, q_bits)
        });
        let r_a = h_a.join().expect("thread A join");
        let r_b = h_b.join().expect("thread B join");
        (r_a, r_b)
    }
}

/// Overlap metrics computed from two `InvokeResult`s.
pub struct OverlapMetrics {
    pub wall_total_us: u128,
    pub overlap_us: u128,
    pub overlap_fraction: f64,
    pub kernel_pcyc_a: u64,
    pub kernel_pcyc_b: u64,
    pub kernel_pcyc_sum: u64,
    pub kernel_pcyc_max: u64,
    pub wall_a_us: u128,
    pub wall_b_us: u128,
}

impl OverlapMetrics {
    pub fn from(a: &InvokeResult, b: &InvokeResult) -> Self {
        let t0 = a.start.min(b.start);
        let t1 = a.end.max(b.end);
        let wall_total_us = t1.duration_since(t0).as_micros();
        let overlap_start = a.start.max(b.start);
        let overlap_end   = a.end.min(b.end);
        let overlap_us = if overlap_end > overlap_start {
            overlap_end.duration_since(overlap_start).as_micros()
        } else { 0 };
        let overlap_fraction = if wall_total_us > 0 {
            overlap_us as f64 / wall_total_us as f64
        } else { 0.0 };
        let wall_a_us = a.end.duration_since(a.start).as_micros();
        let wall_b_us = b.end.duration_since(b.start).as_micros();
        OverlapMetrics {
            wall_total_us, overlap_us, overlap_fraction,
            kernel_pcyc_a: a.kernel_pcyc,
            kernel_pcyc_b: b.kernel_pcyc,
            kernel_pcyc_sum: a.kernel_pcyc + b.kernel_pcyc,
            kernel_pcyc_max: a.kernel_pcyc.max(b.kernel_pcyc),
            wall_a_us, wall_b_us,
        }
    }
}
