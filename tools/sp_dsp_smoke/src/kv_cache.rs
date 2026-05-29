//! §3-HX Sprint J (Path A1) — per-layer K + V DmaBuffer allocation.
//!
//! v0 up-front allocation at ctx_max=4096.  Sprint J.1 (if needed) makes
//! ctx_max configurable from sp_arch_info.context_length; lazy per-layer
//! growth is Sprint K material.  KV layout matches Sprint I/G compute
//! conventions: i16 per element, row-major [n_kv_heads, head_dim, ctx_max].

use crate::dsp_model::SpModelHeader;
use crate::dsp_rpc::{DmaBuffer, FastRpcSession, SpErr};
use std::time::Instant;

pub struct KvCache<'sess> {
    pub layers_k: Vec<DmaBuffer<'sess>>,
    pub layers_v: Vec<DmaBuffer<'sess>>,
    pub ctx_max: usize,
    pub per_layer_bytes: usize,
    pub alloc_wall_ms: u64,
}

impl<'sess> KvCache<'sess> {
    /// Allocate K + V DmaBuffer pair per layer.  Each is sized for the full
    /// ctx_max in i16: n_kv_heads × head_dim × ctx_max × 2 bytes.
    ///
    /// On partial allocation failure (heap exhausted mid-loop), prior
    /// DmaBuffers in layers_k/layers_v Drop in reverse declaration order
    /// via Rust's stack-unwind discipline.
    pub fn alloc(sess: &'sess FastRpcSession, hdr: &SpModelHeader, ctx_max: usize)
        -> Result<Self, SpErr>
    {
        let t0 = Instant::now();
        let per_layer_bytes = (hdr.n_kv_heads as usize)
                            * (hdr.head_dim as usize)
                            * ctx_max
                            * 2;  // i16
        let n_layers = hdr.n_layers as usize;
        let mut layers_k = Vec::with_capacity(n_layers);
        let mut layers_v = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            layers_k.push(sess.alloc_dma(per_layer_bytes)?);
            layers_v.push(sess.alloc_dma(per_layer_bytes)?);
        }
        Ok(KvCache {
            layers_k, layers_v,
            ctx_max,
            per_layer_bytes,
            alloc_wall_ms: t0.elapsed().as_millis() as u64,
        })
    }

    pub fn total_bytes(&self) -> u64 {
        (self.layers_k.len() + self.layers_v.len()) as u64 * self.per_layer_bytes as u64
    }
}
