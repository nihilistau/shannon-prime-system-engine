//! §3-HX Sprint J (Path A1) — full Qwen3 model loader into per-tensor DmaBuffers.
//!
//! Scales Sprint I's single-tile loader pattern to all 28 layers + globals.
//! Lives in sp_dsp_smoke (per Path A1 pivot — sp_daemon cross-compile blocker,
//! see lattice plan + the closure note's Sprint J.5 follow-on).  Sprint K's
//! internal CRT split consumes this model directly; sp_daemon AppState
//! integration is the orthogonal Sprint J.5 work.
//!
//! Reuses .sp-model parser primitives (header + tensor table + Q8 dequant)
//! from Sprint I (see sp_model_layer.rs in this same crate).  This module
//! re-implements the parser inline rather than depending on it to keep each
//! Sprint's smoke binary self-contained.

use crate::dsp_rpc::{DmaBuffer, FastRpcSession, SpErr};
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::marker::PhantomData;
use std::time::Instant;

pub const SP_MODEL_MAGIC_LE: u32 = 0x444D_5053;
pub const SP_HEADER_SIZE: usize = 512;
pub const SP_TENSOR_ENTRY_SIZE: usize = 256;
pub const SP_DT_OK_Q8: u32 = 10;
pub const SP_DT_FROBENIUS_SCALE_FP32: u32 = 12;
pub const SP_DT_F32: u32 = 1;

#[derive(Debug, Clone)]
pub struct SpModelHeader {
    pub magic: u32,
    pub version_major: u16,
    pub version_minor: u16,
    pub arch_id: u32,
    pub tensor_count: u32,
    pub tensor_table_offset: u64,
    pub tensor_data_offset: u64,
    pub file_size: u64,
    // arch_struct (Qwen3) decoded fields (see Sprint J plan §0.B for layout):
    pub vocab_size: u32,
    pub hidden_size: u32,
    pub n_layers: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub intermediate_size: u32,
}

#[derive(Debug, Clone)]
pub struct SpTensorEntry {
    pub name: String,
    pub dtype_id: u32,
    pub n_dims: u32,
    pub dims: [u64; 8],
    pub offset_in_data: u64,
    pub size_bytes: u64,
}

fn rd_u16(b: &[u8], o: usize) -> u16 { u16::from_le_bytes([b[o], b[o+1]]) }
fn rd_u32(b: &[u8], o: usize) -> u32 { u32::from_le_bytes([b[o], b[o+1], b[o+2], b[o+3]]) }
fn rd_u64(b: &[u8], o: usize) -> u64 {
    u64::from_le_bytes([b[o], b[o+1], b[o+2], b[o+3], b[o+4], b[o+5], b[o+6], b[o+7]])
}

pub fn read_header(file: &mut File) -> io::Result<SpModelHeader> {
    file.seek(SeekFrom::Start(0))?;
    let mut buf = [0u8; SP_HEADER_SIZE];
    file.read_exact(&mut buf)?;
    let magic = rd_u32(&buf, 0);
    if magic != SP_MODEL_MAGIC_LE {
        return Err(io::Error::new(io::ErrorKind::InvalidData,
            format!("bad magic 0x{:08x}", magic)));
    }
    let arch = &buf[24..24+56];
    Ok(SpModelHeader {
        magic,
        version_major: rd_u16(&buf, 4),
        version_minor: rd_u16(&buf, 6),
        arch_id:       rd_u32(&buf, 12),
        tensor_count:        rd_u32(&buf, 316),
        tensor_table_offset: rd_u64(&buf, 320),
        tensor_data_offset:  rd_u64(&buf, 328),
        file_size:           rd_u64(&buf, 336),
        vocab_size:          rd_u32(arch, 4),
        hidden_size:         rd_u32(arch, 8),
        n_layers:            rd_u32(arch, 12),
        n_heads:             rd_u32(arch, 16),
        n_kv_heads:          rd_u32(arch, 20),
        head_dim:            rd_u32(arch, 24),
        intermediate_size:   rd_u32(arch, 44),
    })
}

pub fn read_tensor_table(file: &mut File, hdr: &SpModelHeader)
    -> io::Result<Vec<SpTensorEntry>>
{
    file.seek(SeekFrom::Start(hdr.tensor_table_offset))?;
    let n = hdr.tensor_count as usize;
    let mut all = vec![0u8; n * SP_TENSOR_ENTRY_SIZE];
    file.read_exact(&mut all)?;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let base = i * SP_TENSOR_ENTRY_SIZE;
        let name_bytes = &all[base..base + 80];
        let nlen = name_bytes.iter().position(|&b| b == 0).unwrap_or(80);
        let name = String::from_utf8_lossy(&name_bytes[..nlen]).into_owned();
        let mut dims = [0u64; 8];
        for d in 0..8 { dims[d] = rd_u64(&all, base + 88 + d * 8); }
        out.push(SpTensorEntry {
            name,
            dtype_id: rd_u32(&all, base + 80),
            n_dims:   rd_u32(&all, base + 84),
            dims,
            offset_in_data: rd_u64(&all, base + 152),
            size_bytes:     rd_u64(&all, base + 160),
        });
    }
    Ok(out)
}

pub fn find_tensor<'a>(table: &'a [SpTensorEntry], name: &str)
    -> Option<&'a SpTensorEntry>
{
    table.iter().find(|e| e.name == name)
}

// ── DspModel ───────────────────────────────────────────────────────────────

pub struct LayerWeights<'sess> {
    pub w_gate:   DmaBuffer<'sess>,
    pub w_up:     DmaBuffer<'sess>,
    pub w_down:   DmaBuffer<'sess>,
    pub w_q:      DmaBuffer<'sess>,
    pub w_k:      DmaBuffer<'sess>,
    pub w_v:      DmaBuffer<'sess>,
    pub w_o:      DmaBuffer<'sess>,
    pub attn_norm_scale: DmaBuffer<'sess>,
    pub ffn_norm_scale:  DmaBuffer<'sess>,
}

pub struct DspModel<'sess> {
    pub header:          SpModelHeader,
    pub embedding:       DmaBuffer<'sess>,
    pub output_norm:     DmaBuffer<'sess>,
    pub output_proj:     Option<DmaBuffer<'sess>>,
    pub layers:          Vec<LayerWeights<'sess>>,
    pub load_wall_ms:    u64,
    pub total_dma_bytes: u64,
    _marker: PhantomData<&'sess FastRpcSession>,
}

const FP_SCALE: f32 = 64.0;

fn load_q8_weight<'a>(file: &mut File, hdr: &SpModelHeader,
                      table: &[SpTensorEntry], name: &str,
                      sess: &'a FastRpcSession)
    -> Result<DmaBuffer<'a>, SpErr>
{
    let w = find_tensor(table, name)
        .ok_or_else(|| SpErr::Other(format!("missing tensor: {}", name)))?;
    let scale_name = format!("{}.scale", name);
    let s = find_tensor(table, &scale_name)
        .ok_or_else(|| SpErr::Other(format!("missing scale: {}", scale_name)))?;
    if w.dtype_id != SP_DT_OK_Q8 || s.dtype_id != SP_DT_FROBENIUS_SCALE_FP32 {
        return Err(SpErr::Other(format!(
            "{} dtype mismatch (w={}, s={})", name, w.dtype_id, s.dtype_id)));
    }
    let inner = w.dims[0] as usize;
    let outer = w.dims[1] as usize;
    let n_i16 = inner * outer;
    let mut dma = sess.alloc_dma(n_i16 * 2)?;

    let mut q8_buf = vec![0u8; w.size_bytes as usize];
    file.seek(SeekFrom::Start(hdr.tensor_data_offset + w.offset_in_data))
        .map_err(|e| SpErr::Other(format!("seek: {e:?}")))?;
    file.read_exact(&mut q8_buf)
        .map_err(|e| SpErr::Other(format!("read q8: {e:?}")))?;
    let mut scales = vec![0f32; outer];
    file.seek(SeekFrom::Start(hdr.tensor_data_offset + s.offset_in_data))
        .map_err(|e| SpErr::Other(format!("seek scale: {e:?}")))?;
    let scale_bytes = unsafe {
        std::slice::from_raw_parts_mut(scales.as_mut_ptr() as *mut u8, outer * 4)
    };
    file.read_exact(scale_bytes)
        .map_err(|e| SpErr::Other(format!("read scale: {e:?}")))?;

    let dst = unsafe {
        std::slice::from_raw_parts_mut(dma.as_mut_ptr() as *mut i16, n_i16)
    };
    for r in 0..outer {
        let sf = scales[r] * FP_SCALE;
        let base = r * inner;
        for c in 0..inner {
            let v = q8_buf[base + c] as i8 as f32 * sf;
            dst[base + c] = v.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16;
        }
    }
    Ok(dma)
}

fn load_fp32_tensor<'a>(file: &mut File, hdr: &SpModelHeader,
                        table: &[SpTensorEntry], name: &str,
                        sess: &'a FastRpcSession)
    -> Result<DmaBuffer<'a>, SpErr>
{
    let t = find_tensor(table, name)
        .ok_or_else(|| SpErr::Other(format!("missing tensor: {}", name)))?;
    let mut dma = sess.alloc_dma(t.size_bytes as usize)?;
    file.seek(SeekFrom::Start(hdr.tensor_data_offset + t.offset_in_data))
        .map_err(|e| SpErr::Other(format!("seek: {e:?}")))?;
    file.read_exact(dma.as_mut_slice())
        .map_err(|e| SpErr::Other(format!("read: {e:?}")))?;
    Ok(dma)
}

impl<'sess> DspModel<'sess> {
    pub fn load(sess: &'sess FastRpcSession, path: &str) -> Result<Self, SpErr> {
        let t0 = Instant::now();
        let mut file = File::open(path)
            .map_err(|e| SpErr::Other(format!("open {path}: {e:?}")))?;
        let hdr = read_header(&mut file)
            .map_err(|e| SpErr::Other(format!("header: {e:?}")))?;
        let table = read_tensor_table(&mut file, &hdr)
            .map_err(|e| SpErr::Other(format!("tensor table: {e:?}")))?;

        let embedding = load_q8_weight(&mut file, &hdr, &table,
                                        "token_embd.weight", sess)?;
        let output_norm = load_fp32_tensor(&mut file, &hdr, &table,
                                            "output_norm.weight", sess)?;
        let output_proj = match find_tensor(&table, "output.weight") {
            Some(_) => Some(load_q8_weight(&mut file, &hdr, &table,
                                            "output.weight", sess)?),
            None => None,
        };

        let mut layers = Vec::with_capacity(hdr.n_layers as usize);
        for i in 0..hdr.n_layers {
            let p = |suffix: &str| format!("blk.{i}.{suffix}");
            layers.push(LayerWeights {
                w_gate: load_q8_weight(&mut file, &hdr, &table,
                                        &p("ffn_gate.weight"), sess)?,
                w_up:   load_q8_weight(&mut file, &hdr, &table,
                                        &p("ffn_up.weight"), sess)?,
                w_down: load_q8_weight(&mut file, &hdr, &table,
                                        &p("ffn_down.weight"), sess)?,
                w_q:    load_q8_weight(&mut file, &hdr, &table,
                                        &p("attn_q.weight"), sess)?,
                w_k:    load_q8_weight(&mut file, &hdr, &table,
                                        &p("attn_k.weight"), sess)?,
                w_v:    load_q8_weight(&mut file, &hdr, &table,
                                        &p("attn_v.weight"), sess)?,
                w_o:    load_q8_weight(&mut file, &hdr, &table,
                                        &p("attn_output.weight"), sess)?,
                attn_norm_scale: load_fp32_tensor(&mut file, &hdr, &table,
                                                    &p("attn_norm.weight"), sess)?,
                ffn_norm_scale:  load_fp32_tensor(&mut file, &hdr, &table,
                                                    &p("ffn_norm.weight"), sess)?,
            });
        }

        let load_wall_ms = t0.elapsed().as_millis() as u64;
        let mut total_dma_bytes = 0u64;
        for l in &layers {
            total_dma_bytes += (l.w_gate.len() + l.w_up.len() + l.w_down.len()
                              + l.w_q.len() + l.w_k.len() + l.w_v.len()
                              + l.w_o.len() + l.attn_norm_scale.len()
                              + l.ffn_norm_scale.len()) as u64;
        }
        total_dma_bytes += embedding.len() as u64 + output_norm.len() as u64;
        if let Some(op) = &output_proj { total_dma_bytes += op.len() as u64; }

        Ok(DspModel {
            header: hdr,
            embedding,
            output_norm,
            output_proj,
            layers,
            load_wall_ms,
            total_dma_bytes,
            _marker: PhantomData,
        })
    }
}
