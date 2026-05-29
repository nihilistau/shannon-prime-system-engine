//! §3-HX Sprint I — minimal .sp-model parser for single-tile weight loading.
//!
//! Format spec: shannon-prime-system/include/sp/sp_model.h (PPT-LAT-SP-MODEL-v0).
//! - 512 B header at file start (magic "SPMD", version major=0, minor=1)
//! - tensor table at `tensor_table_offset` (= 512), `tensor_count` × 256 B entries
//! - tensor data region at `tensor_data_offset` (multiple of 65536)
//!
//! Sprint I scope: read one 128×128 Q8 tile of one Q8 weight tensor + its
//! per-row FP32 scale companion, dequantize to i16 for the existing Sprint G
//! Halide matmul.  No Fix B aliasing (Sprint J).  No caching across calls.

use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};

// "SPMD" little-endian bytes are 0x53 0x50 0x4D 0x44 → u32 0x444D_5053.
pub const SP_MODEL_MAGIC_LE: u32 = 0x444D_5053;
pub const SP_HEADER_SIZE: usize = 512;
pub const SP_TENSOR_ENTRY_SIZE: usize = 256;
pub const SP_DT_OK_Q8: u32 = 10;
pub const SP_DT_FROBENIUS_SCALE_FP32: u32 = 12;

/// 512-byte header per sp_model.h §3.  Only fields Sprint I uses are
/// exposed by name; the rest stay in the raw byte buffer for future readers.
#[derive(Debug, Clone)]
pub struct SpModelHeader {
    pub magic: u32,
    pub version_major: u16,
    pub version_minor: u16,
    pub header_size: u32,
    pub arch_id: u32,
    pub tensor_count: u32,
    pub tensor_table_offset: u64,
    pub tensor_data_offset: u64,
    pub file_size: u64,
}

/// 256-byte tensor table entry per sp_model.h §4.
#[derive(Debug, Clone)]
pub struct SpTensorEntry {
    pub name: String,           // null-trimmed
    pub dtype_id: u32,
    pub n_dims: u32,
    pub dims: [u64; 8],
    pub offset_in_data: u64,
    pub size_bytes: u64,
    pub block_size: u32,
    pub block_count: u32,
}

fn rd_u16(buf: &[u8], off: usize) -> u16 { u16::from_le_bytes([buf[off], buf[off+1]]) }
fn rd_u32(buf: &[u8], off: usize) -> u32 { u32::from_le_bytes([buf[off], buf[off+1], buf[off+2], buf[off+3]]) }
fn rd_u64(buf: &[u8], off: usize) -> u64 {
    u64::from_le_bytes([buf[off], buf[off+1], buf[off+2], buf[off+3],
                        buf[off+4], buf[off+5], buf[off+6], buf[off+7]])
}

pub fn read_header(file: &mut File) -> io::Result<SpModelHeader> {
    file.seek(SeekFrom::Start(0))?;
    let mut buf = [0u8; SP_HEADER_SIZE];
    file.read_exact(&mut buf)?;
    let magic = rd_u32(&buf, 0);
    if magic != SP_MODEL_MAGIC_LE {
        return Err(io::Error::new(io::ErrorKind::InvalidData,
            format!("bad magic 0x{:08x} (expected 0x{:08x})", magic, SP_MODEL_MAGIC_LE)));
    }
    let version_major = rd_u16(&buf, 4);
    if version_major != 0 {
        return Err(io::Error::new(io::ErrorKind::InvalidData,
            format!("bad version_major {} (expected 0)", version_major)));
    }
    Ok(SpModelHeader {
        magic,
        version_major,
        version_minor: rd_u16(&buf, 6),
        header_size: rd_u32(&buf, 8),
        arch_id: rd_u32(&buf, 12),
        tensor_count: rd_u32(&buf, 316),
        tensor_table_offset: rd_u64(&buf, 320),
        tensor_data_offset: rd_u64(&buf, 328),
        file_size: rd_u64(&buf, 336),
    })
}

pub fn read_tensor_table(file: &mut File, hdr: &SpModelHeader) -> io::Result<Vec<SpTensorEntry>> {
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
            dtype_id:        rd_u32(&all, base + 80),
            n_dims:          rd_u32(&all, base + 84),
            dims,
            offset_in_data:  rd_u64(&all, base + 152),
            size_bytes:      rd_u64(&all, base + 160),
            block_size:      rd_u32(&all, base + 168),
            block_count:     rd_u32(&all, base + 172),
        });
    }
    Ok(out)
}

pub fn find_tensor<'a>(table: &'a [SpTensorEntry], name: &str) -> Option<&'a SpTensorEntry> {
    table.iter().find(|e| e.name == name)
}

/// Read a tile from a Q8 weight + its per-row FP32 scale, dequantize to i16.
///
/// Layout assumption (matches the engine's per-row Q8 arena format):
/// - `w_entry.size_bytes` covers the full Q8 weight (i8 row-major).
/// - `scale_entry.size_bytes` covers one f32 per row of the weight.
/// - `dims[0]` is the inner dim (cols, = hidden_size for `blk.N.ffn_gate.weight`).
/// - `dims[1]` is the outer dim (rows, = intermediate_size).
///
/// `tile_row_start` is the first row index to read; `tile_dim = (rows, cols)`.
/// `fixed_point_scale` converts the f32 dequant to i16 (multiply + round + clamp).
///
/// The returned `Vec<i16>` is `tile_dim.0 * tile_dim.1` elements, row-major:
/// element at `(r, c)` is at index `r * tile_dim.1 + c`.
pub fn read_layer_w_gate_tile(
    file: &mut File,
    hdr: &SpModelHeader,
    w_entry: &SpTensorEntry,
    scale_entry: &SpTensorEntry,
    tile_row_start: usize,
    tile_dim: (usize, usize),
    fixed_point_scale: f32,
) -> io::Result<Vec<i16>> {
    if w_entry.dtype_id != SP_DT_OK_Q8 {
        return Err(io::Error::new(io::ErrorKind::InvalidData,
            format!("weight '{}' is not Q8 (dtype_id={})", w_entry.name, w_entry.dtype_id)));
    }
    if scale_entry.dtype_id != SP_DT_FROBENIUS_SCALE_FP32 {
        return Err(io::Error::new(io::ErrorKind::InvalidData,
            format!("scale '{}' is not FROBENIUS_SCALE_FP32 (dtype_id={})",
                    scale_entry.name, scale_entry.dtype_id)));
    }
    let inner = w_entry.dims[0] as usize;  // cols per row
    if tile_dim.1 > inner {
        return Err(io::Error::new(io::ErrorKind::InvalidData,
            format!("tile cols {} > weight inner dim {}", tile_dim.1, inner)));
    }
    // Read tile_dim.0 rows × tile_dim.1 cols of Q8 bytes (one row at a time,
    // because we read only the first `tile_dim.1` cols of each row).
    let mut tile_q8 = vec![0i8; tile_dim.0 * tile_dim.1];
    for r in 0..tile_dim.0 {
        let row_offset = hdr.tensor_data_offset + w_entry.offset_in_data
                       + ((tile_row_start + r) * inner) as u64;
        file.seek(SeekFrom::Start(row_offset))?;
        let row_buf = unsafe {
            std::slice::from_raw_parts_mut(
                tile_q8.as_mut_ptr().add(r * tile_dim.1) as *mut u8,
                tile_dim.1)
        };
        file.read_exact(row_buf)?;
    }
    // Read per-row scales (f32 each).
    let mut scales = vec![0f32; tile_dim.0];
    let scale_offset = hdr.tensor_data_offset + scale_entry.offset_in_data
                     + (tile_row_start * 4) as u64;
    file.seek(SeekFrom::Start(scale_offset))?;
    let scale_bytes = unsafe {
        std::slice::from_raw_parts_mut(
            scales.as_mut_ptr() as *mut u8,
            tile_dim.0 * 4)
    };
    file.read_exact(scale_bytes)?;
    // Dequantize: i16 = clamp(round(i8 * row_scale * fixed_point_scale)).
    let mut out = Vec::with_capacity(tile_dim.0 * tile_dim.1);
    for r in 0..tile_dim.0 {
        let s = scales[r] * fixed_point_scale;
        for c in 0..tile_dim.1 {
            let v = tile_q8[r * tile_dim.1 + c] as f32 * s;
            let v = v.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16;
            out.push(v);
        }
    }
    Ok(out)
}

// Gate checks (T_MODEL_HEADER_PARSE, T_DMA_TILE_LOAD) live inline in the
// sp_model_layer_smoke bin, since they're on-device gates run against the
// .sp-model file pushed to /data/local/tmp/.
