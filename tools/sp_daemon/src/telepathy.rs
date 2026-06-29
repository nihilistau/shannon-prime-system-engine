//! Telepathy — the LatentBridge: tokenizer-free latent->latent transport between models.
//! Cemented per papers/PPT-LAT-TELEPATHY-LatentBridge-spec.md. `SP_TELEPATHY` is DEFAULT-OFF (null
//! floor): when unset the daemon never touches this path. This module is the first-class in-engine
//! LatentBridge OBJECT + adapter-load + affine transfer + the routing primitive + flags + the
//! fail-closed license gate.
//!
//! Scope (honest): the cross-family TRANSFER (affine map) runs here and is parity-checked vs the proven
//! Python adapter (TELE-1). The PROVEN live transport is the SAME-FAMILY identity inject (RP-1,
//! `gemma4_kv_inject_tokens` in eagle_accept). The cross-family DESTINATION FORWARD (e.g. Qwen) is
//! PENDING — the engine has no Qwen forward yet — so a fully-live Gemma->Qwen transmit is not claimed.
//! The license/attestation enforcement is the SPEC'd commercial boundary (proprietary on the MIT
//! substrate); here it is realized as a fail-closed gate that only disables the bridge's own operation.

use std::fs;

pub const F_DEFAULT_OFF: u32     = 1 << 0;
pub const F_REQUIRE_LICENSE: u32 = 1 << 1;
pub const F_REJECT_FOREIGN: u32  = 1 << 2;

/// The model-pair-specific adapter (z-scored ridge affine), loaded from the flat .bin exported by
/// tools/telepathy/export_adapter.py. Layout (LE): i32 din, i32 dout, f32 gmu[din] gsd[din] qmu[dout] qsd[dout] W[din*dout] (C-order).
#[derive(Clone)]
pub struct AdapterBin {
    pub din: usize, pub dout: usize,
    pub gmu: Vec<f32>, pub gsd: Vec<f32>, pub qmu: Vec<f32>, pub qsd: Vec<f32>, pub w: Vec<f32>,
}
impl AdapterBin {
    pub fn load(path: &str) -> Result<Self, String> {
        let b = fs::read(path).map_err(|e| format!("adapter read {path}: {e}"))?;
        if b.len() < 8 { return Err("adapter bin too small".into()); }
        let i = |o: usize| i32::from_le_bytes([b[o], b[o+1], b[o+2], b[o+3]]) as usize;
        let (din, dout) = (i(0), i(4));
        let need = 8 + 4 * (2*din + 2*dout + din*dout);
        if b.len() < need { return Err(format!("adapter bin short: have {} need {need}", b.len())); }
        let mut o = 8;
        let mut rd = |n: usize| { let mut v = Vec::with_capacity(n);
            for _ in 0..n { v.push(f32::from_le_bytes([b[o], b[o+1], b[o+2], b[o+3]])); o += 4; } v };
        let (gmu, gsd, qmu, qsd) = (rd(din), rd(din), rd(dout), rd(dout));
        let w = rd(din * dout);
        Ok(Self { din, dout, gmu, gsd, qmu, qsd, w })
    }
}

/// The routing PRIMITIVE — a new decision dimension, deliberately separate from the Tool/Action heads
/// (those decide local effects; this decides transport). `Telepathy(bridge_id)` selects a bridge.
#[derive(Debug, PartialEq)]
pub enum RouteDecision { Local, Telepathy(u32) }

/// Default = LOCAL (the null floor) until a hardened **Route head** exists (same probe machinery as the
/// Tool/Action heads; must pass near-miss hardening like TS-2/TS-3 before it may fire). `SP_ROUTE_FORCE`
/// forces a bridge id for testing the seam.
pub fn decide_route(_latent: &[f32]) -> RouteDecision {
    match std::env::var("SP_ROUTE_FORCE") {
        Ok(s) if !s.trim().is_empty() => RouteDecision::Telepathy(s.trim().parse().unwrap_or(0)),
        _ => RouteDecision::Local,
    }
}

pub struct LatentBridge {
    pub src: String, pub dst: String, pub flags: u32, pub scale: f32,
    pub license_ok: bool, pub adapter: AdapterBin,
}
impl LatentBridge {
    /// Fail-closed build: with `F_REQUIRE_LICENSE` and no `SP_TELEPATHY_LICENSE`, `license_ok=false`
    /// and `transfer` returns None (the bridge runs inert — "bricks itself"). Disables ONLY the
    /// bridge's own operation; never any host-external effect.
    pub fn build(src: &str, dst: &str, adapter: AdapterBin, flags: u32) -> Self {
        let license_ok = if flags & F_REQUIRE_LICENSE != 0 {
            std::env::var("SP_TELEPATHY_LICENSE").map(|t| !t.trim().is_empty()).unwrap_or(false)
        } else { true };
        let scale = std::env::var("SP_TELEPATHY_SCALE").ok().and_then(|s| s.parse().ok()).unwrap_or(1.0);
        Self { src: src.into(), dst: dst.into(), flags, scale, license_ok, adapter }
    }
    /// Affine transfer: z-score(in) -> W -> un-z-score(out). None if the license gate fails closed.
    pub fn transfer(&self, x: &[f32]) -> Option<Vec<f32>> {
        if !self.license_ok { return None; }
        let a = &self.adapter; let (din, dout) = (a.din, a.dout);
        if x.len() != din { return None; }
        let mut z = vec![0f32; din];
        for k in 0..din { z[k] = (x[k] - a.gmu[k]) / a.gsd[k]; }
        let mut out = vec![0f32; dout];
        for j in 0..dout {
            let mut s = 0f32;
            for k in 0..din { s += z[k] * a.w[k*dout + j]; }
            out[j] = s * self.scale * a.qsd[j] + a.qmu[j];
        }
        Some(out)
    }
}

/// `SP_TELEPATHY=1` verb: load the LatentBridge, demonstrate the fail-closed license, run the in-engine
/// transfer, parity-check it against the proven Python adapter, and exercise the routing primitive.
/// Pure-Rust (no FFI) — proves the bridge OBJECT + transfer + route + license are cemented in the daemon.
pub fn run_telepathy() -> Result<(), String> {
    let adp = std::env::var("SP_TELEPATHY_ADAPTER").unwrap_or_else(|_| "telepathy_adapter_g2q.bin".into());
    let srcf = std::env::var("SP_TELEPATHY_SRC").unwrap_or_else(|_| "tele_src_latent.bin".into());
    let expf = std::env::var("SP_TELEPATHY_EXPECT").unwrap_or_else(|_| "tele_expected_map.bin".into());
    let read_f32 = |p: &str| -> Result<Vec<f32>, String> {
        let b = fs::read(p).map_err(|e| format!("{p}: {e}"))?;
        Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    };
    let adapter = AdapterBin::load(&adp)?;
    eprintln!("[telepathy] LatentBridge adapter loaded: din={} dout={}  (gemma-3n-E2B -> qwen2.5-coder-0.5b)", adapter.din, adapter.dout);
    let x = read_f32(&srcf)?;
    if x.len() != adapter.din { return Err(format!("src dim {} != din {}", x.len(), adapter.din)); }

    // 1) fail-closed license: REQUIRE_LICENSE, no token -> inert
    let inert = LatentBridge::build("gemma-3n-E2B", "qwen2.5-coder-0.5b", adapter.clone(), F_DEFAULT_OFF | F_REQUIRE_LICENSE);
    let off = inert.transfer(&x);
    eprintln!("[telepathy] fail-closed: SP_TELEPATHY_LICENSE unset -> transfer = {}", if off.is_none() { "None (bridge inert / bricked — correct)" } else { "Some (WRONG)" });

    // 2) with a license token -> live transfer + parity vs Python adapter
    std::env::set_var("SP_TELEPATHY_LICENSE", "dev");
    let bridge = LatentBridge::build("gemma-3n-E2B", "qwen2.5-coder-0.5b", adapter, F_DEFAULT_OFF | F_REQUIRE_LICENSE);
    let y = bridge.transfer(&x).ok_or("transfer refused despite license set")?;
    let exp = read_f32(&expf)?;
    if exp.len() != y.len() { return Err(format!("expect dim {} != dout {}", exp.len(), y.len())); }
    let (mut maxd, mut sse, mut snorm) = (0f32, 0f64, 0f64);
    for j in 0..y.len() { let d = (y[j]-exp[j]).abs(); if d>maxd {maxd=d;} sse += (d as f64)*(d as f64); snorm += (exp[j] as f64)*(exp[j] as f64); }
    let rel = (sse.sqrt()) / (snorm.sqrt() + 1e-9);
    eprintln!("[telepathy] in-engine transfer vs Python adapter: max|Δ|={:.3e}  relL2={:.3e}  (|y|={:.3})", maxd, rel, snorm.sqrt());

    // 3) routing primitive
    let route = decide_route(&x);
    eprintln!("[telepathy] route primitive: decide_route -> {:?}  (default LOCAL = null floor; Route head pending+hardening)", route);
    eprintln!("[telepathy] live transport: SAME-FAMILY identity inject = RP-1 (gemma4_kv_inject_tokens). CROSS-FAMILY destination forward (Qwen) = PENDING (no Qwen engine forward).");

    let verdict = if off.is_none() && rel < 1e-2 { "GREEN" } else { "RED" };
    eprintln!("[telepathy] G-TELEPATHY-WIRE: {verdict}  (fail-closed correct + in-engine transfer == Python within float tol)");
    if verdict != "GREEN" { return Err("G-TELEPATHY-WIRE not green".into()); }
    Ok(())
}
