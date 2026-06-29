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

/// CPU probe for the hardened Route head (same layout as the Tool/Action heads: mu,sd,W1,b1,W2,b2).
/// Returns the argmax class id (route space: 0=LOCAL, 1=TELEPATHY).
fn route_probe(blob: &[f32], h: usize, a: usize, proj: usize, feat: &[f32]) -> i32 {
    let (mu, sd) = (&blob[0..h], &blob[h..2*h]);
    let w1 = &blob[2*h..2*h + proj*h]; let b1 = &blob[2*h + proj*h..2*h + proj*h + proj];
    let w2o = 2*h + proj*h + proj; let w2 = &blob[w2o..w2o + a*proj]; let b2 = &blob[w2o + a*proj..w2o + a*proj + a];
    let mut hid = vec![0f32; proj];
    for p in 0..proj { let mut s = b1[p]; let row = &w1[p*h..p*h + h]; for i in 0..h { s += row[i] * ((feat[i]-mu[i])/sd[i]); } hid[p] = if s>0.0 {s} else {0.0}; }
    let (mut best, mut bv) = (0i32, f32::NEG_INFINITY);
    for c in 0..a { let mut s = b2[c]; let row = &w2[c*proj..c*proj + proj]; for p in 0..proj { s += row[p]*hid[p]; } if s > bv { bv = s; best = c as i32; } }
    best
}

/// The routing PRIMITIVE. When `SP_ROUTE_HEAD` points at the hardened Route head, the decision is
/// HEAD-GOVERNED (LOCAL vs TELEPATHY) — the same near-miss-hardened standard as the Tool/Action heads
/// (TELE-7: isolated-OOD 1.000, false-fire 0.000). Otherwise default LOCAL (null floor). `SP_ROUTE_FORCE`
/// overrides for testing the seam.
pub fn decide_route(latent: &[f32]) -> RouteDecision {
    if let Ok(hp) = std::env::var("SP_ROUTE_HEAD") {
        if !hp.trim().is_empty() {
            if let Ok(b) = std::fs::read(&hp) {
                let blob: Vec<f32> = b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect();
                let (h, a) = (latent.len(), 2usize);          // route space = {LOCAL, TELEPATHY}
                if blob.len() > 2*h + a {
                    let proj = (blob.len() - 2*h - a) / (h + 1 + a);
                    if proj > 0 && blob.len() == 2*h + proj*h + proj + a*proj + a {
                        return if route_probe(&blob, h, a, proj, latent) == 1 { RouteDecision::Telepathy(0) } else { RouteDecision::Local };
                    }
                }
            }
        }
    }
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

    // 3) routing primitive — headless default = LOCAL (null floor)
    let route0 = decide_route(&x);
    eprintln!("[telepathy] route primitive (headless): decide_route -> {:?}  (default LOCAL = null floor)", route0);

    // 3b) the hardened Route head (TELE-7) GOVERNS decide_route in-engine: classify captured feats and
    //     match the Python labels (proves the routing decision meets the Tool/Action heads' standard).
    let mut route_ok = true; let mut route_n = 0usize;
    let fxp = std::env::var("SP_ROUTE_FIXTURE").unwrap_or_else(|_| "route_fixture.bin".into());
    if let Ok(fx) = fs::read(&fxp) {
        let ri = |o: usize| i32::from_le_bytes([fx[o], fx[o+1], fx[o+2], fx[o+3]]);
        let (h, n) = (ri(0) as usize, ri(12) as usize);
        std::env::set_var("SP_ROUTE_HEAD", std::env::var("SP_ROUTE_HEAD_PATH").unwrap_or_else(|_| "_route_head.bin".into()));
        let mut o = 16;
        for _ in 0..n {
            let lab = ri(o); o += 4;
            let feat: Vec<f32> = (0..h).map(|k| { let p = o + k*4; f32::from_le_bytes([fx[p], fx[p+1], fx[p+2], fx[p+3]]) }).collect();
            o += h*4;
            let d = decide_route(&feat);
            let got = matches!(d, RouteDecision::Telepathy(_)) as i32;   // 1=TELEPATHY, 0=LOCAL
            let ok = got == lab; route_ok &= ok; route_n += 1;
            eprintln!("[telepathy] route head: feat(label={}) -> {:?}  {}", lab, d, if ok {"OK"} else {"MISMATCH"});
        }
        std::env::remove_var("SP_ROUTE_HEAD");
    } else { route_ok = false; eprintln!("[telepathy] route fixture {fxp} not found — skipping route-head demo"); }

    eprintln!("[telepathy] live transport: SAME-FAMILY identity inject = RP-1 (gemma4_kv_inject_tokens). CROSS-FAMILY destination forward (Qwen) = PENDING (no Qwen engine forward).");

    let wire_ok = off.is_none() && rel < 1e-2;
    eprintln!("[telepathy] G-TELEPATHY-WIRE: {}  (fail-closed + in-engine transfer == Python)", if wire_ok {"GREEN"} else {"RED"});
    eprintln!("[telepathy] G-ROUTE-WIRE:    {}  ({} fixtures, head-governed decide_route == Python labels)", if route_ok && route_n>0 {"GREEN"} else {"RED"}, route_n);
    if !(wire_ok && route_ok && route_n > 0) { return Err("telepathy gates not all green".into()); }
    Ok(())
}
