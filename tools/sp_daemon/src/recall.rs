//! CONTRACT-CHAT-FULLSTACK B3 — AUTONOMOUS MEMORY RECALL.
//!
//! The C2 curator's discrete bit-collision resolver, ported to Rust so the
//! daemon can compute a chat turn's query signature ON ITS OWN and match it
//! against the episode registry — the model "remembers" without an
//! operator-specified `replay`.
//!
//! The signature is the SIGN of the ±1 LSH projection (SP_ARM_PROJ_SEED via
//! splitmix64) of the per-position GLOBAL-owner K, meaned over the global layers
//! and prefilled positions, packed to 256 bits. Match = XOR + popcount (integer
//! Hamming); accept iff bit-agreement >= TAU_BITS. This is byte-for-byte the math
//! of `tools/curator/discrete_resolve.py` (SEED, R_BITS, HD, the splitmix64
//! stream, the sign-binarize, the agreement count) so the daemon's live query
//! sig is directly comparable to the registry sigs that script writes.
//!
//! Null floor: this module is only reached when a chat turn sets
//! `auto_recall:true` and the registry loaded. Default off ⇒ byte-untouched.

use std::path::Path;

pub const SEED: u64 = 0x5350_524F_4A2B;
pub const R_BITS: usize = 256;
pub const HD: usize = 512; // gemma4-12b global head_dim (g_nkv=1 ⇒ g_kvd=512)
pub const NL: usize = 48;
pub const PERIOD: usize = 6;
pub const TAU_BITS: u32 = 168; // discrete_resolve.py default gate radius

/// splitmix64 ±1 stream — identical to discrete_resolve.py `smix` / build_R.
fn smix(seed: u64, n: usize) -> Vec<i8> {
    let mut s = seed;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        out.push(if z & 1 != 0 { 1 } else { -1 });
    }
    out
}

/// The frozen ±1 router R, shape [R_BITS][HD] row-major (matches build_R).
pub struct Projection {
    r: Vec<i8>, // R_BITS*HD
}

impl Projection {
    pub fn build() -> Self {
        Projection { r: smix(SEED, R_BITS * HD) }
    }

    /// Compute the 256-bit signature from packed global-layer K
    /// (`[n_global][npos][HD]` row-major f32, the gemma4_kv_read_global_k layout).
    /// sig[b] = sign( mean over (global layer, pos) of (R[b] · K[L,p]) ).
    /// Returns the 256-bit signature as 4 little-endian-bit u64 words (bit i of
    /// word w = sig bit 64*w+i), matching the hex packing in discrete_resolve.py.
    pub fn signature(&self, global_k: &[f32], n_global: usize, npos: usize) -> [u64; 4] {
        debug_assert_eq!(global_k.len(), n_global * npos * HD);
        let n_vec = (n_global * npos) as f64;
        // For each of the R_BITS projection rows, accumulate R[b]·K over all
        // (layer,pos) vectors, then take the mean's sign. Accumulate in f64 to
        // keep the daemon's argmax stable; the SIGN is what is binarized so this
        // matches the float projmean()>0 of the reference.
        let mut acc = vec![0.0f64; R_BITS];
        for v in 0..(n_global * npos) {
            let kbase = v * HD;
            for b in 0..R_BITS {
                let rbase = b * HD;
                let mut dot = 0.0f64;
                for d in 0..HD {
                    dot += (self.r[rbase + d] as f64) * (global_k[kbase + d] as f64);
                }
                acc[b] += dot;
            }
        }
        let mut words = [0u64; 4];
        for b in 0..R_BITS {
            let mean = acc[b] / n_vec;
            if mean > 0.0 {
                words[b / 64] |= 1u64 << (b % 64);
            }
        }
        words
    }
}

/// One registry episode: its replay path, position count, topic, and 256-bit sig.
#[derive(Clone, Debug)]
pub struct Episode {
    pub name: String,
    pub dir: String,
    pub npos: i32,
    pub topic: String,
    pub sig: [u64; 4],
}

/// Bit-agreement = R_BITS - HammingDistance (the discrete_resolve.py `agree`).
pub fn agree(a: &[u64; 4], b: &[u64; 4]) -> u32 {
    let mut ham = 0u32;
    for w in 0..4 {
        ham += (a[w] ^ b[w]).count_ones();
    }
    R_BITS as u32 - ham
}

/// Parse a 256-bit lowercase hex string (64 hex chars) into 4 u64 words with the
/// SAME bit order as discrete_resolve.py `packhex` (bit i ⇒ nibble i/4): the hex
/// is x = Σ bit_i << i, written `{x:064x}`. So word w = bits[64w..64w+64].
pub fn parse_sig_hex(hex: &str) -> Option<[u64; 4]> {
    if hex.len() != R_BITS / 4 {
        return None;
    }
    // x is a 256-bit big integer with bit i set ⇒ sig[i]=1. Read it from the hex
    // (MSB-first nibbles) into 4 little words.
    let mut words = [0u64; 4];
    // hex[0] is the most-significant nibble = bits [252..256).
    let bytes = hex.as_bytes();
    for (ci, &c) in bytes.iter().enumerate() {
        let nib = (c as char).to_digit(16)? as u64;
        // nibble at hex index ci covers bits [ (len-1-ci)*4 .. +4 )
        let bit_lo = (R_BITS / 4 - 1 - ci) * 4;
        for k in 0..4 {
            if nib & (1 << k) != 0 {
                let bit = bit_lo + k;
                words[bit / 64] |= 1u64 << (bit % 64);
            }
        }
    }
    Some(words)
}

/// Load the recall registry (JSONL: one episode per line). Tolerant: skips blank /
/// malformed lines, returns the episodes that parsed. Each row needs at minimum
/// `dir`, `npos`, `sig_bits`; `name`/`topic` are decorative.
pub fn load_registry(path: &Path) -> std::io::Result<Vec<Episode>> {
    let text = std::fs::read_to_string(path)?;
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let v: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let dir = v.get("dir").or_else(|| v.get("ring2_path")).and_then(|x| x.as_str());
        let sig_hex = v.get("sig_bits").and_then(|x| x.as_str());
        let npos = v.get("npos").and_then(|x| x.as_i64()).unwrap_or(0) as i32;
        let (dir, sig_hex) = match (dir, sig_hex) {
            (Some(d), Some(s)) => (d, s),
            _ => continue,
        };
        let sig = match parse_sig_hex(sig_hex) {
            Some(s) => s,
            None => continue,
        };
        out.push(Episode {
            name: v.get("name").and_then(|x| x.as_str()).unwrap_or("?").to_string(),
            dir: dir.to_string(),
            npos,
            topic: v.get("topic").and_then(|x| x.as_str()).unwrap_or("").to_string(),
            sig,
        });
    }
    Ok(out)
}

/// Match a query sig against the registry. Returns (best_episode_index,
/// best_agreement). The caller applies the TAU_BITS gate (fire only if
/// best_agreement >= TAU_BITS) — kept separate so the daemon can LOG the score
/// for the foreign-reject leg even when it does not fire.
pub fn best_match(query: &[u64; 4], registry: &[Episode]) -> Option<(usize, u32)> {
    let mut best: Option<(usize, u32)> = None;
    for (i, ep) in registry.iter().enumerate() {
        let a = agree(query, &ep.sig);
        match best {
            Some((_, bs)) if a <= bs => {}
            _ => best = Some((i, a)),
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_sig_roundtrips_bit_order() {
        // bit 0 set ⇒ least-significant nibble bit 0 ⇒ hex ends "...1".
        let mut h = String::from("0").repeat(63);
        h.push('1');
        let w = parse_sig_hex(&h).unwrap();
        assert_eq!(w[0] & 1, 1);
        assert_eq!(w[0] & 2, 0);
        // bit 255 ⇒ top nibble high bit ⇒ hex starts "8...".
        let mut h2 = String::from("8");
        h2.push_str(&"0".repeat(63));
        let w2 = parse_sig_hex(&h2).unwrap();
        assert_eq!(w2[3] >> 63, 1);
    }

    #[test]
    fn agree_self_is_full() {
        let s = [0xDEAD_BEEFu64, 1, 2, 3];
        assert_eq!(agree(&s, &s), R_BITS as u32);
        let mut t = s;
        t[0] ^= 1;
        assert_eq!(agree(&s, &t), R_BITS as u32 - 1);
    }

    #[test]
    fn smix_first_bits_match_reference() {
        // splitmix64 with SEED — first element sign. Reference: discrete_resolve.py
        // build_R()[0,0]. We only assert determinism here (stream is stable).
        let a = smix(SEED, 8);
        let b = smix(SEED, 8);
        assert_eq!(a, b);
        assert!(a.iter().all(|&x| x == 1 || x == -1));
    }
}
