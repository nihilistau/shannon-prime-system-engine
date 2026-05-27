// Phase 6-NET QUIC transport — wire types. TLS, endpoints, and loop added in Tasks 4-7.

// ── Error type ────────────────────────────────────────────────────────────────

pub type ShardError = Box<dyn std::error::Error + Send + Sync + 'static>;
pub type Result<T> = std::result::Result<T, ShardError>;

// ── Wire types ────────────────────────────────────────────────────────────────

/// 64-byte stream header preceding each NTT residue payload.
/// All multi-byte fields are little-endian.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShardBlockHeader {
    pub seq_id:         u64,      //  0..8   global sequence counter
    pub token_pos:      u32,      //  8..12  token position in context
    pub layer_id:       u32,      // 12..16  transformer layer index
    pub prime_selector: u8,       // 16      0 = q1 (1073738753), 1 = q2 (1073732609)
    pub _pad:           [u8; 47], // 17..64  reserved zeros
}
const _: () = assert!(std::mem::size_of::<ShardBlockHeader>() == 64);

/// Residue block transmitted over one QUIC unidirectional stream.
pub struct ResidueBlock {
    pub header:   ShardBlockHeader,
    pub residues: Vec<u32>,
}

// ── Serialization (no serde, no protobuf) ─────────────────────────────────────

pub fn header_to_bytes(h: &ShardBlockHeader) -> [u8; 64] {
    let mut buf = [0u8; 64];
    buf[0..8].copy_from_slice(&h.seq_id.to_le_bytes());
    buf[8..12].copy_from_slice(&h.token_pos.to_le_bytes());
    buf[12..16].copy_from_slice(&h.layer_id.to_le_bytes());
    buf[16] = h.prime_selector;
    buf
}

pub fn header_from_bytes(b: &[u8; 64]) -> ShardBlockHeader {
    ShardBlockHeader {
        seq_id:         u64::from_le_bytes(b[0..8].try_into().unwrap()),
        token_pos:      u32::from_le_bytes(b[8..12].try_into().unwrap()),
        layer_id:       u32::from_le_bytes(b[12..16].try_into().unwrap()),
        prime_selector: b[16],
        _pad:           [0u8; 47],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_is_64_bytes() {
        assert_eq!(std::mem::size_of::<ShardBlockHeader>(), 64);
    }

    #[test]
    fn header_roundtrip() {
        let h = ShardBlockHeader {
            seq_id:         0xDEAD_BEEF_CAFE_1234,
            token_pos:      77,
            layer_id:       3,
            prime_selector: 1,
            _pad:           [0u8; 47],
        };
        let bytes = header_to_bytes(&h);
        let h2 = header_from_bytes(&bytes);
        assert_eq!(h2.seq_id,         h.seq_id);
        assert_eq!(h2.token_pos,      h.token_pos);
        assert_eq!(h2.layer_id,       h.layer_id);
        assert_eq!(h2.prime_selector, h.prime_selector);
    }
}
