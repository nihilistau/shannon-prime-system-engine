use sp_daemon::network::quic_shard::{ShardBlockHeader, ResidueBlock};

#[test]
fn header_is_64_bytes() {
    assert_eq!(std::mem::size_of::<ShardBlockHeader>(), 64);
}

#[test]
fn header_roundtrip() {
    use sp_daemon::network::quic_shard::{header_to_bytes, header_from_bytes};

    let h = ShardBlockHeader {
        seq_id: 0xDEAD_BEEF_CAFE_1234,
        token_pos: 77,
        layer_id: 3,
        prime_selector: 1,
        _pad: [0u8; 47],
    };
    let bytes = header_to_bytes(&h);
    let h2 = header_from_bytes(&bytes);
    assert_eq!(h2.seq_id, h.seq_id);
    assert_eq!(h2.token_pos, h.token_pos);
    assert_eq!(h2.layer_id, h.layer_id);
    assert_eq!(h2.prime_selector, h.prime_selector);
}
