// Phase 6-NET QUIC transport — wire types. TLS, endpoints, and loop added in Tasks 4-7.

use std::net::SocketAddr;
use std::sync::Arc;
use quinn::{Connection, Endpoint, RecvStream};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
use dashmap::DashMap;
use tokio::sync::mpsc;

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

// ── TLS ───────────────────────────────────────────────────────────────────────

/// Dev-mode TLS verifier: accepts any server certificate.
/// INTEGRATION POINT: Replace with Phase 5 ed25519 dominance identity in a later phase.
/// When Phase 5 identity is implemented, swap this struct for one that verifies
/// the peer's ed25519 public key against the known lattice node registry.
#[derive(Debug)]
struct SkipServerVerification;

impl rustls::client::danger::ServerCertVerifier for SkipServerVerification {
    fn verify_server_cert(
        &self,
        _end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &rustls::pki_types::ServerName<'_>,
        _ocsp_response: &[u8],
        _now: rustls::pki_types::UnixTime,
    ) -> std::result::Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> std::result::Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> std::result::Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        vec![
            rustls::SignatureScheme::RSA_PKCS1_SHA256,
            rustls::SignatureScheme::RSA_PKCS1_SHA384,
            rustls::SignatureScheme::RSA_PKCS1_SHA512,
            rustls::SignatureScheme::ECDSA_NISTP256_SHA256,
            rustls::SignatureScheme::ECDSA_NISTP384_SHA384,
            rustls::SignatureScheme::ECDSA_NISTP521_SHA512,
            rustls::SignatureScheme::RSA_PSS_SHA256,
            rustls::SignatureScheme::RSA_PSS_SHA384,
            rustls::SignatureScheme::RSA_PSS_SHA512,
            rustls::SignatureScheme::ED25519,
        ]
    }
}

pub fn make_server_config() -> Result<quinn::ServerConfig> {
    let ck = rcgen::generate_simple_self_signed(vec!["localhost".into()])?;
    let cert_der: CertificateDer<'static> = ck.cert.der().clone();
    let key_der = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(ck.key_pair.serialize_der()));

    let tls = rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(vec![cert_der], key_der)?;

    let quic_server = quinn::crypto::rustls::QuicServerConfig::try_from(tls)?;
    Ok(quinn::ServerConfig::with_crypto(Arc::new(quic_server)))
}

pub fn make_client_config() -> Result<quinn::ClientConfig> {
    let tls = rustls::ClientConfig::builder()
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(SkipServerVerification))
        .with_no_client_auth();

    let quic_client = quinn::crypto::rustls::QuicClientConfig::try_from(tls)?;
    Ok(quinn::ClientConfig::new(Arc::new(quic_client)))
}

// ── Coordinator ───────────────────────────────────────────────────────────────

pub struct SpQuicCoordinator {
    endpoint: Endpoint,
}

impl SpQuicCoordinator {
    /// Bind a QUIC server endpoint on `addr`.
    pub async fn bind(addr: SocketAddr) -> Result<Self> {
        let server_config = make_server_config()?;
        let endpoint = Endpoint::server(server_config, addr)?;
        Ok(Self { endpoint })
    }

    /// Accept the next incoming QUIC connection (blocks until one arrives).
    pub async fn accept_connection(&self) -> Result<Connection> {
        let incoming = self.endpoint.accept().await
            .ok_or("coordinator endpoint closed")?;
        let conn = incoming.await?;
        Ok(conn)
    }

    pub fn local_addr(&self) -> std::io::Result<SocketAddr> {
        self.endpoint.local_addr()
    }
}

// ── Worker ────────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct SpQuicWorker {
    connection: Connection,
}

impl SpQuicWorker {
    /// Dial the coordinator at `server_addr`, binding the client endpoint on
    /// `local_addr` (use port 0 for OS-assigned port).
    pub async fn connect(local_addr: SocketAddr, server_addr: SocketAddr) -> Result<Self> {
        let client_config = make_client_config()?;
        let mut endpoint = Endpoint::client(local_addr)?;
        endpoint.set_default_client_config(client_config);
        let conn = endpoint.connect(server_addr, "localhost")?.await?;
        Ok(Self { connection: conn })
    }

    /// Open a fresh unidirectional stream and write `block`.
    /// Each call to send_block opens its own stream ID — independent delivery,
    /// no HoL coupling between blocks.
    pub async fn send_block(&self, block: &ResidueBlock) -> Result<()> {
        let mut send = self.connection.open_uni().await?;

        send.write_all(&header_to_bytes(&block.header)).await?;
        for r in &block.residues {
            send.write_all(&r.to_le_bytes()).await?;
        }
        send.finish()?;
        Ok(())
    }
}

// ── Receive ───────────────────────────────────────────────────────────────────

/// Read all bytes from a unidirectional stream and decode a ResidueBlock.
/// First 64 bytes = ShardBlockHeader; remaining bytes / 4 = residues.
/// Max payload: 64 + 512 * 4 = 2112 bytes (N ≤ 512).
pub async fn recv_block(mut stream: RecvStream) -> Result<ResidueBlock> {
    let bytes = stream.read_to_end(64 + 512 * 4).await?;

    if bytes.len() < 64 {
        return Err(format!("stream too short: {} bytes (need ≥ 64)", bytes.len()).into());
    }
    if (bytes.len() - 64) % 4 != 0 {
        return Err("residue payload length not a multiple of 4".into());
    }

    let header = header_from_bytes(bytes[0..64].try_into().unwrap());
    let residues = bytes[64..]
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    Ok(ResidueBlock { header, residues })
}

// ── Garner assembly loop ──────────────────────────────────────────────────────

#[derive(Default)]
struct PendingBlock {
    q1:        Option<Vec<u32>>,
    q2:        Option<Vec<u32>>,
    token_pos: u32,
    layer_id:  u32,
}

pub struct GarnerResult {
    pub seq_id:    u64,
    pub token_pos: u32,
    pub layer_id:  u32,
    pub coeffs:    Vec<i64>,
}

/// Accept QUIC connections and streams from shard workers. When both q1 and q2
/// residues arrive for the same seq_id, call ntt_crt_recombine and send the
/// result on `results_tx`. Runs until the coordinator endpoint is dropped.
pub async fn run_garner_loop(
    coordinator: SpQuicCoordinator,
    ntt_n: u32,
    results_tx: mpsc::Sender<GarnerResult>,
) {
    let pending: Arc<DashMap<u64, PendingBlock>> = Arc::new(DashMap::new());

    loop {
        let conn = match coordinator.accept_connection().await {
            Ok(c) => c,
            Err(_) => break,
        };

        let pending = Arc::clone(&pending);
        let results_tx = results_tx.clone();

        tokio::spawn(async move {
            loop {
                let stream = match conn.accept_uni().await {
                    Ok(s) => s,
                    Err(_) => break,
                };

                let pending = Arc::clone(&pending);
                let results_tx = results_tx.clone();

                tokio::spawn(async move {
                    let block = match recv_block(stream).await {
                        Ok(b) => b,
                        Err(_) => return,
                    };

                    let seq_id = block.header.seq_id;
                    let prime_sel = block.header.prime_selector;

                    // Insert residue; check atomically if both primes arrived.
                    // DashMap entry() holds a shard lock for the duration of the block.
                    let both_arrived = {
                        let mut entry = pending.entry(seq_id).or_insert_with(|| PendingBlock {
                            q1: None,
                            q2: None,
                            token_pos: block.header.token_pos,
                            layer_id:  block.header.layer_id,
                        });
                        if prime_sel == 0 {
                            entry.q1 = Some(block.residues);
                        } else {
                            entry.q2 = Some(block.residues);
                        }
                        entry.q1.is_some() && entry.q2.is_some()
                    }; // shard lock released here

                    if both_arrived {
                        // Atomic remove — exactly one task wins if two race here.
                        if let Some((_, pb)) = pending.remove(&seq_id) {
                            if let (Some(q1), Some(q2)) = (pb.q1, pb.q2) {
                                let mut coeffs = vec![0i64; ntt_n as usize];
                                unsafe {
                                    use crate::ntt_ffi::{ntt_crt_recombine, ntt_free, ntt_init};
                                    let ctx = ntt_init(ntt_n);
                                    if !ctx.is_null() {
                                        ntt_crt_recombine(
                                            ctx,
                                            q1.as_ptr(),
                                            q2.as_ptr(),
                                            coeffs.as_mut_ptr(),
                                        );
                                        ntt_free(ctx);
                                    }
                                }
                                let _ = results_tx.send(GarnerResult {
                                    seq_id,
                                    token_pos: pb.token_pos,
                                    layer_id:  pb.layer_id,
                                    coeffs,
                                }).await;
                            }
                        }
                    }
                });
            }
        });
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

    #[test]
    fn tls_configs_construct_without_panic() {
        make_server_config().expect("server config");
        make_client_config().expect("client config");
    }

    // coordinator_binds_on_loopback is inline here because the integration test
    // path (tests/test_quic_shard.rs) is blocked by a pre-existing probe.rs linker
    // issue (same reason tls_configs_construct_without_panic is inline — see Task 4).
    #[tokio::test]
    async fn coordinator_binds_on_loopback() {
        let coord = SpQuicCoordinator::bind("127.0.0.1:0".parse().unwrap())
            .await
            .expect("coordinator bind failed");

        let addr = coord.local_addr().expect("local_addr");
        assert_eq!(addr.ip().to_string(), "127.0.0.1");
        assert_ne!(addr.port(), 0); // OS assigned a real port
    }

    // worker_connect_and_roundtrip is inline here (same probe linker issue
    // blocks tests/test_quic_shard.rs for async QUIC tests — see Task 4/5).
    #[tokio::test]
    async fn worker_connect_and_roundtrip() {
        use std::time::Duration;

        let coord = SpQuicCoordinator::bind("127.0.0.1:0".parse().unwrap())
            .await
            .expect("bind");
        let coord_addr = coord.local_addr().unwrap();

        let accept = tokio::spawn(async move {
            let conn = coord.accept_connection().await.expect("accept");
            let stream = conn.accept_uni().await.expect("uni");
            recv_block(stream).await.expect("recv_block")
        });

        let worker = SpQuicWorker::connect(
            "127.0.0.1:0".parse().unwrap(),
            coord_addr,
        )
        .await
        .expect("connect");

        let block = ResidueBlock {
            header: ShardBlockHeader {
                seq_id: 99,
                token_pos: 7,
                layer_id: 2,
                prime_selector: 0,
                _pad: [0u8; 47],
            },
            residues: (0u32..128).collect(),
        };
        worker.send_block(&block).await.expect("send_block");

        let received = tokio::time::timeout(Duration::from_secs(5), accept)
            .await
            .expect("timeout")
            .expect("task panic");

        assert_eq!(received.header.seq_id, 99);
        assert_eq!(received.header.prime_selector, 0);
        assert_eq!(received.residues.len(), 128);
        assert_eq!(received.residues[0], 0);
        assert_eq!(received.residues[127], 127);
    }

    // garner_loop_reconstructs_single_pair is inline here (same probe linker
    // issue blocks integration tests for async QUIC endpoints — see Task 4/5).
    #[tokio::test]
    async fn garner_loop_reconstructs_single_pair() {
        use crate::ntt_ffi::{ntt_crt_recombine, ntt_free, ntt_init};
        use std::time::Duration;

        const Q1: u32 = 1073738753;
        const Q2: u32 = 1073732609;
        const N: usize = 128;

        let q1: Vec<u32> = (0..N as u32).map(|i| i % Q1).collect();
        let q2: Vec<u32> = (0..N as u32).map(|i| i % Q2).collect();

        // Scalar reference (direct FFI, no network)
        let expected: Vec<i64> = unsafe {
            let ctx = ntt_init(N as u32);
            let mut out = vec![0i64; N];
            ntt_crt_recombine(ctx, q1.as_ptr(), q2.as_ptr(), out.as_mut_ptr());
            ntt_free(ctx);
            out
        };

        let coord = SpQuicCoordinator::bind("127.0.0.1:0".parse().unwrap())
            .await
            .expect("bind");
        let coord_addr = coord.local_addr().unwrap();

        let (tx, mut rx) = mpsc::channel(4);
        tokio::spawn(run_garner_loop(coord, N as u32, tx));
        tokio::time::sleep(Duration::from_millis(20)).await;

        let wa = SpQuicWorker::connect("127.0.0.1:0".parse().unwrap(), coord_addr)
            .await.expect("wa connect");
        wa.send_block(&ResidueBlock {
            header: ShardBlockHeader { seq_id: 7, token_pos: 0, layer_id: 0,
                                       prime_selector: 0, _pad: [0; 47] },
            residues: q1,
        }).await.expect("send q1");

        let wb = SpQuicWorker::connect("127.0.0.1:0".parse().unwrap(), coord_addr)
            .await.expect("wb connect");
        wb.send_block(&ResidueBlock {
            header: ShardBlockHeader { seq_id: 7, token_pos: 0, layer_id: 0,
                                       prime_selector: 1, _pad: [0; 47] },
            residues: q2,
        }).await.expect("send q2");

        // Keep wa, wb alive until result received — prevents FIN loss
        let result = tokio::time::timeout(Duration::from_secs(5), rx.recv())
            .await
            .expect("garner loop timeout")
            .expect("channel closed");

        drop(wa);
        drop(wb);

        assert_eq!(result.seq_id, 7);
        assert_eq!(result.coeffs, expected, "Garner reconstruction not bit-identical");
    }
}
