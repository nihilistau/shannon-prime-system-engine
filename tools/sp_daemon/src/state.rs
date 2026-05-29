use std::net::SocketAddr;
use std::sync::{atomic::{AtomicI32, AtomicU64, AtomicBool}, Arc, Mutex};
use std::time::Instant;
use dashmap::DashMap;
use tokio::sync::broadcast;

// ConnectedPeer lives in the lib (quic_shard) so run_garner_loop can write it
// without creating a lib→binary dependency.
pub use sp_daemon::network::quic_shard::ConnectedPeer;

use ed25519_dalek::SigningKey;
use serde::Serialize;

#[cfg(not(target_os = "android"))]
use crate::session::{SpModel, SpSession};
#[cfg(not(target_os = "android"))]
use crate::sessions::Sessions;
#[cfg(not(target_os = "android"))]
use crate::tokenizer::SptbTokenizer;

/// Events broadcast on the /v1/events SSE channel.
#[derive(Clone, Debug)]
pub enum DaemonEvent {
    /// A chat request completed or was cancelled.
    Chat { chat_id: u64, status: &'static str },
    /// A sieve-fold receipt was minted.
    Mint { receipt_hex: String, sig_hex: String },
}

/// A minted PoUW receipt stored in AppState::receipt_store.
#[derive(Clone, Serialize)]
pub struct ReceiptRecord {
    /// Hex-encoded 152-byte receipt wire format (frozen v1).
    pub payload_hex: String,
    /// Hex-encoded 64-byte ed25519 signature over payload_hex bytes.
    pub sig_hex:     String,
    /// Global sieve-fold counter at mint time.
    pub round:       u64,
}

/// Shared daemon state — threaded through axum via `State<Arc<AppState>>`.
///
/// Drop order (declaration order) keeps models alive past every session:
/// sessions drop before models, so mmap-backed pointers in sp_session remain valid.
#[cfg(not(target_os = "android"))]
pub struct AppState {
    /// Target/verifier model. Kept alive so session mmap-backed pointers remain valid.
    #[allow(dead_code)]
    pub model: SpModel,
    /// Base target session at position 0. Held only briefly during sp_session_clone.
    pub session: Mutex<SpSession>,
    #[allow(dead_code)]
    pub cancel_flag: Arc<AtomicI32>,
    /// Draft model for speculative decoding (Phase 4-SPEC). None in single-model mode.
    #[allow(dead_code)]
    pub draft_model: Option<SpModel>,
    /// Base draft session at position 0. Cloned per spec-decode request.
    pub draft_session: Option<Mutex<SpSession>>,
    /// Active chat registry — maps chat_id → per-chat cancel_flag.
    pub sessions: Arc<Sessions>,
    /// Logits buffer width for this model; set once at startup from sp_arch_info.
    pub vocab_size: usize,
    /// Lifetime token counter for rolling tps in /v1/metrics.
    pub tokens_decoded: AtomicU64,
    /// Daemon start time (for tps denominator).
    pub started_at: Instant,
    /// Broadcast channel for daemon-wide events (/v1/events subscribers).
    pub events_tx: broadcast::Sender<DaemonEvent>,
    /// Tokenizer built from the .sp-tokenizer SPTB blob at startup.
    pub tokenizer: Arc<SptbTokenizer>,
    /// True while a /v1/chat request is actively being processed.
    /// The mining loop backs off when this is set.
    pub inference_active: Arc<AtomicBool>,
    /// Accumulated PoUW receipts minted by the background mining loop.
    pub receipt_store: Arc<Mutex<Vec<ReceiptRecord>>>,
    /// ed25519 node keypair used to sign PoUW receipts.
    pub node_signing_key: SigningKey,
    /// Active QUIC peers indexed by remote SocketAddr.
    /// Populated by run_garner_loop on accept; cleared on connection close.
    /// Empty until the coordinator is wired into daemon startup.
    pub peer_map: Arc<DashMap<SocketAddr, ConnectedPeer>>,
}

/// §3-HX Sprint J.5 — `Send + Sync` wrapper for the DSP-resident model.
///
/// `DspModel<'static>` holds `Vec<DmaBuffer>`, and `DmaBuffer` wraps a raw
/// rpcmem `*mut u8` → `!Send + !Sync`. axum's `State<Arc<AppState>>` requires
/// `Send + Sync`. The unsafe impls are sound for J.5 because the model is
/// **load-and-read-only**: after `DspModel::load` returns on the startup
/// thread, only plain metadata (`header`, `total_dma_bytes`) is read via
/// `/v1/dsp/model_info`; the rpcmem pointers are never dereferenced or invoked
/// across threads. Sprint K, which drives FastRPC invokes against these
/// buffers, must add per-invoke serialization (the deferred `Mutex`).
#[cfg(target_os = "android")]
pub struct ModelHandle(pub crate::dsp_model::DspModel<'static>);
#[cfg(target_os = "android")]
unsafe impl Send for ModelHandle {}
#[cfg(target_os = "android")]
unsafe impl Sync for ModelHandle {}

/// `Send + Sync` wrapper for the DSP-resident KV cache. Same soundness
/// argument as [`ModelHandle`]: J.5 reads only `total_bytes()`.
#[cfg(target_os = "android")]
pub struct KvCacheHandle(pub crate::kv_cache::KvCache<'static>);
#[cfg(target_os = "android")]
unsafe impl Send for KvCacheHandle {}
#[cfg(target_os = "android")]
unsafe impl Sync for KvCacheHandle {}

/// §3-HX Sprint J.5 — android daemon state.
///
/// The android binary host-gates the entire L1 C-ABI inference path (model,
/// session, tokenizer, mining) out of the build, so this is a deliberately
/// minimal `AppState` carrying only the pure-Rust mesh surface plus the cDSP
/// FastRPC session + DSP-resident model. Field names overlap the host struct
/// (`started_at`, `events_tx`, `peer_map`) so handlers that read only those
/// fields compile against both variants without `cfg`.
///
/// `dsp_model` / `kv_cache` are wired in the appstate commit; in this
/// host-gating commit they are absent and `/v1/dsp/model_info` returns 501.
#[cfg(target_os = "android")]
pub struct AppState {
    /// Daemon start time (for uptime / metrics denominators).
    pub started_at: Instant,
    /// Broadcast channel for daemon-wide events (/v1/events subscribers).
    pub events_tx: broadcast::Sender<DaemonEvent>,
    /// Active QUIC peers indexed by remote SocketAddr.
    pub peer_map: Arc<DashMap<SocketAddr, ConnectedPeer>>,
    /// §3-HX Sprint C — FastRpcSession for the V69 cDSP echo skel.
    /// `None` if the skel could not be admitted (alloc failure / missing skel);
    /// `/v1/dsp/echo` then returns 501.  Per-request Mutex serializes the FFI
    /// invoke since FastRPC per-handle thread-safety is single-thread.
    pub dsp_session: Option<Mutex<crate::dsp_rpc::FastRpcSession>>,
    /// §3-HX Sprint J.5 — DSP-resident Qwen3 model, loaded at startup via the
    /// sp_dsp_smoke loader into per-tensor rpcmem DmaBuffers. Backed by a
    /// process-lifetime (leaked `&'static`) FastRpcSession separate from
    /// `dsp_session`. `None` if the load failed → `/v1/dsp/model_info` 501s.
    pub dsp_model: Option<Arc<ModelHandle>>,
    /// Per-layer K/V DmaBuffers allocated at ctx_max=4096. `Mutex` is for
    /// Sprint K's decode-time mutation; J.5 only reads `total_bytes()`.
    pub kv_cache: Option<Arc<Mutex<KvCacheHandle>>>,
}
