use std::sync::{atomic::{AtomicI32, AtomicU64}, Arc, Mutex};
use std::time::Instant;
use tokio::sync::broadcast;

use crate::session::{SpModel, SpSession};
use crate::sessions::Sessions;
use crate::tokenizer::SptbTokenizer;

#[derive(Clone, Debug)]
pub struct ChatEvent {
    pub chat_id: u64,
    pub status: &'static str,
}

/// Shared daemon state — threaded through axum via `State<Arc<AppState>>`.
///
/// Drop order (declaration order) keeps models alive past every session:
/// sessions drop before models, so mmap-backed pointers in sp_session remain valid.
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
    pub events_tx: broadcast::Sender<ChatEvent>,
    /// Tokenizer built from the .sp-tokenizer SPTB blob at startup.
    pub tokenizer: Arc<SptbTokenizer>,
}
