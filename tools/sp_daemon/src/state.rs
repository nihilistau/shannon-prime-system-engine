use std::sync::{atomic::{AtomicI32, AtomicU64}, Arc, Mutex};
use std::time::Instant;
use tokio::sync::broadcast;

use crate::session::{SpModel, SpSession};
use crate::sessions::Sessions;

#[derive(Clone, Debug)]
pub struct ChatEvent {
    pub chat_id: u64,
    pub status: &'static str,
}

/// Shared daemon state — threaded through axum via `State<Arc<AppState>>`.
///
/// Drop order (declaration order) keeps the model alive past every session:
/// sessions drop before model, so mmap-backed pointers in sp_session remain valid.
pub struct AppState {
    /// Kept alive so session mmap-backed pointers remain valid.
    #[allow(dead_code)]
    pub model: SpModel,
    /// Base session, kept at position 0. Held only briefly during sp_session_clone.
    pub session: Mutex<SpSession>,
    /// Cancel flag for the base session (not used for per-chat cancellation).
    #[allow(dead_code)]
    pub cancel_flag: Arc<AtomicI32>,
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
}
