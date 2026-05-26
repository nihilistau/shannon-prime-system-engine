use std::sync::{atomic::AtomicI32, Arc, Mutex};

use crate::session::{SpModel, SpSession};

/// Shared daemon state — threaded through axum via `State<Arc<AppState>>`.
///
/// `model` must outlive every session (sp_session holds raw pointers into
/// the mmap that sp_model manages). Rust drop order guarantees this because
/// struct fields drop in declaration order (session before model).
///
/// `cancel_flag` is the L2-owned atomic passed into sp_session_create.
/// Phase 2-L3.VERBS will flip it from the /v1/abort route.
pub struct AppState {
    /// Kept alive so the session's mmap-backed pointers remain valid.
    #[allow(dead_code)]
    pub model: SpModel,
    pub session: Mutex<SpSession>,
    /// Shared with sp_session_create; VERBS/ABORT will read/write this.
    #[allow(dead_code)]
    pub cancel_flag: Arc<AtomicI32>,
}
