use std::sync::Arc;

use axum::{extract::State, Json};
use serde::Serialize;

use crate::state::AppState;

/// Response shape for GET /v1/metrics.
/// `session_pos` is extra — it proves the FFI call (sp_session_position)
/// worked. Real metrics land in Phase 2-L3.VERBS.
#[derive(Serialize)]
pub struct MetricsResponse {
    pub tokens_per_sec: f32,
    pub ram_svm_bytes:  u64,
    pub peers:          u32,
    pub phase:          &'static str,
    pub session_pos:    u64,
}

pub async fn v1_metrics(State(state): State<Arc<AppState>>) -> Json<MetricsResponse> {
    // Lock the session briefly to read position — sp_session_position is
    // read-only (const sp_session *) but the L1 ABI marks session NOT Sync,
    // so we serialize access through the Mutex.
    let pos = {
        let session = state.session.lock().unwrap();
        session.position().unwrap_or(0) as u64
    };

    Json(MetricsResponse {
        tokens_per_sec: 0.0,      // placeholder — VERBS wires real throughput
        ram_svm_bytes:  0,         // placeholder — VERBS reads arena / SVM usage
        peers:          0,         // placeholder — VERBS reads DHT peer table
        phase:          "lat-phase-2-l1-closed",
        session_pos:    pos,       // live FFI read from sp_session_position
    })
}
