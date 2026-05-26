use std::sync::Arc;

use axum::{routing::{get, post}, Router};

use crate::routes::{v1_abort, v1_chat, v1_events, v1_metrics, v1_peers, v1_receipts};
use crate::state::AppState;

pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/metrics",    get(v1_metrics))
        .route("/v1/chat",       post(v1_chat))
        .route("/v1/abort/:id",  post(v1_abort))
        .route("/v1/receipts",   get(v1_receipts))
        .route("/v1/peers",      get(v1_peers))
        .route("/v1/events",     get(v1_events))
        .with_state(state)
}
