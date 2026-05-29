use std::sync::Arc;

use axum::{routing::{get, post}, Router};
use tower_http::{cors::CorsLayer, services::ServeDir};

use crate::routes::{
    v1_abort, v1_chat, v1_chat_stream_stub, v1_dsp_echo, v1_events, v1_mesh_peers,
    v1_metrics, v1_node_telemetry, v1_pouw_ledger, v1_receipts,
};
use crate::state::AppState;

pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/metrics",        get(v1_metrics))
        .route("/v1/chat",           post(v1_chat))
        .route("/v1/chat/stream",    get(v1_chat_stream_stub))
        .route("/v1/abort/:id",      post(v1_abort))
        .route("/v1/receipts",       get(v1_receipts))
        .route("/v1/events",         get(v1_events))
        .route("/v1/node/telemetry", get(v1_node_telemetry))
        .route("/v1/mesh/peers",     get(v1_mesh_peers))
        .route("/v1/pouw/ledger",    get(v1_pouw_ledger))
        .route("/v1/dsp/echo",       post(v1_dsp_echo))
        .fallback_service(ServeDir::new("frontend_mockups"))
        .layer(CorsLayer::permissive())
        .with_state(state)
}
