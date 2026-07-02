use std::sync::Arc;

use axum::{routing::{get, post}, Router};
use tower_http::{cors::CorsLayer, services::ServeDir};

use crate::routes::{
    v1_abort, v1_capture, v1_chat, v1_chat_stream_stub, v1_debug_backend_counts, v1_dialogue,
    v1_dsp_echo, v1_dsp_model_info, v1_events, v1_mesh_peers, v1_metrics, v1_node_telemetry,
    v1_pouw_ledger, v1_receipts,
};
use crate::state::AppState;

/// Locate the console static dir independently of the launch CWD:
/// 1. SP_CONSOLE_DIR env override;
/// 2. `frontend_mockups` beside the running exe's crate root (exe lives at
///    tools/sp_daemon/target-*/release/sp-daemon.exe → up 3 = tools/sp_daemon);
/// 3. the historical CWD-relative fallback ("frontend_mockups").
fn frontend_dir() -> std::path::PathBuf {
    if let Ok(d) = std::env::var("SP_CONSOLE_DIR") {
        if !d.trim().is_empty() { return std::path::PathBuf::from(d); }
    }
    if let Ok(exe) = std::env::current_exe() {
        let mut p = exe.as_path();
        for _ in 0..4 {
            match p.parent() { Some(par) => p = par, None => break }
            let cand = p.join("frontend_mockups");
            if cand.join("console.html").is_file() { return cand; }
        }
    }
    std::path::PathBuf::from("frontend_mockups")
}

/// Unified router (Phase 2-L3.FG): the L1-backed inference + PoUW + mesh surface
/// serves both host and android (the C ABI links on android now). The two
/// `/v1/dsp/*` handlers each have host (501) and android (real) cfg variants, so
/// the routes are wired unconditionally and resolve per target.
pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/metrics",        get(v1_metrics))
        .route("/v1/chat",           post(v1_chat))
        .route("/v1/chat/stream",    get(v1_chat_stream_stub))
        // Chat-integration: MeMo (Grounding → Entity ID → Synthesis) dialogue.
        .route("/v1/dialogue",       post(v1_dialogue))
        .route("/v1/abort/:id",      post(v1_abort))
        .route("/v1/capture",        post(v1_capture))
        .route("/v1/receipts",       get(v1_receipts))
        .route("/v1/events",         get(v1_events))
        .route("/v1/node/telemetry", get(v1_node_telemetry))
        .route("/v1/mesh/peers",     get(v1_mesh_peers))
        .route("/v1/pouw/ledger",    get(v1_pouw_ledger))
        .route("/v1/dsp/echo",       post(v1_dsp_echo))
        .route("/v1/dsp/model_info", get(v1_dsp_model_info))
        // Sprint WIRE-HEX: hex forward + NTT.5b/c dispatch counters; reads
        // process-static atomics, no L1 calls. Exposes wire_hex_active so
        // the smoke harness validates startup registration as well as
        // first-prefill dispatch.
        .route("/v1/debug/backend_counts", get(v1_debug_backend_counts))
        // Console static files. ServeDir with a RELATIVE path resolves against the
        // process CWD, which depends on where the launcher ran (G-12B-SERVE aftermath:
        // engine-root launches 404'd the console). Resolve robustly: prefer the dir
        // beside the crate (exe at target-*/release/ => ../../../frontend_mockups),
        // fall back to CWD-relative for the historical tools/sp_daemon launch pattern.
        .fallback_service(ServeDir::new(frontend_dir()))
        .layer(CorsLayer::permissive())
        .with_state(state)
}
