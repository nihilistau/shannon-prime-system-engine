use std::convert::Infallible;
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::Arc;

use axum::extract::{Path, State};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::IntoResponse;
use axum::{http::StatusCode, Json};
use serde::{Deserialize, Serialize};
use tokio::task;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::Stream;

use crate::state::AppState;

// ── /v1/metrics ───────────────────────────────────────────────────────────────

#[derive(Serialize)]
pub(crate) struct Metrics {
    tokens_per_sec: f64,
    ram_svm_bytes: u64,
    peers: u32,
    phase: &'static str,
    session_pos: u64,
}

pub async fn v1_metrics(State(state): State<Arc<AppState>>) -> Json<Metrics> {
    let session_pos = {
        let guard = state.session.lock().unwrap();
        guard.position().unwrap_or(0) as u64
    };

    let elapsed = state.started_at.elapsed().as_secs_f64();
    let decoded = state.tokens_decoded.load(Ordering::Relaxed);
    let tps = if elapsed > 0.1 { decoded as f64 / elapsed } else { 0.0 };

    Json(Metrics {
        tokens_per_sec: tps,
        ram_svm_bytes: 0,
        peers: 0,
        phase: "lat-phase-2-l3-verbs-closed",
        session_pos,
    })
}

// ── /v1/chat ──────────────────────────────────────────────────────────────────

/// VERBS debug request shape. Accepts raw token IDs to sidestep the BPE encoder
/// (real text→tokens is Phase 2-L3.TOK scope; the toy fixture has vocab=48).
#[derive(Deserialize)]
pub struct ChatRequest {
    pub prompt_tokens: Vec<i32>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
}
fn default_max_tokens() -> u32 {
    32
}

#[derive(Serialize)]
struct ChatDelta {
    delta: String,
    chat_id: u64,
}

pub async fn v1_chat(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatRequest>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(64);

    // Clone base session — hold Mutex only during sp_session_clone (sub-ms).
    let cancel_child = Arc::new(AtomicI32::new(0));
    let child_result = {
        let guard = state.session.lock().unwrap();
        guard.clone_session(cancel_child.clone())
    };

    let mut child = match child_result {
        Ok(s) => s,
        Err(e) => {
            let _ = tx
                .send(Ok(Event::default().data(format!("{{\"error\":\"{e}\"}}")))).await;
            return Sse::new(ReceiverStream::new(rx)).keep_alive(KeepAlive::default());
        }
    };

    let chat_id = state.sessions.register(cancel_child.clone());
    let sessions = state.sessions.clone();
    let vocab_size = state.vocab_size;
    let app = state.clone();

    task::spawn_blocking(move || {
        let mut logits = vec![0.0f32; vocab_size];

        if !req.prompt_tokens.is_empty() {
            if let Err(e) = child.prefill_chunk(&req.prompt_tokens, &mut logits) {
                let _ = tx.blocking_send(Ok(Event::default().data(
                    format!("{{\"error\":\"{e}\"}}"),
                )));
                sessions.remove(chat_id);
                return;
            }
        }

        let mut next_token = argmax(&logits);

        for _ in 0..req.max_tokens {
            let payload = serde_json::to_string(&ChatDelta {
                delta: format!("<{next_token}>"),
                chat_id,
            })
            .unwrap_or_default();

            if tx.blocking_send(Ok(Event::default().data(payload))).is_err() {
                // Client disconnected — trip cancel so L1 unwinds at next boundary.
                cancel_child.store(1, Ordering::Relaxed);
                sessions.remove(chat_id);
                return;
            }

            app.tokens_decoded.fetch_add(1, Ordering::Relaxed);

            match child.decode_step(next_token, &mut logits) {
                Ok(()) => next_token = argmax(&logits),
                Err(_) => break, // SP_ECANCEL or context full
            }
        }

        let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));
        sessions.remove(chat_id);
    });

    Sse::new(ReceiverStream::new(rx)).keep_alive(KeepAlive::default())
}

fn argmax(logits: &[f32]) -> i32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as i32)
        .unwrap_or(0)
}

// ── /v1/abort/{id} ────────────────────────────────────────────────────────────

pub async fn v1_abort(
    State(state): State<Arc<AppState>>,
    Path(id): Path<u64>,
) -> impl IntoResponse {
    if state.sessions.abort(id) {
        StatusCode::NO_CONTENT
    } else {
        StatusCode::NOT_FOUND
    }
}

// ── /v1/receipts ──────────────────────────────────────────────────────────────

pub async fn v1_receipts() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "receipts": [], "cursor": null }))
}

// ── /v1/peers ─────────────────────────────────────────────────────────────────

pub async fn v1_peers() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "peers": [] }))
}

// ── /v1/events ────────────────────────────────────────────────────────────────

/// Long-lived SSE channel for daemon-wide events.
/// VERBS scope: keep-alive pings only; real chat_completed broadcast is L3.AUTH+ scope.
pub async fn v1_events() -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let stream = futures::stream::pending::<Result<Event, Infallible>>();
    Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(std::time::Duration::from_secs(15))
            .text("ping"),
    )
}
