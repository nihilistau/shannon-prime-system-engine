use std::convert::Infallible;
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::Arc;
use std::time::Duration;

use axum::extract::{Path, State};
use axum::http::{header, HeaderName, HeaderValue};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::{http::StatusCode, Json};
use serde::{Deserialize, Serialize};
use tokio::task;
use tokio_stream::wrappers::{BroadcastStream, ReceiverStream};
use tokio_stream::StreamExt as _;

use crate::state::{AppState, ChatEvent};

// ── SSE header helper ─────────────────────────────────────────────────────────

fn sse_response(sse: impl IntoResponse) -> Response {
    let mut r = sse.into_response();
    r.headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    r.headers_mut().insert(
        HeaderName::from_static("x-accel-buffering"),
        HeaderValue::from_static("no"),
    );
    r
}

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
        phase: "lat-phase-2-l3-sse-closed",
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
) -> impl IntoResponse {
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
            return sse_response(
                Sse::new(ReceiverStream::new(rx)).keep_alive(
                    KeepAlive::new().interval(Duration::from_secs(15)).text("keepalive"),
                ),
            );
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
                let _ = app.events_tx.send(ChatEvent { chat_id, status: "cancelled" });
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
                // Client disconnected.
                cancel_child.store(1, Ordering::Relaxed);
                let _ = app.events_tx.send(ChatEvent { chat_id, status: "cancelled" });
                sessions.remove(chat_id);
                return;
            }

            app.tokens_decoded.fetch_add(1, Ordering::Relaxed);

            match child.decode_step(next_token, &mut logits) {
                Ok(()) => next_token = argmax(&logits),
                Err(_) => break, // SP_ECANCEL or context full
            }
        }

        let is_cancelled = cancel_child.load(Ordering::Relaxed) != 0;
        if is_cancelled {
            let _ = tx.blocking_send(Ok(Event::default().event("cancelled").data("{}")));
            let _ = app.events_tx.send(ChatEvent { chat_id, status: "cancelled" });
        } else {
            let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));
            let _ = app.events_tx.send(ChatEvent { chat_id, status: "done" });
        }
        sessions.remove(chat_id);
    });

    sse_response(
        Sse::new(ReceiverStream::new(rx)).keep_alive(
            KeepAlive::new().interval(Duration::from_secs(15)).text("keepalive"),
        ),
    )
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
/// Emits `event: chat_completed` when a chat finishes or is cancelled.
/// Keepalive comment every 15 s keeps connections alive through proxies.
pub async fn v1_events(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let rx = state.events_tx.subscribe();
    let stream = BroadcastStream::new(rx).filter_map(|result| {
        let ev = result.ok()?;
        let payload = serde_json::json!({
            "chat_id": ev.chat_id,
            "status":  ev.status,
        });
        Some(Ok::<Event, Infallible>(
            Event::default()
                .event("chat_completed")
                .data(payload.to_string()),
        ))
    });

    sse_response(
        Sse::new(stream).keep_alive(
            KeepAlive::new()
                .interval(Duration::from_secs(15))
                .text("keepalive"),
        ),
    )
}
