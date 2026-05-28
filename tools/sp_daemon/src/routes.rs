use std::convert::Infallible;
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::Arc;
use std::time::Duration;

use axum::extract::{Path, State};
use axum::extract::ws::{Message as WsMessage, WebSocket, WebSocketUpgrade};
use axum::http::{header, HeaderName, HeaderValue};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::{http::StatusCode, Json};
use serde::{Deserialize, Serialize};
use tokio::task;
use tokio_stream::wrappers::{BroadcastStream, ReceiverStream};
use tokio_stream::StreamExt as _;

use std::sync::atomic::AtomicBool;
use crate::state::{AppState, DaemonEvent};
use crate::tokenizer::{Message, PushResult, TokenDecodeBuffer};

// ── SSE header helper ──────────────────────────────────────────────────────

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

// ── /v1/metrics ───────────────────────────────────────────────────────────

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
        phase: "lat-phase-2-l3-tok-closed",
        session_pos,
    })
}

// ── /v1/chat ──────────────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct ChatRequest {
    pub prompt: Option<String>,
    pub messages: Option<Vec<Message>>,
    pub prompt_tokens: Option<Vec<i32>>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default)]
    pub stop: Vec<String>,
}

fn default_max_tokens() -> u32 {
    256
}

#[derive(Serialize)]
struct ChatDelta {
    delta: String,
    chat_id: u64,
}

pub async fn v1_chat(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatRequest>,
) -> Response {
    // Exactly one input field required.
    let n_inputs = req.prompt.is_some() as u8
        + req.messages.is_some() as u8
        + req.prompt_tokens.is_some() as u8;
    if n_inputs == 0 {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "one of prompt / messages / prompt_tokens required"})),
        )
            .into_response();
    }
    if n_inputs > 1 {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "only one of prompt / messages / prompt_tokens may be set"})),
        )
            .into_response();
    }

    // Tokenize input.
    let tokenizer = state.tokenizer.clone();
    let tokens: Vec<i32> = if let Some(ids) = req.prompt_tokens {
        ids
    } else if let Some(prompt_text) = req.prompt {
        match tokenizer.encode(&prompt_text) {
            Ok(ids) => ids,
            Err(e) => {
                return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": e})))
                    .into_response();
            }
        }
    } else {
        let messages = req.messages.unwrap();
        let text = match tokenizer.apply_template(&messages) {
            Ok(t) => t,
            Err(te) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({
                        "error": "chat_template_unavailable",
                        "arch_id": te.arch_id,
                        "hint": "use prompt or prompt_tokens"
                    })),
                )
                    .into_response();
            }
        };
        match tokenizer.encode(&text) {
            Ok(ids) => ids,
            Err(e) => {
                return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": e})))
                    .into_response();
            }
        }
    };

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
                Sse::new(ReceiverStream::new(rx))
                    .keep_alive(KeepAlive::new().interval(Duration::from_secs(15)).text("keepalive")),
            );
        }
    };

    let chat_id = state.sessions.register(cancel_child.clone());
    let sessions = state.sessions.clone();
    let vocab_size = state.vocab_size;
    let app = state.clone();
    let max_tokens = req.max_tokens;
    let stop_strings = req.stop;

    // Signal the mining loop to back off for the duration of this request.
    state.inference_active.store(true, Ordering::Relaxed);

    // Guard that clears inference_active when the spawn_blocking closure exits
    // (including early returns and panics).
    struct InferenceGuard(Arc<AtomicBool>);
    impl Drop for InferenceGuard {
        fn drop(&mut self) { self.0.store(false, Ordering::Relaxed); }
    }
    let _guard = InferenceGuard(state.inference_active.clone());

    task::spawn_blocking(move || {
        let _g = _guard; // keep guard alive for the duration of the blocking closure
        let mut logits = vec![0.0f32; vocab_size];
        let mut dec_buf = TokenDecodeBuffer::new(stop_strings);

        if !tokens.is_empty() {
            if let Err(e) = child.prefill_chunk(&tokens, &mut logits) {
                let _ = tx.blocking_send(Ok(Event::default().data(
                    format!("{{\"error\":\"{e}\"}}"),
                )));
                let _ = app.events_tx.send(DaemonEvent::Chat { chat_id, status: "cancelled" });
                sessions.remove(chat_id);
                return;
            }
        }

        let mut next_token = argmax(&logits);

        'decode: for _ in 0..max_tokens {
            // EOS check before emitting.
            if !tokenizer.eos_ids.is_empty() && tokenizer.eos_ids.contains(&next_token) {
                break 'decode;
            }

            let token_bytes = tokenizer.decode_token(next_token);
            let stop_hit = match dec_buf.push(token_bytes) {
                PushResult::Emit(bytes) => {
                    if !bytes.is_empty() {
                        let text = String::from_utf8_lossy(&bytes).into_owned();
                        let payload = serde_json::to_string(&ChatDelta { delta: text, chat_id })
                            .unwrap_or_default();
                        if tx.blocking_send(Ok(Event::default().data(payload))).is_err() {
                            // Client disconnected.
                            cancel_child.store(1, Ordering::Relaxed);
                            let _ = app.events_tx.send(DaemonEvent::Chat { chat_id, status: "cancelled" });
                            sessions.remove(chat_id);
                            return;
                        }
                    }
                    false
                }
                PushResult::Stopped(bytes) => {
                    if !bytes.is_empty() {
                        let text = String::from_utf8_lossy(&bytes).into_owned();
                        let payload = serde_json::to_string(&ChatDelta { delta: text, chat_id })
                            .unwrap_or_default();
                        let _ = tx.blocking_send(Ok(Event::default().data(payload)));
                    }
                    true
                }
            };

            app.tokens_decoded.fetch_add(1, Ordering::Relaxed);

            if stop_hit {
                break 'decode;
            }

            match child.decode_step(next_token, &mut logits) {
                Ok(()) => next_token = argmax(&logits),
                Err(_) => break 'decode,
            }
        }

        // Flush any bytes held back for UTF-8 / stop-string boundary detection.
        let flushed = dec_buf.flush();
        if !flushed.is_empty() {
            let text = String::from_utf8_lossy(&flushed).into_owned();
            let payload = serde_json::to_string(&ChatDelta { delta: text, chat_id })
                .unwrap_or_default();
            let _ = tx.blocking_send(Ok(Event::default().data(payload)));
        }

        let is_cancelled = cancel_child.load(Ordering::Relaxed) != 0;
        if is_cancelled {
            let _ = tx.blocking_send(Ok(Event::default().event("cancelled").data("{}")));
            let _ = app.events_tx.send(DaemonEvent::Chat { chat_id, status: "cancelled" });
        } else {
            let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));
            let _ = app.events_tx.send(DaemonEvent::Chat { chat_id, status: "done" });
        }
        sessions.remove(chat_id);
    });

    sse_response(
        Sse::new(ReceiverStream::new(rx))
            .keep_alive(KeepAlive::new().interval(Duration::from_secs(15)).text("keepalive")),
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

// ── /v1/abort/{id} ────────────────────────────────────────────────────────

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

// ── /v1/receipts ──────────────────────────────────────────────────────────

pub async fn v1_receipts(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let store = state.receipt_store.lock().unwrap();
    let receipts: Vec<_> = store.iter().map(|r| serde_json::json!({
        "payload_hex": r.payload_hex,
        "sig_hex":     r.sig_hex,
        "round":       r.round,
    })).collect();
    drop(store);
    Json(serde_json::json!({ "receipts": receipts, "cursor": null }))
}

// ── /v1/events ────────────────────────────────────────────────────────────

/// Long-lived SSE channel for daemon-wide events.
/// Emits `event: chat_completed` for chat lifecycle and `event: mint` for
/// new PoUW receipts.
pub async fn v1_events(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let rx = state.events_tx.subscribe();
    let stream = BroadcastStream::new(rx).filter_map(|result| {
        let ev = result.ok()?;
        match ev {
            DaemonEvent::Chat { chat_id, status } => {
                let payload = serde_json::json!({ "chat_id": chat_id, "status": status });
                Some(Ok::<Event, Infallible>(
                    Event::default().event("chat_completed").data(payload.to_string()),
                ))
            }
            DaemonEvent::Mint { receipt_hex, sig_hex } => {
                let payload = serde_json::json!({ "receipt_hex": receipt_hex, "sig_hex": sig_hex });
                Some(Ok::<Event, Infallible>(
                    Event::default().event("mint").data(payload.to_string()),
                ))
            }
        }
    });

    sse_response(
        Sse::new(stream).keep_alive(
            KeepAlive::new()
                .interval(Duration::from_secs(15))
                .text("keepalive"),
        ),
    )
}

// ── /v1/node/telemetry (WS) — migrated from console.rs ───────────────────

#[derive(Serialize)]
struct NodeTelemetry {
    node_id: String,
    cpu_temp_c: f32,
    svm_mem_gb: f32,
    dht_peers_active: u32,
    dht_peers_total: u32,
    pouw_frontier: u64,
}

pub async fn v1_node_telemetry(
    State(state): State<Arc<AppState>>,
    ws: WebSocketUpgrade,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| telemetry_loop(socket, state))
}

async fn telemetry_loop(mut socket: WebSocket, state: Arc<AppState>) {
    loop {
        let peers_active = state.peer_map.len() as u32;
        let pouw_frontier = {
            let store = state.receipt_store.lock().unwrap();
            store.len() as u64
        };
        let payload = NodeTelemetry {
            node_id: "q3-beast-canyon".to_string(),
            cpu_temp_c: 58.5,
            svm_mem_gb: 2.4,
            dht_peers_active: peers_active,
            dht_peers_total: 32,
            pouw_frontier,
        };
        let json = serde_json::to_string(&payload).expect("NodeTelemetry serializable");
        if socket.send(WsMessage::Text(json)).await.is_err() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(1000)).await;
    }
}

// ── /v1/mesh/peers — migrated from console.rs ────────────────────────────

#[derive(Serialize)]
struct PeerInfo {
    node_id:    String,
    address:    String,
    shard_id:   String,
    latency_ms: u32,
}

pub async fn v1_mesh_peers(State(state): State<Arc<AppState>>) -> axum::Json<serde_json::Value> {
    let peers: Vec<PeerInfo> = state.peer_map.iter().map(|entry| {
        let addr     = *entry.key();
        let shard_id = entry.value().shard_id;
        PeerInfo {
            node_id:    addr.to_string(),
            address:    addr.to_string(),
            shard_id:   if shard_id == 0 { "q1".into() } else { "q2".into() },
            latency_ms: 45,
        }
    }).collect();
    axum::Json(serde_json::json!({
        "peers":  peers,
        "active": peers.len(),
        "total":  32,
    }))
}

// ── /v1/pouw/ledger — migrated from console.rs ───────────────────────────

pub async fn v1_pouw_ledger(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let rx = state.events_tx.subscribe();
    let stream = BroadcastStream::new(rx).filter_map(|result| {
        let ev = result.ok()?;
        match ev {
            DaemonEvent::Mint { receipt_hex, .. } => {
                let line = format_kste_receipt(&receipt_hex);
                Some(Ok::<Event, Infallible>(Event::default().data(line)))
            }
            _ => None,
        }
    });
    Sse::new(stream)
        .keep_alive(KeepAlive::new().interval(Duration::from_secs(15)).text("keepalive"))
}

fn format_kste_receipt(receipt_hex: &str) -> String {
    if receipt_hex.len() < 288 {
        return format!("[KSTE] <malformed receipt len={}>", receipt_hex.len());
    }
    let round = decode_le_u64_hex(&receipt_hex[272..288]);
    let nonce = &receipt_hex[16..24];
    let hash  = &receipt_hex[144..152];
    format!("[KSTE] Round: {round} | Nonce: 0x{nonce}... | Z_q Hash: 0x{hash}...")
}

fn decode_le_u64_hex(hex16: &str) -> u64 {
    let mut bytes = [0u8; 8];
    for (i, b) in bytes.iter_mut().enumerate() {
        *b = u8::from_str_radix(&hex16[i * 2..i * 2 + 2], 16).unwrap_or(0);
    }
    u64::from_le_bytes(bytes)
}

// ── /v1/chat/stream stub — migrated from console.rs ─────────────────────

pub async fn v1_chat_stream_stub() -> axum::Json<serde_json::Value> {
    axum::Json(serde_json::json!({"status": "stub", "stream": "sse-legacy"}))
}
