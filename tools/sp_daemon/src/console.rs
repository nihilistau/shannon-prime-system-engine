use std::convert::Infallible;
use std::sync::{
    atomic::{AtomicBool, AtomicI32, Ordering},
    Arc,
};
use std::time::Duration;

use axum::{
    extract::{
        ws::{Message as WsMessage, WebSocket, WebSocketUpgrade},
        Json, State,
    },
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse,
    },
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio::task;
use tokio::time::sleep;
use tokio_stream::wrappers::{BroadcastStream, ReceiverStream};
use tokio_stream::StreamExt as _;
use tower_http::{cors::CorsLayer, services::ServeDir};
use tracing::info;

use crate::spec::{argmax, spec_step};
use crate::state::{AppState, DaemonEvent};
use crate::tokenizer::{Message as TokMessage, PushResult, TokenDecodeBuffer};

// ── Telemetry WS ──────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct NodeTelemetry {
    node_id: String,
    cpu_temp_c: f32,
    svm_mem_gb: f32,
    dht_peers_active: u32,
    dht_peers_total: u32,
    pouw_frontier: u64,
}

async fn node_telemetry(
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
            cpu_temp_c: 58.5,   // mocked: C-FFI thermal API not yet exposed
            svm_mem_gb: 2.4,    // mocked: session memory API not yet exposed
            dht_peers_active: peers_active,
            dht_peers_total: 32, // static: dev topology ceiling
            pouw_frontier,
        };
        let json = serde_json::to_string(&payload).expect("NodeTelemetry is always serializable");
        if socket.send(WsMessage::Text(json)).await.is_err() {
            break;
        }
        sleep(Duration::from_millis(1000)).await;
    }
}

// ── Chat SSE (POST /v1/chat) ──────────────────────────────────────────────────

#[derive(Deserialize)]
struct ChatRequest {
    prompt: String,
}

// Number of draft tokens to speculate per spec_step call (Theorem T8.1, §K).
const SPEC_K: usize = 4;
const MAX_TOKENS: u32 = 512;

pub async fn chat_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatRequest>,
) -> Sse<ReceiverStream<Result<Event, Infallible>>> {
    let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(64);

    // Tokenize — apply chat template if available, fall back to raw encode.
    let tokenizer = state.tokenizer.clone();
    let tok_msg = TokMessage { role: "user".into(), content: req.prompt.clone() };
    let tokens = match tokenizer.apply_template(&[tok_msg]) {
        Ok(text) => tokenizer.encode(&text),
        Err(_)   => tokenizer.encode(&req.prompt),
    };
    let tokens = match tokens {
        Ok(ids) => ids,
        Err(e) => {
            let _ = tx.try_send(Ok(Event::default().data(format!("[tokenize error: {e}]"))));
            return mk_sse(rx);
        }
    };

    // Clone target session — hold Mutex only during sp_session_clone (sub-ms).
    let cancel_target = Arc::new(AtomicI32::new(0));
    let mut target_child = {
        let guard = state.session.lock().unwrap();
        match guard.clone_session(cancel_target.clone()) {
            Ok(s) => s,
            Err(e) => {
                let _ = tx.try_send(Ok(Event::default().data(format!("[session error: {e}]"))));
                return mk_sse(rx);
            }
        }
    };

    // Clone draft session if available — hold Mutex only during sp_session_clone.
    // Returns None gracefully when single-model mode or clone fails.
    let cancel_draft = Arc::new(AtomicI32::new(0));
    let draft_child_opt = match &state.draft_session {
        Some(draft_mutex) => {
            let guard = draft_mutex.lock().unwrap();
            guard.clone_session(cancel_draft.clone()).ok()
        }
        None => None,
    };

    // Signal the mining loop to back off for the duration of this request.
    state.inference_active.store(true, Ordering::Relaxed);

    struct InferenceGuard(Arc<AtomicBool>);
    impl Drop for InferenceGuard {
        fn drop(&mut self) { self.0.store(false, Ordering::Relaxed); }
    }
    let _guard = InferenceGuard(state.inference_active.clone());
    let vocab_size = state.vocab_size;

    task::spawn_blocking(move || {
        let _g = _guard;
        let mut target_logits = vec![0.0f32; vocab_size];
        let mut dec_buf = TokenDecodeBuffer::new(vec![]);

        // Prefill target with the prompt.
        if !tokens.is_empty() {
            if let Err(e) = target_child.prefill_chunk(&tokens, &mut target_logits) {
                let _ = tx.blocking_send(Ok(Event::default().data(format!("[prefill error: {e}]"))));
                return;
            }
        }

        // Attempt draft prefill if we have a draft session; on failure degrade silently.
        let mut draft_logits = vec![0.0f32; vocab_size];
        let draft_ready = match draft_child_opt {
            Some(mut dc) => {
                let ok = tokens.is_empty() || dc.prefill_chunk(&tokens, &mut draft_logits).is_ok();
                if ok { Some(dc) } else { None }
            }
            None => None,
        };

        if let Some(mut draft_child) = draft_ready {
            // ── Speculative decode (Theorem T8.1, dual-session) ──────────────
            let mut total: u32 = 0;

            'spec: loop {
                if total >= MAX_TOKENS { break; }

                let result = match spec_step(
                    &mut target_child, &mut draft_child,
                    &target_logits, &draft_logits,
                    SPEC_K, vocab_size,
                ) {
                    Ok(r) => r,
                    Err(_) => break,
                };

                // Emit accepted draft tokens.
                for &tok in &result.accepted {
                    if !tokenizer.eos_ids.is_empty() && tokenizer.eos_ids.contains(&tok) {
                        break 'spec;
                    }
                    if emit_token(tok, &tokenizer, &mut dec_buf, &tx, &cancel_target, &cancel_draft) {
                        return; // client disconnected
                    }
                    total += 1;
                    if total >= MAX_TOKENS { break 'spec; }
                }

                match result.next_draft_logits {
                    Some(dl) => {
                        // All K accepted — advance logits for next iteration.
                        target_logits = result.next_target_logits;
                        draft_logits = dl;
                    }
                    None => {
                        // Rejection: timeline collapse — emit [REWIND] marker.
                        if tx.blocking_send(Ok(Event::default().data("[REWIND]"))).is_err() {
                            cancel_target.store(1, Ordering::Relaxed);
                            cancel_draft.store(1, Ordering::Relaxed);
                            return;
                        }

                        // Target's corrected token at the divergence point.
                        let corrected = argmax(&result.next_target_logits);
                        if !tokenizer.eos_ids.is_empty() && tokenizer.eos_ids.contains(&corrected) {
                            break 'spec;
                        }
                        if emit_token(corrected, &tokenizer, &mut dec_buf, &tx, &cancel_target, &cancel_draft) {
                            return;
                        }
                        total += 1;
                        if total >= MAX_TOKENS { break 'spec; }

                        // Advance both sessions with the corrected token to re-sync KV caches.
                        if target_child.decode_step(corrected, &mut target_logits).is_err() { break 'spec; }
                        if draft_child.decode_step(corrected, &mut draft_logits).is_err() { break 'spec; }
                    }
                }
            }
        } else {
            // ── Autoregressive fallback (single-session, no draft) ────────────
            let mut next_token = argmax(&target_logits);

            'ar: for _ in 0..MAX_TOKENS {
                if !tokenizer.eos_ids.is_empty() && tokenizer.eos_ids.contains(&next_token) {
                    break 'ar;
                }
                if emit_token(next_token, &tokenizer, &mut dec_buf, &tx, &cancel_target, &cancel_draft) {
                    return;
                }
                match target_child.decode_step(next_token, &mut target_logits) {
                    Ok(()) => next_token = argmax(&target_logits),
                    Err(_) => break 'ar,
                }
            }
        }

        // Flush any bytes held back for UTF-8 / stop-string boundary detection.
        let flushed = dec_buf.flush();
        if !flushed.is_empty() {
            let text = String::from_utf8_lossy(&flushed).into_owned();
            let _ = tx.blocking_send(Ok(Event::default().data(text)));
        }
        // tx drops → ReceiverStream ends → SSE closes cleanly (no [DONE] sentinel).
    });

    mk_sse(rx)
}

/// Decode one token via the buffer and send it.
/// Returns `true` if the client disconnected (caller should `return`).
#[inline]
fn emit_token(
    token: i32,
    tokenizer: &crate::tokenizer::SptbTokenizer,
    dec_buf: &mut TokenDecodeBuffer,
    tx: &tokio::sync::mpsc::Sender<Result<Event, Infallible>>,
    cancel_target: &Arc<AtomicI32>,
    cancel_draft: &Arc<AtomicI32>,
) -> bool {
    let bytes = tokenizer.decode_token(token);
    match dec_buf.push(bytes) {
        PushResult::Emit(out) if !out.is_empty() => {
            let text = String::from_utf8_lossy(&out).into_owned();
            if tx.blocking_send(Ok(Event::default().data(text))).is_err() {
                cancel_target.store(1, Ordering::Relaxed);
                cancel_draft.store(1, Ordering::Relaxed);
                return true;
            }
        }
        PushResult::Stopped(out) => {
            if !out.is_empty() {
                let text = String::from_utf8_lossy(&out).into_owned();
                let _ = tx.blocking_send(Ok(Event::default().data(text)));
            }
            // stop-string hit; caller should break its loop but not return (flush still runs)
        }
        _ => {}
    }
    false
}

fn mk_sse(rx: mpsc::Receiver<Result<Event, Infallible>>) -> Sse<ReceiverStream<Result<Event, Infallible>>> {
    Sse::new(ReceiverStream::new(rx))
        .keep_alive(KeepAlive::new().interval(Duration::from_secs(15)).text("keepalive"))
}

// ── DHT Mesh peers (GET /v1/mesh/peers) ──────────────────────────────────────

#[derive(Serialize)]
struct PeerInfo {
    node_id:    String,
    address:    String,
    shard_id:   String,
    latency_ms: u32,
}

async fn mesh_peers(State(state): State<Arc<AppState>>) -> axum::Json<serde_json::Value> {
    // DashMap::iter() holds one shard lock per entry briefly — no long lock.
    let peers: Vec<PeerInfo> = state.peer_map.iter().map(|entry| {
        let addr     = *entry.key();
        let shard_id = entry.value().shard_id;
        PeerInfo {
            node_id:    addr.to_string(), // TODO: replace with TLS identity once wired
            address:    addr.to_string(),
            shard_id:   if shard_id == 0 { "q1".into() } else { "q2".into() },
            latency_ms: 45, // mocked: RTT tracking not yet implemented
        }
    }).collect();
    axum::Json(serde_json::json!({
        "peers": peers,
        "active": peers.len(),
        "total": 32,
    }))
}

// ── PoUW Receipts Ledger (GET /v1/pouw/ledger) ───────────────────────────────
//
// Subscribes to the existing events_tx broadcast channel (DaemonEvent::Mint is
// already emitted by run_mining_loop on every sieve-fold event) and re-formats
// each receipt as a human-readable KSTE display line.
//
// Receipt wire format (152 bytes, hex-encoded):
//   [  0.. 7] magic "SPRCPT01"
//   [  8..72] kste_sig  (64 B) — bytes 8..12 → "nonce" display
//   [ 72..104] seq_hash  (32 B) — bytes 72..76 → Z_q hash display
//   [104..136] pubkey    (32 B)
//   [136..144] round     (u64 LE)
//   [144..152] minted_at_ns (u64 LE)

async fn pouw_ledger(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let rx = state.events_tx.subscribe();
    let stream = BroadcastStream::new(rx).filter_map(|result| {
        let ev = result.ok()?; // drop Lagged errors silently
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

/// Decode a packed receipt hex string into a UI display line.
/// Returns a fallback string if the hex is malformed or too short.
fn format_kste_receipt(receipt_hex: &str) -> String {
    // Minimum length for all fields we read.
    if receipt_hex.len() < 288 {
        return format!("[KSTE] <malformed receipt len={}>", receipt_hex.len());
    }
    let round = decode_le_u64_hex(&receipt_hex[272..288]);
    // kste_sig first 4 bytes → "nonce" shorthand for the UI.
    let nonce = &receipt_hex[16..24];
    // seq_hash first 4 bytes → Z_q hash shorthand.
    let hash  = &receipt_hex[144..152];
    format!("[KSTE] Round: {round} | Nonce: 0x{nonce}... | Z_q Hash: 0x{hash}...")
}

/// Decode 16 hex chars (8 bytes) as a little-endian u64.
fn decode_le_u64_hex(hex16: &str) -> u64 {
    let mut bytes = [0u8; 8];
    for (i, b) in bytes.iter_mut().enumerate() {
        *b = u8::from_str_radix(&hex16[i * 2..i * 2 + 2], 16).unwrap_or(0);
    }
    u64::from_le_bytes(bytes)
}

// ── Stub ──────────────────────────────────────────────────────────────────────

async fn chat_stream_stub() -> axum::Json<serde_json::Value> {
    axum::Json(serde_json::json!({"status": "stub", "stream": "sse-legacy"}))
}

// ── Router / startup ──────────────────────────────────────────────────────────

fn build_console_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/chat", post(chat_handler))
        .route("/v1/chat/stream", get(chat_stream_stub))
        .route("/v1/node/telemetry", get(node_telemetry))
        .route("/v1/mesh/peers", get(mesh_peers))
        .route("/v1/pouw/ledger", get(pouw_ledger))
        .fallback_service(ServeDir::new("frontend_mockups"))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

pub async fn start_operator_console(state: Arc<AppState>, console_port: u16) {
    let app = build_console_router(state);
    let addr: std::net::SocketAddr = ([127, 0, 0, 1], console_port).into();
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .unwrap_or_else(|e| panic!("bind {addr}: {e}"));
    info!("operator console listening on {addr}");
    axum::serve(listener, app)
        .await
        .expect("operator console server error");
}
