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

// ── /v1/debug/backend_counts (Sprint WIRE-HEX) ────────────────────────────
//
// Surfaces the dispatch counters for the optional accelerator backends
// (hex forward, hex NTT). T_WIRE_HEX_BACKEND_DISPATCHES reads
// `hex_forward_count` after one prefill of a known prompt; PASS criterion is
// > 0. wire_hex_active reports whether the registration succeeded at
// startup (independent of whether a prefill has run yet).

#[derive(Serialize)]
pub(crate) struct BackendCounts {
    /// Sprint WIRE-HEX: gemma3_forward_hexagon dispatcher hit count
    /// since process start. > 0 after one prefill when SP_DAEMON_BACKEND=hex
    /// is set AND the daemon was built with --features wire_hex_backend.
    /// Always 0 on host builds and on android without the feature.
    hex_forward_count: u64,
    /// Sprint WIRE-HEX: whether sp_session_register_forward_backend was
    /// invoked successfully on the target session at startup. False when
    /// SP_DAEMON_BACKEND is unset / != "hex", when the feature was off at
    /// build time, or when registration failed (see daemon log).
    wire_hex_active: bool,
    /// NTT.5b/c: hex NTT forward + inverse dispatch counters (Bluestein
    /// inner kernels via FastRPC). Always 0 when SP_ENGINE_NTT_ATTN_HEX
    /// is unset. Independent of WIRE-HEX (different ABI hook).
    ntt_hex_forward_count: u64,
    ntt_hex_inverse_count: u64,
    /// Sprint WIRE-CPU: engine CPU AVX-512 backend dispatcher hit count
    /// since process start. > 0 after one prefill when SP_DAEMON_BACKEND=cpu
    /// is set AND the daemon was built with --features wire_cpu_backend.
    /// Always 0 without the feature.
    cpu_forward_count: u64,
    /// Sprint WIRE-CPU: whether sp_session_register_forward_backend was
    /// invoked successfully on the target session at startup. False when
    /// SP_DAEMON_BACKEND is unset / != "cpu", when the feature was off at
    /// build time, or when registration failed.
    wire_cpu_active: bool,
    /// Sprint WIRE-CUDA: gemma3_forward_cuda / qwen3_forward_cuda dispatcher
    /// hit count since process start. > 0 after one prefill when
    /// SP_DAEMON_BACKEND=cuda is set AND the daemon was built with
    /// --features wire_cuda_backend. Always 0 on builds without the feature.
    cuda_forward_count: u64,
    /// Sprint WIRE-CUDA: whether sp_session_register_forward_backend was
    /// invoked successfully on the target session at startup. False when
    /// SP_DAEMON_BACKEND is unset / != "cuda", when the feature was off at
    /// build time, or when registration failed (see daemon log).
    wire_cuda_active: bool,
    /// Sprint WIRE-VULKAN: gemma3_forward_vulkan / qwen3_forward_vulkan
    /// dispatcher hit count since process start. > 0 after one prefill
    /// when SP_DAEMON_BACKEND=vulkan is set AND the daemon was built with
    /// --features wire_vulkan_backend. Counter is bumped BEFORE the engine
    /// call (per the wire_vulkan trampoline) — increments even if the
    /// engine returns the known OOM error from a Vulkan device that lacks
    /// budget for the model arena (the pre-existing M_GEMMA3_VULKAN /
    /// M_QWEN3_VULKAN OOM bug; see WIRE-VULKAN-OOM-BUGFIX follow-on).
    /// Always 0 when the feature is off at build time.
    vulkan_forward_count: u64,
    /// Sprint WIRE-VULKAN: whether sp_session_register_forward_backend was
    /// invoked successfully on the target session at startup. False when
    /// SP_DAEMON_BACKEND is unset / != "vulkan", when the feature was off
    /// at build time, or when registration failed (see daemon log).
    wire_vulkan_active: bool,
}

pub async fn v1_debug_backend_counts(State(state): State<Arc<AppState>>) -> Json<BackendCounts> {
    let hex_forward_count = {
        #[cfg(all(target_os = "android", feature = "wire_hex_backend"))]
        { sp_daemon::hex_forward_dispatch::dispatch_count() }
        #[cfg(not(all(target_os = "android", feature = "wire_hex_backend")))]
        { 0u64 }
    };
    let (ntt_hex_forward_count, ntt_hex_inverse_count) = {
        #[cfg(target_os = "android")]
        { sp_daemon::ntt_hex_dispatch::dispatch_counts() }
        #[cfg(not(target_os = "android"))]
        { (0u64, 0u64) }
    };
    let cpu_forward_count = {
        // crate::cpu_forward_dispatch lives in the BINARY crate (main.rs),
        // not the lib crate, because WIRE-CPU is host-targeted and the
        // binary already has L1 bindings via its own `mod ffi`. Routes is
        // a binary-crate module, so `crate::cpu_forward_dispatch` resolves.
        #[cfg(feature = "wire_cpu_backend")]
        { crate::cpu_forward_dispatch::dispatch_count() }
        #[cfg(not(feature = "wire_cpu_backend"))]
        { 0u64 }
    };
    let cuda_forward_count = {
        #[cfg(feature = "wire_cuda_backend")]
        { sp_daemon::cuda_forward_dispatch::dispatch_count() }
        #[cfg(not(feature = "wire_cuda_backend"))]
        { 0u64 }
    };
    let vulkan_forward_count = {
        #[cfg(feature = "wire_vulkan_backend")]
        { sp_daemon::vulkan_forward_dispatch::dispatch_count() }
        #[cfg(not(feature = "wire_vulkan_backend"))]
        { 0u64 }
    };
    Json(BackendCounts {
        hex_forward_count,
        wire_hex_active: state.wire_hex_active,
        ntt_hex_forward_count,
        ntt_hex_inverse_count,
        cpu_forward_count,
        wire_cpu_active: state.wire_cpu_active,
        cuda_forward_count,
        wire_cuda_active: state.wire_cuda_active,
        vulkan_forward_count,
        wire_vulkan_active: state.wire_vulkan_active,
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
    // CONTRACT-CHAT-FULLSTACK B1/S1 — byte-exact "auditable mode". The turn
    // decodes through the exact-integer islands + dual-prime CRT-NTT attention on
    // the resident cache (run-to-run bit-identical AND build-independent, the
    // AUDITABILITY axis + the FP-reorder-immunity fix). S1 makes this the
    // DEFAULT chat decode path: `None` (field absent) ⇒ ON. Set `false`
    // explicitly to opt out to the Stage-A float path; `raw_logits:true` also
    // forces it off (the float determinism/null-floor reference). Only honored on
    // the 12B kvdecode (resident-cache) chat path.
    #[serde(default)]
    pub byteexact: Option<bool>,
    // CONTRACT-CHAT-FULLSTACK B2 (§6d-b) — XBAR episode REPLAY into this turn. When
    // set to an episode directory (holding ep.mf/ep.k/ep.v), the resident cache
    // replays that stored episode's owner K/V at [dpos,dpos+replay_npos) right after
    // the prompt prefill and before decode, so a prior memory rolls into the live
    // turn (SP_REPLAY recall, C2 #222). Default None = the B1/Stage-A path untouched
    // (byte-identical null floor). Only honored on the 12B kvdecode (resident-cache)
    // chat path. `replay_npos` bounds how many positions to recall (0 = unset → skip).
    #[serde(default)]
    pub replay: Option<String>,
    #[serde(default)]
    pub replay_npos: i32,
    // CONTRACT-CHAT-FULLSTACK A2-polish — null-floor opt-out. Default false =
    // the DEFAULT served chat, with full control-token suppression
    // (`<image|>`/`<audio|>`/`<|turn>`/… masked to -inf) so output is CLEAN at
    // every temperature incl. greedy. When true, suppression is SKIPPED (the
    // sampler is built with an empty suppress set), so the raw, un-suppressed
    // logits drive selection — this reproduces the prior greedy output
    // bit-for-bit and is the reference the byte-exact / B1 determinism leg
    // compares against (pair with temperature:0 + byteexact:true). Use only for
    // the auditability/determinism null floor; not for normal chat.
    #[serde(default)]
    pub raw_logits: bool,
    // CONTRACT-CHAT-FULLSTACK B5 — the SINGLE LATENT ENTRY POINT (CONTRACT §6).
    // When set true, the prompt is ingested through the ONE residual seam
    // (gemma4_kv_inject_tokens: per token, embd[id]*sqrt(E) staged into the inject
    // override → the model mints K/V natively) INSTEAD of gemma4_kv_prefill(ids).
    // The residual entering layer 0 is bit-identical to prefill by construction
    // (same embed arithmetic + the real id as the step token so PLE matches), so
    // output is bit-identical — this PROVES text, audio, and memory all enter
    // through the one seam. Default None ⇒ the prefill path (untouched null floor);
    // flip the daemon default by passing single_entry:true. Only honored on the
    // 12B kvdecode (resident-cache) chat path.
    #[serde(default)]
    pub single_entry: Option<bool>,
    // CONTRACT-CHAT-FULLSTACK B5 — the GENERIC residual-frame channel. Raw E-float
    // residual vectors fed straight to gemma4_kv_inject_seq (the seam AUDIO/EAR and
    // MEMORY-as-residual sources also use). Each inner Vec is one E-length frame;
    // they are injected at consecutive positions right after the prompt prefill and
    // before decode, each minted at `inject_ph` (default = the gemma-4 audio
    // placeholder 258881). Use with `prompt`/`messages` (the text turn scaffold) so
    // the model has context to digest the frames. Default None = no frames (null
    // floor). Only honored on the 12B kvdecode (resident-cache) chat path.
    #[serde(default)]
    pub inject_frames: Option<Vec<Vec<f32>>>,
    #[serde(default = "default_inject_ph")]
    pub inject_ph: i32,
    // CONTRACT-CHAT-FULLSTACK A2 — sampling knobs (L2 owns sampling over the
    // full-vocab logits row). Flattened so the request body carries them at the
    // top level: {"prompt":"...","temperature":0.7,"top_p":0.95,...}. Defaults
    // are the contract's pre-registered values; `temperature:0` = greedy argmax
    // (bit-identical to the prior hardcoded path, the G-CHAT-A2 determinism leg).
    #[serde(flatten)]
    pub sampling: crate::sampler::SamplingParams,
}

fn default_max_tokens() -> u32 {
    256
}

/// B5: default placeholder token for the generic inject_frames channel — the
/// gemma-4 audio_token_id (the KAI-3 audio port's mint placeholder).
fn default_inject_ph() -> i32 {
    258881
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
        // CONTRACT-CHAT-FULLSTACK S1: build the prompt at the TOKEN level so the
        // gemma-4 turn structure uses its REAL control tokens (`<|turn>`=105 /
        // `<turn|>`=106), not the literal `<start_of_turn>`/`<end_of_turn>`
        // strings that the encoder shatters into ordinary text (the
        // "gemma3-template-on-a-gemma4-model" bug). apply_template_ids emits the
        // control ids directly and routes only the role/content through the C BPE
        // encoder (per-fragment forced BOS stripped).
        match tokenizer.apply_template_ids(&messages) {
            Ok(ids) => ids,
            Err(e) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({
                        "error": "chat_template_unavailable",
                        "detail": e,
                        "hint": "use prompt or prompt_tokens"
                    })),
                )
                    .into_response();
            }
        }
    };

    // CONTRACT-CHAT-FULLSTACK S1 debug: log the head of the assembled prompt
    // token ids so the turn-token structure (<|turn>=105 … <turn|>=106) is
    // verifiable in the daemon log.
    {
        let head: Vec<i32> = tokens.iter().take(12).copied().collect();
        let tail: Vec<i32> = tokens.iter().rev().take(6).rev().copied().collect();
        tracing::info!("S1 prompt ids: n={} head={:?} tail={:?}", tokens.len(), head, tail);
    }

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
    // B1 / S1: byte-exact "auditable mode" for this turn (resident-cache chat
    // path only). CONTRACT-CHAT-FULLSTACK S1 makes byte-exact the DEFAULT chat
    // decode path: the exact-integer islands + dual-prime CRT-NTT attention are
    // run-to-run bit-identical AND build-independent (integer arithmetic), which
    // removes the FP-codegen reorder fragility that flipped a thin rank-2 margin
    // coherent↔garbage across rebuilds from the same HEAD. `raw_logits` (the
    // null-floor / determinism reference opt-out) forces it OFF to recover the
    // Stage-A float path bit-for-bit; an explicit `byteexact:false` also opts out.
    let byteexact = if req.raw_logits {
        false
    } else {
        // Default ON (field absent ⇒ None ⇒ true); explicit `byteexact:false` opts out.
        req.byteexact.unwrap_or(true)
    };
    // B2 (§6d-b): per-turn XBAR episode replay (None = no recall = null floor).
    let replay_dir = req.replay.clone();
    let replay_npos = req.replay_npos;
    // B5 (§6e): the single latent entry point. single_entry routes prompt ingest
    // through gemma4_kv_inject_tokens (the residual seam) instead of prefill;
    // inject_frames feeds raw residual frames (audio/memory) through the same seam.
    let single_entry = req.single_entry.unwrap_or(false);
    let inject_frames = req.inject_frames.clone();
    let inject_ph = req.inject_ph;
    // A2: build the per-request sampler from the (flattened) ChatRequest knobs.
    let sampling = req.sampling.clone();
    // A2-polish: token ids the sampler must never emit — the full control /
    // placeholder set (`<image|>`/`<audio|>`/`<|turn>`/`<turn|>`/… + structural
    // specials), masked to -inf on BOTH the sampled and the greedy path so the
    // default served chat is clean everywhere. Computed id-agnostically from the
    // tokenizer's id_to_bytes. When `raw_logits` is set, the suppress set is
    // EMPTY ⇒ the raw, un-suppressed logits drive selection (the null-floor /
    // determinism reference — reproduces the prior greedy output bit-for-bit).
    let suppress_ids = if req.raw_logits {
        Vec::new()
    } else {
        state.tokenizer.suppress_token_ids()
    };
    // Issue #115: when the client supplies no stop strings, fall back to the
    // arch's chat-format terminator (gemma's `<end_of_turn>`) so the console
    // stream ends at the turn boundary instead of running to max_tokens.
    let stop_strings = if req.stop.is_empty() {
        state.tokenizer.default_stops()
    } else {
        req.stop
    };

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
        // A2: the L2 sampler for this turn (temp=0 ⇒ strict argmax null floor).
        let mut sampler = crate::sampler::Sampler::with_suppress(sampling, suppress_ids);

        // Issue #115 (12B chat path): when the CUDA persistent-KV decode backend
        // is registered (SP_DAEMON_BACKEND=cuda + SP_DAEMON_KVDECODE=1 +
        // SP_CUDA_DECODE_INT8=1), the 12B's full-vocab head is only materializable
        // through the resident cache — sp_prefill_chunk's §6 forward bridge trips
        // "g4 probe: FULL head needs the f32 embd". Drive the single resident
        // cache directly here (serialized by its Mutex; reset per request via
        // rewind). The session-clone + prefill_chunk/decode_step path stays the
        // fallback for models whose head fits the prefill bridge (Qwen3 etc.).
        #[cfg(feature = "wire_cuda_backend")]
        let kvdecode = app.cuda_kvdecode_handle.is_some();
        #[cfg(not(feature = "wire_cuda_backend"))]
        let kvdecode = false;

        if kvdecode {
            // Drop the unused clone — the resident cache, not the clone, holds KV.
            drop(child);
            run_kvdecode_chat(
                &app, chat_id, &tokens, max_tokens, vocab_size,
                &mut logits, &mut dec_buf, &tx, &cancel_child, &sessions,
                &mut sampler, byteexact, replay_dir, replay_npos,
                single_entry, inject_frames, inject_ph,
            );
            return;
        }

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

        let mut next_token = sampler.sample(&mut logits);
        sampler.observe(next_token);

        // A2-polish: this arch's turn boundary is the `<|turn>`/`<turn|>` token
        // (no `<end_of_turn>` token exists), so treat those ids as EOS-equivalent.
        let turn_stop_ids = tokenizer.turn_stop_ids();
        'decode: for _ in 0..max_tokens {
            // EOS / turn-stop check before emitting (stop cleanly at the turn
            // boundary — the marker never reaches the stream).
            if (!tokenizer.eos_ids.is_empty() && tokenizer.eos_ids.contains(&next_token))
                || turn_stop_ids.contains(&next_token)
            {
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
                Ok(()) => {
                    next_token = sampler.sample(&mut logits);
                    sampler.observe(next_token);
                }
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

/// Issue #115 — 12B chat over the resident CUDA persistent-KV decode cache.
///
/// The 12B's tied full-vocab head cannot be materialized by the §6 prefill
/// bridge, so prompt ingest + token decode run on the single session-resident
/// `gemma4_kv_*` cache (the G-WIRE-CUDA-DECODE-GEMMA4 path). The cache is global
/// + stateful (one `dpos`), so we serialize on its Mutex and `rewind` to 0 at
/// the start of each request. The prefill head's argmax for the FIRST generated
/// token is obtained by prefilling `tokens[..n-1]` then `decode_step(tokens[n-1])`
/// (which returns that token's next-position logits) — no separate seq-peek
/// needed. Emit / EOS / stop-string handling mirrors the fallback decode loop.
#[cfg(feature = "wire_cuda_backend")]
#[allow(clippy::too_many_arguments)]
fn run_kvdecode_chat(
    app: &Arc<AppState>,
    chat_id: u64,
    tokens: &[i32],
    max_tokens: u32,
    vocab_size: usize,
    logits: &mut [f32],
    dec_buf: &mut TokenDecodeBuffer,
    tx: &tokio::sync::mpsc::Sender<Result<Event, Infallible>>,
    cancel_child: &Arc<AtomicI32>,
    sessions: &crate::sessions::Sessions,
    sampler: &mut crate::sampler::Sampler,
    byteexact: bool,
    replay_dir: Option<String>,
    replay_npos: i32,
    single_entry: bool,
    inject_frames: Option<Vec<Vec<f32>>>,
    inject_ph: i32,
) {
    use sp_daemon::cuda_kvdecode_dispatch as kv;

    let send_err = |msg: String| {
        let _ = tx.blocking_send(Ok(Event::default().data(format!("{{\"error\":\"{msg}\"}}"))));
        let _ = app.events_tx.send(DaemonEvent::Chat { chat_id, status: "cancelled" });
    };

    if tokens.is_empty() {
        send_err("kvdecode: empty prompt after tokenization".into());
        sessions.remove(chat_id);
        return;
    }

    // Serialize on the resident cache for the whole request (one GPU cache).
    let guard = match app.cuda_kvdecode_handle.as_ref() {
        Some(m) => m.lock().unwrap(),
        None => {
            send_err("kvdecode: handle missing".into());
            sessions.remove(chat_id);
            return;
        }
    };
    let handle = guard.0;

    // B1: byte-exact "auditable mode" for this turn. Set under the cache Mutex so
    // the whole decode runs on the exact-integer substrate, then reset to the
    // float null floor on EVERY exit path (RAII guard) — a request that did not
    // ask for it must see the byte-identical Stage-A path. SAFETY: handle live;
    // we hold the cache Mutex (no concurrent decode).
    struct ByteExactGuard(*mut std::ffi::c_void);
    impl Drop for ByteExactGuard {
        fn drop(&mut self) {
            // Reset to float; ignore errors (best-effort cleanup at turn end).
            let _ = unsafe { kv::set_byteexact(self.0, false) };
        }
    }
    let _bx_guard = if byteexact {
        if let Err(e) = unsafe { kv::set_byteexact(handle, true) } {
            send_err(format!("kvdecode set_byteexact(on): {e}"));
            sessions.remove(chat_id);
            return;
        }
        Some(ByteExactGuard(handle))
    } else {
        None
    };

    // Reset the resident cache to dpos=0 so each request is clean.
    // CONTRACT-CHAT-FULLSTACK B2 RING-FIX: use gemma4_kv_reset (counter reset, NO
    // journal replay) instead of rewind(pos). On the SWA ring path rewind(pos)
    // replays the undo-journal in reverse and reads jK/jV[L]+j*kvd for j up to
    // pos-1 — OUT OF BOUNDS past the flat Jmax*kvd journal once pos>Jmax (the
    // diagnosed B2 ring-reset bug). reset() zeroes dpos/commit_pos/jcur; stale
    // ring slots are never read because the next turn rewrites them in position
    // order. Equivalent to rewind(pos) on the non-ring full-cache path (slot==pos,
    // writes to [0,dpos) on the next turn), so no behavior change there.
    // SAFETY: handle is a live sp_g4_kv* owned by AppState; we hold its Mutex.
    let pos = unsafe { kv::position(handle) };
    if pos > 0 {
        if let Err(e) = unsafe { kv::reset(handle) } {
            send_err(format!("kvdecode reset: {e}"));
            sessions.remove(chat_id);
            return;
        }
    }

    // Prefill prompt[..n-1] into the resident cache, then decode_step(last) to
    // obtain the first generated token's logits. For a 1-token prompt, skip the
    // prefill and decode_step the lone token directly.
    //
    // B5 (§6e) — the SINGLE LATENT ENTRY POINT. When single_entry is set, the
    // prompt head is ingested through gemma4_kv_inject_tokens (the residual seam:
    // per token, embd[id]*sqrt(E) staged into the inject override → the model mints
    // K/V natively) instead of gemma4_kv_prefill(ids). The residual entering layer 0
    // is bit-identical to prefill by construction, so this is the same ingest
    // through the ONE seam audio + memory also use. The last token still goes
    // through decode_step(last) below (it returns the first generation logits) — for
    // single_entry we route the last token through inject_tokens too (its K/V is
    // minted via the seam) and then decode_step(last) re-runs that position to fetch
    // logits; bit-identical either way since the cache state at position p depends
    // only on tokens[0..=p].
    let (head, last) = tokens.split_at(tokens.len() - 1);
    if !head.is_empty() {
        let r = if single_entry {
            unsafe { kv::inject_tokens(handle, head) }
        } else {
            unsafe { kv::prefill(handle, head) }
        };
        if let Err(e) = r {
            send_err(format!("kvdecode {} head: {e}", if single_entry { "inject_tokens" } else { "prefill" }));
            sessions.remove(chat_id);
            return;
        }
    }
    // B5 (§6e) — the GENERIC residual-frame channel. After the prompt scaffold is
    // ingested, inject any raw residual frames (audio/memory source) at consecutive
    // positions through the same seam (gemma4_kv_inject_seq via inject_frames). Each
    // inner Vec is one E-length frame; a malformed (ragged) batch is rejected.
    if let Some(frames) = inject_frames.as_ref() {
        if !frames.is_empty() {
            let e_dim = frames[0].len();
            if e_dim == 0 || frames.iter().any(|f| f.len() != e_dim) {
                send_err("kvdecode inject_frames: ragged/empty frames (each must be E floats)".into());
                sessions.remove(chat_id);
                return;
            }
            let n_frames = frames.len() as i32;
            let flat: Vec<f32> = frames.iter().flatten().copied().collect();
            if let Err(e) = unsafe { kv::inject_frames(handle, &flat, n_frames, inject_ph) } {
                send_err(format!("kvdecode inject_frames(n={n_frames}, ph={inject_ph}): {e}"));
                sessions.remove(chat_id);
                return;
            }
        }
    }
    // B2 (§6d-b): XBAR episode REPLAY into the live turn. Recall a stored episode's
    // owner K/V into the cache at [dpos,dpos+npos) right after the prompt prefill,
    // so the last prompt token + every generated token attend across the recalled
    // memory (SP_REPLAY, C2 #222). Done under the cache Mutex; reject = rewind. A
    // turn that names no episode skips this entirely (byte-identical null floor).
    if let Some(ref dir) = replay_dir {
        if !dir.is_empty() && replay_npos > 0 {
            // SAFETY: handle live; we hold the cache Mutex (no concurrent decode).
            if let Err(e) = unsafe { kv::replay(handle, dir, replay_npos, false) } {
                send_err(format!("kvdecode replay({dir}, {replay_npos}): {e}"));
                sessions.remove(chat_id);
                return;
            }
        }
    }
    if logits.len() != vocab_size {
        send_err("kvdecode: logits buffer size mismatch".into());
        sessions.remove(chat_id);
        return;
    }
    if let Err(e) = unsafe { kv::decode_step(handle, last[0], logits) } {
        send_err(format!("kvdecode decode_step(prefill-tail): {e}"));
        sessions.remove(chat_id);
        return;
    }

    let tokenizer = app.tokenizer.clone();
    // A2-polish: this arch has no `<end_of_turn>` token; its turn boundary is
    // the `<|turn>`/`<turn|>` token. Treat those ids as EOS-equivalent so the
    // resident-cache 12B chat stops cleanly at the turn (the marker never decodes
    // into the stream). Belt-and-braces with default_stops()'s stop-strings.
    let turn_stop_ids = tokenizer.turn_stop_ids();
    let mut next_token = sampler.sample(logits);
    sampler.observe(next_token);

    'decode: for _ in 0..max_tokens {
        if (!tokenizer.eos_ids.is_empty() && tokenizer.eos_ids.contains(&next_token))
            || turn_stop_ids.contains(&next_token)
        {
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

        // Feed the just-emitted token; get logits for the next position.
        // SAFETY: handle live; logits is vocab_size f32 (checked above).
        if let Err(_e) = unsafe { kv::decode_step(handle, next_token, logits) } {
            break 'decode;
        }
        next_token = sampler.sample(logits);
        sampler.observe(next_token);
    }

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
}

// A2: `fn argmax` moved to `crate::sampler::argmax` (the temp=0 null floor);
// both decode loops now go through `Sampler::sample`.

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

// ── POST /v1/dsp/echo — §3-HX Sprint C ───────────────────────────────────
//
// Routes a raw octet-stream body through the V69 cDSP echo skel via
// FastRpcSession + DmaBuffer.  On non-android target_os, returns 501 (the
// FastRPC FFI is gated out at compile time).  On android with no admitted
// session (alloc failure, missing skel on device), also returns 501.
//
// Max body size: 8 MB (Phase 3-HX Sprint C contract; verified end-to-end
// via the parallel `sp_dsp_smoke/src/axum_server.rs` aarch64-android binary).

const DSP_ECHO_MAX_PAYLOAD: usize = 8 * 1024 * 1024;

#[cfg(target_os = "android")]
pub async fn v1_dsp_echo(
    State(state): State<Arc<AppState>>,
    body: axum::body::Bytes,
) -> Response {
    use crate::dsp_rpc::{make_scalars, DmaBuffer, RemoteArg, RemoteBuf, SpErr};
    use std::ffi::c_void;

    let n = body.len();
    if n == 0 {
        return (StatusCode::BAD_REQUEST, "empty body").into_response();
    }
    if n > DSP_ECHO_MAX_PAYLOAD {
        return (StatusCode::PAYLOAD_TOO_LARGE,
                format!("body {n} > {DSP_ECHO_MAX_PAYLOAD}")).into_response();
    }
    let Some(sess_mu) = state.dsp_session.as_ref() else {
        return (StatusCode::NOT_IMPLEMENTED, "cDSP session not admitted").into_response();
    };

    // Wrap the blocking FFI in spawn_blocking so we don't stall the tokio runtime.
    // The session Mutex serializes concurrent requests at the FFI boundary
    // (FastRPC per-handle thread-safety is single-thread).
    let body = body.to_vec();
    let state2 = state.clone();
    let result: Result<Vec<u8>, SpErr> = task::spawn_blocking(move || {
        let sess_mu = state2.dsp_session.as_ref().expect("checked above");
        let sess = sess_mu.lock().expect("dsp session mutex poisoned");
        let mut in_buf:  DmaBuffer = sess.alloc_dma(n)?;
        let mut out_buf: DmaBuffer = sess.alloc_dma(n)?;
        in_buf.as_mut_slice().copy_from_slice(&body);
        for b in out_buf.as_mut_slice().iter_mut() { *b = 0; }

        let mut prim_in: [u32; 2] = [n as u32, n as u32];
        let mut args = [
            RemoteArg { buf: RemoteBuf { pv: prim_in.as_mut_ptr() as *mut c_void, nlen: 8 }},
            RemoteArg { buf: RemoteBuf { pv: in_buf.as_mut_ptr() as *mut c_void,  nlen: n }},
            RemoteArg { buf: RemoteBuf { pv: out_buf.as_mut_ptr() as *mut c_void, nlen: n }},
        ];
        sess.invoke(make_scalars(2, 2, 1), &mut args)?;
        Ok(out_buf.as_slice().to_vec())
    })
    .await
    .unwrap_or_else(|e| Err(SpErr::HandleOpen(-(format!("join: {e}").len() as i32))));
    let _ = sess_mu;

    match result {
        Ok(out) => (StatusCode::OK, out).into_response(),
        Err(e)  => (StatusCode::INTERNAL_SERVER_ERROR,
                    format!("dsp_rpc: {e:?}")).into_response(),
    }
}

#[cfg(not(target_os = "android"))]
pub async fn v1_dsp_echo(
    State(_state): State<Arc<AppState>>,
    body: axum::body::Bytes,
) -> Response {
    let n = body.len();
    if n > DSP_ECHO_MAX_PAYLOAD {
        return (StatusCode::PAYLOAD_TOO_LARGE,
                format!("body {n} > {DSP_ECHO_MAX_PAYLOAD}")).into_response();
    }
    // Host build: FastRPC FFI is gated out.  The route exists so the daemon
    // exposes a uniform surface across host/dev and android/prod; clients
    // get a clear 501 instead of a 404 (which would suggest the route is
    // not deployed at all).
    (StatusCode::NOT_IMPLEMENTED, "v1/dsp/echo requires target_os=android").into_response()
}

// Host build: no DSP-resident model (the loader is android-only). Returns 501
// rather than 404 so the /v1/dsp surface is uniform across host and android.
#[cfg(not(target_os = "android"))]
pub async fn v1_dsp_model_info() -> Response {
    (StatusCode::NOT_IMPLEMENTED, "v1/dsp/model_info requires target_os=android").into_response()
}

// ── §3-HX cDSP model_info (android-only) ─────────────────────────────────────
// Phase 2-L3.FG: /v1/chat + /v1/pouw/ledger now run for real on android (the L1
// forward + sieve C ABI link), so their J.5 501 stubs are gone — the unified
// host handlers above serve both targets.

// /v1/dsp/model_info — reports the DSP-resident model's layer count + total
// DmaBuffer footprint (T_APPSTATE_INTEGRATION). 501 if the model failed to
// load (FastRpcSession unavailable / skel not admitted / bad model path).
#[cfg(target_os = "android")]
pub async fn v1_dsp_model_info(State(state): State<Arc<AppState>>) -> Response {
    let Some(model) = state.dsp_model.as_ref() else {
        return (StatusCode::NOT_IMPLEMENTED, "model not loaded").into_response();
    };
    let hdr = &model.0.header;
    let kv_cache_bytes = state
        .kv_cache
        .as_ref()
        .map(|k| k.lock().unwrap().0.total_bytes())
        .unwrap_or(0);
    Json(serde_json::json!({
        "n_layers":         hdr.n_layers,
        "hidden_size":      hdr.hidden_size,
        "n_heads":          hdr.n_heads,
        "n_kv_heads":       hdr.n_kv_heads,
        "vocab_size":       hdr.vocab_size,
        "total_dma_bytes":  model.0.total_dma_bytes,
        "load_wall_ms":     model.0.load_wall_ms,
        "kv_cache_bytes":   kv_cache_bytes,
    }))
    .into_response()
}

// ── Chat-integration: POST /v1/dialogue ─────────────────────────────────
//
// Single-shot JSON endpoint that drives the M.2 MeMo (Grounding → Entity ID
// → Synthesis) dialogue. Returns the final answer + 3 base64-encoded
// 64-byte SpinorReceipts (one per turn) per `reference-spinor-receipt-layout`.
//
// Returns HTTP 501 if Memory model isn't loaded (--memo-model not passed
// at daemon startup, or model load failed).
//
// This is the Option B parallel endpoint chosen in PLAN-CHAT-INTEGRATION.md;
// the existing /v1/chat is untouched and continues to serve single-model
// SSE streaming chat against the Executive model.

const DIALOGUE_MAX_PROMPT_TOKENS: usize = 64;
const DIALOGUE_MAX_TURN_TOKENS: usize = 8;

#[derive(Deserialize)]
pub struct DialogueRequest {
    pub prompt: String,
}

#[derive(Serialize)]
struct DialogueResponse {
    response: String,
    receipts: Vec<String>, // base64-encoded 64-byte SpinorReceipts
    wall_ms: u64,
    turn_us: [u64; 3],
}

/// STANDARD base64 encoder (RFC 4648). Hand-rolled to avoid adding a
/// dependency for ~6 lines of code; verified against the standard
/// alphabet `ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/`
/// with `=` padding. 64 input bytes → 88 output chars (ceil(64/3)*4 = 88).
fn base64_encode(input: &[u8]) -> String {
    const ALPHABET: &[u8; 64] =
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(((input.len() + 2) / 3) * 4);
    let mut i = 0;
    while i + 3 <= input.len() {
        let n = ((input[i] as u32) << 16) | ((input[i + 1] as u32) << 8) | (input[i + 2] as u32);
        out.push(ALPHABET[((n >> 18) & 0x3F) as usize] as char);
        out.push(ALPHABET[((n >> 12) & 0x3F) as usize] as char);
        out.push(ALPHABET[((n >> 6) & 0x3F) as usize] as char);
        out.push(ALPHABET[(n & 0x3F) as usize] as char);
        i += 3;
    }
    let rem = input.len() - i;
    if rem == 1 {
        let n = (input[i] as u32) << 16;
        out.push(ALPHABET[((n >> 18) & 0x3F) as usize] as char);
        out.push(ALPHABET[((n >> 12) & 0x3F) as usize] as char);
        out.push('=');
        out.push('=');
    } else if rem == 2 {
        let n = ((input[i] as u32) << 16) | ((input[i + 1] as u32) << 8);
        out.push(ALPHABET[((n >> 18) & 0x3F) as usize] as char);
        out.push(ALPHABET[((n >> 12) & 0x3F) as usize] as char);
        out.push(ALPHABET[((n >> 6) & 0x3F) as usize] as char);
        out.push('=');
    }
    out
}

pub async fn v1_dialogue(
    State(state): State<Arc<AppState>>,
    Json(req): Json<DialogueRequest>,
) -> Response {
    // 501 if Memory model isn't loaded.
    if state.memo_model.is_none() || state.memo_session.is_none() || state.memo_tokenizer.is_none() {
        return (
            StatusCode::NOT_IMPLEMENTED,
            Json(serde_json::json!({
                "error": "memo_model_not_loaded",
                "hint": "start sp-daemon with --memo-model / --memo-tokenizer or SP_MEMO_MODEL_PATH / SP_MEMO_TOKENIZER_PATH",
            })),
        )
            .into_response();
    }

    if req.prompt.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "prompt required"})),
        )
            .into_response();
    }

    // Clone Executive base session (mirrors /v1/chat lines 150-154).
    let exec_cancel = Arc::new(AtomicI32::new(0));
    let exec_child = {
        let guard = state.session.lock().unwrap();
        guard.clone_session(exec_cancel.clone())
    };
    let mut exec_child = match exec_child {
        Ok(s) => s,
        Err(e) => {
            return (StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": format!("exec clone: {e}")})))
                .into_response();
        }
    };

    // Clone Memory base session.
    let memo_cancel = Arc::new(AtomicI32::new(0));
    let memo_child = {
        let guard = state.memo_session.as_ref().expect("checked above").lock().unwrap();
        guard.clone_session(memo_cancel.clone())
    };
    let mut memo_child = match memo_child {
        Ok(s) => s,
        Err(e) => {
            return (StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": format!("memo clone: {e}")})))
                .into_response();
        }
    };

    let exec_tokenizer = state.tokenizer.clone();
    let exec_vocab = state.vocab_size;
    let memo_vocab = state.memo_vocab_size;
    let prompt = req.prompt;

    // Signal the mining loop to back off for the duration (mirrors /v1/chat:176).
    state.inference_active.store(true, std::sync::atomic::Ordering::Relaxed);
    struct InferenceGuard(Arc<AtomicBool>);
    impl Drop for InferenceGuard {
        fn drop(&mut self) { self.0.store(false, std::sync::atomic::Ordering::Relaxed); }
    }
    let _guard = InferenceGuard(state.inference_active.clone());

    // Drive the dialogue on a spawn_blocking thread (L1 forward is sync).
    let result = tokio::task::spawn_blocking(move || {
        let _g = _guard;
        let caps = sp_daemon::dialogue::DialogueCaps {
            max_prompt_tokens: DIALOGUE_MAX_PROMPT_TOKENS,
            max_query_tokens:  DIALOGUE_MAX_TURN_TOKENS,
            max_response_tokens: DIALOGUE_MAX_TURN_TOKENS,
            max_answer_tokens: DIALOGUE_MAX_TURN_TOKENS,
        };
        let mut pool = sp_daemon::dialogue::DialoguePool::new(exec_vocab, memo_vocab, &caps);
        crate::dialogue_runner::run_dialogue(
            &mut exec_child,
            &mut memo_child,
            exec_tokenizer.as_ref(),
            &mut pool,
            &prompt,
            &caps,
        )
    }).await;

    match result {
        Ok(Ok(outcome)) => {
            // ledger-autowire: best-effort append of all 3 receipts to the
            // shared PoUW ledger BEFORE building the HTTP response. Per the
            // sprint plan + the broader M.4 design, the ledger is
            // observational, not a transactional gate — a lock or append
            // failure logs a warning and the response still ships. The
            // critical section is ~10 µs total (3 × p99=3 µs per M.4
            // measurement) so contention is structurally irrelevant.
            if let Some(ledger) = &state.ledger {
                match ledger.lock() {
                    Ok(mut guard) => {
                        for (idx, r) in outcome.receipts.iter().enumerate() {
                            if let Err(e) = guard.append(r) {
                                tracing::warn!(
                                    error = %e,
                                    receipt_idx = idx,
                                    "ledger-autowire: append failed; response still returns"
                                );
                            }
                        }
                    }
                    Err(poisoned) => {
                        tracing::warn!(
                            error = ?poisoned,
                            "ledger-autowire: mutex poisoned; skipping append"
                        );
                    }
                }
            }
            let receipts_b64: Vec<String> = outcome
                .receipts
                .iter()
                .map(|r| base64_encode(&r.as_bytes()))
                .collect();
            let body = DialogueResponse {
                response: outcome.final_answer,
                receipts: receipts_b64,
                wall_ms: outcome.total_wall_us / 1000,
                turn_us: outcome.turn_us,
            };
            (StatusCode::OK, Json(serde_json::to_value(&body).unwrap_or(serde_json::json!({})))).into_response()
        }
        Ok(Err(e)) => {
            (StatusCode::INTERNAL_SERVER_ERROR,
             Json(serde_json::json!({"error": format!("run_dialogue: {e}")})))
            .into_response()
        }
        Err(e) => {
            (StatusCode::INTERNAL_SERVER_ERROR,
             Json(serde_json::json!({"error": format!("spawn_blocking: {e}")})))
            .into_response()
        }
    }
}

#[cfg(test)]
mod chat_integration_tests {
    use super::*;

    #[test]
    fn base64_encode_known_vectors() {
        // RFC 4648 test vectors.
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foob"), "Zm9vYg==");
        assert_eq!(base64_encode(b"fooba"), "Zm9vYmE=");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn base64_encode_64_bytes_to_88_chars() {
        let input = [0xA5u8; 64];
        let out = base64_encode(&input);
        assert_eq!(out.len(), 88);
        // First char of all-0xA5 input: 0xA5=10100101 → first 6 bits = 101001 = 41 = 'p'
        assert!(out.starts_with('p'));
    }
}