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
        let guard = state.session.as_ref().expect("L1 session unavailable (qwen36 lane)").lock().unwrap();
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
    // CONTRACT-CHAT-FULLSTACK B3 — AUTONOMOUS MEMORY RECALL. When true (and the
    // daemon loaded an episode registry via SP_RECALL_REGISTRY), the daemon
    // computes THIS turn's 256-bit C2 query signature from the resident cache's
    // global-layer K (after the prompt prefill), Hamming-matches it against the
    // registry, and — if the best match clears TAU_BITS — auto-replays that
    // episode into the live turn BEFORE decode. No operator `replay` field
    // needed: the model selects the relevant memory ON ITS OWN. Default
    // None/false = OFF = the byte-untouched non-recall path (the null floor; the
    // foreign-reject safety leg also runs only when this is on). An explicit
    // `replay` still wins (operator override) and suppresses auto-recall. Only
    // honored on the 12B kvdecode (resident-cache) chat path.
    #[serde(default)]
    pub auto_recall: Option<bool>,
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
    // N5: live end-of-turn logit bias (the coherence knob). None => the daemon's
    // SP_EOT_BIAS env default. Tunable per-request from the console GUI.
    #[serde(default)]
    pub eot_bias: Option<f32>,
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

    // B4 NIGHTSHIFT: capture THIS turn's last user message text (raw, NO chat
    // template — match the curator, which captured raw needle sentences) so the
    // turn-end consolidation hook can mint a position-0 standalone episode. Only
    // meaningful when `messages` was supplied; `prompt`/`prompt_tokens` callers
    // pass None (no consolidation). Cheap clone; ignored unless SP_B4_NIGHTSHIFT=1.
    let raw_user: Option<String> = req.messages.as_ref().and_then(|ms| {
        ms.iter().rev().find(|m| m.role == "user").map(|m| m.content.clone())
    });
    // RECALL MULTI-TURN FIX (2026-07-02, G-ONECONFIG-LIVE C-phase root cause): recall
    // delivery rebuilds the prompt and previously kept ONLY the last user message —
    // the conversation history vanished, so a turn-2 question about turn-1 content
    // could never be answered on a recall turn. Keep a clone of the full message
    // list so the systemecho delivery can preserve the conversation.
    let orig_msgs: Option<Vec<Message>> = req.messages.clone();

    // LIVE CONSOLIDATION HOOK: dump the current conversation to SP_CURRENT_CONVO so the
    // harness agency scheduler can consolidate the LIVE chat (durable facts -> mid-term
    // registry, transcript -> long-term MEM-OKF full+summary) on its heartbeat -- no manual
    // step. Best-effort, messages-only; unset env = no-op (byte-identical null floor).
    if let Ok(conv_path) = std::env::var("SP_CURRENT_CONVO") {
        if let Some(ms) = req.messages.as_ref() {
            let arr: Vec<serde_json::Value> = ms.iter()
                .map(|m| serde_json::json!({"role": m.role, "content": m.content}))
                .collect();
            if let Ok(s) = serde_json::to_string(&arr) {
                let _ = std::fs::write(&conv_path, s);
            }
        }
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

    // ── NORTHSTAR serve (CONTRACT-QWEN36-SERVE): the qwen36 35B-A3B lane ─────
    // When the loaded model is the GDN+MoE hybrid (arch_id 8), the gemma L1
    // session/kvdecode machinery below does not apply: decode via qwen36_step
    // (persistent recurrent state, GPU hybrid hooks booted at daemon start —
    // G-MOE-GPU4-PINNED 6.073 tok/s / 337x) and stream the same ChatDelta/[DONE]
    // SSE shape the console already speaks. v1 = greedy argmax (the gate's
    // determinism leg); sampling knobs come after G-QWEN36-SERVE is GREEN.
    if let Some(lane) = state.qwen36_lane.as_ref() {
        let lane = lane.clone();
        let tok = tokenizer.clone();
        let cancel = Arc::new(AtomicI32::new(0));
        let chat_id = state.sessions.register(cancel.clone());
        let sessions = state.sessions.clone();
        let max_new = req.max_tokens as usize;
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(64);
        tokio::task::spawn_blocking(move || {
            let eos = tok.eos_ids.clone();
            // Split-token UTF-8 guard: BPE token bytes may end mid-codepoint;
            // hold the incomplete tail back until the next token completes it.
            let mut pending: Vec<u8> = Vec::new();
            let res = lane.run_turn(&tokens, max_new, &eos, |id| {
                if cancel.load(std::sync::atomic::Ordering::Relaxed) != 0 {
                    return false;
                }
                pending.extend_from_slice(tok.decode_token(id));
                let (text, keep) = match std::str::from_utf8(&pending) {
                    Ok(s) => (s.to_string(), 0usize),
                    Err(e) => {
                        let ok = e.valid_up_to();
                        (String::from_utf8_lossy(&pending[..ok]).into_owned(), pending.len() - ok)
                    }
                };
                pending.drain(..pending.len() - keep);
                if !text.is_empty() {
                    let ev = Event::default().data(
                        serde_json::to_string(&ChatDelta { delta: text, chat_id }).unwrap(),
                    );
                    if tx.blocking_send(Ok(ev)).is_err() {
                        return false; // client went away
                    }
                }
                true
            });
            match res {
                Ok((out, tokps)) => {
                    tracing::info!("qwen36 turn done: {} tokens @ {:.3} tok/s", out.len(), tokps);
                }
                Err(e) => {
                    tracing::error!("qwen36 turn failed: {e}");
                    let _ = tx.blocking_send(Ok(Event::default()
                        .data(format!("{{\"error\":\"{e}\",\"chat_id\":{chat_id}}}"))));
                }
            }
            let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));
            sessions.remove(chat_id);
        });
        return sse_response(
            Sse::new(ReceiverStream::new(rx))
                .keep_alive(KeepAlive::new().interval(Duration::from_secs(15)).text("keepalive")),
        );
    }

    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(64);

    // Clone base session — hold Mutex only during sp_session_clone (sub-ms).
    let cancel_child = Arc::new(AtomicI32::new(0));
    let child_result = {
        let guard = state.session.as_ref().expect("L1 session unavailable (qwen36 lane)").lock().unwrap();
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
    // B3: autonomous recall toggle. An explicit request `auto_recall` always wins;
    // otherwise it defaults to SP_AUTO_RECALL_DEFAULT (so the served console/UI gets
    // recall without sending the flag per-request). Env unset => false = null floor.
    let auto_recall = req.auto_recall
        .unwrap_or(std::env::var("SP_AUTO_RECALL_DEFAULT").ok().as_deref() == Some("1"));
    // B5 (§6e): the single latent entry point. single_entry routes prompt ingest
    // through gemma4_kv_inject_tokens (the residual seam) instead of prefill;
    // inject_frames feeds raw residual frames (audio/memory) through the same seam.
    let single_entry = req.single_entry.unwrap_or(false);
    let eot_bias = req.eot_bias; // N5: per-request override of the SP_EOT_BIAS default
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
                single_entry, inject_frames, inject_ph, auto_recall,
                raw_user, orig_msgs, eot_bias,
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
// ── Persistent O(1) conversation KV (SP_PERSIST_KV) ─────────────────────────────
// The tokens currently committed in the resident KV (prompt + generated). When the next
// turn's prompt is a STRICT PREFIX of this (the conversation just grew by appending) and the
// cache position matches, we PREFILL ONLY THE NEW SUFFIX instead of rewinding to 0 and
// re-prefilling the whole conversation -- the O(n)-per-turn cost that drops tok/s 30->1 in long
// chats. Byte-exact: the reused prefix's KV is the same deterministic computation as
// re-prefilling it. Default-off, and only on the PLAIN decode path (no recall/agency side-calls,
// which reset the cache), so the committed sequence always mirrors the cache exactly.
#[cfg(feature = "wire_cuda_backend")]
static KV_COMMITTED: std::sync::Mutex<Vec<i32>> = std::sync::Mutex::new(Vec::new());

// B4-SEAL (GEODESIC session 2026-07-03): mint the L5 query-key for a LIVE-captured
// episode with the SAME provenance as the curated corpus (write_ep_l5.py): a
// standalone position-0 forward of the episode text on a throwaway scratch session,
// global-Q of the LAST token, mean-over-heads of global layer 5, L2-normed
// (recall::l5_query_embed). Writes <dir>/ep.l5 (raw LE f32[512] — exactly what
// load_episode_l5key reads back after a restart) and returns the key for the
// in-session Episode. None on any failure — the episode then behaves exactly as
// before this fix (W_c/C2-selectable, L5-invisible), never worse.
// THIS is the seam that seals the system: without it, grown memories are invisible
// to the deployed L5 selector (the old l5key: Vec::new() "follow-up" comment).
#[cfg(feature = "wire_cuda_backend")]
fn mint_live_ep_l5(qm: *const std::ffi::c_void, toks: &[i32], dir: &std::path::Path) -> Option<Vec<f32>> {
    use sp_daemon::cuda_kvdecode_dispatch as kv;
    use sp_daemon::recall;
    if toks.len() < 2 { return None; }
    let scratch = unsafe { kv::open(qm, toks.len() as i32 + 8) }.ok()?;
    let n_global = recall::NL / recall::PERIOD;
    let mut ql = vec![0.0f32; n_global * recall::G_NH * recall::HD];
    let key = (|| {
        unsafe { kv::prefill(scratch, &toks[..toks.len() - 1]) }.ok()?;
        unsafe { kv::read_global_q(scratch, toks[toks.len() - 1], &mut ql) }.ok()?;
        let k = recall::l5_query_embed(&ql);
        if k.len() != recall::HD { return None; }
        Some(k)
    })();
    // SAFETY: scratch came from open() above; release is NULL-safe/idempotent.
    unsafe { kv::release_for_model(scratch) };
    if let Some(ref k) = key {
        let bytes: Vec<u8> = k.iter().flat_map(|x| x.to_le_bytes()).collect();
        if std::fs::write(dir.join("ep.l5"), bytes).is_err() {
            tracing::warn!("B4-L5-MINT: ep.l5 sidecar write failed at {}", dir.display());
        }
    } else {
        tracing::warn!("B4-L5-MINT: L5 key mint failed for {} (episode stays L5-invisible)", dir.display());
    }
    key
}

// LAYER-3 MERGE helper: capture an ARBITRARY text as a new live episode, with the
// SAME provenance as the NIGHTSHIFT path (BOS kept + trailing newline, batched forward
// via the resident model, real C2 sig, persisted to the registry if SP_NIGHTSHIFT_PERSIST).
// Used to write the SYNTHESIZED fact a MERGE consolidates into. Returns true on success.
// Kept separate from the inline B4 capture so the proven NIGHTSHIFT path is untouched;
// uses a millis-unique episode name so it can never collide with a forgotten ep dir.
#[cfg(feature = "wire_cuda_backend")]
fn capture_live_episode(app: &Arc<AppState>, text: &str) -> bool {
    use sp_daemon::cuda_kvdecode_dispatch as kv;
    let text_nl = format!("{text}\n");
    let toks = match app.tokenizer.encode(&text_nl) { Ok(t) => t, Err(_) => return false };
    let ntok = toks.len();
    if ntok < 4 { return false; }
    let qm = {
        let mut sguard = app.session.as_ref().expect("L1 session unavailable (qwen36 lane)").lock().unwrap();
        let sraw = sguard.raw_ptr() as *mut sp_daemon::ffi_l1::sp_session;
        (unsafe { sp_daemon::ffi_l1::sp_session_qwen3_model(sraw) }) as *const std::ffi::c_void
    };
    if qm.is_null() { return false; }
    let uniq = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis()).unwrap_or(0);
    let name = format!("ep_live_m{uniq}");
    let engine_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().and_then(|p| p.parent()).unwrap_or_else(|| std::path::Path::new("."));
    let dir = engine_root.join("_nightshift_live").join(&name);
    let dir_str = dir.to_string_lossy().to_string();
    if std::fs::create_dir_all(&dir).is_err() { return false; }
    { let _ = std::fs::write(dir.join("ep.txt"), text);
      let _ = std::fs::write(dir.join("ep.tok"), toks.iter().map(|t| t.to_string()).collect::<Vec<_>>().join("\n")); }
    if unsafe { kv::capture_batched(qm, &toks, &dir_str) }.is_err() { return false; }
    let (gk, ng) = match sp_daemon::recall::load_episode_global_k(&dir_str, ntok as i32) {
        Some(x) => x, None => return false };
    let npos_sig = if ng > 0 { gk.len() / (ng * sp_daemon::recall::HD) } else { 0 };
    let sig = if npos_sig > 0 { app.recall_proj.signature(&gk, ng, npos_sig) } else { [0u64; 4] };
    // B4-SEAL: mint the L5 key so the merged episode is visible to the live selector.
    let l5k = mint_live_ep_l5(qm, &toks, &dir).unwrap_or_default();
    if std::env::var("SP_NIGHTSHIFT_PERSIST").ok().as_deref() == Some("1") {
        if let Ok(reg_path) = std::env::var("SP_RECALL_REGISTRY") {
            let sig_hex = format!("{:016x}{:016x}{:016x}{:016x}", sig[3], sig[2], sig[1], sig[0]);
            let line = serde_json::json!({
                "name": name.clone(), "dir": dir_str.clone(), "npos": ntok as i32,
                "topic": text, "text": text, "sig_bits": sig_hex,
            }).to_string();
            use std::io::Write as _;
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&reg_path) {
                let _ = writeln!(f, "{line}");
            }
        }
    }
    let topic: String = text.chars().take(40).collect();
    let mut ns = app.nightshift.write().unwrap();
    ns.push(sp_daemon::recall::Episode {
        name, dir: dir_str.clone(), npos: ntok as i32, topic,
        text: text.to_string(), sig, gk, gk_ng: ng, tokens: Some(toks),
        l5key: l5k, // B4-SEAL: minted at capture (mint_live_ep_l5); empty only on mint failure.
    });
    tracing::info!("LAYER-3 MERGE: captured synthesized episode -> \"{}\"", text.chars().take(60).collect::<String>());
    true
}

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
    auto_recall: bool,
    raw_user: Option<String>,
    orig_msgs: Option<Vec<Message>>,
    eot_bias_req: Option<f32>,
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

    // ── TELEPATHY served splice (SP_TELEPATHY_CHAT=1, default-off = byte-identical null floor) ──
    // The cemented two-stage delegate (TELE-12/13/14/15) on the LIVE /v1/chat path: stage 1 routes
    // on the latent, stage 2 hands the CLEAN user text to the qwen coder (CPU L1) and streams its
    // answer into the SSE {delta} stream instead of the Gemma decode. NEVER fuses latent+text.
    // Placed BEFORE the GPU cache guard so a delegated turn never touches the Gemma resident cache.
    // NOTE distinct from SP_TELEPATHY (the one-shot parity verb in main.rs, which early-exits before
    // serving) — the served path uses its own flag. v1: route is SP_ROUTE_FORCE-driven; autonomous
    // feat-route (the TELE-7 head on a NON-COMMITTING capture_feat) is v1.1 (capture_feat is
    // async-armed and would commit the cache).
    // ── SPINE pre-cache route seam (SP_SPINE=1): telepathy expressed as
    //    LatentDecision::Route, routed + executed through spine.rs. Default (spine
    //    off) keeps the proven inline branch below, byte-for-byte. ──
    if std::env::var("SP_SPINE").as_deref() == Ok("1")
        && std::env::var("SP_TELEPATHY_CHAT").as_deref() == Ok("1") {
        if let Some(user) = raw_user.as_ref().map(|s| s.trim().to_string()).filter(|s| !s.is_empty()) {
            let decision = crate::spine::route_decision(&user);
            if matches!(decision, crate::spine::LatentDecision::Route { .. }) {
                let emit = |t: String| {
                    let payload = serde_json::to_string(&ChatDelta { delta: t, chat_id }).unwrap_or_default();
                    tx.blocking_send(Ok(Event::default().data(payload))).is_ok()
                };
                let _ = crate::spine::execute_route(&decision, &user, emit);
                let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));
                let _ = app.events_tx.send(DaemonEvent::Chat { chat_id, status: "done" });
                sessions.remove(chat_id);
                return;
            }
        }
    } else if std::env::var("SP_TELEPATHY_CHAT").as_deref() == Ok("1") {
        if let Some(user) = raw_user.as_ref().map(|s| s.trim().to_string()).filter(|s| !s.is_empty()) {
            if let crate::telepathy::RouteDecision::Telepathy(bid) = crate::telepathy::decide_route(&[0.0f32]) {
                let marker = std::env::var("SP_TELEPATHY_MARKER").as_deref() != Ok("0");
                let emit = |t: String| {
                    let payload = serde_json::to_string(&ChatDelta { delta: t, chat_id }).unwrap_or_default();
                    tx.blocking_send(Ok(Event::default().data(payload))).is_ok()
                };
                tracing::info!("TELEPATHY: route -> delegate(bridge {bid}); user={:?}", user);
                if marker { let _ = emit("\u{27E6}delegate: qwen2.5-coder\u{27E7}\n".to_string()); }
                match crate::telepathy::delegate_execute(&user, bid) {
                    Ok(ans) => { let _ = emit(ans); }
                    Err(e)  => { let _ = emit(format!("[delegate error: {e}]")); }
                }
                if marker { let _ = emit("\n\u{27E6}/delegate\u{27E7}".to_string()); }
                let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));
                let _ = app.events_tx.send(DaemonEvent::Chat { chat_id, status: "done" });
                sessions.remove(chat_id);
                return;
            }
        }
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
    // PERSISTENT O(1) KV (SP_PERSIST_KV): when the committed cache is a STRICT PREFIX of this
    // prompt AND the cache position equals its length (the cache holds exactly it), reuse it and
    // prefill only the new suffix -- no reset, no re-prefill of the conversation.
    // RECALL EXTENSION: auto_recall is now ALLOWED. The W_c / q·K relevance scoring reads the
    // cache's global K/Q NON-COMMITTINGLY (read_global_q/k roll the cache back), so on a NULL
    // (no-fire) turn the plain-prompt cache stays pristine and persist engages with full speedup.
    // A PICK injects the chosen episode's K/V at [dpos,dpos+npos) for synthesis -> that turn is
    // marked dirty at the commit point (recalled.is_some()) so the NEXT turn full-prefills clean.
    // STILL EXCLUDED (these rebuild / speculatively inject into the cache in ways a token-sequence
    // committed cannot mirror): the SPECULATIVE recall paths (SP_B3_JUDGE / SP_B3_DISPOSER /
    // SP_INT2), the memory-agency writers (SP_DECIDE / SP_FORGET / SP_B4_NIGHTSHIFT), and the
    // operator replay / single_entry / inject_frames seams.
    // Default-ON (G-PERSIST-KV GREEN 2026-06-30: 6-turn byte-identical on==off; TTFT off 7.47x growth
    // vs on flat; engine receipts tests/perf/_persist_gate_{off,on}.json). SP_PERSIST_KV=0 forces the
    // O(n) re-prefill null floor. The cache-mutating paths below still hard-exclude persist (reset/turn).
    let persist_kv = std::env::var("SP_PERSIST_KV").ok().as_deref() != Some("0")
        && replay_dir.is_none() && !single_entry && inject_frames.is_none()
        && std::env::var("SP_DECIDE").ok().as_deref() != Some("1")
        && std::env::var("SP_FORGET").ok().as_deref() != Some("1")
        && std::env::var("SP_B4_NIGHTSHIFT").ok().as_deref() != Some("1")
        && std::env::var("SP_B3_JUDGE").ok().as_deref() != Some("1")
        && std::env::var("SP_B3_DISPOSER").is_err()
        && std::env::var("SP_INT2").ok().as_deref() != Some("1");
    let mut prefill_from: usize = 0;
    if persist_kv {
        let committed = KV_COMMITTED.lock().unwrap();
        let cl = committed.len();
        // LONGEST-COMMON-PREFIX reuse. The cache holds exactly `committed` (pos == cl). Reuse the
        // common prefix of committed and this prompt, REWINDING the small diverging committed tail.
        // drop==0 is the strict-prefix append (no rewind). A bounded drop (<= REWIND_BOUND, inside
        // the SWA undo-journal SP_G4_KV_JMAX=64) lets a PRE-WARMED persona+tools prefix be reused even
        // though the first real turn diverges right after it (the dummy warm-up tail is rewound).
        // Byte-exact: the reused prefix K/V is the same deterministic compute; the new suffix is
        // prefilled fresh, overwriting the rewound positions.
        const REWIND_BOUND: usize = 32;
        if cl >= 1 && pos as usize == cl {
            let maxp = cl.min(tokens.len().saturating_sub(1));
            let mut lcp = 0usize;
            while lcp < maxp && tokens[lcp] == committed[lcp] { lcp += 1; }
            let drop_n = cl - lcp;
            if lcp >= 1 && lcp < tokens.len() && drop_n <= REWIND_BOUND {
                if drop_n > 0 {
                    if let Err(e) = unsafe { kv::rewind(handle, drop_n as i32) } {
                        send_err(format!("kvdecode persist rewind({drop_n}): {e}"));
                        sessions.remove(chat_id);
                        return;
                    }
                }
                prefill_from = lcp;
                tracing::info!(
                    "PERSIST-KV: reuse {} of {} committed (drop {}); prefill suffix {} (full would be {})",
                    lcp, cl, drop_n, tokens.len() - lcp, tokens.len());
            }
        }
    }
    if prefill_from == 0 && pos > 0 {
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
    // PERSIST-KV: skip the head positions already committed in the reused cache (prefill_from==0
    // on the normal path => the whole head, byte-identical null floor).
    let head = &head[prefill_from.min(head.len())..];
    if !head.is_empty() {
        // #41 BATCH PREFILL (CONTRACT-BATCH-PREFILL, SP_KV_PREFILL_BATCH=1, default-off):
        // on a COLD turn (prefill_from==0 = no persist prefix reused) with a large head,
        // one n-wide batched forward replaces the per-token launch storm (340tok 12-18s /
        // 560tok 71s -> GPU-saturated). FLOAT (not byte-exact) chat speed mode; the C side
        // enforces cold + ring-off + full-cache and ERRORS otherwise, so any precondition
        // miss falls THROUGH to the per-token path (byte-identical null floor). Not for the
        // single_entry seam. Small heads stay per-token (batch alloc overhead not worth it).
        let want_batch = !single_entry
            && prefill_from == 0
            && head.len() > 64
            && std::env::var("SP_KV_PREFILL_BATCH").as_deref() == Ok("1");
        let batched_ok = if want_batch {
            match unsafe { kv::prefill_batched(handle, head) } {
                Ok(()) => { tracing::info!("BATCH-PREFILL: {} tokens via one batched forward (cold, ring-off)", head.len()); true }
                Err(e) => { tracing::info!("BATCH-PREFILL declined ({e}) -> per-token prefill"); false }
            }
        } else { false };
        if !batched_ok {
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
    }
    // G-INT-2-FIX (text-in-context recall): the synthesis tail token. Defaults to
    // last[0] (the original prompt's last token). When the B3-JUDGE PICKs a memory
    // it rebuilds the cache from an AUGMENTED prompt (memory text prepended to the
    // user query, RAG-style) and overrides syn_last with that augmented prompt's
    // last token, so the synthesis decode_step(syn_last) below starts from the
    // text-in-context reconstruction. NULL / judge-off leaves syn_last == last[0]
    // (byte-identical null floor).
    let mut syn_last: i32 = last[0];
    // ATTR-GATE zero-inference decline: when set (by the SP_RECALL_ATTR_GATE branch on
    // attribute-absence), the synthesis forward is SKIPPED and this fixed string is
    // streamed instead — the gemma4 decode loop never runs (see the seam before
    // decode_step(syn_last)). Default None => normal synthesis (null floor preserved).
    let mut symbolic_decline: Option<String> = None;
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
    // B3: AUTONOMOUS MEMORY RECALL. With auto_recall on, NO explicit `replay`,
    // and a loaded registry: compute THIS turn's 256-bit C2 query signature from
    // the resident cache's GLOBAL-layer K (now holding the prompt's prefilled
    // positions), Hamming-match it against the registry, and auto-replay the best
    // episode iff it clears TAU_BITS. This is the headline "it remembers on its
    // own" path: the model self-selects the relevant memory with no operator
    // intervention. The match score is ALWAYS logged (the foreign-reject safety
    // leg needs the below-TAU case to be visible). dpos here == head.len() (the
    // prompt positions); the chosen episode is injected at [dpos,dpos+npos) just
    // like B2, so the last prompt token + every generated token attend over it.
    // B3-v2: the q·K ATTENTION-RELEVANCE selector replaces the v1 centroid-Hamming
    // sig (which did not separate question→passage: right episode argmax only 1/5,
    // episodes mutually agree ~200/256). v2 scores each episode by the model's NATIVE
    // attention relevance — q·K, where q is THIS query's last-token global-layer query
    // (gemma4_kv_read_global_q, the SAME resident path the decode runs on) and K is the
    // episode's stored global-K (ep.k). Ranks on the top-m-mean relevance; fires the
    // argmax iff it clears SP_B3_TAU_QK (a relevance threshold set from the MEASURED
    // separation — no guessing). Score reported either way (the foreign-reject leg
    // needs the below-TAU case visible). `recalled` holds (name, topm*1000 as u32).
    let mut recalled: Option<(String, u32)> = None;
    // G-INT-2: when SP_INT2 Stage-2 has rendered an ACCEPT/NULL verdict, the legacy
    // SP_B3_WC / SP_B3_DISPOSER / q·K fire branches MUST NOT double-fire on the same turn.
    let mut int2_decided = false;
    // B3-JUDGE: when the generative judge fires, stash (ep_name, injected_text)
    // for the post-synthesis CPU-side grounding check (parametric-hallucination flag).
    let mut judge_ground: Option<(String, String)> = None;
    // ── LAYER-2: FORGET (SP_FORGET=1) — the organism removes a memory ──────────────
    // The building block for memory agency. On a forget intent ("forget ...") the best
    // token-overlap match across the curated registry + live nightshift is dropped from
    // the live set AND rewritten out of the persisted registry (all exact-duplicate
    // copies), then confirmed via text-in-context. First operator-triggered; the same
    // removal the model will later invoke itself. Default-off = byte-identical null floor.
    let mut forget_done = false;
    if auto_recall && replay_dir.is_none()
        && std::env::var("SP_FORGET").ok().as_deref() == Some("1")
    {
        let is_forget = raw_user.as_ref().map(|s| {
            let l = s.to_lowercase();
            l.contains("forget") || l.contains("delete that") || l.contains("erase")
        }).unwrap_or(false);
        if is_forget {
            use sp_daemon::recall;
            let ftext = raw_user.as_ref().map(|s| s.trim().to_string()).unwrap_or_default();
            let q = ftext.to_lowercase().replace("forget", " ").replace("please", " ").replace("erase", " ");
            let mut best: Option<(f32, String)> = None; // (overlap, episode text)
            if let Some(reg) = app.recall_registry.as_ref() {
                for ep in reg.iter() {
                    let ov = recall::token_overlap(&q, &ep.text);
                    if best.as_ref().map_or(true, |(b, _)| ov > *b) { best = Some((ov, ep.text.clone())); }
                }
            }
            {
                let ns = app.nightshift.read().unwrap();
                for ep in ns.iter() {
                    let ov = recall::token_overlap(&q, &ep.text);
                    if best.as_ref().map_or(true, |(b, _)| ov > *b) { best = Some((ov, ep.text.clone())); }
                }
            }
            if let Some((ov, text)) = best.filter(|(ov, _)| *ov >= 0.25) {
                // (1) drop from the live nightshift set (all exact copies)
                { let mut ns = app.nightshift.write().unwrap(); ns.retain(|e| e.text != text); }
                // (2) rewrite the persisted registry, dropping every line with this text
                if let Ok(reg_path) = std::env::var("SP_RECALL_REGISTRY") {
                    if let Ok(content) = std::fs::read_to_string(&reg_path) {
                        let kept: Vec<&str> = content.lines().filter(|line| {
                            match serde_json::from_str::<serde_json::Value>(line) {
                                Ok(v) => v.get("text").and_then(|x| x.as_str()) != Some(text.as_str()),
                                Err(_) => !line.trim().is_empty(),
                            }
                        }).collect();
                        let mut out = kept.join("\n");
                        if !out.is_empty() { out.push('\n'); }
                        let _ = std::fs::write(&reg_path, out);
                    }
                }
                tracing::info!("LAYER-2 FORGET: removed memory (overlap {:.3}) -> \"{}\"", ov, text);
                // (3) confirm via text-in-context (the judge-PICK synthesis machinery)
                let aug = format!(
                    "Context: at the user's request you have just permanently removed this memory: \"{}\".\n\nUser: {}\n\nIn one short sentence, confirm that you have forgotten it.",
                    text, ftext);
                let aug_msgs = vec![Message { role: "user".to_string(), content: aug }];
                if let Ok(aug_toks) = app.tokenizer.apply_template_ids(&aug_msgs) {
                    if aug_toks.len() >= 2 {
                        let _ = unsafe { kv::reset_cold(handle) };
                        let (aug_head, aug_last) = aug_toks.split_at(aug_toks.len() - 1);
                        if aug_head.is_empty() || unsafe { kv::prefill(handle, aug_head) }.is_ok() {
                            syn_last = aug_last[0];
                        }
                    }
                }
                forget_done = true;
            } else {
                tracing::info!("LAYER-2 FORGET: forget intent but no memory matched (best overlap < 0.25)");
            }
        }
    }
    // ─────────── ADR-002 Decide→Execute SPINE (SP_SPINE=1) ───────────
    // The unified recall path made literal (papers/PPT-LAT-ADR-002): one immutable
    // LatentView → a priority-folded chain of Deciders → a discrete LatentDecision →
    // the Executor. This reproduces the LIVE one-config stack (L5 cosine recall →
    // attribute-gate zero-inference decline, QONLY-aware) in ~30 lines that used to be
    // a ~1500-line env-branch ladder. Default-off (SP_SPINE unset) ⇒ the inline
    // `else if` below runs byte-for-byte unchanged (the null floor).
    let spine_active = std::env::var("SP_SPINE").as_deref() == Ok("1");
    if spine_active && auto_recall && replay_dir.is_none() && !forget_done
        && app.recall_registry.is_some()
        && std::env::var("SP_RECALL_L5").as_deref() == Ok("1")
        && unsafe { kv::position(handle) } > 0
    {
        use sp_daemon::recall;
        let ruser = raw_user.clone().unwrap_or_default();
        let n_global = recall::NL / recall::PERIOD;
        let mut ql = vec![0.0f32; n_global * recall::G_NH * recall::HD];
        let read_ok = unsafe { kv::read_global_q(handle, last[0], &mut ql) }.is_ok();
        let l5_query = if read_ok { recall::l5_query_embed(&ql) } else { Vec::new() };
        let global_q = if read_ok { ql } else { Vec::new() };
        // Any graduated latent head joins the fold: load the W_c recall head if deployed
        // (SP_B3_WC), and build_pipeline runs it FIRST (a fired head short-circuits cosine).
        let wc = std::env::var("SP_B3_WC").ok().filter(|s| !s.is_empty())
            .and_then(|p| recall::load_wc(&p));
        let ns_snapshot: Vec<recall::Episode> =
            app.nightshift.read().unwrap().iter().cloned().collect();
        let registry = app.recall_registry.as_ref().unwrap();
        let framing = crate::spine::Delivery::from_env(
            std::env::var("SP_RECALL_L5_PROMPT").as_deref().unwrap_or("systemecho"));
        let view = crate::spine::LatentView {
            raw_user: &ruser,
            global_q,
            l5_query,
            registry,
            nightshift: &ns_snapshot,
            interrogative: recall::is_interrogative(&ruser),
            qonly: std::env::var("SP_RECALL_QONLY").as_deref() == Ok("1"),
            tau_l5: std::env::var("SP_RECALL_L5_TAU").ok().and_then(|s| s.parse().ok()).unwrap_or(0.30),
            tau_margin: std::env::var("SP_RECALL_L5_MARGIN").ok().and_then(|s| s.parse().ok()).unwrap_or(0.0),
            attr_tau: std::env::var("SP_RECALL_ATTR_TAU").ok().and_then(|s| s.parse().ok()).unwrap_or(0.5),
            attr_gate: std::env::var("SP_RECALL_ATTR_GATE").as_deref() == Ok("1"),
            framing,
        };
        let decision = crate::spine::decide(&view, &crate::spine::build_pipeline(wc, framing));
        let ctx = crate::spine::ExecCtx {
            handle,
            tokenizer: app.tokenizer.as_ref(),
            orig_msgs: orig_msgs.as_deref(),
            raw_user: &ruser,
            last_tok: last[0],
        };
        let outcome = unsafe { crate::spine::execute(decision, &ctx) };
        if outcome.recalled.is_some() { recalled = outcome.recalled; }
        if outcome.symbolic_decline.is_some() { symbolic_decline = outcome.symbolic_decline; }
        syn_last = outcome.syn_last;
    } else if auto_recall && replay_dir.is_none() && !forget_done {
        if let Some(registry) = app.recall_registry.as_ref() {
            let npos_q = unsafe { kv::position(handle) }; // = head.len() after prefill
            if npos_q > 0 {
                use sp_daemon::recall;
                // q·K relevance top-m (a few strongest matched positions). Tunable via env.
                let topm: usize = std::env::var("SP_B3_QK_TOPM").ok()
                    .and_then(|s| s.parse().ok()).unwrap_or(8);
                // The relevance threshold on the top-m-mean; default +inf so a first
                // telemetry run (env unset) NEVER fires — we read the matrix, then set
                // SP_B3_TAU_QK from the measured target/foreign gap.
                let tau_qk: f32 = std::env::var("SP_B3_TAU_QK").ok()
                    .and_then(|s| s.parse().ok()).unwrap_or(f32::INFINITY);
                // Read the live query's last-token global-layer Q (one non-committing
                // forward of the last prompt token; the cache is rolled back after).
                let n_global = recall::NL / recall::PERIOD;
                let mut qbuf = vec![0.0f32; n_global * recall::G_NH * recall::HD];
                // G-12B-SERVE §C (2026-07-03): the B3-v2 q·K scan below is a full
                // read_global_q forward + an O(episodes × npos) relevance loop + a giant
                // per-episode log line, run EVERY recall turn. When SP_B3_TAU_QK is unset it
                // defaults to +inf ⇒ `fire = score >= tau_qk` is ALWAYS false ⇒ the scan can
                // NEVER recall — pure dead telemetry. qbuf itself is only consumed downstream
                // by SP_B3_QDUMP / SP_INT2 / SP_B3_WC / SP_B3_JUDGE. In the one-config
                // faithful stack NONE of those are set, so on a 500-position conversation this
                // was ~half the recall-turn cost (turn-4 86.9s). When nothing can use it, skip
                // BOTH the forward and the scan loop; `best` stays None (the 2514 fire-arm
                // handles None cleanly) and the L5 stage (its OWN read_global_q at ~1524) is
                // untouched. This is byte-identical whenever any consumer IS set (scan runs).
                let qk_scan_needed = tau_qk.is_finite()
                    || std::env::var("SP_B3_QDUMP").is_ok()
                    || std::env::var("SP_INT2").ok().as_deref() == Some("1")
                    || std::env::var("SP_B3_WC").ok().filter(|s| !s.is_empty()).is_some()
                    || std::env::var("SP_B3_JUDGE").ok().filter(|s| !s.is_empty()).is_some();
                let read_q_res = if qk_scan_needed {
                    unsafe { kv::read_global_q(handle, last[0], &mut qbuf) }
                } else {
                    Ok(0)
                };
                match read_q_res {
                    Ok(_ng) => {
                        // B3-v3 dataset: SP_B3_QDUMP=<dir> persists THIS turn's last-token
                        // global-Q (the exact vector qk_relevance scores) so the offline
                        // contrastive trainer mines (Q, episode-K) pairs from the substrate.
                        // Additive; unset ⇒ no-op (null floor preserved). File q_<chat_id>.bin.
                        if let Ok(dir) = std::env::var("SP_B3_QDUMP") {
                            let qd = recall::G_NH * recall::HD;
                            let mut buf = Vec::with_capacity(8 + qbuf.len() * 4);
                            buf.extend_from_slice(&(n_global as u32).to_le_bytes());
                            buf.extend_from_slice(&(qd as u32).to_le_bytes());
                            for &x in &qbuf { buf.extend_from_slice(&x.to_le_bytes()); }
                            let _ = std::fs::create_dir_all(&dir);
                            let _ = std::fs::write(
                                std::path::Path::new(&dir).join(format!("q_{chat_id}.bin")), buf);
                        }
                        // Score every episode (max + top-m-mean), log the full matrix.
                        // G-12B-SERVE §C: skipped entirely when the result cannot be used
                        // (see qk_scan_needed above) — `best` stays None, which the 2514
                        // fire-arm treats as REJECT (no replay), the correct dead-scan outcome.
                        let mut best: Option<(usize, f32)> = None;
                        if qk_scan_needed {
                            let mut rows: Vec<String> = Vec::with_capacity(registry.len());
                            for (i, ep) in registry.iter().enumerate() {
                                let np = (ep.npos as usize).min(
                                    if ep.gk_ng > 0 { ep.gk.len() / (ep.gk_ng * recall::HD) } else { 0 });
                                let (mx, tm) = recall::qk_relevance(&qbuf, &ep.gk, ep.gk_ng, np, topm);
                                rows.push(format!("{}(max={:.3},topm={:.3})", ep.name, mx, tm));
                                let key = tm;
                                match best { Some((_, b)) if key <= b => {}, _ => best = Some((i, key)) }
                            }
                            tracing::info!("B3-v2 q·K relevance (npos_q={} topm={}): [{}]", npos_q, topm, rows.join(" "));
                        } else {
                            tracing::info!("B3-v2 q·K scan SKIPPED (TAU_QK=+inf, no QDUMP/INT2/WC/JUDGE consumer) — dead telemetry elided, one forward + registry scan saved this turn");
                        }
                        // ===== G-INT-2 STEP 2: STAGE-1 C2-HAMMING CULL (TELEMETRY-ONLY, SP_INT2=1) =====
                        // Compute the LIVE query's C2 sig from the prompt's global-K, build the bounded
                        // candidate set (curated registry ∪ most-recent-W nightshift episodes), rank by
                        // Hamming distance (R_BITS - agree) ascending, take top-K, and LOG the survivors.
                        // This branch is READ-ONLY on the cache (read_global_k is non-mutating) and does
                        // NOT inject / replay / decide NULL — it is the Stage-1 cull plumbing only. Default-off
                        // (env unset) = byte-identical null floor; leaves SP_B3_WC behavior intact when unset.
                        if std::env::var("SP_INT2").ok().as_deref() == Some("1") {
                            let w_horizon: usize = std::env::var("SP_INT2_W").ok()
                                .and_then(|s| s.parse().ok()).unwrap_or(20);   // KAIROS most-recent-W stub
                            let k_keep: usize = std::env::var("SP_INT2_K").ok()
                                .and_then(|s| s.parse().ok()).unwrap_or(20);   // cull budget
                            // (1) live query C2 sig from the prompt's global-K [n_global][npos_q][HD].
                            let mut qkbuf = vec![0.0f32; n_global * recall::HD * (npos_q as usize)];
                            match unsafe { kv::read_global_k(handle, &mut qkbuf, npos_q) } {
                                Ok(ngk) => {
                                    let qsig = app.recall_proj.signature(&qkbuf, ngk as usize, npos_q as usize);
                                    // (2) bounded candidate union = curated registry ∪ last-W nightshift.
                                    let ns_guard = app.nightshift.read().unwrap();
                                    let ns_len = ns_guard.len();
                                    let ns_skip = ns_len.saturating_sub(w_horizon);   // keep most-recent W live
                                    let n_static = registry.len();
                                    let cands: Vec<(bool, &recall::Episode)> = registry.iter()
                                        .map(|e| (false, e))
                                        .chain(ns_guard.iter().skip(ns_skip).map(|e| (true, e)))
                                        .collect();
                                    // (3) Hamming distance (R_BITS - agree); sort ascending; top-K.
                                    let mut scored: Vec<(usize, u32, bool, &recall::Episode)> = cands.iter()
                                        .enumerate()
                                        .map(|(i, &(live, ep))| {
                                            let d = recall::R_BITS as u32 - recall::agree(&qsig, &ep.sig);
                                            (i, d, live, ep)
                                        })
                                        .collect();
                                    scored.sort_by_key(|&(_, d, _, _)| d);
                                    let survivors: Vec<String> = scored.iter().take(k_keep)
                                        .map(|&(_, d, live, ep)| format!("{}{}:{}", ep.name, if live { "(L)" } else { "(C)" }, d))
                                        .collect();
                                    tracing::info!(
                                        "B-INT2 Stage-1 cull (W={} K={}): qsig={:016x}.. cull_pool={} (curated={} live={}) survivors=[{}]",
                                        w_horizon, k_keep, qsig[0], cands.len(), n_static, ns_len.min(w_horizon),
                                        survivors.join(", "));
                                }
                                Err(e) => tracing::warn!("B-INT2 Stage-1: read_global_k(npos_q={}) failed: {e} -- cull skipped", npos_q),
                            }
                            // ===== G-INT-2 STEP 3: STAGE-2 LIVE TEACHER-FORCED CAUSAL ABLATION GATE =====
                            // Promote the offline disposer (SP_B3_DISPOSER==2) to the LIVE Stage-2
                            // gatekeeper. Phase-A proved bounded-N makes the leaky C2 cull unnecessary,
                            // so we DO NOT Hamming-cull on the hot path: ablate ALL W bounded candidates
                            // directly. Candidate set = the last SP_INT2_W episodes of (curated registry
                            // ∪ nightshift) that have a RESOLVABLE secret (ep.secret + ep.tok on disk);
                            // live nightshift episodes lacking these are SKIPPED (the Phase-B extension).
                            // Per candidate (reusing the disposer FFI sequence EXACTLY, in the live turn's
                            // context — the query prompt is already in the cache at dpos==head.len()):
                            //   replay E -> teacher-force ITS ep.secret -> rewind(ng) -> ablate src rows
                            //   -> re-score -> rewind(ng+npos)  ⇒ collapse=ΣΔLL, dpos nets back to anchor.
                            // best=argmin collapse; collapse < SP_INT2_TAU (default -8.0) ⇒ ACCEPT (one
                            // replay into the live cache so the real decode attends the memory); else NULL
                            // (cache nets to anchor, byte-identical to a no-recall turn). SP_INT2 unset =
                            // this whole branch is skipped = byte-identical null floor.
                            {
                                let int2_tau: f32 = std::env::var("SP_INT2_TAU").ok()
                                    .and_then(|s| s.parse().ok()).unwrap_or(-8.0);
                                let lse = |z: &[f32]| -> f32 {
                                    let m = z.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                                    let mut s = 0.0f32; for &v in z { s += (v - m).exp(); } m + s.ln()
                                };
                                let anchor = unsafe { kv::position(handle) };
                                // SWA-ring Jmax hazard: assert npos+ng <= Jmax per candidate (full-cache
                                // served path is latent, but assert it — G-INT-3 ring turns it on).
                                let jmax: i32 = std::env::var("SP_G4_KV_JMAX").ok()
                                    .and_then(|s| s.parse().ok()).unwrap_or(64);
                                // Bounded candidate union = curated registry ∪ last-W nightshift, then
                                // keep only those with a resolvable secret. (We re-acquire the read lock;
                                // the Stage-1 cull above already dropped its guard.)
                                let ns_guard = app.nightshift.read().unwrap();
                                let ns_len = ns_guard.len();
                                let ns_skip = ns_len.saturating_sub(w_horizon);
                                let cands: Vec<(bool, &recall::Episode)> = registry.iter()
                                    .map(|e| (false, e))
                                    .chain(ns_guard.iter().skip(ns_skip).map(|e| (true, e)))
                                    .collect();
                                let mut best_abl: Option<(usize, f32, String)> = None;
                                let mut drows: Vec<String> = Vec::with_capacity(cands.len());
                                let mut n_probed = 0usize;
                                let mut n_skipped = 0usize;
                                let mut probe_steps = 0usize;
                                for (i, &(live, ep)) in cands.iter().enumerate() {
                                    if ep.npos <= 0 || ep.gk.is_empty() { n_skipped += 1; continue; }
                                    // Resolvable secret? ep.secret sidecar = teacher-force target;
                                    // ep.tok = ablation source rows. SKIP candidates lacking either
                                    // (live nightshift episodes without sidecars — Phase-B extension).
                                    let secret_raw = std::fs::read_to_string(
                                        std::path::Path::new(&ep.dir).join("ep.secret")).ok();
                                    let eptok: Vec<i32> = std::fs::read_to_string(
                                        std::path::Path::new(&ep.dir).join("ep.tok"))
                                        .ok().map(|s| s.lines().filter_map(|l| l.trim().parse::<i32>().ok()).collect())
                                        .unwrap_or_default();
                                    let secret_ids: Vec<i32> = match secret_raw {
                                        Some(s) => {
                                            let s = s.trim_end_matches(|c: char| c == '\n' || c == '\r').to_string();
                                            let mut v = app.tokenizer.encode(&s).unwrap_or_default();
                                            if v.first() == Some(&2) { v.remove(0); }
                                            v
                                        }
                                        None => Vec::new(),
                                    };
                                    if secret_ids.is_empty() || eptok.is_empty() {
                                        n_skipped += 1;
                                        if live {
                                            tracing::info!("B-INT2 Stage-2: SKIP '{}' (live, no resolvable secret/tok sidecar)", ep.name);
                                        }
                                        continue;
                                    }
                                    // Jmax assert (hazard §B.1): one candidate's uncommitted span = npos + ng.
                                    let ng_max = secret_ids.len() as i32;
                                    if ep.npos + ng_max > jmax {
                                        tracing::warn!("B-INT2 Stage-2: '{}' npos({})+ng({}) > Jmax({}) -- raising SP_G4_KV_JMAX advised; skipping to preserve pristineness", ep.name, ep.npos, ng_max, jmax);
                                        n_skipped += 1; continue;
                                    }
                                    n_probed += 1;
                                    // Leg 1: inject E, teacher-force the KNOWN secret, record lp_E.
                                    if unsafe { kv::replay(handle, &ep.dir, ep.npos, false) }.is_err() { continue; }
                                    let mut gen: Vec<i32> = Vec::new();
                                    let mut lpe: Vec<f32> = Vec::new();
                                    let mut tok = last[0];
                                    for &s in &secret_ids {
                                        if unsafe { kv::decode_step(handle, tok, logits) }.is_err() { break; }
                                        lpe.push(logits[s as usize] - lse(logits)); gen.push(s); tok = s;
                                    }
                                    let ng = gen.len();
                                    probe_steps += ng + ep.npos as usize;
                                    let _ = unsafe { kv::rewind(handle, ng as i32) };   // undo payload, keep E
                                    if ng == 0 {
                                        drows.push(format!("{}(collapse=nan)", ep.name));
                                        let _ = unsafe { kv::rewind(handle, ep.npos) };   // shear E -> back to anchor
                                        debug_assert_eq!(unsafe { kv::position(handle) }, anchor);
                                        continue;
                                    }
                                    // Source rows: episode positions whose token matches a secret token.
                                    let mut targets: Vec<i32> = Vec::new();
                                    let want: std::collections::HashSet<i32> = gen.iter().copied().collect();
                                    for (p, &t) in eptok.iter().enumerate() {
                                        if p >= ep.npos as usize { break; }
                                        if want.contains(&t) { targets.push(p as i32); }
                                    }
                                    if targets.len() > 12 { targets.truncate(12); }
                                    // Leg 2: ablate src rows, teacher-force the SAME secret, record lp_abl.
                                    let _ = unsafe { kv::ablate(handle, anchor, &targets) };
                                    let mut lpa: Vec<f32> = Vec::with_capacity(ng);
                                    let mut tok = last[0];
                                    for i2 in 0..ng {
                                        if unsafe { kv::decode_step(handle, tok, logits) }.is_err() { break; }
                                        lpa.push(logits[gen[i2] as usize] - lse(logits)); tok = gen[i2];
                                    }
                                    let _ = unsafe { kv::rewind(handle, lpa.len() as i32 + ep.npos) };   // clear payload + episode (restores ablated rows)
                                    // CRITICAL (Phase-A §B): net dpos delta per candidate == 0.
                                    debug_assert_eq!(unsafe { kv::position(handle) }, anchor,
                                        "B-INT2 Stage-2: candidate '{}' did not rewind to anchor", ep.name);
                                    let n = lpe.len().min(lpa.len());
                                    let mut collapse = 0.0f32;
                                    for j in 0..n { collapse += lpa[j] - lpe[j]; }
                                    drows.push(format!("{}{}(collapse={:.2},ntgt={})", ep.name, if live { "(L)" } else { "(C)" }, collapse, targets.len()));
                                    let better = match best_abl { None => true, Some((_, b, _)) => collapse < b };
                                    if better { best_abl = Some((i, collapse, ep.name.clone())); }
                                }
                                tracing::info!(
                                    "B-INT2 Stage-2 ablation (W={} probed={} skipped={} probe_steps={}): collapse=ΣΔLL (more-neg=load-bearing) [{}] TAU={:.3}",
                                    w_horizon, n_probed, n_skipped, probe_steps, drows.join(" "), int2_tau);
                                // The absolute thermodynamic gate: best=argmin collapse.
                                match best_abl {
                                    Some((idx, collapse, name)) if collapse < int2_tau => {
                                        let ep = cands[idx].1;
                                        // Final dpos MUST be at anchor before the accept replay.
                                        debug_assert_eq!(unsafe { kv::position(handle) }, anchor);
                                        match unsafe { kv::replay(handle, &ep.dir, ep.npos, false) } {
                                            Ok(_) => {
                                                tracing::info!("B-INT2 ACCEPT '{}' ΔLL={:.3} < τ={:.3} -> replay@M_target (live decode attends memory)", name, collapse, int2_tau);
                                                recalled = Some((name.clone(), ((-collapse).max(0.0) * 1000.0) as u32));
                                            }
                                            Err(e) => tracing::warn!("B-INT2 ACCEPT '{}': replay({}, {}) failed: {e} -- proceeding clean", name, ep.dir, ep.npos),
                                        }
                                        int2_decided = true;
                                    }
                                    Some((_, collapse, name)) => {
                                        tracing::info!("B-INT2 NULL (best '{}' ΔLL={:.3} >= τ={:.3}) -> clean prompt (null floor)", name, collapse, int2_tau);
                                        int2_decided = true;
                                    }
                                    None => {
                                        tracing::info!("B-INT2 NULL (no resolvable candidate in W={}) -> clean prompt (null floor)", w_horizon);
                                        int2_decided = true;
                                    }
                                }
                            }
                        }
                        // ===== B3-WC DEPLOY: learned W_c head, logsumexp-mean + (E+1) NULL argmax =====
                        // SP_B3_WC=<wc_deploy.bin> => the autonomous instance selector decides recall.
                        // Score every episode by wc lse-mean (the metric the head trained on, int16-exact),
                        // append the s0 NULL slot, argmax over [episodes, NULL]. Episode wins => replay it
                        // (M_target/SP_REPLAY_MTARGET=42 clamps injection mass); NULL wins => clean prompt.
                        // Default-off (env unset) = null floor; run WITHOUT SP_B3_DISPOSER/SP_B3_TAU_QK.
                        if let Some(wcp) = (if int2_decided { None } else { std::env::var("SP_B3_WC").ok().filter(|s| !s.is_empty()) }) {
                            match recall::load_wc(&wcp) {
                                Some(head) => {
                                    // B4 NIGHTSHIFT: score the static curated registry AND a snapshot of
                                    // the LIVE consolidated episodes (static first, then nightshift). The
                                    // (E+1)-NULL argmax is over [all_candidates, NULL=s0]; a live episode
                                    // can beat NULL + every curated episode and fire its own recall.
                                    let ns_guard = app.nightshift.read().unwrap();
                                    let cands: Vec<&recall::Episode> =
                                        registry.iter().chain(ns_guard.iter()).collect();
                                    let n_static = registry.len();
                                    let (mut bi, mut bv) = (usize::MAX, f32::NEG_INFINITY);
                                    let mut wrows: Vec<String> = Vec::with_capacity(cands.len());
                                    for (i, ep) in cands.iter().enumerate() {
                                        let np = (ep.npos as usize).min(
                                            if ep.gk_ng > 0 { ep.gk.len() / (ep.gk_ng * recall::HD) } else { 0 });
                                        let sc = recall::wc_score(&qbuf, &ep.gk, ep.gk_ng, np, &head);
                                        wrows.push(format!("{}={:.3}", ep.name, sc));
                                        if sc > bv { bv = sc; bi = i; }
                                    }
                                    tracing::info!("B3-WC lse-mean (E+1)-argmax s0={:.3} ({} curated + {} live): [{}]",
                                        head.s0, n_static, cands.len() - n_static, wrows.join(" "));
                                    if bi == usize::MAX || !(bv > head.s0) {
                                        tracing::info!("B3-WC: NULL wins (best={:.3} <= s0={:.3}) -> REJECT (clean prompt)", bv, head.s0);
                                    } else {
                                        let ep = cands[bi];
                                        if let Some(ref toks) = ep.tokens {
                                            // B4 live episode: recall by re-injecting its raw tokens through the
                                            // B5 seam (bit-identical to prefill) — no ep.k/ep.v files on disk.
                                            // G-INT-2-FIX: attenuated inject (M_target=42) so the
                                            // recalled memory BINDS instead of hijacking synthesis.
                                            tracing::info!("B3-WC: RECALL '{}' (LIVE/nightshift) score={:.3} > s0={:.3} -> inject_tokens_atten(n={})",
                                                ep.name, bv, head.s0, toks.len());
                                            if let Err(e) = unsafe { kv::inject_tokens_atten(handle, toks) } {
                                                tracing::warn!("B3-WC: inject_tokens_atten('{}', n={}) failed: {e} -- clean prompt", ep.name, toks.len());
                                            } else {
                                                recalled = Some((ep.name.clone(), (bv.max(0.0) * 1000.0) as u32));
                                            }
                                        } else if std::env::var("SP_B3_WC_TEXT").as_deref() == Ok("1") && raw_user.is_some() {
                                            // F2 FAITHFULNESS: TEXT-IN-CONTEXT delivery for fact recall. F1b.1 proved
                                            // pure-KV replay yields 0% obedience on fact-conflict (the attenuated K/V can't
                                            // override a strong parametric prior), while F1 proved an in-context fact gets
                                            // 100% obedience. So on a confident W_c match, deliver the episode TEXT as
                                            // authoritative context and rebuild the cache from the augmented prompt (the
                                            // proven FORGET-synthesis machinery), instead of kv::replay. Default-off (env
                                            // unset) keeps the pure-KV replay null floor (valid for novel needles).
                                            let ruser = raw_user.as_deref().unwrap_or("");
                                            let aug = format!(
                                                "Context (authoritative, from your memory): {}\n\nUsing the context above, answer faithfully even if it differs from what you already know:\n{}",
                                                ep.text, ruser);
                                            let aug_msgs = vec![Message { role: "user".to_string(), content: aug }];
                                            match app.tokenizer.apply_template_ids(&aug_msgs) {
                                                Ok(aug_toks) if aug_toks.len() >= 2 => {
                                                    let _ = unsafe { kv::reset_cold(handle) };
                                                    let (aug_head, aug_last) = aug_toks.split_at(aug_toks.len() - 1);
                                                    if aug_head.is_empty() || unsafe { kv::prefill(handle, aug_head) }.is_ok() {
                                                        syn_last = aug_last[0];
                                                        recalled = Some((ep.name.clone(), (bv.max(0.0) * 1000.0) as u32));
                                                        tracing::info!("B3-WC: RECALL '{}' (curated) score={:.3} > s0={:.3} -> TEXT-IN-CONTEXT synthesis (F2)", ep.name, bv, head.s0);
                                                    } else {
                                                        tracing::warn!("B3-WC TEXT: prefill(aug) failed -- clean prompt");
                                                    }
                                                }
                                                _ => tracing::warn!("B3-WC TEXT: apply_template_ids(aug) failed -- clean prompt"),
                                            }
                                        } else {
                                            // Curated episode: the existing disk-replay path (M_target unchanged). Pure-KV
                                            // null floor; F1b.1: 0% obey on fact-conflict, valid only for novel needles.
                                            tracing::info!("B3-WC: RECALL '{}' (curated) score={:.3} > s0={:.3} -> replay@M_target", ep.name, bv, head.s0);
                                            if let Err(e) = unsafe { kv::replay(handle, &ep.dir, ep.npos, false) } {
                                                tracing::warn!("B3-WC: replay({}, {}) failed: {e} -- clean prompt", ep.dir, ep.npos);
                                            } else {
                                                recalled = Some((ep.name.clone(), (bv.max(0.0) * 1000.0) as u32));
                                            }
                                        }
                                    }
                                }
                                None => tracing::warn!("B3-WC: SP_B3_WC set but load_wc({}) failed -- skipping", wcp),
                            }
                        }
                        // ===== F2b: TOKEN-OVERLAP (Jaccard) selection for NATURAL FACTS =====
                        // W_c is geometric — great for high-entropy novel needles, blind to mutually-
                        // similar natural-language facts (F2: it picked query-INDEPENDENT episodes, 0/15).
                        // Token-overlap of the QUERY vs each episode TEXT discriminates natural facts
                        // (recall::token_overlap, the production Jaccard verifier); on a confident overlap,
                        // deliver the episode TEXT in-context (the F1=100% path), NOT pure-KV replay.
                        // Default-off (SP_RECALL_JACCARD unset) = null floor; runs only if no prior stage decided.
                        if !int2_decided && recalled.is_none()
                            && std::env::var("SP_RECALL_JACCARD").as_deref() == Ok("1")
                        {
                            if let Some(ruser) = raw_user.as_ref().map(|s| s.trim().to_string()).filter(|s| !s.is_empty()) {
                                let tau_ov: f32 = std::env::var("SP_RECALL_JACCARD_TAU").ok()
                                    .and_then(|s| s.parse().ok()).unwrap_or(0.15);
                                let q = ruser.to_lowercase();
                                let (mut bov, mut bname, mut btext) = (0.0f32, String::new(), String::new());
                                let (mut bgk, mut bgkng): (Vec<f32>, usize) = (Vec::new(), 0);
                                {
                                    let ns_guard = app.nightshift.read().unwrap();
                                    for ep in registry.iter().chain(ns_guard.iter()) {
                                        let ov = recall::token_overlap(&q, &ep.text);
                                        if ov > bov { bov = ov; bname = ep.name.clone(); btext = ep.text.clone(); bgk = ep.gk.clone(); bgkng = ep.gk_ng; }
                                    }
                                }
                                if bov >= tau_ov && !btext.is_empty() {
                                    let aug = format!("Context (authoritative, current): {}\n\n{}", btext, ruser);
                                    let aug_msgs = vec![
                                        Message { role: "system".to_string(), content: "You are Shannon-Prime, a local AI with a real working memory. Keep replies short. Use facts you were given faithfully; if you don't know, say so.".to_string() },
                                        Message { role: "user".to_string(), content: aug },
                                    ];
                                    match app.tokenizer.apply_template_ids(&aug_msgs) {
                                        Ok(aug_toks) if aug_toks.len() >= 2 => {
                                            let _ = unsafe { kv::reset_cold(handle) };
                                            let (aug_head, aug_last) = aug_toks.split_at(aug_toks.len() - 1);
                                            if aug_head.is_empty() || unsafe { kv::prefill(handle, aug_head) }.is_ok() {
                                                syn_last = aug_last[0];
                                                recalled = Some((bname.clone(), (bov * 1000.0) as u32));
                                                tracing::info!("RECALL-JACCARD: '{}' overlap={:.3} >= tau={:.3} -> TEXT-IN-CONTEXT (F2b)", bname, bov, tau_ov);
                                                // LN-F3a selector-data label: pair q_<chat_id>.bin (query global-Q, the
                                                // SP_B3_QDUMP rail) with the Jaccard-selected episode. Capture-only.
                                                if let Ok(qd) = std::env::var("SP_B3_QDUMP") {
                                                    let _ = std::fs::write(
                                                        std::path::Path::new(&qd).join(format!("lbl_{chat_id}.txt")),
                                                        format!("{}\t{:.4}\n", bname, bov));
                                                    // LN-1 clean K-dump: the selected episode's IN-MEMORY global-K
                                                    // [gk_ng][npos][HD] — bypasses the opaque on-disk ep.k layout.
                                                    // Mirrors q_<chat_id>.bin: <u32 ng><u32 npos><f32 gk>.
                                                    if bgkng > 0 && !bgk.is_empty() {
                                                        let npos_k = bgk.len() / (bgkng * recall::HD);
                                                        let mut kb = Vec::with_capacity(8 + bgk.len() * 4);
                                                        kb.extend_from_slice(&(bgkng as u32).to_le_bytes());
                                                        kb.extend_from_slice(&(npos_k as u32).to_le_bytes());
                                                        for &x in &bgk { kb.extend_from_slice(&x.to_le_bytes()); }
                                                        let _ = std::fs::write(
                                                            std::path::Path::new(&qd).join(format!("k_{chat_id}.bin")), kb);
                                                    }
                                                }
                                            } else { tracing::warn!("RECALL-JACCARD: prefill(aug) failed -- clean prompt"); }
                                        }
                                        _ => tracing::warn!("RECALL-JACCARD: apply_template_ids(aug) failed -- clean prompt"),
                                    }
                                } else {
                                    tracing::info!("RECALL-JACCARD: best='{}' overlap={:.3} < tau={:.3} -> no recall (clean prompt)", bname, bov, tau_ov);
                                }
                            }
                        }
                        // ===== L5 RECALL (SP_RECALL_L5) — query-to-query paraphrase recall =====
                        // The fact signal is layer-localized in global layer 5 (G-REP-LAYER-L5:
                        // L5 exact->paraphrase recall@1 = 85.2% / all-layer-avg 11.5%; L5-cosine
                        // query-key = 100% exact / 88.5% paraphrase vs Jaccard 100%/8.2%). Match the
                        // live query's L5 embedding (l5_query_embed of read_global_q, mean-heads +
                        // L2-norm) against each episode's stored L5 query-key (ep.l5) by cosine; on a
                        // confident match deliver the episode TEXT in-context (the same F2b delivery
                        // as the Jaccard branch). GATED after Jaccard (skips if a prior stage already
                        // recalled). Default-off (SP_RECALL_L5 unset) = byte-identical null floor.
                        if !int2_decided && recalled.is_none()
                            && std::env::var("SP_RECALL_L5").as_deref() == Ok("1")
                        {
                            if let Some(ruser) = raw_user.as_ref().map(|s| s.trim().to_string()).filter(|s| !s.is_empty()) {
                                // SP_RECALL_QONLY (G-ONECONFIG-LIVE run-1 lever, RUNBOOK-ONE-CONFIG §8):
                                // conversational STATEMENTS skip the L5 stage entirely — the in-registry
                                // cosine background is ≥0.9, so a non-query turn otherwise gets an
                                // irrelevant fact injected. Deterministic, no forward; unset = null floor.
                                let qonly_skip = std::env::var("SP_RECALL_QONLY").as_deref() == Ok("1")
                                    && !recall::is_interrogative(&ruser);
                                if qonly_skip {
                                    tracing::info!("RECALL-L5: QONLY-SKIP (non-interrogative turn) -> clean prompt");
                                }
                                let tau_l5: f32 = std::env::var("SP_RECALL_L5_TAU").ok()
                                    .and_then(|s| s.parse().ok()).unwrap_or(0.30);
                                // SP_RECALL_L5_MARGIN (run-1 lever): absolute cosine cannot separate the
                                // right episode from the in-registry background (both ≥0.9); the top1−top2
                                // GAP can. Gates DELIVERY only — the attr-gate decline still fires (the
                                // SNE shield must not be starved). 0.0/unset = off = null floor.
                                let tau_margin: f32 = std::env::var("SP_RECALL_L5_MARGIN").ok()
                                    .and_then(|s| s.parse().ok()).unwrap_or(0.0);
                                let n_global = recall::NL / recall::PERIOD;
                                let mut ql = vec![0.0f32; n_global * recall::G_NH * recall::HD];
                                let qk5 = if !qonly_skip && unsafe { kv::read_global_q(handle, last[0], &mut ql) }.is_ok() {
                                    recall::l5_query_embed(&ql)
                                } else { Vec::new() };
                                if qonly_skip {
                                    // handled above — fall through with no recall
                                } else if qk5.is_empty() {
                                    tracing::warn!("RECALL-L5: read_global_q/l5_embed unavailable -- clean prompt");
                                } else {
                                    let mut scored: Vec<(f32, String, String)> = Vec::new();
                                    {
                                        let ns_guard = app.nightshift.read().unwrap();
                                        for ep in registry.iter().chain(ns_guard.iter()) {
                                            if ep.l5key.len() != recall::HD { continue; }
                                            scored.push((recall::cos512(&qk5, &ep.l5key), ep.name.clone(), ep.text.clone()));
                                        }
                                    }
                                    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                                    let (mut bcos, mut bname, mut btext) = scored.first()
                                        .map(|t| (t.0, t.1.clone(), t.2.clone()))
                                        .unwrap_or((f32::NEG_INFINITY, String::new(), String::new()));
                                    let (bcos2, bname2) = scored.get(1).map(|t| (t.0, t.1.clone()))
                                        .unwrap_or((f32::NEG_INFINITY, String::new()));
                                    // MARGIN TELEMETRY (always-on when L5 runs; telemetry-then-pin):
                                    // single-episode registries have no top2 -> margin = +inf semantics.
                                    let margin = if bcos2.is_finite() { bcos - bcos2 } else { f32::INFINITY };
                                    tracing::info!("RECALL-L5-MARGIN: top1='{}' cos={:.4} top2='{}' cos2={:.4} margin={:.4}",
                                        bname, bcos, bname2, bcos2, margin);
                                    // ===== MARGIN-GATED TOP-3 JUDGE RERANK (SP_RECALL_L5_RERANK=<eps>) =====
                                    // G-SEL-OFFLINE: the 7 selector misses are same-template periphrasis
                                    // cross-picks with tiny top1-top2 margins (all <0.013; eps=0.015 catches
                                    // 7/7 while firing on only 16/54 correct picks); correct-in-top-3 =
                                    // 59/61. The distinguishing signal is SEMANTIC (cheap levers convicted
                                    // inert), so on an ambiguous margin the 12B READS the top-3 fact TEXTS
                                    // and picks A/B/C in ONE constrained greedy side-pass — the judge
                                    // DECIDES, the recite path DELIVERS (ADR-002; the 61160e9 pattern).
                                    // Fail-open: any forward/parse failure keeps the L5 top-1. The recite/
                                    // decline below resets the cache anyway, so no restore pass is needed.
                                    // Unset/0 = null floor.
                                    // ===== MARGIN-GATED QUERY CANONICALIZATION (SP_RECALL_L5_CANON=<eps>) =====
                                    // G-SEL-CANON: the judge-rerank PARKED (G-SEL-RERANK-61: fixes 5 but
                                    // breaks 4 — in-family A/B/C adjudication is ~unreliable). The sharper
                                    // decide-step: on an ambiguous margin, ONE micro-forward rewrites the
                                    // query's periphrases to PROPER NAMES ("boot-shaped Mediterranean
                                    // country" -> "Italy") — decide-in-clean-text (ADR-002), a task the 12B
                                    // is far more reliable at than slot-picking. The surfaced name appears
                                    // VERBATIM in exactly the right fact, so a deterministic salient-overlap
                                    // count over the L5 top-5 becomes decisive (raw-query Jaccard was inert,
                                    // G-SEL-OFFLINE). Switch rule is CONSERVATIVE: only on strictly-greater
                                    // overlap than the current top1 (ties keep L5 order; fail-open on any
                                    // forward/parse failure). Unset/0 = null floor.
                                    let canon_eps: f32 = std::env::var("SP_RECALL_L5_CANON").ok()
                                        .and_then(|s| s.parse().ok()).unwrap_or(0.0);
                                    if canon_eps > 0.0 && margin < canon_eps && bcos >= tau_l5 && scored.len() >= 2 {
                                        // NAME-THE-SUBJECT framing (run-2/3 lesson: "rewrite the question"
                                        // makes the 12B ANSWER it instead — even one-shot. Naming the
                                        // DESCRIBED subject is the simpler task and its output is exactly
                                        // the overlap token the switch rule needs). Exemplar OUT-OF-CORPUS.
                                        let ccontent = format!(
                                            "A question describes something without naming it. Name the described thing, place, or person — do NOT answer the question.\nExample: \"What is the tallest building in the city that never sleeps?\" describes New York.\nQuestion: \"{ruser}\" describes:");
                                        let cmsgs = vec![Message { role: "user".to_string(), content: ccontent }];
                                        match app.tokenizer.apply_template_ids(&cmsgs) {
                                            Ok(ctoks) if ctoks.len() >= 2 => {
                                                let _ = unsafe { kv::reset(handle) };
                                                let (chead, clast) = ctoks.split_at(ctoks.len() - 1);
                                                let mut ok = chead.is_empty() || unsafe { kv::prefill(handle, chead) }.is_ok();
                                                let mut canon = String::new();
                                                if ok {
                                                    let mut tok = clast[0];
                                                    let turn_stops = app.tokenizer.turn_stop_ids();
                                                    // raw-argmax side-pass MUST mask the served suppress set
                                                    // (gemma <image|>/<audio|> soft tokens + control markers)
                                                    // — the exact placeholder-spam bug the judge verify path
                                                    // already root-caused. Same cure here.
                                                    let suppress = app.tokenizer.suppress_token_ids();
                                                    for _ in 0..32 {
                                                        if unsafe { kv::decode_step(handle, tok, logits) }.is_err() { ok = false; break; }
                                                        let mut best_t = 0usize;
                                                        let mut best_v = f32::NEG_INFINITY;
                                                        for (t, &v) in logits.iter().enumerate() {
                                                            if suppress.contains(&(t as i32)) { continue; }
                                                            if v > best_v { best_v = v; best_t = t; }
                                                        }
                                                        tok = best_t as i32;
                                                        if tok == 1 || turn_stops.contains(&tok) { break; }
                                                        canon.push_str(&String::from_utf8_lossy(app.tokenizer.decode_token(tok)));
                                                        if canon.trim_start().contains('\n') { break; }
                                                    }
                                                }
                                                let canon = canon.trim().lines().next().unwrap_or("").trim().to_string();
                                                if ok && !canon.is_empty() {
                                                    let k5 = scored.len().min(5);
                                                    let ov: Vec<usize> = (0..k5).map(|i| recall::canon_overlap(&canon, &scored[i].2)).collect();
                                                    let (mut bi, mut bov) = (0usize, ov[0]);
                                                    for i in 1..k5 { if ov[i] > bov { bi = i; bov = ov[i]; } }
                                                    if bi != 0 && bov > ov[0] {
                                                        tracing::info!("RECALL-L5 CANON: margin={:.4} < eps={:.4}; canon={:?} overlap {}>{} -> switch '{}' (over top1 '{}')",
                                                            margin, canon_eps, canon, bov, ov[0], scored[bi].1, bname);
                                                        bcos = scored[bi].0; bname = scored[bi].1.clone(); btext = scored[bi].2.clone();
                                                    } else {
                                                        tracing::info!("RECALL-L5 CANON: margin={:.4} < eps={:.4}; canon={:?} keeps top1 '{}' (ov={:?})",
                                                            margin, canon_eps, canon, bname, ov);
                                                    }
                                                } else {
                                                    tracing::warn!("RECALL-L5 CANON: rewrite unusable (ok={} canon={:?}) -- keeping L5 top1 (fail-open)", ok, canon);
                                                }
                                            }
                                            _ => tracing::warn!("RECALL-L5 CANON: apply_template_ids failed -- keeping L5 top1 (fail-open)"),
                                        }
                                    }
                                    let rerank_eps: f32 = std::env::var("SP_RECALL_L5_RERANK").ok()
                                        .and_then(|s| s.parse().ok()).unwrap_or(0.0);
                                    if rerank_eps > 0.0 && margin < rerank_eps && bcos >= tau_l5 && scored.len() >= 2 {
                                        let k = scored.len().min(3);
                                        let letters = ["A", "B", "C"];
                                        // ANTI-POSITION-BIAS (FINDINGS-#3, relearned by G-SEL-RERANK-61 run-1:
                                        // fixed cos-order slots made the judge CREATE adjacent cross-picks —
                                        // same-family capitals swapped wholesale). Deterministic per-query
                                        // Fisher-Yates over the k candidates (fnv1a of the query seeds a
                                        // xorshift), letters stay A/B/C; the pick maps back through `perm`.
                                        let mut seed: u64 = 0xcbf29ce484222325;
                                        for b in ruser.as_bytes() { seed ^= *b as u64; seed = seed.wrapping_mul(0x100000001b3); }
                                        if seed == 0 { seed = 0x9e3779b97f4a7c15; }
                                        let mut rng = move || { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; seed };
                                        let mut perm: Vec<usize> = (0..k).collect();
                                        for i in (1..k).rev() {
                                            let j = (rng() % (i as u64 + 1)) as usize;
                                            perm.swap(i, j);
                                        }
                                        let mut entries = String::new();
                                        for slot in 0..k { entries.push_str(&format!("{}) {}\n", letters[slot], scored[perm[slot]].2)); }
                                        let jcontent = format!(
                                            "You are a memory index. Exactly one fact below directly answers the question.\n\n{entries}\nQUESTION: {ruser}\n\nReply with only the single letter (A, B or C) of the fact that answers the question:");
                                        let jmsgs = vec![Message { role: "user".to_string(), content: jcontent }];
                                        match app.tokenizer.apply_template_ids(&jmsgs) {
                                            Ok(jtoks) if jtoks.len() >= 2 => {
                                                let _ = unsafe { kv::reset(handle) };
                                                let (jhead, jlast) = jtoks.split_at(jtoks.len() - 1);
                                                let mut ok = jhead.is_empty() || unsafe { kv::prefill(handle, jhead) }.is_ok();
                                                let mut reply = String::new();
                                                if ok {
                                                    let mut tok = jlast[0];
                                                    let turn_stops = app.tokenizer.turn_stop_ids();
                                                    let suppress = app.tokenizer.suppress_token_ids();
                                                    for _ in 0..6 {
                                                        if unsafe { kv::decode_step(handle, tok, logits) }.is_err() { ok = false; break; }
                                                        let mut best_t = 0usize;
                                                        let mut best_v = f32::NEG_INFINITY;
                                                        for (t, &v) in logits.iter().enumerate() {
                                                            if suppress.contains(&(t as i32)) { continue; }
                                                            if v > best_v { best_v = v; best_t = t; }
                                                        }
                                                        tok = best_t as i32;
                                                        if tok == 1 || turn_stops.contains(&tok) { break; }
                                                        reply.push_str(&String::from_utf8_lossy(app.tokenizer.decode_token(tok)));
                                                        if !reply.trim().is_empty() { break; } // one letter is all we need
                                                    }
                                                }
                                                let pick = reply.trim().to_uppercase().chars()
                                                    .find(|c| matches!(c, 'A' | 'B' | 'C'))
                                                    .map(|c| (c as u8 - b'A') as usize)
                                                    .filter(|&slot| slot < k)
                                                    .map(|slot| perm[slot]);
                                                match (ok, pick) {
                                                    (true, Some(i)) => {
                                                        if i != 0 {
                                                            tracing::info!("RECALL-L5 RERANK: margin={:.4} < eps={:.4}; judge pick '{}' (over top1 '{}')",
                                                                margin, rerank_eps, scored[i].1, bname);
                                                            bcos = scored[i].0; bname = scored[i].1.clone(); btext = scored[i].2.clone();
                                                        } else {
                                                            tracing::info!("RECALL-L5 RERANK: margin={:.4} < eps={:.4}; judge confirms top1 '{}'", margin, rerank_eps, bname);
                                                        }
                                                    }
                                                    _ => tracing::warn!("RECALL-L5 RERANK: judge unusable (ok={} reply={:?}) -- keeping L5 top1 (fail-open)", ok, reply),
                                                }
                                            }
                                            _ => tracing::warn!("RECALL-L5 RERANK: apply_template_ids failed -- keeping L5 top1 (fail-open)"),
                                        }
                                    }
                                    if bcos >= tau_l5 && !btext.is_empty() {
                                        // ATTR-GROUNDING (SNE crucible fix): on zero-prior data the model
                                        // CONFABULATES a plausible wrong value when asked an attribute the
                                        // delivered fact does not state (80% on the SNE crucible; 5% leak the
                                        // fact's own value). Two composable, default-off levers:
                                        //  - SP_RECALL_STRICT: closed-book delivery — "answer ONLY from the fact;
                                        //    if it doesn't state what's asked, decline exactly." The MODEL grounds
                                        //    with its own semantics (paraphrase-safe: a same-attribute paraphrase
                                        //    still recites; a different-attribute query declines).
                                        //  - SP_RECALL_ATTR_GATE: deterministic lexical pre-check; if the query's
                                        //    salient words are ABSENT from the fact (ratio>=SP_RECALL_ATTR_TAU),
                                        //    FORCE the strict decline framing regardless of the model. Hard
                                        //    fallback (not paraphrase-aware). Both unset = the proven recite path
                                        //    (byte-identical null floor).
                                        let strict = std::env::var("SP_RECALL_STRICT").as_deref() == Ok("1");
                                        let attr_gate = std::env::var("SP_RECALL_ATTR_GATE").as_deref() == Ok("1");
                                        let attr_tau: f32 = std::env::var("SP_RECALL_ATTR_TAU").ok()
                                            .and_then(|s| s.parse().ok()).unwrap_or(0.5);
                                        let absent = if attr_gate { recall::attr_absent_ratio(&ruser, &btext) } else { 0.0 };
                                        // PARAPHRASE GUARD: only decline on attribute-absence when the query
                                        // shares a high-entropy verbatim token with the fact (a private-entity
                                        // ID/code). General-knowledge/paraphrase queries have no such token, so
                                        // the gate stays off for them (recall preserved) — this is what makes
                                        // the deterministic gate globally default-on-safe, not regime-specific.
                                        let force_decline = attr_gate && absent >= attr_tau
                                            && recall::query_has_entity_token(&ruser);
                                        if force_decline {
                                            // ZERO-INFERENCE symbolic decline (ADR-002 Tier-2 executor). The
                                            // fact exists but does NOT state the queried attribute. Set a
                                            // deterministic decline string and SKIP the synthesis forward
                                            // entirely (consumed at the synthesis seam below, before
                                            // decode_step(syn_last)). No gemma4 decode => confabulation/leak
                                            // is mathematically impossible + the turn resolves in microseconds.
                                            symbolic_decline = Some(
                                                "I have a record for that entity, but it does not include that specific detail.".to_string());
                                            recalled = Some((bname.clone(), (bcos * 1000.0) as u32));
                                            tracing::info!("RECALL-L5: '{}' cos={:.3} absent={:.2} -> ATTR-DECLINE (zero-inference symbolic, no forward)",
                                                bname, bcos, absent);
                                        } else if tau_margin > 0.0 && margin < tau_margin {
                                            // MARGIN GATE (delivery only; attr-decline above is NOT starved):
                                            // an ambiguous top1 (no clear gap over top2) is background, not a
                                            // memory hit — deliver nothing, run the clean prompt.
                                            tracing::info!("RECALL-L5: MARGIN-SKIP top1='{}' cos={:.3} margin={:.4} < tau_m={:.4} -> no recall (clean prompt)",
                                                bname, bcos, margin, tau_margin);
                                        } else {
                                            // recite (proven 86.89% path); SP_RECALL_STRICT = closed-book
                                            // model-grounded framing (dead lever, kept default-off).
                                            let aug_msgs = if strict {
                                                vec![
                                                    Message { role: "system".to_string(), content: "You are Shannon-Prime, a local AI with a real working memory. Answer ONLY using the fact on record below. If the fact does not state what the question asks, reply EXACTLY: \"I do not have that information.\" Do not guess, infer, or invent any detail that is not written in the fact.".to_string() },
                                                    Message { role: "user".to_string(), content: format!("Fact on record: {}\n\nQuestion: {}", btext, ruser) },
                                                ]
                                            } else if std::env::var("SP_RECALL_L5_PROMPT").as_deref() == Ok("sandwich") {
                                                // DELIVERY SWEEP (2026-07-02, RUNBOOK §11): instruction AFTER the
                                                // question (recency) + explicit override authority.
                                                vec![
                                                    Message { role: "system".to_string(), content: "You are Shannon-Prime, a local AI with a real working memory. Keep replies short. Use facts you were given faithfully; if you don't know, say so.".to_string() },
                                                    Message { role: "user".to_string(), content: format!("Context (authoritative, from your memory): {}\n\n{}\n\n(Answer using ONLY the context above; it overrides your prior knowledge.)", btext, ruser) },
                                                ]
                                            } else if std::env::var("SP_RECALL_L5_PROMPT").as_deref() == Ok("factecho") {
                                                // DELIVERY SWEEP: prime the answer to copy from the fact.
                                                vec![
                                                    Message { role: "system".to_string(), content: "You are Shannon-Prime, a local AI with a real working memory. Your memory record is the ground truth for this conversation, even where it differs from general knowledge. Keep replies short.".to_string() },
                                                    Message { role: "user".to_string(), content: format!("Fact on record: {}\n\nQuestion: {}\n\nAnswer using the fact on record:", btext, ruser) },
                                                ]
                                            } else if std::env::var("SP_RECALL_L5_PROMPT").as_deref() == Ok("system") {
                                                // DELIVERY SWEEP: fact delivered as SYSTEM authority, clean user turn.
                                                vec![
                                                    Message { role: "system".to_string(), content: format!("You are Shannon-Prime, a local AI with a real working memory. Fact on record (authoritative for this conversation, overrides prior knowledge): {}\nAnswer from this fact; keep replies short.", btext) },
                                                    Message { role: "user".to_string(), content: ruser.clone() },
                                                ]
                                            } else if std::env::var("SP_RECALL_L5_PROMPT").as_deref() == Ok("systemecho") {
                                                // DELIVERY SWEEP round 2 winner: system authority + copy-priming
                                                // (full-61: 88.52% OBEY, 0 LEAK — every correctly-selected episode
                                                // obeyed; misses = selection cross-picks). MULTI-TURN FIX: preserve
                                                // the conversation (G-ONECONFIG-LIVE C-phase root cause: delivery
                                                // used to keep only the last user message, so turn-2 questions about
                                                // turn-1 content were unanswerable on recall turns).
                                                let mut v = vec![
                                                    Message { role: "system".to_string(), content: format!("You are Shannon-Prime, a local AI with a real working memory. Fact on record (authoritative for this conversation, overrides prior knowledge): {}\nEvery answer must repeat the relevant part of the fact on record verbatim. Keep replies short.", btext) },
                                                ];
                                                match orig_msgs.as_ref() {
                                                    Some(ms) if ms.iter().any(|m| m.role == "user") => {
                                                        for m in ms.iter().filter(|m| m.role != "system") { v.push(m.clone()); }
                                                        if let Some(last_u) = v.iter_mut().rev().find(|m| m.role == "user") {
                                                            last_u.content = format!("{}\n\nAnswer using the fact on record:", ruser);
                                                        }
                                                    }
                                                    _ => v.push(Message { role: "user".to_string(), content: format!("{}\n\nAnswer using the fact on record:", ruser) }),
                                                }
                                                v
                                            } else if std::env::var("SP_RECALL_L5_PROMPT").as_deref() == Ok("scaled") {
                                                // SCALED delivery wording (2026-07-02): the plain recite wording's
                                                // paraphrase obedience proved FP/build-FRAGILE (the 86.89% receipt
                                                // is NOT reproducible on the current stack: 40.98% via the receipt's
                                                // own harness, 3 independent rebuilds incl. exact a14fee4 src+lock).
                                                // This explicit-override wording carries its own receipt: OBEY 52/61
                                                // = 85.2% (_g_faithful_recall_scaled.json, seam=scaled) — the
                                                // instruction does the work, not a thin float margin.
                                                vec![
                                                    Message { role: "system".to_string(), content: "You are Shannon-Prime, a local AI with a real working memory. Keep replies short. Use facts you were given faithfully; if you don't know, say so.".to_string() },
                                                    Message { role: "user".to_string(), content: format!("Context (authoritative, from your memory): {}\n\nUsing the context above, answer faithfully even if it differs from what you already know:\n{}", btext, ruser) },
                                                ]
                                            } else {
                                                vec![
                                                    Message { role: "system".to_string(), content: "You are Shannon-Prime, a local AI with a real working memory. Keep replies short. Use facts you were given faithfully; if you don't know, say so.".to_string() },
                                                    Message { role: "user".to_string(), content: format!("Context (authoritative, current): {}\n\n{}", btext, ruser) },
                                                ]
                                            };
                                            match app.tokenizer.apply_template_ids(&aug_msgs) {
                                                Ok(aug_toks) if aug_toks.len() >= 2 => {
                                                    let _ = unsafe { kv::reset_cold(handle) };
                                                    let (aug_head, aug_last) = aug_toks.split_at(aug_toks.len() - 1);
                                                    if aug_head.is_empty() || unsafe { kv::prefill(handle, aug_head) }.is_ok() {
                                                        syn_last = aug_last[0];
                                                        recalled = Some((bname.clone(), (bcos * 1000.0) as u32));
                                                        tracing::info!("RECALL-L5: '{}' cos={:.3} >= tau={:.3} absent={:.2} mode={} -> TEXT-IN-CONTEXT",
                                                            bname, bcos, tau_l5, absent, if strict {"STRICT"} else {"recite"});
                                                    } else { tracing::warn!("RECALL-L5: prefill(aug) failed -- clean prompt"); }
                                                }
                                                _ => tracing::warn!("RECALL-L5: apply_template_ids(aug) failed -- clean prompt"),
                                            }
                                        }
                                    } else {
                                        tracing::info!("RECALL-L5: best='{}' cos={:.3} < tau={:.3} -> no recall (clean prompt)", bname, bcos, tau_l5);
                                    }
                                }
                            }
                        }
                        // ===== B3-JUDGE: 12B GENERATIVE recall judge (SP_B3_JUDGE) =====
                        // The open-set novel-recall wall that defeated every GEOMETRIC signal
                        // (q·K, cosine, C2 sig, the W_c head, causal self-ablation) is broken by
                        // a GENERATIVE judge: the 12B READS the candidate memory TEXTS (the words,
                        // not post-RoPE K vectors) and picks the one that answers THIS query, or
                        // [NULL]. Validated offline at 85.7% recall@1 on _needle_corpus_div (the
                        // EXACT corpus that gave W_c ~50%); harness tools/xbar_lsh/judge_recall_test.py.
                        // Default-off (SP_B3_JUDGE unset) ⇒ this whole block is skipped ⇒
                        // byte-identical null floor. Runs only when no prior stage decided.
                        if !int2_decided && recalled.is_none()
                            && std::env::var("SP_B3_JUDGE").ok().filter(|s| !s.is_empty()).is_some()
                        {
                            // KAIROS-bounded working set: at most SP_B3_JUDGE_K (default 20) episodes.
                            let kbound: usize = std::env::var("SP_B3_JUDGE_K").ok()
                                .and_then(|s| s.parse().ok()).unwrap_or(20);
                            // The user query text (raw, NO chat template — the judge prompt itself
                            // is what carries the template). raw_user is the last user message.
                            let query = raw_user.as_ref().map(|s| s.trim().to_string()).unwrap_or_default();
                            if query.is_empty() {
                                tracing::info!("B3-JUDGE: no user query text -- skipping (clean prompt)");
                            } else {
                                // (1) Assemble the candidate set. KAIROS bounds it to <=kbound. The
                                // FINDINGS-#3 anti-position-bias discipline of the validated harness:
                                // the candidate ORDER is shuffled and each gets a RANDOMIZED copy-able
                                // tag (consonant+digit+consonant) — a fixed [M_A],[M_B],.. order anchors
                                // the model on the first slot (observed: it returned [M_A] for every
                                // query). The shuffle/tag draw is DETERMINISTIC per query (seeded by a
                                // hash of the query text) so a turn is reproducible/byte-exact-when-off.
                                // A tiny xorshift PRNG (no rand crate dep) drives a Fisher-Yates shuffle.
                                let mut seed: u64 = 0xcbf29ce484222325;
                                for b in query.as_bytes() { seed ^= *b as u64; seed = seed.wrapping_mul(0x100000001b3); }
                                if seed == 0 { seed = 0x9e3779b97f4a7c15; }
                                let mut rng = move || {
                                    seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; seed
                                };
                                // ===== KAIROS DUAL-AXIS WORKING-SET ASSEMBLY (LIVE-CONVERSATION) =====
                                // The "first K episodes" window is replaced by a recency+salience
                                // selection over a UNIFIED candidate pool = curated registry ∪ the LIVE
                                // NIGHTSHIFT episodes (the conversation's own immediate past). Two axes
                                // feed the SAME generative judge:
                                //   (1) RECENCY  : the last SP_B3_JUDGE_R episodes of app.nightshift (the
                                //       most-recent conversational turns; NIGHTSHIFT appends ep_live_NNN
                                //       to the end). ALWAYS included -- short-term working memory. This is
                                //       the loop-closure: the judge can recall what was just said.
                                //   (2) SALIENCE : ONLY if the cold pool > the remaining slots, the demoted
                                //       Stage-0 C2-LSH Hamming pre-filter rescues OLDER memories (curated
                                //       registry + nightshift episodes older than the recency-R tail) that
                                //       match THIS query. Compute the live query's C2 sig, score every cold
                                //       candidate by Hamming agreement (R_BITS - hamming) to its STORED sig
                                //       (exact, captured at ingest), take the top (K - R), append.
                                // Candidates are CLONED into a working Vec (<=K episodes; cheap) with a
                                // parallel provenance flag (live = recall via inject_tokens; curated =
                                // replay(dir)). The judge still makes the final pick; Stage-0 only widens
                                // what it can see. NULL FLOOR: with SP_B4_NIGHTSHIFT unset, app.nightshift
                                // is empty ⇒ recency is empty ⇒ the cold pool is the curated registry only
                                // ⇒ behaviour is registry-only, byte-identical to the prior KAIROS path.
                                let rbound: usize = std::env::var("SP_B3_JUDGE_R").ok()
                                    .and_then(|s| s.parse().ok()).unwrap_or(8);
                                // Snapshot the live nightshift episodes (clone; bounded by working set).
                                let ns_snapshot: Vec<recall::Episode> = {
                                    let g = app.nightshift.read().unwrap();
                                    g.iter().cloned().collect()
                                };
                                let n_live = ns_snapshot.len();
                                let n_cur = registry.len();
                                // ===== REORDER (ADR-002 §8.1): judge = reject veto, L5-direct owns delivery.
                                // Precompute the GLOBAL L5-#1 episode (bcos over registry ∪ nightshift by
                                // L5 query-key cosine — the IDENTICAL selection SP_RECALL_L5 delivers at
                                // 86.89%). On a judge PASS we deliver THIS, not the judge's own pick among
                                // the shuffled top-K shortlist (K-sweep: judge-pick delivery = 50% vs
                                // L5-#1 = 86.89%). The judge still sees the K(>=2) shortlist so it engages
                                // its PASS/NULL reject (K=1 degenerates to always-PICK). L5 mode off /
                                // best unavailable ⇒ None ⇒ falls back to the judge's pick (prior behavior,
                                // null floor preserved). (name, text) of the delivery target.
                                let l5_best: Option<(String, String)> =
                                    if std::env::var("SP_B3_JUDGE_L5").as_deref() == Ok("1") {
                                        let ql5g = recall::l5_query_embed(&qbuf);
                                        if ql5g.is_empty() { None } else {
                                            let (mut bcos, mut bname, mut btext) =
                                                (f32::NEG_INFINITY, String::new(), String::new());
                                            for ep in registry.iter().chain(ns_snapshot.iter()) {
                                                if ep.l5key.len() != recall::HD { continue; }
                                                let c = recall::cos512(&ql5g, &ep.l5key);
                                                if c > bcos { bcos = c; bname = ep.name.clone(); btext = ep.text.clone(); }
                                            }
                                            if btext.trim().is_empty() { None } else {
                                                tracing::info!("B3-JUDGE REORDER: L5 global best='{}' cos={:.3} (delivery target on PASS)", bname, bcos);
                                                Some((bname, btext))
                                            }
                                        }
                                    } else { None };
                                let rkeep = rbound.min(kbound).min(n_live);
                                // The working candidate set: each entry is (Episode, is_live).
                                let mut cands: Vec<(recall::Episode, bool)> = Vec::with_capacity(kbound);
                                let mut kairos_rescued: Vec<String> = Vec::new();
                                // (1) recency axis = the last rkeep LIVE episodes (the conversation tail).
                                for j in (n_live - rkeep)..n_live {
                                    cands.push((ns_snapshot[j].clone(), true));
                                }
                                // The cold pool = curated registry (all) ∪ older nightshift (before rkeep).
                                // Built as (Episode, is_live) so dispatch knows the recall path.
                                let cold: Vec<(recall::Episode, bool)> = registry.iter().cloned()
                                    .map(|e| (e, false))
                                    .chain(ns_snapshot[..(n_live - rkeep)].iter().cloned().map(|e| (e, true)))
                                    .collect();
                                let salience_slots = kbound.saturating_sub(cands.len());
                                if cold.len() <= salience_slots {
                                    // everything fits: take the whole cold pool, no Hamming cull needed.
                                    for c in cold.into_iter() { cands.push(c); }
                                } else if salience_slots > 0 {
                                    // STAGE-1 (the E2E composition): SP_B3_JUDGE_WC=<wc_deploy.bin> scores the
                                    // cold pool with the PROVEN W_c head (360/361 selector) and shortlists the
                                    // top (K-R) for the judge+0.6 verifier to ground. W_c is the correct Stage-1
                                    // — C2-Hamming (the else path) was a measured-weaker signal that would
                                    // starve the judge of the right candidate. Unset => Hamming (null floor).
                                    let wc_head = std::env::var("SP_B3_JUDGE_WC").ok()
                                        .filter(|s| !s.is_empty()).and_then(|p| recall::load_wc(&p));
                                    // STAGE-1 L5 (SP_B3_JUDGE_L5=1): rank the cold pool by L5 query-key
                                    // cosine (the 86.89%-live recall selector) and shortlist top-(K-R) for
                                    // the judge. This IS the pass->block: L5 PASSES the shortlist, the
                                    // generative judge PICKS the answerer or [NULL] (the reject). Requires
                                    // ep.l5 keys on the episodes; falls through to W_c/C2 if L5 unavailable.
                                    let ql5 = if std::env::var("SP_B3_JUDGE_L5").as_deref() == Ok("1") {
                                        recall::l5_query_embed(&qbuf)
                                    } else { Vec::new() };
                                    if !ql5.is_empty() {
                                        let mut scored: Vec<(usize, f32)> = cold.iter().enumerate()
                                            .map(|(i, (ep, _))| (i, if ep.l5key.len() == recall::HD {
                                                recall::cos512(&ql5, &ep.l5key) } else { f32::NEG_INFINITY }))
                                            .collect();
                                        scored.sort_by(|a, b| b.1.partial_cmp(&a.1)
                                            .unwrap_or(std::cmp::Ordering::Equal)); // descending L5 cosine
                                        for &(i, _sc) in scored.iter().take(salience_slots) {
                                            kairos_rescued.push(cold[i].0.name.clone());
                                            cands.push(cold[i].clone());
                                        }
                                        tracing::info!(
                                            "B3-JUDGE STAGE-1 L5: shortlisted top-{} of {} cold by L5 query-key cosine",
                                            salience_slots, cold.len());
                                    } else if let Some(head) = wc_head.as_ref() {
                                        let mut scored: Vec<(usize, f32)> = cold.iter().enumerate()
                                            .map(|(i, (ep, _))| (i, recall::wc_score(
                                                &qbuf, &ep.gk, ep.gk_ng, ep.npos as usize, head)))
                                            .collect();
                                        scored.sort_by(|a, b| b.1.partial_cmp(&a.1)
                                            .unwrap_or(std::cmp::Ordering::Equal)); // descending W_c relevance
                                        for &(i, _sc) in scored.iter().take(salience_slots) {
                                            kairos_rescued.push(cold[i].0.name.clone());
                                            cands.push(cold[i].clone());
                                        }
                                        tracing::info!(
                                            "B3-JUDGE STAGE-1 W_c: shortlisted top-{} of {} cold by W_c (head r={})",
                                            salience_slots, cold.len(), head.r);
                                    } else {
                                    // (2) salience axis: rescue the top (K - R) cold candidates by C2 Hamming.
                                    let mut qkbuf = vec![0.0f32; n_global * recall::HD * (npos_q as usize)];
                                    match unsafe { kv::read_global_k(handle, &mut qkbuf, npos_q) } {
                                        Ok(ngk) => {
                                            let qsig = app.recall_proj.signature(
                                                &qkbuf, ngk as usize, npos_q as usize);
                                            let mut scored: Vec<(usize, u32)> = cold.iter().enumerate()
                                                .map(|(i, (ep, _))| (i, recall::agree(&qsig, &ep.sig)))
                                                .collect();
                                            scored.sort_by(|a, b| b.1.cmp(&a.1)); // descending agreement
                                            for &(i, _ag) in scored.iter().take(salience_slots) {
                                                kairos_rescued.push(cold[i].0.name.clone());
                                                cands.push(cold[i].clone());
                                            }
                                            tracing::info!(
                                                "B3-JUDGE KAIROS: live={} cur={} K={} R={} (recency-kept={}) salience_slots={} qsig={:016x}.. rescued=[{}]",
                                                n_live, n_cur, kbound, rbound, rkeep, salience_slots, qsig[0],
                                                kairos_rescued.join(", "));
                                        }
                                        Err(e) => {
                                            tracing::warn!(
                                                "B3-JUDGE KAIROS: read_global_k(npos_q={}) failed: {e} -- recency-only working set (live tail)",
                                                npos_q);
                                        }
                                    }
                                    }
                                }
                                tracing::info!(
                                    "B3-JUDGE KAIROS working set: {} episode(s) ({} live-recency + {} cold), live_total={} cur_total={}",
                                    cands.len(), rkeep, cands.len().saturating_sub(rkeep), n_live, n_cur);
                                // Fisher-Yates shuffle the working set (anti-position-bias, FINDING #3).
                                for i in (1..cands.len()).rev() {
                                    let j = (rng() % (i as u64 + 1)) as usize;
                                    cands.swap(i, j);
                                }
                                // tag pool = consonant+digit+consonant (collision-free copy-able codes);
                                // drawn without replacement per query (sampled order from the same rng).
                                const CONS: &[u8] = b"BCDFGHJKLMNPQRSTVWXZ";
                                const DIG: &[u8] = b"0123456789";
                                let mut pool: Vec<String> = Vec::with_capacity(CONS.len() * 10 * CONS.len());
                                for &a in CONS { for &d in DIG { for &c in CONS {
                                    pool.push(String::from_utf8(vec![a, d, c]).unwrap()); } } }
                                for i in (1..pool.len()).rev() {
                                    let j = (rng() % (i as u64 + 1)) as usize;
                                    pool.swap(i, j);
                                }
                                let mut tags: Vec<String> = Vec::with_capacity(cands.len());
                                let mut texts: Vec<String> = Vec::with_capacity(cands.len());
                                for (slot, (ep, is_live)) in cands.iter().enumerate() {
                                    tags.push(pool[slot % pool.len()].clone());
                                    // detokenize the episode TEXT. LIVE episodes carry the raw turn tokens
                                    // (ep.tokens, captured with forced BOS + trailing \n); curated episodes
                                    // have an ep.tok sidecar on disk. BOS (id 2) dropped; bytes joined.
                                    let eptok: Vec<i32> = if *is_live {
                                        ep.tokens.clone().unwrap_or_default()
                                    } else {
                                        std::fs::read_to_string(
                                            std::path::Path::new(&ep.dir).join("ep.tok"))
                                            .ok().map(|c| c.lines().filter_map(|l| l.trim().parse::<i32>().ok()).collect())
                                            .unwrap_or_default()
                                    };
                                    let mut bytes: Vec<u8> = Vec::new();
                                    for &t in eptok.iter() {
                                        if t == 2 { continue; } // skip forced BOS
                                        bytes.extend_from_slice(app.tokenizer.decode_token(t));
                                    }
                                    let txt = String::from_utf8_lossy(&bytes).trim().to_string();
                                    // fall back to the registry/live topic if tokens were unavailable.
                                    texts.push(if txt.is_empty() { ep.topic.clone() } else { txt });
                                }
                                // (2) Build the judge prompt (the validated harness template) + apply
                                // the gemma chat template via apply_template_ids (FINDING #1: the
                                // messages path = instruct behavior; a raw prompt base-completes/echoes).
                                let mut entries = String::new();
                                for (tg, tx) in tags.iter().zip(texts.iter()) {
                                    entries.push_str(&format!("[{tg}] {tx}\n"));
                                }
                                let judge_content = format!(
                                    "You are a memory index. Each entry below has a TAG in [brackets]. \
Read the QUESTION and reply with ONLY the tag of the single entry that directly \
answers it. If no entry answers it, reply [NULL].\n\n{entries}\nQUESTION: {query}\n\
Tag of the answer (or [NULL]):");
                                // JUDGE-SERVED (Exp E, 2026-06-25): SP_B3_VERIFY swaps the weak single-tag
                                // prompt for the skeptical TAG+EVIDENCE contract. The generative judge gets
                                // the PICK (~85%); the deterministic token_overlap gate (parse site below)
                                // turns it into a 95%-reject judge by demanding a verifiable citation.
                                // Default-off (SP_B3_VERIFY unset) => the single-tag prompt above = null floor.
                                let verify_mode = std::env::var("SP_B3_VERIFY").ok().filter(|s| !s.is_empty()).is_some();
                                let judge_content = if verify_mode {
                                    format!("You are a STRICT memory index. Each entry has a TAG in [brackets].\n\n{entries}\nQUESTION: {query}\n\nMost questions have NO matching entry. Find the entry that directly answers the question; if none does, answer NONE. Then reply on ONE line EXACTLY:\nTAG=<the tag, or NONE> | EVIDENCE=<copy the exact words from that entry that answer it>\nANSWER:")
                                } else { judge_content };
                                let jmsgs = vec![Message {
                                    role: "user".to_string(), content: judge_content }];
                                match app.tokenizer.apply_template_ids(&jmsgs) {
                                    Err(e) => tracing::warn!("B3-JUDGE: apply_template_ids failed: {e} -- clean prompt"),
                                    Ok(jtoks) if jtoks.len() < 2 => {
                                        tracing::warn!("B3-JUDGE: judge prompt too short ({}) -- clean prompt", jtoks.len());
                                    }
                                    Ok(jtoks) => {
                                        // (3) NESTED judge inference. The resident cache currently holds the
                                        // original prompt prefilled at [0, anchor)=head. We reset, run the
                                        // judge token-by-token, parse the tag, then reset + re-prefill head
                                        // to RESTORE the exact pre-branch cache state (no contamination of
                                        // the final synthesis). reset() is the SWA-ring-safe reset (rewind
                                        // past Jmax is the diagnosed ring bug).
                                        // JUDGE-SERVED: in verify mode FORCE the format by prefilling "TAG="
                                        // into the model turn. The 12B otherwise sometimes answers in prose
                                        // (the correct recalled fact, but no tag) and the deterministic parse
                                        // drops a VALID recall (the NIGHTSHIFT live-fact smoke: it reproduced
                                        // "5-RAVEN-9921" but as a sentence). Seed `reply` with "TAG=" so
                                        // parse_tag_evidence sees the forced prefix. Default path unchanged.
                                        let mut jtoks = jtoks;
                                        let mut reply = String::new();
                                        if verify_mode {
                                            if let Ok(tt) = app.tokenizer.encode("TAG=") {
                                                let tt: Vec<i32> = tt.into_iter().filter(|&t| t != 2).collect();
                                                if !tt.is_empty() { jtoks.extend_from_slice(&tt); reply.push_str("TAG="); }
                                            }
                                        }
                                        let _ = unsafe { kv::reset(handle) };
                                        let (jhead, jlast) = jtoks.split_at(jtoks.len() - 1);
                                        let mut judge_ok = jhead.is_empty()
                                            || unsafe { kv::prefill(handle, jhead) }.is_ok();
                                        if judge_ok {
                                            // greedy: decode_step(jlast) gives the first generated logits.
                                            let mut tok = jlast[0];
                                            let turn_stops = app.tokenizer.turn_stop_ids();
                                            // JUDGE-SERVED: EVIDENCE needs room to be copied; the single-tag
                                            // path still stops at 10. (eos / turn_stop break out earlier.)
                                            let jbudget: u32 = if verify_mode { 64 } else { 10 };
                                            // JUDGE-SERVED: the served sampler masks the model's suppress-tokens
                                            // (gemma image/audio soft tokens 258882/258883 + control markers), but
                                            // this raw-argmax side-pass bypassed it -> the placeholders leaked into
                                            // EVIDENCE and derailed the judge. Mask them to -inf here too (the
                                            // structural cure for the <image|>/<audio|> spam). verify-only.
                                            let suppress = if verify_mode { app.tokenizer.suppress_token_ids() } else { Vec::new() };
                                            for _ in 0..jbudget {
                                                if unsafe { kv::decode_step(handle, tok, logits) }.is_err() {
                                                    judge_ok = false; break;
                                                }
                                                if verify_mode {
                                                    for &sid in &suppress {
                                                        let u = sid as usize;
                                                        if u < logits.len() { logits[u] = f32::NEG_INFINITY; }
                                                    }
                                                }
                                                // greedy argmax (temperature 0, penalty 1.0).
                                                let mut bi = 0usize; let mut bv = f32::NEG_INFINITY;
                                                for (i, &v) in logits.iter().enumerate() { if v > bv { bv = v; bi = i; } }
                                                let g = bi as i32;
                                                if app.tokenizer.eos_ids.contains(&g) || turn_stops.contains(&g) { break; }
                                                reply.push_str(&String::from_utf8_lossy(app.tokenizer.decode_token(g)));
                                                tok = g;
                                                // JUDGE-SERVED verify reply is ONE line: stop at the first
                                                // newline so the decode can't ramble into echo / special-token
                                                // spam (the live-smoke false-reject). Single-tag path unchanged.
                                                if verify_mode && reply.trim_start().contains('\n') { break; }
                                            }
                                        }
                                        // G-INT-2-FIX: the judge's nested forward is a DESTRUCTIVE read of
                                        // the resident cache — it advanced dpos past the prompt anchor and
                                        // wrote judge K/V into global slots [n-1, jhead+J). A plain
                                        // reset()+prefill(head) only rewinds the counters and rewrites
                                        // [0,n-1); the judge residue lingers at slots >= n-1. On a NULL turn
                                        // synthesis sits at pos=n-1 so the residue is beyond dpos (never
                                        // read => clean), but on a PICK the injected memory advances dpos
                                        // forward and the synthesis window sweeps the stale judge slots =>
                                        // prompt-echo degeneration (the proven Phase-4 blocker). Restore via
                                        // reset_cold (zeroes every owner K/V + journal) so the reconstruction
                                        // truly starts cold; the inject (below) then lands on a clean head and
                                        // nothing of the judge pass can be attended. Byte-identical to the
                                        // null-floor for the NULL/no-inject path (zeroed slots beyond dpos
                                        // are never read after a fresh prefill).
                                        // INSTRUMENT (root-cause receipt): log dpos at each stage.
                                        let dpos_pre = unsafe { kv::position(handle) };
                                        let _ = unsafe { kv::reset_cold(handle) };
                                        let dpos_postreset = unsafe { kv::position(handle) };
                                        if !head.is_empty() {
                                            if let Err(e) = unsafe { kv::prefill(handle, head) } {
                                                tracing::error!("B3-JUDGE: FATAL re-prefill(head) failed: {e}");
                                            }
                                        }
                                        let dpos_posthead = unsafe { kv::position(handle) };
                                        tracing::info!("B3-JUDGE COLD-RESTORE: dpos pre-restore={} after reset_cold={} after prefill(head)={} (head.len={})",
                                            dpos_pre, dpos_postreset, dpos_posthead, head.len());
                                        // (4) Parse the tag (FINDING #3: copy-able TAG, never an ordinal).
                                        // Longest matching tag wins; [NULL]/no-tag ⇒ reject. Match on the
                                        // full "[M_X]" surface so M_A never substring-collides with M_AB.
                                        let mut picked: Option<usize> = None;
                                        if verify_mode {
                                            // JUDGE-SERVED Stage-2 (Exp E): parse TAG+EVIDENCE; accept the
                                            // pick ONLY if its cited span clears the deterministic
                                            // token-overlap gate (0.6) against that entry's text. This is the
                                            // 95%-reject lever — the picker proposes, the CPU math disposes.
                                            let (pk, ev) = recall::parse_tag_evidence(&reply, &tags);
                                            match pk {
                                                Some(i) => {
                                                    let ov = recall::token_overlap(&ev, &texts[i]);
                                                    if ov >= recall::OVERLAP_THR {
                                                        picked = Some(i);
                                                        tracing::info!("B3-VERIFY: PICK [{}] overlap={:.3} >= {:.2} ACCEPT ev={:?}",
                                                            tags[i], ov, recall::OVERLAP_THR, ev.trim());
                                                    } else {
                                                        tracing::info!("B3-VERIFY: tag [{}] overlap={:.3} < {:.2} REJECT (ungrounded citation) ev={:?}",
                                                            tags[i], ov, recall::OVERLAP_THR, ev.trim());
                                                    }
                                                }
                                                None => tracing::info!("B3-VERIFY: NONE/no-tag -> REJECT (clean prompt)"),
                                            }
                                        } else {
                                            let up = reply.to_uppercase();
                                            let mut best_len = 0usize;
                                            if !up.contains("NULL") {
                                                for (slot, tg) in tags.iter().enumerate() {
                                                    let needle = format!("[{}]", tg);
                                                    if up.contains(&needle.to_uppercase()) && needle.len() > best_len {
                                                        best_len = needle.len(); picked = Some(slot);
                                                    }
                                                }
                                                // tolerate a bare tag without brackets (model sometimes drops them)
                                                if picked.is_none() {
                                                    for (slot, tg) in tags.iter().enumerate() {
                                                        if up.contains(&tg.to_uppercase()) && tg.len() > best_len {
                                                            best_len = tg.len(); picked = Some(slot);
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                        match picked {
                                            None => tracing::info!(
                                                "B3-JUDGE: [NULL] (reply={:?}) -> REJECT (clean prompt, foreign)", reply.trim()),
                                            Some(slot) => {
                                                let (ep, _is_live) = &cands[slot];
                                                // (5) TEXT-IN-CONTEXT RECALL (G-INT-2-FIX, the Phase-4
                                                // sealer). The α-sweep PROVED latent injection of a live
                                                // episode has NO operating point that recites the specific
                                                // fact (high mass=echo-hijack, low mass=parametric-washout).
                                                // The generative judge works by READING TEXT, so synthesis
                                                // reads the recalled memory TEXT too: prepend the picked
                                                // episode's text (texts[slot]) to the user query, RAG-style,
                                                // re-template, and rebuild the cache from that augmented
                                                // prompt. The model then reads + recites the fact natively
                                                // through its own generative pipeline — NO lossy latent KV
                                                // inject for the payload. The Ring-2/XBAR substrate stays the
                                                // SELECTION mechanism (the judge PICK above); recitation goes
                                                // through the native pipeline. We DROP inject_tokens/replay
                                                // for the recitation path (operator decision).
                                                // AUTHORITY FIX (2026-07-01): deliver the CLEAN manifest text
                                                // (ep.text, the same field L5-direct delivers), NOT texts[slot]
                                                // — texts[slot] is the detokenized RAW ep.tok turn (captured
                                                // with forced BOS + trailing \n = chat-template cruft), which
                                                // polluted "Context (authoritative): ..." and dropped the model
                                                // back to parametric (right pick, wrong answer). Fall back to
                                                // the detokenized turn only for live episodes with no manifest.
                                                // REORDER (ADR-002 §8.1): the judge PICK is a PASS signal only.
                                                // Deliver the GLOBAL L5-#1 (l5_best, the 86.89% SP_RECALL_L5
                                                // selection), NOT the judge's own shortlist pick (measured 50%
                                                // — the judge picks #2 or the model resists). The judge already
                                                // did its job: PASS (some memory answers) vs NULL (foreign
                                                // reject). Fall back to the judge's pick text only when L5 mode
                                                // is off / L5 best unavailable (prior behavior, null floor).
                                                let (deliver_name, mem_text) = match l5_best.as_ref() {
                                                    Some((n, t)) if !t.trim().is_empty() => (n.clone(), t.clone()),
                                                    _ => (ep.name.clone(),
                                                          if !ep.text.trim().is_empty() { ep.text.clone() } else { texts[slot].clone() }),
                                                };
                                                // JUDGE-SERVED synthesis: rigid CLOSED-TASK framing. The prior
                                                // conversational "Using that context, answer:" wording induced a
                                                // template-continuation trap — the model answered, then echoed the
                                                // prompt / degenerated (Georgian, python) to fill max_tokens. A
                                                // closed instruction makes it answer concisely and emit the turn-stop
                                                // (the eos/turn_stop break at the synthesis loop then halts it).
                                                // DELIVERY FIX (2026-07-01): deliver through the EXACT clean
                                                // text-in-context format of the proven SP_RECALL_L5 / Jaccard
                                                // branches (faithfulness system prompt + "Context (authoritative,
                                                // current): <fact>\n\n<query>"). The prior single-user "Provide a
                                                // direct, concise answer" framing (NO system prompt) degenerated ->
                                                // tag echo + <image|>/Georgian garbage. The judge is a PICK/NULL
                                                // GATE; recitation goes through the proven 86.89% clean delivery.
                                                let aug_msgs = vec![
                                                    Message { role: "system".to_string(), content: "You are Shannon-Prime, a local AI with a real working memory. Keep replies short. Use facts you were given faithfully; if you don't know, say so.".to_string() },
                                                    Message { role: "user".to_string(), content: format!("Context (authoritative, current): {mem_text}\n\n{query}") },
                                                ];
                                                match app.tokenizer.apply_template_ids(&aug_msgs) {
                                                    Ok(aug_toks) if aug_toks.len() >= 2 => {
                                                        // Establish the synthesis cache from the augmented
                                                        // prompt via the SAME single clean reconstruction the
                                                        // null path uses: reset_cold (zero every owner K/V +
                                                        // journal) -> prefill(aug_head). The synthesis decode
                                                        // loop then starts from decode_step(syn_last) where
                                                        // syn_last = aug_toks[last]. The recalled fact is now
                                                        // in the model's context window.
                                                        // ROOT-CAUSE PROBE (2026-07-01): log reset_cold's result + the
                                                        // resulting dpos. If reset_cold fails or leaves dpos!=0, the aug
                                                        // is prefilled on the judge's polluted cache -> authority loss.
                                                        match unsafe { kv::reset_cold(handle) } {
                                                            Ok(_) => tracing::info!("B3-JUDGE: reset_cold OK, dpos={}", unsafe { kv::position(handle) }),
                                                            Err(e) => tracing::error!("B3-JUDGE: reset_cold FAILED: {e} (dpos={})", unsafe { kv::position(handle) }),
                                                        }
                                                        let (aug_head, aug_last) = aug_toks.split_at(aug_toks.len() - 1);
                                                        let mut ok = true;
                                                        if !aug_head.is_empty() {
                                                            if let Err(e) = unsafe { kv::prefill(handle, aug_head) } {
                                                                tracing::error!("B3-JUDGE: FATAL aug prefill failed: {e} -- clean prompt");
                                                                ok = false;
                                                            }
                                                        }
                                                        if ok {
                                                            syn_last = aug_last[0];
                                                            // provenance/telemetry reflect the DELIVERED fact
                                                            // (deliver_name = L5-#1 on reorder; = judge pick on fallback).
                                                            recalled = Some((deliver_name.clone(), 1000));
                                                            judge_ground = Some((deliver_name.clone(), mem_text.clone()));
                                                            tracing::info!(
                                                                "B3-JUDGE: PASS (judge PICK [{}] '{}' topic='{}' reply={:?}) -> DELIVER '{}' TEXT-IN-CONTEXT (aug n={}, dpos={}) -> synthesis recites natively",
                                                                tags[slot], ep.name, ep.topic, reply.trim(), deliver_name, aug_toks.len(),
                                                                unsafe { kv::position(handle) });
                                                        } else {
                                                            // aug reconstruction failed: fall back to the
                                                            // original clean prompt (reset_cold + prefill(head))
                                                            // so synthesis is at least coherent (null-floor-like).
                                                            let _ = unsafe { kv::reset_cold(handle) };
                                                            if !head.is_empty() {
                                                                let _ = unsafe { kv::prefill(handle, head) };
                                                            }
                                                        }
                                                    }
                                                    other => {
                                                        if let Err(e) = other.map(|_| ()) {
                                                            tracing::warn!("B3-JUDGE: aug apply_template_ids failed: {e} -- clean prompt");
                                                        } else {
                                                            tracing::warn!("B3-JUDGE: aug prompt too short -- clean prompt");
                                                        }
                                                        // leave the restored clean cache (head already prefilled
                                                        // by the cold-restore above) + syn_last == last[0].
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        // B3-v9c THE DISPOSER (Stage-2 post-inject semantic verify-and-rewind).
                        // The q·K Stage-1 ranker (above) PROPOSES; the Disposer DISPOSES. For
                        // each candidate episode it SPECULATIVELY injects (M_target budget if set)
                        // into the resident cache, teacher-forces a Yes/No reasoning bridge via
                        // decode_step, reads margin = logit(Yes)-logit(No) at the answer position,
                        // then O(1)-rewinds the probe (last token + bridge + episode) to vaporise
                        // it (the KAI-1b slot==pos / SWA-journal inverse). The best-margin episode
                        // is re-injected and recalled IFF margin > SP_B3_DISPOSER_TAU. TELEMETRY-
                        // FIRST (the codebase's tau_qk discipline): TAU defaults to +inf so the
                        // first run NEVER fires — we read the per-(query,episode) margin matrix,
                        // verify open-world separation (the v6/v7 Yes/No bridge FAILED in-prompt;
                        // this is the post-LATENT-inject variant, re-measured), THEN pin TAU.
                        // The M_target=42 budget is the safety floor: a wrong FIRE stumbles
                        // (sub-dominant), it does not hijack. Default-off ⇒ falls through to the
                        // existing q·K fire (null floor).
                        let disp_mode = if int2_decided { 0 } else { std::env::var("SP_B3_DISPOSER").ok()
                            .and_then(|s| s.trim().parse::<i32>().ok()).unwrap_or(0) };
                        if disp_mode == 2 {
                            // B3-v10 ABLATION GATE (the Thermodynamic Knockout). Per candidate: inject E,
                            // greedily decode the 8-tok payload, then memset-zero the episode positions
                            // whose tokens match the payload window (ep.tok sidecar), re-score the SAME
                            // payload, and measure collapse = Σ(LL_ablated - LL_E) over [2,8). A TRUE match
                            // is structurally load-bearing ⇒ catastrophic NEGATIVE collapse; a parametric
                            // bleed / empty-match shrugs it off ⇒ ≈0. The audio super-attractor has no
                            // ep.tok ⇒ empty mask ⇒ collapse 0 ⇒ fails the knockout (lexical limit ENFORCES
                            // the semantic boundary). Fire iff collapse < SP_B3_DISPOSER_TAU (telemetry -inf).
                            let abl_tau: f32 = std::env::var("SP_B3_DISPOSER_TAU").ok()
                                .and_then(|s| s.parse().ok()).unwrap_or(f32::NEG_INFINITY);
                            const KGEN: usize = 8;
                            let lse = |z: &[f32]| -> f32 {
                                let m = z.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                                let mut s = 0.0f32; for &v in z { s += (v - m).exp(); } m + s.ln()
                            };
                            let argmax = |z: &[f32]| -> usize {
                                let mut bi = 0usize; let mut bv = f32::NEG_INFINITY;
                                for (i, &v) in z.iter().enumerate() { if v > bv { bv = v; bi = i; } } bi
                            };
                            // v12 TEACHER-FORCED KNOCKOUT: SP_B3_SECRET=<exact secret string>. Instead of
                            // greedy-decoding (which may not recite the secret in [2,8)), teacher-force the
                            // KNOWN secret tokens and score THEIR NLL with vs without the episode's source
                            // rows. Measures the secret's dependency DIRECTLY; window = the full secret.
                            // SECRET resolution is PER-EPISODE (resolved inside the loop below): env
                            // SP_B3_SECRET is the single-shot override; otherwise each episode supplies
                            // its own knockout target via an ep.secret sidecar (the corpus admission
                            // path — ONE daemon boot labels the whole registry, each ep self-tested with
                            // ITS OWN secret). Missing sidecar => empty => that ep falls back to the
                            // greedy [2,8) window (no crash, null floor preserved).
                            let env_secret = std::env::var("SP_B3_SECRET").ok().filter(|s| !s.is_empty());
                            let anchor = unsafe { kv::position(handle) };
                            let mut best_abl: Option<(usize, f32)> = None;
                            let mut drows: Vec<String> = Vec::with_capacity(registry.len());
                            for (i, ep) in registry.iter().enumerate() {
                                if ep.npos <= 0 || ep.gk.is_empty() { continue; }
                                let eptok: Vec<i32> = std::fs::read_to_string(std::path::Path::new(&ep.dir).join("ep.tok"))
                                    .ok().map(|s| s.lines().filter_map(|l| l.trim().parse::<i32>().ok()).collect())
                                    .unwrap_or_default();
                                // per-episode knockout target: env override else this ep's ep.secret
                                // sidecar (leading space preserved; only trailing CR/LF stripped).
                                let secret_ids: Vec<i32> = env_secret.clone()
                                    .or_else(|| std::fs::read_to_string(std::path::Path::new(&ep.dir).join("ep.secret")).ok())
                                    .map(|s| { let s = s.trim_end_matches(|c: char| c == '\n' || c == '\r').to_string();
                                               app.tokenizer.encode(&s).unwrap_or_default() })
                                    .map(|mut v| { if v.first() == Some(&2) { v.remove(0); } v })
                                    .unwrap_or_default();
                                let w0: usize = if secret_ids.is_empty() { 2 } else { 0 };
                                // Leg 1: inject E, greedy-decode the payload, record lp_E.
                                if unsafe { kv::replay(handle, &ep.dir, ep.npos, false) }.is_err() { continue; }
                                let mut gen: Vec<i32> = Vec::new();
                                let mut lpe: Vec<f32> = Vec::new();
                                let mut tok = last[0];
                                if secret_ids.is_empty() {
                                    for _ in 0..KGEN {
                                        if unsafe { kv::decode_step(handle, tok, logits) }.is_err() { break; }
                                        let g = argmax(logits); lpe.push(logits[g] - lse(logits)); gen.push(g as i32); tok = g as i32;
                                    }
                                } else {
                                    for &s in &secret_ids {   // teacher-force the KNOWN secret
                                        if unsafe { kv::decode_step(handle, tok, logits) }.is_err() { break; }
                                        lpe.push(logits[s as usize] - lse(logits)); gen.push(s); tok = s;
                                    }
                                }
                                let ng = gen.len();
                                let _ = unsafe { kv::rewind(handle, ng as i32) };   // undo payload, keep E
                                if ng == 0 { drows.push(format!("{}(collapse=nan)", ep.name)); let _ = unsafe { kv::rewind(handle, ep.npos) }; continue; }
                                // Target positions: episode indices whose token matches a payload-window token.
                                let mut targets: Vec<i32> = Vec::new();
                                if !eptok.is_empty() {
                                    let want: std::collections::HashSet<i32> = gen[w0.min(ng)..ng].iter().copied().collect();
                                    for (p, &t) in eptok.iter().enumerate() {
                                        if p >= ep.npos as usize { break; }
                                        if want.contains(&t) { targets.push(p as i32); }
                                    }
                                    if targets.len() > 12 { targets.truncate(12); }
                                }
                                // Leg 2: ablate targets, teacher-force the SAME payload, record lp_ablated.
                                let _ = unsafe { kv::ablate(handle, anchor, &targets) };
                                let mut lpa: Vec<f32> = Vec::with_capacity(ng);
                                let mut tok = last[0];
                                for i2 in 0..ng {
                                    if unsafe { kv::decode_step(handle, tok, logits) }.is_err() { break; }
                                    lpa.push(logits[gen[i2] as usize] - lse(logits)); tok = gen[i2];
                                }
                                let _ = unsafe { kv::rewind(handle, lpa.len() as i32 + ep.npos) };   // clear payload + episode (restores ablated rows)
                                let n = lpe.len().min(lpa.len());
                                let mut collapse = 0.0f32;
                                for j in w0..n { collapse += lpa[j] - lpe[j]; }
                                drows.push(format!("{}(collapse={:.2},ntgt={})", ep.name, collapse, targets.len()));
                                let better = match best_abl { None => true, Some((_, b)) => collapse < b };
                                if better { best_abl = Some((i, collapse)); }
                            }
                            tracing::info!("B3-DISPOSER ABLATION collapse=ΣΔLL_ablated[2,{}) (more-neg=load-bearing): [{}] TAU={:.3}", KGEN, drows.join(" "), abl_tau);
                            if let Some((idx, collapse)) = best_abl {
                                let ep = &registry[idx];
                                let fire = collapse < abl_tau && ep.npos > 0 && !ep.gk.is_empty();
                                tracing::info!("B3-DISPOSER: best='{}' (topic='{}') collapse={:.3} ⇒ {}",
                                    ep.name, ep.topic, collapse, if fire { "AUTHORIZE (load-bearing, re-inject)" } else { "REJECT (rewound; clean prompt)" });
                                if fire {
                                    if let Err(e) = unsafe { kv::replay(handle, &ep.dir, ep.npos, false) } {
                                        tracing::warn!("B3-DISPOSER: ablation re-inject({}, {}) failed: {e} — proceeding without recall", ep.dir, ep.npos);
                                    } else {
                                        recalled = Some((ep.name.clone(), ((-collapse).max(0.0) * 1000.0) as u32));
                                    }
                                }
                            }
                        } else if disp_mode == 1 {
                            let disp_tau: f32 = std::env::var("SP_B3_DISPOSER_TAU").ok()
                                .and_then(|s| s.parse().ok()).unwrap_or(f32::INFINITY);
                            // v9h: ΔLL polarity. +1 (default) = argmax payload-ΔLL (current); -1 =
                            // argmin = the episode that DISRUPTS the continuation MOST (the
                            // "truth-is-surprising" hypothesis). Toggle to adjudicate on the metal.
                            let dcont_sign: f32 = std::env::var("SP_B3_DCONT_SIGN").ok()
                                .and_then(|s| s.parse().ok()).unwrap_or(1.0);
                            // MULTI-TOKEN Δ-CONTINUATION signal (v9e). First-token Δcont was blind
                            // (1/3): biographical answers open with boilerplate ("Robert"/"He"); the
                            // memory-specific facts ("The Bill","Herons") surface positions 3-8. So
                            // measure the SEMANTIC PAYLOAD: under REAL E greedily decode K tokens =
                            // the model's own continuation g[0..K); then under ZEROED E (same npos,
                            // null content) teacher-force THAT SAME sequence and read each token's
                            // log-prob. signal = Σ ΔLL = Σ (log p_real(g_i) - log p_zero(g_i)) over
                            // the PAYLOAD window i∈[2,K) (skip the boilerplate). A true match forces
                            // tokens that are HIGHLY SURPRISING to the zeroed cache (large +ΔLL); a
                            // mismatch yields the prompt-driven continuation the zeroed cache already
                            // expects (~0). This is the teacher-forced ΔLL on the model's OWN answer —
                            // content-aware, not a raw distribution magnitude, so the per-episode
                            // content-mass bias is suppressed. Ranks on the payload-window ΔLL.
                            const KGEN: usize = 8;
                            let lse = |z: &[f32]| -> f32 {
                                let m = z.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                                let mut s = 0.0f32; for &v in z { s += (v - m).exp(); } m + s.ln()
                            };
                            let argmax = |z: &[f32]| -> usize {
                                let mut bi = 0usize; let mut bv = f32::NEG_INFINITY;
                                for (i, &v) in z.iter().enumerate() { if v > bv { bv = v; bi = i; } } bi
                            };
                            let mut best_disp: Option<(usize, f32)> = None;
                            let mut drows: Vec<String> = Vec::with_capacity(registry.len());
                            for (i, ep) in registry.iter().enumerate() {
                                if ep.npos <= 0 || ep.gk.is_empty() { continue; }
                                // Leg 1: REAL E — greedily decode the continuation, record its log-probs.
                                if unsafe { kv::replay(handle, &ep.dir, ep.npos, false) }.is_err() { continue; }
                                let mut gen: Vec<i32> = Vec::with_capacity(KGEN);
                                let mut lpr: Vec<f32> = Vec::with_capacity(KGEN);
                                let mut tok = last[0];
                                for _ in 0..KGEN {
                                    if unsafe { kv::decode_step(handle, tok, logits) }.is_err() { break; }
                                    let g = argmax(logits);
                                    lpr.push(logits[g] - lse(logits));
                                    gen.push(g as i32);
                                    tok = g as i32;
                                }
                                let _ = unsafe { kv::rewind(handle, gen.len() as i32 + ep.npos) };
                                if gen.is_empty() { drows.push(format!("{}(pay=nan)", ep.name)); continue; }
                                // Leg 2: ZEROED E — teacher-force the SAME sequence, read its log-probs.
                                if unsafe { kv::replay(handle, &ep.dir, ep.npos, true) }.is_err() { continue; }
                                let mut lpz: Vec<f32> = Vec::with_capacity(gen.len());
                                let mut tok = last[0];
                                for i2 in 0..gen.len() {
                                    if unsafe { kv::decode_step(handle, tok, logits) }.is_err() { break; }
                                    lpz.push(logits[gen[i2] as usize] - lse(logits));
                                    tok = gen[i2];
                                }
                                let _ = unsafe { kv::rewind(handle, lpz.len() as i32 + ep.npos) };
                                // ΔLL = log p_real - log p_zero, summed over the payload window [2,n).
                                let n = lpr.len().min(lpz.len());
                                let (mut dll_all, mut dll_pay) = (0.0f32, 0.0f32);
                                for j in 0..n { let d = lpr[j] - lpz[j]; dll_all += d; if j >= 2 { dll_pay += d; } }
                                drows.push(format!("{}(pay={:.2},all={:.2})", ep.name, dll_pay, dll_all));
                                let better = match best_disp { None => true, Some((_, b)) => dcont_sign * dll_pay > dcont_sign * b };
                                if better { best_disp = Some((i, dll_pay)); }
                            }
                            tracing::info!("B3-DISPOSER Δcont(multi-tok) ΣΔLL pay[2,{})/all: [{}] TAU={:.3}", KGEN, drows.join(" "), disp_tau);
                            if let Some((idx, dll)) = best_disp {
                                let ep = &registry[idx];
                                let fire = dll > disp_tau && ep.npos > 0 && !ep.gk.is_empty();
                                tracing::info!("B3-DISPOSER: best='{}' (topic='{}') ΣΔLL_pay={:.3} ⇒ {}",
                                    ep.name, ep.topic, dll, if fire { "AUTHORIZE (re-inject)" } else { "REJECT (rewound; clean prompt)" });
                                if fire {
                                    // SAFETY: handle live; cache Mutex held.
                                    if let Err(e) = unsafe { kv::replay(handle, &ep.dir, ep.npos, false) } {
                                        tracing::warn!("B3-DISPOSER: authorized re-inject({}, {}) failed: {e} — proceeding without recall", ep.dir, ep.npos);
                                    } else {
                                        recalled = Some((ep.name.clone(), (dll.max(0.0) * 1000.0) as u32));
                                    }
                                }
                            }
                        } else if let Some((idx, score)) = (if int2_decided { None } else { best }) {
                            let ep = &registry[idx];
                            let fire = score >= tau_qk && ep.npos > 0 && !ep.gk.is_empty();
                            tracing::info!(
                                "B3-v2 recall: best='{}' (topic='{}') topm={:.3} TAU={:.3} ⇒ {}",
                                ep.name, ep.topic, score, tau_qk,
                                if fire { "FIRE (auto-replay)" } else { "REJECT (below TAU; no replay)" }
                            );
                            if fire {
                                // SAFETY: handle live; we hold the cache Mutex.
                                if let Err(e) = unsafe { kv::replay(handle, &ep.dir, ep.npos, false) } {
                                    tracing::warn!("B3-v2 recall: auto-replay({}, {}) failed: {e} — proceeding without recall", ep.dir, ep.npos);
                                } else {
                                    recalled = Some((ep.name.clone(), (score * 1000.0) as u32));
                                }
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!("B3-v2 recall: read_global_q failed: {e} — proceeding without recall");
                    }
                }
            }
        } else {
            tracing::info!("B3-v2 recall: auto_recall set but no registry loaded (SP_RECALL_REGISTRY) — no-op");
        }
    }
    // Surface the recall decision on the SSE stream as a structured event so the
    // console can show a "recalled episode" indicator. Emitted before the answer
    // tokens (a separate event name so it never appears in the text delta).
    if let Some((ref name, score)) = recalled {
        let payload = serde_json::json!({"recalled": name, "agree": score, "chat_id": chat_id});
        let _ = tx.blocking_send(Ok(Event::default().event("recall").data(payload.to_string())));
    } else if auto_recall && replay_dir.is_none() {
        let payload = serde_json::json!({"recalled": serde_json::Value::Null, "chat_id": chat_id});
        let _ = tx.blocking_send(Ok(Event::default().event("recall").data(payload.to_string())));
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
    // ATTR-GATE ZERO-INFERENCE SYMBOLIC DECLINE (ADR-002 Tier-2 executor). The recall
    // stage decided the delivered fact does NOT state the queried attribute (private
    // entity, attribute-absent). Emit the deterministic decline and RETURN before the
    // synthesis forward — the gemma4 decode loop never runs, so confabulation/leak is
    // mathematically impossible and the turn resolves at string-allocation speed.
    // reset_cold keeps the persistent KV clean for the next turn.
    if let Some(msg) = symbolic_decline.take() {
        let payload = serde_json::to_string(&ChatDelta { delta: msg, chat_id }).unwrap_or_default();
        let _ = tx.blocking_send(Ok(Event::default().data(payload)));
        let _ = unsafe { kv::reset_cold(handle) };
        let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));
        let _ = app.events_tx.send(DaemonEvent::Chat { chat_id, status: "done" });
        tracing::info!("ATTR-DECLINE: zero-inference symbolic decline streamed (no gemma4 decode)");
        sessions.remove(chat_id);
        return;
    }
    // ── GEODESIC G-FM-STEER (ADR-003 §4.2) — SP_STEER_VEC=<raw f32 LE bin> +
    // SP_STEER_ALPHA=<f32>. Arms persistent pre-head steering for THIS turn's
    // synthesis when a recall fired (the F3 treatment was measured on recall
    // turns); explicitly DISARMS on non-recall turns so a prior arm never leaks
    // across turns on the resident session. Env unset ⇒ the verb is never called
    // ⇒ byte-identical null floor (session calloc's steer_active=0).
    if let (Ok(vf), Some(al)) = (std::env::var("SP_STEER_VEC"),
            std::env::var("SP_STEER_ALPHA").ok().and_then(|s| s.trim().parse::<f32>().ok())) {
        static STEER_VEC: std::sync::OnceLock<Vec<f32>> = std::sync::OnceLock::new();
        let v = STEER_VEC.get_or_init(|| match std::fs::read(&vf) {
            Ok(b) => b.chunks_exact(4)
                      .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            Err(e) => { tracing::warn!("FM-STEER: read {vf} failed ({e}) — steering disabled"); Vec::new() }
        });
        let steer_on = recalled.is_some() && !v.is_empty() && al != 0.0;
        match unsafe { kv::steer_set(handle,
                if steer_on { v.as_slice() } else { &[] },
                if steer_on { al } else { 0.0 }) } {
            Ok(()) => { if steer_on {
                tracing::info!("FM-STEER: armed alpha={al} dim={} (recall turn)", v.len());
            } }
            Err(e) => tracing::warn!("FM-STEER: steer_set failed ({e})"),
        }
    }
    // ── GEODESIC F3 CAPTURE (ADR-003 §5) — SP_F3_CAPTURE=<dir>; unset ⇒ nothing armed
    // ⇒ byte-identical null floor. Two one-shot taps of the post-output_norm hidden
    // (kv::capture_feat_arm): (1) armed on the syn_last step directly below = the
    // answer-turn LAST-PROMPT-TOKEN state (the feature the LM head consumes to pick
    // the first answer token — for a recall turn this is the state under the delivered
    // fact + faithfulness prompt; for a clean turn, the parametric state); (2) armed on
    // the first in-loop step = the FIRST-ANSWER-TOKEN state. Zero extra forwards — the
    // tap is two D2H copies on steps the turn runs anyway. Files + meta written at
    // turn end (after the decode loop). Offline-batch rail ONLY: capture_feat commits
    // like any decode (routes.rs:787 note is about PRE-cache routing, not this site).
    const F3_E: usize = 3840; // gemma4-12B hidden (UNIFICATION substrate map)
    let f3_dir = std::env::var("SP_F3_CAPTURE").ok().filter(|s| !s.is_empty());
    let mut f3_prompt: Vec<f32> = Vec::new();
    let mut f3_first: Vec<f32> = Vec::new();
    if f3_dir.is_some() {
        f3_prompt = vec![0f32; F3_E];
        if let Err(e) = unsafe { kv::capture_feat_arm(handle, &mut f3_prompt) } {
            tracing::warn!("F3-CAPTURE: arm(prompt) failed ({e}) — capture skipped this turn");
            f3_prompt = Vec::new();
        }
    }
    // G-INT-2-FIX: synthesis starts from syn_last (== last[0] for null/clean turns;
    // the augmented prompt's last token when the B3-JUDGE PICKed a memory text-in-context).
    if let Err(e) = unsafe { kv::decode_step(handle, syn_last, logits) } {
        // F3: an armed capture fires inside the step; on step FAILURE the armed pointer
        // may persist on the session — leak the buffer so a late write can never hit
        // freed memory (dangling-write guard; error path only, serve is aborting anyway).
        if !f3_prompt.is_empty() { std::mem::forget(std::mem::take(&mut f3_prompt)); }
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
    // SP_EOT_DEBUG (read-only): on free-gen turns, log the BEST stop-token RANK (# vocab
    // logits strictly greater) from the RAW forward logits per generated token. High rank
    // throughout => the model never wants to end (forward/mapping bug); low rank but not
    // chosen => sampler/top_k. Default-off => zero overhead.
    let eot_dbg = judge_ground.is_none()
        && std::env::var("SP_EOT_DEBUG").ok().as_deref() == Some("1");
    let eot_stop_ids: Vec<usize> = tokenizer.eos_ids.iter().chain(turn_stop_ids.iter())
        .map(|&x| x as usize).filter(|&x| x < logits.len()).collect();
    let stop_rank = |lg: &[f32]| -> (i32, usize) {
        let mut bid = -1i32; let mut bl = f32::NEG_INFINITY;
        for &s in &eot_stop_ids { if lg[s] > bl { bl = lg[s]; bid = s as i32; } }
        if bid < 0 { (-1, usize::MAX) } else { (bid, lg.iter().filter(|&&v| v > bl).count()) }
    };
    let mut eot_gen: Vec<i32> = Vec::new();
    let mut eot_ranks: Vec<usize> = Vec::new();
    // PERSIST-KV: the tokens we commit to the KV this turn (each is decode_step'd in the loop
    // body below). Appended to the prompt to form the next turn's reusable committed prefix.
    let mut committed_gen: Vec<i32> = Vec::new();
    if eot_dbg { eot_ranks.push(stop_rank(logits).1); }
    // SP_EOT_BIAS: the forward drives the end-of-turn token to ~rank 1 at a real turn
    // boundary but a hair short of winning, so the model never stops and degenerates.
    // A small fixed bias on the stop tokens tips a rank-1 boundary to chosen WITHOUT
    // firing mid-answer (where eot sits at rank ~1000, far out of reach). 0 = null floor.
    let eot_bias: f32 = eot_bias_req.unwrap_or_else(|| std::env::var("SP_EOT_BIAS").ok()
        .and_then(|s| s.trim().parse().ok()).unwrap_or(0.0));
    if eot_bias != 0.0 { for &s in &eot_stop_ids { logits[s] += eot_bias; } }
    let mut next_token = sampler.sample(logits);
    sampler.observe(next_token);
    // B3-JUDGE grounding: accumulate the synthesized answer for the post-hoc check.
    let mut answer_text = String::new();

    'decode: for _ in 0..max_tokens {
        if (!tokenizer.eos_ids.is_empty() && tokenizer.eos_ids.contains(&next_token))
            || turn_stop_ids.contains(&next_token)
        {
            break 'decode;
        }
        if eot_dbg && eot_gen.len() < 64 { eot_gen.push(next_token); }

        let token_bytes = tokenizer.decode_token(next_token);
        // JUDGE-SERVED hard stop (recall turns only): the 12B won't self-terminate the
        // text-in-context synthesis and degenerates AFTER the correct answer (code
        // fences, newlines, repetition). A concise factual recall answer never contains
        // a newline or backtick — the moment one appears, halt BEFORE streaming it so
        // the babble never reaches the client. judge_ground gate => normal chat unchanged.
        if judge_ground.is_some() {
            let tb = String::from_utf8_lossy(token_bytes);
            if tb.contains('\n') || tb.contains('`') { break 'decode; }
        }
        let stop_hit = match dec_buf.push(token_bytes) {
            PushResult::Emit(bytes) => {
                if !bytes.is_empty() {
                    let text = String::from_utf8_lossy(&bytes).into_owned();
                    answer_text.push_str(&text);
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

        // PERSIST-KV: record this token as committed -- the decode_step below writes its K/V into
        // the cache, so it becomes part of the reusable committed prefix for the next turn. Placed
        // AFTER every early break above so committed_gen never lists a token the cache doesn't hold.
        committed_gen.push(next_token);
        // GEODESIC F3 tap (2): first in-loop step = the FIRST-ANSWER-TOKEN state.
        // Armed immediately before the unconditional decode_step below so the one-shot
        // ALWAYS fires into a live buffer. f3_first: [] = not yet armed; len==F3_E =
        // captured; len==1 = failed-sentinel (never retried, never written).
        if !f3_prompt.is_empty() && f3_first.is_empty() {
            let mut b = vec![0f32; F3_E];
            match unsafe { kv::capture_feat_arm(handle, &mut b) } {
                Ok(()) => f3_first = b, // move keeps the heap buffer address — armed ptr stays valid
                Err(e) => { tracing::warn!("F3-CAPTURE: arm(first) failed ({e})"); f3_first = vec![f32::NAN]; }
            }
        }
        // Feed the just-emitted token; get logits for the next position.
        // SAFETY: handle live; logits is vocab_size f32 (checked above).
        if let Err(_e) = unsafe { kv::decode_step(handle, next_token, logits) } {
            // F3 dangling-write guard (see the syn_last site): step failed with a
            // possibly-armed capture — leak the buffer rather than risk a late write.
            if f3_first.len() == F3_E { std::mem::forget(std::mem::take(&mut f3_first)); }
            break 'decode;
        }
        if eot_dbg && eot_ranks.len() < 64 { eot_ranks.push(stop_rank(logits).1); }
        if eot_bias != 0.0 { for &s in &eot_stop_ids { logits[s] += eot_bias; } }
        next_token = sampler.sample(logits);
        sampler.observe(next_token);
    }

    if eot_dbg {
        let ranks: Vec<String> = eot_ranks.iter().take(40).map(|r| r.to_string()).collect();
        let gids: Vec<String> = eot_gen.iter().take(40).map(|g| g.to_string()).collect();
        tracing::info!("EOT-DEBUG: stop_ids={:?} ngen={} stop_rank_per_step=[{}] gen_ids=[{}]",
            eot_stop_ids, eot_gen.len(), ranks.join(","), gids.join(","));
    }

    let flushed = dec_buf.flush();
    if !flushed.is_empty() {
        let text = String::from_utf8_lossy(&flushed).into_owned();
        answer_text.push_str(&text);
        let payload = serde_json::to_string(&ChatDelta { delta: text, chat_id })
            .unwrap_or_default();
        let _ = tx.blocking_send(Ok(Event::default().data(payload)));
    }

    // ── GEODESIC F3 CAPTURE — persist the pair + meta (ADR-003 §5, G-F3-CAPTURE).
    // File f3_<chat_id>.bin: magic "F3P1" + E:u32 + nframes:u32 + pad:u32, then
    // nframes×E LE f32 (frame 0 = last-prompt-token state, frame 1 = first-answer-
    // token state when captured). One f3_meta.jsonl row per turn (the harness joins
    // by `user` text); f3_env.txt dumped once per serve (the re-baseline law:
    // receipts carry the full SP_* env).
    if let Some(ref f3d) = f3_dir {
        if !f3_prompt.is_empty() {
            let _ = std::fs::create_dir_all(f3d);
            let envp = std::path::Path::new(f3d).join("f3_env.txt");
            if !envp.exists() {
                let mut ed = String::new();
                for (k, v) in std::env::vars() {
                    if k.starts_with("SP_") || k.starts_with("CUBLAS") {
                        ed.push_str(&format!("{k}={v}\n"));
                    }
                }
                let _ = std::fs::write(&envp, ed);
            }
            let has_first = f3_first.len() == F3_E;
            let nframes: u32 = if has_first { 2 } else { 1 };
            let mut buf = Vec::with_capacity(16 + F3_E * nframes as usize * 4);
            buf.extend_from_slice(b"F3P1");
            buf.extend_from_slice(&(F3_E as u32).to_le_bytes());
            buf.extend_from_slice(&nframes.to_le_bytes());
            buf.extend_from_slice(&0u32.to_le_bytes());
            for &x in &f3_prompt { buf.extend_from_slice(&x.to_le_bytes()); }
            if has_first { for &x in &f3_first { buf.extend_from_slice(&x.to_le_bytes()); } }
            let _ = std::fs::write(
                std::path::Path::new(f3d).join(format!("f3_{chat_id}.bin")), buf);
            let mode = if recalled.is_some() {
                std::env::var("SP_RECALL_L5_PROMPT").unwrap_or_else(|_| "recite".into())
            } else { "clean".to_string() };
            let meta = serde_json::json!({
                "chat_id": chat_id,
                "mode": mode,
                "recalled": recalled.as_ref().map(|(n, c)|
                    serde_json::json!({"ep": n, "cos_milli": c})),
                "user": raw_user.as_deref().unwrap_or(""),
                "answer": answer_text,
                "n_gen": committed_gen.len(),
                "frames": nframes,
            });
            use std::io::Write as _;
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true)
                .open(std::path::Path::new(f3d).join("f3_meta.jsonl")) {
                let _ = writeln!(f, "{meta}");
            }
            tracing::info!("F3-CAPTURE: f3_{}.bin frames={} mode={} recalled={:?}",
                chat_id, nframes, meta["mode"], recalled);
        }
    }

    // ── B3-JUDGE GROUNDING CHECK (CPU-side, telemetry) ──────────────────────────
    // If the generative judge injected a memory, verify the synthesized answer
    // actually USES the memory's salient nouns. If the model ignored the injected
    // payload and answered parametrically, flag a parametric-hallucination (the
    // discrete-substrate reject the operator specified). Telemetry-only: it does
    // not alter the (already-streamed) answer; it surfaces the discrepancy in the
    // daemon log so a wrong/ignored recall is auditable.
    if let Some((ref ep_name, ref mem_text)) = judge_ground {
        let ans_lc = answer_text.to_lowercase();
        // salient tokens = the memory's words >=4 chars, minus common stopwords.
        const STOP: &[&str] = &["the","and","for","that","with","this","from","are",
            "was","were","authorizes","requires","access","code","recovery","using",
            "your","you","have","will","when","which","what","into","over","each"];
        let salient: Vec<String> = mem_text.split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() >= 4)
            .map(|w| w.to_lowercase())
            .filter(|w| !STOP.contains(&w.as_str()))
            .collect();
        let n_salient = salient.len();
        let n_used = salient.iter().filter(|w| ans_lc.contains(w.as_str())).count();
        let grounded = n_salient == 0 || n_used > 0;
        if grounded {
            tracing::info!("B3-JUDGE grounding: OK '{}' used {}/{} salient mem tokens", ep_name, n_used, n_salient);
        } else {
            tracing::warn!("B3-JUDGE grounding: PARAMETRIC-HALLUCINATION FLAG '{}' -- answer used 0/{} salient mem tokens (model ignored injected memory)", ep_name, n_salient);
        }
    }

    let is_cancelled = cancel_child.load(Ordering::Relaxed) != 0;
    if is_cancelled {
        let _ = tx.blocking_send(Ok(Event::default().event("cancelled").data("{}")));
        let _ = app.events_tx.send(DaemonEvent::Chat { chat_id, status: "cancelled" });
    } else {
        let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));
        let _ = app.events_tx.send(DaemonEvent::Chat { chat_id, status: "done" });
    }

    // ── B4 NIGHTSHIFT: between-turn memory CONSOLIDATION ────────────────────────
    // After the assistant reply has finished decoding for THIS turn, capture the
    // raw USER message as a LIVE episode so the W_c head can self-select it on a
    // LATER turn (the chat GROWS its memory). The capture is a position-0 STANDALONE
    // prefill on a SCRATCH session (NOT a read of the live conversation cache) so the
    // episode-K has the same provenance as the curated registry-K the head trained on.
    // Env-gated SP_B4_NIGHTSHIFT=1; unset ⇒ this whole hook is skipped ⇒ byte-identical
    // null floor. We still hold the resident-cache Mutex (`guard`/`handle` from the top
    // of this fn) for the WHOLE capture, so no concurrent decode races the scratch open.
    // A capture failure logs a warning and does NOT break the (already-sent) response.
    // TODO(B4-v2): admit via the teacher-forced ablation oracle (collapse < TAU=-8)
    // before append — v1 captures EVERY sufficiently-long user turn (no relevance gate).
    // LAYER-3 DECIDE: carries the just-captured (episode_name, text) out of the
    // NIGHTSHIFT block to the DECIDE pass below, so the model can judge whether the
    // new fact supersedes an existing memory. None unless a capture happened this turn.
    let mut decide_new: Option<(String, String)> = None;
    if std::env::var("SP_B4_NIGHTSHIFT").ok().as_deref() == Some("1") {
        // B4-v2 ADMISSION GATE (interim): only ASSERTIONS become memories. Skip a turn
        // that was (a) answered FROM memory (recalled.is_some() => a query consuming a
        // memory, not a new fact) or (b) interrogative (ends with '?'). This kills the
        // junk-question capture that let a true-foreign query false-recall a stored
        // question (the live-smoke regression). The rigorous gate is the teacher-forced
        // ablation oracle (collapse < TAU=-8), which ALSO rejects parametric facts — the
        // TODO above; this cheap gate is the safety floor it will subsume.
        // A forget command turn must NEVER become a memory (it removed one; storing
        // "Forget the secret vault code." re-pollutes the registry). Exclude both the
        // matched case (forget_done) and any forget phrasing (matched or not).
        let is_forget_turn = raw_user.as_ref().map(|s| {
            let l = s.to_lowercase();
            l.contains("forget") || l.contains("delete that") || l.contains("erase")
        }).unwrap_or(false);
        // LAYER-3: capture STATEMENTS even when a related memory was recalled (so a
        // superseding fact like "my favorite color is now green" IS stored and the
        // DECIDE pass below can retire the old one). Skip interrogatives via wh-word /
        // '?' detection (the spider-question false-recall fix), NOT the recalled.is_none()
        // coupling that previously blocked the supersede case. N2 stays LOOSE by design.
        let lc = raw_user.as_ref().map(|s| s.trim().to_lowercase()).unwrap_or_default();
        let first_word = lc.split_whitespace().next().unwrap_or("");
        // Skip questions AND request-imperatives ("tell me about X", "describe...") --
        // they are not facts to store. wh-words + '?' + common request verbs/prefixes.
        let is_question = lc.ends_with('?')
            || lc.starts_with("tell me") || lc.starts_with("do you") || lc.starts_with("can you")
            || matches!(first_word,
                "what"|"who"|"whom"|"whose"|"where"|"when"|"why"|"how"|"which"
                |"tell"|"describe"|"list"|"show"|"explain"|"summarize"|"recall"|"remind");
        let admit = !forget_done
            && !is_forget_turn
            && !is_question
            && !lc.is_empty();
        if let Some(text) = raw_user.as_ref().map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty()).filter(|_| admit) {
            // (a) tokenize the RAW user content (no chat template — match the curator).
            // PROVENANCE FIX (B4-v3): byte-match the curator's sp_tok_enc EXACTLY. The
            // curator captured each needle from a `.txt` file that ends in a trailing
            // newline ("...6428.\n") with the gemma4 C encoder's FORCED leading BOS
            // (id 2) KEPT — yielding e.g. npos=22 for ep_n_div_000 (tok[0]=2 BOS,
            // tok[-1]=107 newline). The prior live path trimmed the text (no trailing
            // newline) AND stripped the BOS, losing BOTH boundary tokens (npos=20) — a
            // 2-token tokenization mismatch that produced a DIFFERENT ep.k and collapsed
            // the W_c score (live 0.084 vs curated 9.858 on identical text). So: append
            // the trailing "\n" before encoding and DO NOT strip the auto-prepended BOS.
            let text_nl = format!("{text}\n");
            match app.tokenizer.encode(&text_nl) {
                Ok(toks) => {
                    // KEEP the forced-BOS (id 2): the curator's ep.tok starts with it.
                    let ntok = toks.len();
                    if ntok < 4 {
                        tracing::info!("B4-NIGHTSHIFT: skip (too short, ntok={ntok})");
                    } else {
                        // (b) qm = the session's borrowed qwen3_model* (shares loaded weights).
                        let qm = {
                            let mut sguard = app.session.as_ref().expect("L1 session unavailable (qwen36 lane)").lock().unwrap();
                            let sraw = sguard.raw_ptr() as *mut sp_daemon::ffi_l1::sp_session;
                            // SAFETY: session is locked + valid for this borrow.
                            (unsafe { sp_daemon::ffi_l1::sp_session_qwen3_model(sraw) }) as *const std::ffi::c_void
                        };
                        if qm.is_null() {
                            tracing::warn!("B4-NIGHTSHIFT: sp_session_qwen3_model NULL — capture skipped");
                        } else {
                            // (c) Option-2 PROVENANCE FIX: capture through the SAME batched
                            // `gemma4_decode_cuda` forward the curator used. It writes
                            // ep.k/ep.v/ep.mf into a temp episode dir under the engine, so the live
                            // episode's K is BYTE-COMPATIBLE with the curated registry and the deployed
                            // W_c head selects it with ZERO retraining (no scratch per-token prefill,
                            // no K-norm calibration). gemma4_decode_cuda reads SP_XBAR_SWA_RING /
                            // SP_XBAR_SWA_W / SP_BYTEEXACT — NONE of which the nightshift launcher sets
                            // (it sets SP_DAEMON_KVDECODE_RING_W, a different var), so this runs in the
                            // curator's clean full-cache float config and matches the curated ep.k. We
                            // hold the resident-cache Mutex (`guard`/`handle`) for the whole capture, so
                            // the SP_XBAR_RECALL_WRITE set+unset inside the glue is serialized.
                            let idx = { app.nightshift.read().unwrap().len() };
                            let engine_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                                .parent().and_then(|p| p.parent())
                                .unwrap_or_else(|| std::path::Path::new("."));
                            let dir = engine_root
                                .join("_nightshift_live")
                                .join(format!("ep_live_{:03}", idx));
                            let dir_str = dir.to_string_lossy().to_string();
                            if let Err(e) = std::fs::create_dir_all(&dir) {
                                tracing::warn!("B4-NIGHTSHIFT: mkdir {dir_str} failed: {e} — no episode appended");
                            } else {
                                // SAFETY: qm valid (session held above); we hold the resident-cache
                                // Mutex so no concurrent device forward / SP_XBAR_RECALL_WRITE env race.
                                { let p = dir.join("ep.txt"); let _ = std::fs::write(&p, &text); let q = dir.join("ep.tok"); let _ = std::fs::write(&q, toks.iter().map(|t| t.to_string()).collect::<Vec<_>>().join("\n")); }
                                let cap_rc = unsafe { kv::capture_batched(qm, &toks, &dir_str) };
                                match cap_rc {
                                    Err(e) => tracing::warn!("B4-NIGHTSHIFT: batched capture failed: {e} — no episode appended"),
                                    Ok(()) => {
                                        // Load the just-written ep.k as global-K with the SAME extractor
                                        // the curated registry uses (byte-compatible by construction).
                                        match sp_daemon::recall::load_episode_global_k(&dir_str, ntok as i32) {
                                            None => tracing::warn!(
                                                "B4-NIGHTSHIFT: load_episode_global_k('{dir_str}') -> None — no episode appended"),
                                            Some((gk, ng)) => {
                                                // G-INT-2 STEP 1: compute the LIVE episode's REAL C2 sig
                                                // from its just-captured global-K, using the SAME Projection
                                                // (SEED/R_BITS/HD) the curated registry sigs come from, so live
                                                // and curated sigs are directly Hamming-comparable. npos here is
                                                // the actual position count packed in gk (load_episode_global_k
                                                // clamps ntok to the captured Pmax), derived from gk.len().
                                                let npos_sig = if ng > 0 { gk.len() / (ng * sp_daemon::recall::HD) } else { 0 };
                                                let sig = if npos_sig > 0 {
                                                    app.recall_proj.signature(&gk, ng, npos_sig)
                                                } else { [0u64; 4] };
                                                // B4-SEAL: mint the L5 query-key at capture time so the GROWN
                                                // episode is immediately visible to the deployed L5 selector
                                                // (and to load_episode_l5key after restart via <dir>/ep.l5).
                                                let l5k = mint_live_ep_l5(qm, &toks, &dir).unwrap_or_default();
                                                // N3 PERSIST: append this live episode to the active registry
                                                // file so it survives a daemon restart (default-off
                                                // SP_NIGHTSHIFT_PERSIST=1 = null floor). Done here, before `sig`
                                                // is moved into the Episode below. On restart load_registry reads
                                                // it back (curated path; ep.k/ep.v/ep.mf already on disk at dir).
                                                if std::env::var("SP_NIGHTSHIFT_PERSIST").ok().as_deref() == Some("1") {
                                                    if let Ok(reg_path) = std::env::var("SP_RECALL_REGISTRY") {
                                                        let pidx = app.nightshift.read().unwrap().len();
                                                        let sig_hex = format!("{:016x}{:016x}{:016x}{:016x}", sig[3], sig[2], sig[1], sig[0]);
                                                        let line = serde_json::json!({
                                                            "name": format!("ep_live_{:03}", pidx),
                                                            "dir": dir_str.clone(), "npos": ntok as i32,
                                                            "topic": text.clone(), "text": text.clone(), "sig_bits": sig_hex,
                                                        }).to_string();
                                                        use std::io::Write as _;
                                                        match std::fs::OpenOptions::new().create(true).append(true).open(&reg_path) {
                                                            Ok(mut f) => { let _ = writeln!(f, "{line}");
                                                                tracing::info!("B4-NIGHTSHIFT-PERSIST: appended ep_live_{:03} -> {}", pidx, reg_path); }
                                                            Err(e) => tracing::warn!("B4-NIGHTSHIFT-PERSIST: append {} failed: {e}", reg_path),
                                                        }
                                                    }
                                                }
                                                let mut ns = app.nightshift.write().unwrap();
                                                let idx = ns.len();
                                                tracing::info!(
                                                    "B4-NIGHTSHIFT: ep_live_{:03} C2-sig = {:016x}... (was [0;4])",
                                                    idx, sig[0]);
                                                let topic: String = text.chars().take(40).collect();
                                                // tokens: Some(toks) so the recall side still injects via
                                                // kv::inject_tokens (unchanged); ep.k/ep.v/ep.mf now also
                                                // exist on disk at `dir` if a future path prefers kv::replay.
                                                ns.push(sp_daemon::recall::Episode {
                                                    name: format!("ep_live_{:03}", idx),
                                                    dir: dir_str.clone(),
                                                    npos: ntok as i32,
                                                    topic,
                                                    text: text.clone(),
                                                    sig,
                                                    gk,
                                                    gk_ng: ng,
                                                    tokens: Some(toks),
                                                    l5key: l5k, // B4-SEAL: minted at capture; empty only on mint failure
                                                });
                                                let total = ns.len();
                                                tracing::info!(
                                                    "B4-NIGHTSHIFT: consolidated (batched) -> 'ep_live_{:03}' npos={} ng={} — registry now has {} live episode(s)",
                                                    idx, ntok, ng, total
                                                );
                                                // LAYER-3: hand the new fact to the DECIDE pass below.
                                                decide_new = Some((format!("ep_live_{:03}", idx), text.clone()));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                Err(e) => tracing::warn!("B4-NIGHTSHIFT: tokenize failed: {e} — no episode appended"),
            }
        }
    }

    // ── LAYER-3: DECIDE — the model curates its own memory ──────────────────────────
    // When NIGHTSHIFT just captured a new fact, show the model its RELATED existing
    // memories and let IT decide whether the new one supersedes/contradicts an old one
    // (e.g. "my favorite color is now green" retires "...is blue"). The model's verdict
    // — not a dedup rule — drives the removal: this is the emergent agency layer. Runs
    // AFTER the response streamed (cache free) and AFTER capture (the new episode
    // exists). SP_DECIDE=1; unset ⇒ no model-call, no removal = byte-identical null floor.
    if std::env::var("SP_DECIDE").ok().as_deref() == Some("1") {
        if let Some((new_name, new_text)) = decide_new {
            use sp_daemon::recall;
            // Related existing memories (share subject with the new fact), excluding the
            // just-captured episode. token_overlap >= 0.3 ≈ "about the same thing".
            let mut cands: Vec<(f32, String)> = Vec::new();
            if let Some(reg) = app.recall_registry.as_ref() {
                for ep in reg.iter() {
                    if ep.name == new_name || ep.text == new_text { continue; }
                    let ov = recall::token_overlap(&new_text, &ep.text);
                    if ov >= 0.3 && !cands.iter().any(|(_, ct)| ct == &ep.text) { cands.push((ov, ep.text.clone())); }
                }
            }
            {
                let ns = app.nightshift.read().unwrap();
                for ep in ns.iter() {
                    if ep.name == new_name || ep.text == new_text { continue; }
                    let ov = recall::token_overlap(&new_text, &ep.text);
                    if ov >= 0.3 && !cands.iter().any(|(_, ct)| ct == &ep.text) { cands.push((ov, ep.text.clone())); }
                }
            }
            cands.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            cands.truncate(5);
            if !cands.is_empty() {
                let mut decided = false; // set true when stage-1 supersede forgets
                // Present the new fact + numbered related memories; the model picks.
                let mut listing = String::new();
                for (i, (_, t)) in cands.iter().enumerate() {
                    listing.push_str(&format!("{}. \"{}\"\n", i + 1, t));
                }
                let prompt = format!(
                    "A NEW fact was just stated:\n\"{}\"\nHere are earlier memories:\n{}\nIs the NEW fact a REPLACEMENT for one of these memories -- does it state a new value for the SAME attribute, so that the NEW fact and that memory CANNOT both be true at the same time (for example a favorite color changing, or a current home city changing)? Only then give that memory's number. If the NEW fact is about a DIFFERENT attribute and can be true ALONGSIDE the memories (for example a person's job AND the city they live in are both true), answer NONE.\nReply with the memory number that is replaced, or NONE.",
                    new_text, listing);
                let dmsgs = vec![Message { role: "user".to_string(), content: prompt }];
                if let Ok(dtoks) = app.tokenizer.apply_template_ids(&dmsgs) {
                    if dtoks.len() >= 2 {
                        // Force the answer format + prevent template echo: prefill "CHANGED="
                        // into the model turn (the judge's prefill-TAG trick). The model then
                        // completes a number or NONE; reply is seeded with the forced prefix.
                        let mut dtoks = dtoks;
                        let mut reply = String::new();
                        if let Ok(tt) = app.tokenizer.encode("CHANGED=") {
                            let tt: Vec<i32> = tt.into_iter().filter(|&t| t != 2).collect();
                            if !tt.is_empty() { dtoks.extend_from_slice(&tt); reply.push_str("CHANGED="); }
                        }
                        let _ = unsafe { kv::reset(handle) };
                        let (dhead, dlast) = dtoks.split_at(dtoks.len() - 1);
                        let ok = dhead.is_empty() || unsafe { kv::prefill(handle, dhead) }.is_ok();
                        if ok {
                            let mut tok = dlast[0];
                            let turn_stops = app.tokenizer.turn_stop_ids();
                            let suppress = app.tokenizer.suppress_token_ids();
                            for _ in 0..12 {
                                if unsafe { kv::decode_step(handle, tok, logits) }.is_err() { break; }
                                for &sid in &suppress { let u = sid as usize; if u < logits.len() { logits[u] = f32::NEG_INFINITY; } }
                                let mut bi = 0usize; let mut bv = f32::NEG_INFINITY;
                                for (i, &v) in logits.iter().enumerate() { if v > bv { bv = v; bi = i; } }
                                let g = bi as i32;
                                if app.tokenizer.eos_ids.contains(&g) || turn_stops.contains(&g) { break; }
                                reply.push_str(&String::from_utf8_lossy(app.tokenizer.decode_token(g)));
                                tok = g;
                                if reply.contains('\n') { break; }
                            }
                        }
                        // Restore a clean cache (the next turn re-prefills from scratch).
                        let _ = unsafe { kv::reset_cold(handle) };
                        tracing::info!("LAYER-3 DECIDE: new=\"{}\" {} cand(s) -> reply=\"{}\"",
                            new_text.chars().take(50).collect::<String>(), cands.len(), reply.trim());
                        // Parse CHANGED=<n>; execute the model's choice via the forget removal
                        // (drop from the live set + rewrite the persisted registry). NONE / any
                        // unparseable reply ⇒ no removal (safe default).
                        let ru = reply.to_uppercase();
                        let n: Option<usize> = if ru.contains("NONE") { None } else {
                            ru.chars()
                                .skip_while(|c| !c.is_ascii_digit())
                                .take_while(|c| c.is_ascii_digit())
                                .collect::<String>().parse().ok()
                        };
                        {
                            if let Some(n) = n {
                                if n >= 1 && n <= cands.len() {
                                    let victim = cands[n - 1].1.clone();
                                    { let mut ns = app.nightshift.write().unwrap(); ns.retain(|e| e.text != victim); }
                                    if let Ok(reg_path) = std::env::var("SP_RECALL_REGISTRY") {
                                        if let Ok(content) = std::fs::read_to_string(&reg_path) {
                                            let kept: Vec<&str> = content.lines().filter(|line| {
                                                match serde_json::from_str::<serde_json::Value>(line) {
                                                    Ok(v) => v.get("text").and_then(|x| x.as_str()) != Some(victim.as_str()),
                                                    Err(_) => !line.trim().is_empty(),
                                                }
                                            }).collect();
                                            let mut out = kept.join("\n"); if !out.is_empty() { out.push('\n'); }
                                            let _ = std::fs::write(&reg_path, out);
                                        }
                                    }
                                    tracing::info!("LAYER-3 DECIDE: the model chose to FORGET #{} -> \"{}\" (superseded by the new fact)", n, victim);
                                    decided = true;
                                }
                            }
                        }
                    }
                }
                // ── STAGE 2: MERGE — consolidate two partial facts into ONE synthesized
                // truth. Runs only if stage-1 found NO supersede. Compares the new fact
                // with the single best-overlap candidate; if the model judges they describe
                // the SAME subject, it returns the combined fact, and the daemon drops BOTH
                // old episodes and captures the one consolidation (the operator's "holy grail").
                if !decided {
                    let cand0 = cands[0].1.clone();
                    let mprompt = format!(
                        "Two facts may describe the SAME thing and belong together as ONE memory:\nA: \"{}\"\nB: \"{}\"\nIf A and B are about the same subject and should be combined, reply with the single complete combined fact on one line, prefixed exactly: MERGE:: <combined fact>\nIf they are about DIFFERENT things, reply NONE.",
                        new_text, cand0);
                    let mmsgs = vec![Message { role: "user".to_string(), content: mprompt }];
                    if let Ok(mtoks) = app.tokenizer.apply_template_ids(&mmsgs) {
                        if mtoks.len() >= 2 {
                            let _ = unsafe { kv::reset(handle) };
                            let (mhead, mlast) = mtoks.split_at(mtoks.len() - 1);
                            let ok = mhead.is_empty() || unsafe { kv::prefill(handle, mhead) }.is_ok();
                            let mut mreply = String::new();
                            if ok {
                                let mut tok = mlast[0];
                                let turn_stops = app.tokenizer.turn_stop_ids();
                                let suppress = app.tokenizer.suppress_token_ids();
                                for _ in 0..40 {
                                    if unsafe { kv::decode_step(handle, tok, logits) }.is_err() { break; }
                                    for &sid in &suppress { let u = sid as usize; if u < logits.len() { logits[u] = f32::NEG_INFINITY; } }
                                    let mut bi = 0usize; let mut bv = f32::NEG_INFINITY;
                                    for (i, &v) in logits.iter().enumerate() { if v > bv { bv = v; bi = i; } }
                                    let g = bi as i32;
                                    if app.tokenizer.eos_ids.contains(&g) || turn_stops.contains(&g) { break; }
                                    mreply.push_str(&String::from_utf8_lossy(app.tokenizer.decode_token(g)));
                                    tok = g;
                                    if mreply.contains('\n') { break; }
                                }
                            }
                            let _ = unsafe { kv::reset_cold(handle) };
                            let mtrim = mreply.trim().to_string();
                            tracing::info!("LAYER-3 MERGE: A=\"{}\" B=\"{}\" -> reply=\"{}\"",
                                new_text.chars().take(40).collect::<String>(),
                                cand0.chars().take(40).collect::<String>(), mtrim);
                            // Parse "MERGE:: <combined>" (case-insensitive). NONE / no marker /
                            // empty combined ⇒ keep both (safe default, no removal).
                            let lower = mtrim.to_lowercase();
                            if let Some(mp) = lower.find("merge") {
                                if !lower[..mp].contains("none") {
                                    let rest = mtrim.get(mp + 5..).unwrap_or("");
                                    let combined = rest
                                        .trim_start_matches(|c: char| c == ':' || c == '=' || c == '-' || c == '>' || c == ' ')
                                        .trim().trim_matches('"').trim().to_string();
                                    if combined.chars().count() >= 4 {
                                        // Drop BOTH old episodes (the just-captured new fact + the
                                        // candidate) from the live set and the persisted registry...
                                        let drop_a = new_text.clone();
                                        let drop_b = cand0.clone();
                                        { let mut ns = app.nightshift.write().unwrap();
                                          ns.retain(|e| e.text != drop_a && e.text != drop_b); }
                                        if let Ok(reg_path) = std::env::var("SP_RECALL_REGISTRY") {
                                            if let Ok(content) = std::fs::read_to_string(&reg_path) {
                                                let kept: Vec<&str> = content.lines().filter(|line| {
                                                    match serde_json::from_str::<serde_json::Value>(line) {
                                                        Ok(v) => { let t = v.get("text").and_then(|x| x.as_str());
                                                            t != Some(drop_a.as_str()) && t != Some(drop_b.as_str()) },
                                                        Err(_) => !line.trim().is_empty(),
                                                    }
                                                }).collect();
                                                let mut out = kept.join("\n"); if !out.is_empty() { out.push('\n'); }
                                                let _ = std::fs::write(&reg_path, out);
                                            }
                                        }
                                        // ...and capture the single synthesized consolidation.
                                        let okc = capture_live_episode(app, &combined);
                                        tracing::info!("LAYER-3 MERGE: consolidated \"{}\" + \"{}\" -> \"{}\" (capture {})",
                                            drop_a.chars().take(30).collect::<String>(),
                                            drop_b.chars().take(30).collect::<String>(),
                                            combined.chars().take(50).collect::<String>(),
                                            if okc { "ok" } else { "FAILED" });
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // PERSIST-KV: record the cache's new contents (this turn's prompt + the tokens we committed)
    // so the next turn can reuse it as a strict prefix and prefill only its new suffix. Gated by
    // persist_kv (the plain decode path) -- in that config NO side-call above touched the cache, so
    // [tokens ++ committed_gen] mirrors it exactly. Any other config never sets persist_kv, so the
    // committed state is simply left stale and the next turn's strict-prefix+position guard rejects
    // it -> full reset+prefill (the byte-identical null floor). Cleared if anything went wrong is
    // unnecessary: the guard already fails closed.
    if persist_kv {
        let mut c = KV_COMMITTED.lock().unwrap();
        c.clear();
        // RECALL EXTENSION: re-arm the reusable prefix ONLY when the cache is the pristine plain
        // prompt + generated -- i.e. no episode was injected for synthesis this turn. A PICK
        // (recalled.is_some()) leaves the cache holding the episode K/V at [dpos,dpos+npos), which
        // a token-sequence committed cannot represent; leaving it cleared makes the next turn
        // reset + full-prefill (clearing the injected episode) -- correct, just not O(1) that turn.
        if recalled.is_none() {
            c.reserve(tokens.len() + committed_gen.len());
            c.extend_from_slice(tokens);
            c.extend_from_slice(&committed_gen);
        }
    }

    sessions.remove(chat_id);
}

// A2: `fn argmax` moved to `crate::sampler::argmax` (the temp=0 null floor);
// both decode loops now go through `Sampler::sample`.

// ── POST /v1/capture (B4 SCALE-UP curator) ───────────────────────────────
// Batch-capture an episode's ep.k/ep.v/ep.mf by running ONE curated forward on the
// RESIDENT model (reuses loaded weights — no 9GB reload per needle). Tokenizes the
// raw needle text EXACTLY as the B4 provenance fix does: KEEP the forced-BOS (id 2)
// AND append a trailing "\n" before encoding — this is why live==curated (the head
// scores ep.k identically). Calls kv::capture_batched under the kvdecode/session
// lock. Default behaviour unchanged for all other endpoints. Returns {ok, npos} or
// {error}. Used by tools/xbar_lsh/b4_capture_driver.py to populate a corpus in ONE
// model load.
#[derive(serde::Deserialize)]
pub struct CaptureReq {
    pub text: String,
    pub out_dir: String,
}

pub async fn v1_capture(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CaptureReq>,
) -> Response {
    use sp_daemon::cuda_kvdecode_dispatch as kv;
    let app = state.clone();
    let text = req.text.trim().to_string();
    let out_dir = req.out_dir.clone();
    if text.is_empty() {
        return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error":"empty text"}))).into_response();
    }
    // PROVENANCE: byte-match the curator's sp_tok_enc — BOS kept + trailing newline.
    let text_nl = format!("{text}\n");
    let toks = match app.tokenizer.encode(&text_nl) {
        Ok(t) => t,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR,
                          Json(serde_json::json!({"error": format!("tokenize: {e}")}))).into_response(),
    };
    let ntok = toks.len();
    if ntok < 4 {
        return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error":"too short","npos":ntok}))).into_response();
    }
    // Run the blocking CUDA capture on a blocking thread; hold the session lock for
    // the qm borrow + the whole capture (serializes SP_XBAR_RECALL_WRITE env inside).
    let res = task::spawn_blocking(move || -> Result<usize, String> {
        let qm = {
            let mut sguard = app.session.as_ref().expect("L1 session unavailable (qwen36 lane)").lock().unwrap();
            let sraw = sguard.raw_ptr() as *mut sp_daemon::ffi_l1::sp_session;
            (unsafe { sp_daemon::ffi_l1::sp_session_qwen3_model(sraw) }) as *const std::ffi::c_void
        };
        if qm.is_null() {
            return Err("sp_session_qwen3_model NULL".to_string());
        }
        std::fs::create_dir_all(&out_dir).map_err(|e| format!("mkdir {out_dir}: {e}"))?;
        unsafe { kv::capture_batched(qm, &toks, &out_dir) }?;
        Ok(ntok)
    }).await;
    match res {
        Ok(Ok(npos)) => (StatusCode::OK, Json(serde_json::json!({"ok":true,"npos":npos}))).into_response(),
        Ok(Err(e))   => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error":e}))).into_response(),
        Err(e)       => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error":format!("join: {e}")}))).into_response(),
    }
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
        let guard = state.session.as_ref().expect("L1 session unavailable (qwen36 lane)").lock().unwrap();
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
