use std::fs;
use std::path::PathBuf;
use std::sync::{atomic::{AtomicI32, AtomicU64, AtomicBool}, Arc, Mutex};
use std::time::Instant;
use dashmap::DashMap;
use tokio::sync::mpsc;

use ed25519_dalek::SigningKey;
use rand::rngs::OsRng;
use tracing::info;

use sp_daemon::network::quic_shard::{run_garner_loop, SpQuicCoordinator};

/// NTT transform size for CRT residue reconstruction.
/// Matches the test topology (N=128). Tuned to model layer width in BLOCK-SYNC phase.
const QUIC_NTT_N: u32 = 128;

fn pid_file() -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push("sp-daemon.pid");
    p
}

// ── Commands (called from main, synchronous) ───────────────────────────────

/// Spawn the daemon inner process detached from the current session, write
/// the PID file, and return so the calling process can exit.
pub fn cmd_start(model: &str, tokenizer: &str, draft_model: &str, draft_tokenizer: &str, quic_port: u16) {
    let exe = std::env::current_exe().expect("current_exe");
    let quic_port_s = quic_port.to_string();
    let mut cmd = std::process::Command::new(&exe);
    cmd.args([
        "--daemon-inner",
        "--model",         model,
        "--tokenizer",     tokenizer,
        "--draft-model",   draft_model,
        "--draft-tokenizer", draft_tokenizer,
        "--quic-port",     &quic_port_s,
    ]);

    // Windows: DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP so the child is
    // not attached to the parent's console or process group.
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const DETACHED_PROCESS: u32 = 0x0000_0008;
        const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
        cmd.creation_flags(DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP);
    }

    let child = cmd.spawn().expect("failed to spawn daemon inner process");
    let pid = child.id();
    // Write PID before the parent exits so `stop` can locate the process.
    fs::write(pid_file(), pid.to_string()).expect("failed to write PID file");
    eprintln!("sp-daemon started (pid={pid}, pid_file={})", pid_file().display());
}

/// Send SIGTERM (Unix) or taskkill (Windows) to the running daemon, then
/// remove the PID file. Waiting for clean sp_session_destroy is handled by
/// the daemon's graceful-shutdown handler in `run_inner`.
pub fn cmd_stop() {
    let pid_path = pid_file();
    let pid_str = fs::read_to_string(&pid_path).unwrap_or_else(|_| {
        eprintln!("no PID file at {} — is sp-daemon running?", pid_path.display());
        std::process::exit(1);
    });
    let pid: u32 = pid_str.trim().parse().expect("corrupt PID file");
    send_term(pid);
    fs::remove_file(&pid_path).ok();
    eprintln!("sp-daemon stop signal sent to pid={pid}");
}

/// No-op for v0. Phase 2-L3.FG will introduce hot-reload of the model.
pub fn cmd_reload() {
    eprintln!("sp-daemon reload: no-op for v0");
}

// ── Inner daemon ───────────────────────────────────────────────────────────

/// The actual long-lived server. Called when the process is the child spawned
/// by `cmd_start` (detected via `--daemon-inner` argv flag in main.rs).
pub async fn run_inner(model_path: &str, tok_path: &str, draft_model_path: &str, draft_tok_path: &str, quic_port: u16) {
    // Detach from the parent's controlling terminal on Unix.
    // On Windows, DETACHED_PROCESS in cmd_start already did this.
    #[cfg(unix)]
    unsafe { libc::setsid() };

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "sp_daemon=info".into()),
        )
        .init();

    info!("sp-daemon inner starting (model={model_path})");

    // ── Target model ──────────────────────────────────────────────────────
    let model = crate::session::SpModel::load(model_path, tok_path)
        .expect("sp_model_load failed — check SP_MODEL_PATH / SP_TOKENIZER_PATH");

    let arch = model.arch_info().expect("sp_model_arch failed");
    let vocab_size = arch.vocab_size as usize;
    info!("arch: vocab={} n_layers={} hidden={}", arch.vocab_size, arch.n_layers, arch.hidden_dim);

    let cancel_flag = Arc::new(AtomicI32::new(0));
    let session = crate::session::SpSession::create(&model, Arc::clone(&cancel_flag))
        .expect("sp_session_create failed");

    let pos = session.position().expect("sp_session_position");
    info!("L1 FFI OK — session_position={pos}");

    let tokenizer = crate::tokenizer::SptbTokenizer::build(&model, arch.arch_id)
        .expect("SptbTokenizer::build failed — check .sp-tokenizer blob");
    info!("tokenizer built: arch_id={} eos_ids={:?}", arch.arch_id, tokenizer.eos_ids);

    // ── Draft model (Phase 4-SPEC) — optional ─────────────────────────────
    let (draft_model, draft_session) = if !draft_model_path.is_empty() {
        let dm = crate::session::SpModel::load(draft_model_path, draft_tok_path)
            .expect("draft sp_model_load failed");
        let draft_arch = dm.arch_info().expect("draft sp_model_arch failed");
        info!("draft arch: vocab={} n_layers={} hidden={}",
            draft_arch.vocab_size, draft_arch.n_layers, draft_arch.hidden_dim);
        let d_cancel = Arc::new(AtomicI32::new(0));
        let ds = crate::session::SpSession::create(&dm, d_cancel)
            .expect("draft sp_session_create failed");
        (Some(dm), Some(Mutex::new(ds)))
    } else {
        info!("no draft model — single-model mode");
        (None, None)
    };

    let (events_tx, _) =
        tokio::sync::broadcast::channel::<crate::state::DaemonEvent>(64);

    let node_signing_key = SigningKey::generate(&mut OsRng);
    let mining_signing_key = node_signing_key.clone();
    let pubkey_hex: String = node_signing_key.verifying_key().to_bytes()
        .iter().fold(String::new(), |mut s, b| { use std::fmt::Write; let _ = write!(s, "{b:02x}"); s });
    info!("node pubkey: {pubkey_hex}");

    let inference_active = Arc::new(AtomicBool::new(false));
    let receipt_store    = Arc::new(Mutex::new(Vec::new()));

    let state = Arc::new(crate::state::AppState {
        model,
        session: Mutex::new(session),
        cancel_flag,
        draft_model,
        draft_session,
        sessions: crate::sessions::Sessions::new(),
        vocab_size,
        tokens_decoded: AtomicU64::new(0),
        started_at: Instant::now(),
        events_tx: events_tx.clone(),
        tokenizer,
        inference_active: inference_active.clone(),
        receipt_store:    receipt_store.clone(),
        node_signing_key,
        peer_map: Arc::new(DashMap::new()),
    });

    // ── Background PoUW mining task ────────────────────────────────────────
    tokio::spawn(crate::mining::run_mining_loop(
        mining_signing_key,
        inference_active,
        receipt_store,
        events_tx,
    ));

    // ── Operator Console (127.0.0.1:3000) ─────────────────────────────────
    tokio::spawn(crate::console::start_operator_console(Arc::clone(&state)));

    // ── QUIC DHT Coordinator ───────────────────────────────────────────────
    // Binds on 0.0.0.0:<quic_port> (LAN-accessible, unlike the HTTP server
    // which is loopback-only).  quic_port=0 disables the coordinator.
    if quic_port != 0 {
        let quic_addr: std::net::SocketAddr = ([0, 0, 0, 0], quic_port).into();
        match SpQuicCoordinator::bind(quic_addr).await {
            Ok(coordinator) => {
                info!("SP_INFO: QUIC Coordinator listening on {quic_addr}");
                // Garner results channel: receiver intentionally discarded here.
                // Results will be consumed by the inference pipeline in BLOCK-SYNC phase.
                let (garner_tx, _garner_rx) = mpsc::channel(64);
                tokio::spawn(run_garner_loop(
                    coordinator,
                    QUIC_NTT_N,
                    garner_tx,
                    Arc::clone(&state.peer_map),
                ));
            }
            Err(e) => {
                tracing::warn!("QUIC coordinator bind failed on {quic_addr}: {e} — DHT mesh disabled");
            }
        }
    } else {
        info!("QUIC mesh disabled (quic_port=0); pass --quic-port <N> or SP_QUIC_PORT=<N> to enable");
    }

    // ── HTTP server ────────────────────────────────────────────────────────
    let app = crate::server::build_router(Arc::clone(&state));

    // Bind only to loopback. 0.0.0.0 is explicitly forbidden:
    // single-user developer-device assumption (PPT-LAT-Roadmap §14.3.1).
    // LAN exposure and TLS are v1+ scope (Phase 2-L3.AUTH onwards).
    let addr: std::net::SocketAddr = ([127, 0, 0, 1], 8080).into();
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("bind 127.0.0.1:8080 — is another instance already running?");
    info!("listening on {addr}");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .expect("server error");

    // Drop order: session → model (fields drop in declaration order in AppState).
    // sp_session_destroy runs, then sp_model_unload. Both are synchronous.
    info!("sp-daemon inner stopped");
}

// ── Helpers ────────────────────────────────────────────────────────────────

/// Resolves on SIGTERM (Unix) or Ctrl-C (all platforms).
async fn shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        let mut sigterm = signal(SignalKind::terminate()).unwrap();
        tokio::select! {
            _ = sigterm.recv()            => {},
            _ = tokio::signal::ctrl_c()  => {},
        }
    }
    #[cfg(not(unix))]
    {
        tokio::signal::ctrl_c().await.ok();
    }
    tracing::info!("shutdown signal received");
}

#[cfg(unix)]
fn send_term(pid: u32) {
    unsafe { libc::kill(pid as libc::pid_t, libc::SIGTERM) };
}

#[cfg(windows)]
fn send_term(pid: u32) {
    // /F = force (necessary for detached console processes); GenerateConsoleCtrlEvent
    // for clean session drain is Phase 2-L3.FG scope.
    std::process::Command::new("taskkill")
        .args(["/PID", &pid.to_string(), "/F"])
        .spawn()
        .ok();
}

#[cfg(not(any(unix, windows)))]
fn send_term(_pid: u32) {
    eprintln!("send_term: not implemented for this platform");
}
