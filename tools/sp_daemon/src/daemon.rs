use std::fs;
use std::path::PathBuf;
use std::sync::{atomic::{AtomicI32, AtomicU64}, Arc, Mutex};
use std::time::Instant;

use tracing::info;

fn pid_file() -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push("sp-daemon.pid");
    p
}

// ── Commands (called from main, synchronous) ───────────────────────────────

/// Spawn the daemon inner process detached from the current session, write
/// the PID file, and return so the calling process can exit.
pub fn cmd_start(model: &str, tokenizer: &str) {
    let exe = std::env::current_exe().expect("current_exe");
    let mut cmd = std::process::Command::new(&exe);
    cmd.args(["--daemon-inner", "--model", model, "--tokenizer", tokenizer]);

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
pub async fn run_inner(model_path: &str, tok_path: &str) {
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

    // ── L1 FFI setup ──────────────────────────────────────────────────────
    // Load the .sp-model and create a persistent session.  cancel_flag is
    // L2-owned (Arc<AtomicI32>); raw pointer handed to L1 via sp_session_create.
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

    let (events_tx, _) =
        tokio::sync::broadcast::channel::<crate::state::ChatEvent>(64);

    let state = Arc::new(crate::state::AppState {
        model,
        session: Mutex::new(session),
        cancel_flag,
        sessions: crate::sessions::Sessions::new(),
        vocab_size,
        tokens_decoded: AtomicU64::new(0),
        started_at: Instant::now(),
        events_tx,
    });

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
