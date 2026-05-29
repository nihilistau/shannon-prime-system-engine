mod daemon;
#[cfg(target_os = "android")]
mod dsp_rpc;
// §3-HX Sprint J.5 — import (not move) the Sprint J loader from sp_dsp_smoke.
// #[path] keeps the loader files in their crate; `use crate::dsp_rpc::…` inside
// them resolves to this crate's dsp_rpc (signatures verified identical), so no
// duplicate FastRpcSession type and no smoke [lib] target. Tech-debt: the
// cross-tree path marks the loader's eventual real home (a shared crate).
#[cfg(target_os = "android")]
#[path = "../../sp_dsp_smoke/src/dsp_model.rs"]
mod dsp_model;
#[cfg(target_os = "android")]
#[path = "../../sp_dsp_smoke/src/kv_cache.rs"]
mod kv_cache;
// §3-HX Sprint J.5 — the L1 C-ABI inference path (ffi/session/forward) and the
// sieve-backed mining/spec/tokenizer modules do not cross-compile or link for
// aarch64-android (build.rs sets sp_no_link; the math-core C ABI is Phase
// 2-L3.FG scope). They are host-only; the android binary serves the DSP-loader
// surface instead (see daemon::run_inner android variant).
#[cfg(not(target_os = "android"))]
mod ffi;
#[cfg(not(target_os = "android"))]
mod mining;
mod routes;
mod server;
#[cfg(not(target_os = "android"))]
mod session;
#[cfg(not(target_os = "android"))]
mod sessions;
#[cfg(not(target_os = "android"))]
mod sieve_ffi;
#[cfg(not(target_os = "android"))]
mod spec;
mod state;
#[cfg(not(target_os = "android"))]
mod tokenizer;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "sp-daemon",
    about = "Shannon Prime L3 HTTP/SSE daemon (lat-phase-2-l3-core)",
    long_about = "Long-lived daemon that wraps the frozen L1 C ABI in an HTTP \
                  server on 127.0.0.1:8080. All four frontends (mobile, desktop, \
                  watch, CLI) attach to this process."
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Cmd>,

    // Internal flag: this process IS the inner daemon (spawned by `start`).
    #[arg(long, hide = true)]
    daemon_inner: bool,
    #[arg(long, default_value = "", hide = true)]
    model: String,
    #[arg(long, default_value = "", hide = true)]
    tokenizer: String,
    #[arg(long, default_value = "", hide = true)]
    draft_model: String,
    #[arg(long, default_value = "", hide = true)]
    draft_tokenizer: String,
    #[arg(long, default_value = "0", hide = true)]
    quic_port: u16,
    #[arg(long, default_value = "8080", hide = true)]
    port: u16,
    #[arg(long, default_value = "", hide = true)]
    peer: String,
    #[arg(long, default_value = "", hide = true)]
    peers: String,
}

#[derive(Subcommand)]
enum Cmd {
    /// Spawn the daemon process and exit.
    Start {
        /// Path to the .sp-model file (or set SP_MODEL_PATH).
        #[arg(long, env = "SP_MODEL_PATH")]
        model: String,
        /// Path to the .sp-tokenizer file (or set SP_TOKENIZER_PATH).
        #[arg(long, env = "SP_TOKENIZER_PATH")]
        tokenizer: String,
        /// Path to draft .sp-model for Phase 4-SPEC speculative decode (optional).
        #[arg(long, env = "SP_DRAFT_MODEL_PATH", default_value = "")]
        draft_model: String,
        /// Path to draft .sp-tokenizer (optional, required if --draft-model is set).
        #[arg(long, env = "SP_DRAFT_TOKENIZER_PATH", default_value = "")]
        draft_tokenizer: String,
        /// UDP port for the QUIC DHT mesh coordinator (set SP_QUIC_PORT or 0 to disable).
        #[arg(long, env = "SP_QUIC_PORT", default_value = "0")]
        quic_port: u16,
        /// TCP port for the main HTTP API server (set SP_HTTP_PORT).
        #[arg(long, env = "SP_HTTP_PORT", default_value = "8080")]
        port: u16,
        /// Dial this QUIC peer address on startup (e.g. 127.0.0.1:5000). Back-compat alias for --peers with one entry.
        #[arg(long, default_value = "")]
        peer: String,
        /// Comma-separated list of QUIC peer addresses to dial on startup (set SP_PEERS).
        #[arg(long, env = "SP_PEERS", default_value = "")]
        peers: String,
    },
    /// Stop the running daemon (sends SIGTERM / taskkill).
    Stop,
    /// Reload configuration (no-op for v0).
    Reload,
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    if cli.daemon_inner {
        daemon::run_inner(
            &cli.model, &cli.tokenizer,
            &cli.draft_model, &cli.draft_tokenizer,
            cli.quic_port, cli.port, &cli.peer, &cli.peers,
        ).await;
        return;
    }

    match cli.command {
        Some(Cmd::Start { model, tokenizer, draft_model, draft_tokenizer, quic_port, port, peer, peers }) =>
            daemon::cmd_start(&model, &tokenizer, &draft_model, &draft_tokenizer, quic_port, port, &peer, &peers),
        Some(Cmd::Stop) => daemon::cmd_stop(),
        Some(Cmd::Reload) => daemon::cmd_reload(),
        None => {
            eprintln!("Usage: sp-daemon <start|stop|reload>");
            std::process::exit(1);
        }
    }
}
