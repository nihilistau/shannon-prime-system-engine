mod daemon;
mod ffi;
mod mining;
mod routes;
mod server;
mod session;
mod sessions;
mod sieve_ffi;
mod spec;
mod state;
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
        daemon::run_inner(&cli.model, &cli.tokenizer, &cli.draft_model, &cli.draft_tokenizer).await;
        return;
    }

    match cli.command {
        Some(Cmd::Start { model, tokenizer, draft_model, draft_tokenizer }) =>
            daemon::cmd_start(&model, &tokenizer, &draft_model, &draft_tokenizer),
        Some(Cmd::Stop) => daemon::cmd_stop(),
        Some(Cmd::Reload) => daemon::cmd_reload(),
        None => {
            eprintln!("Usage: sp-daemon <start|stop|reload>");
            std::process::exit(1);
        }
    }
}
