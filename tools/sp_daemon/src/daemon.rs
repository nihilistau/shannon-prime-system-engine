use std::fs;
use std::path::PathBuf;
use std::sync::{atomic::{AtomicI32, AtomicU64, AtomicBool}, Arc, Mutex};
use std::time::Instant;
use dashmap::DashMap;
use tokio::sync::mpsc;

use ed25519_dalek::SigningKey;
use rand::rngs::OsRng;
use tracing::info;

// The QUIC DHT garner loop performs NTT-CRT shard recombination via the
// network::ntt_ffi C ABI, which does not link on aarch64-android (Phase
// 2-L3.FG scope). It is host-only; the android daemon serves the mesh surface
// from an empty peer_map (no inference cluster on a single on-device node).
#[cfg(not(target_os = "android"))]
use sp_daemon::network::quic_shard::{run_garner_loop, SpQuicCoordinator, SpQuicWorker};

/// NTT transform size for CRT residue reconstruction.
/// Matches the test topology (N=128). Tuned to model layer width in BLOCK-SYNC phase.
#[cfg(not(target_os = "android"))]
const QUIC_NTT_N: u32 = 128;

fn pid_file() -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push("sp-daemon.pid");
    p
}

// ── Commands (called from main, synchronous) ───────────────────────────────

/// Spawn the daemon inner process detached from the current session, write
/// the PID file, and return so the calling process can exit.
pub fn cmd_start(model: &str, tokenizer: &str, draft_model: &str, draft_tokenizer: &str, memo_model: &str, memo_tokenizer: &str, pouw_ledger_path: &str, quic_port: u16, http_port: u16, peer: &str, peers: &str) {
    let exe = std::env::current_exe().expect("current_exe");
    let quic_port_s = quic_port.to_string();
    let http_port_s = http_port.to_string();
    let mut cmd = std::process::Command::new(&exe);
    cmd.args([
        "--daemon-inner",
        "--model",           model,
        "--tokenizer",       tokenizer,
        "--draft-model",     draft_model,
        "--draft-tokenizer", draft_tokenizer,
        // Chat-integration: Memory model for /v1/dialogue (empty = disabled).
        "--memo-model",      memo_model,
        "--memo-tokenizer",  memo_tokenizer,
        // ledger-autowire: PoUW receipt ledger path (empty = disabled).
        "--pouw-ledger-path", pouw_ledger_path,
        "--quic-port",       &quic_port_s,
        "--port",            &http_port_s,
        "--peer",            peer,
        "--peers",           peers,
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
///
/// Phase 2-L3.FG: unified across host + android. The L1 forward (chat) + sieve
/// mining (ledger) run on both (the C ABI links on android now). The cDSP DSP
/// model + echo session load in a `cfg(android)` block; the QUIC garner mesh
/// stays host-only (NTT-CRT cluster, out of scope — android serves empty peers).
pub async fn run_inner(model_path: &str, tok_path: &str, draft_model_path: &str, draft_tok_path: &str, memo_model_path: &str, memo_tok_path: &str, pouw_ledger_path: &str, quic_port: u16, http_port: u16, peer: &str, peers: &str) {
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
    let mut session = crate::session::SpSession::create(&model, Arc::clone(&cancel_flag))
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

    // ── Chat-integration: Memory model for /v1/dialogue — optional ────────
    let (memo_model, memo_session, memo_tokenizer, memo_vocab_size) = if !memo_model_path.is_empty() {
        let mm = crate::session::SpModel::load(memo_model_path, memo_tok_path)
            .expect("memo sp_model_load failed — check SP_MEMO_MODEL_PATH / SP_MEMO_TOKENIZER_PATH");
        let memo_arch = mm.arch_info().expect("memo sp_model_arch failed");
        info!("memo arch: vocab={} n_layers={} hidden={}",
            memo_arch.vocab_size, memo_arch.n_layers, memo_arch.hidden_dim);
        let m_cancel = Arc::new(AtomicI32::new(0));
        let ms = crate::session::SpSession::create(&mm, m_cancel)
            .expect("memo sp_session_create failed");
        let mt = crate::tokenizer::SptbTokenizer::build(&mm, memo_arch.arch_id)
            .expect("memo SptbTokenizer::build failed — check Memory .sp-tokenizer blob");
        info!("memo tokenizer built: arch_id={} eos_ids={:?}", memo_arch.arch_id, mt.eos_ids);
        (
            Some(mm),
            Some(Mutex::new(ms)),
            Some(mt), // SptbTokenizer::build already returns Arc<Self>
            memo_arch.vocab_size as usize,
        )
    } else {
        info!("no memo model — /v1/dialogue endpoint will return HTTP 501");
        (None, None, None, 0usize)
    };

    // ── ledger-autowire: open the PoUW receipt ledger if --pouw-ledger-path
    // was passed. Open failures bail the daemon (operator can correct
    // misconfig at startup rather than at first dialogue). Empty path
    // disables autowire (None → /v1/dialogue handler skips append silently).
    let ledger: Option<Arc<Mutex<sp_daemon::pouw_ledger::Ledger>>> = if !pouw_ledger_path.is_empty() {
        match sp_daemon::pouw_ledger::Ledger::open(pouw_ledger_path) {
            Ok(l) => {
                info!("ledger-autowire: PoUW ledger opened at {} ({} pre-existing bytes)",
                    pouw_ledger_path, l.len_bytes());
                Some(Arc::new(Mutex::new(l)))
            }
            Err(e) => {
                panic!("ledger-autowire: PoUW ledger open failed at {pouw_ledger_path}: {e}");
            }
        }
    } else {
        info!("ledger-autowire: PoUW ledger autowire disabled (--pouw-ledger-path empty)");
        None
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

    // §3-HX cDSP bridge (android-only): open the echo session + load the
    // DSP-resident model at startup. Any failure degrades to None (the
    // corresponding /v1/dsp/* endpoint returns 501) — never crashes the daemon.
    #[cfg(target_os = "android")]
    let (dsp_session, dsp_model, kv_cache) = {
        const ECHO_SKEL_URI: &str =
            "file:///libsp_echo_skel.so?sp_echo_skel_handle_invoke&_modver=1.0&_dom=cdsp";
        const MODEL_SKEL_URI: &str =
            "file:///libsp_compute_skel.so?sp_compute_skel_handle_invoke&_modver=1.0&_dom=cdsp";
        const CTX_MAX: usize = 4096;

        let dsp_session = match crate::dsp_rpc::FastRpcSession::new(ECHO_SKEL_URI) {
            Ok(s) => { info!("§3-HX Sprint C: cDSP echo session open"); Some(Mutex::new(s)) }
            Err(e) => {
                tracing::warn!("§3-HX Sprint C: FastRpcSession::new failed: {e:?} — /v1/dsp/echo will 501");
                None
            }
        };

        // Model session is a SEPARATE FastRpcSession leaked to 'static so the
        // ~1.4 GB DmaBuffer<'sess> borrows live for the process lifetime.
        let (dsp_model, kv_cache) = match crate::dsp_rpc::FastRpcSession::new(MODEL_SKEL_URI) {
            Ok(s) => {
                let sess: &'static crate::dsp_rpc::FastRpcSession = Box::leak(Box::new(s));
                match crate::dsp_model::DspModel::load(sess, model_path) {
                    Ok(m) => {
                        info!("L3.FG: DSP model loaded — {} layers, {} MB DMA, {} ms",
                            m.header.n_layers, m.total_dma_bytes / (1024 * 1024), m.load_wall_ms);
                        let kv = match crate::kv_cache::KvCache::alloc(sess, &m.header, CTX_MAX) {
                            Ok(kv) => {
                                info!("L3.FG: KV cache — {} MB, {} ms",
                                    kv.total_bytes() / (1024 * 1024), kv.alloc_wall_ms);
                                Some(std::sync::Arc::new(Mutex::new(crate::state::KvCacheHandle(kv))))
                            }
                            Err(e) => { tracing::warn!("L3.FG: KvCache::alloc failed: {e:?}"); None }
                        };
                        (Some(std::sync::Arc::new(crate::state::ModelHandle(m))), kv)
                    }
                    Err(e) => {
                        tracing::warn!("L3.FG: DspModel::load failed: {e:?} — /v1/dsp/model_info will 501");
                        (None, None)
                    }
                }
            }
            Err(e) => {
                tracing::warn!("L3.FG: model FastRpcSession::new failed: {e:?} — /v1/dsp/model_info will 501");
                (None, None)
            }
        };
        (dsp_session, dsp_model, kv_cache)
    };

    // ── §4-NTT Sprint NTT.5b: optional Hexagon NTT compute-backend ─────────
    //
    // When SP_ENGINE_NTT_ATTN_HEX=1 is set AND a Memory model session is
    // loaded, we (a) open a dedicated FastRpcSession against the compute
    // skel, (b) wrap it in an Arc<ComputeBackend>, (c) leak an Arc::clone
    // into a Box<ComputeBackend> raw pointer that L1 can hold, and (d) call
    // sp_session_register_compute_backend on the Memory session. The
    // backend is also stashed in AppState so the Arc count keeps the
    // FastRpcSession alive past the L1 raw pointer's last invocation.
    //
    // Unset env OR no Memory model OR FastRpcSession open failure = None
    // (the L1 register call is skipped; existing host path stays intact).
    //
    // NTT.5b ships the registration only. Consumption — i.e. flipping
    // forward.c's NTT-attention routing through the backend instead of the
    // host ntt_crt path — is OUT OF SCOPE per the sprint spec.
    #[cfg(target_os = "android")]
    let ntt_hex_backend: Option<std::sync::Arc<sp_daemon::ntt_hex_dispatch::ComputeBackend>> = {
        let env_set = std::env::var("SP_ENGINE_NTT_ATTN_HEX")
            .map(|v| v.trim() == "1")
            .unwrap_or(false);
        if env_set && memo_session.is_some() {
            const COMPUTE_SKEL_URI: &str =
                "file:///libsp_compute_skel.so?sp_compute_skel_handle_invoke&_modver=1.0&_dom=cdsp";
            // Use lib-crate path so the FastRpcSession type matches the
            // ComputeBackend::new constructor (both live in the lib crate).
            match sp_daemon::dsp_rpc::FastRpcSession::new(COMPUTE_SKEL_URI) {
                Ok(s) => {
                    info!("NTT.5b: Hexagon compute backend session open (SP_ENGINE_NTT_ATTN_HEX=1)");
                    let backend = std::sync::Arc::new(
                        sp_daemon::ntt_hex_dispatch::ComputeBackend::new(std::sync::Arc::new(s))
                    );
                    // Register with the Memory session via the L1 ABI.
                    // The raw pointer is the Arc::clone-leaked Box; AppState's
                    // ntt_hex_backend field keeps the Arc alive so the pointer
                    // stays valid until daemon shutdown.
                    if let Some(memo_mu) = memo_session.as_ref() {
                        let backend_for_l1 = std::sync::Arc::clone(&backend);
                        let leaked: *mut sp_daemon::ntt_hex_dispatch::ComputeBackend =
                            std::sync::Arc::into_raw(backend_for_l1) as *mut _;
                        let (fwd, inv) = sp_daemon::ntt_hex_dispatch::ComputeBackend::dispatch_fns();
                        // memo_session is a Mutex<SpSession>; take the lock briefly
                        // to get the raw L1 session pointer for the register call.
                        let mut guard = memo_mu.lock().unwrap();
                        // SAFETY: SpSession.ptr stays valid for SpSession's lifetime;
                        // we hold the Mutex guard so no concurrent forward is running.
                        let session_raw: *mut crate::ffi::sp_session = guard.raw_ptr();
                        let rc = unsafe {
                            crate::ffi::sp_session_register_compute_backend(
                                session_raw,
                                leaked as *mut std::os::raw::c_void,
                                Some(fwd),
                                Some(inv),
                            )
                        };
                        drop(guard);
                        if rc == crate::ffi::sp_status_SP_OK {
                            info!("NTT.5b: sp_session_register_compute_backend OK on Memory session");
                        } else {
                            tracing::warn!("NTT.5b: sp_session_register_compute_backend rc={rc} — backend stored on AppState but L1 link failed");
                            // Reclaim the leaked Arc so we don't double-count;
                            // the AppState Arc still has its own ref.
                            unsafe { std::sync::Arc::from_raw(leaked); }
                        }
                    }
                    Some(backend)
                }
                Err(e) => {
                    tracing::warn!("NTT.5b: FastRpcSession::new failed: {e:?} — backend disabled (host path)");
                    None
                }
            }
        } else {
            if env_set && memo_session.is_none() {
                info!("NTT.5b: SP_ENGINE_NTT_ATTN_HEX=1 set but no Memory model — backend disabled");
            }
            None
        }
    };

    // ── Sprint WIRE-HEX: optional full-forward backend on the TARGET session ─
    //
    // When SP_DAEMON_BACKEND=hex is set AND the daemon was built with
    // --features wire_hex_backend (so libsp_hex_daemon_backend.a is linked),
    // register the engine's gemma3_forward_hexagon dispatcher with the
    // target session via the new sp_l1.h:§6 hook. After registration,
    // sp_prefill_chunk routes through the cDSP V69 HVX backend instead of
    // math-core's reference forward (the 6-month gap fix).
    //
    // Decode (persistent KV) is NOT hooked — sp_hex_host.c re-runs the full
    // forward over the accumulated history per call and has no persistent-KV
    // API. Decode keeps using the math-core reference. Documented honestly
    // in CLOSURE-WIRE-HEX.md.
    //
    // Unset env OR wrong feature set = skipped (existing reference path).
    // Wrong arch (not gemma3) or wrong arena (not Q8) = the first prefill
    // will return SP_EBADSTATE; daemon logs the cause via sp_last_error.
    #[cfg(all(target_os = "android", feature = "wire_hex_backend"))]
    let wire_hex_active = {
        let env_set = std::env::var("SP_DAEMON_BACKEND")
            .map(|v| v.trim().eq_ignore_ascii_case("hex"))
            .unwrap_or(false);
        if env_set {
            // `session` here is the raw SpSession (not yet wrapped in Mutex);
            // we own it exclusively so no locking needed.
            // The binary-crate `crate::ffi::sp_session` and the lib-crate
            // `sp_daemon::ffi_l1::sp_session` are bindgen outputs from the
            // same sp_l1.h header — byte-identical opaque structs but
            // distinct Rust types. Cast through *mut to bridge the alias.
            let session_raw: *mut sp_daemon::ffi_l1::sp_session =
                session.raw_ptr() as *mut sp_daemon::ffi_l1::sp_session;
            // SAFETY: we own `session` exclusively; no concurrent forward.
            match unsafe { sp_daemon::hex_forward_dispatch::register_with_session(session_raw) } {
                Ok(()) => {
                    info!("WIRE-HEX: sp_session_register_forward_backend OK on TARGET session — prefill routes to gemma3_forward_hexagon (cDSP V69 HVX)");
                    true
                }
                Err(e) => {
                    tracing::warn!("WIRE-HEX: registration failed: {e} — falling back to math-core reference forward");
                    false
                }
            }
        } else {
            false
        }
    };
    #[cfg(not(all(target_os = "android", feature = "wire_hex_backend")))]
    let wire_hex_active = false;
    if !wire_hex_active {
        // Host build OR feature off OR env unset — log once for clarity.
        #[cfg(all(target_os = "android", feature = "wire_hex_backend"))]
        info!("WIRE-HEX: feature linked but SP_DAEMON_BACKEND!=hex — staying on math-core reference forward");
    }

    // ── Sprint WIRE-CPU: optional full-forward backend on the TARGET session ─
    //
    // When SP_DAEMON_BACKEND=cpu is set AND the daemon was built with
    // --features wire_cpu_backend (so sp_cpu_daemon_backend is linked),
    // register the engine's per-arch CPU forward dispatcher with the
    // target session via the same sp_l1.h:§6 hook WIRE-HEX uses. After
    // registration, sp_prefill_chunk routes through the engine's
    // gemma3_forward_cpu / qwen3_forward_cpu / qwen25_forward_cpu_impl
    // (cpu_overlay.c AVX2 dot_f32 + optional AVX-512 primitives) instead
    // of math-core's reference forward.
    //
    // HOST target — no Android cross-compile. The CPU backend has no
    // per-session statics (cpu_overlay.c reads gate-knob env vars at each
    // call); release_for_model is a no-op.
    //
    // Decode (persistent KV) is NOT hooked, same architectural pattern as
    // WIRE-HEX — gemma3_forward_cpu re-runs the full forward per call.
    //
    // Unset env OR wrong feature set = skipped (existing reference path).
    #[cfg(feature = "wire_cpu_backend")]
    let wire_cpu_active = {
        let env_set = std::env::var("SP_DAEMON_BACKEND")
            .map(|v| v.trim().eq_ignore_ascii_case("cpu"))
            .unwrap_or(false);
        if env_set {
            // `session` here is the raw SpSession (not yet wrapped in Mutex);
            // we own it exclusively so no locking needed.
            //
            // Unlike WIRE-HEX which had to bridge the binary-crate ffi to
            // the lib-crate ffi_l1 via a pointer cast, WIRE-CPU's trampoline
            // lives in the binary crate (host-only, no need for the
            // lib-crate sibling). The raw pointer is passed straight through
            // as `*mut crate::ffi::sp_session`.
            let session_raw: *mut crate::ffi::sp_session = session.raw_ptr();
            // SAFETY: we own `session` exclusively; no concurrent forward.
            match unsafe { crate::cpu_forward_dispatch::register_with_session(session_raw) } {
                Ok(()) => {
                    info!("WIRE-CPU: sp_session_register_forward_backend OK on TARGET session — prefill routes to engine CPU AVX-512 backend (gemma3_forward_cpu / qwen3_forward_cpu)");
                    true
                }
                Err(e) => {
                    tracing::warn!("WIRE-CPU: registration failed: {e} — falling back to math-core reference forward");
                    false
                }
            }
        } else {
            false
        }
    };
    #[cfg(not(feature = "wire_cpu_backend"))]
    let wire_cpu_active = false;
    if !wire_cpu_active {
        #[cfg(feature = "wire_cpu_backend")]
        info!("WIRE-CPU: feature linked but SP_DAEMON_BACKEND!=cpu — staying on math-core reference forward");
    }

    // ── Sprint WIRE-CUDA: optional full-forward backend on the TARGET session ─
    //
    // When SP_DAEMON_BACKEND=cuda is set AND the daemon was built with
    // --features wire_cuda_backend (so libsp_cuda_daemon_backend.lib is
    // linked), register the engine's gemma3_forward_cuda / qwen3_forward_cuda
    // dispatcher with the target session via sp_l1.h:§6. After registration,
    // sp_prefill_chunk routes through the CUDA PTX backend instead of
    // math-core's reference forward.
    //
    // Decode (persistent KV) is NOT hooked — the CUDA whole-forward path
    // re-runs the full forward over accumulated history per call (ppl-style
    // usage); hooking decode would be devastatingly slow without a per-backend
    // persistent-KV variant — different sprint.
    //
    // Host-only (no target_os = "android" constraint). Unset env OR wrong
    // feature set = skipped (existing reference path). Arch routing
    // (SP_ARCH_GEMMA3 vs SP_ARCH_QWEN3) is done by the C glue.
    #[cfg(feature = "wire_cuda_backend")]
    let wire_cuda_active = {
        let env_set = std::env::var("SP_DAEMON_BACKEND")
            .map(|v| v.trim().eq_ignore_ascii_case("cuda"))
            .unwrap_or(false);
        if env_set {
            // The binary-crate `crate::ffi::sp_session` and the lib-crate
            // `sp_daemon::ffi_l1::sp_session` are bindgen outputs from the
            // same sp_l1.h header — byte-identical opaque structs but
            // distinct Rust types. Cast through *mut to bridge the alias.
            let session_raw: *mut sp_daemon::ffi_l1::sp_session =
                session.raw_ptr() as *mut sp_daemon::ffi_l1::sp_session;
            // SAFETY: we own `session` exclusively at this point; no concurrent forward.
            match unsafe { sp_daemon::cuda_forward_dispatch::register_with_session(session_raw) } {
                Ok(()) => {
                    info!("WIRE-CUDA: sp_session_register_forward_backend OK on TARGET session — prefill routes to gemma3_forward_cuda / qwen3_forward_cuda (CUDA PTX)");
                    true
                }
                Err(e) => {
                    tracing::warn!("WIRE-CUDA: registration failed: {e} — falling back to math-core reference forward");
                    false
                }
            }
        } else {
            false
        }
    };
    #[cfg(not(feature = "wire_cuda_backend"))]
    let wire_cuda_active = false;
    if !wire_cuda_active {
        #[cfg(feature = "wire_cuda_backend")]
        info!("WIRE-CUDA: feature linked but SP_DAEMON_BACKEND!=cuda — staying on math-core reference forward");
    }

    // ── Sprint WIRE-VULKAN: optional full-forward backend on TARGET session ──
    //
    // When SP_DAEMON_BACKEND=vulkan is set AND the daemon was built with
    // --features wire_vulkan_backend (so libsp_vulkan_daemon_backend.{a,lib}
    // is linked along with the Vulkan loader), register the engine's
    // gemma3_forward_vulkan / qwen3_forward_vulkan dispatcher with the
    // target session via the sp_l1.h:§6 hook. After registration,
    // sp_prefill_chunk routes through the host GPU's Vulkan compute path
    // instead of math-core's reference forward — host-side analog of the
    // WIRE-HEX wiring on android.
    //
    // Decode (persistent KV) is NOT hooked — vulkan_forward.cpp re-runs the
    // full forward over the accumulated history per call and has no
    // persistent-KV API. Decode keeps using the math-core reference.
    // Documented honestly in CLOSURE-WIRE-VULKAN.md.
    //
    // Unset env OR wrong feature set = skipped (existing reference path).
    // Wrong arch (not Gemma3 / Qwen3) = the C glue's arch switch surfaces
    // sp_set_error("vulkan: unsupported arch ...") and returns -1; the
    // first prefill returns SP_EBADSTATE; daemon log shows the cause via
    // sp_last_error.
    //
    // Known prior OOM bug: M_GEMMA3_VULKAN + M_QWEN3_VULKAN ctests fail with
    // vkAllocateMemory: VkResult -2 on this host (RTX 2060, 6 GB VRAM).
    // The wiring still registers cleanly; the first prefill may hit the
    // same OOM. See ctest-vulkan-validate.log + WIRE-VULKAN-OOM-BUGFIX
    // follow-on.
    #[cfg(feature = "wire_vulkan_backend")]
    let wire_vulkan_active = {
        let env_set = std::env::var("SP_DAEMON_BACKEND")
            .map(|v| v.trim().eq_ignore_ascii_case("vulkan"))
            .unwrap_or(false);
        if env_set {
            // `session` here is the raw SpSession (not yet wrapped in Mutex);
            // we own it exclusively so no locking needed. Same crate::ffi <->
            // sp_daemon::ffi_l1 cast pattern as WIRE-HEX (both bindgen the
            // same sp_l1.h header; byte-identical opaque structs; distinct
            // Rust types). Cast through *mut to bridge the alias.
            let session_raw: *mut sp_daemon::ffi_l1::sp_session =
                session.raw_ptr() as *mut sp_daemon::ffi_l1::sp_session;
            // SAFETY: we own `session` exclusively; no concurrent forward.
            match unsafe { sp_daemon::vulkan_forward_dispatch::register_with_session(session_raw) } {
                Ok(()) => {
                    info!("WIRE-VULKAN: sp_session_register_forward_backend OK on TARGET session — prefill routes to gemma3_forward_vulkan / qwen3_forward_vulkan (host GPU compute)");
                    true
                }
                Err(e) => {
                    tracing::warn!("WIRE-VULKAN: registration failed: {e} — falling back to math-core reference forward");
                    false
                }
            }
        } else {
            false
        }
    };
    #[cfg(not(feature = "wire_vulkan_backend"))]
    let wire_vulkan_active = false;
    if !wire_vulkan_active {
        #[cfg(feature = "wire_vulkan_backend")]
        info!("WIRE-VULKAN: feature linked but SP_DAEMON_BACKEND!=vulkan — staying on math-core reference forward");
    }

    let state = Arc::new(crate::state::AppState {
        model,
        session: Mutex::new(session),
        cancel_flag,
        draft_model,
        draft_session,
        sessions: crate::sessions::Sessions::new(),
        wire_hex_active,
        wire_cpu_active,
        wire_cuda_active,
        wire_vulkan_active,
        vocab_size,
        tokens_decoded: AtomicU64::new(0),
        started_at: Instant::now(),
        events_tx: events_tx.clone(),
        tokenizer,
        inference_active: inference_active.clone(),
        receipt_store:    receipt_store.clone(),
        node_signing_key,
        peer_map: Arc::new(DashMap::new()),
        // Chat-integration: Memory model wiring (None when --memo-model unset).
        memo_session,
        memo_tokenizer,
        memo_model,
        memo_vocab_size,
        // ledger-autowire: shared PoUW ledger handle (None when --pouw-ledger-path unset).
        ledger,
        #[cfg(target_os = "android")]
        dsp_session,
        #[cfg(target_os = "android")]
        dsp_model,
        #[cfg(target_os = "android")]
        kv_cache,
        #[cfg(target_os = "android")]
        ntt_hex_backend,
    });

    // ── Background PoUW mining task ────────────────────────────────────────
    tokio::spawn(crate::mining::run_mining_loop(
        mining_signing_key,
        inference_active,
        receipt_store,
        events_tx,
    ));

    // ── QUIC DHT Coordinator + peer dials (host-only) ──────────────────────
    // The garner loop does NTT-CRT shard recombination for the inference
    // cluster — out of L3.FG scope (android serves an empty peer_map on a
    // single on-device node). Binds 0.0.0.0:<quic_port>; quic_port=0 disables.
    #[cfg(not(target_os = "android"))]
    {
        if quic_port != 0 {
            let quic_addr: std::net::SocketAddr = ([0, 0, 0, 0], quic_port).into();
            match SpQuicCoordinator::bind(quic_addr).await {
                Ok(coordinator) => {
                    info!("SP_INFO: QUIC Coordinator listening on {quic_addr}");
                    // Garner results channel: receiver intentionally discarded here.
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

        // Outbound peer dials (--peer single F4 form; --peers F5 bootstrap list).
        spawn_peer_dial(peer);
        for addr in peers.split(',').filter(|s| !s.is_empty()) {
            spawn_peer_dial(addr);
        }
    }
    #[cfg(target_os = "android")]
    let _ = (quic_port, peer, peers); // QUIC mesh is host-only (garner = NTT-CRT cluster)

    // ── HTTP server ────────────────────────────────────────────────────────
    let app = crate::server::build_router(Arc::clone(&state));

    let addr: std::net::SocketAddr = ([127, 0, 0, 1], http_port).into();
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .unwrap_or_else(|e| panic!("bind {addr}: {e}"));
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

/// Spawn a background task that dials `addr_str` as a QUIC peer and keeps
/// the connection alive via a 60s sleep loop.  On parse or dial failure:
/// logs a warning and exits silently — does NOT crash the daemon.
#[cfg(not(target_os = "android"))]
fn spawn_peer_dial(addr_str: &str) {
    if addr_str.is_empty() { return; }
    let addr_str = addr_str.to_string();
    tokio::spawn(async move {
        match addr_str.parse::<std::net::SocketAddr>() {
            Ok(peer_addr) => {
                let local: std::net::SocketAddr = ([0u8, 0, 0, 0], 0u16).into();
                match SpQuicWorker::connect(local, peer_addr).await {
                    Ok(worker) => {
                        info!("SP_INFO: QUIC connected to peer {peer_addr}");
                        loop {
                            tokio::time::sleep(std::time::Duration::from_secs(60)).await;
                            let _ = &worker;
                        }
                    }
                    Err(e) => tracing::warn!("SP_WARN: QUIC dial to {peer_addr} failed: {e}"),
                }
            }
            Err(e) => tracing::warn!("SP_WARN: invalid peer address {addr_str}: {e}"),
        }
    });
}

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
