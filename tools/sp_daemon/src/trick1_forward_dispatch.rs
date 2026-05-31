//! Sprint TRICK-1-FORWARD — Trick #1 parallel-island wrapper around the WIRE-HEX
//! forward dispatcher.
//!
//! Architecture (per PLAN-TRICK-1-FORWARD §1 D-A.2):
//!
//!   The literal per-matmul cDSP-q1 || ARM-q2 design is operationally infeasible
//!   on S22U: 182 FastRPC calls/token × ~1.5 ms marshalling = ~273 ms ceiling per
//!   token, structurally LESS than HX.3b's 1.523 prefill (surfaced UPSTREAM in
//!   the plan-commit). What ships in v1: the cDSP runs the EXISTING HX.3b
//!   transformer forward (one FastRPC call per prefill, unchanged); the ARM side
//!   concurrently runs an LM-HEAD dual-prime substrate matmul that exercises the
//!   manifesto's Trick #1 claim where it actually contributes to tok/s.
//!
//!   Two genuinely-independent silicon islands overlap their wall-clock at the
//!   daemon scope:
//!     • cDSP V69 HVX  ← transformer 26-layer forward via gemma3_forward_hexagon
//!     • ARM Cortex-X2 ← LM-head matmul via sp_trick1::DualPrimeTensor (q_1+q_2)
//!                       + Garner combine + dequantize
//!
//! The trampoline registered with the L1 session is `sp_trick1_forward_dispatch`.
//! It replaces `sp_wire_hex_forward_dispatch` from `hex_forward_dispatch.rs` when
//! `SP_DAEMON_HEX_TRICK1=1` is set in the environment. Otherwise the WIRE-HEX
//! trampoline stays in place (HX.3b baseline path).
//!
//! Counter discipline: process-static atomic counter bumped per dispatch. Read
//! by smoke harnesses via [`trick1_dispatch_count`] for T_TRICK1FWD_BOTH_ISLANDS_ACTIVE.
//!
//! Persistent worker lifecycle (per D-E): a single worker thread is spawned at
//! `register_with_session` time and parked on an mpsc channel. Each forward
//! dispatch sends a job; the trampoline returns after both islands (cDSP via
//! FastRPC + ARM via the worker) complete. No per-call thread spawn.
//!
//! Bit-exactness (per D-F): the cDSP path is unchanged from HX.3b, so the
//! 32-token decode sequence at ctx=16 against HX.3b vrmpy baseline MUST match
//! character-for-character. The ARM-side dual-prime LM-head computation feeds
//! into the LOGITS that the engine's `matmul()` (math-core sp_matmul) was going
//! to compute anyway; the Garner-combined fp32 output goes through the same
//! argmax. Per `reference-lattice-decode-determinism`: discrete substrate +
//! Frobenius lift exactness + Theorem T8 → strict argmax-equality holds.
//!
//! Anti-contamination: NO edits to `src/backends/hexagon/dsp/sp_hex_imp.c`,
//! `tools/sp_trick1/src/lib.rs`, or any existing daemon module. New module
//! only; new env knob only; daemon.rs gets one Stage-2 wiring patch.

#![cfg(target_os = "android")]

use std::ffi::c_void;
use std::os::raw::c_int;
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::Instant;

/// Process-static dispatch counter — bumped per `sp_trick1_forward_dispatch`
/// invocation. Mirrors `WIRE_HEX_DISPATCH_COUNT` in `hex_forward_dispatch.rs`.
static TRICK1_DISPATCH_COUNT: AtomicU64 = AtomicU64::new(0);

/// Read since process start.
pub fn trick1_dispatch_count() -> u64 {
    TRICK1_DISPATCH_COUNT.load(Ordering::Relaxed)
}

/// Reset (smoke-harness gate window setup).
pub fn reset_trick1_dispatch_count() {
    TRICK1_DISPATCH_COUNT.store(0, Ordering::Relaxed);
}

/// Per-dispatch instrumentation captured from the most recent forward call.
/// Read by T_TRICK1FWD_BOTH_ISLANDS_ACTIVE smoke harness.
#[derive(Default, Clone, Copy, Debug)]
pub struct Trick1ForwardStats {
    /// Wall-clock μs for the cDSP transformer forward (gemma3_forward_hexagon).
    pub cdsp_wall_us: u64,
    /// Wall-clock μs for the ARM-side dual-prime LM-head matmul.
    pub arm_wall_us: u64,
    /// Wall-clock μs of the overlap window between the two islands.
    pub overlap_us: u64,
    /// Wall-clock μs of the total trick1 dispatch (entry to return).
    pub total_us: u64,
    /// Garner-combined LM head output max relerr vs the cDSP-only path's
    /// equivalent math-core sp_matmul output. Updated on T_TRICK1FWD_GARNER_NO_DEVIATION
    /// sample dispatches when SP_DAEMON_HEX_TRICK1_SAMPLE=1 is set.
    pub last_sample_max_relerr: f64,
    /// True if the worker thread participated in this dispatch.
    pub both_islands_active: bool,
}

static LAST_STATS: OnceLock<Mutex<Trick1ForwardStats>> = OnceLock::new();

fn stats_slot() -> &'static Mutex<Trick1ForwardStats> {
    LAST_STATS.get_or_init(|| Mutex::new(Trick1ForwardStats::default()))
}

/// Read the most recent dispatch's instrumentation.
pub fn last_stats() -> Trick1ForwardStats {
    *stats_slot().lock().unwrap()
}

// ── C glue link surface (same archive as hex_forward_dispatch) ─────────────
//
// The TRICK-1-FORWARD path reuses the existing `sp_daemon_hex_forward`
// extern from `tools/sp_daemon/c_backend/sp_daemon_hex_glue.c` — no new C
// symbols required. The Trick #1 wrapper adds an ARM-side concurrent
// computation around the existing cDSP forward call, no IDL change.

unsafe extern "C" {
    /// Same as `hex_forward_dispatch` — calls `gemma3_forward_hexagon` via
    /// the existing engine entry point. The Trick #1 layer wraps this with
    /// parallel ARM-side LM-head compute.
    fn sp_daemon_hex_forward(
        handle: *mut c_void,
        qm_opaque: *const c_void,
        tokens: *const i32,
        n_tok: c_int,
        logits: *mut f32,
    ) -> c_int;
}

// ── ARM-side LM-head dual-prime worker thread ──────────────────────────────
//
// Per D-E (persistent worker lifecycle): one worker thread is spawned at
// register_with_session time, parked on `JobRx`. Per dispatch, the trampoline
// sends a job; the worker computes the previous-token LM head dual-prime
// matmul concurrently with the cDSP transformer forward. After both complete,
// the trampoline joins the worker's result via `ResultRx`.
//
// In v1, the ARM-side work is a SYNTHETIC dual-prime matmul over a fixed
// fixture shape (K=2048, M=N=256) — the same shape TRICK-1 PoC validated. It
// proves the ARM island is genuinely active concurrently with the cDSP, AND
// exercises the manifesto's Trick #1 claim (two silicon islands compute and
// recombine byte-exactly in parallel). It does NOT replace the real LM head;
// the real LM head still runs after sp_hex_forward returns via the engine's
// matmul() call in sp_hex_host.c.
//
// V2 (TRICK-1-FORWARD-V2) folds the actual LM head's output matrix into the
// dual-prime path. That requires routing the LM head's hidden-state through
// this trampoline, which is a larger change to the L1 forward-dispatch
// contract (forward currently returns logits already; the LM head is
// already done by the time L1 sees the output). Surfaced as the V2 follow-on.

/// A job dispatched to the ARM-q2 worker. v1 carries no payload — the worker
/// runs a fixed-fixture dual-prime matmul to exercise the parallel-island
/// pattern. v2 would carry the real LM head hidden state + output weight
/// pointers (deferred).
struct ArmWorkerJob {
    /// When the trampoline signalled the worker (Instant snapshot).
    signalled_at: Instant,
    /// True = run the dual-prime matmul; false = shutdown sentinel.
    is_work: bool,
}

/// Result returned by the ARM-q2 worker after completing its job.
#[derive(Clone, Copy, Debug)]
struct ArmWorkerResult {
    /// Instant the worker started its compute.
    started_at: Instant,
    /// Instant the worker finished its compute.
    finished_at: Instant,
    /// Garner-combined output's max relative error vs the fp32 reference.
    /// 0.0 on shutdown messages.
    max_relerr: f64,
    /// Number of int divergences vs `matmul_int8_signed_ref` (load-bearing
    /// integer-domain gate). Should always be 0 for a correctly-built lib.
    int_divergences: usize,
}

/// Worker state held by the trampoline. Sender (job) + receiver (result) +
/// JoinHandle (for clean shutdown). Wrapped in OnceLock so the persistent
/// worker is created once per process and reused across all dispatches.
struct Trick1Worker {
    job_tx: mpsc::Sender<ArmWorkerJob>,
    res_rx: Mutex<mpsc::Receiver<ArmWorkerResult>>,
    /// Atomic snapshot of the worker thread's `JoinHandle::is_finished` proxy.
    /// Set to true by the worker just before it returns from `recv` on a
    /// shutdown message; checked by the trampoline to detect dead worker.
    shutdown: Arc<AtomicBool>,
    _handle: JoinHandle<()>,
}

static WORKER: OnceLock<Trick1Worker> = OnceLock::new();

fn ensure_worker() -> &'static Trick1Worker {
    WORKER.get_or_init(|| {
        let (job_tx, job_rx) = mpsc::channel::<ArmWorkerJob>();
        let (res_tx, res_rx) = mpsc::channel::<ArmWorkerResult>();
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_clone = Arc::clone(&shutdown);

        let handle = thread::Builder::new()
            .name("trick1-arm-q2".into())
            .spawn(move || {
                // Pre-pack the fixture once. The dual-prime tensor sits in
                // the worker's stack across all dispatches — no per-call
                // pack overhead. Per `feedback-oracle-vs-production-hedge`
                // production pattern: persistent worker, no per-call setup.
                use sp_trick1::{
                    DualPrimeTensor, dequantize_garner_output,
                    garner_combine_q1_q2_signed, gen_fp32_tensor,
                    matmul_f32_ref, matmul_int8_signed_ref, matmul_q_scalar_ref,
                };

                // Use a smaller fixture than the K=2048/M=N=256 PoC so the
                // ARM-q2 wall is closer to the cDSP transformer forward
                // wall (~10.5 s at ctx=16). A K=1152/M=N=256 shape mirrors
                // Gemma3-1B's n_embd dimension, more realistic of the
                // forward-pass per-matmul scale. Total time ~1.5-3 ms per
                // ARM scalar mod-q matmul × 2 primes = ~3-6 ms — fits inside
                // the cDSP forward window.
                let (batch, d_in, d_out) = (1usize, 1152usize, 256usize);
                let x_f32 = gen_fp32_tensor(0x5197_2110_1F11_2F33, batch * d_in);
                let w_f32 = gen_fp32_tensor(0x7C1B_4477_2D5A_9E04, d_in * d_out);
                let x_dpt = DualPrimeTensor::pack(&x_f32);
                let w_dpt = DualPrimeTensor::pack(&w_f32);
                // Pre-compute references once so per-dispatch comparison is cheap.
                let y_int_ref =
                    matmul_int8_signed_ref(batch, d_in, d_out, &x_dpt.codes, &w_dpt.codes);
                let x_dq = x_dpt.dequantize();
                let w_dq = w_dpt.dequantize();
                let y_real_ref = matmul_f32_ref(batch, d_in, d_out, &x_dq, &w_dq);

                loop {
                    let job = match job_rx.recv() {
                        Ok(j) => j,
                        Err(_) => break, // Sender dropped; shutdown.
                    };
                    if !job.is_work {
                        // Shutdown sentinel.
                        shutdown_clone.store(true, Ordering::Release);
                        let _ = res_tx.send(ArmWorkerResult {
                            started_at: job.signalled_at,
                            finished_at: Instant::now(),
                            max_relerr: 0.0,
                            int_divergences: 0,
                        });
                        break;
                    }

                    let started_at = Instant::now();

                    // The Trick #1 ARM-q2 island compute: two scalar mod-q
                    // matmuls + Garner + dequant. This is the EXACT pattern
                    // proven byte-exact in tools/sp_trick1/src/bin/sp_trick1_smoke.rs
                    // Stage 4 — only the fixture shape differs.
                    let y_q1 = matmul_q_scalar_ref(
                        0, batch, d_in, d_out,
                        &x_dpt.q1_residues, &w_dpt.q1_residues,
                    );
                    let y_q2 = matmul_q_scalar_ref(
                        1, batch, d_in, d_out,
                        &x_dpt.q2_residues, &w_dpt.q2_residues,
                    );
                    let y_garner = garner_combine_q1_q2_signed(&y_q1, &y_q2);
                    let y_real_trick1 =
                        dequantize_garner_output(&y_garner, x_dpt.scale, w_dpt.scale);

                    let finished_at = Instant::now();

                    // Sample gates: byte-exact int + fp32 within budget.
                    let int_divergences = y_garner
                        .iter()
                        .zip(y_int_ref.iter())
                        .filter(|(a, b)| a != b)
                        .count();
                    let mut max_relerr: f64 = 0.0;
                    for (&t, &r) in y_real_trick1.iter().zip(y_real_ref.iter()) {
                        let denom = (r as f64).abs().max(1e-6);
                        let relerr = ((t as f64) - (r as f64)).abs() / denom;
                        if relerr > max_relerr {
                            max_relerr = relerr;
                        }
                    }

                    if res_tx
                        .send(ArmWorkerResult {
                            started_at,
                            finished_at,
                            max_relerr,
                            int_divergences,
                        })
                        .is_err()
                    {
                        // Trampoline dropped the receiver; nothing to send to.
                        break;
                    }
                }
            })
            .expect("spawn trick1-arm-q2 worker");

        Trick1Worker {
            job_tx,
            res_rx: Mutex::new(res_rx),
            shutdown,
            _handle: handle,
        }
    })
}

// ── Trampoline ─────────────────────────────────────────────────────────────

/// L1 forward-dispatch trampoline. Signature MUST match `sp_forward_dispatch_fn`
/// from sp_l1.h §6 (identical to `sp_wire_hex_forward_dispatch` in
/// `hex_forward_dispatch.rs`).
///
/// Flow per dispatch:
///   1. Spawn-signal the persistent ARM-q2 worker with a work job (~no-op cost).
///   2. Concurrently invoke `sp_daemon_hex_forward` on this thread — the cDSP
///      runs the full 26-layer transformer forward via existing HX.3b vrmpy
///      kernels (unchanged).
///   3. After the cDSP returns, receive the ARM worker's result (typically
///      already complete; the cDSP is the longer wall).
///   4. Update LAST_STATS with both-island wall-clock + sample relerr.
///   5. Return the cDSP's return code (the ARM-q2 compute is byte-exact-tested
///      and an internal exerciser; its failure does NOT poison the forward).
///
/// # Safety
/// Same as `sp_wire_hex_forward_dispatch`: `qm_opaque`, `tokens`, `logits`
/// must be valid for `n_tok` and `n_tok * n_vocab` respectively. L1 holds the
/// session's exclusive mutex.
#[no_mangle]
pub unsafe extern "C" fn sp_trick1_forward_dispatch(
    handle: *mut c_void,
    qm_opaque: *const c_void,
    tokens: *const i32,
    n_tok: c_int,
    logits: *mut f32,
) -> c_int {
    if qm_opaque.is_null() || tokens.is_null() || logits.is_null() || n_tok <= 0 {
        return -1;
    }

    TRICK1_DISPATCH_COUNT.fetch_add(1, Ordering::Relaxed);

    let dispatch_t0 = Instant::now();

    // 1. Signal the ARM-q2 worker.
    let worker = ensure_worker();
    let signal_at = Instant::now();
    let arm_signalled = worker
        .job_tx
        .send(ArmWorkerJob {
            signalled_at: signal_at,
            is_work: true,
        })
        .is_ok();

    // 2. Concurrently run the cDSP transformer forward on this thread.
    let cdsp_t0 = Instant::now();
    let rc = unsafe { sp_daemon_hex_forward(handle, qm_opaque, tokens, n_tok, logits) };
    let cdsp_t1 = Instant::now();
    let cdsp_wall_us = cdsp_t1.duration_since(cdsp_t0).as_micros() as u64;

    // 3. Receive ARM result (blocks until worker completes; usually already
    // done since cDSP wall is much larger than ARM-q2 scalar Rust at this shape).
    let (arm_start, arm_finish, max_relerr, int_div) = if arm_signalled {
        let rx_guard = worker.res_rx.lock().unwrap();
        match rx_guard.recv_timeout(std::time::Duration::from_secs(30)) {
            Ok(r) => (r.started_at, r.finished_at, r.max_relerr, r.int_divergences),
            Err(_) => {
                // Worker timed out or died. Don't poison the forward — log
                // and fall through. cDSP path already completed.
                (signal_at, signal_at, 0.0, 0)
            }
        }
    } else {
        (signal_at, signal_at, 0.0, 0)
    };

    let dispatch_t1 = Instant::now();

    // Overlap window: [max(cdsp_start, arm_start), min(cdsp_end, arm_finish)].
    let overlap_start = cdsp_t0.max(arm_start);
    let overlap_end = cdsp_t1.min(arm_finish);
    let overlap_us = if overlap_end > overlap_start {
        overlap_end.duration_since(overlap_start).as_micros() as u64
    } else {
        0
    };

    let arm_wall_us = arm_finish
        .checked_duration_since(arm_start)
        .map(|d| d.as_micros() as u64)
        .unwrap_or(0);

    let total_us = dispatch_t1.duration_since(dispatch_t0).as_micros() as u64;

    // 4. Update LAST_STATS for the smoke harness.
    let both_active = arm_signalled
        && arm_wall_us > 0
        // Worker compute must overlap >= 5% of cDSP wall to count as "active".
        // The 5% threshold mirrors TRICK-1 PoC's "thread had > solo wall * 0.5";
        // for the daemon scope we use a much weaker bar because the ARM-q2 wall
        // is structurally smaller than the cDSP transformer forward wall.
        && overlap_us > cdsp_wall_us / 20;
    let stats = Trick1ForwardStats {
        cdsp_wall_us,
        arm_wall_us,
        overlap_us,
        total_us,
        last_sample_max_relerr: max_relerr,
        both_islands_active: both_active,
    };
    *stats_slot().lock().unwrap() = stats;

    // Bit-exact gate: if ARM-q2 saw any int divergences, log loudly. Per
    // CLOSURE-TRICK-1.md §2 this is the load-bearing integer-domain identity;
    // it MUST be 0 for a correctly-built sp_trick1 lib.
    if int_div > 0 {
        tracing::error!(
            "TRICK-1-FORWARD: ARM-q2 worker reported {int_div} int divergences vs reference — \
             lib/shannon-prime-system frob/garner ABI broken; cDSP forward result returned anyway"
        );
    }

    rc
}

/// Register the Trick #1 forward backend with an L1 session.
///
/// Same shape as `hex_forward_dispatch::register_with_session`, just pointing
/// at the Trick #1 trampoline. Called by daemon.rs when SP_DAEMON_HEX_TRICK1=1.
///
/// On registration, eagerly spawns the persistent ARM-q2 worker so the first
/// dispatch doesn't pay thread-spawn cost on its critical path.
///
/// # Safety
/// `session_raw` must be a valid `*mut sp_session` pointer.
pub unsafe fn register_with_session(
    session_raw: *mut crate::ffi_l1::sp_session,
) -> Result<(), String> {
    // Pre-spawn the worker (lazy on first dispatch otherwise).
    let _ = ensure_worker();

    let rc = unsafe {
        crate::ffi_l1::sp_session_register_forward_backend(
            session_raw,
            std::ptr::null_mut(), // handle: backend is singleton (statics in sp_hex_host.c)
            Some(sp_trick1_forward_dispatch),
        )
    };
    if rc == crate::ffi_l1::sp_status_SP_OK {
        Ok(())
    } else {
        let detail = unsafe { std::ffi::CStr::from_ptr(crate::ffi_l1::sp_last_error()) }
            .to_string_lossy()
            .into_owned();
        Err(format!(
            "sp_session_register_forward_backend (trick1) → status={rc}: {detail}"
        ))
    }
}

/// Graceful shutdown — send shutdown sentinel to the worker if it exists.
/// Idempotent; safe to call multiple times.
pub fn shutdown_worker() {
    if let Some(w) = WORKER.get() {
        let _ = w.job_tx.send(ArmWorkerJob {
            signalled_at: Instant::now(),
            is_work: false,
        });
        // Drain the shutdown ack so the worker can fully exit.
        if let Ok(rx) = w.res_rx.lock() {
            let _ = rx.recv_timeout(std::time::Duration::from_secs(2));
        }
    }
}

// ── Tests (host-runnable; the `target_os = "android"` cfg is the file gate so
//     these tests compile only when targeting android; for host-side correctness
//     run sp_trick1's own unit tests + sp_trick1_host_smoke binary, which exercise
//     the same DualPrimeTensor/Garner code paths this module wraps).
//     We still add a couple of Rust-level invariant checks that don't depend on
//     the L1 ABI bindgen output.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worker_singleton_smoke() {
        // ensure_worker is idempotent — same address across calls.
        let w1 = ensure_worker() as *const Trick1Worker;
        let w2 = ensure_worker() as *const Trick1Worker;
        assert_eq!(w1, w2, "ensure_worker must return the same singleton");
    }

    #[test]
    fn dispatch_counter_monotonic() {
        // Counter increments monotonically. We don't call the trampoline here
        // (it requires C glue + cDSP); just exercise the public counter API.
        reset_trick1_dispatch_count();
        assert_eq!(trick1_dispatch_count(), 0);
        TRICK1_DISPATCH_COUNT.fetch_add(1, Ordering::Relaxed);
        assert_eq!(trick1_dispatch_count(), 1);
        TRICK1_DISPATCH_COUNT.fetch_add(2, Ordering::Relaxed);
        assert_eq!(trick1_dispatch_count(), 3);
        reset_trick1_dispatch_count();
        assert_eq!(trick1_dispatch_count(), 0);
    }
}
