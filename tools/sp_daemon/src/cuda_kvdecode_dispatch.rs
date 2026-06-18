//! Sprint WIRE-CUDA-DECODE-GEMMA4 — persistent-KV decode backend dispatch.
//!
//! Symmetric to `cuda_forward_dispatch.rs` (WIRE-CUDA, PREFILL), but for the
//! **token-by-token DECODE** path the prefill hook cannot serve. See
//! `tools/sp_daemon/WIRE-CUDA-DECODE-GEMMA4.md` for the full design.
//!
//! WHY a second module: `sp_session_register_forward_backend` /
//! `sp_forward_dispatch_fn` (sp_l1.h §6) is PREFILL-ONLY — it re-runs the full
//! forward over the accumulated history per call, AND for a 12B OK_Q4B model the
//! tied full-vocab LM head is materialized only inside the DECODE path, so
//! driving decode through the prefill entry trips the guard at
//! `cuda_forward.cu:1627` (`-4: g4 probe: FULL head needs the f32 embd`).
//! The fix is a stateful, session-resident KV-decode verb mirroring the
//! already-frozen `gemma4_kv_*` C ABI (cuda_forward.cu, declared in
//! tests/test_gemma4_cuda.c:65-79).
//!
//! Architecture (mirrors WIRE-CUDA forward, Shape B):
//!  1. The future L1 verb `sp_session_register_kvdecode_backend` (sp_l1.h §6b,
//!     designed in the addendum, NOT yet in the frozen header) takes a
//!     dispatch TABLE (open/prefill/decode_step/rewind/position/close) over a
//!     session-resident handle, not a single stateless forward fn.
//!  2. This module's C glue (`sp_daemon_cuda_kvdecode_*` in
//!     `c_backend_cuda/sp_daemon_cuda_glue.c`) adapts that table onto the
//!     `gemma4_kv_*` symbols already compiled into
//!     `libsp_cuda_daemon_backend`.
//!  3. AppState owns the `sp_g4_kv*` handle lifetime (state.rs
//!     `cuda_kvdecode_handle`); daemon.rs opens it at startup when
//!     `SP_DAEMON_BACKEND=cuda` + `SP_DAEMON_KVDECODE=1` (INTEGRATION step).
//!
//! Null floor: this module compiles ONLY under `--features wire_cuda_backend`
//! (the same feature that links the CUDA lib carrying `gemma4_kv_*` — no new
//! feature). Without it the daemon binary is byte-identical to pre-WIRE-CUDA.
//!
//! SCAFFOLD: the device-wiring bodies are stubbed with `TODO(WIRE-CUDA-DECODE)`.
//! The real `gemma4_kv_*` calls + the `sp_session_register_kvdecode_backend`
//! header verb land at INTEGRATION (addendum §7). This file is the reviewable
//! skeleton + the link surface, and it COMPILES.

#![cfg(feature = "wire_cuda_backend")]

use std::ffi::c_void;
use std::os::raw::c_int;
use std::sync::atomic::{AtomicU64, Ordering};

/// Decode-step counter bumped per `decode_step` call. Read by the
/// `G-WIRE-CUDA-DECODE-GEMMA4` smoke harness to validate the verb was reached.
static KVDECODE_STEP_COUNT: AtomicU64 = AtomicU64::new(0);

/// Read the decode-step count since process start.
pub fn step_count() -> u64 {
    KVDECODE_STEP_COUNT.load(Ordering::Relaxed)
}

/// Reset the decode-step count (smoke-harness gate window setup).
pub fn reset_step_count() {
    KVDECODE_STEP_COUNT.store(0, Ordering::Relaxed);
}

// ── C glue link surface ─────────────────────────────────────────────────────
// These symbols live in `c_backend_cuda/sp_daemon_cuda_glue.c`, compiled into
// `libsp_cuda_daemon_backend` by `build-host-cuda-backend.bat`. build.rs links
// the static lib when the `wire_cuda_backend` feature is on. Each adapts one
// row of the `sp_kvdecode_dispatch_fn` table (addendum §2) onto a `gemma4_kv_*`
// symbol. The opaque `*mut c_void` handle is an `sp_g4_kv*` on the engine side.
unsafe extern "C" {
    /// `gemma4_kv_open(m, pmax)` -> `sp_g4_kv*` (as opaque handle).
    /// Returns NULL on failure (sp_last_error carries detail).
    fn sp_daemon_cuda_kvdecode_open(qm_opaque: *const c_void, pmax: c_int) -> *mut c_void;

    /// `gemma4_kv_prefill(s, toks, n)`. 0 on success.
    fn sp_daemon_cuda_kvdecode_prefill(
        handle: *mut c_void,
        tokens: *const i32,
        n_tok: c_int,
    ) -> c_int;

    /// One persistent-KV decode step at the live dpos. Writes the full-vocab
    /// logits row `[n_vocab]` for the NEXT position and advances dpos.
    /// TODO(WIRE-CUDA-DECODE): backed by the additive `gemma4_kv_decode_logits`
    /// symbol (addendum §3.1 option A) — NOT the argmax-only `gemma4_kv_decode`.
    fn sp_daemon_cuda_kvdecode_step(
        handle: *mut c_void,
        token: i32,
        logits: *mut f32,
    ) -> c_int;

    /// `gemma4_kv_rewind(s, n)`. O(1) cold-evict (`dpos -= n`). 0 on success.
    fn sp_daemon_cuda_kvdecode_rewind(handle: *mut c_void, n: c_int) -> c_int;

    /// `gemma4_kv_pos(s)`. Current dpos, or -1 on NULL.
    fn sp_daemon_cuda_kvdecode_position(handle: *const c_void) -> c_int;

    /// `gemma4_kv_close(s)`. Frees the resident cache. NULL-safe.
    fn sp_daemon_cuda_kvdecode_close(handle: *mut c_void);

    /// CONTRACT-CHAT-FULLSTACK B1 — `gemma4_kv_byteexact_set(s, on)`. Toggles
    /// per-session byte-exact "auditable mode" on the resident cache. 0 on success.
    fn sp_daemon_cuda_kvdecode_byteexact(handle: *mut c_void, on: c_int) -> c_int;

    /// CONTRACT-CHAT-FULLSTACK B2 (§6d-b) — `gemma4_kv_replay(s, epdir, npos, zero)`.
    /// Recall a stored episode's owner K/V into the resident cache at
    /// `[dpos, dpos+npos)` and advance dpos (SP_REPLAY into the live turn). `epdir`
    /// is a NUL-terminated path holding ep.mf/ep.k/ep.v; `zero!=0` = zeroed reject
    /// control. 0 on success.
    fn sp_daemon_cuda_kvdecode_replay(
        handle: *mut c_void,
        epdir: *const std::os::raw::c_char,
        npos: c_int,
        zero: c_int,
    ) -> c_int;
}

/// Open a session-resident KV-decode cache on the CUDA backend.
///
/// Returns the opaque `sp_g4_kv*` handle (as `*mut c_void`) on success, or an
/// error string. `pmax` is the max resident position count (context budget);
/// `qm_opaque` is the session-borrowed `qwen3_model*` (must be `SP_ARCH_GEMMA4`;
/// the tied head needs `SP_CUDA_DECODE_INT8=1` in the environment — see
/// `gemma4_kv_open` at cuda_forward.cu:3670).
///
/// # Safety
/// `qm_opaque` must point to a live `qwen3_model` borrowed from a session
/// (valid for the session lifetime).
pub unsafe fn open(qm_opaque: *const c_void, pmax: i32) -> Result<*mut c_void, String> {
    if qm_opaque.is_null() || pmax <= 0 {
        return Err("kvdecode open: NULL model or non-positive pmax".to_string());
    }
    // SAFETY: caller guarantees qm_opaque validity; pmax checked positive.
    let h = unsafe { sp_daemon_cuda_kvdecode_open(qm_opaque, pmax) };
    if h.is_null() {
        Err(last_error())
    } else {
        Ok(h)
    }
}

/// Ingest prompt history into the resident cache (stores K/V at `[dpos,dpos+n)`).
///
/// # Safety
/// `handle` must be a live `sp_g4_kv*` from [`open`]; `tokens` valid for `n_tok`.
pub unsafe fn prefill(handle: *mut c_void, tokens: &[i32]) -> Result<(), String> {
    if handle.is_null() || tokens.is_empty() {
        return Err("kvdecode prefill: NULL handle or empty tokens".to_string());
    }
    // SAFETY: handle live per caller; tokens slice gives ptr+len.
    let rc = unsafe {
        sp_daemon_cuda_kvdecode_prefill(handle, tokens.as_ptr(), tokens.len() as c_int)
    };
    if rc == 0 { Ok(()) } else { Err(last_error()) }
}

/// Decode one token, returning the full-vocab logits row for the next position.
///
/// `logits` must be a caller-allocated slice of `n_vocab` f32. L2 owns sampling
/// (greedy / temperature / top-p / spec-decode verify) over the returned row.
///
/// # Safety
/// `handle` must be a live `sp_g4_kv*` from [`open`]; `logits.len()` must equal
/// the model's `n_vocab` (the glue writes exactly that many floats).
pub unsafe fn decode_step(
    handle: *mut c_void,
    token: i32,
    logits: &mut [f32],
) -> Result<(), String> {
    if handle.is_null() || logits.is_empty() {
        return Err("kvdecode decode_step: NULL handle or empty logits".to_string());
    }
    // Bump BEFORE the call so the counter reflects attempted steps (the gate
    // criterion is "the verb was reached", not "CUDA succeeded").
    KVDECODE_STEP_COUNT.fetch_add(1, Ordering::Relaxed);
    // SAFETY: handle live per caller; logits slice gives the n_vocab buffer.
    let rc = unsafe { sp_daemon_cuda_kvdecode_step(handle, token, logits.as_mut_ptr()) };
    if rc == 0 { Ok(()) } else { Err(last_error()) }
}

/// O(1) cold-evict: shear the logical decode position back by `n`.
///
/// # Safety
/// `handle` must be a live `sp_g4_kv*` from [`open`].
pub unsafe fn rewind(handle: *mut c_void, n: i32) -> Result<(), String> {
    if handle.is_null() || n < 0 {
        return Err("kvdecode rewind: NULL handle or negative n".to_string());
    }
    // SAFETY: handle live per caller.
    let rc = unsafe { sp_daemon_cuda_kvdecode_rewind(handle, n) };
    if rc == 0 { Ok(()) } else { Err(last_error()) }
}

/// CONTRACT-CHAT-FULLSTACK B1 — toggle byte-exact ("auditable") mode on the
/// resident cache. `on=true` routes the islands+attention through the
/// exact-integer dual-prime CRT-NTT substrate (run-to-run bit-identical);
/// `on=false` restores the float Stage-A path (byte-identical null floor).
/// The caller MUST hold the cache Mutex (the chat path sets it on at request
/// start, off at request end).
///
/// # Safety
/// `handle` must be a live `sp_g4_kv*` from [`open`].
pub unsafe fn set_byteexact(handle: *mut c_void, on: bool) -> Result<(), String> {
    if handle.is_null() {
        return Err("kvdecode set_byteexact: NULL handle".to_string());
    }
    // SAFETY: handle live per caller; glue forwards to gemma4_kv_byteexact_set.
    let rc = unsafe { sp_daemon_cuda_kvdecode_byteexact(handle, if on { 1 } else { 0 }) };
    if rc == 0 { Ok(()) } else { Err(last_error()) }
}

/// CONTRACT-CHAT-FULLSTACK B2 (§6d-b) — replay a stored episode into the resident
/// cache at `[dpos, dpos+npos)` (SP_REPLAY recall into the live turn). `epdir`
/// holds ep.mf/ep.k/ep.v; `zero=true` injects the zeroed reject control. On reject
/// the caller undoes it with `rewind(handle, npos)`. The caller MUST hold the
/// cache Mutex (the chat path replays before decode under the Mutex).
///
/// # Safety
/// `handle` must be a live `sp_g4_kv*` from [`open`].
pub unsafe fn replay(handle: *mut c_void, epdir: &str, npos: i32, zero: bool) -> Result<(), String> {
    if handle.is_null() || npos <= 0 {
        return Err("kvdecode replay: NULL handle or non-positive npos".to_string());
    }
    let c_epdir = match std::ffi::CString::new(epdir) {
        Ok(s) => s,
        Err(_) => return Err("kvdecode replay: epdir has interior NUL".to_string()),
    };
    // SAFETY: handle live per caller; c_epdir owns the NUL-terminated buffer for
    // the duration of the call; glue forwards to gemma4_kv_replay.
    let rc = unsafe {
        sp_daemon_cuda_kvdecode_replay(handle, c_epdir.as_ptr(), npos, if zero { 1 } else { 0 })
    };
    if rc == 0 { Ok(()) } else { Err(last_error()) }
}

/// Current decode position (`dpos`), or -1 on NULL.
///
/// # Safety
/// `handle` must be a live `sp_g4_kv*` from [`open`] (or NULL).
pub unsafe fn position(handle: *const c_void) -> i32 {
    // SAFETY: glue handles NULL.
    unsafe { sp_daemon_cuda_kvdecode_position(handle) }
}

/// Free the resident cache. Idempotent / NULL-safe.
///
/// # Safety
/// `handle` must be a live `sp_g4_kv*` from [`open`] (or NULL); not used after.
pub unsafe fn close(handle: *mut c_void) {
    if !handle.is_null() {
        // SAFETY: handle live per caller; not used after close.
        unsafe { sp_daemon_cuda_kvdecode_close(handle) };
    }
}

// ── §6b dispatch-table trampolines ──────────────────────────────────────────
// The bindgen `sp_kvdecode_dispatch_fn` fn-ptr fields are typed over the opaque
// `sp_kvdecode_handle` + `qwen3_model*` (as `*const c_void`); the C glue uses
// plain `void*`. These thin `extern "C"` trampolines bridge the two (an
// `sp_kvdecode_handle*` IS the glue's `void*` handle — an `sp_g4_kv*`). They
// forward straight to the already-filled glue symbols, with `decode_step`
// bumping the gate's step counter so the harness can confirm the verb was hit.

type KvHandle = crate::ffi_l1::sp_kvdecode_handle;

unsafe extern "C" fn tramp_open(
    qm_opaque: *const c_void,
    pmax: c_int,
    out: *mut *mut KvHandle,
) -> c_int {
    // SAFETY: glue checks NULL qm; `out` written with the opaque handle.
    let h = unsafe { sp_daemon_cuda_kvdecode_open(qm_opaque, pmax) };
    if h.is_null() {
        return -1;
    }
    if !out.is_null() {
        unsafe { *out = h as *mut KvHandle };
    }
    0
}

unsafe extern "C" fn tramp_prefill(h: *mut KvHandle, tokens: *const i32, n_tok: c_int) -> c_int {
    // SAFETY: glue validates args; handle is the opaque void* cast.
    unsafe { sp_daemon_cuda_kvdecode_prefill(h as *mut c_void, tokens, n_tok) }
}

unsafe extern "C" fn tramp_decode_step(h: *mut KvHandle, token: i32, logits: *mut f32) -> c_int {
    KVDECODE_STEP_COUNT.fetch_add(1, Ordering::Relaxed);
    // SAFETY: glue forwards to gemma4_kv_decode_logits on the resident cache.
    unsafe { sp_daemon_cuda_kvdecode_step(h as *mut c_void, token, logits) }
}

unsafe extern "C" fn tramp_rewind(h: *mut KvHandle, n: c_int) -> c_int {
    // SAFETY: glue validates args.
    unsafe { sp_daemon_cuda_kvdecode_rewind(h as *mut c_void, n) }
}

unsafe extern "C" fn tramp_position(h: *const KvHandle) -> c_int {
    // SAFETY: glue is NULL-safe.
    unsafe { sp_daemon_cuda_kvdecode_position(h as *const c_void) }
}

unsafe extern "C" fn tramp_close(h: *mut KvHandle) {
    // SAFETY: glue is NULL-safe.
    unsafe { sp_daemon_cuda_kvdecode_close(h as *mut c_void) };
}

/// The dispatch table handed to L1. `'static` so the pointer stays valid for
/// the whole process — L1 stores `&DT` and re-emits the fn pointers per decode.
static DT: crate::ffi_l1::sp_kvdecode_dispatch_fn = crate::ffi_l1::sp_kvdecode_dispatch_fn {
    open: Some(tramp_open),
    prefill: Some(tramp_prefill),
    decode_step: Some(tramp_decode_step),
    rewind: Some(tramp_rewind),
    position: Some(tramp_position),
    close: Some(tramp_close),
};

/// Register the CUDA KV-decode backend with an L1 session.
///
/// Opens the resident `sp_g4_kv` cache (via the glue `open`) and registers the
/// §6b dispatch table with the session through
/// `sp_session_register_kvdecode_backend`. After this returns, the session's
/// `sp_decode_step` routes the single-token forward through `tramp_decode_step`
/// → `gemma4_kv_decode_logits` on the resident handle.
///
/// Returns the opaque KV handle on success so the caller (AppState) can own its
/// lifetime and pass it back at `close` time (the resident cache is freed by
/// `release_for_model`). The caller MUST drive `prefill` (history ingest) on
/// the returned handle before the first `sp_decode_step`.
///
/// # Safety
/// `session_raw` must be a valid `*mut sp_session` with the L2-side Mutex held;
/// `qm_opaque` the session's borrowed `qwen3_model*` (valid for the session
/// lifetime).
pub unsafe fn register_with_session(
    session_raw: *mut crate::ffi_l1::sp_session,
    qm_opaque: *const c_void,
    pmax: i32,
) -> Result<*mut c_void, String> {
    // Step 1: open the resident KV cache.
    // SAFETY: caller guarantees qm_opaque + session validity.
    let handle = unsafe { open(qm_opaque, pmax) }?;

    // Step 2: point sp_decode_step at the glue dispatch table on this session.
    // SAFETY: caller holds the SpSession's Mutex; no concurrent decode.
    let rc = unsafe {
        crate::ffi_l1::sp_session_register_kvdecode_backend(
            session_raw,
            handle as *mut KvHandle,
            &DT as *const crate::ffi_l1::sp_kvdecode_dispatch_fn,
        )
    };
    if rc != crate::ffi_l1::sp_status_SP_OK {
        // Roll back the resident cache so we don't leak it on a failed register.
        unsafe { close(handle) };
        return Err(format!(
            "sp_session_register_kvdecode_backend → status={rc}: {}",
            last_error()
        ));
    }

    Ok(handle)
}

/// Tear down a resident KV cache opened via [`register_with_session`].
/// Called at AppState shutdown. Idempotent / NULL-safe.
///
/// # Safety
/// `handle` must be an `sp_g4_kv*` from [`register_with_session`] (or NULL).
pub unsafe fn release_for_model(handle: *mut c_void) {
    // SAFETY: close is NULL-safe; handle not used after.
    unsafe { close(handle) };
}

/// Fetch the last engine error string via the L1 ABI (`sp_last_error`).
fn last_error() -> String {
    // SAFETY: sp_last_error returns a process-static NUL-terminated C string.
    unsafe { std::ffi::CStr::from_ptr(crate::ffi_l1::sp_last_error()) }
        .to_string_lossy()
        .into_owned()
}
