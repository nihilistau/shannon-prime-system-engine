//! Safe Rust wrappers around the L1 opaque handles (sp_model, sp_session).
//!
//! Ownership rules from the L1 ABI contract (sp_l1.h):
//!   - sp_model is immutable after load → Send + Sync (many sessions per model).
//!   - sp_session is single-thread mutable → Send but NOT Sync.
//!   - cancel_flag storage must outlive sp_session_destroy → held in Arc.
use std::ffi::CString;
use std::ptr;
use std::sync::atomic::AtomicI32;
use std::sync::Arc;

use crate::ffi;

// ── SpModel ────────────────────────────────────────────────────────────────

pub struct SpModel(*mut ffi::sp_model);

// SAFETY: sp_model is immutable after sp_model_load returns.
unsafe impl Send for SpModel {}
unsafe impl Sync for SpModel {}

impl Drop for SpModel {
    fn drop(&mut self) {
        unsafe { ffi::sp_model_unload(self.0) };
    }
}

impl SpModel {
    pub fn load(model_path: &str, tok_path: &str) -> Result<Self, String> {
        let model_c = CString::new(model_path).map_err(|e| e.to_string())?;
        let tok_c = CString::new(tok_path).map_err(|e| e.to_string())?;
        let mut ptr: *mut ffi::sp_model = ptr::null_mut();
        let status =
            unsafe { ffi::sp_model_load(model_c.as_ptr(), tok_c.as_ptr(), &mut ptr) };
        if status == ffi::sp_status_SP_OK {
            Ok(SpModel(ptr))
        } else {
            let detail = unsafe { std::ffi::CStr::from_ptr(ffi::sp_last_error()) }
                .to_string_lossy()
                .into_owned();
            Err(format!("sp_model_load → status={status}: {detail}"))
        }
    }

    pub fn as_ptr(&self) -> *const ffi::sp_model {
        self.0 as *const _
    }
}

// ── SpSession ──────────────────────────────────────────────────────────────

pub struct SpSession {
    ptr: *mut ffi::sp_session,
    /// Keeps the cancel_flag allocation live for the session's lifetime
    /// (L1 contract: cancel_flag storage must outlive sp_session_destroy).
    _cancel_flag: Arc<AtomicI32>,
}

// SAFETY: session is Send (single thread at a time, enforced by Mutex<SpSession>
// in AppState). NOT Sync — callers must not share &SpSession across threads.
unsafe impl Send for SpSession {}

impl Drop for SpSession {
    fn drop(&mut self) {
        unsafe { ffi::sp_session_destroy(self.ptr) };
    }
}

impl SpSession {
    /// Create a session over an already-loaded model.
    ///
    /// `cancel_flag` is L2-owned. The raw pointer is passed to L1 and must
    /// remain valid until this `SpSession` is dropped. The `Arc` guarantees that.
    ///
    /// cancel_flag type: `volatile int *` in C (sp_l1.h:128).
    /// Rust side: `Arc<AtomicI32>` — AtomicI32 has the same layout as c_int;
    /// L1 only reads the flag with relaxed ordering (header guarantee), so the
    /// cast to *mut c_int is sound.
    pub fn create(model: &SpModel, cancel_flag: Arc<AtomicI32>) -> Result<Self, String> {
        let cfg = ffi::sp_session_config {
            max_context: 0,       // 0 = arch default
            deterministic: 1,     // bit-exact reductions for CORE
            arm_bank_kb: 0,
            sieve_capacity: 0,
            flags: 0,
            precision_override: 0, // defer to arch_info.preferred_precision
        };
        let mut ptr: *mut ffi::sp_session = ptr::null_mut();
        // SAFETY: cancel_flag Arc keeps the allocation alive until drop.
        let cancel_raw = cancel_flag.as_ptr() as *mut std::os::raw::c_int;
        let status = unsafe {
            ffi::sp_session_create(model.as_ptr(), &cfg, cancel_raw, &mut ptr)
        };
        if status == ffi::sp_status_SP_OK {
            Ok(SpSession { ptr, _cancel_flag: cancel_flag })
        } else {
            let detail = unsafe { std::ffi::CStr::from_ptr(ffi::sp_last_error()) }
                .to_string_lossy()
                .into_owned();
            Err(format!("sp_session_create → status={status}: {detail}"))
        }
    }

    /// Current sequence position (number of tokens consumed so far).
    /// Reads sp_session_position — the FFI proof call in /v1/metrics.
    pub fn position(&self) -> Result<usize, String> {
        let mut pos: usize = 0;
        let status = unsafe { ffi::sp_session_position(self.ptr, &mut pos) };
        if status == ffi::sp_status_SP_OK {
            Ok(pos)
        } else {
            Err(format!("sp_session_position → status={status}"))
        }
    }
}
