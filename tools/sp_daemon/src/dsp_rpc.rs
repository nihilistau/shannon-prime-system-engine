//! §3-HX Sprint A — FastRPC bridge for the Hexagon V69 cDSP.
//!
//! Dynamic loader for `libcdsprpc.so` (Android aarch64; userspace FastRPC).
//! Admission via `DSPRPC_CONTROL_UNSIGNED_MODULE` (Path B per
//! `reference-signed-pd-developer-path`) — Knack's S22U has no testsig on
//! `/vendor/etc/`, so signed-PD admission (Path A) is unavailable.
//!
//! Pattern source: `remote.h:793,803,811,840` (Hexagon SDK 5.5.6.0).
//! Prior-cohort validation: `shannon_prime_hexagon.c:93-100,148-154`.
//!
//! This module is compiled only on `target_os = "android"`. Host builds
//! (Windows / Linux x86) skip it via #[cfg].

#![cfg(target_os = "android")]

use std::ffi::{c_int, c_void, CString};
use std::os::raw::c_char;

use libloading::{Library, Symbol};

// ── Constants (from remote.h) ──────────────────────────────────────────────

/// CDSP_DOMAIN_ID — V69 cDSP on Snapdragon. remote.h:125.
pub const CDSP_DOMAIN_ID: c_int = 3;

/// DSPRPC_CONTROL_UNSIGNED_MODULE — session_control req ID. remote.h:641.
pub const DSPRPC_CONTROL_UNSIGNED_MODULE: u32 = 2;

/// AEE_ERPC = AEE_EOFFSET (0x80000400 on hex) + 0x200.
/// Surfaced to host as 0x80000600 on signature mismatch / admission failure.
pub const AEE_ERPC: c_int = 0x8000_0600u32 as i32;

// ── Wire types (#[repr(C)] from remote.h:161-210) ──────────────────────────

#[repr(C)]
#[derive(Clone, Copy)]
pub struct RemoteBuf {
    pub pv:   *mut c_void,
    pub nlen: usize,
}

#[repr(C)]
pub union RemoteArg {
    pub buf: RemoteBuf,
    pub h:   u32,
}

/// `struct remote_rpc_control_unsigned_module` — remote.h:423-428.
#[repr(C)]
struct RemoteRpcControlUnsignedModule {
    domain: c_int,
    enable: c_int,
}

// ── Function pointer types ─────────────────────────────────────────────────

type FnSessionControl = unsafe extern "C" fn(req: u32,
                                              data: *mut c_void,
                                              datalen: u32) -> c_int;

// Sprint A uses remote_handle64 (multi-domain) per SDK S22U pattern —
// matches the qaic-emitted "<iface>_skel_handle_invoke" symbol.  remote_handle
// (single-domain, u32) is a separate ABI.
type FnHandleOpen = unsafe extern "C" fn(name: *const c_char,
                                          ph: *mut u64) -> c_int;

type FnHandleInvoke = unsafe extern "C" fn(h: u64,
                                            scalars: u32,
                                            pra: *mut RemoteArg) -> c_int;

type FnHandleClose = unsafe extern "C" fn(h: u64) -> c_int;

// ── Errors ──────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum SpErr {
    /// `Library::new("libcdsprpc.so")` failed.
    LibLoad(String),
    /// `Library::get(<symbol>)` failed for the named symbol.
    Symbol(&'static str),
    /// `remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, ...)` returned non-zero.
    UnsignedPdReject(c_int),
    /// `remote_handle_open(...)` returned 0x80000600 (AEE_ERPC).
    ///
    /// Per `reference-signed-pd-developer-path`, five usual causes:
    ///   1. Skel not signed AND no Path B/C admission negotiated
    ///   2. Skel signed with wrong dev key
    ///   3. Skel pushed to path NOT in ADSP_LIBRARY_PATH
    ///   4. vendor.fastrpc.process.attrs=0x8 forces Unsigned Sandbox
    ///   5. Skel built for wrong DSP arch (-mv66 vs V69 device)
    SignatureMismatch(c_int),
    /// `remote_handle_open(...)` returned other non-zero.
    HandleOpen(c_int),
    /// `remote_handle_invoke(...)` returned non-zero.
    Invoke(c_int),
}

// ── Scalars helper (REMOTE_SCALARS_MAKE — remote.h:113) ────────────────────

/// Build the `dwScalars` u32 for `remote_handle_invoke`.
///
/// Layout: `(method:5 << 24) | (n_in:8 << 16) | (n_out:8 << 8)`.
/// `method` = IDL method index (qaic-generated; 0 for the first method).
/// `n_in`   = number of input buffers in `args` (counted from front).
/// `n_out`  = number of output buffers in `args` (counted after inputs).
/// `args.len()` MUST equal `n_in + n_out`.
pub fn make_scalars(method: u32, n_in: u32, n_out: u32) -> u32 {
    ((method & 0x1f) << 24) | ((n_in & 0xff) << 16) | ((n_out & 0xff) << 8)
}

// ── FastRpcSession ─────────────────────────────────────────────────────────

/// One FastRPC session = one loaded `libcdsprpc.so` + one open `remote_handle`.
///
/// Lifecycle: `new` → load libcdsprpc.so → resolve symbols →
///   `remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE)` →
///   `remote_handle_open(skel_uri)` → return.
/// `Drop` calls `remote_handle_close` (best-effort; errors logged not panicked).
pub struct FastRpcSession {
    // Keep the library loaded for the lifetime of the session — symbol
    // pointers below are only valid while `_lib` is held.
    _lib:           Library,
    fn_invoke:      FnHandleInvoke,
    fn_close:       FnHandleClose,
    handle:         u64,
}

impl FastRpcSession {
    /// Open an Unsigned-PD FastRPC session and load `skel_uri`.
    ///
    /// `skel_uri` is the qaic-generated URI plus the domain selector,
    /// e.g. `"file:///libshannonprime_echo_skel.so?_dom=cdsp"`.
    ///
    /// Per Path B: enables Unsigned PD on CDSP_DOMAIN_ID=3 via
    /// `remote_session_control` BEFORE the handle open. If the device's
    /// libcdsprpc.so version is old enough that the session-control symbol
    /// is absent, the loader returns `SpErr::Symbol("remote_session_control")`.
    pub fn new(skel_uri: &str) -> Result<Self, SpErr> {
        // 1. Load libcdsprpc.so. On aarch64-android the linker finds it via
        //    /vendor/lib64 / /system/lib64 by default.
        let lib = unsafe {
            Library::new("libcdsprpc.so")
                .map_err(|e| SpErr::LibLoad(format!("libcdsprpc.so: {e}")))?
        };

        // 2. Resolve the four symbols we need.
        let fn_session: Symbol<FnSessionControl> = unsafe {
            lib.get(b"remote_session_control\0")
                .map_err(|_| SpErr::Symbol("remote_session_control"))?
        };
        let fn_open: Symbol<FnHandleOpen> = unsafe {
            lib.get(b"remote_handle64_open\0")
                .map_err(|_| SpErr::Symbol("remote_handle64_open"))?
        };
        let fn_invoke: Symbol<FnHandleInvoke> = unsafe {
            lib.get(b"remote_handle64_invoke\0")
                .map_err(|_| SpErr::Symbol("remote_handle64_invoke"))?
        };
        let fn_close: Symbol<FnHandleClose> = unsafe {
            lib.get(b"remote_handle64_close\0")
                .map_err(|_| SpErr::Symbol("remote_handle64_close"))?
        };

        // Bind the resolved symbols to raw fn pointers we can store past
        // the `Symbol` scope.
        let fn_session_raw: FnSessionControl = *fn_session;
        let fn_open_raw:    FnHandleOpen     = *fn_open;
        let fn_invoke_raw:  FnHandleInvoke   = *fn_invoke;
        let fn_close_raw:   FnHandleClose    = *fn_close;
        // Drop the Symbol wrappers; raw pointers stay valid as long as
        // `lib` is held (and lib moves into Self below).
        drop(fn_session);
        drop(fn_open);
        drop(fn_invoke);
        drop(fn_close);

        // 3. Enable Unsigned PD on CDSP. Path B admission gate.
        let mut ctrl = RemoteRpcControlUnsignedModule {
            domain: CDSP_DOMAIN_ID,
            enable: 1,
        };
        let rc = unsafe {
            fn_session_raw(
                DSPRPC_CONTROL_UNSIGNED_MODULE,
                &mut ctrl as *mut _ as *mut c_void,
                std::mem::size_of::<RemoteRpcControlUnsignedModule>() as u32,
            )
        };
        if rc != 0 {
            return Err(SpErr::UnsignedPdReject(rc));
        }

        // 4. Open the skel handle (remote_handle64 = u64).
        let uri_c = CString::new(skel_uri)
            .map_err(|_| SpErr::HandleOpen(-1))?;
        let mut handle: u64 = 0;
        let rc = unsafe { fn_open_raw(uri_c.as_ptr(), &mut handle as *mut u64) };
        if rc != 0 {
            return Err(if rc == AEE_ERPC {
                SpErr::SignatureMismatch(rc)
            } else {
                SpErr::HandleOpen(rc)
            });
        }

        Ok(FastRpcSession {
            _lib: lib,
            fn_invoke: fn_invoke_raw,
            fn_close:  fn_close_raw,
            handle,
        })
    }

    /// Invoke a method on the open handle. `args.len()` must equal
    /// `n_in + n_out` as encoded in `scalars`.
    pub fn invoke(&self, scalars: u32, args: &mut [RemoteArg]) -> Result<(), SpErr> {
        let rc = unsafe { (self.fn_invoke)(self.handle, scalars, args.as_mut_ptr()) };
        if rc == 0 { Ok(()) } else { Err(SpErr::Invoke(rc)) }
    }
}

impl Drop for FastRpcSession {
    fn drop(&mut self) {
        // Best-effort close. Errors are logged but cannot be propagated
        // out of Drop. The OS will reap the handle when the process exits
        // regardless.
        let rc = unsafe { (self.fn_close)(self.handle) };
        if rc != 0 {
            eprintln!("[sp-daemon] FastRpcSession::drop: remote_handle_close \
                       returned {rc:#x} (handle={})", self.handle);
        }
    }
}
