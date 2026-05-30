//! K.2-spike POC harness — `sp_npu_spike_smoke`.
//!
//! Per K2-SPIKE-DESIGN.md and the prompt's T_K2_SPIKE_POC gate:
//!
//!   1. Initialize QNN HTP backend + context via the C shim
//!      (`sp_qnn_init`).
//!   2. Round-trip an INT8 buffer through a single-op ElementWiseAdd
//!      graph on the NPU (`sp_qnn_run_add_smoke`).
//!   3. Verify byte-exact equality between observed output and the
//!      expected `c[i] = a[i] + b[i]` (with values chosen to stay in
//!      signed-int8 range so the add doesn't saturate).
//!   4. Report wall-clock for `graphExecute` round-trip (target <
//!      100 ms sanity bound).
//!   5. Clean up.
//!
//! Test vector (small + deterministic):
//!   N = 64
//!   a[i] = (i as i32 % 32 - 16) as i8       // -16..15
//!   b[i] = (i as i32 % 16 -  8) as i8       //  -8..7
//!   expected c[i] = a[i] + b[i] in [-24..22]
//!
//! Build:
//!   cargo build --target aarch64-linux-android --release --bin sp_npu_spike_smoke
//!
//! Deploy + run on S22U (see Cargo.toml for the full adb sequence).

#[cfg(not(target_os = "android"))]
fn main() {
    eprintln!("sp_npu_spike_smoke: host build skipped (Android-only).");
    eprintln!("Build target: aarch64-linux-android. Cross-compile and");
    eprintln!("deploy to Knack's S22U per Cargo.toml header.");
}

#[cfg(target_os = "android")]
use std::ffi::{c_char, c_int, CStr, CString};

#[cfg(target_os = "android")]
extern "C" {
    fn sp_qnn_init(htp_so_path: *const c_char) -> c_int;
    fn sp_qnn_run_add_smoke(
        a: *const i8,
        b: *const i8,
        c: *mut i8,
        n: u32,
        out_wall_ns: *mut u64,
    ) -> c_int;
    fn sp_qnn_shutdown();
    fn sp_qnn_last_error() -> *const c_char;
}

#[cfg(target_os = "android")]
fn err_str() -> String {
    unsafe {
        let p = sp_qnn_last_error();
        if p.is_null() {
            String::from("<null>")
        } else {
            CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    }
}

#[cfg(target_os = "android")]
fn main() {
    // 0. Pre-flight: print env we care about.
    eprintln!("[sp-npu-spike] === K.2-spike POC harness ===");
    eprintln!("[sp-npu-spike] LD_LIBRARY_PATH  = {:?}",
              std::env::var("LD_LIBRARY_PATH").ok());
    eprintln!("[sp-npu-spike] ADSP_LIBRARY_PATH = {:?}",
              std::env::var("ADSP_LIBRARY_PATH").ok());

    let htp_path =
        std::env::var("SP_QNN_HTP_SO").unwrap_or_else(|_| "libQnnHtp.so".to_string());
    eprintln!("[sp-npu-spike] dlopen target: {htp_path}");

    let c_path = CString::new(htp_path.as_str()).expect("CString");

    // 1. Initialize.
    let rc = unsafe { sp_qnn_init(c_path.as_ptr()) };
    if rc != 0 {
        eprintln!("[sp-npu-spike] FAIL init rc={rc} err={}", err_str());
        eprintln!("[sp-npu-spike] === T_K2_SPIKE_POC: FAIL ===");
        eprintln!("[sp-npu-spike] (upstream blocker -- surface per feedback-no-silent-gate-revisions)");
        std::process::exit(2);
    }
    eprintln!("[sp-npu-spike] init OK");

    // 2. Build inputs.
    const N: usize = 64;
    let mut a = [0i8; N];
    let mut b = [0i8; N];
    let mut expected = [0i8; N];
    for i in 0..N {
        let ai = (i as i32 % 32 - 16) as i8;   // [-16, 15]
        let bi = (i as i32 %  16 - 8) as i8;   // [-8, 7]
        a[i] = ai;
        b[i] = bi;
        expected[i] = ai + bi;                  // [-24, 22]; safe in int8
    }
    let mut c = [0i8; N];

    // 3. Run the smoke graph.
    let mut wall_ns: u64 = 0;
    let t_total_start = std::time::Instant::now();
    let rc = unsafe {
        sp_qnn_run_add_smoke(
            a.as_ptr(),
            b.as_ptr(),
            c.as_mut_ptr(),
            N as u32,
            &mut wall_ns,
        )
    };
    let t_total_us = t_total_start.elapsed().as_micros();
    if rc != 0 {
        eprintln!("[sp-npu-spike] FAIL run rc={rc} err={}", err_str());
        eprintln!("[sp-npu-spike] === T_K2_SPIKE_POC: FAIL ===");
        unsafe { sp_qnn_shutdown(); }
        std::process::exit(3);
    }

    // 4. Verify.
    let mut mismatches = 0usize;
    for i in 0..N {
        if c[i] != expected[i] {
            if mismatches < 10 {
                eprintln!(
                    "[sp-npu-spike] mismatch i={i} a={} b={} c={} expected={}",
                    a[i], b[i], c[i], expected[i]);
            }
            mismatches += 1;
        }
    }

    let wall_ms = wall_ns as f64 / 1_000_000.0;
    eprintln!("[sp-npu-spike] graphExecute wall = {wall_ms:.3} ms ({wall_ns} ns)");
    eprintln!("[sp-npu-spike] total run wall    = {t_total_us} us (includes graphCreate+Tensor+addNode+Finalize)");
    eprintln!("[sp-npu-spike] mismatches = {mismatches} / {N}");

    let pass_correctness = mismatches == 0;
    let pass_wall_bound  = wall_ms < 100.0;
    eprintln!("[sp-npu-spike] T_K2_SPIKE_POC correctness (bytes-equal): {}",
              if pass_correctness { "PASS" } else { "FAIL" });
    eprintln!("[sp-npu-spike] T_K2_SPIKE_POC sanity bound (<100 ms):    {}",
              if pass_wall_bound { "PASS" } else { "FAIL" });

    // 5. Shutdown.
    unsafe { sp_qnn_shutdown(); }
    eprintln!("[sp-npu-spike] shutdown OK");

    let exit = if pass_correctness && pass_wall_bound { 0 } else { 1 };
    eprintln!("[sp-npu-spike] === T_K2_SPIKE_POC: {} ===",
              if exit == 0 { "PASS" } else { "FAIL" });
    std::process::exit(exit);
}
