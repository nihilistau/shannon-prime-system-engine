//! §4-NTT Sprint NTT.5b Stage 3 — T_NTT5B_HOST_HEX_BIT_EXACT smoke.
//!
//! Drives `sp_pr_bluestein_inner` along two paths on the SAME input:
//!   1. Host path:  sp_pr_bluestein_set_backend(NULL,NULL,NULL) — math-core's
//!                  ntt_crt host pipeline
//!   2. Hex path:   sp_pr_bluestein_set_backend(&backend, fwd, inv) — the 4×
//!                  forward + 1× inverse inner NTT calls dispatch through
//!                  FastRPC to the Hexagon V69 HVX kernels (method 17 +
//!                  method 18).
//!
//! Compares the returned int64 byte-exact. Any divergence is a wiring bug
//! (NTT.4 already validated the underlying HVX kernels: T_NTT4_POLY_MUL_EXACT
//! 12/12 byte-exact, see CLOSURE-NTT-4.md).
//!
//! Sweep: N ∈ {64, 128, 256} (the Bluestein-admissible inner-product N values
//! relevant to the production attention overlay; inner M ∈ {128, 256, 512}),
//! 100 random seeds per N, both paths per seed → 600 dispatched-path runs +
//! 300 host-path baselines.
//!
//! Pre-call: ntt_twiddle_init(N=512) to warm VTCM tables (idempotent; same as
//! sp_ntt_4_polymul_smoke.rs:107-119).
//!
//! Build:
//!   cd tools/sp_daemon  (for the .cargo/config.toml NDK env)
//!   cargo build --target aarch64-linux-android --release \
//!       --manifest-path ../sp_dsp_smoke/Cargo.toml \
//!       --bin sp_ntt_5b_bluestein_hex_smoke
//!
//! Run:
//!   adb push libsp_compute_skel.so /data/local/tmp/
//!   adb push target/aarch64-linux-android/release/sp_ntt_5b_bluestein_hex_smoke \
//!        /data/local/tmp/
//!   adb shell 'ADSP_LIBRARY_PATH="/data/local/tmp;" \
//!     /data/local/tmp/sp_ntt_5b_bluestein_hex_smoke'

#[cfg(not(target_os = "android"))]
fn main() {
    eprintln!("sp_ntt_5b_bluestein_hex_smoke: host build skipped (target_os = android required)");
}

#[cfg(target_os = "android")]
mod dsp_rpc;

#[cfg(target_os = "android")]
mod ntt5b_ffi {
    //! FFI binding to math-core's Bluestein wrapper + the L1 backend ABI.
    //! These symbols live in libsp_poly_ring.a (sp_pr_bluestein_*) and the
    //! ABI typedef in libsp_session.a (sp_compute_ntt_dispatch_fn).
    use std::os::raw::c_int;

    #[repr(C)]
    pub struct SpPrBluesteinCtx {
        _opaque: [u8; 0],
    }

    /// Dispatch fn signature matches sp_compute_ntt_dispatch_fn in sp_l1.h.
    pub type SpComputeNttDispatchFn = unsafe extern "C" fn(
        handle: *mut std::ffi::c_void,
        q_idx: c_int,
        n: c_int,
        in_buf: *const u32,
        out_buf: *mut u32,
    ) -> c_int;

    extern "C" {
        pub fn sp_pr_bluestein_init(n: u32) -> *mut SpPrBluesteinCtx;
        pub fn sp_pr_bluestein_free(ctx: *mut SpPrBluesteinCtx);
        pub fn sp_pr_bluestein_inner(
            ctx: *mut SpPrBluesteinCtx,
            q: *const i32,
            k: *const i32,
        ) -> i64;
        pub fn sp_pr_bluestein_set_backend(
            ctx: *mut SpPrBluesteinCtx,
            handle: *mut std::ffi::c_void,
            forward: Option<SpComputeNttDispatchFn>,
            inverse: Option<SpComputeNttDispatchFn>,
        );
    }
}

#[cfg(target_os = "android")]
fn main() {
    use dsp_rpc::{make_scalars, FastRpcSession, RemoteArg, RemoteBuf};
    use ntt5b_ffi::*;
    use std::ffi::c_void;
    use std::os::raw::c_int;
    use std::sync::Arc;
    use std::time::Instant;

    eprintln!("[NTT.5b] sp_ntt_5b_bluestein_hex_smoke — T_NTT5B_HOST_HEX_BIT_EXACT");
    eprintln!("[NTT.5b]   host Bluestein vs Hex-backend Bluestein on N in {{64, 128, 256}}");

    // ── Open FastRPC session ──
    eprintln!("[NTT.5b] opening FastRpcSession (Path B Unsigned PD)...");
    let sess: Arc<FastRpcSession> = match FastRpcSession::new(
        "file:///libsp_compute_skel.so?sp_compute_skel_handle_invoke&_modver=1.0&_dom=cdsp",
    ) {
        Ok(s) => { eprintln!("[NTT.5b] session open"); Arc::new(s) }
        Err(e) => { eprintln!("[NTT.5b] session FAIL: {e:?}"); std::process::exit(1); }
    };

    // ── method 14: ntt_twiddle_init for N=512 (covers 128/256/512) ──
    eprintln!("[NTT.5b] priming VTCM twiddles via ntt_twiddle_init(N=512)...");
    {
        let mut prim_in: [u32; 1] = [512u32];
        let mut args = [
            RemoteArg { buf: RemoteBuf { pv: prim_in.as_mut_ptr() as *mut c_void, nlen: 4 }},
        ];
        match sess.invoke(make_scalars(14, 1, 0), &mut args) {
            Ok(_) => eprintln!("[NTT.5b] ntt_twiddle_init ok"),
            Err(e) => {
                eprintln!("[NTT.5b] ntt_twiddle_init FAIL: {e:?}");
                std::process::exit(1);
            }
        }
    }

    // ── Trampoline backend (mirrors sp_daemon::ntt_hex_dispatch but lives
    //    standalone in this smoke binary so it can be cross-checked without
    //    the daemon's full link graph) ──
    struct ComputeBackend {
        session: Arc<FastRpcSession>,
    }

    fn invoke_method(
        backend: &ComputeBackend, method: u32, q_idx: i32, n: i32,
        in_buf: &[u32], out_buf: &mut [u32],
    ) -> Result<(), ()> {
        let n_bytes = (n as usize) * 4;
        let mut prim_in: [u32; 4] = [q_idx as u32, n as u32, n_bytes as u32, n_bytes as u32];
        let mut in_bytes: Vec<u8> = Vec::with_capacity(n_bytes);
        for v in in_buf { in_bytes.extend_from_slice(&v.to_le_bytes()); }
        let mut out_bytes: Vec<u8> = vec![0u8; n_bytes];
        let mut args = [
            RemoteArg { buf: RemoteBuf { pv: prim_in.as_mut_ptr() as *mut c_void, nlen: 16 }},
            RemoteArg { buf: RemoteBuf { pv: in_bytes.as_mut_ptr() as *mut c_void, nlen: n_bytes }},
            RemoteArg { buf: RemoteBuf { pv: out_bytes.as_mut_ptr() as *mut c_void, nlen: n_bytes }},
        ];
        if backend.session.invoke(make_scalars(method, 2, 1), &mut args).is_err() {
            return Err(());
        }
        for (i, c) in out_bytes.chunks_exact(4).enumerate() {
            out_buf[i] = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        }
        Ok(())
    }

    unsafe extern "C" fn smoke_forward(
        handle: *mut c_void, q_idx: c_int, n: c_int,
        in_buf: *const u32, out_buf: *mut u32,
    ) -> c_int {
        if handle.is_null() || in_buf.is_null() || out_buf.is_null() || n <= 0 { return -1; }
        let backend = &*(handle as *const ComputeBackend);
        let in_slice = std::slice::from_raw_parts(in_buf, n as usize);
        let out_slice = std::slice::from_raw_parts_mut(out_buf, n as usize);
        match invoke_method(backend, 17, q_idx as i32, n as i32, in_slice, out_slice) {
            Ok(()) => 0, Err(()) => -1,
        }
    }

    unsafe extern "C" fn smoke_inverse(
        handle: *mut c_void, q_idx: c_int, n: c_int,
        in_buf: *const u32, out_buf: *mut u32,
    ) -> c_int {
        if handle.is_null() || in_buf.is_null() || out_buf.is_null() || n <= 0 { return -1; }
        let backend = &*(handle as *const ComputeBackend);
        let in_slice = std::slice::from_raw_parts(in_buf, n as usize);
        let out_slice = std::slice::from_raw_parts_mut(out_buf, n as usize);
        match invoke_method(backend, 18, q_idx as i32, n as i32, in_slice, out_slice) {
            Ok(()) => 0, Err(()) => -1,
        }
    }

    let backend = ComputeBackend { session: Arc::clone(&sess) };
    let backend_ptr = &backend as *const ComputeBackend as *mut c_void;

    // ── PRNG (xorshift64*; mirrors poly_ring_test.c's rng_coeff_blue) ──
    let mut rng_state: u64 = 0x5B3B5B7B5B7B5B7Bu64;
    fn rng_next(state: &mut u64) -> u64 {
        let mut x = *state;
        x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
        *state = x;
        x.wrapping_mul(0x2545F4914F6CDD1Du64)
    }
    fn rng_coeff_blue(state: &mut u64) -> i32 {
        // NTT.5a's rng_coeff_blue range: [-2^14, 2^14). Per the coefficient
        // bit-exactness invariant (|c_k| < M/2 ≈ 2^59 for N ≤ 256).
        let v = (rng_next(state) & 0x7FFFu64) as i32;
        v - (1i32 << 14)
    }

    // ── Driver ──
    let ns: [u32; 3] = [64u32, 128u32, 256u32];
    let seeds_per_n = 100u32;
    let mut total_runs = 0u32;
    let mut total_divergences = 0u32;
    let mut first_divergence: Option<(u32, u32, i64, i64)> = None;
    let mut per_n_results: Vec<(u32, u32)> = Vec::new();
    let mut wall_host_total = std::time::Duration::ZERO;
    let mut wall_hex_total = std::time::Duration::ZERO;

    let t_start = Instant::now();

    for &n in &ns {
        // Allocate one host ctx + one dispatched ctx for this N.
        let host_ctx = unsafe { sp_pr_bluestein_init(n) };
        let disp_ctx = unsafe { sp_pr_bluestein_init(n) };
        if host_ctx.is_null() || disp_ctx.is_null() {
            eprintln!("[NTT.5b]   N={n}: sp_pr_bluestein_init returned NULL");
            unsafe { sp_pr_bluestein_free(host_ctx); sp_pr_bluestein_free(disp_ctx); }
            continue;
        }
        // host_ctx stays unset = host path; disp_ctx gets the backend.
        unsafe {
            sp_pr_bluestein_set_backend(disp_ctx, backend_ptr,
                Some(smoke_forward), Some(smoke_inverse));
        }

        let mut n_divergences = 0u32;
        let mut q_buf = vec![0i32; n as usize];
        let mut k_buf = vec![0i32; n as usize];
        for seed_ix in 0..seeds_per_n {
            for i in 0..(n as usize) {
                q_buf[i] = rng_coeff_blue(&mut rng_state);
                k_buf[i] = rng_coeff_blue(&mut rng_state);
            }
            // Host path
            let t0 = Instant::now();
            let host_v = unsafe { sp_pr_bluestein_inner(host_ctx, q_buf.as_ptr(), k_buf.as_ptr()) };
            wall_host_total += t0.elapsed();
            // Hex path
            let t0 = Instant::now();
            let hex_v = unsafe { sp_pr_bluestein_inner(disp_ctx, q_buf.as_ptr(), k_buf.as_ptr()) };
            wall_hex_total += t0.elapsed();

            total_runs += 1;
            if host_v != hex_v {
                if first_divergence.is_none() {
                    first_divergence = Some((n, seed_ix, host_v, hex_v));
                    eprintln!("[NTT.5b]   ! N={n} seed_ix={seed_ix}: host={host_v} hex={hex_v}");
                }
                total_divergences += 1;
                n_divergences += 1;
            }
        }
        per_n_results.push((n, n_divergences));

        unsafe {
            // Unregister before free (defensive; free clears state but tidiness).
            sp_pr_bluestein_set_backend(disp_ctx, std::ptr::null_mut(), None, None);
            sp_pr_bluestein_free(host_ctx);
            sp_pr_bluestein_free(disp_ctx);
        }
        eprintln!("[NTT.5b]   N={n}: divergences={n_divergences}/{seeds_per_n}");
    }

    let elapsed = t_start.elapsed();

    eprintln!("\n[NTT.5b] -- T_NTT5B_HOST_HEX_BIT_EXACT SUMMARY --");
    eprintln!("[NTT.5b] total_runs          : {total_runs}");
    eprintln!("[NTT.5b] divergence_count    : {total_divergences}");
    for (n, d) in per_n_results.iter() {
        eprintln!("[NTT.5b]   N={n}: divergences={d}");
    }
    eprintln!("[NTT.5b] elapsed             : {elapsed:?}");
    if let Some((n, sx, h, x)) = first_divergence {
        eprintln!("[NTT.5b] first_divergence    : N={n} seed_ix={sx} host={h} hex={x}");
    }

    // ── Wall-clock matrix (informational; T_NTT5B_WALL_CLOCK_INFORMATIONAL) ──
    eprintln!("\n[NTT.5b] -- Wall-clock matrix (informational) --");
    let avg_host_us = (wall_host_total.as_micros() as f64) / (total_runs as f64);
    let avg_hex_us  = (wall_hex_total.as_micros()  as f64) / (total_runs as f64);
    eprintln!("[NTT.5b]   host avg per inner: {avg_host_us:.1} us");
    eprintln!("[NTT.5b]   hex  avg per inner: {avg_hex_us:.1} us");
    eprintln!("[NTT.5b]   ratio hex/host    : {:.2}x", avg_hex_us / avg_host_us);
    eprintln!("[NTT.5b]   NOTE: hex path expected SLOWER at small N (FastRPC marshalling");
    eprintln!("[NTT.5b]         per inner NTT dominates). NTT.6 long-context tiling is where");
    eprintln!("[NTT.5b]         the silicon win materializes — see spec §scope.");

    if total_divergences == 0 && total_runs == ns.len() as u32 * seeds_per_n {
        eprintln!("\n[NTT.5b] T_NTT5B_HOST_HEX_BIT_EXACT PASS  ({total_runs}/{total_runs} runs byte-exact)");
        std::process::exit(0);
    } else {
        eprintln!("\n[NTT.5b] T_NTT5B_HOST_HEX_BIT_EXACT FAIL  ({total_divergences}/{total_runs} divergences)");
        std::process::exit(1);
    }
}
