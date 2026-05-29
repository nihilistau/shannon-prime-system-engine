//! §3-HX Sprint A smoke test — exercises FastRpcSession against the echo skel
//! on the connected S22U device via FastRPC + Unsigned PD (Path B).
//!
//! Build:
//!   cargo build --target aarch64-linux-android --release
//!
//! Deploy + run (the deploy-s22u-echo-skel.bat helper):
//!   adb push libshannonprime_echo_skel.so /data/local/tmp/
//!   adb push test_dsp_rpc                  /data/local/tmp/
//!   adb shell "chmod +x /data/local/tmp/test_dsp_rpc"
//!   adb shell "ADSP_LIBRARY_PATH=\"/data/local/tmp;\" /data/local/tmp/test_dsp_rpc"
//!
//! Pass criteria:
//!   T_RPC_ECHO_1 (16 B) bitwise OK + T_RPC_ECHO_2 (4 KB) + T_RPC_ECHO_3 (1 MB)
//!   + UNSIGNED_PD_ADMITTED (the session opened at all = Path B works)
//!   exit code 0
//! Fail:
//!   exit code != 0; stderr names the failed gate.

// On host x86 builds, this binary does nothing but exit 0 with a friendly
// message — there's no libcdsprpc.so to dynamic-link against.
#[cfg(not(target_os = "android"))]
fn main() {
    eprintln!("sp-dsp-smoke: host build (target_os != android) — skipped");
    eprintln!("Build with: cargo build --target aarch64-linux-android --release");
}

#[cfg(target_os = "android")]
mod dsp_rpc;

#[cfg(target_os = "android")]
fn main() {
    use dsp_rpc::{make_scalars, FastRpcSession, RemoteArg, RemoteBuf, SpErr};
    use std::ffi::c_void;
    use std::time::Instant;

    // URI per qaic-generated sp_echo_URI macro (sp_echo.h line 1):
    //   "file:///libsp_echo_skel.so?sp_echo_skel_handle_invoke&_modver=1.0"
    // The `_handle_invoke` suffix comes from `interface sp_echo : remote_handle64`
    // (multi-domain) — the canonical pattern per SDK S22U workspace.
    // `&_dom=cdsp` selects cDSP domain (remote.h:142).
    let skel_uri = "file:///libsp_echo_skel.so?sp_echo_skel_handle_invoke&_modver=1.0&_dom=cdsp";

    eprintln!("[sp-dsp-smoke] opening FastRpcSession (Unsigned PD admission, Path B)...");
    let session = match FastRpcSession::new(skel_uri) {
        Ok(s) => {
            eprintln!("[sp-dsp-smoke] UNSIGNED_PD_ADMITTED — session open");
            s
        }
        Err(SpErr::UnsignedPdReject(rc)) => {
            eprintln!("[sp-dsp-smoke] FAIL: UnsignedPdReject({rc:#x}) — \
                       remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE) rejected.\n\
                       Likely causes: device libcdsprpc.so too old, or unsigned PD \
                       support disabled by vendor.fastrpc.process.attrs.");
            std::process::exit(1);
        }
        Err(SpErr::SignatureMismatch(rc)) => {
            eprintln!("[sp-dsp-smoke] FAIL: SignatureMismatch({rc:#x}) — see \
                       reference-signed-pd-developer-path five-cause map. \
                       Path B admission ran but handle_open still got AEE_ERPC.");
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("[sp-dsp-smoke] FAIL: {e:?}");
            std::process::exit(1);
        }
    };

    let mut fails = 0;
    for (name, size) in [
        ("T_RPC_ECHO_1", 16usize),
        ("T_RPC_ECHO_2", 4096),
        ("T_RPC_ECHO_3", 1024 * 1024),
    ] {
        let mut src: Vec<u8> = (0..size).map(|i| ((i * 0x9E + 0x37) & 0xFF) as u8).collect();
        let mut dst: Vec<u8> = vec![0u8; size];
        // Per qaic-emitted sp_echo_stub.c:297-319 marshalling pattern for
        // `ping(in seq<octet>, rout seq<octet>)`:
        //   - 3 remote_args total
        //   - arg[0] = primIn buffer (8 B: [in_len:u32, out_len:u32]) — counted
        //              as 1st input buffer
        //   - arg[1] = in_buf data — 2nd input buffer
        //   - arg[2] = out_buf data — 1st output buffer
        //   - Scalars: REMOTE_SCALARS_MAKEX(0, method=2, n_in=2, n_out=1, 0, 0)
        let mut prim_in: [u32; 2] = [size as u32, size as u32];
        let scalars = make_scalars(2, 2, 1);
        let mut args = [
            RemoteArg { buf: RemoteBuf {
                pv: prim_in.as_mut_ptr() as *mut c_void,
                nlen: 8,
            }},
            RemoteArg { buf: RemoteBuf {
                pv: src.as_mut_ptr() as *mut c_void,
                nlen: src.len(),
            }},
            RemoteArg { buf: RemoteBuf {
                pv: dst.as_mut_ptr() as *mut c_void,
                nlen: dst.len(),
            }},
        ];

        match session.invoke(scalars, &mut args) {
            Ok(()) if dst == src => eprintln!("[sp-dsp-smoke] {name} ({size} B) PASS"),
            Ok(()) => {
                eprintln!("[sp-dsp-smoke] {name} FAIL: bytes diverged at idx \
                           {:?}", dst.iter().zip(src.iter()).position(|(d, s)| d != s));
                fails += 1;
            }
            Err(e) => {
                eprintln!("[sp-dsp-smoke] {name} FAIL invoke: {e:?}");
                fails += 1;
            }
        }
    }

    drop(session);  // exercises Drop / remote_handle_close
    eprintln!("[sp-dsp-smoke] session closed cleanly");

    // ── T_RPC_LEAK_1: 1000-cycle create/invoke/drop ─────────────────────────
    // Each iter: open session → 1 invoke (16 B) → drop. Verifies pool/handle
    // cleanup is leak-free across many cycles. Bench wall ≈ N × ~ms per cycle.
    eprintln!("[sp-dsp-smoke] T_RPC_LEAK_1: running 1000 create/invoke/drop cycles...");
    let mut leak_fails = 0;
    for iter in 0..1000 {
        let s = match FastRpcSession::new(skel_uri) {
            Ok(s) => s,
            Err(e) => { eprintln!("  leak cycle {iter} new: {e:?}"); leak_fails += 1; break; }
        };
        let mut src = [0xA5u8; 16];
        let mut dst = [0u8; 16];
        let mut prim_in: [u32; 2] = [16, 16];
        let mut args = [
            RemoteArg { buf: RemoteBuf { pv: prim_in.as_mut_ptr() as *mut c_void, nlen: 8 }},
            RemoteArg { buf: RemoteBuf { pv: src.as_mut_ptr() as *mut c_void, nlen: 16 }},
            RemoteArg { buf: RemoteBuf { pv: dst.as_mut_ptr() as *mut c_void, nlen: 16 }},
        ];
        if let Err(e) = s.invoke(make_scalars(2, 2, 1), &mut args) {
            eprintln!("  leak cycle {iter} invoke: {e:?}");
            leak_fails += 1;
            break;
        }
        if dst != src {
            eprintln!("  leak cycle {iter}: bytes diverged");
            leak_fails += 1;
            break;
        }
        // s drops here
    }
    if leak_fails == 0 {
        eprintln!("[sp-dsp-smoke] T_RPC_LEAK_1 (1000 cycles) PASS");
    } else {
        eprintln!("[sp-dsp-smoke] T_RPC_LEAK_1 FAIL: {leak_fails} cycles broke");
        fails += 1;
    }

    // ═════════════════════════════════════════════════════════════════════
    // §3-HX Sprint B — DmaBuffer zero-copy via rpcmem_alloc
    // ═════════════════════════════════════════════════════════════════════
    eprintln!("\n[sp-dsp-smoke] ═══ Sprint B: DmaBuffer tests ═══");

    // Re-open a session for the Sprint B suite (previous one was dropped).
    let session = match FastRpcSession::new(skel_uri) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[sp-dsp-smoke] Sprint B session reopen FAIL: {e:?}");
            std::process::exit(1);
        }
    };

    // T_DMA_ALLOC_FREE: alloc(1 MB) → drop → no crash
    {
        let buf = session.alloc_dma(1024 * 1024);
        match buf {
            Ok(b) => {
                eprintln!("[sp-dsp-smoke] T_DMA_ALLOC_FREE (1 MB) PASS — ptr={:p}", b.as_ptr());
                drop(b);
            }
            Err(e) => {
                eprintln!("[sp-dsp-smoke] T_DMA_ALLOC_FREE FAIL: {e:?}");
                fails += 1;
            }
        }
    }

    // T_DMA_PING_BITWISE: invoke sp_echo_ping with DmaBuffer-backed in/out.
    // EXACT-SIZE DISCIPLINE: alloc_dma(size) MUST match the IDL Len → off-by-
    // one yields AEE_EUNSUPPORTED silent fail per reference-hexagon-working-setup.
    for (name, size) in [
        ("T_DMA_PING_16B",  16usize),
        ("T_DMA_PING_4KB",  4096),
        ("T_DMA_PING_1MB",  1024 * 1024),
    ] {
        let mut in_buf  = match session.alloc_dma(size) {
            Ok(b) => b,
            Err(e) => { eprintln!("[sp-dsp-smoke] {name} alloc in_buf: {e:?}"); fails += 1; continue; }
        };
        let mut out_buf = match session.alloc_dma(size) {
            Ok(b) => b,
            Err(e) => { eprintln!("[sp-dsp-smoke] {name} alloc out_buf: {e:?}"); fails += 1; continue; }
        };
        // Pattern-fill in_buf, zero out_buf.
        for (i, b) in in_buf.as_mut_slice().iter_mut().enumerate() {
            *b = ((i * 0x9E + 0x37) & 0xFF) as u8;
        }
        for b in out_buf.as_mut_slice().iter_mut() { *b = 0; }

        let mut prim_in: [u32; 2] = [size as u32, size as u32];
        let mut args = [
            RemoteArg { buf: RemoteBuf { pv: prim_in.as_mut_ptr() as *mut c_void, nlen: 8 }},
            RemoteArg { buf: RemoteBuf { pv: in_buf.as_mut_ptr() as *mut c_void, nlen: size }},
            RemoteArg { buf: RemoteBuf { pv: out_buf.as_mut_ptr() as *mut c_void, nlen: size }},
        ];

        match session.invoke(make_scalars(2, 2, 1), &mut args) {
            Ok(()) if out_buf.as_slice() == in_buf.as_slice() => {
                eprintln!("[sp-dsp-smoke] {name} ({size} B) PASS");
            }
            Ok(()) => {
                eprintln!("[sp-dsp-smoke] {name} FAIL: bytes diverged");
                fails += 1;
            }
            Err(e) => {
                eprintln!("[sp-dsp-smoke] {name} FAIL invoke: {e:?}");
                fails += 1;
            }
        }
        // in_buf/out_buf drop here → rpcmem_free
    }

    // T_DMA_VS_HEAP: 1000-iter 1 MB ping wall comparison.
    // Per Sprint A finding: 1 MB ping via Vec<u8> takes ~ms (FastRPC marshal
    // copy).  rpcmem-backed should be measurably faster.
    {
        const N: usize = 1024 * 1024;
        const ITERS: u32 = 1000;

        // Pre-allocate to keep alloc cost out of the timed loop.
        let mut heap_in  = vec![0u8; N];
        let mut heap_out = vec![0u8; N];
        for (i, b) in heap_in.iter_mut().enumerate() {
            *b = ((i * 0xA7 + 0x11) & 0xFF) as u8;
        }
        let mut dma_in  = session.alloc_dma(N).unwrap();
        let mut dma_out = session.alloc_dma(N).unwrap();
        dma_in.as_mut_slice().copy_from_slice(&heap_in);

        let mut prim_in: [u32; 2] = [N as u32, N as u32];

        // Heap path
        let t0 = Instant::now();
        for _ in 0..ITERS {
            let mut args = [
                RemoteArg { buf: RemoteBuf { pv: prim_in.as_mut_ptr() as *mut c_void, nlen: 8 }},
                RemoteArg { buf: RemoteBuf { pv: heap_in.as_mut_ptr() as *mut c_void, nlen: N }},
                RemoteArg { buf: RemoteBuf { pv: heap_out.as_mut_ptr() as *mut c_void, nlen: N }},
            ];
            if session.invoke(make_scalars(2, 2, 1), &mut args).is_err() {
                eprintln!("[sp-dsp-smoke] T_DMA_VS_HEAP heap-leg invoke FAIL"); fails += 1; break;
            }
        }
        let heap_wall = t0.elapsed();

        // DMA path
        let t0 = Instant::now();
        for _ in 0..ITERS {
            let mut args = [
                RemoteArg { buf: RemoteBuf { pv: prim_in.as_mut_ptr() as *mut c_void, nlen: 8 }},
                RemoteArg { buf: RemoteBuf { pv: dma_in.as_mut_ptr() as *mut c_void, nlen: N }},
                RemoteArg { buf: RemoteBuf { pv: dma_out.as_mut_ptr() as *mut c_void, nlen: N }},
            ];
            if session.invoke(make_scalars(2, 2, 1), &mut args).is_err() {
                eprintln!("[sp-dsp-smoke] T_DMA_VS_HEAP dma-leg invoke FAIL"); fails += 1; break;
            }
        }
        let dma_wall = t0.elapsed();

        let ratio = dma_wall.as_secs_f64() / heap_wall.as_secs_f64();
        eprintln!("[sp-dsp-smoke] T_DMA_VS_HEAP (1 MB × {ITERS} iter): heap {heap_wall:?}  dma {dma_wall:?}  ratio {ratio:.3}× (<1.0 = dma faster)");
    }

    // T_DMA_LEAK_1: 1000 alloc/use/drop cycles with verify
    {
        let mut leak_fails = 0;
        for iter in 0..1000 {
            let mut a = match session.alloc_dma(64) {
                Ok(b) => b,
                Err(e) => { eprintln!("  dma leak cycle {iter}: {e:?}"); leak_fails += 1; break; }
            };
            let mut b = match session.alloc_dma(64) {
                Ok(b) => b,
                Err(e) => { eprintln!("  dma leak cycle {iter}: {e:?}"); leak_fails += 1; break; }
            };
            for (i, x) in a.as_mut_slice().iter_mut().enumerate() { *x = (i & 0xFF) as u8; }
            for x in b.as_mut_slice().iter_mut() { *x = 0; }
            let mut prim_in: [u32; 2] = [64, 64];
            let mut args = [
                RemoteArg { buf: RemoteBuf { pv: prim_in.as_mut_ptr() as *mut c_void, nlen: 8 }},
                RemoteArg { buf: RemoteBuf { pv: a.as_mut_ptr() as *mut c_void, nlen: 64 }},
                RemoteArg { buf: RemoteBuf { pv: b.as_mut_ptr() as *mut c_void, nlen: 64 }},
            ];
            if session.invoke(make_scalars(2, 2, 1), &mut args).is_err()
                || a.as_slice() != b.as_slice() {
                leak_fails += 1; break;
            }
        }
        if leak_fails == 0 {
            eprintln!("[sp-dsp-smoke] T_DMA_LEAK_1 (1000 alloc/invoke/drop cycles) PASS");
        } else {
            eprintln!("[sp-dsp-smoke] T_DMA_LEAK_1 FAIL: {leak_fails} cycles broke");
            fails += 1;
        }
    }

    drop(session);
    eprintln!("[sp-dsp-smoke] Sprint B session closed cleanly");

    if fails == 0 {
        eprintln!("[sp-dsp-smoke] ALL GATES PASS");
        std::process::exit(0);
    } else {
        eprintln!("[sp-dsp-smoke] {fails} gate(s) FAILED");
        std::process::exit(1);
    }
}
