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

    // URI for the echo skel. qaic emits `<iface>_URI` in the generated header
    // as `"file:///lib<name>_skel.so?<iface>_skel_invoke&_modver=1.0"` (see
    // prior cohort `sp_hex.h:274`). We hardcode the equivalent so this smoke
    // binary doesn't link the qaic-generated stub (which would require
    // sp-daemon's full build chain). `&_dom=cdsp` selects the cDSP domain
    // (remote.h:142).
    let skel_uri = "file:///libshannonprime_echo_skel.so?echo_skel_invoke&_modver=1.0&_dom=cdsp";

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
        // Method 0 = echo.ping(in_buf, rout_buf).
        // Scalars: method=0, n_in=1, n_out=1.
        let scalars = make_scalars(0, 1, 1);
        let mut args = [
            RemoteArg { buf: RemoteBuf { pv: src.as_mut_ptr() as *mut c_void, nlen: src.len() } },
            RemoteArg { buf: RemoteBuf { pv: dst.as_mut_ptr() as *mut c_void, nlen: dst.len() } },
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

    if fails == 0 {
        eprintln!("[sp-dsp-smoke] ALL GATES PASS");
        std::process::exit(0);
    } else {
        eprintln!("[sp-dsp-smoke] {fails} gate(s) FAILED");
        std::process::exit(1);
    }
}
