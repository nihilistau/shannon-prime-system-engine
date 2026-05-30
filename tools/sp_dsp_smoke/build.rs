// build.rs — Sprint NTT.0 Stage 1.
//
// Adds math-core's `sp_ntt_crt` static library to the link line so the
// NTT.0 smoke harness can call math-core's `ntt_forward` host-side as the
// T_NTT0_SCALAR_BIT_EXACT oracle.
//
// Mirrors the L3.FG link pattern from tools/sp_daemon/build.rs:
//   - For aarch64-android target: link build-android-libs/core/ntt_crt/libsp_ntt_crt.a
//   - For Windows MSVC host:      link build-cpu/lib/.../core/ntt_crt/sp_ntt_crt.lib
//
// NOTE: math-core's `sp_ntt_crt` library has NO transitive math-core deps
// (just stdint + libc); we link it standalone.  If a future change introduces
// transitive deps, this build.rs needs to grow a MODULES table like
// sp_daemon's; the link error would surface clearly.

use std::env;
use std::path::PathBuf;

fn main() {
    // Cargo manifest dir = tools/sp_dsp_smoke
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let engine_root = manifest_dir.parent().unwrap().parent().unwrap();

    println!("cargo:rerun-if-env-changed=SP_SYSTEM_BUILD_DIR");
    println!("cargo:rerun-if-env-changed=SP_NTT_CRT_LIB_DIR");

    let target_os  = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    // The NTT.0 smoke is `cfg(target_os = "android")` gated.  Host (Windows
    // MSVC / Linux gcc) builds compile a stub `main` only and do NOT call
    // the FFI symbols; emit no link directives.  This matches the existing
    // sp_dsp_smoke pattern (libloading is also cfg-gated to android).
    if target_os != "android" {
        return;
    }

    let build_dir = env::var("SP_SYSTEM_BUILD_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| engine_root.join("build-android-libs"));

    let search = build_dir.join("core").join("ntt_crt");
    println!("cargo:rustc-link-search=native={}", search.display());
    println!("cargo:rustc-link-lib=static=sp_ntt_crt");
    println!("cargo:rustc-link-lib=m");
}
