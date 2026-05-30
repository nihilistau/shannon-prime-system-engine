// K.2-spike — build.rs
//
// Compiles src/sp_npu_shim.c against the QNN SDK headers on the host. The shim
// implements 4 entrypoints (sp_qnn_init, sp_qnn_run_add_smoke, sp_qnn_shutdown,
// sp_qnn_last_error) that wrap the multi-level QnnInterface dispatch lifecycle
// + the Qnn_Tensor_t / Qnn_OpConfig_t struct construction in C, where the
// header definitions live. Rust only calls the 4 clean C entrypoints — no
// re-derivation of the QNN data-struct shapes in Rust.
//
// The shim does NOT link against libQnnHtp.so; instead it dlopens it at
// runtime (matching the libloading-bridge approach in
// tools/sp_dsp_smoke/src/dsp_rpc.rs). This avoids static linking against the
// SDK + lets the runtime SDK version differ from the build-time headers.
//
// Required env: QNN_SDK_ROOT — points at C:\Qualcomm\AIStack\QAIRT\<version>\
//
// The build is only enabled for target_os = "android" so host-side `cargo
// check` on Windows doesn't need the SDK.

fn main() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "android" {
        // Host build — skip the C compile + run a stub main from Rust.
        println!("cargo:warning=sp-npu-spike: target_os={target_os}, skipping QNN C shim build (Android-only)");
        return;
    }

    let qnn_root = std::env::var("QNN_SDK_ROOT").unwrap_or_else(|_| {
        // Default to Knack's installed v2.45 SDK if env not set.
        r"C:\Qualcomm\AIStack\QAIRT\2.45.40.260406".to_string()
    });
    let qnn_include = format!("{qnn_root}\\include\\QNN");
    println!("cargo:rerun-if-env-changed=QNN_SDK_ROOT");
    println!("cargo:rerun-if-changed=src/sp_npu_shim.c");
    println!("cargo:warning=sp-npu-spike: using QNN_SDK_ROOT={qnn_root}");

    cc::Build::new()
        .file("src/sp_npu_shim.c")
        .include(&qnn_include)
        // Android NDK r27d aarch64-linux-android21 toolchain is configured via
        // .cargo/config.toml for cargo; the cc crate auto-detects through
        // CC_aarch64-linux-android env var or by interrogating the linker
        // setting. Set CC manually if cc fails to autodetect.
        .flag_if_supported("-Wno-unused-parameter")
        .flag_if_supported("-Wno-missing-field-initializers")
        .define("QNN_API", "")
        .define("QNN_INTERFACE", "")
        .compile("sp_npu_shim");
}
