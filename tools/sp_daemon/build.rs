use std::env;
use std::path::PathBuf;

// Math-core modules in link order (high-level → primitive).
// Each entry: (subdir_name, link_lib_name).
// Paths: {SP_SYSTEM_BUILD_DIR}/core/{subdir}/[lib]{link_lib_name}.[a|lib]
const MODULES: &[(&str, &str)] = &[
    ("session",          "sp_session"),
    ("forward",          "sp_forward"),
    ("forward_dispatch", "sp_forward_dispatch"),
    ("forward_kernels",  "sp_forward_kernels"),
    ("model",            "sp_model"),
    ("arena",            "sp_arena"),
    ("frobenius",        "sp_frobenius"),
    ("sieve",            "sp_sieve"),
    ("kste",             "sp_kste"),
    ("vht2",             "sp_vht2"),
    ("poly_ring",        "sp_poly_ring"),
    ("ntt_crt",          "sp_ntt_crt"),
    ("ok_arith",         "sp_ok_arith"),
    ("gguf",             "sp_gguf"),
    ("io_format",        "sp_io_format"),
    ("io_hash",          "sp_io_hash"),
    ("weight_dtype",     "sp_weight_dtype"),
];

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    // tools/sp_daemon/ → engine root is ../..
    let engine_root = manifest_dir.parent().unwrap().parent().unwrap();

    // ── Headers ────────────────────────────────────────────────────────────
    // sp_l1.h lives in the math-core submodule include tree.
    // Override with SP_SYSTEM_INCLUDE for out-of-tree builds.
    let include_dir = env::var("SP_SYSTEM_INCLUDE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| engine_root.join("lib/shannon-prime-system/include"));

    println!("cargo:rerun-if-env-changed=SP_SYSTEM_INCLUDE");
    println!("cargo:rerun-if-env-changed=SP_SYSTEM_BUILD_DIR");
    println!("cargo:rerun-if-changed={}", include_dir.join("sp/sp_l1.h").display());
    println!("cargo:rerun-if-changed={}", include_dir.join("sp/sp_model.h").display());
    println!("cargo:rerun-if-changed={}", include_dir.join("sp/sp_status.h").display());
    // M.5 (routing): KSTE encoder header is the routing primitive input.
    println!("cargo:rerun-if-changed={}", include_dir.join("sp/kste.h").display());

    // Phase 2-L3.FG: android now bindgens + links the math-core C ABI (the
    // early sp_no_link skip is removed). bindgen runs on the host but targets
    // aarch64 via BINDGEN_EXTRA_CLANG_ARGS_aarch64-linux-android (--target +
    // --sysroot, set in .cargo/config.toml).

    // ── bindgen ────────────────────────────────────────────────────────────
    // Bindgen the frozen L1 header. sp_l1.h includes sp_model.h + sp_status.h
    // transitively; bindgen sees all three through -I. M.5 routing also needs
    // sp_kste_encode + sp_kste_tree_t from sp/kste.h which is NOT transitively
    // included from sp_l1.h, so we add it via a synthesized wrapper header in
    // OUT_DIR.
    // Requires libclang. On Windows set:
    //   LIBCLANG_PATH=C:\Program Files\LLVM\bin
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let wrapper_path = out_dir.join("sp_bindgen_wrapper.h");
    std::fs::write(
        &wrapper_path,
        "/* Auto-generated bindgen wrapper. */\n\
         #include \"sp/sp_l1.h\"\n\
         /* M.5 (routing): KSTE encoder + Tier-0/Tier-1 dominance API. */\n\
         #include \"sp/kste.h\"\n",
    ).expect("could not write sp_bindgen_wrapper.h");

    let bindings = bindgen::Builder::default()
        .header(wrapper_path.to_string_lossy())
        .clang_arg(format!("-I{}", include_dir.display()))
        .allowlist_function("sp_.*")
        .allowlist_type("sp_.*")
        .allowlist_var("SP_.*")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("bindgen failed — set LIBCLANG_PATH if libclang is not on PATH");

    bindings
        .write_to_file(out_dir.join("sp_bindings.rs"))
        .expect("could not write sp_bindings.rs");

    // ── Link ───────────────────────────────────────────────────────────────
    // Phase 2-L3.FG: aarch64-android now links the cross-compiled math-core
    // (build-android-libs.bat → build-android-libs/core/<m>/libsp_<m>.a). The
    // old sp_no_link skip is gone; android falls through to the link loop below.
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    // SP_SYSTEM_BUILD_DIR: root of the math-core build directory.
    //
    //   Engine-embedded (Windows MinGW / MSVC, engine build-cpu):
    //     {engine}/build-cpu/lib/shannon-prime-system/  → core/{m}/sp_{m}.lib
    //   aarch64-android (Phase 2-L3.FG, build-android-libs.bat):
    //     {engine}/build-android-libs/                   → core/{m}/libsp_{m}.a
    //
    // Both resolve libs under the same core/{module}/ sub-path; only the file
    // prefix/extension differs, which Cargo derives from the target.
    let build_dir = env::var("SP_SYSTEM_BUILD_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            if target_os == "android" {
                engine_root.join("build-android-libs")
            } else {
                // Default: engine's MinGW build-cpu embedded submodule (Windows).
                engine_root.join("build-cpu/lib/shannon-prime-system")
            }
        });

    // CARGO_CFG_TARGET_ENV differentiates MSVC ("msvc") from MinGW/Linux ("gnu"/"").
    // On MSVC, cargo:rustc-link-lib=static stops propagating to [[bin]] targets when
    // the [lib] in this package (lib.rs) does not itself reference the C FFI symbols.
    // cargo:rustc-link-arg bypasses that boundary and reaches the binary link step
    // directly with a full absolute path — no /LIBPATH lookup required.
    let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();

    for (module_dir, lib_name) in MODULES {
        let search = build_dir.join("core").join(module_dir);
        println!("cargo:rustc-link-search=native={}", search.display());
        if target_env == "msvc" {
            let lib_path = search.join(format!("{lib_name}.lib"));
            println!("cargo:rustc-link-arg={}", lib_path.display());
        } else {
            println!("cargo:rustc-link-lib=static={lib_name}");
        }
    }

    if target_os == "linux" || target_os == "android" {
        println!("cargo:rustc-link-lib=m");
    }

    // ── Sprint WIRE-HEX: link the daemon-callable Hexagon V69 backend ──
    //
    // When the `wire_hex_backend` Cargo feature is on AND we're cross-compiling
    // for android, link the standalone static lib built by
    // `tools/sp_daemon/build-android-hex-backend.bat`:
    //   build-android-hex-backend/libsp_hex_daemon_backend.a
    //
    // That archive contains:
    //   - src/backends/hexagon/sp_hex_host.c (gemma3_forward_hexagon)
    //   - src/backends/cpu/cpu_overlay.c     (matmul/embed_row/as_f32/sp_kernels_read_env)
    //   - generated sp_hex_stub.c            (qaic FastRPC client stub)
    //   - tools/sp_daemon/c_backend/sp_daemon_hex_glue.c (the §6 dispatcher)
    //
    // Plus the FastRPC runtime libs from Hexagon SDK 5.5.6.0:
    //   libcdsprpc.so   (intra-device IPC; provides remote_handle64_* + remote_session_control)
    //   rpcmem.a        (rpcmem_alloc/free/init)
    //
    // Build order: `build-android-libs.bat` first (math-core archives the
    // hex glue depends on transitively), then `build-android-hex-backend.bat`,
    // then `cargo build --target aarch64-linux-android --features wire_hex_backend`.
    let wire_hex = env::var("CARGO_FEATURE_WIRE_HEX_BACKEND").is_ok();
    if wire_hex && target_os == "android" {
        let hex_lib_dir = env::var("SP_HEX_BACKEND_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| engine_root.join("build-android-hex-backend"));
        let hex_archive = hex_lib_dir.join("libsp_hex_daemon_backend.a");
        if !hex_archive.exists() {
            panic!(
                "WIRE-HEX: hex backend archive not found at {} — run build-android-hex-backend.bat first",
                hex_archive.display()
            );
        }
        println!("cargo:rustc-link-search=native={}", hex_lib_dir.display());
        println!("cargo:rustc-link-lib=static=sp_hex_daemon_backend");

        // FastRPC runtime libs (Hexagon SDK 5.5.6.0). HEXAGON_SDK_ROOT is set
        // by scripts/env/env-hexagon.bat; this build step expects it present.
        let hex_sdk = env::var("HEXAGON_SDK_ROOT")
            .unwrap_or_else(|_| String::from("C:/Qualcomm/Hexagon_SDK/5.5.6.0"));
        let rpcmem_dir  = format!("{hex_sdk}/ipc/fastrpc/rpcmem/prebuilt/android_aarch64");
        let cdsprpc_dir = format!("{hex_sdk}/ipc/fastrpc/remote/ship/android_aarch64");
        println!("cargo:rustc-link-search=native={rpcmem_dir}");
        println!("cargo:rustc-link-search=native={cdsprpc_dir}");
        // rpcmem.a is a non-standard archive name (not librpcmem.a). Use an
        // explicit -l:rpcmem.a directive so the GNU/Clang linker accepts it.
        println!("cargo:rustc-link-arg=-l:rpcmem.a");
        println!("cargo:rustc-link-lib=dylib=cdsprpc");
        println!("cargo:rerun-if-env-changed=SP_HEX_BACKEND_DIR");
        println!("cargo:rerun-if-env-changed=HEXAGON_SDK_ROOT");
        println!("cargo:warning=WIRE-HEX: linking libsp_hex_daemon_backend.a + libcdsprpc.so + rpcmem.a");
    }
}
