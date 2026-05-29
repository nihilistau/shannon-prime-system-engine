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

    // Phase 2-L3.FG: android now bindgens + links the math-core C ABI (the
    // early sp_no_link skip is removed). bindgen runs on the host but targets
    // aarch64 via BINDGEN_EXTRA_CLANG_ARGS_aarch64-linux-android (--target +
    // --sysroot, set in .cargo/config.toml).

    // ── bindgen ────────────────────────────────────────────────────────────
    // Bindgen the frozen L1 header. sp_l1.h includes sp_model.h + sp_status.h
    // transitively; bindgen sees all three through -I.
    // Requires libclang. On Windows set:
    //   LIBCLANG_PATH=C:\Program Files\LLVM\bin
    let bindings = bindgen::Builder::default()
        .header(include_dir.join("sp/sp_l1.h").to_string_lossy())
        .clang_arg(format!("-I{}", include_dir.display()))
        .allowlist_function("sp_.*")
        .allowlist_type("sp_.*")
        .allowlist_var("SP_.*")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("bindgen failed — set LIBCLANG_PATH if libclang is not on PATH");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
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
}
