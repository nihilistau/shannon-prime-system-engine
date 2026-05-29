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

    // Skip bindgen + link on Android cross-compile.
    // The Android target is the §3-HX Sprint A FastRPC bridge (dsp_rpc.rs),
    // which does NOT depend on the L1 ABI bindings. Lib-side files (network,
    // ntt_ffi, dsp_rpc) verified to not include!(sp_bindings.rs).
    // Engine-on-Android (full forward path) is Phase 2-L3.FG scope.
    let target_os_early = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os_early == "android" {
        println!("cargo:rustc-cfg=sp_no_link");
        return;
    }

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
    // aarch64-android cross-compile: no pre-built device libs for CORE.
    // Type-check passes; link is Phase 2-L3.FG scope.
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "android" {
        println!("cargo:rustc-cfg=sp_no_link");
        return;
    }

    // SP_SYSTEM_BUILD_DIR: root of the math-core CMake build directory.
    //
    //   Standalone math-core (Linux / MinGW GCC):
    //     /path/to/shannon-prime-system/build/
    //     → libs at {dir}/core/{module}/libsp_{module}.a
    //
    //   Engine-embedded (Windows MinGW / MSVC, engine build-cpu):
    //     /path/to/shannon-prime-system-engine/build-cpu/lib/shannon-prime-system/
    //     → libs at {dir}/core/{module}/sp_{module}.lib
    //
    // Both follow the same core/{module}/ sub-path — only the lib file
    // prefix/extension differs, and Cargo resolves that from the target.
    let build_dir = env::var("SP_SYSTEM_BUILD_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            // Default: engine's MinGW build-cpu embedded submodule (Windows).
            engine_root.join("build-cpu/lib/shannon-prime-system")
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

    if target_os == "linux" {
        println!("cargo:rustc-link-lib=m");
    }
}
