pub mod network;
pub mod ntt_ffi;

// §4-MeMo Sprint M.2 — Zero-copy dialogue loop (Grounding → Entity ID →
// Synthesis) + Spinor receipt envelope. Host-build-safe: the module itself
// compiles on host (struct + tests); the L1-driven `run_dialogue()` helper
// lives in the android binary (sp_memo_m2_dialogue_smoke.rs).
pub mod dialogue;

#[cfg(target_os = "android")]
pub mod dsp_rpc;

// §4-MeMo Sprint M.1 — re-export the L1 C ABI bindings from the lib crate so
// android binaries (e.g. sp_memo_m1_smoke) pick up the link dependency on
// the math-core static libs via the lib's own link graph. On host this is
// harmless; on android this is what propagates -lsp_session etc. through
// to per-binary link steps (cargo's `rustc-link-lib` from build.rs reaches
// binaries only via the lib crate's symbol graph).
#[cfg(target_os = "android")]
#[allow(non_upper_case_globals, non_camel_case_types, non_snake_case, dead_code, clippy::all)]
pub mod ffi_l1 {
    include!(concat!(env!("OUT_DIR"), "/sp_bindings.rs"));
}

// M.5 (routing): KSTE-routed sparse Memory activation primitive. Public from
// the lib crate so sp_memo_m5_routing_smoke binary can `use sp_daemon::memo_routing`.
// Host build also exposes this (build.rs wires KSTE encoder symbols into the lib's
// own link closure via the bindgen wrapper header).
pub mod memo_routing;
