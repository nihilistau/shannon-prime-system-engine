pub mod network;
pub mod ntt_ffi;

// §4-MeMo Sprint M.2 — Zero-copy dialogue loop (Grounding → Entity ID →
// Synthesis) + Spinor receipt envelope. Host-build-safe: the module itself
// compiles on host (struct + tests); the L1-driven `run_dialogue()` helper
// lives in the android binary (sp_memo_m2_dialogue_smoke.rs).
pub mod dialogue;

// Chat-integration: the daemon-callable MeMo dialogue runner lives in the
// binary crate (src/dialogue_runner.rs declared from main.rs) because it
// depends on session/tokenizer modules that are binary-crate-local. The
// host-safe primitives (SpinorReceipt, DialoguePool, argmax) stay in
// `dialogue` above.

#[cfg(target_os = "android")]
pub mod dsp_rpc;

// §4-NTT Sprint NTT.5b — Hexagon backend dispatch trampolines for math-core's
// sp_compute_ntt_dispatch_fn ABI. Routes the inner Bluestein NTT calls through
// FastRPC method 17 (forward, VTCM-aware HVX) and method 18 (HVX INTT). Held
// in AppState as Option<Arc<ComputeBackend>>; registered with the Memory L1
// session at daemon startup when SP_ENGINE_NTT_ATTN_HEX=1 is set.
#[cfg(target_os = "android")]
pub mod ntt_hex_dispatch;

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

// M.4 (ledger): PoUW receipt ledger + mesh replay primitive. Host-buildable
// (file I/O + SpinorReceipt::as_bytes round-trip, no L1 ABI needed); the
// android smoke binary sp_memo_m4_ledger_smoke drives it on the S22U.
pub mod pouw_ledger;
