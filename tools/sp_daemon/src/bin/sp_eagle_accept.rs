//! sp_eagle_accept — LIVE EAGLE/MTP single-token acceptance probe on the served 12B.
//!
//! Loads the SAME 12B the daemon serves, opens the resident KV cache, loads the
//! gemma4-assistant draft (gemma4_draft_open), then greedy-decodes the target while at
//! each step (a) capturing the post-output_norm feature (gemma4_kv_capture_feat), (b)
//! reading the target's greedy next token from gemma4_kv_decode_logits, and (c) running
//! the draft (gemma4_draft_step) seeded by that feature + last token, reading the LIVE
//! KV ring. Accept rate = fraction where draft argmax == target greedy. This is the
//! payoff metric the whole MTP campaign drives toward — the draft's single-token agreement
//! with the served model. Tweak live via SP_DRAFT_ASCALE=one|rsqrt (no rebuild).
//!
//! Build: cargo build --release --features wire_cuda_backend --bin sp_eagle_accept
//! Run:   set SP_CUDA_DECODE_INT8=1, STOP the serving daemon (frees the GPU), then
//!        sp_eagle_accept <12b.sp-model> <12b.sp-tokenizer> <draft.gguf> [n_decode]
#![cfg(feature = "wire_cuda_backend")]

mod ffi {
    #![allow(non_upper_case_globals, non_camel_case_types, non_snake_case, dead_code, clippy::all)]
    include!(concat!(env!("OUT_DIR"), "/sp_bindings.rs"));
}
use ffi::sp_status_SP_OK;
use std::ffi::{c_void, CString};
use std::os::raw::{c_char, c_int};
use std::ptr;
use std::sync::atomic::AtomicI32;
use std::sync::Arc;

#[repr(C)]
struct sp_g4_kv {
    _opaque: [u8; 0],
}
extern "C" {
    fn gemma4_kv_open(m: *const c_void, pmax: c_int) -> *mut sp_g4_kv;
    fn gemma4_kv_prefill(s: *mut sp_g4_kv, toks: *const i32, n: c_int) -> c_int;
    fn gemma4_kv_decode_logits(s: *mut sp_g4_kv, token: i32, logits: *mut f32) -> c_int;
    fn gemma4_kv_capture_feat(s: *mut sp_g4_kv, feat: *mut f32) -> c_int;
    fn gemma4_kv_close(s: *mut sp_g4_kv);
    fn gemma4_draft_open(path: *const c_char) -> c_int;
    fn gemma4_draft_step(
        s: *mut sp_g4_kv,
        feat: *const f32,
        token: c_int,
        out_token: *mut c_int,
        out_hnext: *mut f32,
    ) -> c_int;
    fn gemma4_draft_close();
}
fn last_err() -> String {
    unsafe { std::ffi::CStr::from_ptr(ffi::sp_last_error()) }.to_string_lossy().into_owned()
}
fn argmax(v: &[f32]) -> i32 {
    let (mut bi, mut bv) = (0usize, v[0]);
    for (i, &x) in v.iter().enumerate() { if x > bv { bv = x; bi = i; } }
    bi as i32
}

fn main() {
    if std::env::var("SP_CUDA_DECODE_INT8").ok().as_deref() != Some("1") {
        std::env::set_var("SP_CUDA_DECODE_INT8", "1");
        eprintln!("[probe] SP_CUDA_DECODE_INT8 not set — defaulting to 1 (tied head)");
    }
    let a: Vec<String> = std::env::args().collect();
    let model_path = a.get(1).cloned().or_else(|| std::env::var("SP_MODEL_PATH").ok())
        .expect("usage: sp_eagle_accept <12b.sp-model> <12b.sp-tokenizer> <draft.gguf> [n]");
    let tok_path = a.get(2).cloned().or_else(|| std::env::var("SP_TOKENIZER_PATH").ok())
        .expect("usage: sp_eagle_accept <12b.sp-model> <12b.sp-tokenizer> <draft.gguf> [n]");
    let draft_path = a.get(3).cloned().or_else(|| std::env::var("SP_DRAFT_GGUF").ok())
        .expect("need draft gguf path (arg 3 or SP_DRAFT_GGUF)");
    let n_decode: usize = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(48);

    let prompt: Vec<i32> = vec![2, 106, 1645, 108, 1841, 603, 573, 6996, 576, 6081, 235336, 108, 106, 2516, 108];
    let pmax: i32 = (prompt.len() + n_decode + 8) as i32;
    println!("sp_eagle_accept — LIVE single-token acceptance probe");
    println!("  target : {model_path}");
    println!("  draft  : {draft_path}");
    println!("  prompt : {} tokens   n_decode : {n_decode}   ascale : {}",
             prompt.len(), std::env::var("SP_DRAFT_ASCALE").unwrap_or_else(|_| "rsqrt(default)".into()));

    // ── load target 12B (same as the daemon) ──
    let model_c = CString::new(model_path).unwrap();
    let tok_c = CString::new(tok_path).unwrap();
    let mut model_ptr: *mut ffi::sp_model = ptr::null_mut();
    assert_eq!(unsafe { ffi::sp_model_load(model_c.as_ptr(), tok_c.as_ptr(), &mut model_ptr) },
               sp_status_SP_OK, "sp_model_load failed");
    let mut arch: ffi::sp_arch_info = unsafe { std::mem::zeroed() };
    assert_eq!(unsafe { ffi::sp_model_arch(model_ptr, &mut arch) }, sp_status_SP_OK, "sp_model_arch failed");
    let vocab = arch.vocab_size as usize;
    let hidden = arch.hidden_dim as usize;
    println!("  arch   : vocab={vocab} n_layers={} hidden={hidden}", arch.n_layers);

    let cancel = Arc::new(AtomicI32::new(0));
    let cfg = ffi::sp_session_config {
        max_context: pmax as usize, deterministic: 1, arm_bank_kb: 0,
        sieve_capacity: 0, flags: 0, precision_override: 0,
    };
    let mut sess: *mut ffi::sp_session = ptr::null_mut();
    assert_eq!(unsafe { ffi::sp_session_create(model_ptr, &cfg, cancel.as_ptr() as *mut c_int, &mut sess) },
               sp_status_SP_OK, "sp_session_create failed");
    let qm = unsafe { ffi::sp_session_qwen3_model(sess) };
    assert!(!qm.is_null(), "sp_session_qwen3_model NULL");

    let s = unsafe { gemma4_kv_open(qm, pmax) };
    if s.is_null() { panic!("gemma4_kv_open failed: {}", last_err()); }
    let draft_c = CString::new(draft_path).unwrap();
    if unsafe { gemma4_draft_open(draft_c.as_ptr()) } != 0 { panic!("gemma4_draft_open failed: {}", last_err()); }

    // prefill all-but-last so the first decode_logits processes the last prompt token cleanly
    let np = prompt.len();
    assert_eq!(unsafe { gemma4_kv_prefill(s, prompt.as_ptr(), (np - 1) as c_int) }, 0, "prefill failed");
    let mut last = prompt[np - 1];

    let mut feat = vec![0.0f32; hidden];
    let mut logits = vec![0.0f32; vocab];
    let (mut accept, mut count) = (0usize, 0usize);
    let mut samples: Vec<(i32, i32)> = Vec::new();
    for _ in 0..n_decode {
        assert_eq!(unsafe { gemma4_kv_capture_feat(s, feat.as_mut_ptr()) }, 0, "capture_feat failed");
        assert_eq!(unsafe { gemma4_kv_decode_logits(s, last, logits.as_mut_ptr()) }, 0,
                   "decode_logits failed");
        let tgt = argmax(&logits);
        let mut dtok: c_int = -1;
        let rc = unsafe { gemma4_draft_step(s, feat.as_ptr(), last, &mut dtok, ptr::null_mut()) };
        assert_eq!(rc, 0, "gemma4_draft_step failed: {}", last_err());
        if dtok == tgt { accept += 1; }
        if samples.len() < 12 { samples.push((dtok, tgt)); }
        count += 1;
        last = tgt;
    }
    println!("  samples (draft -> target): {:?}", samples);
    let rate = 100.0 * accept as f64 / count as f64;
    println!("  ACCEPT(K=1): {accept}/{count} = {rate:.1}%");
    println!("G-EAGLE-ACCEPT-LIVE: {}", if accept > 0 { "RAN" } else { "RAN(0% — tweak ascale/rope)" });

    unsafe {
        gemma4_draft_close();
        gemma4_kv_close(s);
        ffi::sp_session_destroy(sess);
        ffi::sp_model_unload(model_ptr);
    }
}
