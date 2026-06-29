//! SP_EAGLE_ACCEPT — LIVE EAGLE/MTP single-token acceptance probe, as a daemon
//! one-shot MODE (not a standalone bin): the token-management contract
//! (apply_template_ids / suppress_token_ids / eos_ids) lives in the binary-private
//! tokenizer+sampler modules, so the framework-faithful probe must run here, the
//! same way SP_NIGHTSHIFT_OFFLINE / SP_KAIROS_ALPHA do. Mirrors nightshift_curator's
//! SpModel+SptbTokenizer+session+gemma4_kv setup.
//!
//! Loads the served 12B + the gemma4-assistant draft, greedy-decodes the target while
//! (a) framing the prompt with apply_template_ids (real <|turn>=105 / <turn|>=106 ids),
//! (b) suppressing soft/control tokens (suppress_token_ids: 258882/258883 + pipe-controls)
//! before argmax on BOTH the target and the draft, (c) stopping on eos_ids. Accept rate =
//! fraction where the draft's argmax == the target's greedy next token.
//!
//! Run: SP_EAGLE_ACCEPT=1 SP_MODEL_PATH=…b1.sp-model SP_TOKENIZER_PATH=…b1.sp-tokenizer \
//!      SP_DRAFT_GGUF=…gemma-4-12b-it-F16-MTP.gguf [SP_EAGLE_N=48] [SP_DRAFT_ASCALE=one] sp-daemon
#![cfg(feature = "wire_cuda_backend")]

use std::ffi::{c_void, CString};
use std::os::raw::{c_char, c_int};
use std::sync::atomic::AtomicI32;
use std::sync::Arc;

use crate::session::{SpModel, SpSession};
use crate::tokenizer::{Message, SptbTokenizer};

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
        suppress: *const c_int,
        n_suppress: c_int,
        out_token: *mut c_int,
        out_hnext: *mut f32,
    ) -> c_int;
    fn gemma4_draft_close();
}

fn argmax(v: &[f32]) -> i32 {
    let (mut bi, mut bv) = (0usize, v[0]);
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            bi = i;
        }
    }
    bi as i32
}

pub fn run_eagle_accept(
    model_path: &str,
    tok_path: &str,
    draft_gguf: &str,
    n_decode: usize,
) -> Result<(usize, usize), String> {
    std::env::set_var("SP_CUDA_DECODE_INT8", "1"); // tied head (required by gemma4_kv_open)

    let model = SpModel::load(model_path, tok_path)?;
    let arch = model.arch_info()?;
    let vocab = arch.vocab_size as usize;
    let hidden = arch.hidden_dim as usize;
    let tok = SptbTokenizer::build(&model, arch.arch_id, tok_path)?;
    let cancel = Arc::new(AtomicI32::new(0));
    let mut session = SpSession::create(&model, cancel)?;
    let sraw = session.raw_ptr() as *mut sp_daemon::ffi_l1::sp_session;
    let qm = (unsafe { sp_daemon::ffi_l1::sp_session_qwen3_model(sraw) }) as *const c_void;
    if qm.is_null() {
        return Err("eagle: sp_session_qwen3_model NULL".to_string());
    }

    // FRAME via the token-management contract (real <|turn>/<turn|> ids, not literal strings).
    let msgs = vec![Message {
        role: "user".to_string(),
        content: "What is the capital of France?".to_string(),
    }];
    let prompt = tok.apply_template_ids(&msgs)?;
    if prompt.len() < 2 {
        return Err(format!("eagle: framed prompt too short ({})", prompt.len()));
    }
    // SUPPRESS set (soft/control tokens: 258882/258883 + pipe-controls + named specials).
    let suppress: Vec<i32> = tok.suppress_token_ids();
    eprintln!(
        "[eagle] target={model_path}\n[eagle] draft={draft_gguf}\n[eagle] prompt={} tok  vocab={vocab} hidden={hidden}  suppress={} ids  eos={:?}  ascale={}",
        prompt.len(), suppress.len(), tok.eos_ids,
        std::env::var("SP_DRAFT_ASCALE").unwrap_or_else(|_| "rsqrt(default)".into())
    );

    let pmax = (prompt.len() + n_decode + 8) as c_int;
    let s = unsafe { gemma4_kv_open(qm, pmax) };
    if s.is_null() {
        return Err("eagle: gemma4_kv_open NULL".to_string());
    }
    let dc = CString::new(draft_gguf).unwrap();
    if unsafe { gemma4_draft_open(dc.as_ptr()) } != 0 {
        unsafe { gemma4_kv_close(s) };
        return Err("eagle: gemma4_draft_open failed".to_string());
    }

    // prefill all-but-last so the first decode_logits processes the last prompt token cleanly.
    let np = prompt.len();
    if unsafe { gemma4_kv_prefill(s, prompt.as_ptr(), (np - 1) as c_int) } != 0 {
        return Err("eagle: prefill failed".to_string());
    }
    let mut last = prompt[np - 1];

    // K-step EAGLE DRIVE: draft K tokens via the recurrence (h_{k+1} = draft post_proj
    // out_hnext; token feedback), verify the prefix against the target's greedy, accept the
    // matching prefix. Measures mean ACCEPT-LENGTH (the real spec-decode quality metric) +
    // single-token accept (K=1 case) + a sequential-verify tok/s baseline. The realized
    // throughput win = (mean_accept+1) tokens per target forward, UNLOCKED by batched verify.
    let k = std::env::var("SP_EAGLE_K").ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(4);
    let dump = std::env::var("SP_EAGLE_DUMP").ok(); // optional flywheel-data dir (feature,target-token)
    let mut feat = vec![0.0f32; hidden];
    let mut hnext = vec![0.0f32; hidden];
    let mut logits = vec![0.0f32; vocab];
    let mut accept_lens: Vec<usize> = Vec::new();
    let (mut single_hit, mut single_n) = (0usize, 0usize);
    let mut tgt_seq: Vec<i32> = Vec::new();
    let mut samples: Vec<(i32, i32)> = Vec::new();
    let mut dump_rows: Vec<(Vec<f32>, i32)> = Vec::new();

    let suppress_logits = |lg: &mut [f32]| {
        for &id in &suppress { if (id as usize) < vocab { lg[id as usize] = f32::NEG_INFINITY; } }
    };

    // prime: feature + target greedy for the position right after `last`.
    unsafe { gemma4_kv_capture_feat(s, feat.as_mut_ptr()) };
    if unsafe { gemma4_kv_decode_logits(s, last, logits.as_mut_ptr()) } != 0 {
        return Err("eagle: prime decode failed".to_string());
    }
    suppress_logits(&mut logits);
    let mut g = argmax(&logits);

    let t0 = std::time::Instant::now();
    let mut target_tokens = 0usize;
    while target_tokens < n_decode && !tok.eos_ids.contains(&g) {
        // draft K tokens from (feat, last) via the EAGLE recurrence.
        let mut drafts: Vec<i32> = Vec::with_capacity(k);
        let mut dh = feat.clone();
        let mut din = last;
        for _ in 0..k {
            let mut dk: c_int = -1;
            let rc = unsafe {
                gemma4_draft_step(s, dh.as_ptr(), din, suppress.as_ptr(),
                                  suppress.len() as c_int, &mut dk, hnext.as_mut_ptr())
            };
            if rc != 0 { return Err("eagle: draft_step failed".to_string()); }
            drafts.push(dk);
            din = dk;
            std::mem::swap(&mut dh, &mut hnext); // dh := post_proj h_next (recurrence)
        }
        single_n += 1;
        if drafts[0] == g { single_hit += 1; }
        if samples.len() < 12 { samples.push((drafts[0], g)); }
        if dump.is_some() { dump_rows.push((feat.clone(), g)); }

        // verify: accept the leading drafts matching the target greedy continuation.
        let mut m = 0usize;
        while m < k && drafts[m] == g && !tok.eos_ids.contains(&g) {
            m += 1;
            tgt_seq.push(g);
            last = g;
            target_tokens += 1;
            unsafe { gemma4_kv_capture_feat(s, feat.as_mut_ptr()) };
            if unsafe { gemma4_kv_decode_logits(s, g, logits.as_mut_ptr()) } != 0 { g = -1; break; }
            suppress_logits(&mut logits);
            g = argmax(&logits);
        }
        if g == -1 { break; }
        if m == 0 {
            // draft[0] missed -> commit the target's corrected token (1 step) and move on.
            tgt_seq.push(g);
            last = g;
            target_tokens += 1;
            unsafe { gemma4_kv_capture_feat(s, feat.as_mut_ptr()) };
            if unsafe { gemma4_kv_decode_logits(s, g, logits.as_mut_ptr()) } != 0 { break; }
            suppress_logits(&mut logits);
            g = argmax(&logits);
        }
        accept_lens.push(m);
    }
    let elapsed = t0.elapsed().as_secs_f64();
    unsafe { gemma4_draft_close(); gemma4_kv_close(s); }

    if let Some(dir) = dump {
        let _ = std::fs::create_dir_all(&dir);
        let mut feats: Vec<u8> = Vec::with_capacity(dump_rows.len() * hidden * 4);
        let mut toks: Vec<u8> = Vec::with_capacity(dump_rows.len() * 4);
        for (h, t) in &dump_rows {
            for &v in h { feats.extend_from_slice(&v.to_le_bytes()); }
            toks.extend_from_slice(&t.to_le_bytes());
        }
        let _ = std::fs::write(format!("{dir}/feat.f32"), &feats);
        let _ = std::fs::write(format!("{dir}/tgt.i32"), &toks);
        eprintln!("[eagle] flywheel dump: {} (feature,target) rows -> {dir}", dump_rows.len());
    }

    let cont: String = tgt_seq.iter().flat_map(|&t| tok.decode_token(t).to_vec())
        .collect::<Vec<u8>>().into_iter().map(|b| b as char).collect();
    let mean_acc = if accept_lens.is_empty() { 0.0 } else { accept_lens.iter().sum::<usize>() as f64 / accept_lens.len() as f64 };
    let single = if single_n > 0 { 100.0 * single_hit as f64 / single_n as f64 } else { 0.0 };
    let toks_s = if elapsed > 0.0 { target_tokens as f64 / elapsed } else { 0.0 };
    eprintln!("[eagle] target continuation: {:?}", cont);
    eprintln!("[eagle] samples (draft0 -> target): {:?}", samples);
    eprintln!("[eagle] K={k} rounds={} mean_accept_len={mean_acc:.3} (potential {:.2}x w/ batched verify) | single-token={single:.1}% | {target_tokens} tok in {elapsed:.1}s = {toks_s:.1} tok/s (seq verify)",
              accept_lens.len(), mean_acc + 1.0);
    Ok((single_hit, single_n))
}
