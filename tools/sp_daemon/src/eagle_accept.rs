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
    fn gemma4_kv_reset(s: *mut sp_g4_kv) -> c_int;
    fn gemma4_kv_ctx_dump(s: *mut sp_g4_kv, kg: *mut f32, vg: *mut f32, ks: *mut f32, vs: *mut f32,
                          kvd_g: *mut c_int, kvd_s: *mut c_int, npos: *mut c_int) -> c_int;
    fn gemma4_kv_ctx_geom(s: *mut sp_g4_kv, g_nkv: *mut c_int, g_hd: *mut c_int, s_nkv: *mut c_int,
                          s_hd: *mut c_int, period: *mut c_int, kvfs: *mut c_int,
                          g_base: *mut f32, s_base: *mut f32) -> c_int;
    fn gemma4_embd_row(token: c_int, e: c_int, out: *mut f32) -> c_int;
    fn gemma4_kv_decode_batch(s: *mut sp_g4_kv, toks: *const i32, b: c_int,
                              logits_out: *mut f32, feat_out: *mut f32) -> c_int;
    fn gemma4_kv_rewind(s: *mut sp_g4_kv, delta: c_int) -> c_int;
}

/// KAIROS action space (Latent Interceptor) — keep in sync with CONTRACT-LATENT-INTERCEPTOR.md.
pub const LI_ACTIONS: [&str; 5] = ["NO_OP", "KEEP", "FORGET", "E2B_TOOL", "ACTION"];

/// Apply the tiny Latent Interceptor probe (mu,sd,W1,b1,W2,b2 from li_head.bin) to a feature
/// (host f32[H]) -> action id. Pure CPU matmul (proj~256), microseconds. Layout per sp_li_train.py.
fn li_probe(blob: &[f32], h: usize, a: usize, proj: usize, feat: &[f32]) -> i32 {
    let mu = &blob[0..h];
    let sd = &blob[h..2 * h];
    let w1 = &blob[2 * h..2 * h + proj * h];
    let b1 = &blob[2 * h + proj * h..2 * h + proj * h + proj];
    let w2o = 2 * h + proj * h + proj;
    let w2 = &blob[w2o..w2o + a * proj];
    let b2 = &blob[w2o + a * proj..w2o + a * proj + a];
    let mut hid = vec![0.0f32; proj];
    for p in 0..proj {
        let mut s = b1[p];
        let row = &w1[p * h..p * h + h];
        for i in 0..h { s += row[i] * ((feat[i] - mu[i]) / sd[i]); }
        hid[p] = if s > 0.0 { s } else { 0.0 }; // ReLU
    }
    let mut best = 0i32; let mut bestv = f32::NEG_INFINITY;
    for c in 0..a {
        let mut s = b2[c];
        let row = &w2[c * proj..c * proj + proj];
        for p in 0..proj { s += row[p] * hid[p]; }
        if s > bestv { bestv = s; best = c as i32; }
    }
    best
}

fn parse_kairos_tape(text: &str) -> Vec<(String, i32)> {
    let mut events = Vec::new();
    for line in text.lines() {
        let l = line.trim();
        if l.is_empty() || l.starts_with('#') { continue; }
        let (mut toks, mut cur, mut q): (Vec<String>, String, bool) = (Vec::new(), String::new(), false);
        for ch in l.chars() {
            match ch {
                '"' => q = !q,
                c if c.is_whitespace() && !q => { if !cur.is_empty() { toks.push(std::mem::take(&mut cur)); } }
                c => cur.push(c),
            }
        }
        if !cur.is_empty() { toks.push(cur); }
        if toks.len() != 5 { continue; }
        let aid = li_action_id(&toks[4]);
        if aid < 0 { continue; }
        let payload = if toks[2] == "-" { String::new() } else { format!(" payload=\"{}\"", toks[2]) };
        events.push((format!("EVENT kind={} salience={}{}", toks[1], toks[3], payload), aid));
    }
    events
}

/// LIVE KAIROS ORACLE (the Latent Interceptor action gate). Per tick: 12B prefills the event frame
/// (~35ms, unavoidable), taps the frame-end latent, runs the CPU probe (~us) -> action. NO_OP ->
/// SHORT-CIRCUIT (skip the ~440ms 12B decode); any action class -> wake the 12B decode. Logs the
/// per-tick decision + the banked compute. Reuses the gemma4_kv handle + the eagle feature tap.
pub fn run_li_oracle(model_path: &str, tok_path: &str, tape_path: &str, head_path: &str) -> Result<(), String> {
    std::env::set_var("SP_CUDA_DECODE_INT8", "1");
    let model = SpModel::load(model_path, tok_path)?;
    let arch = model.arch_info()?;
    let vocab = arch.vocab_size as usize;
    let hidden = arch.hidden_dim as usize;
    let tok = SptbTokenizer::build(&model, arch.arch_id, tok_path)?;
    let cancel = Arc::new(AtomicI32::new(0));
    let mut session = SpSession::create(&model, cancel)?;
    let sraw = session.raw_ptr() as *mut sp_daemon::ffi_l1::sp_session;
    let qm = (unsafe { sp_daemon::ffi_l1::sp_session_qwen3_model(sraw) }) as *const c_void;
    if qm.is_null() { return Err("li_oracle: qm NULL".to_string()); }

    let raw = std::fs::read(head_path).map_err(|e| format!("head {head_path}: {e}"))?;
    let blob: Vec<f32> = raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    let a = LI_ACTIONS.len();
    // len = 2H + A + proj*(H+1+A)  ->  proj
    let proj = (blob.len() - 2 * hidden - a) / (hidden + 1 + a);
    eprintln!("[li-oracle] head={head_path} hidden={hidden} proj={proj} actions={a}");

    const CONTRACT: &str = "You are a background kernel daemon. Each tick you receive one environment event. \
Decide exactly one action: NO_OP (idle, do nothing), KEEP (remember this fact), FORGET (evict stale state), \
E2B_TOOL (run a tool/compute), or ACTION (intervene). Most events are NO_OP.";
    let events = parse_kairos_tape(&std::fs::read_to_string(tape_path).map_err(|e| format!("tape: {e}"))?);
    if events.is_empty() { return Err("li_oracle: no events".to_string()); }

    let s = unsafe { gemma4_kv_open(qm, 512) };
    if s.is_null() { return Err("li_oracle: kv_open NULL".to_string()); }
    let mut feat = vec![0.0f32; hidden];
    let mut logits = vec![0.0f32; vocab];
    let suppress: Vec<i32> = tok.suppress_token_ids();
    let (mut correct, mut skips, mut woke, mut false_wake, mut missed) = (0usize, 0usize, 0usize, 0usize, 0usize);
    let (mut saved_ms, mut spent_ms) = (0.0f64, 0.0f64);
    let decode_cap = 24usize;

    for (i, (body, gt)) in events.iter().enumerate() {
        let frame = format!("{CONTRACT}\n\nCURRENT EVENT: {body}\nRespond with exactly one of: NO_OP, KEEP, FORGET, E2B_TOOL, ACTION.");
        let ids = match tok.apply_template_ids(&vec![Message { role: "user".to_string(), content: frame }]) { Ok(p) => p, Err(_) => continue };
        if ids.len() < 2 || ids.len() >= 512 { continue; }
        unsafe { gemma4_kv_reset(s) };
        let np = ids.len();
        let tpf = std::time::Instant::now();
        if unsafe { gemma4_kv_prefill(s, ids.as_ptr(), (np - 1) as c_int) } != 0 { continue; }
        unsafe { gemma4_kv_capture_feat(s, feat.as_mut_ptr()) };
        if unsafe { gemma4_kv_decode_logits(s, ids[np - 1], logits.as_mut_ptr()) } != 0 { continue; }
        let prefill_ms = tpf.elapsed().as_secs_f64() * 1000.0;
        let tpr = std::time::Instant::now();
        let action = li_probe(&blob, hidden, a, proj, &feat);
        let probe_us = tpr.elapsed().as_secs_f64() * 1e6;

        if action == *gt { correct += 1; }
        if action == 0 {
            // NO_OP -> short-circuit: the 477ms 12B decode is SKIPPED.
            skips += 1;
            let est = 440.0; saved_ms += est;
            if *gt != 0 { missed += 1; } // wrongly skipped a real event
            eprintln!("[li-oracle] tick {i:>3} action=NO_OP gt={} prefill={prefill_ms:.0}ms probe={probe_us:.0}us  DECODE SKIPPED (~{est:.0}ms banked)",
                      LI_ACTIONS[*gt as usize]);
        } else {
            // action -> wake the 12B to decode the execution (bounded).
            woke += 1;
            if *gt == 0 { false_wake += 1; } // wrongly woke on an idle tick
            let td = std::time::Instant::now();
            let mut last = argmax(&logits);
            for _ in 0..decode_cap {
                if tok.eos_ids.contains(&last) { break; }
                unsafe { gemma4_kv_decode_logits(s, last, logits.as_mut_ptr()) };
                for &id in &suppress { if (id as usize) < vocab { logits[id as usize] = f32::NEG_INFINITY; } }
                last = argmax(&logits);
            }
            let dms = td.elapsed().as_secs_f64() * 1000.0; spent_ms += dms;
            eprintln!("[li-oracle] tick {i:>3} action={:<8} gt={:<8} prefill={prefill_ms:.0}ms probe={probe_us:.0}us  WOKE 12B decode={dms:.0}ms",
                      LI_ACTIONS[action as usize], LI_ACTIONS[*gt as usize]);
        }
    }
    unsafe { gemma4_kv_close(s) };
    let n = events.len();
    eprintln!("[li-oracle] DONE {n} ticks | accuracy={:.3} ({correct}/{n}) | NO_OP-skips={skips} woke={woke} | false_wake={false_wake} missed={missed}",
              correct as f64 / n as f64);
    eprintln!("[li-oracle] COMPUTE: decode spent={spent_ms:.0}ms on {woke} woke ticks; BANKED ~{saved_ms:.0}ms on {skips} skipped ticks ({:.0}% of ticks gated)",
              100.0 * skips as f64 / n as f64);
    Ok(())
}
fn li_action_id(s: &str) -> i32 {
    LI_ACTIONS.iter().position(|&a| a.eq_ignore_ascii_case(s.trim())).map(|i| i as i32).unwrap_or(-1)
}

/// LATENT INTERCEPTOR CAPTURE: run the 12B over a KAIROS event tape; per tick frame the event
/// against the kernel contract, prefill it, capture the FRAME-END feature[hidden] (the post-
/// output_norm hidden the LM head consumes) + the action label (tape `expect`). The interceptor
/// learns feature -> action WITHOUT the 262k vocab head. Each event is evaluated against a clean
/// contract context (KV reset between events = the pruned-heartbeat semantics).
pub fn run_li_capture(
    model_path: &str,
    tok_path: &str,
    tape_path: &str,
    out_dir: &str,
) -> Result<usize, String> {
    std::env::set_var("SP_CUDA_DECODE_INT8", "1");
    let model = SpModel::load(model_path, tok_path)?;
    let arch = model.arch_info()?;
    let vocab = arch.vocab_size as usize;
    let hidden = arch.hidden_dim as usize;
    let tok = SptbTokenizer::build(&model, arch.arch_id, tok_path)?;
    let cancel = Arc::new(AtomicI32::new(0));
    let mut session = SpSession::create(&model, cancel)?;
    let sraw = session.raw_ptr() as *mut sp_daemon::ffi_l1::sp_session;
    let qm = (unsafe { sp_daemon::ffi_l1::sp_session_qwen3_model(sraw) }) as *const c_void;
    if qm.is_null() { return Err("li_capture: qm NULL".to_string()); }

    const CONTRACT: &str = "You are a background kernel daemon. Each tick you receive one environment event. \
Decide exactly one action: NO_OP (idle, do nothing), KEEP (remember this fact), FORGET (evict stale state), \
E2B_TOOL (run a tool/compute), or ACTION (intervene). Most events are NO_OP.";

    let tape = std::fs::read_to_string(tape_path).map_err(|e| format!("tape {tape_path}: {e}"))?;
    let mut events: Vec<(String, i32)> = Vec::new();
    for line in tape.lines() {
        let l = line.trim();
        if l.is_empty() || l.starts_with('#') { continue; }
        // tick_idx kind payload(may be quoted) salience expect
        let mut toks: Vec<String> = Vec::new();
        let (mut cur, mut q) = (String::new(), false);
        for ch in l.chars() {
            match ch {
                '"' => q = !q,
                c if c.is_whitespace() && !q => { if !cur.is_empty() { toks.push(std::mem::take(&mut cur)); } }
                c => cur.push(c),
            }
        }
        if !cur.is_empty() { toks.push(cur); }
        if toks.len() != 5 { continue; }
        let aid = li_action_id(&toks[4]);
        if aid < 0 { continue; }
        let payload = if toks[2] == "-" { String::new() } else { format!(" payload=\"{}\"", toks[2]) };
        let body = format!("EVENT kind={} salience={}{}", toks[1], toks[3], payload);
        events.push((body, aid));
    }
    if events.is_empty() { return Err("li_capture: no events parsed".to_string()); }

    let pmax = 512 as c_int;
    let s = unsafe { gemma4_kv_open(qm, pmax) };
    if s.is_null() { return Err("li_capture: gemma4_kv_open NULL".to_string()); }
    let mut feat = vec![0.0f32; hidden];
    let mut logits = vec![0.0f32; vocab];
    let mut feats: Vec<f32> = Vec::with_capacity(events.len() * hidden);
    let mut labels: Vec<i32> = Vec::with_capacity(events.len());
    let mut dist = [0usize; 5];

    for (i, (body, aid)) in events.iter().enumerate() {
        let frame = format!("{CONTRACT}\n\nCURRENT EVENT: {body}\nRespond with exactly one of: NO_OP, KEEP, FORGET, E2B_TOOL, ACTION.");
        let msgs = vec![Message { role: "user".to_string(), content: frame }];
        let ids = match tok.apply_template_ids(&msgs) { Ok(p) => p, Err(_) => continue };
        if ids.len() < 2 || ids.len() as c_int >= pmax { continue; }
        if unsafe { gemma4_kv_reset(s) } != 0 { return Err("li_capture: reset".to_string()); }
        let np = ids.len();
        if unsafe { gemma4_kv_prefill(s, ids.as_ptr(), (np - 1) as c_int) } != 0 { continue; }
        unsafe { gemma4_kv_capture_feat(s, feat.as_mut_ptr()) };
        if unsafe { gemma4_kv_decode_logits(s, ids[np - 1], logits.as_mut_ptr()) } != 0 { continue; }
        feats.extend_from_slice(&feat);
        labels.push(*aid);
        dist[*aid as usize] += 1;
        if i % 50 == 0 { eprintln!("[li-capture] {i}/{}", events.len()); }
    }
    unsafe { gemma4_kv_close(s) };
    std::fs::create_dir_all(out_dir).map_err(|e| format!("mkdir {out_dir}: {e}"))?;
    let f32b: Vec<u8> = feats.iter().flat_map(|x| x.to_le_bytes()).collect();
    let i32b: Vec<u8> = labels.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(format!("{out_dir}/feat.f32"), &f32b).map_err(|e| format!("feat: {e}"))?;
    std::fs::write(format!("{out_dir}/label.i32"), &i32b).map_err(|e| format!("label: {e}"))?;
    std::fs::write(format!("{out_dir}/manifest.jsonl"),
        format!("{{\"n\":{},\"hidden\":{hidden},\"n_actions\":5,\"actions\":[\"NO_OP\",\"KEEP\",\"FORGET\",\"E2B_TOOL\",\"ACTION\"]}}\n", labels.len()))
        .map_err(|e| format!("manifest: {e}"))?;
    eprintln!("[li-capture] DONE {} samples -> {out_dir} (hidden={hidden}) dist NO_OP={} KEEP={} FORGET={} E2B={} ACTION={}",
              labels.len(), dist[0], dist[1], dist[2], dist[3], dist[4]);
    Ok(labels.len())
}

/// EAGLE flywheel CAPTURE: greedy-roll the target over a corpus, dumping per generated
/// position (feature h, input token, target label) + the full-sequence KV the draft attends.
/// This is the engine-matched training data for #2 (the draft learns OUR engine's distribution).
pub fn run_eagle_capture(
    model_path: &str,
    tok_path: &str,
    corpus_path: &str,
    out_dir: &str,
    gen_len: usize,
) -> Result<usize, String> {
    std::env::set_var("SP_CUDA_DECODE_INT8", "1");
    let model = SpModel::load(model_path, tok_path)?;
    let arch = model.arch_info()?;
    let vocab = arch.vocab_size as usize;
    let hidden = arch.hidden_dim as usize;
    let tok = SptbTokenizer::build(&model, arch.arch_id, tok_path)?;
    let cancel = Arc::new(AtomicI32::new(0));
    let mut session = SpSession::create(&model, cancel)?;
    let sraw = session.raw_ptr() as *mut sp_daemon::ffi_l1::sp_session;
    let qm = (unsafe { sp_daemon::ffi_l1::sp_session_qwen3_model(sraw) }) as *const c_void;
    if qm.is_null() { return Err("capture: qm NULL".to_string()); }
    let suppress: Vec<i32> = tok.suppress_token_ids();

    let corpus = std::fs::read_to_string(corpus_path).map_err(|e| format!("corpus {corpus_path}: {e}"))?;
    let lines: Vec<&str> = corpus.lines().map(|l| l.trim()).filter(|l| !l.is_empty()).collect();
    let max_prompt = 64usize;
    let pmax = (max_prompt + gen_len + 8) as c_int;
    let s = unsafe { gemma4_kv_open(qm, pmax) };
    if s.is_null() { return Err("capture: gemma4_kv_open NULL".to_string()); }

    // geometry for KV-buffer sizing.
    let (mut gnkv, mut ghd, mut snkv, mut shd, mut period, mut kvfs) = (0i32, 0i32, 0i32, 0i32, 0i32, 0i32);
    let (mut gb, mut sb) = (0f32, 0f32);
    unsafe { gemma4_kv_ctx_geom(s, &mut gnkv, &mut ghd, &mut snkv, &mut shd, &mut period, &mut kvfs, &mut gb, &mut sb); }
    let kvd_g = (gnkv * ghd) as usize;
    let kvd_s = (snkv * shd) as usize;
    std::fs::create_dir_all(out_dir).map_err(|e| format!("mkdir {out_dir}: {e}"))?;
    let mut manifest = String::new();
    manifest.push_str(&format!(
        "{{\"hidden\":{hidden},\"vocab\":{vocab},\"g_nkv\":{gnkv},\"g_hd\":{ghd},\"s_nkv\":{snkv},\"s_hd\":{shd},\"period\":{period},\"kvfs\":{kvfs},\"g_base\":{gb},\"s_base\":{sb},\"gen_len\":{gen_len}}}\n"
    ));

    let mut kg = vec![0.0f32; (pmax as usize) * kvd_g.max(1)];
    let mut vg = vec![0.0f32; (pmax as usize) * kvd_g.max(1)];
    let mut ks = vec![0.0f32; (pmax as usize) * kvd_s.max(1)];
    let mut vs = vec![0.0f32; (pmax as usize) * kvd_s.max(1)];
    let mut feat = vec![0.0f32; hidden];
    let mut xbuf = vec![0.0f32; hidden];
    let mut logits = vec![0.0f32; vocab];

    let mut seqs = 0usize;
    for (si, line) in lines.iter().enumerate() {
        let msgs = vec![Message { role: "user".to_string(), content: line.to_string() }];
        let mut prompt = match tok.apply_template_ids(&msgs) { Ok(p) => p, Err(_) => continue };
        if prompt.len() < 2 || prompt.len() > max_prompt { prompt.truncate(max_prompt); }
        if unsafe { gemma4_kv_reset(s) } != 0 { return Err("capture: reset".to_string()); }
        let np = prompt.len();
        if unsafe { gemma4_kv_prefill(s, prompt.as_ptr(), (np - 1) as c_int) } != 0 { continue; }
        let mut last = prompt[np - 1];
        let (mut feats, mut xs, mut inps, mut lbls, mut atts): (Vec<f32>, Vec<f32>, Vec<i32>, Vec<i32>, Vec<i32>) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for _ in 0..gen_len {
            unsafe { gemma4_kv_capture_feat(s, feat.as_mut_ptr()) };
            if unsafe { gemma4_kv_decode_logits(s, last, logits.as_mut_ptr()) } != 0 { break; }
            for &id in &suppress { if (id as usize) < vocab { logits[id as usize] = f32::NEG_INFINITY; } }
            let g = argmax(&logits);
            let attend = np as i32 + inps.len() as i32; // KV positions the draft attends for this step
            unsafe { gemma4_embd_row(last, hidden as c_int, xbuf.as_mut_ptr()); } // draft pre_proj input x
            xs.extend_from_slice(&xbuf);
            feats.extend_from_slice(&feat);
            inps.push(last);
            lbls.push(g);
            atts.push(attend);
            last = g;
            if tok.eos_ids.contains(&g) { break; }
        }
        if inps.is_empty() { continue; }
        let mut kvd_g_o = 0i32; let mut kvd_s_o = 0i32; let mut npos = 0i32;
        unsafe { gemma4_kv_ctx_dump(s, kg.as_mut_ptr(), vg.as_mut_ptr(), ks.as_mut_ptr(), vs.as_mut_ptr(),
                                    &mut kvd_g_o, &mut kvd_s_o, &mut npos); }
        let dir = format!("{out_dir}/seq_{si:04}");
        std::fs::create_dir_all(&dir).ok();
        let w = |name: &str, bytes: &[u8]| { let _ = std::fs::write(format!("{dir}/{name}"), bytes); };
        let f32b = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let i32b = |v: &[i32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        w("feat.f32", &f32b(&feats));
        w("x.f32", &f32b(&xs));
        w("inp.i32", &i32b(&inps));
        w("lbl.i32", &i32b(&lbls));
        w("att.i32", &i32b(&atts));
        w("kg.f32", &f32b(&kg[..(npos as usize) * kvd_g])); w("vg.f32", &f32b(&vg[..(npos as usize) * kvd_g]));
        w("ks.f32", &f32b(&ks[..(npos as usize) * kvd_s])); w("vs.f32", &f32b(&vs[..(npos as usize) * kvd_s]));
        manifest.push_str(&format!("{{\"seq\":{si},\"gen\":{},\"npos\":{npos},\"kvd_g\":{kvd_g},\"kvd_s\":{kvd_s}}}\n", inps.len()));
        seqs += 1;
        if si % 25 == 0 { eprintln!("[capture] seq {si}/{} gen={} npos={npos}", lines.len(), inps.len()); }
    }
    unsafe { gemma4_kv_close(s) };
    std::fs::write(format!("{out_dir}/manifest.jsonl"), manifest).map_err(|e| format!("manifest: {e}"))?;
    eprintln!("[capture] DONE {seqs} sequences -> {out_dir} (kvd_g={kvd_g} kvd_s={kvd_s} hidden={hidden})");
    Ok(seqs)
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
    // SP_EAGLE_PROMPT overrides the probe prompt (held-out / OOD A/B).
    let prompt_text = std::env::var("SP_EAGLE_PROMPT")
        .unwrap_or_else(|_| "What is the capital of France?".to_string());
    let msgs = vec![Message {
        role: "user".to_string(),
        content: prompt_text,
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
    // BOTTLENECK PROFILE: prefill = (np-1) forwards with ONE host sync; decode below = one sync +
    // a full-vocab D2H per token. prefill_ms/token << decode_ms/token => sync/D2H-bound (cheap
    // batched verify wins); ~equal => per-forward compute-bound (needs batched-GEMM).
    let np = prompt.len();
    let t_pf = std::time::Instant::now();
    if unsafe { gemma4_kv_prefill(s, prompt.as_ptr(), (np - 1) as c_int) } != 0 {
        return Err("eagle: prefill failed".to_string());
    }
    let pf_ms = t_pf.elapsed().as_secs_f64() * 1000.0;
    eprintln!("[eagle] PROFILE prefill {} tok in {:.1}ms = {:.1} ms/tok (one host sync, no per-tok logits D2H)",
              np - 1, pf_ms, pf_ms / (np - 1).max(1) as f64);
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

    // BOTTLENECK ISOLATION (SP_EAGLE_PROFILE=1): B sequential decode_logits (B syncs + B full-vocab
    // D2H) vs ONE gemma4_kv_decode_batch(B) (one sync). Same B tokens, same start (rewind between).
    // ratio >> 1 => per-token overhead is collapsible (batched verify wins without batched-GEMM).
    if std::env::var("SP_EAGLE_PROFILE").as_deref() == Ok("1") {
        let bb = 8usize;
        let probe_toks: Vec<i32> = (0..bb).map(|_| last).collect();
        let mut blog = vec![0.0f32; bb * vocab];
        // sequential
        let ts = std::time::Instant::now();
        for &t in &probe_toks { let _ = unsafe { gemma4_kv_decode_logits(s, t, logits.as_mut_ptr()) }; }
        let seq_ms = ts.elapsed().as_secs_f64() * 1000.0;
        unsafe { gemma4_kv_rewind(s, bb as c_int) };
        // batched
        let tb = std::time::Instant::now();
        let rc = unsafe { gemma4_kv_decode_batch(s, probe_toks.as_ptr(), bb as c_int, blog.as_mut_ptr(), std::ptr::null_mut()) };
        let bat_ms = tb.elapsed().as_secs_f64() * 1000.0;
        unsafe { gemma4_kv_rewind(s, bb as c_int) };
        eprintln!("[eagle] PROFILE B={bb}: {bb}x seq decode_logits = {seq_ms:.1}ms ({:.1} ms/tok) | 1x decode_batch = {bat_ms:.1}ms ({:.1} ms/tok) | speedup {:.2}x  rc={rc}",
                  seq_ms / bb as f64, bat_ms / bb as f64, seq_ms / bat_ms.max(0.001));
        // draft-step cost (the suspected real bottleneck): time bb draft_steps (no KV advance).
        let mut hn = vec![0.0f32; hidden];
        let mut dh = feat.clone();
        let td = std::time::Instant::now();
        for _ in 0..bb {
            let mut dk: c_int = -1;
            unsafe { gemma4_draft_step(s, dh.as_ptr(), last, suppress.as_ptr(), suppress.len() as c_int, &mut dk, hn.as_mut_ptr()); }
            std::mem::swap(&mut dh, &mut hn);
        }
        let draft_ms = td.elapsed().as_secs_f64() * 1000.0;
        eprintln!("[eagle] PROFILE B={bb}: draft_step = {draft_ms:.1}ms ({:.1} ms/step)  vs decode {:.1} ms/tok  -> draft/decode = {:.2}x",
                  draft_ms / bb as f64, seq_ms / bb as f64, (draft_ms / bb as f64) / (seq_ms / bb as f64).max(0.001));
    }

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
        if !drafts.is_empty() && drafts[0] == g { single_hit += 1; } // K=0 => plain-decode baseline
        if samples.len() < 12 && !drafts.is_empty() { samples.push((drafts[0], g)); }
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
