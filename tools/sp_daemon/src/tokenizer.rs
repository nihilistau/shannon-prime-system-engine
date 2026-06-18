//! SPTB tokenizer adapter — parses the .sp-tokenizer blob and wraps it in the
//! `tokenizers` crate for prompt encoding / token decoding.

use std::ffi::{c_char, c_int, CString};
use std::sync::Arc;

use ahash::AHashMap;
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::{AddedToken, Tokenizer};

use crate::session::SpModel;

const TYPE_BPE_LLAMA3: u32 = 1;
const TYPE_BPE_GPT2: u32 = 2;
// Issue #115: GEMMA4_BPE on-disk family tag (sp_tok_type_id::SP_TOK_GEMMA4_BPE).
// The Rust `tokenizers` crate cannot drive the 514k-merge U+2581-piece gemma4
// BPE, so this lane routes encode through the engine's PROVEN C encoder
// (T_G4_TOK_PARITY: byte-for-byte vs the llama oracle).
const TYPE_GEMMA4_BPE: u32 = 4;

const ARCH_QWEN3: u32 = 2;
const ARCH_GEMMA3: u32 = 3;
const ARCH_QWEN25: u32 = 6;
const ARCH_GEMMA4: u32 = 7;

// ── Engine C tokenizer FFI (issue #115) ──────────────────────────────────────
// The proven gemma4 BPE encoder, compiled from src/tokenizer/{tokenizer,
// gemma4_bpe}.c by build.rs (cc) and linked into the daemon. We FFI the small
// load/encode/free surface (bindgen'd from sp_engine/tokenizer.h into
// crate::ffi_l1) rather than reimplementing the encoder in Rust.
use crate::ffi::{
    sp_tokenizer, sp_tokenizer_encode, sp_tokenizer_free, sp_tokenizer_load_tokfile,
    sp_tokenizer_vocab_size,
};

/// RAII wrapper around the engine C `sp_tokenizer` (gemma4 lane).
///
/// SAFETY / Send+Sync: after `sp_tokenizer_load_tokfile` returns, the handle is
/// READ-ONLY — `sp_tokenizer_encode` only reads the loaded vocab/merge tables
/// (no internal mutation, no shared global state) and the engine builds no
/// thread-local caches. The daemon shares one handle across rayon/tokio
/// blocking threads via Arc<AppState>, so concurrent `encode` calls are sound.
/// The handle is freed exactly once on Drop.
struct CTokenizer {
    handle: *mut sp_tokenizer,
}

impl CTokenizer {
    /// Load a .sp-tokenizer through the engine's blob loader. The loader
    /// dispatches on the on-disk family tag; for gemma4 it builds the
    /// 514k-merge BPE tables.
    fn load(path: &str) -> Result<Self, String> {
        let c_path = CString::new(path)
            .map_err(|_| format!("tokenizer path has interior NUL: {path}"))?;
        // SAFETY: c_path outlives the call; the loader copies what it needs.
        let handle = unsafe { sp_tokenizer_load_tokfile(c_path.as_ptr()) };
        if handle.is_null() {
            return Err(format!(
                "sp_tokenizer_load_tokfile failed for {path} (see daemon stderr for the engine diagnostic)"
            ));
        }
        Ok(CTokenizer { handle })
    }

    fn vocab_size(&self) -> u32 {
        // SAFETY: handle is a valid, non-null sp_tokenizer for self's lifetime.
        unsafe { sp_tokenizer_vocab_size(self.handle) }
    }

    /// Encode UTF-8 text to token IDs via the proven gemma4 BPE (parse_special=1
    /// so chat-template control surfaces like <start_of_turn> emit as single
    /// IDs). BOS is auto-prepended by the engine (gemma4 forces add_bos=1).
    fn encode(&self, text: &str) -> Result<Vec<i32>, String> {
        // First call sizes the output; the engine returns the full count even
        // when it exceeds the buffer, so we grow-and-retry on truncation.
        let mut cap: usize = text.len() + 16;
        loop {
            let mut out = vec![0i32; cap];
            // SAFETY: text ptr/len describe a valid UTF-8 slice; out has `cap`
            // i32 slots; handle is a valid read-only sp_tokenizer.
            let n = unsafe {
                sp_tokenizer_encode(
                    self.handle,
                    text.as_ptr() as *const c_char,
                    text.len(),
                    /*parse_special=*/ 1,
                    out.as_mut_ptr(),
                    cap as c_int,
                )
            };
            if n < 0 {
                return Err("sp_tokenizer_encode returned -1".to_string());
            }
            let n = n as usize;
            if n <= cap {
                out.truncate(n);
                return Ok(out);
            }
            // Truncated — the return value is the true count; resize and retry.
            cap = n;
        }
    }
}

impl Drop for CTokenizer {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // SAFETY: handle was produced by sp_tokenizer_load_tokfile and is
            // freed exactly once (Drop runs once; handle nulled implicitly).
            unsafe { sp_tokenizer_free(self.handle) };
        }
    }
}

// SAFETY: see CTokenizer doc — the handle is read-only after load.
unsafe impl Send for CTokenizer {}
unsafe impl Sync for CTokenizer {}

// ── Message / TemplateError ────────────────────────────────────────────────

#[derive(serde::Deserialize, Debug, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug)]
pub struct TemplateError {
    pub arch_id: u32,
}

impl std::fmt::Display for TemplateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "no chat template for arch_id={}", self.arch_id)
    }
}

// ── SPTB parser ────────────────────────────────────────────────────────────

struct SptbData {
    type_id: u32,
    vocab: Vec<String>,
    merges: Vec<(String, String)>,
}

fn read_u32(buf: &[u8], pos: &mut usize) -> Option<u32> {
    let bytes: [u8; 4] = buf.get(*pos..*pos + 4)?.try_into().ok()?;
    *pos += 4;
    Some(u32::from_le_bytes(bytes))
}

fn read_str(buf: &[u8], pos: &mut usize) -> Option<String> {
    let len = read_u32(buf, pos)? as usize;
    let s = std::str::from_utf8(buf.get(*pos..*pos + len)?).ok()?.to_string();
    *pos += len;
    Some(s)
}

fn parse_sptb(blob: &[u8]) -> Result<SptbData, String> {
    let mut pos = 0usize;

    let magic = read_u32(blob, &mut pos).ok_or("SPTB: truncated magic")?;
    if magic != 0x42545053 {
        return Err(format!("SPTB: bad magic 0x{magic:08X}"));
    }
    let type_id    = read_u32(blob, &mut pos).ok_or("SPTB: truncated type_id")?;
    let vocab_size = read_u32(blob, &mut pos).ok_or("SPTB: truncated vocab_size")? as usize;
    let n_merges   = read_u32(blob, &mut pos).ok_or("SPTB: truncated n_merges")? as usize;

    let mut vocab = Vec::with_capacity(vocab_size);
    for i in 0..vocab_size {
        vocab.push(read_str(blob, &mut pos)
            .ok_or_else(|| format!("SPTB: truncated vocab[{i}]"))?);
    }

    // SentencePiece: skip f32 scores that follow vocab
    if type_id == 0 {
        let skip = vocab_size * 4;
        if pos + skip > blob.len() {
            return Err("SPTB: truncated scores section".to_string());
        }
        pos += skip;
    }

    let mut merges = Vec::with_capacity(n_merges);
    for i in 0..n_merges {
        let s = read_str(blob, &mut pos)
            .ok_or_else(|| format!("SPTB: truncated merge[{i}]"))?;
        let sep = s.find(' ')
            .ok_or_else(|| format!("SPTB: merge[{i}] has no space separator"))?;
        merges.push((s[..sep].to_string(), s[sep + 1..].to_string()));
    }

    Ok(SptbData { type_id, vocab, merges })
}

// ── GPT2 byte-level inverse mapping ───────────────────────────────────────

fn gpt2_char_to_byte(c: char) -> Option<u8> {
    let cp = c as u32;
    match cp {
        0x0021..=0x007E => Some(cp as u8),
        0x00A1..=0x00AC => Some(cp as u8),
        0x00AE..=0x00FF => Some(cp as u8),
        0x0100..=0x0120 => Some((cp - 0x0100) as u8),
        0x0121           => Some(0x7F),
        0x0122..=0x0142 => Some((cp - 0x0122 + 0x80) as u8),
        0x0143           => Some(0xAD),
        _ => None,
    }
}

fn gpt2_decode(token_str: &str) -> Vec<u8> {
    token_str.chars().filter_map(gpt2_char_to_byte).collect()
}

// ── EOS detection ──────────────────────────────────────────────────────────

fn find_eos_ids(arch_id: u32, vocab: &[String]) -> Vec<i32> {
    let names: &[&str] = match arch_id {
        ARCH_QWEN3 | ARCH_QWEN25 => &["<|im_end|>", "<|endoftext|>"],
        // gemma4 (arch_id=7) shares gemma3's turn markers. The live daemon
        // logged eos_ids=[] for arch_id=7, so decode never stopped on EOS —
        // issue #115 fixes that here.
        ARCH_GEMMA3 | ARCH_GEMMA4 => &["<eos>", "<end_of_turn>"],
        _ => &[],
    };
    vocab.iter().enumerate()
        .filter_map(|(id, tok)| {
            if names.contains(&tok.as_str()) { Some(id as i32) } else { None }
        })
        .collect()
}

// ── SptbTokenizer ──────────────────────────────────────────────────────────

pub struct SptbTokenizer {
    inner: Option<Tokenizer>,
    /// Issue #115: the engine C gemma4 BPE encoder, loaded from the same
    /// .sp-tokenizer path. `Some` only for the GEMMA4_BPE lane (type_id=4);
    /// `encode` prefers it when present. Decode stays on the Rust id_to_bytes
    /// path (the ▁→space mapping already decodes gemma4 correctly).
    c_tokenizer: Option<CTokenizer>,
    id_to_bytes: Vec<Vec<u8>>,
    pub eos_ids: Vec<i32>,
    pub arch_id: u32,
    type_id: u32,
}

impl SptbTokenizer {
    /// `tok_path` is the .sp-tokenizer file the daemon was started with
    /// (--tokenizer / SP_TOKENIZER_PATH). For the gemma4 lane (type_id=4) it is
    /// re-opened through the engine's proven C encoder; for the BPE/SPM lanes it
    /// is unused (the Rust `tokenizers`/id_to_bytes paths handle them).
    pub fn build(model: &SpModel, arch_id: u32, tok_path: &str) -> Result<Arc<Self>, String> {
        let blob = model.tokenizer_blob()?;
        let data = parse_sptb(blob)?;

        let is_bpe = data.type_id == TYPE_BPE_GPT2 || data.type_id == TYPE_BPE_LLAMA3;
        let is_gemma4 = data.type_id == TYPE_GEMMA4_BPE;

        let inner = if is_bpe {
            let vocab_map: AHashMap<String, u32> = data.vocab.iter().enumerate()
                .map(|(i, s)| (s.clone(), i as u32))
                .collect();

            let bpe = BPE::builder()
                .vocab_and_merges(vocab_map, data.merges.clone())
                .build()
                .map_err(|e| format!("BPE::build: {e}"))?;

            let mut tokenizer = Tokenizer::new(bpe);
            tokenizer.with_pre_tokenizer(Some(ByteLevel::new(false, true, true)));

            // Add tokens that look like special markers so they aren't split
            let specials: Vec<AddedToken> = data.vocab.iter()
                .filter(|tok| tok.starts_with('<') && tok.ends_with('>'))
                .map(|tok| AddedToken::from(tok.as_str(), true))
                .collect();
            if !specials.is_empty() {
                tokenizer.add_special_tokens(&specials);
            }

            Some(tokenizer)
        } else {
            None
        };

        let id_to_bytes: Vec<Vec<u8>> = if is_bpe {
            data.vocab.iter().map(|s| gpt2_decode(s)).collect()
        } else {
            // SentencePiece: ▁ (U+2581) represents a word-boundary space
            data.vocab.iter().map(|s| s.replace('\u{2581}', " ").into_bytes()).collect()
        };

        let eos_ids = find_eos_ids(arch_id, &data.vocab);

        // Issue #115: the gemma4 lane routes encode through the engine's proven
        // C BPE encoder. Load it from the same .sp-tokenizer path and sanity-
        // check the vocab size against the parsed blob (catches a path mismatch).
        let c_tokenizer = if is_gemma4 {
            let c = CTokenizer::load(tok_path)?;
            let cvs = c.vocab_size() as usize;
            if cvs != data.vocab.len() {
                return Err(format!(
                    "gemma4 C tokenizer vocab {cvs} != blob vocab {} (path mismatch?: {tok_path})",
                    data.vocab.len()
                ));
            }
            Some(c)
        } else {
            None
        };

        Ok(Arc::new(SptbTokenizer {
            inner,
            c_tokenizer,
            id_to_bytes,
            eos_ids,
            arch_id,
            type_id: data.type_id,
        }))
    }

    pub fn encode(&self, text: &str) -> Result<Vec<i32>, String> {
        // Issue #115: gemma4 uses the engine C encoder (the Rust `tokenizers`
        // crate cannot drive the 514k-merge U+2581-piece BPE).
        if let Some(c) = self.c_tokenizer.as_ref() {
            return c.encode(text);
        }
        let tok = self.inner.as_ref()
            .ok_or_else(|| format!("encode not supported for tokenizer type_id={}", self.type_id))?;
        let enc = tok.encode(text, false)
            .map_err(|e| format!("tokenizer encode: {e}"))?;
        Ok(enc.get_ids().iter().map(|&id| id as i32).collect())
    }

    pub fn decode_token(&self, id: i32) -> &[u8] {
        match self.id_to_bytes.get(id as usize) {
            Some(b) => b,
            None => &[],
        }
    }

    /// Default stop strings for this arch's chat format, used when the request
    /// supplies none. Issue #115: this gemma4 artifact's vocab has no
    /// `<end_of_turn>` *token* (the turn terminator is the literal text
    /// `<end_of_turn>`, which BPE-splits into several pieces), so EOS-by-id
    /// cannot catch it — the stop-string path on `TokenDecodeBuffer` does.
    /// `<eos>` (id 1) stays covered by `eos_ids`.
    pub fn default_stops(&self) -> Vec<String> {
        match self.arch_id {
            ARCH_GEMMA3 | ARCH_GEMMA4 => vec!["<end_of_turn>".to_string()],
            _ => Vec::new(),
        }
    }

    /// CONTRACT-CHAT-FULLSTACK A2 — token ids the text-chat sampler must never
    /// emit. The gemma4-12b artifact carries an image-placeholder control token
    /// that decodes to the literal bytes `<image|>`; it has an abnormally high
    /// LM-head logit and is the `<image|>` decode-loop attractor the contract
    /// flags. It is never a valid TEXT output, so we mask its logit to -inf in
    /// the sampler (standard suppress-tokens / bad-words practice). Computed
    /// from `id_to_bytes` so it stays correct regardless of the token's id.
    /// Returns every id whose decoded bytes contain an image-placeholder marker.
    pub fn suppress_token_ids(&self) -> Vec<i32> {
        let mut out = Vec::new();
        for (id, bytes) in self.id_to_bytes.iter().enumerate() {
            // Match the placeholder marker bytes. `<image|>` (this artifact) and
            // the canonical gemma `<image_soft_token>` are both caught.
            if contains_subslice(bytes, b"<image") {
                out.push(id as i32);
            }
        }
        out
    }

    pub fn apply_template(&self, messages: &[Message]) -> Result<String, TemplateError> {
        match self.arch_id {
            ARCH_QWEN3 | ARCH_QWEN25 => Ok(chatml_template(messages)),
            // gemma4 (arch_id=7) uses the identical <start_of_turn>…<end_of_turn>
            // format as gemma3 (issue #115 — the `messages` path now works).
            ARCH_GEMMA3 | ARCH_GEMMA4 => Ok(gemma3_template(messages)),
            _ => Err(TemplateError { arch_id: self.arch_id }),
        }
    }
}

/// A2 helper: true if `hay` contains `needle` as a contiguous subslice.
fn contains_subslice(hay: &[u8], needle: &[u8]) -> bool {
    if needle.is_empty() || needle.len() > hay.len() {
        return needle.is_empty();
    }
    hay.windows(needle.len()).any(|w| w == needle)
}

// ── Chat templates ─────────────────────────────────────────────────────────

fn chatml_template(messages: &[Message]) -> String {
    let mut out = String::new();
    if messages.first().map(|m| m.role != "system").unwrap_or(true) {
        out.push_str("<|im_start|>system\n<|im_end|>\n");
    }
    for msg in messages {
        out.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", msg.role, msg.content));
    }
    out.push_str("<|im_start|>assistant\n");
    out
}

fn gemma3_template(messages: &[Message]) -> String {
    let mut out = String::new();
    for msg in messages {
        let role = if msg.role == "assistant" { "model" } else { msg.role.as_str() };
        out.push_str(&format!("<start_of_turn>{}\n{}<end_of_turn>\n", role, msg.content));
    }
    out.push_str("<start_of_turn>model\n");
    out
}

// ── TokenDecodeBuffer ──────────────────────────────────────────────────────

pub enum PushResult {
    Emit(Vec<u8>),
    Stopped(Vec<u8>),
}

pub struct TokenDecodeBuffer {
    buf: Vec<u8>,
    stop_bytes: Vec<Vec<u8>>,
    hold: usize,
}

impl TokenDecodeBuffer {
    pub fn new(stop_strings: Vec<String>) -> Self {
        let stop_bytes: Vec<Vec<u8>> = stop_strings.iter()
            .map(|s| s.as_bytes().to_vec())
            .collect();
        let max_len = stop_bytes.iter().map(|s| s.len()).max().unwrap_or(0);
        let hold = max_len.saturating_sub(1);
        TokenDecodeBuffer { buf: Vec::new(), stop_bytes, hold }
    }

    pub fn push(&mut self, bytes: &[u8]) -> PushResult {
        self.buf.extend_from_slice(bytes);

        for stop in &self.stop_bytes {
            if let Some(pos) = memmem(&self.buf, stop) {
                let before = self.buf[..pos].to_vec();
                self.buf.clear();
                return PushResult::Stopped(before);
            }
        }

        if self.buf.len() <= self.hold {
            return PushResult::Emit(vec![]);
        }

        let candidate_end = self.buf.len() - self.hold;
        let emit_end = valid_utf8_up_to(&self.buf[..candidate_end]);
        if emit_end == 0 {
            return PushResult::Emit(vec![]);
        }

        let emit = self.buf[..emit_end].to_vec();
        self.buf.drain(..emit_end);
        PushResult::Emit(emit)
    }

    /// Flush remaining buffer bytes (call on stream end or cancel).
    pub fn flush(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.buf)
    }
}

fn memmem(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

fn valid_utf8_up_to(buf: &[u8]) -> usize {
    match std::str::from_utf8(buf) {
        Ok(_) => buf.len(),
        Err(e) => e.valid_up_to(),
    }
}
