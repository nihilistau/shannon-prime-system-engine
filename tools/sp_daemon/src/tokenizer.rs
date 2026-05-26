//! SPTB tokenizer adapter — parses the .sp-tokenizer blob and wraps it in the
//! `tokenizers` crate for prompt encoding / token decoding.

use std::sync::Arc;

use ahash::AHashMap;
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::{AddedToken, Tokenizer};

use crate::session::SpModel;

const TYPE_BPE_LLAMA3: u32 = 1;
const TYPE_BPE_GPT2: u32 = 2;

const ARCH_QWEN3: u32 = 2;
const ARCH_GEMMA3: u32 = 3;
const ARCH_QWEN25: u32 = 6;

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
        ARCH_GEMMA3 => &["<eos>", "<end_of_turn>"],
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
    id_to_bytes: Vec<Vec<u8>>,
    pub eos_ids: Vec<i32>,
    pub arch_id: u32,
    type_id: u32,
}

impl SptbTokenizer {
    pub fn build(model: &SpModel, arch_id: u32) -> Result<Arc<Self>, String> {
        let blob = model.tokenizer_blob()?;
        let data = parse_sptb(blob)?;

        let is_bpe = data.type_id == TYPE_BPE_GPT2 || data.type_id == TYPE_BPE_LLAMA3;

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

        Ok(Arc::new(SptbTokenizer {
            inner,
            id_to_bytes,
            eos_ids,
            arch_id,
            type_id: data.type_id,
        }))
    }

    pub fn encode(&self, text: &str) -> Result<Vec<i32>, String> {
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

    pub fn apply_template(&self, messages: &[Message]) -> Result<String, TemplateError> {
        match self.arch_id {
            ARCH_QWEN3 | ARCH_QWEN25 => Ok(chatml_template(messages)),
            ARCH_GEMMA3 => Ok(gemma3_template(messages)),
            _ => Err(TemplateError { arch_id: self.arch_id }),
        }
    }
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
