//! L2 sampler — CONTRACT-CHAT-FULLSTACK Stage A2.
//!
//! The kvdecode ABI (`gemma4_kv_decode_logits`, cuda_kvdecode_dispatch.rs:139)
//! deliberately returns the full-vocab logits row so that **L2 owns sampling**.
//! This module is that owner: temperature / top-k / top-p (nucleus) /
//! repetition+frequency penalty over the generated history, with a seedable,
//! deterministic RNG.
//!
//! NULL FLOOR / DETERMINISM CONTRACT (gate G-CHAT-A2):
//!   `temperature == 0.0` ⇒ pure argmax, byte-for-byte identical to the prior
//!   hardcoded `fn argmax`. No penalty, no top-k/p, no RNG draw — the exact
//!   same `max_by partial_cmp` tie-break (first index wins). This preserves the
//!   determinism the greedy path had and keeps the speedup-vs-numerics split in
//!   A1 honest (temp=0 is the fixed reference both A1 legs compare against).
//!
//! Given a seed, sampling is fully reproducible: same logits + same history +
//! same seed ⇒ same token. The RNG is a small SplitMix64 (no external dep).

use serde::Deserialize;

/// Sampling knobs, plumbed from `ChatRequest`. Defaults are the contract's
/// pre-registered values (temperature 0.7, top_p 0.95, top_k 40, rep_pen 1.1).
#[derive(Debug, Clone, Deserialize)]
pub struct SamplingParams {
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default = "default_top_k")]
    pub top_k: u32,
    #[serde(default = "default_repetition_penalty")]
    pub repetition_penalty: f32,
    /// Optional frequency penalty (subtractive, scaled by occurrence count).
    /// 0.0 = off (default). Distinct from repetition_penalty (multiplicative).
    #[serde(default)]
    pub frequency_penalty: f32,
    /// Optional seed for the RNG. `None` ⇒ derive from a process clock so
    /// successive requests differ; `Some(s)` ⇒ fully reproducible.
    #[serde(default)]
    pub seed: Option<u64>,
}

fn default_temperature() -> f32 {
    0.7
}
fn default_top_p() -> f32 {
    0.95
}
fn default_top_k() -> u32 {
    40
}
fn default_repetition_penalty() -> f32 {
    1.1
}

impl Default for SamplingParams {
    fn default() -> Self {
        SamplingParams {
            temperature: default_temperature(),
            top_p: default_top_p(),
            top_k: default_top_k(),
            repetition_penalty: default_repetition_penalty(),
            frequency_penalty: 0.0,
            seed: None,
        }
    }
}

/// SplitMix64 — tiny deterministic PRNG. Avoids an external `rand` dependency
/// for the ~10 lines of sampling RNG we need; well-distributed for f64 draws.
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        // Avoid the all-zero fixed point of SplitMix64.
        Rng {
            state: seed ^ 0x9E37_79B9_7F4A_7C15,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform f64 in [0, 1).
    fn next_f64(&mut self) -> f64 {
        // 53-bit mantissa.
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// The stateful sampler for one chat turn. Owns its RNG and tracks the
/// generated-token history (for repetition / frequency penalty).
pub struct Sampler {
    params: SamplingParams,
    rng: Rng,
    /// Generated token ids so far this turn, for the repetition penalty.
    history: Vec<i32>,
    /// Token ids whose logit is forced to -inf before sampling (e.g. the
    /// `<image|>` placeholder control token — the decode-loop attractor). Only
    /// applied on the SAMPLED path; the temp=0 greedy null floor is left exactly
    /// equal to the old argmax (the G-CHAT-A2 determinism leg).
    suppress: Vec<i32>,
}

impl Sampler {
    pub fn new(params: SamplingParams) -> Self {
        Self::with_suppress(params, Vec::new())
    }

    pub fn with_suppress(params: SamplingParams, suppress: Vec<i32>) -> Self {
        let seed = params.seed.unwrap_or_else(|| {
            // Non-deterministic default seed from a monotonic clock.
            use std::time::{SystemTime, UNIX_EPOCH};
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0x1234_5678_9ABC_DEF0)
        });
        Sampler {
            rng: Rng::new(seed),
            history: Vec::new(),
            params,
            suppress,
        }
    }

    /// True when this sampler is the strict-argmax null floor (temp == 0).
    #[inline]
    pub fn is_greedy(&self) -> bool {
        self.params.temperature <= 0.0
    }

    /// Record a chosen token into the penalty history.
    pub fn observe(&mut self, token: i32) {
        self.history.push(token);
    }

    /// Pick the next token from a full-vocab logits row.
    ///
    /// `temperature == 0` ⇒ strict argmax (byte-identical to the old `argmax`):
    /// no penalty, no truncation, no RNG. Otherwise: apply repetition/frequency
    /// penalties over `history`, scale by temperature, truncate to top-k then
    /// top-p (nucleus), softmax over the survivors, and draw.
    pub fn sample(&mut self, logits: &mut [f32]) -> i32 {
        if logits.is_empty() {
            return 0;
        }

        // ── temp == 0: strict argmax null floor (matches the old fn argmax) ──
        // NOTE: suppression is deliberately NOT applied here — the greedy path
        // stays byte-identical to the prior hardcoded argmax (determinism gate).
        if self.is_greedy() {
            return argmax(logits);
        }

        // ── suppress placeholder/control tokens (the <image|> loop attractor) ──
        for &id in &self.suppress {
            let idx = id as usize;
            if idx < logits.len() {
                logits[idx] = f32::NEG_INFINITY;
            }
        }

        // ── repetition / frequency penalty over generated history ──
        if (self.params.repetition_penalty - 1.0).abs() > f32::EPSILON
            || self.params.frequency_penalty.abs() > f32::EPSILON
        {
            // Count occurrences once (frequency penalty scales by count).
            // History is short (<= max_tokens), so a per-call scan is cheap.
            let rp = self.params.repetition_penalty;
            let fp = self.params.frequency_penalty;
            // Walk history; apply the multiplicative rep-penalty once per unique
            // token and the additive frequency penalty per occurrence.
            let mut seen_rep: std::collections::HashSet<i32> =
                std::collections::HashSet::new();
            for &t in &self.history {
                let idx = t as usize;
                if idx >= logits.len() {
                    continue;
                }
                if (rp - 1.0).abs() > f32::EPSILON && seen_rep.insert(t) {
                    // Standard HF-style rep penalty: divide if logit > 0, else
                    // multiply (push the logit further negative).
                    let l = logits[idx];
                    logits[idx] = if l > 0.0 { l / rp } else { l * rp };
                }
                if fp.abs() > f32::EPSILON {
                    logits[idx] -= fp;
                }
            }
        }

        // ── temperature scale ──
        let temp = self.params.temperature.max(1e-6);
        for l in logits.iter_mut() {
            *l /= temp;
        }

        // ── build a candidate list (index, logit), sorted desc by logit ──
        let mut cand: Vec<(usize, f32)> = logits
            .iter()
            .enumerate()
            .map(|(i, &l)| (i, l))
            .collect();
        // Partial sort by logit descending (NaN treated as smallest).
        cand.sort_unstable_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });

        // ── top-k truncation ──
        let k = if self.params.top_k == 0 {
            cand.len()
        } else {
            (self.params.top_k as usize).min(cand.len())
        };
        cand.truncate(k);

        // ── softmax over the survivors (numerically stable) ──
        let max_logit = cand.first().map(|&(_, l)| l).unwrap_or(0.0);
        let mut probs: Vec<f64> = cand
            .iter()
            .map(|&(_, l)| ((l - max_logit) as f64).exp())
            .collect();
        let sum: f64 = probs.iter().sum();
        if sum <= 0.0 || !sum.is_finite() {
            // Degenerate (all -inf / NaN) — fall back to argmax of the survivors.
            return cand.first().map(|&(i, _)| i as i32).unwrap_or(0);
        }
        for p in probs.iter_mut() {
            *p /= sum;
        }

        // ── top-p (nucleus): keep the smallest prefix whose mass >= top_p ──
        let top_p = self.params.top_p.clamp(0.0, 1.0) as f64;
        let mut nucleus = cand.len();
        if top_p < 1.0 {
            let mut acc = 0.0f64;
            for (i, &p) in probs.iter().enumerate() {
                acc += p;
                if acc >= top_p {
                    nucleus = i + 1;
                    break;
                }
            }
        }
        let nucleus = nucleus.max(1);

        // Renormalize the nucleus.
        let nucleus_sum: f64 = probs[..nucleus].iter().sum();

        // ── draw ──
        let r = self.rng.next_f64() * nucleus_sum;
        let mut acc = 0.0f64;
        for i in 0..nucleus {
            acc += probs[i];
            if r < acc {
                return cand[i].0 as i32;
            }
        }
        // Floating-point fall-through: return the last nucleus member.
        cand[nucleus - 1].0 as i32
    }
}

/// Strict argmax — first index wins ties (matches the prior `fn argmax` in
/// routes.rs exactly, the temp=0 null floor for G-CHAT-A2).
pub fn argmax(logits: &[f32]) -> i32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as i32)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temp_zero_is_argmax() {
        // The null-floor invariant (G-CHAT-A2): temp=0 ⇒ the sampler returns
        // EXACTLY what the old hardcoded `fn argmax` returned. std `max_by`
        // returns the LAST element among equal maxima, so with two 5.0 maxima
        // (indices 1 and 4) the result is index 4 — and crucially, identical
        // between the sampler's greedy path and the free `argmax` helper (which
        // is byte-for-byte the prior routes.rs argmax).
        let mut s = Sampler::new(SamplingParams {
            temperature: 0.0,
            ..Default::default()
        });
        let mut logits = vec![0.1, 5.0, -2.0, 4.9, 5.0];
        let via_sampler = s.sample(&mut logits);
        let logits2 = vec![0.1, 5.0, -2.0, 4.9, 5.0];
        let via_argmax = argmax(&logits2);
        assert_eq!(via_sampler, via_argmax, "temp=0 must equal the old argmax");
        assert_eq!(via_sampler, 4, "max_by returns the last of equal maxima");

        // Unique-max case: unambiguous.
        let mut s2 = Sampler::new(SamplingParams { temperature: 0.0, ..Default::default() });
        let mut l3 = vec![0.1, 9.0, -2.0, 4.9, 5.0];
        assert_eq!(s2.sample(&mut l3), 1);
    }

    #[test]
    fn seeded_is_deterministic() {
        let p = SamplingParams {
            temperature: 0.8,
            top_p: 0.95,
            top_k: 40,
            repetition_penalty: 1.1,
            frequency_penalty: 0.0,
            seed: Some(42),
        };
        let logits_master = vec![1.0f32, 2.0, 3.0, 0.5, -1.0, 2.5, 1.5, 0.0];
        let mut a = Sampler::new(p.clone());
        let mut b = Sampler::new(p);
        for _ in 0..32 {
            let mut la = logits_master.clone();
            let mut lb = logits_master.clone();
            let ta = a.sample(&mut la);
            let tb = b.sample(&mut lb);
            assert_eq!(ta, tb, "seeded sampler must be reproducible");
            a.observe(ta);
            b.observe(tb);
        }
    }

    #[test]
    fn top_k_one_is_argmax_even_with_temp() {
        let mut s = Sampler::new(SamplingParams {
            temperature: 1.0,
            top_k: 1,
            top_p: 1.0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            seed: Some(7),
        });
        let mut logits = vec![0.1, 9.0, 3.0, 2.0];
        // top_k=1 collapses the candidate set to the single max ⇒ deterministic.
        assert_eq!(s.sample(&mut logits), 1);
    }

    #[test]
    fn repetition_penalty_suppresses_repeat() {
        // With a strong rep penalty + top_k=1, a previously-seen high-logit
        // token should be demoted below an unseen competitor.
        let mut s = Sampler::new(SamplingParams {
            temperature: 1.0,
            top_k: 1,
            top_p: 1.0,
            repetition_penalty: 4.0,
            frequency_penalty: 0.0,
            seed: Some(1),
        });
        s.observe(1); // token 1 was generated before
        // Logit 1 (10.0) is highest but penalized /4 = 2.5; token 2 (3.0) wins.
        let mut logits = vec![0.0, 10.0, 3.0, 1.0];
        assert_eq!(s.sample(&mut logits), 2);
    }
}
