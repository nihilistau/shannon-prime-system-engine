---
type: design
title: Gemma3 SentencePiece tokenizer (SP2) + PPL loop (SP3) → close T_FRO_4
description: "Goal. SP2: bring up the Gemma3 SPM (SentencePiece) \"llama\" tokenizer — encode only — matching the stock llama.cpp token IDs byte-for-byte."
tags: [design]
timestamp: 2026-05-22T07:26:39Z
resource: ./docs/superpowers/plans/2026-05-22-gemma3-spm-ppl.md
sp_status: ACTIVE
sp_gate: none
sp_commit: TBD
sp_repro: none
---

# Gemma3 SentencePiece tokenizer (SP2) + PPL loop (SP3) → close T_FRO_4

> **For agentic workers:** TDD per task; oracle parity is the gate, mirroring TOK_ENCODE / E_CPU_2. Steps use checkbox (`- [ ]`) syntax.

**Goal.** SP2: bring up the Gemma3 **SPM (SentencePiece) "llama"** tokenizer — encode only — matching the stock llama.cpp token IDs byte-for-byte. SP3: a perplexity loop on Gemma3-1B replicating `llama.cpp/examples/perplexity`, used to close **T_FRO_4** (Frobenius-Q8 PPL within 0.1% of the engine's own f32 PPL), with an ungated oracle (llama.cpp f16 PPL) cross-check so a silent forward/tokenizer regression can't pass the relative gate.

**Architecture.** The existing `sp_tokenizer` is GPT2/Qwen2 byte-level BPE. Make it **model-type aware**: dispatch on `tokenizer.ggml.model` (`"gpt2"` → existing BPE; `"llama"` → new SPM path). SPM reuses the GGUF arrays directly (`tokenizer.ggml.{tokens,scores,token_type}`) — same single-source-of-truth as TOK_ENCODE; no `tokenizer.model` protobuf parse. PPL reuses `gemma3_forward` (or `qwen3_forward` for the Qwen3 cross-check) over fixed-size context chunks.

**Pins from the oracle (`lattice-llama/src/llama-vocab.cpp` `llm_tokenizer_spm_session` + the SPM case of `impl::tokenize`), confirmed:**
- **Algorithm:** greedy **bigram-merge by unigram score**, NOT Viterbi. Split text → UTF-8-char symbols (doubly-linked). Seed a max-heap with every adjacent bigram whose merged text is a vocab token, keyed by that token's score. Pop highest; if both symbols still live and `left.n+right.n==bigram.size`, merge right into left, relink, push `(left.prev,left)` and `(left,left.next)`. Then walk the chain and `resegment` each symbol.
- **Comparator (byte-parity critical):** max on score; **tie → smaller left index wins** (`(l.score<r.score) || (l.score==r.score && l.left>r.left)`).
- **resegment:** if symbol text is a vocab token → emit it; else if in `rev_merge` → recurse into the two stored halves; else **byte fallback** → emit `byte_to_token(b)` for each byte (`<0xXX>`, type 6, present for all 256).
- **Normalization:** `llama_escape_whitespace`: replace every `" "` → `▁` (U+2581 = `\xe2\x96\x81`). Gemma `add_space_prefix=0` ⇒ **no** leading space prefix added. Vocab tokens store the literal `▁`-marker bytes, so escape-then-byte-equal-lookup is exact.
- **Specials / BOS:** `tokenizer_st_partition` splits out special-token surfaces first; with `parse_special=false` CONTROL/UNKNOWN are not matched (USER_DEFINED still is). `add_bos=1` ⇒ prepend BOS (id **2**); `add_eos=0` ⇒ no EOS. bos=2, eos=1, unk=3, pad=0.

**PPL methodology (replicate `perplexity.cpp`):** non-overlapping chunks of `n_ctx` tokens (default 512); score only positions `[n_ctx/2 .. n_ctx-1]` (each scored token has ≥ n_ctx/2 context); skip BOS in scoring; `NLL = −Σ log softmax(logits_t)[token_{t+1}] / N_scored`; `PPL = exp(NLL)`. `parse_special=false`.

**Runtime decision (Blocking #1).** A 262144-vocab LM head × 26 layers makes Gemma3-1B prefill expensive (M_GEMMA3 = 3.3 s incl. load for 6 tokens). **Measure per-token cost first** (a timed ~128-tok forward) before sizing. Use a **small deterministic WikiText-2-raw slice** sized so a single PPL run is a few minutes, and confirm `σ(NLL)/√N` gives `σ(PPL)/PPL ≪ 1e-3` (sampling noise must not swamp the 0.1% gate). **`T_FRO_4` is a phase-close / SLOW gate** — its own `ctest` invocation (label `SLOW` or `-R T_FRO_4`), NOT in the every-commit fast suite. Optional head optimization: compute the LM head only at scored positions `[n_ctx/2..]` (halves head cost).

**Oracle cross-check isolation (Blocking #2).** Split into two independent checks so an accounting/tokenization bug can't masquerade as the §8.6.1 precision floor: (a) **hard tokenizer parity on the corpus** — SPM-encode the whole slice, assert byte-equal to the oracle IDs (one SPM_ENCODE fixture IS a multi-KB corpus chunk); (b) **ungated PPL cross-check** — `sp_perplexity` f32 vs `llama-perplexity` on the same slice + n_ctx, accounting replicated exactly (scored positions, BOS skip, last-chunk handling). With (a) asserting identical IDs, (b)'s only free variable is forward + accounting.

**BOS API decision.** Read `tokenizer.ggml.add_bos_token` at load; `sp_tokenizer_encode` auto-prepends BOS when it is 1 (no-op for Qwen3 where it is 0 ⇒ TOK_ENCODE unchanged; matches the oracle's `add_special=true`). No signature change.

---

## SP2 — Task 1: model-type dispatch + SPM vocab load (Qwen3 TOK regression as guard)

**Files:** `include/sp_engine/tokenizer.h`, `src/tokenizer/tokenizer.c`.

- [ ] **Step 1.** In the tokenizer struct add: `int spm` (model is "llama"), `const float *scores` (+ owned copy when `own`), byte-token id table `int32_t byte_tok[256]` (resolved from `<0xXX>` at load), and the special ids (bos/eos/unk). Read `tokenizer.ggml.model`; set `spm` when `== "llama"`. Read `tokenizer.ggml.scores` (FLOAT32 arr) and `tokenizer.ggml.token_type` (INT32 arr) via `gguf_find_kv` + `arr_data`/`arr_type` validation (add a tiny `gguf_kv_f32_array`/`gguf_kv_i32_array` accessor if cleaner). For BPE models scores may be absent — guard.
- [ ] **Step 2.** Build the `byte_tok[256]` table: for `b` in 0..255, look up `"<0x%02X>"` in the vocab hashmap. Assert all 256 resolve for an SPM model.
- [ ] **Step 3. Build + Qwen3 TOK regression (guard).** `scripts\build\build-cpu.bat && ctest --test-dir build-cpu -R "TOK_" --output-on-failure` — TOK_DECODE + TOK_ENCODE on Qwen3 must stay green (BPE path untouched).
- [ ] **Step 4. Commit.** `[lat-2-CPU] Tokenizer: model-type dispatch + SPM vocab/scores/byte-token load`.

## SP2 — Task 2: SPM bigram-merge encode

**Files:** `src/tokenizer/tokenizer.c` (+ decl already in `tokenizer.h`).

- [ ] **Step 1.** `sp_tokenizer_encode` dispatches: `spm` → `spm_encode`, else the existing BPE path. `spm_encode`: (a) special-token partition (reuse the existing longest-match-first scan; honor `parse_special`), (b) prepend BOS if `add_bos` (the engine caller decides; expose via the existing signature — for PPL we add BOS), (c) per raw fragment: `escape_whitespace` into a scratch buffer (`" "`→3 bytes `\xe2\x96\x81`), then run the bigram-merge.
- [ ] **Step 2. The merge core.** Port `llm_tokenizer_spm_session::tokenize` + `try_add_bigram` + `resegment` to C: a `symbol{prev,next,*text,n}` array, a binary max-heap of `bigram{left,right,score,size}` with the exact comparator, a `rev_merge` open-addressing map `text→(left,right)`. UTF-8 symbol split via a leading-byte length helper. Byte fallback uses `byte_tok[]`.
- [ ] **Step 3. Oracle parity fixture.** Use `dump_logits.exe` (already emits token IDs) on 4–5 fixed prompts incl. leading/internal spaces, punctuation, CJK, digits, and a byte-fallback trigger; capture the IDs. (Or a tiny `dump_tokens` if cleaner.)
- [ ] **Step 4. Test `SPM_ENCODE`** (`tests/test_spm.c`): `encode(prompt)` == oracle IDs for each fixture (BOS-prepended), exact. Register in `tests/CMakeLists.txt`.
- [ ] **Step 5. Build + run.** `ctest -R "SPM_ENCODE|TOK_" --output-on-failure`. Commit `[lat-2-CPU] Gemma3 SPM encode (bigram-merge, byte fallback) — SPM_ENCODE parity`.

## SP3 — Task 3: PPL loop

**Files:** `src/forward/ppl.c` (+ decl in a header), `tests/test_ppl.c`, `tests/CMakeLists.txt`, `tests/fixtures/wiki.test.raw.slice`.

- [ ] **Step 1.** `sp_perplexity(model, tokenizer, text, n_ctx, *ppl)`: tokenize (BOS + SPM), chunk into non-overlapping `n_ctx` windows, run the arch forward per chunk, accumulate NLL over `[n_ctx/2..]`, return `exp(mean NLL)`. Arch-dispatch (`gemma3_forward` for Gemma3, `qwen3_forward` for Qwen3). Honors `SP_ENGINE_FROB`/`SP_ARENA`.
- [ ] **Step 2. Fixture.** Check in a deterministic WikiText-2-raw test slice (~64–128 KB).
- [ ] **Step 3. Oracle cross-check.** Build `llama-perplexity` from `lattice-llama` on the same slice + `n_ctx`; record `PPL_oracle` (f16). One-off, noted in the test comment / a fixture `.txt`.
- [ ] **Step 4. Test `T_FRO_4`** (`tests/test_ppl.c`): compute `PPL_f32` (pure-f32) and `PPL_q8` (`SP_ENGINE_FROB=1` or `SP_ARENA=q8`). **Gate:** `|PPL_q8 − PPL_f32| / PPL_f32 < 1e-3`. **Ungated cross-check (logged):** `|PPL_f32 − PPL_oracle| / PPL_oracle` within the §8.6.1 precision floor (a few ×1e-3, consistent with M_GEMMA3_CPU KL 1.65e-6). Register `T_FRO_4`.
- [ ] **Step 5.** Build + `ctest -R "T_FRO_4|SPM|TOK|M_GEMMA3|E_CPU_2" --output-on-failure`, then full regression. Commit `[lat-2-CPU] SP3 PPL loop + T_FRO_4 (Gemma3-1B Frobenius-Q8 within 0.1% of f32; oracle cross-check)`. Tag `lat-phase-2-cpu-fro4-closed`.

---

## Out of scope (explicit)
- **SPM decode** (▁ unescape, byte-token reassembly, leading-space detokenize convention) — not needed for T_FRO_4; deferred to a Gemma3-generation follow-up.
- Quantized-Gemma3 GGUFs, other backends (2-CU/VK/HX mirror later), φ-RoPE (roadmap §20 research track).

## Success criteria
1. `SPM_ENCODE` green (byte-parity vs oracle on all fixtures). 2. `T_FRO_4` green (Q8 within 0.1% of f32) + oracle cross-check within the precision floor. 3. Qwen3 TOK_/E_CPU_/GEMMA3 regression unchanged. 4. MSVC `/W4` clean.
