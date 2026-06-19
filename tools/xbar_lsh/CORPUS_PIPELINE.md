---
type: runbook
title: XBAR novel-needle corpus pipeline (labeler→head loop, scale half)
description: Mint a split-entropy non-parametric needle corpus, capture token-aligned episodes, and admission-gate them with the teacher-forced ablation oracle before training the learned recall head.
tags: [xbar, b3, autonomous-recall, labeler, learned-head, corpus]
timestamp: 2026-06-19
resource: tools/xbar_lsh/mint_corpus.py
sp_status: WIP
sp_gate: G-CHAT-B3-ADMISSION (pending metal)
sp_commit: pending
sp_repro: see Steps below
---

# XBAR novel-needle corpus pipeline

The labeler half of the loop is proven (G-CHAT-B3-LABELER: perfect-diagonal,
~15-nat margin). This pipeline produces the **scaled, admission-gated training
corpus** the learned head consumes. The teacher-forced ablation gate is reused
as the **admission oracle**: a needle enters the training set ONLY if its own
secret is structurally load-bearing on its own episode.

## Why split-entropy (the OOD-grammatical-shock guard)

A CSPRNG secret in a *semantic* slot ("the capital of France is 7xQ-99pZ") would
crater ΔLL through **token-refusal** — the unconditioned model refuses to assign
mass to base58 garbage in a noun slot — NOT through episodic dependency. That
contaminates the label. So entropy source is matched to the archetype's
structural expectation:

- **code** → CSPRNG. The slot expects random alphanumerics (vault codes / keys),
  so pure entropy isolates raw episodic retention with no OOD anomaly. Shape stays
  in-distribution (digit-WORD-4digit); the selection + digits carry the entropy.
- **contradiction** → structured-fictional. Phonetically valid, grammatically
  coherent, demonstrably non-existent entity (Oricon-Prime style) overriding a
  parametric fact. Tests the model following the injected memory's semantic graph.
- **relational** → structured-fictional. Invented protocol + multi-hop threshold.

Every needle states its secret **exactly once** (the v10/needle1 self-redundancy
bug: a duplicated secret lets the model recover from a single-copy ablation →
weak collapse → contaminated label). `mint_corpus.py` asserts secret-once,
query-prefix, and secret-is-tail before emitting.

## Steps

1. **Mint** (sandbox/host, no GPU):
   ```
   python tools/xbar_lsh/mint_corpus.py --n 200 --out _needle_corpus \
     --epdir-root "D:\F\shannon-prime-repos\shannon-prime-system-engine\_needle_corpus\eps" \
     --seed-tag scale
   ```
   → `_needle_corpus/<id>.txt`, `corpus_manifest.jsonl` (id, archetype, text,
   query, secret, topic, sig_bits), `registry.jsonl` (npos=-1 placeholder).
   Add a deliberate PARAMETRIC CONTROL (e.g. `ctrl_paris.txt` = "The capital of
   France is Paris.", secret " Paris") that MUST be rejected.

2. **Capture** (metal, 12B per needle):
   ```
   _b3_corpus_capture.bat _needle_corpus
   ```
   Per needle: `sp_tok_enc` → `.tok` → `_b3_capture_ep` → `eps/ep_<name>/` (ep.k/v/mf/tok).
   `patch_npos.py` then fills real npos into registry.jsonl. ep.tok = the exact
   input token stream ⇒ ablation token-alignment guaranteed by construction.

3. **Admission gate** (metal; the auto-reject):
   For each needle N, launch the ablation labeler with N's secret + a registry
   containing N (+ the parametric control as a cross-check):
   ```
   _b3_admission.bat " <secret>" <registry.jsonl>
   ```
   POST N's query to `/v1/chat`; read the daemon log:
   `B3-DISPOSER ABLATION collapse=... [ep_N(collapse=X,ntgt=..) ep_ctrl(collapse=Y,..)]`
   **Admission rule:** ACCEPT N iff `collapse(N's secret, ep_N) < TAU` (TAU=-8.0,
   the v13 pin). The parametric control must land ≈0 (REJECT) — that is the proof
   the auto-reject catches leaks.

4. **Build labels**: accepted (query, episode)→relevant; off-diagonal +
   rejected → not-relevant. This is the contrastive training set.

5. **Train head**: reuse `tools/xbar_lsh/b3_train_wc` on the ablation-validated
   labels. Features: query global-Q (SP_B3_QDUMP) + episode global-K (ep.gk),
   both query-time. Export int16 W_c.

6. **Deploy**: recall.rs W_c hook + M=42 safety budget. Converts the OFFLINE
   validator into a LIVE autonomous selector.

## Scaling note (50→200 needles)

Step 3 currently launches one daemon per secret (SP_B3_SECRET is read once per
process). For 200 needles that is 200 model loads. The additive enhancement that
collapses it to a single pass: read the matched episode's secret from an
`ep.secret` sidecar at request-time when SP_B3_SECRET is empty (falls back to the
env var = null floor preserved). That is a small, testable routes.rs change —
left as a separate unit, not built blind.
