---
type: contract
title: "CONTRACT-EAGLE-FLYWHEEL — engine-matched MTP draft finetune (#2)"
description: "The EAGLE finetune flywheel that lifts SP-MTP draft acceptance by closing the train/serve gap."
tags: [contract, eagle, mtp, spec-decode, flywheel]
timestamp: 2026-06-29T23:30:00Z
resource: ./tools/eagle/CONTRACT-EAGLE-FLYWHEEL.md
sp_status: GREEN
sp_gate: G-EAGLE-FLYWHEEL-AB
sp_commit: 66d84c4
sp_repro: "tools/sp_daemon/_eagle_mode_run.bat (baseline) ; tools/sp_daemon/_eagle_remeasure.bat (finetuned)"
---

# CONTRACT-EAGLE-FLYWHEEL — engine-matched MTP draft finetune (#2)

## Result (G-EAGLE-FLYWHEEL-AB, 2026-06-29)

A/B on the served gemma4-12b-b1 (SP_BYTEEXACT exact-integer target), same prompt
("capital of France" + capitals continuation), SP_EAGLE_K=4, N=128, ascale=one,
SP_CUDA_DECODE_INT8=1. Draft loaded from GGUF via `gemma4_draft_open` (no transcode needed
to measure). Output on stderr.

| draft | mean_accept_len | single-token | accept | tok/s (seq verify) | potential w/ batched verify |
|---|---|---|---|---|---|
| baseline (gemma-4-12b-it-F16-MTP.gguf, off-the-shelf) | 0.258 | 25.8% | 33/128 | 2.0 | 1.26x |
| **finetuned (gemma4-mtp-draft-ft.gguf)** | **1.042** | **96.6%** | **115/119** | 2.2 | **2.04x** |

**4x accept-length lift; single-token 25.8% -> 96.6%.** Continuation byte-coherent (target
verify is exact; the draft only proposes). This confirms the boundary thesis: the off-the-shelf
draft (distilled vs FP gemma-4-12b-it) was the wall; matching OUR engine's distribution (OK_Q4B +
exact-integer/AltUp) is the lever. tok/s is still sequential-verify limited — the batched-verify
verb (previously gated by #2) is now worth building and would realize ~2.04x.

**Scope / honesty:** the measurement prompt is in-distribution (the corpus includes capital-of-X
prompts). The lift (0.258->1.042) is far too large to be noise, but a held-out-prompt A/B is the
next rigor step before a throughput claim. 1 GPU (local RTX 2060), 3 epochs, 310-prompt corpus.

## Pipeline (all GREEN)

1. **Capture** (`6f3c79d`): `SP_EAGLE_CAPTURE=1` one-shot greedy-rolls the served 12B over a corpus,
   dumping per generated position feat.f32[3840], x.f32[3840] (=target_embd[tok]*sqrt(3840)),
   inp/lbl/att.i32, and the full-seq target KV the draft attends (kg/vg[512] global owner kvfs-1,
   ks/vs[2048] SWA owner kvfs-2) + manifest. CUDA: `gemma4_kv_ctx_dump` / `gemma4_kv_ctx_geom` /
   `gemma4_embd_row`. Corpus: `make_corpus.py` (310 prompts). 310 seqs / 1070 MB.
2. **Train** (`f19dc47`, `66d84c4`): `sp_eagle_train.py` — differentiable torch port of the proven
   draft forward (pre_proj + 4 GQA-attn-over-captured-KV sandwich blocks + post_proj + tied head);
   CE vs captured label; trains pre/post/4-layers/output_norm (154.4M/47), head frozen. Batched head
   (bs128) + GPU-accumulate. Export = copy GGUF + in-place dtype-preserving patch (readback-validated).
   3 epochs: train_acc 0.823 -> 0.928 -> 0.951.
3. **Measure** (this contract): point `SP_DRAFT_GGUF` at the finetuned GGUF, re-run the accept drive.

## Held-out A/B (rigor, G-EAGLE-FLYWHEEL-AB-OOD, 2026-06-30)

OOD prompt ("write a short story about a lighthouse keeper who befriends a whale" — NOT in the
corpus): baseline 0.176/15.2% -> finetuned **0.709/68.5%** = the 4x accept lift GENERALIZES
(coherent story output). Not an overfitting artifact. Receipts: _ab_receipts/heldout_*.txt.

## THROUGHPUT — the honest negative (G-EAGLE-THROUGHPUT, 2026-06-30)

Profiled (SP_EAGLE_PROFILE) on the float probe, RTX 2060:

| config | tok/s | note |
|---|---|---|
| **plain greedy decode (K=0)** | **24.9** | the REAL baseline (no speculation) |
| spec, baseline draft, K=4, fast draft | 8.4 | post suppress-fix |
| spec, finetuned draft, K=4, fast draft | 4.4 | post suppress-fix |
| spec, finetuned draft, K=4, slow draft | 2.2 | pre suppress-fix |

Component costs: decode_logits=38ms/tok; draft_step=89ms -> **24ms** (suppress fix: device mask
kernel replaced 6248 sync cudaMemcpy/step); gemma4_kv_decode_batch (loop, one sync) = **1.00x**
(no amortization — sync was never the cost; the per-forward GPU compute is).

**Conclusion: speculative decoding does NOT yield a throughput win on this engine/HW.** (1) The
verify is sequential = 1 target forward per token, same as plain decode, so spec only ADDS the
draft cost. (2) The draft (24ms) is ~0.64x a target forward (38ms), not << , because the draft
carries the same 262144-vocab head as the target. Even a true batched-GEMM verify (weight-bound
forward, ~26ms weights) reaches only ~breakeven vs plain, since the draft cost dominates. A real
win would require BOTH draft_step -> ~5ms AND a batched-GEMM multi-position verify, for ~1.7x best
case. Receipts: _ab_receipts/profile_after_suppressfix.txt, finetuned_fastdraft.txt; K=0 run.

**Banked assets (not wasted):** the finetuned draft is a 96.6%-single-token target predictor that
generalizes OOD — valuable independent of throughput (cross-machine where target>>draft, or as a
predictor for recall/XBAR). The suppress fix is a permanent engine speedup (draft 3.7x faster).
gemma4_kv_decode_batch kept as the honest-negative verb (+ ready if a batched-GEMM forward lands).

## Next (decision pending)

- If throughput is critical: the two-arc path (draft_step -> ~5ms + batched-GEMM verify) for ~1.7x.
- Else: bank the finetuned draft + suppress fix; plain decode (24.9 tok/s) stands. Optionally still
  transcode the finetuned draft -> sp Q4 + scale the corpus (the draft QUALITY asset is reusable).
