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

## Next

- Held-out-prompt A/B (rigor).
- Batched-verify CUDA verb (now justified) -> realize ~2x tok/s.
- Transcode finetuned draft -> sp Q4 for deployment; scale corpus (-> Colab L4/RTX6000) for breadth.
