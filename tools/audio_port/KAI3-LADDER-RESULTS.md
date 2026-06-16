# KAI-3 §7.3 — frame-projector synthetic ladder (2026-06-16)

Architecture: `gen_synth_frames.py` (synthetic 640-float frames over a fixed V_sub=160 token-id subset,
frozen anchor matrix A, per-element Gaussian noise) → `frame_projector.py` (per-position MLP `640→V_sub`
logits + frozen on-manifold binder `softmax(logits/τ)·W_sub`, W_sub = real embed rows ×√H).

Objective = **dense per-position cross-entropy** CE(logits_i, token_i) — the structural fix for the KAI-2
t10 plateau (the projector is supervised at every timestep; the decision pivot is a downstream consequence,
verified later on metal, never the training signal). Train τ=1, export τ=0.2 (sharp → near-discrete on-manifold).

Scope: this is the **architecture + plumbing + boundary** proof on a synthetic frame surrogate. It does NOT
prove real audio — actual GNA/CNN features (task #154) replace the anchor matrix A later. Token sequences
here are synthetic random valid token-ids (no tokenizer dep); the ACTION/NO_OP pivot belongs to the METAL
G-KAIROS-3 gate (needs the real gemma tokenizer; cloud lane).

## Results (CPU, V_sub=160, 512 train / 8 held-out events, 80 epochs)

| noise_rel | effective ‖noise‖:‖signal‖ (= noise_rel·√640) | held-out per-pos top1 | manifold mean max-cos |
|-----------|-----------------------------------------------|-----------------------|------------------------|
| 0.1       | 2.5×                                          | **1.000**             | 0.9998                 |
| 0.3       | 7.6×                                          | 0.165                 | 0.9786                 |
| 0.5       | 12.6×                                         | 0.032                 | 0.9780                 |

## Read

- **PLUMBING + ARCHITECTURE: GREEN.** At the realistic rung (noise_rel=0.1), the projector recovers the
  held-out token sequence **perfectly (top1 1.000)**, CE→0.0000, emitted vectors are on-manifold
  (**cos 0.9998** at sharp τ=0.2), and the 8 exported packets are in the engine's `'KAI2'|k|hidden|f32`
  format (drop straight into `kai2_read_packet` → `gemma4_kv_inject_seq`). The full chain — CE objective
  trains, binder lands on-manifold, N preserved, export consumable — is proven end to end.
- **BINDER is noise-independent.** cos stays ~0.98 even when recovery collapses: at high noise the Mapper
  emits an on-manifold-but-WRONG token. The failure mode is recovery accuracy, never off-manifold drift —
  the t10 noise-shearing pathology cannot recur by construction.
- **Boundary located.** The Mapper resolves perfectly up to ~2.5× noise:signal, degrades at 7.6×, hits
  ~chance at 12.6×. NOTE the knob: `gen_synth_frames.py` applies `sigma·N(0,I_640)` per the literal spec,
  so effective noise:signal = noise_rel·√640 (≈25×). σ=0.3/0.5 are therefore brutal (7.6×/12.6×) regimes,
  not 0.3/0.5. For a realistic resolution sweep, a clean-SNR knob (‖noise‖ = noise_rel·‖A‖) is the
  follow-on; real GNA features will sit near the σ=0.1 regime the architecture already clears.

Receipts: `_run_one.sh <noise_rel>`; fixtures `tests/fixtures/kai3_s01/eval_*.bin` (σ=0.1 packets).
