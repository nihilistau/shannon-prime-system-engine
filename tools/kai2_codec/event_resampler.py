"""
KAI-2 phase-2 codec — the event resampler (DESIGN DRAFT, 2026-06-15).

Why this exists
---------------
KAI-2 phase-1 proved the latent-interrupt SEAM (gemma4_kv_inject) is bit-exact inert
(G-KAIROS-2 self-null GREEN) and live across all 48 layers. But the A/B showed that an
UNTRAINED k=1 mean-pool of a 44-token event does NOT pivot the resident to an ACTION:
averaging destroys the sequence/phase structure. Text delivery costs ~44 steps; a working
compressed packet would collapse that to ~1 (the ~44x prize).

The fix, grounded in how OUR model (VERIFIED: google/gemma-4-12B "Unified", model_type
gemma4_unified, the encoder-free 5th Gemma-4 size) natively ingests continuous modalities.
config.json ground truth: the 12B is NATIVELY MULTIMODAL incl. AUDIO — it "projects raw image
patches and audio waveforms directly into the LLM's embedding space through lightweight linear
layers." The native audio injection = SOFT-TOKEN SUBSTITUTION: a placeholder audio_token_id=258881
(bracketed by boa=256000 / eoa=258883) whose embedding is REPLACED by the projected audio-frame
vector — which is EXACTLY what gemma4_kv_inject does at the post-embed residual (dpos). Audio
runs at dim 640, audio_samples_per_token=640 (40ms/frame @16kHz ~25 tok/s), then a projector
lifts 640 -> text hidden 3840. So the frozen model was JOINTLY TRAINED to interpret continuous
non-text vectors at our exact inject port.

SUPERSEDED AS PRIMARY (2026-06-15): tensor inspection of the local bf16 checkpoint showed the
native audio projector is a SINGLE linear (model.embed_audio.embedding_projection.weight [3840,640],
no bias) — no pooler, no encoder. So the adopted codec is a single nn.Linear (see
tools/kai2_codec/train_kai2_codec.py :: KAI2Codec); this Perceiver resampler is the HEAVIER FALLBACK
only (kept for the case the single-linear packet underfits the action-pivot).

Two routes (see CONTRACT-KAIROS §6.3):
  ROUTE A (native-audio mimic, strongest): extract the audio projector from the HF safetensors,
    deliver the event AS audio (TTS or synthesized 640-sample frames) -> real audio-soft-token
    vectors -> inject at audio_token_id positions. Frozen model natively "hears" it; maybe ZERO
    training. Decisive cheap probe G-KAIROS-2-NATIVE: dump real audio embeddings from HF once,
    inject on our engine, watch the pivot.
  ROUTE B (this file, the fallback): a learned Perceiver resampler -> k<=4 soft vectors, trained
    by distillation from the working TEXT path (no audio weights needed). The audio path PROVES
    the target manifold exists + is natively injectable; this lands a learned packet on it.
NOTE: our .sp-model is TEXT-ONLY (sp_transcode captured no audio/vision projection tensors), so
Route A needs the HF audio-projector weights; Route B needs none. E = text hidden_size = 3840.

Training signal (cheap + label-free): DISTILLATION from the text path we already proved works.
  teacher = frozen 12B reading the full text frame -> its action-token logits / hidden traj.
  student = inject the resampler's k vectors -> match the teacher's action logits.
This is the same forward-KL-distill recipe that won the LSH router (tools/xbar_lsh/train_lsh.py).

COMPUTE: a frozen 12B in the loop does NOT fit the 2060 (needs bf16 ~24GB) -> cloud lane
(the P2.b RunPod/Colab lane). This file is the architecture + loss DRAFT only; the cloud
wiring (load HF Gemma-3-12B, the inject forward hook, the corpus) is marked TODO.

Pre-registered target (contract G-KAIROS-2 phase 2): the trained k<=4 packet pivots the
resident to the correct <ACTION> in <= 2 decode steps (vs 44 text-delivery steps), with
selectivity (idle/low-salience event -> NO_OP) held.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class EventResampler(nn.Module):
    """Variable-length event token embeddings -> k fixed soft vectors in E-dim.

    Cross-attention (Perceiver) pooling: k learned query latents attend over the event's
    token embeddings, so the output preserves which-token-said-what (unlike a mean-pool).
    Output vectors live in the model's embedding space (E) and are injected as residual
    entries at consecutive dpos coordinates. Frozen-LLM training updates only THIS module.
    """

    def __init__(self, e_dim: int, k: int = 4, n_heads: int = 8, depth: int = 2, p_drop: float = 0.0):
        super().__init__()
        self.e_dim, self.k = e_dim, k
        # k learned query latents (the "soft token" slots)
        self.queries = nn.Parameter(torch.randn(k, e_dim) * (e_dim ** -0.5))
        self.in_norm = nn.LayerNorm(e_dim)
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "q_norm": nn.LayerNorm(e_dim),
                "kv_norm": nn.LayerNorm(e_dim),
                "attn": nn.MultiheadAttention(e_dim, n_heads, dropout=p_drop, batch_first=True),
                "ff_norm": nn.LayerNorm(e_dim),
                "ff": nn.Sequential(nn.Linear(e_dim, 4 * e_dim), nn.GELU(), nn.Linear(4 * e_dim, e_dim)),
            })
            for _ in range(depth)
        ])
        self.out_norm = nn.LayerNorm(e_dim)
        # scale the output into the residual's natural magnitude (post-embed-scale lives ~ sqrt(E));
        # learnable so the injected packet matches what the layers expect to "see".
        self.out_scale = nn.Parameter(torch.ones(1) * (e_dim ** 0.5))

    def forward(self, event_emb: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """event_emb: (B, T, E) post-embed-scale token embeddings of the event frame.
        returns: (B, k, E) soft vectors to inject at k consecutive residual coordinates."""
        B = event_emb.shape[0]
        ctx = self.in_norm(event_emb)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)            # (B, k, E)
        for blk in self.blocks:
            a, _ = blk["attn"](blk["q_norm"](q), blk["kv_norm"](ctx), ctx,
                               key_padding_mask=key_padding_mask, need_weights=False)
            q = q + a
            q = q + blk["ff"](blk["ff_norm"](q))
        q = self.out_norm(q)
        return q * (self.out_scale / (self.e_dim ** 0.5))           # back to residual magnitude


def distill_loss(student_action_logits: torch.Tensor,
                 teacher_action_logits: torch.Tensor,
                 tau: float = 1.0,
                 hidden_student: torch.Tensor | None = None,
                 hidden_teacher: torch.Tensor | None = None,
                 lambda_hidden: float = 0.0) -> torch.Tensor:
    """Forward-KL distillation of the teacher's (text-frame) next-token distribution onto the
    student's (injected-packet) distribution at the decision step, + optional hidden-state match.
    Both logits: (B, V) at the step where the teacher emits <ACTION>. Frozen-LLM ⇒ grads flow
    only into the resampler (which produced the injected vectors)."""
    p_t = F.softmax(teacher_action_logits / tau, dim=-1)
    log_p_s = F.log_softmax(student_action_logits / tau, dim=-1)
    loss = F.kl_div(log_p_s, p_t, reduction="batchmean") * (tau * tau)
    if lambda_hidden > 0.0 and hidden_student is not None and hidden_teacher is not None:
        loss = loss + lambda_hidden * F.mse_loss(hidden_student, hidden_teacher)
    return loss


# ─────────────────────────────────────────────────────────────────────────────────────────
# CLOUD-WIRING TODO (next session, P2.b RunPod/Colab lane — does NOT fit the 2060):
#   1. Load HF google/gemma-4-12B (bf16) FROZEN; grab the post-embed-scale embeddings for both
#      (a) the full text frame [teacher] and (b) the event content tokens [resampler input].
#   2. Forward hook at the post-embed residual: replace the first k positions' residual with
#      the resampler output (the PyTorch analog of gemma4_kv_inject), run frozen, read the
#      action-step logits = student_action_logits. Teacher = frozen model on the full frame.
#   3. Corpus: synthesize (event-frame, salience, expect=ACTION/NO_OP) tuples from the §2b tape
#      format; balance salient/idle for the selectivity arm.
#   4. Train ONLY EventResampler with distill_loss (+ a NO_OP/idle negative so idle events do
#      NOT pivot — the selectivity half of the gate).
#   5. Export the trained k-vector packets per event-class -> a .bin the C engine injects via
#      gemma4_kv_inject; re-run SP_G4_KAIROS_INTERRUPT for the real G-KAIROS-2 latency/selectivity.
# ─────────────────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # shape smoke (CPU, no checkpoint): resampler maps (B,T,E) -> (B,k,E)
    E, k = 3840, 4                    # gemma-4-12B text hidden_size (VERIFIED from config.json)
    m = EventResampler(E, k=k)
    x = torch.randn(2, 44, E)         # 2 events, 44-token frames
    y = m(x)
    assert y.shape == (2, k, E), y.shape
    print(f"EventResampler OK: (B=2,T=44,E={E}) -> {tuple(y.shape)}; params={sum(p.numel() for p in m.parameters()):,}")
