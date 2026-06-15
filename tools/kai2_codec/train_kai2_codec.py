#!/usr/bin/env python3
"""
train_kai2_codec.py — KAI-2 phase-2 codec distillation runner (P2.b cloud lane).

GROUNDED ON THE METAL (verified 2026-06-15 from the local bf16 checkpoint
D:\\Files\\Models\\Gemma4\\gemma-4-12b-bucket\\model.safetensors):

  gemma-4-12B "Unified" is ENCODER-FREE. Its NATIVE AUDIO port is a SINGLE linear:
      model.embed_audio.embedding_projection.weight : bf16 [3840, 640]   (no bias)
  processor_config.json: audio_samples_per_token=640, audio_ms_per_token=40,
      sampling_rate=16000, feature_size=640, audio_seq_length=750 (=30 s @ 25 tok/s).
  i.e. 640 raw samples (40 ms @ 16 kHz) --nn.Linear(640->3840)--> residual stream.
  No pooler, no 400M middleman. The continuous-modality port IS a single matmul into
  hidden_size=3840 — EXACTLY the residual-entry seam gemma4_kv_inject writes to
  (post-embed, pre-layer, at dpos). So the operator's "treat the event like a 40 ms
  frame, single linear" is not an analogy: it is the model's native mechanism.

THE KAI-2 CODEC (operator's design, adopted) mirrors that native projector: a single
nn.Linear maps a FIXED-WIDTH raw event vector -> k*hidden soft tokens. We do NOT need a
Perceiver (tools/kai2_codec/event_resampler.py is now the heavier fallback only).

DISTILLATION (frozen teacher, Forward-KL):
  teacher = frozen 12B reading the FULL text event frame  -> action-pivot logits (the §6.2
            measured "text path": ~44 delivery steps then ACTION/NO_OP).
  student = the SAME frozen 12B but the event tokens' input embeddings are REPLACED by the
            codec's k soft vectors (the soft-token substitution gemma4_kv_inject does) -> logits.
  loss    = Forward-KL(student || teacher) at the decision position. Backprop ONLY the codec.
  Selectivity is inherited: a balanced corpus (salient->ACTION, idle->NO_OP) means matching the
  teacher's distribution reproduces NO_OP on idle events for free (the §6.3 gate's 2x2 arm).

Pre-registered gate (CONTRACT-KAIROS §6.3): null floor preserved (engine decode byte-untouched),
trained k<=4 packet pivots the resident to the correct <ACTION> in <=2 decode steps (vs 44 text),
selectivity 2x2 held. This script trains the packet; the C engine harness runs the gate.

COMPUTE: the frozen 12B in the loop needs bf16 ~24 GB -> cloud (P2.b RunPod/Colab A6000/A100).
Use --smoke for a CPU/2060 shape+gradient sanity run with a tiny random stand-in teacher.

Usage:
  # cloud, real teacher from the local (or HF) checkpoint:
  python train_kai2_codec.py --model google/gemma-4-12B --k 4 --epochs 3 --out codec_k4.pt
  python train_kai2_codec.py --model /path/to/gemma-4-12b-bucket --k 4 ...
  # local sanity (no 24 GB load):
  python train_kai2_codec.py --smoke
"""
from __future__ import annotations
import argparse, json, os, struct
import torch
import torch.nn as nn
import torch.nn.functional as F

HIDDEN_DEFAULT = 3840          # gemma-4-12B text hidden_size (verified config.json)
RAW_EVENT_DIM_DEFAULT = 2048   # fixed-width raw-event buffer (event bytes -> normalized floats)


# ───────────────────────── the codec (operator's single-linear design) ─────────────────────────
class KAI2Codec(nn.Module):
    """Fixed-width raw event vector -> k soft vectors in the model's embedding space.

    A SINGLE nn.Linear, mirroring gemma-4-12B's native embed_audio.embedding_projection
    [3840,640]. raw_event_dim is the width of the event's continuous encoding (bytes->floats,
    or pooled token embeddings — see encode_event_raw); we map it straight to k*hidden and
    reshape to k sequential residual-entry vectors. No pooler, no encoder.
    """

    def __init__(self, raw_event_dim: int, hidden_dim: int = HIDDEN_DEFAULT, k: int = 4):
        super().__init__()
        self.k, self.hidden_dim = k, hidden_dim
        self.projection = nn.Linear(raw_event_dim, hidden_dim * k)
        # residual-magnitude scale (Gemma scales embeddings by sqrt(hidden)); learnable so the
        # injected packet matches what the layers expect to "see" at the seam.
        self.out_scale = nn.Parameter(torch.ones(1) * (hidden_dim ** 0.5))

    def forward(self, raw_event: torch.Tensor) -> torch.Tensor:   # (B, raw_event_dim) -> (B, k, H)
        y = self.projection(raw_event).view(-1, self.k, self.hidden_dim)
        return y * (self.out_scale / (self.hidden_dim ** 0.5))


def distillation_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                      temperature: float = 1.0) -> torch.Tensor:
    """Forward-KL of the teacher's action-pivot distribution onto the student. logits: (B, V) at
    the decision step. Frozen LLM => grads flow only into the codec that made the injected vectors."""
    log_p_s = F.log_softmax(student_logits / temperature, dim=-1)
    p_t = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * (temperature * temperature)


# ───────────────────────── raw-event front-end (event -> fixed-width vector) ─────────────────────
def encode_event_raw(event_text: str, raw_event_dim: int = RAW_EVENT_DIM_DEFAULT) -> torch.Tensor:
    """The literal "treat the event like a waveform" encoding: UTF-8 bytes of the event payload,
    padded/truncated to raw_event_dim, mapped to [-1,1]. A single continuous fixed-width array,
    exactly the shape the native audio projector consumes. (Alternative: pooled token embeddings;
    kept simple + tokenizer-free here so the codec front-end has zero model dependency.)"""
    b = event_text.encode("utf-8")[:raw_event_dim]
    buf = torch.zeros(raw_event_dim, dtype=torch.float32)
    if b:
        arr = torch.frombuffer(bytearray(b), dtype=torch.uint8).float()
        buf[: arr.numel()] = (arr / 127.5) - 1.0
    return buf


# ───────────────────────── corpus (event frame, raw vector, expect) ─────────────────────────────
def build_corpus(path: str | None):
    """Each item: dict(text=<event frame text>, expect=<"ACTION"|"NO_OP">). If no path, a tiny
    balanced built-in tape (salient->ACTION, idle->NO_OP) so the runner is self-contained."""
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]
    salient = [
        "EVENT build_id=4471 status=FAILED tests=3_broken salience=0.85",
        "EVENT disk mount=/ used=96% salience=0.80",
        "EVENT cert domain=api ttl=12h expiring salience=0.78",
        "EVENT deploy svc=auth health=crashloop salience=0.90",
    ]
    idle = [
        "EVENT heartbeat ok cpu=12% salience=0.10",
        "EVENT cron job=rotate_logs done salience=0.15",
        "EVENT user login region=eu ok salience=0.20",
        "EVENT cache warm hit_rate=0.99 salience=0.08",
    ]
    return ([{"text": t, "expect": "ACTION"} for t in salient] +
            [{"text": t, "expect": "NO_OP"} for t in idle])


# ───────────────────────── teacher/student forward (the frozen 12B in the loop) ──────────────────
SYSTEM = ("You are a background monitor. Read the event. Emit <ACTION> if it needs intervention "
          "(salience>=0.5), else NO_OP.\nEVENT: ")
DECIDE = "\nDECIDE: "


def load_teacher(model_id: str):
    from transformers import AutoModelForMultimodalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForMultimodalLM.from_pretrained(model_id, dtype="auto", device_map="auto")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    # the text decoder + input embedding table (robust to wrapper layout)
    lm = getattr(model, "language_model", None) or getattr(model, "model", model)
    embed = model.get_input_embeddings()
    return model, lm, embed, tok


def decision_logits_text(model, embed, tok, event_text, device):
    """Teacher path: full text event in the scaffold; logits at the decision step (next token)."""
    ids = tok(SYSTEM + event_text + DECIDE, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model(input_ids=ids)
    return out.logits[:, -1, :]              # (1, V)


def decision_logits_inject(model, embed, tok, event_text, codec, raw_dim, device):
    """Student path: replace the event tokens' embeddings with k codec soft vectors, keep scaffold."""
    pre = tok(SYSTEM, return_tensors="pt").input_ids.to(device)
    post = tok(DECIDE, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    pre_e, post_e = embed(pre), embed(post)                                  # (1,Tp,H),(1,Tq,H)
    raw = encode_event_raw(event_text, raw_dim).unsqueeze(0).to(device)
    soft = codec(raw).to(pre_e.dtype)                                        # (1,k,H) — grad path
    inputs_embeds = torch.cat([pre_e, soft, post_e], dim=1)
    out = model(inputs_embeds=inputs_embeds)
    return out.logits[:, -1, :]


# ───────────────────────────────────────── train ────────────────────────────────────────────────
def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    corpus = build_corpus(args.corpus)
    codec = KAI2Codec(args.raw_dim, args.hidden, args.k).to(device)
    opt = torch.optim.AdamW(codec.parameters(), lr=args.lr)

    if args.smoke:
        # tiny random stand-in teacher: just exercises codec shapes + KL gradient on the 2060/CPU.
        V = 256
        teach = nn.Linear(args.hidden * args.k, V).to(device)   # fake "decision head" over k vectors
        for p in teach.parameters():
            p.requires_grad_(False)
        for ep in range(args.epochs):
            tot = 0.0
            for it in corpus:
                raw = encode_event_raw(it["text"], args.raw_dim).unsqueeze(0).to(device)
                soft = codec(raw)
                s_log = teach(soft.flatten(1))
                t_log = torch.zeros_like(s_log)
                t_log[0, 1 if it["expect"] == "ACTION" else 0] = 8.0    # teacher "pivots" to a class
                loss = distillation_loss(s_log, t_log, args.tau)
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item()
            print(f"[smoke] epoch {ep} mean_KL={tot/len(corpus):.4f}")
        print(f"[smoke] OK codec params={sum(p.numel() for p in codec.parameters()):,} "
              f"(single linear {args.raw_dim}->{args.hidden*args.k})")
        torch.save({"codec": codec.state_dict(), "args": vars(args)}, args.out)
        return

    model, lm, embed, tok = load_teacher(args.model)
    for ep in range(args.epochs):
        tot = 0.0
        for it in corpus:
            t_log = decision_logits_text(model, embed, tok, it["text"], device)
            s_log = decision_logits_inject(model, embed, tok, it["text"], codec, args.raw_dim, device)
            loss = distillation_loss(s_log, t_log, args.tau)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        print(f"[train] epoch {ep} mean_KL={tot/len(corpus):.4f}")
    torch.save({"codec": codec.state_dict(), "args": vars(args)}, args.out)
    print(f"[train] saved {args.out}")


# ───────────────────────── export k-vector packets for the C engine (gemma4_kv_inject) ───────────
def export_packets(args):
    """Write per-event .bin packets (k x hidden float32, row-major) the engine injects via
    gemma4_kv_inject. Header: magic 'KAI2', u32 k, u32 hidden, then k*hidden float32."""
    ckpt = torch.load(args.out, map_location="cpu")
    a = ckpt["args"]
    codec = KAI2Codec(a["raw_dim"], a["hidden"], a["k"]); codec.load_state_dict(ckpt["codec"]); codec.eval()
    os.makedirs(args.packets_dir, exist_ok=True)
    for i, it in enumerate(build_corpus(args.corpus)):
        raw = encode_event_raw(it["text"], a["raw_dim"]).unsqueeze(0)
        with torch.no_grad():
            vecs = codec(raw)[0].contiguous().float().numpy()    # (k, hidden)
        fn = os.path.join(args.packets_dir, f"event_{i:03d}_{it['expect']}.bin")
        with open(fn, "wb") as f:
            f.write(b"KAI2"); f.write(struct.pack("<II", a["k"], a["hidden"])); f.write(vecs.tobytes())
        print(f"[export] {fn}  k={a['k']} hidden={a['hidden']} expect={it['expect']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-12B", help="HF id or local bucket path (frozen teacher)")
    ap.add_argument("--corpus", default=None, help="jsonl of {text,expect}; default = built-in tape")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=HIDDEN_DEFAULT)
    ap.add_argument("--raw_dim", type=int, default=RAW_EVENT_DIM_DEFAULT)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--out", default="kai2_codec_k4.pt")
    ap.add_argument("--smoke", action="store_true", help="tiny random teacher; CPU/2060 shape+grad sanity")
    ap.add_argument("--export", action="store_true", help="after/without training, export .bin packets")
    ap.add_argument("--packets_dir", default="kai2_packets")
    args = ap.parse_args()
    if args.export and os.path.exists(args.out):
        export_packets(args)
    else:
        train(args)
        if args.export:
            export_packets(args)


if __name__ == "__main__":
    main()
