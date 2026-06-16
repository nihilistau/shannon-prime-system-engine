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
import argparse, json, os, struct, sys
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

    def __init__(self, raw_event_dim: int, hidden_dim: int = HIDDEN_DEFAULT, k: int = 4, n_sub: int = 0, tau: float = 1.0):
        super().__init__()
        self.k, self.hidden_dim, self.n_sub, self.tau = k, hidden_dim, n_sub, tau
        # n_sub>0 = ON-MANIFOLD HEAD (t9): emit softmax over a frozen subset of the model's real embedding
        # rows ⇒ output is a convex combination of native embeddings, STRUCTURALLY on-manifold (a one-hot
        # softmax == the exact EMB injection that pivots). n_sub==0 = legacy free single-linear (t4-t8).
        out = (n_sub if n_sub > 0 else hidden_dim) * k
        self.projection = nn.Linear(raw_event_dim, out)
        if n_sub == 0:
            self.out_scale = nn.Parameter(torch.ones(1) * (hidden_dim ** 0.5))
        self.WE = None  # (n_sub, H) frozen embedding subset ×embscale; set externally, NOT a param/buffer

    def forward(self, raw_event: torch.Tensor) -> torch.Tensor:   # (B, raw_event_dim) -> (B, k, H)
        if self.n_sub > 0:
            logits = self.projection(raw_event).view(-1, self.k, self.n_sub)
            w = torch.softmax(logits / self.tau, dim=-1)          # (B,k,N) mixture over real rows
            return w @ self.WE.to(w.dtype)                        # (B,k,H) — convex combo, on-manifold
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
# HELD-OUT EVAL events = the original 8 built-ins. They are EXCLUDED from gen_corpus() so the metal gate
# (which hard-codes event_000 "build_id=4471..." salient + event_004 "heartbeat..." idle) is a TRUE
# generalization test — the codec never trains on them. export_packets() exports for THESE.
EVAL_EVENTS = (
    [{"text": t, "expect": "ACTION"} for t in [
        "EVENT build_id=4471 status=FAILED tests=3_broken salience=0.85",
        "EVENT disk mount=/ used=96% salience=0.80",
        "EVENT cert domain=api ttl=12h expiring salience=0.78",
        "EVENT deploy svc=auth health=crashloop salience=0.90"]] +
    [{"text": t, "expect": "NO_OP"} for t in [
        "EVENT heartbeat ok cpu=12% salience=0.10",
        "EVENT cron job=rotate_logs done salience=0.15",
        "EVENT user login region=eu ok salience=0.20",
        "EVENT cache warm hit_rate=0.99 salience=0.08"]])


def gen_corpus(n: int = 512, seed: int = 0):
    """Template grammar (type × fields × salience) → n balanced events. Breaks the per-event shortcut
    (8→512) and provides the held-out generalization split. Excludes EVAL_EVENTS verbatim."""
    import random
    rng = random.Random(seed)
    SAL = ["EVENT build_id={bid} status=FAILED tests={t}_broken salience={s}",
           "EVENT disk mount={mnt} used={u}% salience={s}",
           "EVENT cert domain={d} ttl={h}h expiring salience={s}",
           "EVENT deploy svc={svc} health=crashloop salience={s}",
           "EVENT oom_killed proc={svc} mem={u}% salience={s}",
           "EVENT latency svc={svc} p99={p}ms breach salience={s}",
           "EVENT queue name={svc} depth={bid} backlog salience={s}",
           "EVENT security ip={d} brute_force attempts={t} salience={s}"]
    IDLE = ["EVENT heartbeat ok cpu={u}% salience={s}",
            "EVENT cron job={svc} done salience={s}",
            "EVENT user login region={d} ok salience={s}",
            "EVENT cache warm hit_rate=0.{p} salience={s}",
            "EVENT backup svc={svc} ok bytes={bid} salience={s}",
            "EVENT metric svc={svc} p99={p}ms nominal salience={s}",
            "EVENT health svc={svc} green salience={s}",
            "EVENT scale svc={svc} replicas={t} steady salience={s}"]
    eval_set = {e["text"] for e in EVAL_EVENTS}
    def fill(tmpl, sal):
        s = round(rng.uniform(0.50, 0.99) if sal else rng.uniform(0.00, 0.49), 2)
        return tmpl.format(bid=rng.randint(1000, 9999), t=rng.randint(1, 9), u=rng.randint(80, 99) if sal else rng.randint(1, 40),
                           d=rng.choice(["api", "auth", "cdn", "db", "eu", "us", "ap", "edge"]), h=rng.randint(1, 24),
                           svc=rng.choice(["auth", "web", "db", "cache", "queue", "worker", "gw", "scheduler"]),
                           p=rng.randint(10, 99), mnt=rng.choice(["/", "/var", "/data", "/srv"]), s=("%.2f" % s))
    out = []
    half = n // 2
    for sal, tmpls in ((True, SAL), (False, IDLE)):
        seen = set()
        while len([o for o in out if (o["expect"] == "ACTION") == sal]) < half:
            txt = fill(rng.choice(tmpls), sal)
            if txt in eval_set or txt in seen:
                continue
            seen.add(txt)
            out.append({"text": txt, "expect": "ACTION" if sal else "NO_OP"})
    rng.shuffle(out)
    return out


def build_corpus(path: str | None):
    """jsonl path → that corpus; else the 512-event generated grammar (t8 anchor regime)."""
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]
    return gen_corpus(512, 0)


# ───────────────────────── teacher/student forward (the frozen 12B in the loop) ──────────────────
# EXACT gemma chat-template scaffold the 6h soak proved selective (the raw template-less version lobotomized
# the 12B's instruction-following -> non-selective teacher -> the v1 codec memorized that failure, G-KAIROS-2
# RED 2026-06-16). SYSTEM = the system daemon turn + the OPEN of the event user-turn; DECIDE = the CLOSE of the
# user-turn + the model-turn start. The k codec soft-vectors are injected BETWEEN them, in place of the event text.
SYSTEM = ("<start_of_turn>user\n"
          "You are a background kernel daemon. Each tick you receive one environment event "
          "that carries a salience score between 0.0 and 1.0. Reply with EXACTLY one of:\n"
          "NO_OP\n<ACTION>short imperative</ACTION>\n"
          "RULE: if salience >= 0.5 the event requires intervention, so reply with an <ACTION> line. "
          "If salience < 0.5, reply NO_OP. Follow the rule exactly. Do not explain."
          "<end_of_turn>\n<start_of_turn>model\nUnderstood.<end_of_turn>\n"
          "<start_of_turn>user\n")
DECIDE = "<end_of_turn>\n<start_of_turn>model\n"


@torch.no_grad()
def _ok_q4b_fakequant_(w: torch.Tensor, blk: int = 32) -> None:
    """In-place OK_Q4B fake-quant of a frozen weight: per-32-block symmetric int4 [-7,7] + per-block
    (f16) scale, exactly the engine's .sp-model OK_Q4B codec. dequant = round(w/s*7)*s/7, s=max|w| per
    block along the contiguous (input) dim. Frozen weights ⇒ constants ⇒ no STE needed; the codec's
    gradients flow through the quantized-VALUE forward, so it learns the served geometry (Step 2)."""
    orig_shape = w.shape; cols = w.shape[-1]
    if cols % blk != 0:
        return  # leave odd-width tensors (norms etc.) alone
    f = w.reshape(-1, cols // blk, blk).float()
    s = f.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    s = s.half().float()                                   # f16 block scale (match the container)
    code = torch.clamp(torch.round(f / s * 7.0), -7, 7)
    w.copy_((code * s / 7.0).reshape(orig_shape).to(w.dtype))


def _apply_ok_q4b(model) -> int:
    """Fake-quant every 2D weight (Linear matmuls + the embedding table) to OK_Q4B in place."""
    n = 0
    for mod in model.modules():
        wt = getattr(mod, "weight", None)
        if isinstance(wt, torch.nn.Parameter) and wt.dim() >= 2 and wt.shape[-1] >= 32:
            _ok_q4b_fakequant_(wt.data); n += 1
    return n


def load_teacher(model_id: str, fakeq: bool = False):
    from transformers import AutoModelForMultimodalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForMultimodalLM.from_pretrained(model_id, dtype="auto", device_map="auto")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if fakeq:
        nq = _apply_ok_q4b(model)
        print(f"[train] OK_Q4B FAKE-QUANT applied to {nq} weight tensors (Step 2: distill against the "
              f"served quantized geometry)", flush=True)
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


# Option 2 (AltUp-constrained, 2026-06-16): the inputs_embeds bypass FAILED to transfer to the C-engine
# (G-KAIROS-2 RED) because gemma-4 is AltUp — the residual is the 0th of N prediction streams, and the
# per-layer embeddings (PLE) gathered from the TOKEN IDs drive the AltUp predictors at every block. The
# engine's gemma4_kv_inject overrides ONLY the post-embed 0th stream at placeholder positions, leaving the
# placeholder token's PLE intact; the AltUp predictors then "correct" the 0th-stream override back toward
# the placeholder ⇒ injection erased (proven on metal: INJSCALE 0==1==62==1000, NOPLE on/off all byte-
# identical). FIX: distill against the SAME mechanics. Feed real input_ids with a PLACEHOLDER token at the
# k soft positions (so HF runs the identical PLE/AltUp gathers the engine does), and overwrite ONLY the
# 0th-stream base embedding at those positions via a forward HOOK on embed_tokens — mirroring engine
# cuda_forward.cu lines 3355-3357 exactly. The codec must now learn a 0th-stream vector that SURVIVES the
# 48-layer AltUp prediction gauntlet. Placeholder = gemma's native continuous-modality token (audio_token_id):
# the model already knows how to take an externally-supplied 0th-stream embedding at that token id.
PLACEHOLDER_ID = 258881  # gemma-4 audio_token_id; overwritten from config at load (load_teacher)


def decision_logits_inject(model, embed, tok, event_text, codec, raw_dim, device, k):
    """Student path (Option 2): placeholder tokens carry PLE through AltUp; a forward hook overwrites ONLY
    the 0th-stream base embedding at the k placeholder positions with the codec vectors — the exact
    mechanical twin of the C-engine's gemma4_kv_inject seam."""
    pre = tok(SYSTEM, return_tensors="pt").input_ids.to(device)              # (1,Tp)  BOS + sys + user-open
    post = tok(DECIDE, return_tensors="pt", add_special_tokens=False).input_ids.to(device)  # (1,Tq)
    Tp = pre.shape[1]
    ph = torch.full((1, k), PLACEHOLDER_ID, dtype=pre.dtype, device=device)  # k placeholder tokens
    input_ids = torch.cat([pre, ph, post], dim=1)                            # real token sequence (PLE-bearing)
    raw = encode_event_raw(event_text, raw_dim).unsqueeze(0).to(device)
    soft = codec(raw)                                                        # (1,k,H) — grad path into codec
    emb_mod = model.get_input_embeddings()
    soft_c = soft.to(next(emb_mod.parameters()).dtype)
    def _hook(_m, _a, output):                                               # fires POST-embed (post-scale), like engine post-k_embed_scale
        output = output.clone()
        output[:, Tp:Tp + k, :] = soft_c                                     # overwrite ONLY 0th stream at placeholders
        return output
    h = emb_mod.register_forward_hook(_hook)
    try:
        out = model(input_ids=input_ids)                                     # AltUp/PLE fire on the placeholder ids
    finally:
        h.remove()
    return out.logits[:, -1, :], soft                                        # soft (1,k,H) for the anchor loss


def anchor_loss(soft, event_text, embed, tok, device):
    """t8 manifold-anchor: min-cosine distance from each codec vector to the event's REAL token embeddings.
    L_anchor = (1/k) Σ_i min_j (1 − cos(C_i, E_j)). Forces the codec output ONTO the native embedding
    manifold — the diagnostic proved bf16/quant-aware codecs landed at random-noise cosine (0.078) and were
    sheared to nothing on metal, while real embeddings pivot. Scale-invariant (cosine), so codec-raw vs
    scaled E_j is fine."""
    ids = tok(event_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)  # (1,n)
    with torch.no_grad():
        E = embed(ids)[0]                                                    # (n,H) fixed target
    C = soft[0]                                                              # (k,H) grad
    Cn = F.normalize(C, dim=-1); En = F.normalize(E.to(C.dtype), dim=-1)
    sims = Cn @ En.t()                                                       # (k,n)
    return (1.0 - sims.max(dim=1).values).mean()


@torch.no_grad()
def manifold_maxcos(codec, embed, events, raw_dim, device, n_ev=4):
    """Pre-export gate: mean over a few events of (mean over the k codec vectors of) max cosine to ANY
    embedding row. Random baseline ≈0.07; on-manifold ≫. Aborts export if the codec is still noise."""
    Wn = F.normalize(embed.weight.float(), dim=-1)                           # (V,H) — Q4B rows if --fakeq
    vals = []
    for it in events[:n_ev]:
        raw = encode_event_raw(it["text"], raw_dim).unsqueeze(0).to(device)
        Cn = F.normalize(codec(raw)[0].float(), dim=-1)                      # (k,H)
        best = torch.full((Cn.shape[0],), -1.0, device=device)
        for s in range(0, Wn.shape[0], 65536):
            best = torch.maximum(best, (Cn @ Wn[s:s+65536].t()).max(1).values)
        vals.append(best.mean().item())
    return sum(vals) / len(vals)


def teacher_decision(model, tok, event_text, device):
    """Greedy-generate the teacher's decision on the TEXT path; classify ACTION vs NO_OP (mirrors the
    engine gate's detection). Used for the pre-distillation selectivity gate."""
    ids = tok(SYSTEM + event_text + DECIDE, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model.generate(input_ids=ids, max_new_tokens=8, do_sample=False)
    txt = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=False).replace("\n", " ")
    if "ACTION" in txt:
        return "ACTION", txt
    if "OP" in txt or "NOOP" in txt:
        return "NO_OP", txt
    return "NEITHER", txt


# ───────────────────────────────────────── train ────────────────────────────────────────────────
def build_subset(corpus, embed, tok, device, cap=4096):
    """t9 on-manifold head: the union of token ids across the corpus + EVAL_EVENTS + scaffold = the exact
    event vocabulary the codec must span (held-out events share the template grammar). embed(ids) is the
    ScaledWordEmbedding output (already ×√H) = the post-scale residual a real token produces == the EMB
    injection. So a one-hot softmax over this subset reproduces a pivoting injection exactly."""
    ids = set()
    for it in list(corpus) + list(EVAL_EVENTS):
        ids.update(tok(it["text"], add_special_tokens=False).input_ids)
    for s in (SYSTEM, DECIDE):
        ids.update(tok(s, add_special_tokens=False).input_ids)
    ids = sorted(ids)[:cap]
    with torch.no_grad():
        WE = embed(torch.tensor(ids, dtype=torch.long, device=device)).float()   # (N,H) post-scale
    return ids, WE


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    corpus = build_corpus(args.corpus)

    if args.smoke:
        codec = KAI2Codec(args.raw_dim, args.hidden, args.k).to(device)
        opt = torch.optim.AdamW(codec.parameters(), lr=args.lr)
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

    model, lm, embed, tok = load_teacher(args.model, getattr(args, "fakeq", False))
    # Option 2: pin the placeholder token to gemma's native continuous-modality id (config-driven).
    global PLACEHOLDER_ID
    cfg = model.config
    pid = getattr(cfg, "audio_token_id", None)
    if pid is None and hasattr(cfg, "text_config"): pid = getattr(cfg.text_config, "audio_token_id", None)
    if pid is not None: PLACEHOLDER_ID = int(pid)
    print(f"[train] PLACEHOLDER_ID={PLACEHOLDER_ID} (gemma native continuous-modality token; PLE-bearing); "
          f"vocab={getattr(cfg,'vocab_size',getattr(getattr(cfg,'text_config',None),'vocab_size','?'))}", flush=True)
    # ── PRE-DISTILLATION TEACHER-SELECTIVITY GATE (ABORT if the distillation target is invalid) ──
    sal = next(it for it in corpus if it["expect"] == "ACTION")
    idl = next(it for it in corpus if it["expect"] == "NO_OP")
    sd, stx = teacher_decision(model, tok, sal["text"], device)
    idd, itx = teacher_decision(model, tok, idl["text"], device)
    selective = (sd == "ACTION" and idd == "NO_OP")
    print(f"[teacher-check] salient->{sd} (\"{stx[:48]}\") | idle->{idd} (\"{itx[:48]}\") | SELECTIVE={selective}", flush=True)
    if not selective:
        print("[teacher-check] ABORT rc=4: teacher is NOT selective on this scaffold — distilling against it is "
              "garbage-in. Fix the scaffold/prompt before training.", flush=True)
        sys.exit(4)
    # ── build the codec: on-manifold softmax head (t9) or legacy free linear (t4-t8) ──
    if getattr(args, "onmanifold", False):
        sub_ids, WE = build_subset(corpus, embed, tok, device)
        codec = KAI2Codec(args.raw_dim, args.hidden, args.k, n_sub=len(sub_ids), tau=args.head_tau).to(device)
        codec.WE = WE; args.n_sub = len(sub_ids)
        print(f"[train] ON-MANIFOLD head: subset N={len(sub_ids)} event-vocab tokens; codec params="
              f"{sum(p.numel() for p in codec.parameters()):,} (softmax over real embedding rows)", flush=True)
    else:
        codec = KAI2Codec(args.raw_dim, args.hidden, args.k).to(device); args.n_sub = 0
    opt = torch.optim.AdamW(codec.parameters(), lr=args.lr)
    # ── distill (t8: KL + manifold-anchor, held-out split, cos-diag pre-export gate) ──
    import random as _rnd
    _rnd.Random(1).shuffle(corpus)
    nval = max(2, int(len(corpus) * 0.2)); val = corpus[:nval]; train_set = corpus[nval:]
    print(f"[train] corpus={len(corpus)} train={len(train_set)} val={len(val)} (held-out generalization split)", flush=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)  # LR anneal (t3 lesson)
    lam0 = getattr(args, "anchor_lambda", 1.0); onm = codec.n_sub > 0
    vkl_curve = []; best = float("inf"); best_state = None; best_ep = -1
    for ep in range(args.epochs):
        lam = 0.0 if onm else max(0.1, lam0 * (1.0 - ep / max(1, args.epochs)))  # on-manifold = structural, no anchor
        codec.train(); tkl = tan = 0.0
        for it in train_set:
            t_log = decision_logits_text(model, embed, tok, it["text"], device)
            s_log, soft = decision_logits_inject(model, embed, tok, it["text"], codec, args.raw_dim, device, args.k)
            kl = distillation_loss(s_log, t_log, args.tau)
            if onm:
                loss = kl
            else:
                an = anchor_loss(soft, it["text"], embed, tok, device); loss = kl + lam * an; tan += an.item()
            opt.zero_grad(); loss.backward(); opt.step()
            tkl += kl.item()
        sched.step()
        codec.eval(); vkl = 0.0
        with torch.no_grad():
            for it in val:
                t_log = decision_logits_text(model, embed, tok, it["text"], device)
                s_log, _ = decision_logits_inject(model, embed, tok, it["text"], codec, args.raw_dim, device, args.k)
                vkl += distillation_loss(s_log, t_log, args.tau).item()
        vkl /= len(val); vkl_curve.append(vkl)
        if vkl < best:   # save-best by HELD-OUT KL (generalization, not train memorization)
            best = vkl; best_ep = ep
            best_state = {kk: vv.detach().cpu().clone() for kk, vv in codec.state_dict().items()}
        extra = f"head_tau={codec.tau:.2f}" if onm else f"lam={lam:.3f} train_anch={tan/max(1,len(train_set)):.4f}"
        print(f"[train] ep {ep} {extra} train_KL={tkl/len(train_set):.4f} val_KL={vkl:.4f} best_valKL={best:.4f}@{best_ep}", flush=True)
    print(f"[train] VAL_KL_curve={['%.4f' % x for x in vkl_curve]}", flush=True)
    print(f"[train] BEST val_KL={best:.4f} @epoch {best_ep} (held-out; saved this)", flush=True)
    if best_state is not None:
        codec.load_state_dict(best_state)
    # ── COS-DIAG PRE-EXPORT MANIFOLD GATE (abort if still noise) ──
    mx = manifold_maxcos(codec, embed, val, args.raw_dim, device)
    print(f"[manifold-gate] val mean max-cos to embed table = {mx:.4f} (random baseline ~0.07; need >> )", flush=True)
    if mx < 0.30:
        print(f"[manifold-gate] ABORT rc=5: codec still OFF-MANIFOLD ({mx:.4f}≈noise) — not exporting. "
              f"Raise --anchor_lambda or epochs.", flush=True)
        torch.save({"codec": codec.state_dict(), "args": vars(args), "best_valkl": best, "maxcos": mx,
                    "WE": (codec.WE.detach().cpu() if codec.n_sub > 0 else None), "n_sub": codec.n_sub}, args.out)
        sys.exit(5)
    torch.save({"codec": codec.state_dict(), "args": vars(args), "best_valkl": best, "best_ep": best_ep, "maxcos": mx,
                "WE": (codec.WE.detach().cpu() if codec.n_sub > 0 else None), "n_sub": codec.n_sub}, args.out)
    print(f"[train] saved {args.out}", flush=True)


# ───────────────────────── export k-vector packets for the C engine (gemma4_kv_inject) ───────────
def export_packets(args):
    """Write per-event .bin packets (k x hidden float32, row-major) the engine injects via
    gemma4_kv_inject. Header: magic 'KAI2', u32 k, u32 hidden, then k*hidden float32."""
    ckpt = torch.load(args.out, map_location="cpu")
    a = ckpt["args"]
    nsub = ckpt.get("n_sub", a.get("n_sub", 0))
    codec = KAI2Codec(a["raw_dim"], a["hidden"], a["k"], n_sub=nsub, tau=a.get("head_tau", 0.5))
    if nsub > 0:
        codec.WE = ckpt["WE"]
    codec.load_state_dict(ckpt["codec"]); codec.eval()
    os.makedirs(args.packets_dir, exist_ok=True)
    # Export for the HELD-OUT EVAL_EVENTS (never trained on) so the metal gate = generalization test.
    for i, it in enumerate(EVAL_EVENTS):
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
    ap.add_argument("--fakeq", action="store_true", help="Step 2: OK_Q4B fake-quant the frozen teacher weights before distill")
    ap.add_argument("--anchor_lambda", type=float, default=1.0, help="t8: weight on the manifold-anchor loss (λ-annealed)")
    ap.add_argument("--onmanifold", action="store_true", help="t9: softmax-over-real-embedding-rows head (structurally on-manifold)")
    ap.add_argument("--head_tau", type=float, default=0.5, help="t9: softmax temperature of the on-manifold head (lower=sharper toward discrete tokens)")
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
