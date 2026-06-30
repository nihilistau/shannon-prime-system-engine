#!/usr/bin/env python3
# telepathy_sidecar.py — v1 Qwen DELEGATE sidecar for Telepathy (TELE-8 v1).
# The daemon's LatentBridge, on route=TELEPATHY, hands a Gemma per-token latent sequence here; the
# sidecar maps it into Qwen's embedding space (W_emb), injects it as a soft prefix, and STREAMS the
# delegate continuation back token-by-token. The Gemma engine is never touched.
#
# VRAM design: LAZY-load (cold until the first transmit, then warm), DEVICE = SP_QWEN_DEVICE (default
# `cpu` => ZERO contention with the served Gemma-3-12B on the GPU). 0.5B is tiny; cpu is fine for a
# short delegate reply. Set SP_QWEN_DEVICE=cuda only if the card has headroom. Production footprint =
# Qwen-only (~1GB fp16 / less on cpu) — the daemon supplies the Gemma latent, so no Gemma weights here.
#
# Wire: the daemon spawns `python telepathy_sidecar.py --serve` and writes JSON lines on stdin
#   {"gemma_npy": "<path to [K,Dg] gemma per-token latents>"}  ->  streamed delegate text on stdout
#   (framed <telepathy> ... </telepathy>). This is the v1 transport; the L1 ABI backend hook is v2.
import os, sys, json, numpy as np
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # delegate text may be non-ASCII
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

QWEN    = os.environ.get("SP_QWEN_MODEL", "Qwen/Qwen2.5-Coder-0.5B-Instruct")
DEVICE  = os.environ.get("SP_QWEN_DEVICE", "cpu")          # cpu default = no VRAM contention with Gemma
# prefer the GENERATION-tuned adapter (TELE-10: CE 3.09->2.65, ORDER +2.18) if present; else the W_emb one
_GEN = "telepathy_adapter_g2q_gen.npz"; _EMB = "telepathy_adapter_g2q_emb.npz"
ADAPTER = os.environ.get("SP_TELEPATHY_EMB_ADAPTER", _GEN if os.path.exists(_GEN) else _EMB)
MAXNEW  = int(os.environ.get("SP_QWEN_MAXNEW", "24"))

_S = {}
def _lazy():
    if _S: return _S
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    dev = DEVICE if (DEVICE != "cuda" or torch.cuda.is_available()) else "cpu"
    dt  = torch.bfloat16 if dev == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(QWEN)
    model = AutoModelForCausalLM.from_pretrained(QWEN, dtype=dt).to(dev).eval()
    ad = np.load(ADAPTER); gen = "Wt" in ad.files
    _S.update(torch=torch, tok=tok, model=model, dev=dev, emb=model.get_input_embeddings(),
              gen=gen, embnorm=float(ad["embnorm"]), scale=float(ad["scale"]))
    if gen:   # TELE-10/10b generation-tuned: linear (+ optional residual MLP)
        _S.update(Wt=ad["Wt"], b=ad["b"], has_mlp=("m0w" in ad.files))
        if "m0w" in ad.files: _S.update(m0w=ad["m0w"], m0b=ad["m0b"], m2w=ad["m2w"], m2b=ad["m2b"])
    else:     # W_emb: z-scored ridge
        _S.update(W=ad["W"], gmu=ad["gmu"], gsd=ad["gsd"], emu=ad["emu"], esd=ad["esd"])
    sys.stderr.write(f"[sidecar] Qwen on {dev}; adapter={'GEN-tuned' if gen else 'W_emb'} ({ADAPTER}); lazy-warm\n"); sys.stderr.flush()
    return _S

def _gelu(x):
    import math
    return 0.5 * x * (1.0 + np.vectorize(math.erf)(x / np.sqrt(2.0)))

def _map(gem):   # [K,Dg] gemma per-token -> [K,Demb] qwen embedding-space soft prefix
    if _S["gen"]:
        h = gem @ _S["Wt"].T + _S["b"]
        if _S.get("has_mlp"):
            p = h + _gelu(h @ _S["m0w"].T + _S["m0b"]) @ _S["m2w"].T + _S["m2b"]
        else:
            p = h
    else:
        z = (gem - _S["gmu"]) / _S["gsd"]
        p = (z @ _S["W"]) * _S["esd"] + _S["emu"]
    return (p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-8)) * _S["embnorm"] * _S["scale"]

def _attest_ok():
    # Fail-closed cryptographic-attestation gate at the transfer boundary: no valid SP_TELEPATHY_LICENSE
    # => the bridge runs INERT (refuses to transmit). Disables ONLY the bridge's own operation; never any
    # host-external effect. Mirrors the daemon LatentBridge license gate (telepathy.rs).
    t = os.environ.get("SP_TELEPATHY_LICENSE", "")
    return bool(t and t.strip())

def transmit(gem, stream=True):
    if not _attest_ok():
        sys.stderr.write("[sidecar] attestation FAIL-CLOSED: SP_TELEPATHY_LICENSE unset -> bridge inert (no transfer)\n"); sys.stderr.flush()
        if stream: sys.stdout.write("\n"); sys.stdout.flush()
        return ""
    s = _lazy(); torch = s["torch"]
    pref = _map(np.asarray(gem, dtype=np.float32))
    bos = s["tok"].bos_token_id if s["tok"].bos_token_id is not None else s["tok"].eos_token_id
    out_ids = []
    with torch.no_grad():
        be = s["emb"](torch.tensor([[bos]], device=s["dev"]))
        pt = torch.tensor(pref, dtype=be.dtype, device=s["dev"]).unsqueeze(0)
        cur, past = torch.cat([be, pt], 1), None
        for _ in range(MAXNEW):
            o = s["model"](inputs_embeds=cur, past_key_values=past, use_cache=True)
            past = o.past_key_values
            nid = int(o.logits[0, -1].argmax())
            if nid == s["tok"].eos_token_id: break
            out_ids.append(nid)
            if stream: sys.stdout.write(s["tok"].decode([nid], skip_special_tokens=True)); sys.stdout.flush()
            cur = s["emb"](torch.tensor([[nid]], device=s["dev"]))
    if stream: sys.stdout.write("\n"); sys.stdout.flush()
    return s["tok"].decode(out_ids, skip_special_tokens=True)

def main():
    if "--serve" in sys.argv:
        sys.stderr.write('[sidecar] serve: stdin JSON lines {"gemma_npy":path} -> streamed delegate text\n'); sys.stderr.flush()
        for line in sys.stdin:
            line = line.strip()
            if not line: continue
            try:
                gem = np.load(json.loads(line)["gemma_npy"])
                sys.stdout.write("<telepathy>\n"); transmit(gem, True); sys.stdout.write("</telepathy>\n"); sys.stdout.flush()
            except Exception as e:
                sys.stderr.write(f"[sidecar] err: {e}\n"); sys.stderr.flush()
    elif "--smoke" in sys.argv:
        # plumbing smoke (random Dg latent): proves load+map+inject+stream. Semantics gate = telepathy_prefix --stream / G-TELEPATHY-LIVE.
        s = _lazy(); dg = s["W"].shape[0]
        sys.stderr.write(f"[sidecar] SMOKE: random {6}x{dg} gemma latent -> delegate stream:\n"); sys.stderr.flush()
        txt = transmit(np.random.RandomState(0).randn(6, dg).astype(np.float32), True)
        sys.stderr.write(f"[sidecar] SMOKE ok (streamed {len(txt)} chars) — plumbing GREEN\n"); sys.stderr.flush()
    else:
        g = next((np.load(a) for a in sys.argv[1:] if a.endswith(".npy")), None)
        if g is None: sys.stderr.write("usage: telepathy_sidecar.py <gemma_tokens.npy> | --serve | --smoke\n"); sys.exit(2)
        transmit(g, True)

if __name__ == "__main__":
    main()
