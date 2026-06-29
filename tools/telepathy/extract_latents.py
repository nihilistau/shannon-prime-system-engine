#!/usr/bin/env python3
# extract_latents.py <repo_id> <text_file> <out.npy> [--layer L] [--pool mean|last]
# Mean-pool a chosen hidden layer over tokens -> one sentence latent per line. Works for plain causal
# LMs (qwen2) and multimodal wrappers (gemma3n) by feeding text-only input_ids with output_hidden_states.
import sys, argparse, numpy as np, torch

def load(repo, dtype):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    model = None
    import transformers
    from transformers import AutoModelForCausalLM
    def _mk(cls, **kw):
        try:    return cls.from_pretrained(repo, dtype=dtype, trust_remote_code=True, **kw)
        except TypeError: return cls.from_pretrained(repo, torch_dtype=dtype, trust_remote_code=True, **kw)
    # text-only class first for multimodal checkpoints (avoids vision/timm/torchvision tower)
    loaders = []
    if "gemma-3n" in repo.lower(): loaders.append(("gemma3n-text", getattr(transformers, "Gemma3nForCausalLM", None)))
    loaders += [("causal", AutoModelForCausalLM), ("auto", getattr(transformers, "AutoModel", None))]
    for name, cls in loaders:
        if cls is None: continue
        try:
            model = _mk(cls)
            print(f"[load] {repo} via {name} ({type(model).__name__})"); break
        except Exception as e:
            print(f"[load] {name} failed: {repr(e)[:160]}")
    if model is None: raise RuntimeError("no loader worked")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    return tok, model.eval()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo"); ap.add_argument("text_file"); ap.add_argument("out")
    ap.add_argument("--layer", type=int, default=-1)   # -1 = last hidden layer
    ap.add_argument("--pool", default="mean")
    ap.add_argument("--bs", type=int, default=8)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    texts = [l.rstrip("\n") for l in open(a.text_file, encoding="utf-8") if l.strip()]
    tok, model = load(a.repo, dtype); model.to(dev)
    vecs = []
    with torch.no_grad():
        for i in range(0, len(texts), a.bs):
            batch = texts[i:i+a.bs]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=64).to(dev)
            out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, output_hidden_states=True)
            hs = out.hidden_states[a.layer]                       # [B,T,D] (or [n_altup,B,T,D] for gemma-3n)
            if hs.dim() == 4: hs = hs[0]                           # gemma-3n AltUp: take the primary stream
            m = enc.attention_mask.unsqueeze(-1).to(hs.dtype)     # [B,T,1]
            if a.pool == "mean":
                v = (hs * m).sum(1) / m.sum(1).clamp(min=1)
            else:  # last non-pad token
                idx = enc.attention_mask.sum(1) - 1
                v = hs[torch.arange(hs.size(0)), idx]
            vecs.append(v.float().cpu().numpy())
            if i % 64 == 0: print(f"  {i}/{len(texts)}", flush=True)
    arr = np.concatenate(vecs, 0)
    np.save(a.out, arr)
    print(f"[done] {a.repo} -> {a.out} shape={arr.shape}")

if __name__ == "__main__":
    main()
