#!/usr/bin/env python3
# live_transmit_demo.py — the first live Gemma->Qwen TELEPATHY transmit via the sidecar (loads Qwen once).
# Each src_tok_<i>.npy is a REAL gemma-3n-E2B per-token latent; the sidecar maps it (W_emb) into Qwen's
# embedding space, injects it as a soft prefix, and Qwen unfolds a delegate reply. No tokenizer hand-off.
import os, json, numpy as np
os.environ.setdefault("SP_QWEN_DEVICE", "cuda"); os.environ.setdefault("SP_QWEN_MAXNEW", "20")
import telepathy_sidecar as sc
man = json.load(open("src_tok_manifest.json"))
print("=== FAIL-CLOSED (no SP_TELEPATHY_LICENSE) ===")
os.environ.pop("SP_TELEPATHY_LICENSE", None)
r = sc.transmit(np.load("src_tok_0.npy"), stream=False)
print(f"  inert result = {r!r}  (empty => bridge refused, correct)\n")
print("=== LICENSED LIVE Gemma->Qwen TRANSMIT (SP_TELEPATHY_LICENSE=dev) ===")
os.environ["SP_TELEPATHY_LICENSE"] = "dev"
for e in man:
    out = sc.transmit(np.load(f"src_tok_{e['i']}.npy"), stream=False)
    print(f"  SRC[{e['i']}] '{e['text'][:46]}'\n     QWEN delegate -> {out!r}\n")
print("DONE")
