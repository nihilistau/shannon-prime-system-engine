#!/usr/bin/env python3
# save_src_tokens.py — encode a few SOURCE texts with gemma-3n-E2B -> per-token last-hidden (altup
# stream 0) -> save each as src_tok_<i>.npy [K,Dg]. These are the real Gemma latents the daemon would
# hand the sidecar on a TELEPATHY route; here we pre-encode a handful for the live-transmit gate.
import numpy as np, torch, transformers, json
from transformers import AutoTokenizer
GEMMA="google/gemma-3n-E2B-it"; dev="cuda" if torch.cuda.is_available() else "cpu"
SRC=["count the number of r letters in the word strawberry",
     "write a python function that reverses a linked list",
     "what is the time complexity of binary search",
     "explain how a hash map handles collisions"]
tok=AutoTokenizer.from_pretrained(GEMMA,trust_remote_code=True)
m=transformers.Gemma3nForCausalLM.from_pretrained(GEMMA,dtype=torch.bfloat16,trust_remote_code=True).to(dev).eval()
man=[]
with torch.no_grad():
    for i,t in enumerate(SRC):
        enc=tok(t,return_tensors="pt",truncation=True,max_length=48).to(dev)
        hs=m(**enc,output_hidden_states=True).hidden_states[-1]
        if hs.dim()==4: hs=hs[0]
        v=hs[0].float().cpu().numpy()
        np.save(f"src_tok_{i}.npy",v); man.append({"i":i,"text":t,"shape":list(v.shape)})
        print(f"saved src_tok_{i}.npy {v.shape}  '{t[:40]}'",flush=True)
open("src_tok_manifest.json","w").write(json.dumps(man))
print("DONE")
