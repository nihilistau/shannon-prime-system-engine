#!/usr/bin/env python3
# strict_eval.py — honest re-eval of the answer adapter: STRICT exact match (first token == gold, exactly),
# to expose how much of the lenient 0.321 was degenerate-repetition gaming the startswith() gate.
import os, json, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
QWEN="Qwen/Qwen2.5-Coder-0.5B-Instruct"; dev="cuda" if torch.cuda.is_available() else "cpu"
TE=[json.loads(l) for l in open("answer_test.jsonl",encoding="utf-8") if l.strip()]
GEM=list(np.load("answer_gemma_cache.npz",allow_pickle=True)["gem"]); GTE=GEM[-len(TE):]
qtok=AutoTokenizer.from_pretrained(QWEN); qm=AutoModelForCausalLM.from_pretrained(QWEN,dtype=torch.float32).to(dev).eval()
emb=qm.get_input_embeddings(); ad=np.load("telepathy_adapter_g2q_ans.npz")
Wt,b=torch.tensor(ad["Wt"],device=dev),torch.tensor(ad["b"],device=dev)
m0w,m0b,m2w,m2b=[torch.tensor(ad[k],device=dev) for k in("m0w","m0b","m2w","m2b")]
embnorm,scale=float(ad["embnorm"]),float(ad["scale"]); bos=qtok.bos_token_id or qtok.eos_token_id
def prefix(g):
    h=torch.tensor(g,dtype=torch.float32,device=dev)@Wt.T+b
    p=h+(torch.nn.functional.gelu(h@m0w.T+m0b)@m2w.T+m2b)
    return p/(p.norm(dim=1,keepdim=True)+1e-6)*embnorm*scale
def degenerate(s):  # all same char, or len-1 unique
    t="".join(s.split()); return len(set(t))<=1 and len(t)>1
lenient=strict=degen=0
with torch.no_grad():
    for g,r in zip(GTE,TE):
        be=emb(torch.tensor([[bos]],device=dev)); pt=prefix(g).unsqueeze(0)
        gen=qm.generate(inputs_embeds=torch.cat([be,pt],1),max_new_tokens=6,do_sample=False)
        out=qtok.decode(gen[0],skip_special_tokens=True).strip().lower(); gold=r["a"].lower()
        toks=out.split()
        if gold in toks or out.startswith(gold): lenient+=1
        if toks and toks[0]==gold: strict+=1
        if degenerate(out): degen+=1
n=len(TE)
print(f"[strict-eval] N={n}")
print(f"  LENIENT (startswith/in-split) = {lenient/n:.3f}  (the gamed number)")
print(f"  STRICT  (first token == gold) = {strict/n:.3f}  (the honest number)")
print(f"  DEGENERATE outputs (repetition/garbage) = {degen/n:.3f}")
