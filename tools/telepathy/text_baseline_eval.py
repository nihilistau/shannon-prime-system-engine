#!/usr/bin/env python3
# text_baseline_eval.py — the control I should have run: does Qwen answer these questions from TEXT
# (no bridge)? If yes, the TELE-11 failure is the LATENT CHANNEL, not the delegate's capability.
import json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
QWEN="Qwen/Qwen2.5-Coder-0.5B-Instruct"; dev="cuda" if torch.cuda.is_available() else "cpu"
TE=[json.loads(l) for l in open("answer_test.jsonl",encoding="utf-8") if l.strip()]
tok=AutoTokenizer.from_pretrained(QWEN); m=AutoModelForCausalLM.from_pretrained(QWEN,dtype=torch.float32).to(dev).eval()
import re
fair=0; arith_n=arith_ok=0; samples=[]
def cat(q): return "arith" if ("times" in q or "plus" in q) else "other"
with torch.no_grad():
    for r in TE:
        msgs=[{"role":"user","content":r["q"]}]
        enc=tok.apply_chat_template(msgs,add_generation_prompt=True,return_tensors="pt")
        ids=(enc if torch.is_tensor(enc) else enc["input_ids"]).to(dev)
        g=m.generate(ids,max_new_tokens=24,do_sample=False)
        out=tok.decode(g[0,ids.shape[1]:],skip_special_tokens=True).strip().lower(); gold=r["a"].lower()
        toks=set(re.findall(r"[a-z0-9]+",out))
        ok = gold in toks                          # FAIR: gold appears anywhere in the answer
        fair+=ok
        if cat(r["q"])=="arith": arith_n+=1; arith_ok+=ok
        if len(samples)<8: samples.append((r["q"],r["a"],out[:36]))
n=len(TE)
print(f"[text-baseline] N={n}  FAIR (gold appears in answer) = {fair/n:.3f}")
print(f"  arithmetic subset: {arith_ok}/{arith_n} = {arith_ok/max(1,arith_n):.3f}")
for q,a,o in samples: print(f"  Q='{q[:38]}' gold='{a}' -> {o!r}")
