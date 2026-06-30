#!/usr/bin/env python3
# hybrid_eval.py — TELE-12: the HYBRID proof (zero training). Split per the channel boundary:
#   LATENT carries the OPERATION (gist/intent — its strong suit); TEXT carries the bare OPERANDS (precise
#   payload — what the latent loses). Decisive iff hybrid >> both single-channel controls.
# 4 conditions, fair gold-in-answer metric, arithmetic subset (plus/minus/times):
#   1 text-full      : Qwen gets the full question as text                 (upper bound ~0.8)
#   2 latent-only    : latent prefix + "The answer is" scaffold            (TELE-11b ~0.08, operands lost)
#   3 operands-only  : TEXT "<a> <b>" with NO operation word              (ambiguous op -> low)
#   4 HYBRID         : latent prefix (op) + TEXT " <a> <b>. The answer is" (op from latent, operands from text)
import os, re, json, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
QWEN="Qwen/Qwen2.5-Coder-0.5B-Instruct"; dev="cuda" if torch.cuda.is_available() else "cpu"
TE=[json.loads(l) for l in open("answer_test.jsonl",encoding="utf-8") if l.strip()]
GEM=list(np.load("answer_gemma_cache.npz",allow_pickle=True)["gem"]); GTE=GEM[-len(TE):]
# arithmetic subset only (two operands + an op word)
rows=[];
for g,r in zip(GTE,TE):
    nums=re.findall(r"\d+", r["q"])
    if len(nums)==2 and any(op in r["q"] for op in("plus","minus","times")):
        rows.append((g,r,nums[0],nums[1]))
print(f"[hybrid] arithmetic test items: {len(rows)}")
tok=AutoTokenizer.from_pretrained(QWEN); m=AutoModelForCausalLM.from_pretrained(QWEN,dtype=torch.float32).to(dev).eval()
emb=m.get_input_embeddings(); ad=np.load("telepathy_adapter_g2q_ans.npz")
Wt,b=torch.tensor(ad["Wt"],device=dev),torch.tensor(ad["b"],device=dev)
m0w,m0b,m2w,m2b=[torch.tensor(ad[k],device=dev) for k in("m0w","m0b","m2w","m2b")]
embnorm,scale=float(ad["embnorm"]),float(ad["scale"]); bos=tok.bos_token_id or tok.eos_token_id
def prefix(g):
    h=torch.tensor(g,dtype=torch.float32,device=dev)@Wt.T+b
    p=h+(torch.nn.functional.gelu(h@m0w.T+m0b)@m2w.T+m2b)
    return p/(p.norm(dim=1,keepdim=True)+1e-6)*embnorm*scale
def emb_text(t): return emb(tok(t,add_special_tokens=False,return_tensors="pt").input_ids.to(dev))
def gen_embeds(seq):
    with torch.no_grad(): g=m.generate(inputs_embeds=seq,max_new_tokens=10,do_sample=False)
    return tok.decode(g[0],skip_special_tokens=True).strip().lower()
def gen_text(t):
    enc=tok.apply_chat_template([{"role":"user","content":t}],add_generation_prompt=True,return_tensors="pt")
    ids=(enc if torch.is_tensor(enc) else enc["input_ids"]).to(dev)
    with torch.no_grad(): g=m.generate(ids,max_new_tokens=12,do_sample=False)
    return tok.decode(g[0,ids.shape[1]:],skip_special_tokens=True).strip().lower()
def ok(out,gold): return gold.lower() in re.findall(r"[a-z0-9]+",out)
be=emb(torch.tensor([[bos]],device=dev))
c1=c2=c3=c4=0; samples=[]
for g,r,a,b2 in rows:
    gold=r["a"]
    o1=gen_text(r["q"])                                                   # text-full
    o2=gen_embeds(torch.cat([be,prefix(g).unsqueeze(0),emb_text(" The answer is")],1))   # latent-only
    o3=gen_text(f"{a} {b2}")                                              # operands-only (no op)
    o4=gen_embeds(torch.cat([be,prefix(g).unsqueeze(0),emb_text(f" {a} {b2}. The answer is")],1))  # HYBRID
    c1+=ok(o1,gold); c2+=ok(o2,gold); c3+=ok(o3,gold); c4+=ok(o4,gold)
    if len(samples)<6: samples.append((r["q"],gold,o4[:24]))
n=len(rows)
print(f"\n==== HYBRID proof (arith, N={n}, fair gold-in-answer) ====")
print(f"  1 text-full     = {c1/n:.3f}  (upper bound)")
print(f"  2 latent-only   = {c2/n:.3f}  (gist only; operands lost)")
print(f"  3 operands-only = {c3/n:.3f}  (text operands, NO op)")
print(f"  4 HYBRID        = {c4/n:.3f}  (latent op + text operands)")
print(f"  VERDICT: hybrid {'>>' if c4/n>max(c2/n,c3/n)+0.1 else '~'} controls "
      f"=> {'latent supplies the op, text the operands (PROVEN)' if c4/n>max(c2/n,c3/n)+0.1 else 'no clear hybrid gain'}")
for q,gd,o in samples: print(f"  Q='{q[:30]}' gold='{gd}' hybrid->{o!r}")
