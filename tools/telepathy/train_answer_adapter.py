#!/usr/bin/env python3
# train_answer_adapter.py — TELE-11 (c): answer-labeled fidelity. Input = Gemma latent of the QUESTION,
# target = the computed ANSWER. CE backprop through FROZEN Qwen to the adapter (linear+residual-MLP,
# warm-started from the TELE-10b gen adapter). Gate = EXACT-MATCH task-correctness (generate from the
# prefix alone, check the answer appears) -- the honest "answering vs echoing" test.
import os, json, numpy as np, torch, torch.nn as nn, transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
GEMMA="google/gemma-3n-E2B-it"; QWEN="Qwen/Qwen2.5-Coder-0.5B-Instruct"; dev="cuda" if torch.cuda.is_available() else "cpu"
def load_jsonl(p): return [json.loads(l) for l in open(p,encoding="utf-8") if l.strip()]
TR=load_jsonl("answer_train.jsonl"); TE=load_jsonl("answer_test.jsonl")
print(f"[data] train={len(TR)} test={len(TE)}")

# gemma per-token latents for all QUESTIONS (cache)
cache="answer_gemma_cache.npz"
qs=[r["q"] for r in TR]+[r["q"] for r in TE]
if os.path.exists(cache):
    GEM=list(np.load(cache,allow_pickle=True)["gem"])
else:
    gt=AutoTokenizer.from_pretrained(GEMMA,trust_remote_code=True)
    gm=transformers.Gemma3nForCausalLM.from_pretrained(GEMMA,dtype=torch.bfloat16,trust_remote_code=True).to(dev).eval()
    GEM=[]
    with torch.no_grad():
        for i,q in enumerate(qs):
            enc=gt(q,return_tensors="pt",truncation=True,max_length=40).to(dev)
            hs=gm(**enc,output_hidden_states=True).hidden_states[-1]
            if hs.dim()==4: hs=hs[0]
            GEM.append(hs[0].float().cpu().numpy())
            if i%60==0: print(f"  gemma {i}/{len(qs)}",flush=True)
    del gm; torch.cuda.empty_cache(); np.savez(cache,gem=np.array(GEM,dtype=object)); print("[cache] saved")
ntr=len(TR); GEM_TR=GEM[:ntr]; GEM_TE=GEM[ntr:]

# frozen qwen + warm-start adapter from the TELE-10b gen adapter
qtok=AutoTokenizer.from_pretrained(QWEN); qm=AutoModelForCausalLM.from_pretrained(QWEN,dtype=torch.float32).to(dev).eval()
for p in qm.parameters(): p.requires_grad_(False)
emb=qm.get_input_embeddings(); ad=np.load("telepathy_adapter_g2q_gen.npz")
Dg,De=ad["Wt"].shape[1],ad["Wt"].shape[0]
lin=nn.Linear(Dg,De).to(dev); mlp=nn.Sequential(nn.Linear(De,256),nn.GELU(),nn.Linear(256,De)).to(dev)
with torch.no_grad():
    lin.weight.copy_(torch.tensor(ad["Wt"],device=dev)); lin.bias.copy_(torch.tensor(ad["b"],device=dev))
    mlp[0].weight.copy_(torch.tensor(ad["m0w"],device=dev)); mlp[0].bias.copy_(torch.tensor(ad["m0b"],device=dev))
    mlp[2].weight.copy_(torch.tensor(ad["m2w"],device=dev)); mlp[2].bias.copy_(torch.tensor(ad["m2b"],device=dev))
embnorm=float(ad["embnorm"]); logscale=nn.Parameter(torch.tensor(float(np.log(float(ad["scale"]))),device=dev))
opt=torch.optim.Adam(list(lin.parameters())+list(mlp.parameters())+[logscale],lr=float(os.environ.get("SP_GEN_LR","3e-5")))
import re
bos=qtok.bos_token_id if qtok.bos_token_id is not None else qtok.eos_token_id
SCAFFOLD=" The answer is"                 # text-grounding anchor: separates 'read the latent' from 'generate'
scaf_ids=qtok(SCAFFOLD,add_special_tokens=False,return_tensors="pt").input_ids.to(dev)
def prefix(g):
    h=lin(torch.tensor(g,dtype=torch.float32,device=dev)); p=h+mlp(h)
    return p/(p.norm(dim=1,keepdim=True)+1e-6)*embnorm*torch.exp(logscale)
def ctx(g):                               # [1,1+K+S,De] = bos + latent prefix + scaffold text
    parts=torch.cat([emb(torch.tensor([[bos]],device=dev)), prefix(g).unsqueeze(0), emb(scaf_ids)],1)
    return parts, parts.shape[1]
def ce(g,a):
    c,clen=ctx(g); aid=qtok(" "+a,add_special_tokens=False,return_tensors="pt").input_ids.to(dev)
    full=torch.cat([c, emb(aid)],1)
    lab=torch.cat([torch.full((1,clen),-100,device=dev), aid],1)
    return qm(inputs_embeds=full,labels=lab).loss
def exact_match(GE,rows):
    hit=0
    with torch.no_grad():
        for g,r in zip(GE,rows):
            c,_=ctx(g)
            gen=qm.generate(inputs_embeds=c,max_new_tokens=10,do_sample=False)
            out=qtok.decode(gen[0],skip_special_tokens=True).strip().lower()
            if r["a"].lower() in re.findall(r"[a-z0-9]+",out): hit+=1   # FAIR: gold appears in the answer
    return hit/len(rows)
base_em=exact_match(GEM_TE,TE); print(f"[base] warm-start exact-match (test) = {base_em:.3f}")
rng=np.random.RandomState(0); order=list(range(ntr))
for ep in range(8):
    rng.shuffle(order); tot=0.0
    for i in order:
        opt.zero_grad(); l=ce(GEM_TR[i],TR[i]["a"]); l.backward(); opt.step(); tot+=l.item()
    em=exact_match(GEM_TE,TE)
    print(f"[ep {ep}] train CE={tot/ntr:.4f}  test exact-match={em:.3f}",flush=True)
fin=exact_match(GEM_TE,TE)
print(f"[answer-tuned] test EXACT-MATCH: {base_em:.3f} -> {fin:.3f}  (task-correctness; answering not echoing)")
np.savez("telepathy_adapter_g2q_ans.npz", Wt=lin.weight.detach().cpu().numpy(), b=lin.bias.detach().cpu().numpy(),
         m0w=mlp[0].weight.detach().cpu().numpy(), m0b=mlp[0].bias.detach().cpu().numpy(),
         m2w=mlp[2].weight.detach().cpu().numpy(), m2b=mlp[2].bias.detach().cpu().numpy(),
         embnorm=np.float32(embnorm), scale=np.float32(float(torch.exp(logscale))), src=GEMMA, dst=QWEN)
print("[save] answer adapter -> telepathy_adapter_g2q_ans.npz")
print("[samples] test Q -> latent prefix + 'The answer is' scaffold -> Qwen:")
for g,r in list(zip(GEM_TE,TE))[:8]:
    with torch.no_grad():
        c,_=ctx(g); gen=qm.generate(inputs_embeds=c,max_new_tokens=10,do_sample=False)
    out=qtok.decode(gen[0],skip_special_tokens=True).strip().replace("\n"," ").encode("ascii","replace").decode()
    ok = r["a"].lower() in re.findall(r"[a-z0-9]+", out.lower())
    print(f"  Q='{r['q'][:34]}' gold='{r['a']}' -> {out[:30]!r} {'OK' if ok else ''}")
print("DONE")
