#!/usr/bin/env python3
# train_gen_adapter.py — TELE-10 v2 FIDELITY: a GENERATION-tuned bridge adapter.
# Objective shift: representation cosine (W_emb) -> causal generation CE, backprop'd END-TO-END through
# the FROZEN Qwen forward to the adapter only (prefix-tuning). No proxy near the embed layer (that is the
# cosine trap). Warm-started from W_emb so the proven +1.45-nat positional bandwidth (TELE-5) is kept,
# then fine-tuned. Loss = CE of Qwen reconstructing the SOURCE from the injected prefix (faithful transmit;
# no answer labels available). HF Qwen is the differentiable target (sp-Qwen O_K isn't a torch graph) —
# sp-Qwen calibration is a deploy-time check.
import os, numpy as np, torch, torch.nn as nn, transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
GEMMA="google/gemma-3n-E2B-it"; QWEN="Qwen/Qwen2.5-Coder-0.5B-Instruct"
dev="cuda" if torch.cuda.is_available() else "cpu"
texts=[l.rstrip("\n") for l in open("pairs.txt",encoding="utf-8") if l.strip()][:160]

# 1) gemma per-token latents for the corpus (cached)
cache="gemma_tok_cache.npz"
if os.path.exists(cache):
    GEM=list(np.load(cache,allow_pickle=True)["gem"]); print(f"[cache] {len(GEM)} gemma token seqs")
else:
    gt=AutoTokenizer.from_pretrained(GEMMA,trust_remote_code=True)
    gm=transformers.Gemma3nForCausalLM.from_pretrained(GEMMA,dtype=torch.bfloat16,trust_remote_code=True).to(dev).eval()
    GEM=[]
    with torch.no_grad():
        for i,t in enumerate(texts):
            enc=gt(t,return_tensors="pt",truncation=True,max_length=40).to(dev)
            hs=gm(**enc,output_hidden_states=True).hidden_states[-1]
            if hs.dim()==4: hs=hs[0]
            GEM.append(hs[0].float().cpu().numpy())
            if i%40==0: print(f"  gemma {i}/{len(texts)}",flush=True)
    del gm; torch.cuda.empty_cache(); np.savez(cache,gem=np.array(GEM,dtype=object)); print("[cache] saved")

# 2) frozen qwen + warm-started adapter
qtok=AutoTokenizer.from_pretrained(QWEN)
qm=AutoModelForCausalLM.from_pretrained(QWEN,dtype=torch.float32).to(dev).eval()
for p in qm.parameters(): p.requires_grad_(False)
emb=qm.get_input_embeddings()
ad=np.load("telepathy_adapter_g2q_emb.npz"); W=ad["W"]; gmu,gsd,emu,esd=ad["gmu"],ad["gsd"],ad["emu"],ad["esd"]
embnorm=float(ad["embnorm"]); scale0=float(ad["scale"]); Dg,De=W.shape
Wfold=(np.diag(1/gsd)@W@np.diag(esd)).astype(np.float32); bfold=(emu-(gmu/gsd)@W*esd).astype(np.float32)
lin=nn.Linear(Dg,De).to(dev)
with torch.no_grad(): lin.weight.copy_(torch.tensor(Wfold.T,device=dev)); lin.bias.copy_(torch.tensor(bfold,device=dev))
logscale=nn.Parameter(torch.tensor(float(np.log(scale0)),device=dev))
LR=float(os.environ.get("SP_GEN_LR","3e-5"))
opt=torch.optim.Adam(list(lin.parameters())+[logscale],lr=LR)
print(f"[cfg] lr={LR} (low: warm-start is already strong, avoid overshoot)")
bos=qtok.bos_token_id if qtok.bos_token_id is not None else qtok.eos_token_id

def prefix(gnp):
    p=lin(torch.tensor(gnp,dtype=torch.float32,device=dev))
    return p/(p.norm(dim=1,keepdim=True)+1e-6)*embnorm*torch.exp(logscale)
def ids_of(i): return qtok(texts[i],return_tensors="pt",truncation=True,max_length=40).input_ids.to(dev)
def ce(i):
    pref=prefix(GEM[i]); ids=ids_of(i); K=pref.shape[0]
    full=torch.cat([emb(torch.tensor([[bos]],device=dev)), pref.unsqueeze(0), emb(ids)],1)
    labels=torch.cat([torch.full((1,1+K),-100,device=dev), ids],1)
    return qm(inputs_embeds=full, labels=labels).loss

n=len(GEM); rng=np.random.RandomState(0); idx=rng.permutation(n); tr=idx[:int(n*0.85)]; te=idx[int(n*0.85):]
def te_ce():
    with torch.no_grad(): return float(np.mean([ce(int(i)).item() for i in te]))
base=te_ce(); print(f"[base] held-out CE (W_emb warm-start, untuned) = {base:.4f}")
for ep in range(6):
    rng.shuffle(tr); tot=0.0
    for i in tr:
        opt.zero_grad(); l=ce(int(i)); l.backward(); opt.step(); tot+=l.item()
    print(f"[ep {ep}] train CE={tot/len(tr):.4f}  heldout CE={te_ce():.4f}  scale={float(torch.exp(logscale)):.3f}",flush=True)
fin=te_ce(); print(f"[gen-tuned] held-out CE: {base:.4f} -> {fin:.4f}  (lower = better reconstruction = more faithful transmit)")

# 3) save the gen adapter (raw linear + norm) for the sidecar
np.savez("telepathy_adapter_g2q_gen.npz", Wt=lin.weight.detach().cpu().numpy(), b=lin.bias.detach().cpu().numpy(),
         embnorm=np.float32(embnorm), scale=np.float32(float(torch.exp(logscale))), src=GEMMA, dst=QWEN)
print("[save] gen adapter -> telepathy_adapter_g2q_gen.npz")

# 4) permutation gate (TELE-5) on the gen adapter: did we KEEP positional bandwidth?
def ll(i, mode, rs=0):
    with torch.no_grad():
        p=prefix(GEM[i])
        if mode=="shuf": p=p[torch.tensor(np.random.RandomState(rs).permutation(p.shape[0]),device=dev)]
        ids=ids_of(i); K=p.shape[0]
        full=torch.cat([emb(torch.tensor([[bos]],device=dev)), p.unsqueeze(0), emb(ids)],1)
        labels=torch.cat([torch.full((1,1+K),-100,device=dev), ids],1)
        return -qm(inputs_embeds=full,labels=labels).loss.item()
corr=np.mean([ll(int(i),"corr") for i in te]); shuf=np.mean([np.mean([ll(int(i),"shuf",r) for r in (1,2,3)]) for i in te])
print(f"[perm] ORDER_gain (corr-shuf) on gen adapter = {corr-shuf:+.4f}  (TELE-5 was +1.45; want still >0)")

# 5) a couple generations (repr-safe)
print("[gen] samples (gen-tuned prefix):")
for i in list(te)[:3]:
    with torch.no_grad():
        be=emb(torch.tensor([[bos]],device=dev)); pt=prefix(GEM[i]).unsqueeze(0)
        g=qm.generate(inputs_embeds=torch.cat([be,pt],1),max_new_tokens=16,do_sample=False)
    out=qtok.decode(g[0],skip_special_tokens=True).encode("ascii","replace").decode()
    print(f"  src='{texts[i][:40]}' -> {out[:60]!r}")
print("DONE")
