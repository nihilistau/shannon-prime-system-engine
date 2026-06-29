#!/usr/bin/env python3
# telepathy_prefix.py — the READABLE-PREFIX gate. Map a gemma per-token sequence into Qwen's INPUT-
# EMBEDDING space (W_emb, char-span aligned) and inject it as a soft-token prefix Qwen attends to from
# position 0. Then PROVE Qwen reads POSITIONAL STRUCTURE (not aggregate mass) via the permutation gap:
#   teacher-forced reconstruction LL of the source text under prefixes =
#     none  -> mean-replicated (mass) -> shuffled (multiset/bag) -> correct order
#   mass_gain   = LL(mean)-LL(none)      (pure aggregate mass; permutation-invariant)
#   multiset    = LL(shuf)-LL(mean)      (right vectors, wrong order)
#   ORDER_gain  = LL(corr)-LL(shuf)      (positional structure, isolated)  <-- the proof
# ORDER_gain>0 cannot come from aggregate bias (mass is permutation-invariant).
import numpy as np, torch, transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
GEMMA="google/gemma-3n-E2B-it"; QWEN="Qwen/Qwen2.5-Coder-0.5B-Instruct"
dev="cuda" if torch.cuda.is_available() else "cpu"

def gemma_tokens(texts):
    tok=AutoTokenizer.from_pretrained(GEMMA,trust_remote_code=True)
    m=transformers.Gemma3nForCausalLM.from_pretrained(GEMMA,dtype=torch.bfloat16,trust_remote_code=True).to(dev).eval()
    out=[]
    with torch.no_grad():
        for t in texts:
            enc=tok(t,return_tensors="pt",return_offsets_mapping=True,truncation=True,max_length=48)
            off=enc.pop("offset_mapping")[0].tolist(); enc={k:v.to(dev) for k,v in enc.items()}
            hs=m(**enc,output_hidden_states=True).hidden_states[-1]
            if hs.dim()==4: hs=hs[0]
            out.append((hs[0].float().cpu().numpy(),off))
    del m; torch.cuda.empty_cache(); return out

def align(gem,qwn_off,qwn_emb):
    G,E,tid=[],[],[]
    for ti,((gv,go),qo,qe) in enumerate(zip(gem,qwn_off,qwn_emb)):
        for j,(qa,qb) in enumerate(qo):
            if qb<=qa: continue
            ov=[gv[i] for i,(ga,gb) in enumerate(go) if gb>qa and ga<qb and gb>ga]
            if not ov: continue
            G.append(np.mean(ov,axis=0)); E.append(qe[j]); tid.append(ti)
    return np.array(G),np.array(E),np.array(tid)

def main():
    texts=[l.rstrip("\n") for l in open("pairs.txt",encoding="utf-8") if l.strip()][:160]
    tokq=AutoTokenizer.from_pretrained(QWEN)
    modelq=AutoModelForCausalLM.from_pretrained(QWEN,dtype=torch.bfloat16).to(dev).eval()
    emb_layer=modelq.get_input_embeddings()
    # qwen per-token input embeddings + offsets
    qoff=[]; qemb=[]; qids=[]
    with torch.no_grad():
        for t in texts:
            enc=tokq(t,return_tensors="pt",return_offsets_mapping=True,truncation=True,max_length=48)
            qoff.append(enc["offset_mapping"][0].tolist())
            ids=enc["input_ids"].to(dev); qids.append(ids[0])
            qemb.append(emb_layer(ids)[0].float().cpu().numpy())
    print("[prefix] gemma per-token extraction...")
    gem=gemma_tokens(texts)
    G,E,tid=align(gem,qoff,qemb)
    print(f"[align] {len(G)} token pairs; Dg={G.shape[1]} Demb={E.shape[1]}")
    ntext=len(texts); rng=np.random.RandomState(0); perm=rng.permutation(ntext)
    te_txt=set(perm[int(ntext*0.8):].tolist())
    tr=np.array([i for i in range(len(G)) if tid[i] not in te_txt])
    gmu,gsd=G[tr].mean(0),G[tr].std(0)+1e-6; emu,esd=E[tr].mean(0),E[tr].std(0)+1e-6
    Wemb=np.linalg.solve(((G[tr]-gmu)/gsd).T@((G[tr]-gmu)/gsd)+100*np.eye(G.shape[1]), ((G[tr]-gmu)/gsd).T@((E[tr]-emu)/esd))
    def mapg(gv):  # gemma hidden rows -> qwen embedding-space rows
        return (((gv-gmu)/gsd)@Wemb)*esd+emu

    # emb norm for scaling soft tokens to embedding scale
    embnorm=np.linalg.norm(E[tr],axis=1).mean()
    def run_ll(text_ti, mode, scale, rs):
        # build prefix soft tokens (qwen-embedding space) from this text's gemma sequence
        gv,go=gem[text_ti]; pref=mapg(gv)                       # [K, Demb]
        if mode=="none": pref=np.zeros((0,pref.shape[1]),np.float32)
        elif mode=="mean": pref=np.repeat(pref.mean(0,keepdims=True),len(pref),axis=0)
        elif mode=="shuf": pref=pref[np.random.RandomState(rs).permutation(len(pref))]
        elif mode=="rev":  pref=pref[::-1]
        # normalize prefix rows to embedding scale * scale
        if len(pref):
            pref=pref/(np.linalg.norm(pref,axis=1,keepdims=True)+1e-8)*embnorm*scale
        ids=qids[text_ti].unsqueeze(0)                          # [1,T] the source text
        with torch.no_grad():
            xemb=emb_layer(ids)                                 # [1,T,D]
            pt=torch.tensor(pref,dtype=xemb.dtype,device=dev).unsqueeze(0) if len(pref) else None
            full = torch.cat([pt,xemb],1) if pt is not None else xemb
            K=pt.shape[1] if pt is not None else 0
            labels=torch.cat([torch.full((1,K),-100,device=dev), ids],1)
            o=modelq(inputs_embeds=full, labels=labels)
        return -o.loss.item()

    te=sorted(te_txt)
    def avg(mode,scale,rs=0):
        return float(np.mean([run_ll(ti,mode,scale,rs) for ti in te]))
    # small scale pick by ORDER gain on first few
    calib=te[:len(te)//2]
    best=(1.0,-1e9)
    for s in [0.5,1.0,2.0]:
        c=np.mean([run_ll(ti,"corr",s,0) for ti in calib]); sh=np.mean([run_ll(ti,"shuf",s,1) for ti in calib])
        print(f"  scale={s}: calib ORDER_gain={c-sh:+.4f}")
        if c-sh>best[1]: best=(s,c-sh)
    s=best[0]; print(f"[scale] {s}")

    fin=te[len(te)//2:]
    def favg(mode,rs=0): return float(np.mean([run_ll(ti,mode,s,rs) for ti in fin]))
    ll_none=favg("none"); ll_mean=favg("mean"); ll_rev=favg("rev")
    ll_shuf=float(np.mean([np.mean([run_ll(ti,"shuf",s,r) for r in (1,2,3)]) for ti in fin]))
    ll_corr=favg("corr")
    # per-text order win-rate
    wins=sum(1 for ti in fin if run_ll(ti,"corr",s,0) > np.mean([run_ll(ti,"shuf",s,r) for r in (1,2,3)]))
    print(f"\n==== READABLE-PREFIX gate (held-out, scale={s}) ====")
    print(f"LL none={ll_none:+.3f}  mean={ll_mean:+.3f}  shuf={ll_shuf:+.3f}  rev={ll_rev:+.3f}  corr={ll_corr:+.3f}")
    print(f"mass_gain   (mean-none) = {ll_mean-ll_none:+.4f}  (aggregate mass; permutation-invariant)")
    print(f"multiset    (shuf-mean) = {ll_shuf-ll_mean:+.4f}  (right vectors, wrong order)")
    print(f"ORDER_gain  (corr-shuf) = {ll_corr-ll_shuf:+.4f}  <-- positional structure, isolated")
    print(f"per-text corr>shuf win-rate = {wins/len(fin):.3f}  (chance 0.5)")
    print(f"VERDICT positional-attention={'GREEN' if (ll_corr-ll_shuf)>0 and wins/len(fin)>0.7 else 'AMBER' if (ll_corr-ll_shuf)>0 else 'RED'}")

if __name__=="__main__":
    main()
