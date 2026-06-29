#!/usr/bin/env python3
# telepathy_multivec.py — does the bridge survive PER-POSITION (multi-vector), and does the pooled-fit
# adapter hold per-token? Builds per-token paired latents via CHARACTER-SPAN alignment (gemma & qwen
# tokenize the same string differently -> align by char offsets, not by index), then compares:
#   (A) pooled-fit W (telepathy_adapter_g2q.npz) applied PER-TOKEN     [the user's question]
#   (B) a refit per-token ridge W_tok on span-aligned pairs
# Metrics on held-out texts: mean per-token cos to the true qwen token hidden + per-token retrieval@1
# (within-sentence: can the mapped vector pick its own position out of the sentence -> real bandwidth).
import sys, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers

GEMMA="google/gemma-3n-E2B-it"; QWEN="Qwen/Qwen2.5-Coder-0.5B-Instruct"

def load(repo,dtype,dev):
    tok=AutoTokenizer.from_pretrained(repo,trust_remote_code=True)
    if "gemma-3n" in repo.lower():
        m=transformers.Gemma3nForCausalLM.from_pretrained(repo,dtype=dtype,trust_remote_code=True)
    else:
        m=AutoModelForCausalLM.from_pretrained(repo,dtype=dtype,trust_remote_code=True)
    return tok,m.to(dev).eval()

def per_token(repo,texts,dev):
    """return list per text of (vecs[T,D] float32, offsets[(a,b)])"""
    tok,model=load(repo,torch.bfloat16 if dev=='cuda' else torch.float32,dev)
    out=[]
    with torch.no_grad():
        for t in texts:
            enc=tok(t,return_tensors="pt",return_offsets_mapping=True,truncation=True,max_length=64)
            off=enc.pop("offset_mapping")[0].tolist()
            enc={k:v.to(dev) for k,v in enc.items()}
            o=model(**enc,output_hidden_states=True)
            hs=o.hidden_states[-1]
            if hs.dim()==4: hs=hs[0]                       # gemma-3n altup -> primary stream
            v=hs[0].float().cpu().numpy()                  # [T,D]
            out.append((v,off))
    del model; torch.cuda.empty_cache()
    return out

def align(gem, qwn):
    """char-span align: for each qwen token, average gemma tokens whose span overlaps. -> G_aln,Q paired."""
    G,Q,txtid=[],[],[]
    for ti,((gv,go),(qv,qo)) in enumerate(zip(gem,qwn)):
        for j,(qa,qb) in enumerate(qo):
            if qb<=qa: continue                            # special/empty token
            ov=[gv[i] for i,(ga,gb) in enumerate(go) if gb>qa and ga<qb and gb>ga]
            if not ov: continue
            G.append(np.mean(ov,axis=0)); Q.append(qv[j]); txtid.append(ti)
    return np.array(G), np.array(Q), np.array(txtid)

def zfit(X): return X.mean(0), X.std(0)+1e-6
def ridge(X,Y,lam): return np.linalg.solve(X.T@X+lam*np.eye(X.shape[1]), X.T@Y)
def cosm(a,b):
    a=a/(np.linalg.norm(a,axis=1,keepdims=True)+1e-8); b=b/(np.linalg.norm(b,axis=1,keepdims=True)+1e-8)
    return (a*b).sum(1)

def main():
    dev="cuda" if torch.cuda.is_available() else "cpu"
    texts=[l.rstrip("\n") for l in open("pairs.txt",encoding="utf-8") if l.strip()][:220]
    print(f"[multivec] {len(texts)} texts, per-token extraction...")
    gem=per_token(GEMMA,texts,dev); print("  gemma per-token done")
    qwn=per_token(QWEN,texts,dev);  print("  qwen per-token done")
    G,Q,tid=align(gem,qwn)
    print(f"[align] {len(G)} span-aligned token pairs ({len(G)/len(texts):.1f}/text), Dg={G.shape[1]} Dq={Q.shape[1]}")
    ntext=len(texts); rng=np.random.RandomState(0); perm=rng.permutation(ntext); te_txt=set(perm[int(ntext*0.8):].tolist())
    tr=np.array([i for i in range(len(G)) if tid[i] not in te_txt]); te=np.array([i for i in range(len(G)) if tid[i] in te_txt])

    # (B) refit per-token ridge on span-aligned pairs
    gmu,gsd=zfit(G[tr]); qmu,qsd=zfit(Q[tr])
    Wtok=ridge((G[tr]-gmu)/gsd,(Q[tr]-qmu)/qsd,100.0)
    predB=(((G[te]-gmu)/gsd)@Wtok)*qsd+qmu
    cosB=cosm(predB,Q[te]).mean()

    # (A) pooled-fit adapter applied per-token
    ad=np.load("telepathy_adapter_g2q.npz")
    predA=(((G[te]-ad["gmu"])/ad["gsd"])@ad["W_fwd"])*ad["qsd"]+ad["qmu"]
    cosA=cosm(predA,Q[te]).mean()

    # bandwidth: per-token retrieval@1 WITHIN each held-out sentence (mapped vec picks its own position)
    def within_retr(pred):
        hit=tot=0
        for tindex in te_txt:
            idx=[k for k in te if tid[k]==tindex]
            if len(idx)<3: continue
            P=pred[[list(te).index(k) for k in idx]]; T=Q[idx]
            Pn=P/(np.linalg.norm(P,axis=1,keepdims=True)+1e-8); Tn=T/(np.linalg.norm(T,axis=1,keepdims=True)+1e-8)
            order=np.argsort(-(Pn@Tn.T),axis=1)
            hit+=sum(m in order[m,:1] for m in range(len(idx))); tot+=len(idx)
        return hit/max(1,tot)
    rA=within_retr(predA); rB=within_retr(predB)

    print(f"\n==== PER-POSITION (multi-vector) adapter test (held-out tokens) ====")
    print(f"(A) POOLED-fit W applied per-token : cos={cosA:.3f}  within-sentence retr@1={rA:.3f}")
    print(f"(B) REFIT  per-token W_tok         : cos={cosB:.3f}  within-sentence retr@1={rB:.3f}")
    print(f"sentence-level baseline (TELE-1 pooled-to-pooled) was cos~0.876 for reference")
    print(f"VERDICT pooled-holds-per-token={'YES' if cosA>0.7 else 'NO (refit needed)'} | "
          f"refit-lift=+{cosB-cosA:.3f} cos, +{rB-rA:.3f} retr@1")
    np.savez("telepathy_adapter_g2q_tok.npz",W_fwd=Wtok.astype(np.float32),
             gmu=gmu.astype(np.float32),gsd=gsd.astype(np.float32),qmu=qmu.astype(np.float32),qsd=qsd.astype(np.float32),
             src=GEMMA,dst=QWEN,kind="per_token_span_aligned")
    print("[save] per-token adapter -> telepathy_adapter_g2q_tok.npz")

if __name__=="__main__":
    main()
