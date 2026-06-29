#!/usr/bin/env python3
# fit_adapter.py — calibrate + GATE the Telepathy gemma<->qwen adapter.
#   inputs: gemma.npy [N,Dg], qwen.npy [N,Dq] (paired, same row = same text), foreign_gemma.npy [M,Dg]
#   linear adapter = z-scored ridge (closed form). Optional --mlp shallow fallback.
# Gates reported:
#   ROUNDTRIP  : cosine(g -> q -> g , g) on held-out  (cycle consistency)
#   PARITY/sem : retrieval@1/@5 of the true qwen target after mapping g->q (semantic identity preserved)
#   REJECT     : separation between in-domain mapped-distance and FOREIGN mapped-distance (AUC)
import sys, argparse, numpy as np

def zscore_fit(X): mu = X.mean(0); sd = X.std(0)+1e-6; return mu, sd
def zscore(X, mu, sd): return (X-mu)/sd
def ridge(X, Y, lam):  # X[n,di] Y[n,do] -> W[di,do]
    di = X.shape[1]; A = X.T@X + lam*np.eye(di); return np.linalg.solve(A, X.T@Y)
def cos(a, b):
    a=a/ (np.linalg.norm(a,axis=-1,keepdims=True)+1e-8); b=b/(np.linalg.norm(b,axis=-1,keepdims=True)+1e-8)
    return (a*b).sum(-1)
def retrieval_at_k(pred, targ, ks=(1,5)):
    # for each pred row, rank all targ rows by cosine; is the matching index in top-k
    P=pred/(np.linalg.norm(pred,axis=1,keepdims=True)+1e-8); T=targ/(np.linalg.norm(targ,axis=1,keepdims=True)+1e-8)
    S=P@T.T; order=np.argsort(-S,axis=1); res={}
    for k in ks:
        hit=sum(i in order[i,:k] for i in range(len(pred))); res[k]=hit/len(pred)
    return res
def auc(pos, neg):  # prob a random pos scores below a random neg (pos=in-domain dist small, neg=foreign dist large)
    # we want in-domain distances < foreign distances -> AUC of (foreign>indomain)
    c=0; t=0
    for p in pos:
        for n in neg:
            t+=1; c+= (n>p)+0.5*(n==p)
    return c/t

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--gemma",required=True); ap.add_argument("--qwen",required=True); ap.add_argument("--foreign",required=True)
    ap.add_argument("--lam",type=float,default=100.0); ap.add_argument("--mlp",action="store_true")
    ap.add_argument("--test",type=float,default=0.2); ap.add_argument("--save",default=None)
    a=ap.parse_args()
    G=np.load(a.gemma).astype(np.float64); Q=np.load(a.qwen).astype(np.float64); F=np.load(a.foreign).astype(np.float64)
    n=len(G); assert len(Q)==n, f"pair mismatch {len(G)} vs {len(Q)}"
    rng=np.random.RandomState(0); idx=rng.permutation(n); nt=int(n*(1-a.test))
    tr,te=idx[:nt],idx[nt:]
    gmu,gsd=zscore_fit(G[tr]); qmu,qsd=zscore_fit(Q[tr])
    Gz=zscore(G,gmu,gsd); Qz=zscore(Q,qmu,qsd); Fz=zscore(F,gmu,gsd)
    print(f"[data] N={n} train={len(tr)} test={len(te)} | Dg={G.shape[1]} Dq={Q.shape[1]} foreign={len(F)}")

    if a.mlp:
        import torch, torch.nn as nn
        dev="cuda" if torch.cuda.is_available() else "cpu"
        Xtr=torch.tensor(Gz[tr],dtype=torch.float32,device=dev); Ytr=torch.tensor(Qz[tr],dtype=torch.float32,device=dev)
        net=nn.Sequential(nn.Linear(G.shape[1],512),nn.GELU(),nn.Linear(512,Q.shape[1])).to(dev)
        opt=torch.optim.Adam(net.parameters(),1e-3,weight_decay=1e-4)
        for ep in range(3000):
            opt.zero_grad(); loss=((net(Xtr)-Ytr)**2).mean(); loss.backward(); opt.step()
        with torch.no_grad():
            Wf=None; predq=net(torch.tensor(Gz,dtype=torch.float32,device=dev)).cpu().numpy()
        # back map via separate net
        net2=nn.Sequential(nn.Linear(Q.shape[1],512),nn.GELU(),nn.Linear(512,G.shape[1])).to(dev)
        opt2=torch.optim.Adam(net2.parameters(),1e-3,weight_decay=1e-4)
        Ytr2=torch.tensor(Gz[tr],dtype=torch.float32,device=dev)
        for ep in range(3000):
            opt2.zero_grad(); loss=((net2(Ytr)-Ytr2)**2).mean(); loss.backward(); opt2.step()
        with torch.no_grad():
            predg=net2(torch.tensor(predq,dtype=torch.float32,device=dev)).cpu().numpy()
            predFq=net(torch.tensor(Fz,dtype=torch.float32,device=dev)).cpu().numpy()
        tag="MLP(512,GELU)"
    else:
        Wfwd=ridge(Gz[tr],Qz[tr],a.lam); Wbwd=ridge(Qz[tr],Gz[tr],a.lam)
        predq=Gz@Wfwd; predg=predq@Wbwd; predFq=Fz@Wfwd
        tag=f"ridge(lam={a.lam})"
        if a.save:
            np.savez(a.save, W_fwd=Wfwd.astype(np.float32), W_bwd=Wbwd.astype(np.float32),
                     gmu=gmu.astype(np.float32), gsd=gsd.astype(np.float32),
                     qmu=qmu.astype(np.float32), qsd=qsd.astype(np.float32),
                     src="google/gemma-3n-E2B-it", dst="Qwen/Qwen2.5-Coder-0.5B-Instruct", lam=a.lam)
            print(f"[save] adapter -> {a.save}")

    # ---- metrics on held-out test ----
    fwd_cos=cos(predq[te],Qz[te]).mean()
    rt_cos =cos(predg[te],Gz[te]).mean()
    base_cos=cos(Gz[te][rng.permutation(len(te))],Qz[te]).mean() if G.shape[1]==Q.shape[1] else float('nan')
    ret=retrieval_at_k(predq[te],Qz[te])
    # reject: distance of mapped point to nearest TRUE qwen (train set as the gallery)
    gallery=Qz[tr]
    def nndist(P):
        Pn=P/(np.linalg.norm(P,axis=1,keepdims=True)+1e-8); Gn=gallery/(np.linalg.norm(gallery,axis=1,keepdims=True)+1e-8)
        return 1-(Pn@Gn.T).max(1)   # 1 - max cosine = nearest-neighbor cosine distance
    d_in=nndist(predq[te]); d_for=nndist(predFq)
    rej_auc=auc(d_in,d_for)
    print(f"\n==== Telepathy gemma->qwen adapter [{tag}] ====")
    print(f"ROUNDTRIP  cos(g->q->g, g) held-out = {rt_cos:.3f}   (random-pair baseline ~ {base_cos:.3f})")
    print(f"SEMANTIC   fwd cos(g->q, q) held-out = {fwd_cos:.3f}")
    print(f"SEMANTIC   retrieval@1 = {ret[1]:.3f}  retrieval@5 = {ret[5]:.3f}   (chance@1 = {1/len(te):.3f})")
    print(f"REJECT     in-domain nn-dist mean={d_in.mean():.3f}  foreign nn-dist mean={d_for.mean():.3f}  AUC={rej_auc:.3f}")
    print(f"VERDICT    roundtrip {'GREEN' if rt_cos>0.6 else 'AMBER' if rt_cos>0.4 else 'RED'} | "
          f"retr@5 {'GREEN' if ret[5]>0.5 else 'AMBER' if ret[5]>0.2 else 'RED'} | "
          f"reject {'GREEN' if rej_auc>0.8 else 'AMBER' if rej_auc>0.65 else 'RED'}")

if __name__=="__main__":
    main()
