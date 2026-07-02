"""rep_combine.py — the multi-signal recall head: combine 3 cheap orthogonal signals.

Signals per (incoming query, stored episode key):
  J  = Jaccard token overlap  (lexical; 100% exact / ~8% paraphrase)
  L5 = cosine of L5 query embeddings (directional/semantic; 100% exact / 88.5% para)
  MD = KSTE-MD magnitude-shape similarity (-L1 of Dickson sigma0(+)sigma1; orthogonal)
Test whether a GATED (jaccard-else-L5) or LEARNED (logistic) head over these beats any single
signal on BOTH exact and paraphrase. Held-out split for the learned head. n=61 fact-conflicts.

Usage: python rep_combine.py <exact_dir> <para_dir> <facts.json>
"""
import os, sys, glob, struct, json, re
import numpy as np
np.seterr(all="ignore")
L5, HD, G_NH, LMAX = 5, 512, 16, 8

def load(p):
    b=open(p,"rb").read(); ng,d=struct.unpack("<II",b[:8]); return ng,np.frombuffer(b[8:],"<f4").astype(np.float64)
def dq(d):
    o={}
    for p in glob.glob(os.path.join(d,"q_*.bin")):
        c=int(os.path.basename(p)[2:-4]); ng,a=load(p); o[c]=a.reshape(ng,G_NH,HD)
    return [o[c] for c in sorted(o)]
def l5e(vecs):
    M=np.stack([v[L5].mean(0) for v in vecs]); return M/(np.linalg.norm(M,axis=1,keepdims=True)+1e-30)
def toks(s): return set(re.findall(r"[a-z0-9]+", s.lower()))
def jac(a,b):
    A,B=toks(a),toks(b); return len(A&B)/max(1,len(A|B))

def md_sig(vf):
    """faithful python port of kste_md on a float vector (scaled to int32)."""
    v=np.clip(vf*1e6,-2.147e9,2.147e9).astype(np.int64)
    a=np.abs(v); order=np.argsort(-a, kind="stable"); amax=max(1,int(a[order[0]]))
    nA=int(np.sum(a[order[:14]]>0))
    used=0; nB=nC=MBB=MCC=0; dmax=0
    for r in order[14:74]:
        av=int(a[r]);
        if av==0: break
        L=1+int((LMAX-1)*av//amax); L=min(max(L,1),LMAX)
        if used+L>60: L=60-used
        if L<=0: break
        used+=L; internal=L*(L-1)//2
        if v[r]>=0: nB+=L; MBB+=internal
        else: nC+=L; MCC+=internal
        dmax=max(dmax,L+1)
    ntot=nA+nB+nC
    return np.array([nA,nB,nC,dmax,ntot, 0,nB,nC,0,MBB,0,0,0,MCC], float)

def recall1_rank(score):  # score[i,j] higher=more similar; correct=diagonal
    return float(np.mean(np.argmax(score,axis=1)==np.arange(score.shape[0])))
def zrows(M):
    return (M-M.mean(1,keepdims=True))/(M.std(1,keepdims=True)+1e-9)

ex_d,pa_d,fj=sys.argv[1],sys.argv[2],sys.argv[3]
EX,PA=dq(ex_d),dq(pa_d); F=json.load(open(fj,encoding="utf-8"))
n=min(len(EX),len(PA),len(F)); EX,PA,F=EX[:n],PA[:n],F[:n]
Ex5,Pa5=l5e(EX),l5e(PA)
MDx=np.stack([md_sig(EX[i][L5].mean(0)) for i in range(n)])   # exact L5 md sig (episode key)
MDe=MDx                                                       # key = exact
MDpi=np.stack([md_sig(PA[i][L5].mean(0)) for i in range(n)])
print(f"=== REP-COMBINE  n={n}  (J lexical + L5 directional + MD magnitude-shape) ===\n")

def feats(incoming_text_idx, inc_field, inc_L5, inc_MD):
    J =np.array([[jac(F[i][inc_field], F[j]["fact"]) for j in range(n)] for i in range(n)])
    L =inc_L5 @ Ex5.T
    MD=-np.stack([[np.abs(inc_MD[i]-MDe[j]).sum() for j in range(n)] for i in range(n)])
    return J,L,MD

def run(tag, inc_field, inc_L5, inc_MD):
    J,L,MD = feats(None, inc_field, inc_L5, inc_MD)
    print(f"[{tag}]  singles:  J={100*recall1_rank(J):5.1f}%   L5={100*recall1_rank(L):5.1f}%   MD={100*recall1_rank(MD):5.1f}%")
    # gated: if best lexical overlap for query i >= tau use J-argmax, else L5-argmax
    tau=0.30; hit=0
    for i in range(n):
        pick = np.argmax(J[i]) if J[i].max()>=tau else np.argmax(L[i])
        hit += (pick==i)
    print(f"        GATED(J if maxJ>= {tau} else L5): {100*hit/n:5.1f}%")
    return J,L,MD

Jx,Lx,MDx_ = run("EXACT incoming", "q",   Ex5, MDx)
Jp,Lp,MDp_ = run("PARA  incoming", "para",Pa5, MDpi)

# learned logistic head over z-scored [J,L,MD], trained on EXACT+PARA train split, tested held-out
def design(J,L,MD):
    return np.stack([zrows(J),zrows(L),zrows(MD)],axis=-1)  # [n,n,3]
Dx,Dp = design(Jx,Lx,MDx_), design(Jp,Lp,MDp_)
rng=np.random.default_rng(0); perm=rng.permutation(n); tr,te=perm[:46],perm[46:]
X=[];y=[]
for split_D in (Dx,Dp):
    for i in tr:
        for j in range(n):
            X.append(split_D[i,j]); y.append(1.0 if j==i else 0.0)
X=np.array(X); y=np.array(y)
w=np.zeros(3); b=0.0; lr=0.5
for _ in range(3000):
    z=X@w+b; p=1/(1+np.exp(-z)); g=p-y
    w-=lr*(X.T@g/len(y)+1e-3*w); b-=lr*g.mean()
def rec_learned(D, idx):
    hit=0
    for i in idx:
        s=D[i]@w+b; hit+=(np.argmax(s)==i)
    return 100*hit/len(idx)
print(f"\n[LEARNED logistic head]  weights J={w[0]:+.2f} L5={w[1]:+.2f} MD={w[2]:+.2f}")
print(f"  held-out (n={len(te)}):  EXACT={rec_learned(Dx,te):5.1f}%   PARA={rec_learned(Dp,te):5.1f}%")
print(f"  train    (n={len(tr)}):  EXACT={rec_learned(Dx,tr):5.1f}%   PARA={rec_learned(Dp,tr):5.1f}%")
print("\n(goal: a combiner that holds ~100% EXACT and rescues PARA, beating any single signal)")
