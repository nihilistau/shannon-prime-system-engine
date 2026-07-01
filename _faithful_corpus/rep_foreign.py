import os,sys,glob,struct,json,re
import numpy as np
np.seterr(all="ignore")
L5,HD,G_NH=5,512,16
def load(p):
    b=open(p,"rb").read(); ng,d=struct.unpack("<II",b[:8]); return ng,np.frombuffer(b[8:],"<f4").astype(np.float64)
def dq(d):
    o={}
    for p in glob.glob(os.path.join(d,"q_*.bin")):
        c=int(os.path.basename(p)[2:-4]); ng,a=load(p); o[c]=a.reshape(ng,G_NH,HD)
    return [o[c] for c in sorted(o)]
def l5(vecs):
    M=np.stack([v[L5].mean(0) for v in vecs]); return M/(np.linalg.norm(M,axis=1,keepdims=True)+1e-30)
def toks(s): return set(re.findall(r"[a-z0-9]+",s.lower()))
def jac(a,b):
    A,B=toks(a),toks(b); return len(A&B)/max(1,len(A|B))

eng=sys.argv[1]
KEYS=l5(dq(os.path.join(eng,"_faithful_corpus","qdump")))           # exact-query L5 keys [61,512]
PARA=l5(dq(os.path.join(eng,"_faithful_corpus","qdump_para")))      # in-memory paraphrase queries
FOR =l5(dq(os.path.join(eng,"_faithful_corpus","qdump_foreign")))   # foreign queries
F=json.load(open(os.path.join(eng,"_faithful_corpus","facts.json"),encoding="utf-8"))
FQ=json.load(open(os.path.join(eng,"_faithful_corpus","foreign_queries.json"),encoding="utf-8"))
ep_text=[f["fact"] for f in F]
print(f"keys={len(KEYS)} para(in-mem)={len(PARA)} foreign={len(FOR)}")

def feats(Q, qtexts):
    C=Q@KEYS.T                                   # [n,61]
    s=np.sort(C,axis=1)
    top1=s[:,-1]; top2=s[:,-2]; margin=top1-top2
    mj=np.array([max(jac(qtexts[i],t) for t in ep_text) for i in range(len(Q))])
    return top1,margin,mj
pt1,pm,pj = feats(PARA,[ (F[i].get("para") or F[i]["q"]) for i in range(len(PARA)) ])
ft1,fm,fj = feats(FOR, FQ[:len(FOR)])

def stats(name,a): print(f"  {name:16s} n={len(a):3d}  mean={a.mean():.3f} p10={np.percentile(a,10):.3f} p50={np.percentile(a,50):.3f} p90={np.percentile(a,90):.3f}")
print("\n[distributions]  (in-memory PARA should ACCEPT; FOREIGN should REJECT)")
print(" top1 L5-cos:"); stats("para",pt1); stats("foreign",ft1)
print(" margin(top1-top2):"); stats("para",pm); stats("foreign",fm)
print(" max Jaccard:"); stats("para",pj); stats("foreign",fj)

def gate_report(name, pscore, fscore, sweep):
    # accept if score>=tau. find tau giving ~=85% para-accept, report foreign false-accept.
    best=None
    for tau in sweep:
        pa=(pscore>=tau).mean(); fa=(fscore>=tau).mean()
        if pa>=0.85:
            if best is None or fa<best[2]: best=(tau,pa,fa)
    if best: print(f"  {name:22s} @tau={best[0]:.3f}: para-accept={best[1]:.2%}  foreign-FALSE-accept={best[2]:.2%}")
    else:    print(f"  {name:22s}: cannot reach 85% para-accept")
print("\n[single-signal reject gates] (target: keep para-accept>=85%, minimize foreign false-accept)")
gate_report("L5 top1-cos alone", pt1, ft1, np.linspace(0.80,1.0,101))
gate_report("margin alone",      pm,  fm,  np.linspace(0,0.3,151))
gate_report("maxJaccard alone",  pj,  fj,  np.linspace(0,0.5,101))

# multi-signal: accept if top1>=tc AND margin>=tm ; grid for max(para-accept - foreign-accept) with para>=0.85
best=None
for tc in np.linspace(0.80,0.99,40):
  for tm in np.linspace(0,0.15,40):
    pa=((pt1>=tc)&(pm>=tm)).mean(); fa=((ft1>=tc)&(fm>=tm)).mean()
    if pa>=0.85 and (best is None or fa<best[3]): best=(tc,tm,pa,fa)
print("\n[multi-signal gate]  accept iff (L5 top1 >= tc) AND (margin >= tm)")
if best: print(f"  @tc={best[0]:.3f} tm={best[1]:.3f}: para-accept={best[2]:.2%}  foreign-FALSE-accept={best[3]:.2%}")
else: print("  cannot reach 85% para-accept")
