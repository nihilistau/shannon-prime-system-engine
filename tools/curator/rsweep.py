#!/usr/bin/env python3
# Does the DISCRETE (sign-bit Hamming) resolver separate if we widen the hash? Sweep r.
import os, numpy as np
SEED=0x5350524F4A2B; HD=512; NL,PERIOD=48,6; MASK64=(1<<64)-1
def smix(seed,n):
    s=seed&MASK64; out=np.empty(n,dtype=np.int8)
    for i in range(n):
        s=(s+0x9E3779B97F4A7C15)&MASK64; z=s
        z=((z^(z>>30))*0xBF58476D1CE4E5B9)&MASK64
        z=((z^(z>>27))*0x94D049BB133111EB)&MASK64
        z=z^(z>>31); out[i]=1 if (z&1) else -1
    return out
def gl(): return [L for L in range(NL) if (L%PERIOD)==PERIOD-1]
def loadK(d):
    raw=np.fromfile(os.path.join(d,"ep.k"),dtype="<f4"); P=raw.size//(NL*HD)
    return raw.reshape(NL,P,HD),P
eng="/sessions/friendly-dreamy-ramanujan/mnt/shannon-prime-system-engine"
eps={"ep_toy":(f"{eng}/_p33_ep",16),"ep_wiki":(f"{eng}/_c2_ep_wiki",84)}
Kc={n:loadK(d) for n,(d,_) in eps.items()}
ksc=float(np.std(Kc["ep_toy"][0][gl()])); rng=np.random.default_rng(20260617)
print(f"{'r':>4} | {'ep_toy self':>11} {'ep_wiki self':>12} | {'max non-target':>14} | bit-gap | clean?")
for r in (32,64,128,256,512):
    Rfull=smix(SEED,r*HD).astype(np.float32).reshape(r,HD)
    def sig(name,half):  # half=0 -> first half (sig), 1 -> second (cue)
        K,P=Kc[name]; rp=list(range(min(eps[name][1],P))); h=len(rp)//2
        pos=rp[:h] if half==0 else rp[h:]
        v=np.stack([Rfull@K[L,p] for L in gl() for p in pos],0).mean(0)
        return (v>0)
    S={n:sig(n,0) for n in eps}; C={n:sig(n,1) for n in eps}
    NEG=[((rng.normal(0,ksc,size=(len(gl())*24,HD)).astype(np.float32))@Rfull.T).mean(0)>0 for _ in range(8)]
    def ag(a,b): return int(np.sum(a==b))
    self_toy=ag(C["ep_toy"],S["ep_toy"]); self_wiki=ag(C["ep_wiki"],S["ep_wiki"])
    nontgt=0
    for ce in eps:
        nontgt=max(nontgt, max(ag(C[ce],S[o]) for o in eps if o!=ce), max(ag(C[ce],n) for n in NEG))
    posmin=min(self_toy,self_wiki); gap=posmin-nontgt
    print(f"{r:>4} | {self_toy:>6}/{r:<4} {self_wiki:>6}/{r:<5} | {nontgt:>10}/{r:<3} | {gap:>+6} | {'YES' if gap>0 else 'no'}  (chance={r//2})")
