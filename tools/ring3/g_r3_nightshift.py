#!/usr/bin/env python3
# R3.4 G-R3-NIGHTSHIFT — the idle-loop consolidation state machine (the thermodynamic garbage collector).
# Path A, parameter-free. Moves episodes from the expensive resident Ring-2 pool into the dense superposed
# Ring-3 index BEFORE resident capacity runs out, under the irreversible-aware G-R3-LOSS gate.
#
# Per episode:  SELECT (a Ring-2 resident episode) -> BIND (addr (*) id into the active Ring-3 vector, in a
#   SHADOW copy) -> SHADOW-GATE (re-verify EVERY bound episode in the vector still recalls@1 above margin>0 —
#   crosstalk from the new bind must not knock out an earlier one) -> if PASS: PROMOTE (commit the bind) +
#   EVICT (free the resident slot; the verbatim ep.k STAYS on Optane for retrieve-and-verify) ; if FAIL or
#   count==CAP: SEAL the vector read-only and initialize a fresh empty one (the episode re-tries in the new vector).
#
# The SEAL is gate-driven (capacity), with CAP=32 as the pre-registered safety cap (R3.2 budget @ D=1024).
# Proven twice: (A) D=1024 CAP=32 production — gate holds to the cap; (B) small D — the GATE fires BEFORE the
# cap, proving the seal is the math, not a magic constant. Telemetry: resident footprint stays bounded.
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ok_bind as ok  # NATIVE exact-integer negacyclic CRT-NTT bind (Leg A)
SEED=0x5350524F4A2B; R_BITS=256; HD=512; NL,PERIOD=48,8; MASK64=(1<<64)-1; CAP=32
def smix(seed,n):
    s=seed&MASK64; out=np.empty(n,dtype=np.int8)
    for i in range(n):
        s=(s+0x9E3779B97F4A7C15)&MASK64; z=s
        z=((z^(z>>30))*0xBF58476D1CE4E5B9)&MASK64; z=((z^(z>>27))*0x94D049BB133111EB)&MASK64
        z=z^(z>>31); out[i]=1 if (z&1) else -1
    return out
def gl(): return [L for L in range(NL) if (L%PERIOD)==PERIOD-1]
def loadK(d):
    raw=np.fromfile(os.path.join(d,"ep.k"),dtype="<f4"); P=raw.size//(NL*HD); return raw.reshape(NL,P,HD),P,raw.nbytes
def ep_sig_seed(epdir,npos):
    R=smix(SEED,R_BITS*HD).astype(np.float32).reshape(R_BITS,HD)
    K_,P,_=loadK(epdir); rp=list(range(min(npos,P)))
    v=np.stack([R@K_[L,p] for L in gl() for p in rp],0).mean(0); b=(v>0); s=0
    for i in range(R_BITS):
        if b[i]: s|=(1<<i)
    return s & MASK64

class ActiveVector:
    def __init__(self,D):
        self.D=D; self.M=np.zeros(D,dtype=np.int64); self.eps=[]; self.addr={}; self.id={}; self.sealed=False
    def _carrier(self,seed): return ok.carrier(seed, self.D)
    def _id(self,seed): return ok.idvec(seed, self.D)
    def _cconv(self,a,b): return ok.bind(a,b)        # NATIVE engine bind
    def _ccorr(self,a,b): return ok.unbind(a,b)      # NATIVE engine unbind
    def _cos(self,a,b): return ok.cos(a,b)
    def _recall_ok(self, M, eps):
        # SHADOW-GATE: every bound episode must recall@1 (correct id is argmax) with margin>0 over the others
        for q in eps:
            est=self._ccorr(M, self.addr[q])
            sims={e:self._cos(est,self.id[e]) for e in eps}
            best=max(sims,key=sims.get); margin=sims[q]-max(v for e,v in sims.items() if e!=q) if len(eps)>1 else sims[q]
            if best!=q or margin<=0: return False
        return True
    def try_bind(self, name, seed):
        if self.sealed or len(self.eps)>=CAP: return False
        self.addr[name]=self._carrier(seed); self.id[name]=self._id(seed)
        shadow=self.M + self._cconv(self.addr[name], self.id[name])
        if self._recall_ok(shadow, self.eps+[name]):
            self.M=shadow; self.eps.append(name); return True       # PROMOTE
        del self.addr[name]; del self.id[name]; return False        # gate fail -> caller seals
    def seal(self): self.sealed=True

def nightshift(D, batch, ep_bytes):
    vectors=[ActiveVector(D)]; consolidated=[]; resident=set(b[0] for b in batch); freed=0
    print(f"  [run D={D} CAP={CAP}] batch={len(batch)} episodes; resident pool starts at {len(resident)}")
    for name,seed in batch:
        v=vectors[-1]
        if not v.try_bind(name,seed):
            v.seal()
            why = "CAP" if len(v.eps)>=CAP else "gate (capacity)"
            print(f"    seal vector #{len(vectors)} at {len(v.eps)} episodes [{why}] -> init fresh vector")
            vectors.append(ActiveVector(D));
            assert vectors[-1].try_bind(name,seed), "fresh vector must accept"
        # PROMOTE+EVICT: pointer in Ring-3, free the resident slot (verbatim stays on Optane)
        consolidated.append(name); resident.discard(name); freed+=ep_bytes.get(name,0)
    # final verify: every consolidated episode recalls@1 in its (now sealed-or-active) vector
    allok=True
    for v in vectors:
        if not v._recall_ok(v.M, v.eps): allok=False
    sealed=sum(1 for v in vectors if v.sealed); sizes=[len(v.eps) for v in vectors]
    r3_resident_bytes = sum(v.D*8 for v in vectors) + len(consolidated)*8   # M (f8) per vector + 8B pointer/episode
    print(f"    -> {len(consolidated)} consolidated across {len(vectors)} vectors {sizes}; sealed={sealed}; resident pool now {len(resident)}")
    print(f"    -> resident KV freed (demoted to Optane): {freed/1e6:.1f} MB ; Ring-3 resident footprint: {r3_resident_bytes/1024:.1f} KB")
    print(f"    -> all consolidated episodes recall@1 in their vector: {'GREEN' if allok else 'RED'}")
    cap_seal = all(s==CAP for s in sizes[:-1]) if D>=1024 else True     # production: full vectors sealed at CAP
    gate_seal = (max(sizes)<CAP) if D<1024 else True                   # small D: GATE fires before CAP
    return allok, cap_seal, gate_seal, sizes

def main():
    eng=os.environ.get("SP_R3_ENG", os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","..")))
    realK={"ep_toy":(f"{eng}/_p33_ep",16),"ep_wiki":(f"{eng}/_c2_ep_wiki",84)}
    ep_bytes={}; seeds={}
    for n,(d,p) in realK.items():
        _,_,nb=loadK(d); ep_bytes[n]=nb; seeds[n]=ep_sig_seed(d,p)
    # batch = the 2 real Ring-2 episodes + synthetic episode descriptors (random sig seeds), ~8.3MB resident each
    rng=np.random.default_rng(31337); batch=[("ep_toy",seeds["ep_toy"]),("ep_wiki",seeds["ep_wiki"])]
    for i in range(38):
        nm=f"ep{i:02d}"; batch.append((nm,int(rng.integers(1,2**62)))); ep_bytes[nm]=8_300_000
    print("[ns] NIGHTSHIFT consolidation loop — SELECT->BIND->SHADOW-GATE->PROMOTE+EVICT->SATURATE&SEAL\n")
    print("[ns] (A) production D=1024, CAP=32 (R3.2 budget): gate holds to the cap, vector seals at 32, fresh vector continues")
    okA,capA,_,szA=nightshift(1024, batch, ep_bytes)
    print("\n[ns] (B) small D=128: capacity ~D/8 << CAP — the SHADOW-GATE must fire BEFORE the count cap (seal is the MATH, not 32)")
    okB,_,gateB,szB=nightshift(128, batch, ep_bytes)
    ok = okA and capA and okB and gateB
    print(f"\n[ns]   (A) all recall@1 + vectors seal at CAP=32: {'PASS' if (okA and capA) else 'FAIL'} (sizes {szA})")
    print(f"[ns]   (B) gate-driven seal before CAP (max vector {max(szB)} < 32): {'PASS' if (okB and gateB) else 'FAIL'} (sizes {szB})")
    print(f"\n[gate] G-R3-NIGHTSHIFT {'GREEN -- autonomous consolidation: bind -> shadow-gate (whole-set recall) -> promote+evict (verbatim to Optane) -> saturate&seal; seal is gate-driven with CAP=32 safety cap; resident pool bounded' if ok else 'RED'}")
    return 0 if ok else 1

if __name__=="__main__":
    import sys; sys.exit(main())
