#!/usr/bin/env python3
# R3.1 G-R3-BIND — prove the VSA/HRR superposition math on the real Ring-2 episode tensors (Path A, parameter-free).
#
# Store:  M = sum_i ( addr_i (*) id_i )           (*) = circular convolution (== the engine's Zq/NTT pointwise multiply)
#   addr_i : a carrier SEEDED by episode i's real C2 256-bit signature (content-derived from ep.k global keys,
#            so a live cue regenerates the same address; ties Ring-3 to the proven C2 resolver).
#   id_i   : a clean random +/-1 id label — the "episode signature" surfaced on recall; points back to the
#            Ring-2 verbatim episode for the exact #222 verify (NOT a reconstructed span — the SS4 trap stays shut).
# Recall: id_est = M (corr) addr_j ; cleanup = argmax cos(id_est, id_k) over the codebook -> episode j.
#
# Metric (the cleanup standard, NOT a ratio-to-possibly-negative denominator):
#   margin_j = cos(correct) - max cos(wrong)   > 0  iff recall@1 is correct & strictly above crosstalk.
#   z_j      = (cos(correct) - mean(wrong)) / std(wrong)   = the signal z-score above the crosstalk distribution (N>=3).
# Pass (pre-registered): N=2 both real margins>0 (recall@1) for the substrate +/-1 carrier; capacity recall@5>=0.90 to N=64 @ D=1024.
#
# Domain note: this proves the binding ALGEBRA + capacity in the real domain via FFT circular convolution, which is
# mathematically the engine's NTT-over-Zq (DESIGN-VSA-ring3 SS2). The DEPLOYMENT binds over Zq via the engine NTT/CRT
# (exact integer, no float drift) — that engine port is R3.x, not this math proof.
import os, numpy as np
SEED=0x5350524F4A2B; R_BITS=256; HD=512; NL,PERIOD=48,6; MASK64=(1<<64)-1; D=1024

def smix(seed,n):
    s=seed&MASK64; out=np.empty(n,dtype=np.int8)
    for i in range(n):
        s=(s+0x9E3779B97F4A7C15)&MASK64; z=s
        z=((z^(z>>30))*0xBF58476D1CE4E5B9)&MASK64; z=((z^(z>>27))*0x94D049BB133111EB)&MASK64
        z=z^(z>>31); out[i]=1 if (z&1) else -1
    return out
def gl(): return [L for L in range(NL) if (L%PERIOD)==PERIOD-1]
def loadK(d):
    raw=np.fromfile(os.path.join(d,"ep.k"),dtype="<f4"); P=raw.size//(NL*HD); return raw.reshape(NL,P,HD),P
def ep_sig_seed(epdir,npos):
    R=smix(SEED,R_BITS*HD).astype(np.float32).reshape(R_BITS,HD)
    K,P=loadK(epdir); rp=list(range(min(npos,P)))
    v=np.stack([R@K[L,p] for L in gl() for p in rp],0).mean(0); b=(v>0); s=0
    for i in range(R_BITS):
        if b[i]: s|=(1<<i)
    return s & MASK64
def carrier(seed,kind):
    rng=np.random.default_rng(seed % (2**63))
    if kind=="unitary":
        th=rng.uniform(0,2*np.pi,D//2+1); F=np.exp(1j*th); F[0]=1.0
        if D%2==0: F[-1]=1.0
        return np.fft.irfft(F,n=D)
    return rng.integers(0,2,D).astype(np.float64)*2-1     # +/-1 Rademacher (substrate-native)
def idvec(seed):
    rng=np.random.default_rng((seed^0xABCDEF) % (2**63)); return rng.integers(0,2,D).astype(np.float64)*2-1
def cconv(a,b): return np.fft.irfft(np.fft.rfft(a)*np.fft.rfft(b),n=D)
def ccorr(a,b): return np.fft.irfft(np.fft.rfft(a)*np.conj(np.fft.rfft(b)),n=D)
def cos(a,b): return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))

def main():
    eng=os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..",".."))
    real=[("ep_toy",f"{eng}/_p33_ep",16),("ep_wiki",f"{eng}/_c2_ep_wiki",84)]
    rs=[ep_sig_seed(d,n) for _,d,n in real]
    print(f"[r3] D={D}  real sig-seeds: ep_toy={hex(rs[0])[:12]}.. ep_wiki={hex(rs[1])[:12]}..",flush=True)
    def run(kind,N):
        rng=np.random.default_rng(777+N)
        seeds=list(rs)+[int(rng.integers(1,2**62)) for _ in range(N-2)]
        addrs=[carrier(s,kind) for s in seeds]; ids=[idvec(s) for s in seeds]
        M=np.zeros(D)
        for a,v in zip(addrs,ids): M+=cconv(a,v)
        h1=h5=0; mar=[]; zs=[]
        for j in range(N):
            est=ccorr(M,addrs[j]); sims=np.array([cos(est,ids[k]) for k in range(N)]); o=np.argsort(-sims)
            if o[0]==j: h1+=1
            if j in o[:5]: h5+=1
            ot=np.delete(sims,j); mar.append(sims[j]-ot.max()); zs.append((sims[j]-ot.mean())/(ot.std()+1e-9) if ot.size>1 else float('nan'))
        return h1/N,h5/N,mar,zs
    print(f"\n{'kind':>8}{'N':>5} | {'rec@1':>6} {'rec@5':>6} | {'min_margin':>10} {'mean_z':>7} | toy_mgn wiki_mgn",flush=True)
    res={}
    for kind in ("unitary","pm1"):
        for N in (2,8,32,64,128,256):
            r1,r5,m,z=run(kind,N); res[(kind,N)]=(r1,r5,m,z); zz=np.nanmean(z) if N>2 else float('nan')
            print(f"{kind:>8}{N:>5} | {r1:>6.3f} {r5:>6.3f} | {min(m):>10.4f} {zz:>7.2f} | {m[0]:>7.4f} {m[1]:>7.4f}",flush=True)
    m2=res[("pm1",2)][2]
    n2=(res[("pm1",2)][0]==1.0 and m2[0]>0 and m2[1]>0 and res[("unitary",2)][0]==1.0)
    cap=all(res[("pm1",N)][1]>=0.90 for N in (8,32,64))
    print(f"\n[r3] N=2 ep_toy+ep_wiki (pm1): recall@1={res[('pm1',2)][0]:.0f}/1 margins=({m2[0]:.4f},{m2[1]:.4f}) -> {'PASS' if n2 else 'FAIL'}",flush=True)
    print(f"[r3] capacity pm1 recall@5>=0.90 to N=64 @ D={D}: {'PASS' if cap else 'FAIL'}",flush=True)
    print(f"\n[gate] G-R3-BIND {'GREEN -- VSA superposition recalls the right episode id strictly above crosstalk (margin>0); graceful capacity ~N=64 @ D=1024; +/-1 substrate carrier tracks the ideal unitary one' if (n2 and cap) else 'RED'}",flush=True)
    return 0 if (n2 and cap) else 1

if __name__=="__main__":
    import sys; sys.exit(main())
