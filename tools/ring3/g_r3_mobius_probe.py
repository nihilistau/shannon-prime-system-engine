import sys, numpy as np
sys.path.insert(0,"/tmp")
import ok_bind as ok
D=1024; N=32
rng=np.random.default_rng(2026)
seeds=[int(rng.integers(1,2**62)) for _ in range(N)]
addrs=[ok.carrier(s,D) for s in seeds]; ids=[ok.idvec(s,D) for s in seeds]
M=np.zeros(D,dtype=np.int64)
for a,v in zip(addrs,ids): M=M+ok.bind(a,v)
def recall(Mv):
    h=0
    for j in range(N):
        est=ok.unbind(Mv,addrs[j]); sims=[ok.cos(est,ids[k]) for k in range(N)]
        if int(np.argmax(sims))==j: h+=1
    return h/N
# (1) density of the dense holographic superposition
nz=int(np.count_nonzero(M)); print(f"[M] D={D} N={N}  nonzero components: {nz}/{D} = {100*nz/D:.1f}%  (dense by construction)")
print(f"[M] baseline recall@1 (full dense M): {recall(M):.3f}")
# (2) square-free index mask (1-based index i in [1..D]); density of square-free ints ~ 6/pi^2
def squarefree(n):
    i=2
    while i*i<=n:
        if n%(i*i)==0: return False
        i+=1
    return True
sf=np.array([squarefree(i) for i in range(1,D+1)])
print(f"[sf] square-free indices in [1..{D}]: {sf.sum()}/{D} = {100*sf.mean():.2f}%  (theory 6/pi^2={100*6/np.pi**2:.2f}%)")
# (3) Mobius 'compression' as the directive frames it: keep only square-free-indexed components
Msf=M.copy(); Msf[~sf]=0
print(f"[mobius] recall@1 after keeping ONLY square-free components (composite zeroed): {recall(Msf):.3f}")
# (4) is there ANY exploitable redundancy? entropy proxy: unique-value ratio + can composite be reconstructed from divisors?
# test the embedding-style reconstruction f(i)=sum_{d|i,d<i} ... requires multiplicative structure; measure residual
err=[]
for i in range(2,D+1):
    divs=[d for d in range(1,i) if i%d==0 and squarefree(d)]
    if not divs: continue
    recon=M[np.array(divs)-1].sum()/len(divs)   # any divisor-based reconstruction
    err.append(abs(recon - M[i-1]))
err=np.array(err)
print(f"[mobius] divisor-reconstruction of composite components: mean|err|={err.mean():.1f} vs |M| typ={np.abs(M).mean():.1f}  (ratio {err.mean()/np.abs(M).mean():.2f}x => NO multiplicative redundancy)")
