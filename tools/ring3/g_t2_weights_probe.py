#!/usr/bin/env python3
# G-T2-WEIGHTS: test T2 Mobius square-free reconstruction on a real embedding block.
# Input: _embed_block.bf16 (first N rows of embed_tokens, bf16). Output: Mobius-energy split +
# composite-row reconstruction cos vs random baseline. See tests/fixtures/xbar_r3/G-T2-WEIGHTS.log.
import numpy as np, sys
N=int(sys.argv[2]) if len(sys.argv)>2 else 4096; E=int(sys.argv[3]) if len(sys.argv)>3 else 3840
raw=np.fromfile(sys.argv[1] if len(sys.argv)>1 else "_embed_block.bf16",dtype="<u2")[:N*E].reshape(N,E)
f=(raw.astype(np.uint32)<<16).view(np.float32).astype(np.float64)
def mu_sf(n):
    m=1; sf=True; x=n; p=2
    while p*p<=x:
        if x%p==0:
            x//=p
            if x%p==0:
                sf=False
                while x%p==0: x//=p
            m=-m
        p+=1
    if x>1: m=-m
    return (0 if not sf else m), sf
mu=np.zeros(N,dtype=np.int64); sf=np.zeros(N,bool)
for n in range(1,N): mu[n],sf[n]=mu_sf(n)
divs=[[] for _ in range(N)]
for d in range(1,N):
    for n in range(d,N,d): divs[n].append(d)
g=np.zeros((N,E))
for n in range(1,N):
    for d in divs[n]:
        if mu[n//d]: g[n]+=mu[n//d]*f[d]
en_sf=float((np.linalg.norm(g[1:],axis=1)[sf[1:]]**2).sum()); en_nsf=float((np.linalg.norm(g[1:],axis=1)[~sf[1:]]**2).sum())
cos=lambda a,b: float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
comp=[n for n in range(2,N) if not sf[n]][:400]
recon=lambda n: sum((g[d] for d in divs[n] if sf[d]), np.zeros(E))
rng=np.random.default_rng(0)
print(f"non-sf Mobius energy {100*en_nsf/(en_sf+en_nsf):.2f}%  recon_cos {np.mean([cos(recon(n),f[n]) for n in comp]):.4f}  rand {np.mean([cos(f[rng.integers(1,N)],f[n]) for n in comp]):.4f}")
