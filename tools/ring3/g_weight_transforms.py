#!/usr/bin/env python3
# G-WEIGHT-TRANSFORMS: can geometry compress the 12B weights? Tests SVD low-rank, Walsh-Hadamard
# incoherence (the corrected Spinor), and LLL orthogonality-defect, on an extracted embed_tokens block.
# Result: weights are high-rank/high-entropy (no lossless number-theoretic compression); the only real
# lever is the Hadamard/NTT incoherence rotation -> ~2-2.6x lower quant error at fixed bits (QuIP-style).
# See tests/fixtures/xbar_r3/G-WEIGHT-TRANSFORMS.log. Input: _embed_block.bf16 [4096,3840] bf16 (scratch).
import numpy as np, sys
N,E=4096,3840
X=((np.fromfile(sys.argv[1] if len(sys.argv)>1 else "_embed_block.bf16",dtype="<u2")[:N*E].astype(np.uint32)<<16).view(np.float32)).astype(np.float64).reshape(N,E)
fro=np.linalg.norm(X); relL2=lambda A: float(np.linalg.norm(X-A)/fro)
def qrow(T,b): qm=(1<<(b-1))-1; s=(np.abs(T).max(1,keepdims=True)+1e-12)/qm; return np.clip(np.round(T/s),-qm-1,qm)*s
print("baseline:", {f"int{b}":round(relL2(qrow(X,b)),5) for b in (8,4)})
S=np.linalg.svd(X,compute_uv=False); en=np.cumsum(S**2)/np.sum(S**2)
print("svd energy-rank:", {f:int(np.searchsorted(en,f))+1 for f in (0.5,0.9,0.99,0.999)}, "breakeven", int(N*E/(N+E)))
P=4096
def fwht(a):
    a=a.astype(np.float64).copy(); n=a.shape[-1]; h=1
    while h<n:
        for i in range(0,n,2*h):
            x=a[:,i:i+h].copy(); y=a[:,i+h:i+2*h].copy(); a[:,i:i+h]=x+y; a[:,i+h:i+2*h]=x-y
        h*=2
    return a
Xp=np.zeros((N,P)); Xp[:,:E]=X; R=fwht(Xp)/np.sqrt(P)
for b in (8,4):
    Xrec=(fwht(qrow(R,b)*np.sqrt(P))/P)[:,:E]
    print(f"int{b}: direct {relL2(qrow(X,b)):.4e}  hadamard+quant {float(np.linalg.norm(X-Xrec)/fro):.4e}")
