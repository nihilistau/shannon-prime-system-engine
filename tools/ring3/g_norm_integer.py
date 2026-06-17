#!/usr/bin/env python3
# G-NORM-INTEGER: exact-integer RMSNorm prototype -- the keystone fp32-island replacement for the
# byte-exact forward. y = x * sqrt(E/sum x^2) * (1+w), all bigint fixed-point: Sigma x^2 exact (int64),
# 1/sqrt via exact integer isqrt (deterministic), (1+w) in fixed-point. Proves (a) ~1e-5 fidelity vs the
# float RMSNorm and (b) reduction-order immunity (byte-exact), which the float sp_rmsnorm_bridge lacks.
import numpy as np, math, random
def int_rmsnorm(x, w, Q=16, IB=20, Qw=16):
    E=len(x); xi=np.round(x*(1<<Q)).astype(np.int64)
    sumsq=int(np.sum(xi.astype(object)**2))                    # EXACT
    inv=math.isqrt((E<<(2*(Q+IB)))//sumsq)                     # round(2^IB * sqrt(E/sum x^2)), EXACT
    wi=np.round((1.0+w)*(1<<Qw)).astype(np.int64)
    yi=[int(xi[i])*inv*int(wi[i]) for i in range(E)]           # 2^(Q+IB+Qw) fixed-point, fully integer
    return np.array(yi,dtype=np.float64)/float(1<<(Q+IB+Qw)), xi
if __name__=="__main__":
    E=3840; rng=np.random.default_rng(0); x=rng.standard_normal(E)*0.7; w=rng.standard_normal(E)*0.1
    rms=math.sqrt(np.mean(x*x)); yf=(x/rms)*(1.0+w)
    yi,xi=int_rmsnorm(x,w)
    print(f"fidelity relerr {np.linalg.norm(yi-yf)/np.linalg.norm(yf):.3e}")
    perms=[list(range(E))]+[random.Random(s).sample(range(E),E) for s in range(4)]
    iset={int(np.sum(xi[pm].astype(object)**2)) for pm in perms}
    print(f"byte-exact: sum(x^2) {len(iset)} distinct over 5 orders (1=reduction-order-immune)")
