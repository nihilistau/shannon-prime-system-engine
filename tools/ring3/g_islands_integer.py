#!/usr/bin/env python3
# G-BYTEEXACT-ISLANDS: exact-integer prototypes for the softmax + GELU fp32-islands (companion to
# g_norm_integer.py). Shared primitive = fixed-point exp via 2^x (integer poly, coeffs (ln2)^k/k!).
# Proves ~1e-6 fidelity vs float and byte-exactness (the sums are reduction-order-immune; the per-element
# transcendentals are deterministic integer functions). The byte-exact forward's float islands, closed.
import numpy as np, math, random
FB=30; ONE=1<<FB; LOG2E=int(round(math.log2(math.e)*ONE))
C=[int(round((math.log(2)**k/math.factorial(k))*ONE)) for k in range(7)]
def exp2_frac(r):
    acc=C[6]
    for k in (5,4,3,2,1,0): acc=(acc*r>>FB)+C[k]
    return acc
def exp_fixed(d):                                  # e^d, d<=0 (FB-fixed) -> FB-fixed
    g=-(d*LOG2E>>FB); n=g>>FB; r=g-(n<<FB)
    return exp2_frac(ONE-r)>>(n+1) if r else (ONE>>n)
def softmax_int(zf, Z=1<<14):
    zi=np.round(zf*Z).astype(np.int64); m=int(zi.max())
    e=[exp_fixed((int(z)-m)*ONE//Z) for z in zi]; S=sum(e)
    return np.array([ei/S for ei in e]), e, S
def tanh_fixed(t):
    s=1 if t>=0 else -1; a=abs(t); e2=exp_fixed(-(2*a))
    return s*(ONE-((2*e2<<FB)//(ONE+e2)))
K=int(round(math.sqrt(2/math.pi)*ONE)); A=int(round(0.044715*ONE))
def gelu_int(xf, Z=1<<16):
    out=[]
    for xq in np.round(xf*Z).astype(np.int64):
        x=int(xq)*ONE//Z; x3=(((x*x>>FB)*x)>>FB); inner=(K*(x+(A*x3>>FB))>>FB)
        out.append(((x>>1)*(ONE+tanh_fixed(inner))>>FB)/ONE)
    return np.array(out)
if __name__=="__main__":
    rng=np.random.default_rng(0)
    z=rng.standard_normal(256)*4; pf=np.exp(z-z.max()); pf/=pf.sum(); pi,e,S=softmax_int(z)
    perms=[list(range(256))]+[random.Random(s).sample(range(256),256) for s in range(4)]
    print(f"softmax: max|dp| {np.abs(pf-pi).max():.2e} KL {np.sum(pf*np.log((pf+1e-30)/(pi+1e-30))):.2e} | sum order-immune: {len({sum(e[i] for i in pm) for pm in perms})==1}")
    x=rng.standard_normal(512)*3; gf=0.5*x*(1+np.tanh(math.sqrt(2/math.pi)*(x+0.044715*x**3)))
    print(f"gelu: relerr {np.linalg.norm(gf-gelu_int(x))/np.linalg.norm(gf):.2e}")
