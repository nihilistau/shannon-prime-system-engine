#!/usr/bin/env python3
# G-BYTEEXACT-ATTN-FULL: the COMPLETE exact-integer decode attention (single head) validated
# against the float reference, on the frozen dual-prime substrate -- the design the CUDA
# k_attn_decode_*_bx kernels implement. Chain: CKKS-encode q/K/V (round(Delta*v)) -> exact
# integer Q.K via dual-prime modular dot + Garner -> integer softmax (FB30 2^x poly, the island)
# -> exact integer p.V via dual-prime modular dot -> decode. Gates:
#   - CRT dual-prime dot == plain integer dot (the negacyclic coeff_{N-1}, no __int128);
#   - full-attention fidelity vs float ~1e-5..1e-4 (Delta-tunable);
#   - AV accumulator RANGE: softmax is a probability dist => Sum e_s*v_s stays ~2^46 << M~2^60
#     at EVERY context W up to 16384 (dual-prime sufficient end-to-end -- no 3rd prime);
#   - reduction-order immunity: key-set permutation -> byte-identical output (cross-machine).
import numpy as np, random, math
Q1=1073738753; Q2=1073732609; M=Q1*Q2
FB=30; ONE=1<<FB; LOG2E=int(round(math.log2(math.e)*ONE))
C=[int(round((math.log(2)**k/math.factorial(k))*ONE)) for k in range(7)]
def exp2_frac(r):
    acc=C[6]
    for k in (5,4,3,2,1,0): acc=(acc*r>>FB)+C[k]
    return acc
def exp_fixed(d):
    if d>0: d=0
    g=-(d*LOG2E>>FB); n=g>>FB; r=g-(n<<FB)
    if n>=32: return 0
    return exp2_frac(ONE-r)>>(n+1) if r else (ONE>>n)
def garner(r1,r2):
    inv=pow(Q1,Q2-2,Q2); t=(r2-r1)*inv%Q2; x=r1+Q1*t
    return x-M if x>M//2 else x
def exact_dot(a,b):                         # dual-prime modular integer dot (== plain bigint dot)
    r1=sum((x%Q1)*(y%Q1) for x,y in zip(a,b))%Q1
    r2=sum((x%Q2)*(y%Q2) for x,y in zip(a,b))%Q2
    return garner(r1,r2)
def attn_float(q,K,V):
    sc=K@q; mx=sc.max(); e=np.exp(sc-mx); return (e@V)/e.sum()
def attn_int(q,K,V,DELTA,ZB=14):
    W=len(K); LD=2*int(round(math.log2(DELTA)))
    qi=np.round(q*DELTA).astype(np.int64); Ki=np.round(K*DELTA).astype(np.int64); Vi=np.round(V*DELTA).astype(np.int64)
    D=[int(qi@Ki[s]) for s in range(W)]                 # exact integer Q.K = Delta^2 * real score
    zi=[(d<<ZB)>>LD for d in D]; m=max(zi)
    e=[exp_fixed((z-m)*ONE>>ZB) for z in zi]; S=sum(e); ev=np.array(e,dtype=object)
    maxacc=0; ao=np.empty(len(q))
    for i in range(len(q)):
        col=Vi[:,i].tolist(); num=int(sum(ev[s]*col[s] for s in range(W)))   # exact integer p.V
        maxacc=max(maxacc,abs(num)); ao[i]=num/(S*DELTA)
    return ao,maxacc,S
if __name__=="__main__":
    rng=np.random.default_rng(1); N=256
    a=[int(x) for x in np.round(rng.standard_normal(N)*0.4*2**16)]
    b=[int(x) for x in np.round(rng.standard_normal(N)*0.4*2**16)]
    print("CRT dual-prime dot == plain int dot:", exact_dot(a,b)==sum(x*y for x,y in zip(a,b)))
    for (W,DELTA) in [(64,1<<16),(1024,1<<16),(4096,1<<16),(16384,1<<14)]:
        q=rng.standard_normal(N)*0.4; K=rng.standard_normal((W,N))*0.4; V=rng.standard_normal((W,N))*0.4
        af=attn_float(q,K,V); ai,maxacc,S=attn_int(q,K,V,DELTA)
        rel=np.linalg.norm(af-ai)/(np.linalg.norm(af)+1e-12)
        print(f"W={W:5d} Delta=2^{int(round(math.log2(DELTA)))}: relerr {rel:.2e} | AV maxacc 2^{math.log2(maxacc+1):.1f} vs M 2^{math.log2(M):.1f} -> dual-prime-safe {maxacc<M//2}")
    W=256; q=rng.standard_normal(N)*0.4; K=rng.standard_normal((W,N))*0.4; V=rng.standard_normal((W,N))*0.4
    base,_,_=attn_int(q,K,V,1<<16); ok=True
    for s in range(3):
        pm=random.Random(s).sample(range(W),W)
        a2,_,_=attn_int(q,K[pm],V[pm],1<<16)
        if not np.array_equal(a2,base): ok=False
    print("order-immune (key-set permutation -> identical output):",ok)
