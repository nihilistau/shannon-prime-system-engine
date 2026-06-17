#!/usr/bin/env python3
# G-R3-BIND-on-O_K (Leg A) -- swap the float-FFT VSA bind for the engine-native
# exact-integer NEGACYCLIC dual-prime CRT-NTT (frozen primes q1=1073738753,
# q2=1073732609, M=q1*q2). Prove three things, falsifiably:
#  (1) C-ENGINE PARITY: the numpy-int negacyclic algebra is BIT-IDENTICAL to the
#      native libsp_poly_ring/libsp_ntt_crt primitives reached by ctypes --
#      sp_pr_mul, ntt_forward o pointwise o inverse, sp_pr_inner, and
#      sp_pr_score_kstore(encode(k)). (the EXACTNESS CONTRACT in the headers)
#  (2) MARGIN PARITY: integer bind recalls the correct id with margin>0, matching
#      the float negacyclic reference at the same degree. For the substrate-native
#      +/-1 carrier the encode boundary is LOSSLESS (int recall == float recall).
#  (3) REDUCTION-ORDER IMMUNITY: the integer superposition M is BYTE-IDENTICAL
#      under permuted episode-summation order; the float real carrier M is NOT
#      (floating-point addition is non-associative).
import os, sys, ctypes as C, numpy as np

N = 512
M_MOD = 1152908312643096577
ENG = os.environ.get("SP_R3_ENG", os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","..")))
LIB = os.environ.get("SP_R3_LIB", "/tmp/spbuild/libsp.so")

# ---- the float baseline's content-derived addressing (reused verbatim) ----
SEED=0x5350524F4A2B; R_BITS=256; HD=512; NL,PERIOD=48,6; MASK64=(1<<64)-1
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

# ---- carriers ----
def pm1(seed):
    rng=np.random.default_rng(seed%(2**63)); return (rng.integers(0,2,N)*2-1).astype(np.int64)
def unitary_real(seed):
    rng=np.random.default_rng(seed%(2**63))
    th=rng.uniform(0,2*np.pi,N//2+1); F=np.exp(1j*th); F[0]=1.0; F[-1]=1.0
    return np.fft.irfft(F,n=N)
def idvec(seed):
    rng=np.random.default_rng((seed^0xABCDEF)%(2**63)); return (rng.integers(0,2,N)*2-1).astype(np.int64)

# ---- exact integer negacyclic algebra (ground truth: Z[x]/(x^N+1)) ----
def negmul_int(a,b):
    a=np.asarray(a,dtype=np.int64); b=np.asarray(b,dtype=np.int64)
    lin=np.convolve(a,b)                       # 2N-1, exact int64 for our ranges
    out=lin[:N].copy(); out[:N-1]-=lin[N:2*N-1]
    return out
def negmul_float(a,b):
    a=np.asarray(a,dtype=np.float64); b=np.asarray(b,dtype=np.float64)
    lin=np.convolve(a,b); out=lin[:N].copy(); out[:N-1]-=lin[N:2*N-1]; return out
def involute(a):
    a=np.asarray(a); out=np.empty(N,dtype=a.dtype); out[0]=a[0]; out[1:]=-a[N-1:0:-1]; return out

DELTA=1<<14
def enc(v): return np.rint(np.asarray(v,dtype=np.float64)*DELTA).astype(np.int64)
def cos(a,b):
    a=np.asarray(a,dtype=np.float64); b=np.asarray(b,dtype=np.float64)
    return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))

# ============================ (1) C-ENGINE PARITY ============================
def c_parity():
    lib=C.CDLL(LIB)
    lib.ntt_init.restype=C.c_void_p; lib.ntt_init.argtypes=[C.c_uint32]
    lib.sp_pr_init.restype=C.c_void_p; lib.sp_pr_init.argtypes=[C.c_uint32]
    lib.ntt_forward.argtypes=[C.c_void_p,C.POINTER(C.c_int32),C.POINTER(C.c_uint32),C.POINTER(C.c_uint32)]
    lib.ntt_inverse.argtypes=[C.c_void_p,C.POINTER(C.c_uint32),C.POINTER(C.c_uint32),C.POINTER(C.c_int64)]
    lib.ntt_pointwise_mul.argtypes=[C.c_void_p]+[C.POINTER(C.c_uint32)]*6
    lib.sp_pr_mul.argtypes=[C.c_void_p,C.POINTER(C.c_int32),C.POINTER(C.c_int32),C.POINTER(C.c_int64)]
    lib.sp_pr_inner.restype=C.c_int64; lib.sp_pr_inner.argtypes=[C.c_void_p,C.POINTER(C.c_int32),C.POINTER(C.c_int32)]
    lib.sp_pr_kstore_encode.argtypes=[C.c_void_p,C.POINTER(C.c_int32),C.POINTER(C.c_uint32)]
    lib.sp_pr_query_begin.argtypes=[C.c_void_p,C.POINTER(C.c_int32)]
    lib.sp_pr_score_kstore.restype=C.c_int64; lib.sp_pr_score_kstore.argtypes=[C.c_void_p,C.POINTER(C.c_uint32)]
    nctx=lib.ntt_init(N); pctx=lib.sp_pr_init(N)
    assert nctx and pctx, "ctx alloc failed"
    def i32(x): return np.ascontiguousarray(x,dtype=np.int32)
    def u32(n): return np.zeros(n,dtype=np.uint32)
    def i64(n): return np.zeros(n,dtype=np.int64)
    p32=lambda x:x.ctypes.data_as(C.POINTER(C.c_int32))
    pu =lambda x:x.ctypes.data_as(C.POINTER(C.c_uint32))
    p64=lambda x:x.ctypes.data_as(C.POINTER(C.c_int64))
    rng=np.random.default_rng(12345); checks=0; fails=0; firstfail=None
    for t in range(64):
        a=i32(rng.integers(-16384,16385,N)); b=i32(rng.integers(-16384,16385,N))
        ref=negmul_int(a,b)
        # sp_pr_mul
        o=i64(N); lib.sp_pr_mul(pctx,p32(a),p32(b),p64(o)); checks+=1
        if not np.array_equal(o,ref): fails+=1; firstfail=firstfail or ("sp_pr_mul",t)
        # ntt forward o pointwise o inverse (the deployment bind path)
        r1a=u32(N);r2a=u32(N);r1b=u32(N);r2b=u32(N)
        lib.ntt_forward(nctx,p32(a),pu(r1a),pu(r2a)); lib.ntt_forward(nctx,p32(b),pu(r1b),pu(r2b))
        o1=u32(N);o2=u32(N); lib.ntt_pointwise_mul(nctx,pu(r1a),pu(r2a),pu(r1b),pu(r2b),pu(o1),pu(o2))
        inv=i64(N); lib.ntt_inverse(nctx,pu(o1),pu(o2),p64(inv)); checks+=1
        if not np.array_equal(inv,ref): fails+=1; firstfail=firstfail or ("ntt_fwd_pw_inv",t)
        # sp_pr_inner  &  sp_pr_score_kstore(encode(k))
        q=i32(rng.integers(-1000,1001,N)); k=i32(rng.integers(-1000,1001,N))
        dot=int(np.dot(q.astype(np.int64),k.astype(np.int64)))
        inn=lib.sp_pr_inner(pctx,p32(q),p32(k)); checks+=1
        if inn!=dot: fails+=1; firstfail=firstfail or ("sp_pr_inner",t)
        kres=u32(2*N); lib.sp_pr_kstore_encode(pctx,p32(k),pu(kres))
        lib.sp_pr_query_begin(pctx,p32(q)); sc=lib.sp_pr_score_kstore(pctx,pu(kres)); checks+=1
        if sc!=dot: fails+=1; firstfail=firstfail or ("score_kstore",t)
    return checks,fails,firstfail

# ============================ (2) MARGIN PARITY =============================
def make(mode,seeds):
    if mode=="int_pm1":   av=[pm1(s) for s in seeds];                 ids=[idvec(s) for s in seeds];                  mul=negmul_int
    elif mode=="flt_pm1": av=[pm1(s).astype(np.float64) for s in seeds]; ids=[idvec(s).astype(np.float64) for s in seeds]; mul=negmul_float
    elif mode=="int_uni": av=[enc(unitary_real(s)) for s in seeds];   ids=[idvec(s) for s in seeds];                  mul=negmul_int
    elif mode=="flt_uni": av=[unitary_real(s) for s in seeds];        ids=[idvec(s).astype(np.float64) for s in seeds]; mul=negmul_float
    return av,ids,mul
def run(mode,Nep,rs):
    rng=np.random.default_rng(777+Nep); seeds=list(rs)+[int(rng.integers(1,2**62)) for _ in range(Nep-2)]
    av,ids,mul=make(mode,seeds)
    M=mul(av[0],ids[0])
    for i in range(1,Nep): M=M+mul(av[i],ids[i])
    h1=h5=0; mar=[]
    for j in range(Nep):
        est=mul(M,involute(av[j])); sims=np.array([cos(est,ids[k]) for k in range(Nep)]); o=np.argsort(-sims)
        if o[0]==j:h1+=1
        if j in o[:5]:h5+=1
        ot=np.delete(sims,j); mar.append(float(sims[j]-ot.max()))
    return h1/Nep,h5/Nep,mar

# ====================== (3) REDUCTION-ORDER IMMUNITY =======================
def immunity(rs):
    Nep=16; rng=np.random.default_rng(777+Nep); seeds=list(rs)+[int(rng.integers(1,2**62)) for _ in range(Nep-2)]
    # integer real (unitary) carrier -- the case where float WOULD round
    ai=[enc(unitary_real(s)) for s in seeds]; di=[idvec(s) for s in seeds]
    ib=[negmul_int(ai[i],di[i]) for i in range(Nep)]
    ibase=np.zeros(N,dtype=np.int64)
    for b in ib: ibase=ibase+b
    int_ok=True
    for p in range(8):
        acc=np.zeros(N,dtype=np.int64)
        for idx in np.random.default_rng(p).permutation(Nep): acc=acc+ib[idx]
        if not np.array_equal(acc,ibase): int_ok=False
    # float real carrier -- non-associative
    fa=[unitary_real(s) for s in seeds]; fi=[idvec(s).astype(np.float64) for s in seeds]
    fb=[negmul_float(fa[i],fi[i]) for i in range(Nep)]
    fbase=np.zeros(N)
    for b in fb: fbase=fbase+b
    maxd=0.0
    for p in range(8):
        acc=np.zeros(N)
        for idx in np.random.default_rng(p).permutation(Nep): acc=acc+fb[idx]
        maxd=max(maxd,float(np.max(np.abs(acc-fbase))))
    return int_ok, maxd

def main():
    print(f"[r3-ok] N={N}  M={M_MOD}  DELTA={DELTA}  LIB={LIB}",flush=True)
    real=[("ep_toy",f"{ENG}/_p33_ep",16),("ep_wiki",f"{ENG}/_c2_ep_wiki",84)]
    rs=[ep_sig_seed(d,n) for _,d,n in real]
    print(f"[r3-ok] real C2 sig-seeds: ep_toy={hex(rs[0])[:12]}.. ep_wiki={hex(rs[1])[:12]}..",flush=True)

    checks,fails,ff=c_parity()
    print(f"\n=== (1) C-ENGINE PARITY (ctypes -> libsp_poly_ring/libsp_ntt_crt) ===",flush=True)
    print(f"  checks={checks} fails={fails} firstfail={ff}",flush=True)
    print(f"  -> numpy-int negacyclic algebra {'BIT-IDENTICAL' if fails==0 else 'DIVERGES FROM'} native sp_pr_mul / ntt_fwd.pw.inv / sp_pr_inner / score_kstore",flush=True)

    print(f"\n=== (2) MARGIN PARITY (recall@1 / recall@5 / min-margin) @ deg N={N} ===",flush=True)
    print(f"{'mode':>8}{'Nep':>5} | {'rec@1':>6} {'rec@5':>6} | {'min_margin':>10} | toy_mgn  wiki_mgn",flush=True)
    res={}
    for mode in ("int_pm1","flt_pm1","int_uni","flt_uni"):
        for Nep in (2,8,16,32):
            r1,r5,m=run(mode,Nep,rs); res[(mode,Nep)]=(r1,r5,m)
            print(f"{mode:>8}{Nep:>5} | {r1:>6.3f} {r5:>6.3f} | {min(m):>10.4f} | {m[0]:>7.4f} {m[1]:>7.4f}",flush=True)
    pm1_lossless=all(abs(res[("int_pm1",Nep)][0]-res[("flt_pm1",Nep)][0])<1e-12 and
                     abs(res[("int_pm1",Nep)][1]-res[("flt_pm1",Nep)][1])<1e-12 for Nep in (2,8,16,32))
    n2=(res[("int_pm1",2)][0]==1.0 and res[("int_pm1",2)][2][0]>0 and res[("int_pm1",2)][2][1]>0)
    print(f"  -> +/-1 carrier int==float recall (lossless encode): {pm1_lossless}",flush=True)
    print(f"  -> N=2 ep_toy+ep_wiki int recall@1=1, both margins>0: {n2}",flush=True)

    int_ok,fmaxd=immunity(rs)
    print(f"\n=== (3) REDUCTION-ORDER IMMUNITY (16 episodes, 8 permutations) ===",flush=True)
    print(f"  integer M byte-identical across all permutations: {int_ok}",flush=True)
    print(f"  float real-carrier M max|diff| across permutations: {fmaxd:.6e}  (non-associative: {fmaxd>0})",flush=True)

    green = (fails==0) and pm1_lossless and n2 and int_ok and (fmaxd>0)
    print(f"\n[gate] G-R3-BIND-on-O_K (Leg A) {'GREEN' if green else 'RED'} -- "
          + ("exact-integer negacyclic CRT-NTT bind is bit-identical to the native engine primitives, "
             "recalls the correct episode id with margin>0 matching the float baseline (encode lossless on the +/-1 substrate carrier), "
             "and yields a reduction-order-immune (byte-identical) superposition the float lane cannot" if green else "see failing leg above"),flush=True)
    return 0 if green else 1

if __name__=="__main__":
    sys.exit(main())
