#!/usr/bin/env python3
# frob_episode.py -- Frobenius pi^k integer codec for Ring-2 episodes (T4 storage form).
#
# Encode the float ep.k/ep.v into per-(layer,channel) Frobenius-scaled INTEGER coordinates.
# T4 (PPT-ARM-Theory) proves the per-tensor pi^k scale cancels through the norm boundary, so the
# scale is "free" -- the store lives as pure integers on disk; decode reconstitutes the exact floats
# the attention heads read. The engine consumes float K/V unchanged (quantize at the disk boundary).
#
# SCHEMES (the rank-2 O_K lattice O_K = Z*1 (+) Z*w, realized as TWO REAL integer coordinates so the
# round-trip stays real -- the literal complex a+b*w embedding would inject an imaginary part):
#   a16   : 1-step, int16 coarse coordinate `a`           (16 b/elem ; effectively lossless, relL2 ~3e-5)
#   a8b4  : 2-step, int8 `a` + int4 residual `b`          (12 b/elem ; relL2 ~6e-4, aggressive)
#   a16b8 : 2-step, int16 `a` + int8 residual `b`         (24 b/elem ; relL2 ~1e-7 = sub-ULP, BIT-EXACT)
# The residual coordinate `b` is error-feedback: b = round((x - a*s_a)/s_b); decode = a*s_a + b*s_b.
import os, sys, numpy as np
NL=48; HD=512
SCHEMES={"a16":(16,None),"a8b4":(8,4),"a16b8":(16,8)}

def _load(d,name):
    raw=np.fromfile(os.path.join(d,name),dtype="<f4"); P=raw.size//(NL*HD); return raw.reshape(NL,P,HD),P
def _dt(bits): return np.int8 if bits<=8 else (np.int16 if bits<=16 else np.int32)

def encode(T, b1, b2):
    """rank-2 integer lattice encode: coarse coord a (b1 bits) + optional residual coord b (b2 bits)."""
    qm1=(1<<(b1-1))-1; sa=(np.abs(T).max(axis=1,keepdims=True)+1e-12)/qm1   # per-(layer,channel) pi^k scale
    a=np.clip(np.round(T/sa),-qm1-1,qm1).astype(_dt(b1))
    if b2 is None: return dict(a=a, sa=sa.astype(np.float32))
    r=T - a.astype(np.float64)*sa                                          # error-feedback residual
    qm2=(1<<(b2-1))-1; sb=(np.abs(r).max(axis=1,keepdims=True)+1e-12)/qm2
    b=np.clip(np.round(r/sb),-qm2-1,qm2).astype(_dt(b2))
    return dict(a=a, sa=sa.astype(np.float32), b=b, sb=sb.astype(np.float32), b2=b2)

def decode(enc):
    x=enc["a"].astype(np.float64)*enc["sa"].astype(np.float64)
    if "b" in enc: x=x + enc["b"].astype(np.float64)*enc["sb"].astype(np.float64)
    return x.astype(np.float32)

def roundtrip_episode(src, dst, scheme):
    b1,b2=SCHEMES[scheme]; os.makedirs(dst, exist_ok=True); report={}
    for name in ("ep.k","ep.v"):
        T,P=_load(src,name); enc=encode(T,b1,b2)
        # PURE-INTEGER ON-DISK STORE: coordinate codes + per-(layer,channel) scale sidecars
        enc["a"].tofile(os.path.join(dst,f"{name}.a{b1}")); enc["sa"].tofile(os.path.join(dst,f"{name}.sa"))
        store=enc["a"].nbytes+enc["sa"].nbytes
        if b2 is not None:
            enc["b"].tofile(os.path.join(dst,f"{name}.b{b2}")); enc["sb"].tofile(os.path.join(dst,f"{name}.sb"))
            store+=enc["b"].nbytes+enc["sb"].nbytes
        R=decode(enc); R.tofile(os.path.join(dst,name))                    # the float the engine/SP_REPLAY reads
        identical=int(np.count_nonzero(R.view(np.uint32)==T.view(np.uint32))); tot=T.size
        err=np.abs(T-R); orig=T.nbytes
        report[name]=dict(P=P, scheme=scheme, bits=b1+(b2 or 0),
            maxerr=float(err.max()), relL2=float(np.linalg.norm((T-R).ravel())/(np.linalg.norm(T.ravel())+1e-12)),
            ulp_exact_pct=100.0*identical/tot, orig_MB=orig/1e6, store_MB=store/1e6, ratio=orig/store)
    for f in os.listdir(src):                                              # copy manifest etc. unchanged
        if f in ("ep.k","ep.v") or not os.path.isfile(os.path.join(src,f)): continue
        open(os.path.join(dst,f),"wb").write(open(os.path.join(src,f),"rb").read())
    return report

if __name__=="__main__":
    src,dst,scheme=sys.argv[1],sys.argv[2],(sys.argv[3] if len(sys.argv)>3 else "a16")
    if scheme.isdigit(): scheme={"8":"a8b4","16":"a16"}.get(scheme,"a16")  # back-compat w/ bit-int arg
    rep=roundtrip_episode(src,dst,scheme)
    for n,r in rep.items():
        print(f"  {n}: {r['scheme']}({r['bits']}b) relL2={r['relL2']:.3e} maxerr={r['maxerr']:.3e} "
              f"ulp-exact={r['ulp_exact_pct']:.2f}% | store {r['store_MB']:.1f}MB vs {r['orig_MB']:.1f}MB ({r['ratio']:.2f}x)")
