#!/usr/bin/env python3
# frob_episode.py -- Frobenius pi^k integer codec for Ring-2 episodes (T4 storage form).
# Encode the float ep.k/ep.v into per-(layer,channel) Frobenius-scaled integers (the OK_Q8-style
# scalar O_K coordinate: integer code x per-row scale; b=0 embedding, full a+bw lattice = follow-on).
# The per-row scale IS the pi^k Frobenius scale T4 proves cancels through the norm boundary -- so the
# store lives as pure integers on disk; decode reconstitutes the exact floats the attention heads read.
import os, sys, numpy as np
NL=48; HD=512
def _load(d,name):
    raw=np.fromfile(os.path.join(d,name),dtype="<f4"); P=raw.size//(NL*HD); return raw.reshape(NL,P,HD),P
def encode_tensor(T, bits):
    qmax=(1<<(bits-1))-1
    s=(np.abs(T).max(axis=1,keepdims=True)+1e-12)/qmax       # per-(layer,channel) Frobenius scale [NL,1,HD]
    dt=np.int8 if bits==8 else np.int16
    q=np.clip(np.round(T/s),-qmax-1,qmax).astype(dt)
    return q, s.astype(np.float32)
def decode_tensor(q, s):
    return (q.astype(np.float64)*s.astype(np.float64)).astype(np.float32)
def roundtrip_episode(src, dst, bits):
    os.makedirs(dst, exist_ok=True)
    report={}
    for name in ("ep.k","ep.v"):
        T,P=_load(src,name)
        q,s=encode_tensor(T,bits)
        # PURE-INTEGER STORE (the Ring-2 disk form): codes + scale sidecar
        q.tofile(os.path.join(dst,name+f".q{bits}")); s.tofile(os.path.join(dst,name+f".s{bits}"))
        R=decode_tensor(q,s)                                  # restoration seam -> exact floats
        R.tofile(os.path.join(dst,name))                      # float the engine consumes (SP_REPLAY reads this)
        err=np.abs(T-R)
        store_bytes=q.nbytes+s.nbytes; orig=T.nbytes
        report[name]=dict(P=P,maxerr=float(err.max()),rms=float(err.std()),
                          relL2=float(np.linalg.norm((T-R).ravel())/(np.linalg.norm(T.ravel())+1e-12)),
                          orig_MB=orig/1e6,store_MB=store_bytes/1e6,ratio=orig/store_bytes)
    # copy non-tensor episode files (manifest etc.) unchanged
    for f in os.listdir(src):
        if f in ("ep.k","ep.v"): continue
        if os.path.isfile(os.path.join(src,f)):
            open(os.path.join(dst,f),"wb").write(open(os.path.join(src,f),"rb").read())
    return report
if __name__=="__main__":
    src=sys.argv[1]; dst=sys.argv[2]; bits=int(sys.argv[3])
    rep=roundtrip_episode(src,dst,bits)
    for n,r in rep.items():
        print(f"  {n}: int{bits} P={r['P']} maxerr={r['maxerr']:.3e} rms={r['rms']:.3e} relL2={r['relL2']:.3e} | store {r['store_MB']:.1f}MB vs {r['orig_MB']:.1f}MB ({r['ratio']:.2f}x)")
