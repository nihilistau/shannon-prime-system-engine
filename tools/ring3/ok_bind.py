#!/usr/bin/env python3
# ok_bind.py -- the Ring-3 VSA bind/unbind, executed on the NATIVE exact-integer
# dual-prime negacyclic CRT-NTT C engine (core/poly_ring sp_pr_mul via ctypes).
# Proven bit-identical to the numpy reference + reduction-order-immune in
# G-R3-BIND-on-O_K (Leg A). No host float FFT anywhere on the bind/unbind path.
#
# Degree: the engine ring supports N in {128,256,512}. A logical dim D is tiled as
# a DIRECT SUM of <=512 negacyclic blocks (D=1024 = 512 (+) 512), each bound with
# the exact engine product -- so capacity tracks total D while every coefficient is
# exact integer. Carriers/ids are +/-1 (substrate-native): the encode boundary is
# the identity, lossless (Leg A).
import os, ctypes as C, numpy as np
LIB=os.environ.get("SP_R3_LIB","/tmp/spbuild/libsp.so")
_lib=None; _ctx={}
def _load():
    global _lib
    if _lib is None:
        _lib=C.CDLL(LIB)
        _lib.sp_pr_init.restype=C.c_void_p; _lib.sp_pr_init.argtypes=[C.c_uint32]
        _lib.sp_pr_mul.argtypes=[C.c_void_p,C.POINTER(C.c_int32),C.POINTER(C.c_int32),C.POINTER(C.c_int64)]
    return _lib
def _ctxfor(N):
    _load()
    if N not in _ctx:
        c=_lib.sp_pr_init(N)
        if not c: raise RuntimeError(f"sp_pr_init({N}) failed (N must be 128/256/512)")
        _ctx[N]=c
    return _ctx[N]
def _blocks(D):
    if D in (128,256,512): return [D]
    if D==1024: return [512,512]
    bl=[]; r=D
    while r>=512: bl.append(512); r-=512
    if r in (128,256,512): bl.append(r)
    elif r>0: raise ValueError(f"D={D} not tileable into {{128,256,512}} blocks")
    return bl
def _negmul(a,b,N):  # exact negacyclic product over Z[x]/(x^N+1) via the native engine
    lib=_load(); ctx=_ctxfor(N)
    ai=np.ascontiguousarray(a,dtype=np.int32); bi=np.ascontiguousarray(b,dtype=np.int32)
    o=np.zeros(N,dtype=np.int64)
    lib.sp_pr_mul(ctx, ai.ctypes.data_as(C.POINTER(C.c_int32)),
                  bi.ctypes.data_as(C.POINTER(C.c_int32)), o.ctypes.data_as(C.POINTER(C.c_int64)))
    return o
def _involute(a,N):
    a=np.asarray(a); out=np.empty(N,dtype=a.dtype); out[0]=a[0]; out[1:]=-a[N-1:0:-1]; return out
def bind(addr,idv):     # M-contribution = addr (x) id   (negacyclic, native)
    addr=np.asarray(addr); idv=np.asarray(idv); D=addr.shape[0]; out=np.zeros(D,dtype=np.int64); o=0
    for N in _blocks(D): out[o:o+N]=_negmul(addr[o:o+N],idv[o:o+N],N); o+=N
    return out
def unbind(M,addr):     # est = M (x) addr*   (negacyclic correlation, native)
    M=np.asarray(M); addr=np.asarray(addr); D=addr.shape[0]; out=np.zeros(D,dtype=np.int64); o=0
    for N in _blocks(D): out[o:o+N]=_negmul(M[o:o+N],_involute(addr[o:o+N],N),N); o+=N
    return out
def carrier(seed,D): return (np.random.default_rng(int(seed)%(2**63)).integers(0,2,D)*2-1).astype(np.int64)
def idvec(seed,D):   return (np.random.default_rng((int(seed)^0xABCDEF)%(2**63)).integers(0,2,D)*2-1).astype(np.int64)
def cos(a,b):
    a=np.asarray(a,dtype=np.float64); b=np.asarray(b,dtype=np.float64)
    return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
