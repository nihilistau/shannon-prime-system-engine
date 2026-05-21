#!/usr/bin/env python3
"""Minimal GGUF metadata reader — extracts tokenizer.ggml.* and decodes ref IDs.
Throwaway prototyping aid for the tokenizer-encode bring-up (not engine code)."""
import struct, sys

GGUF_TYPES = {0:'u8',1:'i8',2:'u16',3:'i16',4:'u32',5:'i32',6:'f32',7:'bool',
              8:'str',9:'arr',10:'u64',11:'i64',12:'f64'}
SCALAR_FMT = {0:'<B',1:'<b',2:'<H',3:'<h',4:'<I',5:'<i',6:'<f',7:'<B',10:'<Q',11:'<q',12:'<d'}
SCALAR_SZ  = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}

class R:
    def __init__(self, b): self.b=b; self.p=0
    def u32(self): v=struct.unpack_from('<I',self.b,self.p)[0]; self.p+=4; return v
    def u64(self): v=struct.unpack_from('<Q',self.b,self.p)[0]; self.p+=8; return v
    def s(self):
        n=self.u64(); v=self.b[self.p:self.p+n]; self.p+=n; return v
    def scalar(self,t):
        v=struct.unpack_from(SCALAR_FMT[t],self.b,self.p)[0]; self.p+=SCALAR_SZ[t]; return v

def read_value(r,t):
    if t==8: return r.s()
    if t==9:
        at=r.u32(); n=r.u64()
        if at==8: return [r.s() for _ in range(n)]
        return [r.scalar(at) for _ in range(n)]
    return r.scalar(t)

def parse(path):
    b=open(path,'rb').read()
    r=R(b)
    assert b[:4]==b'GGUF', 'bad magic'
    r.p=4
    ver=r.u32(); nt=r.u64(); nkv=r.u64()
    kv={}
    for _ in range(nkv):
        key=r.s().decode('utf-8'); t=r.u32(); kv[key]=read_value(r,t)
    return ver,nt,nkv,kv

if __name__=='__main__':
    path=sys.argv[1] if len(sys.argv)>1 else r'D:\Files\models\Qwen\Qwen3-0.6B-GGUF\Qwen3-0.6B-f16.gguf'
    ver,nt,nkv,kv=parse(path)
    print(f'version={ver} tensors={nt} kv={nkv}')
    for k in ('tokenizer.ggml.model','tokenizer.ggml.pre','tokenizer.ggml.add_bos_token',
              'tokenizer.ggml.add_eos_token','tokenizer.ggml.bos_token_id',
              'tokenizer.ggml.eos_token_id'):
        if k in kv: print(f'  {k} = {kv[k]!r}')
    toks=kv.get('tokenizer.ggml.tokens')
    merges=kv.get('tokenizer.ggml.merges')
    ttype=kv.get('tokenizer.ggml.token_type')
    print(f'  n_tokens={len(toks) if toks else 0} n_merges={len(merges) if merges else 0}')
    if merges: print('  first merges:', [m.decode() for m in merges[:5]])
    # decode the ref IDs
    ids=[785, 10250, 8168, 2022, 315, 458, 7546, 374, 279, 2745, 49189, 315, 49433,
         6693, 1985, 374, 429, 7546, 26, 419, 54272, 315, 49368, 3147, 10163, 279,
         5810, 5109, 553, 43492, 13]
    print('  pieces:', [toks[i].decode('utf-8','replace') for i in ids])
