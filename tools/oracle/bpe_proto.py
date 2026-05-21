#!/usr/bin/env python3
"""Prototype Qwen2 byte-level BPE encoder — proves the algorithm reproduces the
stock-llama.cpp token IDs (incl. specials + Unicode) before porting to C.
Throwaway parity oracle (not engine code). Run: python -X utf8 bpe_proto.py"""
import sys, struct, regex as re
sys.argv=['x']
exec(open(r'tools/oracle/gguf_peek.py').read().split('if __name__')[0])

MODEL = r'D:\Files\models\Qwen\Qwen3-0.6B-GGUF\Qwen3-0.6B-f16.gguf'
import os
PROBE = os.environ.get('SP_PROBE_DIR', r'C:\Users\Knack\AppData\Local\Temp\probes')

# Qwen2 pre-tokenizer regex (verbatim from llama.cpp LLAMA_VOCAB_PRE_TYPE_QWEN2)
QWEN2_RE = ("(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])"
            "|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+"
            "|\\p{N}"
            "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*"
            "|\\s*[\\r\\n]+"
            "|\\s+(?!\\S)"
            "|\\s+")
RX = re.compile(QWEN2_RE)

def byte_encoder():
    bs=list(range(33,127))+list(range(161,173))+list(range(174,256))
    cs=bs[:]; n=0
    for b in range(256):
        if b not in bs: bs.append(b); cs.append(256+n); n+=1
    return {b:chr(c) for b,c in zip(bs,cs)}

class Tok:
    def __init__(self, path):
        ver,nt,nkv,kv=parse(path)
        self.toks=[t.decode('utf-8') for t in kv['tokenizer.ggml.tokens']]
        self.ttype=kv.get('tokenizer.ggml.token_type')
        merges=[m.decode('utf-8') for m in kv['tokenizer.ggml.merges']]
        self.tok2id={t:i for i,t in enumerate(self.toks)}
        self.rank={}
        for i,m in enumerate(merges):
            a,b=m.split(' '); self.rank[(a,b)]=i
        self.b2u=byte_encoder()
        # special tokens: CONTROL(3) + USER_DEFINED(4), longest surface first
        self.specials=[]
        if self.ttype:
            for i,ty in enumerate(self.ttype):
                if ty in (3,4): self.specials.append((self.toks[i], i))
            self.specials.sort(key=lambda x:-len(x[0]))

    def _bpe(self, piece):
        syms=list(piece)
        while len(syms)>1:
            best=None; bi=-1
            for i in range(len(syms)-1):
                r=self.rank.get((syms[i],syms[i+1]))
                if r is not None and (best is None or r<best): best=r; bi=i
            if bi<0: break
            syms=syms[:bi]+[syms[bi]+syms[bi+1]]+syms[bi+2:]
        return syms

    def _encode_text(self, text, out):
        for m in RX.finditer(text):
            enc=''.join(self.b2u[b] for b in m.group().encode('utf-8'))
            for s in self._bpe(enc): out.append(self.tok2id[s])

    def encode(self, text, parse_special=True):
        out=[]
        frags=[(text,False)]
        if parse_special:
            for surf,tid in self.specials:
                nf=[]
                for frag,isid in frags:
                    if isid: nf.append((frag,isid)); continue
                    i=0
                    while True:
                        j=frag.find(surf,i)
                        if j<0: nf.append((frag[i:],False)); break
                        if j>i: nf.append((frag[i:j],False))
                        nf.append((tid,True)); i=j+len(surf)
                frags=nf
        for frag,isid in frags:
            if isid: out.append(frag)
            elif frag: self._encode_text(frag,out)
        return out

def read_ids(path):
    b=open(path,'rb').read()
    magic,nt,nv=struct.unpack_from('<III',b,0)
    return list(struct.unpack_from('<%di'%nt,b,12))

if __name__=='__main__':
    t=Tok(MODEL)
    cases=[
        ("main", "The prime factorization of an integer is the multiset of primes whose product is that integer; this lattice of divisibility orders the natural numbers by dominance.",
         r'D:\Files\models\Qwen\Qwen3-0.6B-GGUF\qwen3_ref.bin'),
        ("ml", "价格 is 価格 — naïve €5,00. 中文测试 हिन्दी", PROBE+r"\ml.bin"),
        ("digits", "abc123def 4567 x0", PROBE+r"\digits.bin"),
        ("ws", "a  b   c", PROBE+r"\ws.bin"),
        ("specials", "<|im_start|>user\nHello<|im_end|>", PROBE+r"\specials.bin"),
    ]
    allok=True
    for name,prompt,refpath in cases:
        try: ref=read_ids(refpath)
        except FileNotFoundError: print(f'{name}: SKIP (no {refpath})'); continue
        got=t.encode(prompt)
        ok=got==ref; allok&=ok
        print(f'{name}: {"MATCH" if ok else "MISMATCH"}  (got {len(got)} / ref {len(ref)})')
        if not ok:
            print('   got:',got); print('   ref:',ref)
    print('ALL MATCH' if allok else 'FAILURES PRESENT')
