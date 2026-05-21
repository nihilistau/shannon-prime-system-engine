#!/usr/bin/env python3
"""Emit the byte-escaped C fixture rows for the TOK_ENCODE parity cases, from
the validated oracle probes. Throwaway generator (run by hand when probes change).
  python -X utf8 tools/oracle/gen_encode_fixture.py"""
import os, struct

def read_ids(path):
    b = open(path, 'rb').read()
    magic, nt, nv = struct.unpack_from('<III', b, 0)
    return list(struct.unpack_from('<%di' % nt, b, 12))

D = os.environ.get('SP_PROBE_DIR', r'C:\Users\Knack\AppData\Local\Temp\probes')
cases = [
    ('ml',       '价格 is 価格 — naïve €5,00. 中文测试 हिन्दी', os.path.join(D, 'ml.bin')),
    ('digits',   'abc123def 4567 x0',                          os.path.join(D, 'digits.bin')),
    ('ws',       'a  b   c',                                   os.path.join(D, 'ws.bin')),
    ('specials', '<|im_start|>user\nHello<|im_end|>',          os.path.join(D, 'specials.bin')),
]

def esc(b):
    return ''.join('\\x%02x' % c for c in b)

for name, p, rp in cases:
    ids = read_ids(rp)
    enc = p.encode('utf-8')
    print('    { "%s", "%s", %d,' % (name, esc(enc), len(ids)))
    print('      (const int32_t[]){ %s } },' % ', '.join(map(str, ids)))
