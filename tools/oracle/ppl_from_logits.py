#!/usr/bin/env python3
"""Compute perplexity from a dump_logits .bin over the second-half positions
(single window, no chunking) — the oracle cross-check for sp_perplexity at
n_ctx == n_tokens. Usage: ppl_from_logits.py <ref.bin>"""
import struct, math, array, sys

path = sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/ppl/wiki.tiny.ref.bin"
with open(path, "rb") as f:
    magic, nt, nv = struct.unpack("<III", f.read(12))
    ids = list(struct.unpack("<%di" % nt, f.read(4 * nt)))
    first = nt // 2
    nll = 0.0
    cnt = 0
    for p in range(nt):
        row = array.array("f")
        row.frombytes(f.read(4 * nv))
        if p < first or p >= nt - 1:
            continue
        tgt = ids[p + 1]
        m = max(row)
        s = sum(math.exp(x - m) for x in row)
        logp = row[tgt] - m - math.log(s)
        nll += -logp
        cnt += 1
print(f"oracle f16: nt={nt} first={first} scored={cnt} PPL={math.exp(nll/cnt):.5f}")
