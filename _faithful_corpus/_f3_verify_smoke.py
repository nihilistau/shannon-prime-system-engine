import json, struct, collections, sys
d = sys.argv[1] if len(sys.argv) > 1 else '_faithful_corpus/f3/A'
metas = [json.loads(l) for l in open(d + '/f3_meta.jsonl', encoding='utf-8')]
byu = collections.defaultdict(list)
for m in metas: byu[m['user']].append(m)
print('items:', len(byu), 'rows:', len(metas))
allid = True
for u, ms in byu.items():
    h = open(f"{d}/f3_{ms[0]['chat_id']}.bin", 'rb').read(16)
    magic = h[:4]; e, nf, pad = struct.unpack('<3I', h[4:16])
    line = f"{u[:40]!r:44} mode={ms[0]['mode']:10} rec={bool(ms[0]['recalled'])} hdr={magic}/{e}/{nf}"
    if len(ms) == 2:
        p0 = open(f"{d}/f3_{ms[0]['chat_id']}.bin", 'rb').read()[16:]
        p1 = open(f"{d}/f3_{ms[1]['chat_id']}.bin", 'rb').read()[16:]
        ident = p0 == p1; allid &= ident
        line += f" rerun_identical={ident}"
    line += f" ans={ms[0]['answer'][:32]!r}"
    print(line)
print('DETERMINISM:', 'PASS' if allid else 'FAIL')
