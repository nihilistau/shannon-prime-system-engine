"""rank_diag.py — for each V3 query, find the RANK of the CORRECT episode in the L5
cosine list (from SP_RECALL_L5_DUMPRANK telemetry). Decides re-rank vs re-key.

registry order == facts_v3 order == grow order, so fact[i] <-> episode name[i].
"""
import json, re, os, sys

ENG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = json.load(open(f"{ENG}/_v3_corpus/facts_v3.json", encoding="utf-8"))
reg = [json.loads(l) for l in open(f"{ENG}/_v3_corpus/registry.jsonl", encoding="utf-8") if l.strip()]
names = [r["name"] for r in reg]                       # episode name in grow order
para2idx = {f["para"].strip(): i for i, f in enumerate(F)}

log = open(f"{ENG}/_v3_serve_dump.log", encoding="utf-8", errors="replace").read()
# lines: RECALL-L5-DUMPRANK: q="..." ranked=[name=cos name=cos ...]
rx = re.compile(r'RECALL-L5-DUMPRANK: q="(.*?)" ranked=\[(.*?)\]')
rows = rx.findall(log)
# keep the LAST occurrence per query (this run)
qmap = {}
for q, ranked in rows:
    qmap[q] = ranked

print(f"facts={len(F)} episodes={len(names)} dump-queries={len(qmap)}\n")
crosspick_ids = {"dynamite_inv","hamlet_author","starry_night","radium_disc",
                 "david_sculptor","evolution_theory","first_element","telescope_inv"}
hdr = f"{'id':16} {'correct_rank':12} {'top1_is_correct':15} {'correct_cos':11} {'top1_cos':9} note"
print(hdr); print("-"*len(hdr))
rank_hist = {}
for i, f in enumerate(F):
    para = f["para"].strip()
    ranked = qmap.get(para)
    if ranked is None:
        # try loose match (json may re-encode unicode/quotes)
        cand = [v for k, v in qmap.items() if k[:30] == para[:30]]
        ranked = cand[0] if cand else None
    if ranked is None:
        print(f"{f['id']:16} {'NO-DUMP':12} (query not found; QONLY-skip or attr-decline)")
        continue
    pairs = [p.rsplit("=", 1) for p in ranked.split()]
    lst = [(nm, float(c)) for nm, c in pairs]
    correct = names[i]
    rank = next((r for r,(nm,_) in enumerate(lst) if nm == correct), None)
    top1_nm, top1_c = lst[0]
    ccos = next((c for nm,c in lst if nm==correct), None)
    rr = rank+1 if rank is not None else ">8"
    rank_hist[rr] = rank_hist.get(rr,0)+1
    mark = "*" if f["id"] in crosspick_ids else " "
    print(f"{mark}{f['id']:15} {str(rr):12} {str(rank==0):15} "
          f"{('%.4f'%ccos) if ccos is not None else 'n/a':11} {('%.4f'%top1_c):9} "
          f"{'CROSSPICK' if f['id'] in crosspick_ids else ''}")
print("\nrank histogram (correct-episode rank -> count):", dict(sorted(rank_hist.items(), key=lambda x:str(x[0]))))
# focused: for the crosspicks, is correct in top-3 / top-5 / buried?
print("\n=== CROSSPICK correct-rank detail ===")
for f in F:
    if f["id"] not in crosspick_ids: continue
    para = f["para"].strip(); ranked = qmap.get(para)
    if ranked is None:
        cand=[v for k,v in qmap.items() if k[:30]==para[:30]]; ranked=cand[0] if cand else None
    if ranked is None: print(f"{f['id']}: NO DUMP"); continue
    i = para2idx[para]; correct = names[i]
    pairs = [p.rsplit("=",1) for p in ranked.split()]
    lst = [(nm,float(c)) for nm,c in pairs]
    rank = next((r for r,(nm,_) in enumerate(lst) if nm==correct), None)
    shortlist = " ".join(f"{r+1}:{nm[-6:]}={c:.3f}" for r,(nm,c) in enumerate(lst[:5]))
    print(f"{f['id']:16} want={f['obey']:10} correct={correct[-6:]} rank={rank+1 if rank is not None else '>8'}  top5[{shortlist}]")
