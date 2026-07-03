"""wc_rank_diag.py — from the B3-WC lse-mean log lines, find the rank of the CORRECT
episode under the EXISTING W_c relevance metric (vs L5-cosine which buried 8/30).

The B3-WC line lists episodes in REGISTRY order == facts order == query order. The gate
queries in facts order, so the j-th B3-WC line is query j whose correct episode is index j.
"""
import re, os, json
ENG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = json.load(open(f"{ENG}/_v3_corpus/facts_v3.json", encoding="utf-8"))
log = open(f"{ENG}/_v3_serve_wcprobe.log", encoding="utf-8", errors="replace").read()
rx = re.compile(r"B3-WC lse-mean.*?: \[(.*?)\]")
lines = rx.findall(log)
# keep the last 30 (this run)
lines = lines[-30:]
print(f"B3-WC score lines parsed: {len(lines)}  facts: {len(F)}\n")
cross = {"dynamite_inv","hamlet_author","starry_night","radium_disc","david_sculptor",
         "evolution_theory","first_element","telescope_inv"}
rank_hist = {}
crosspick_ranks = {}
r1 = 0
for j, line in enumerate(lines):
    pairs = [p.rsplit("=",1) for p in line.split()]
    scores = [(nm, float(v)) for nm, v in pairs]
    order = sorted(range(len(scores)), key=lambda i: -scores[i][1])  # index by desc score
    # correct episode is index j (registry order == facts order)
    rank = order.index(j) + 1 if j < len(scores) else None
    rank_hist[rank] = rank_hist.get(rank, 0) + 1
    if rank == 1: r1 += 1
    fid = F[j]["id"] if j < len(F) else f"idx{j}"
    if fid in cross: crosspick_ranks[fid] = rank
    mark = "*" if fid in cross else " "
    top = scores[order[0]]
    print(f"{mark}{fid:16} correct_rank={str(rank):4} correct_score={scores[j][1]:+7.3f} top1={top[0][-6:]}={top[1]:+.3f}")
print(f"\nW_c correct-rank-1: {r1}/{len(lines)}   (L5-cosine was 22/30)")
print("rank histogram:", dict(sorted(rank_hist.items(), key=lambda x: (x[0] is None, x[0]))))
print("\n=== CROSSPICK ranks under W_c (were >8 under L5) ===")
for fid in cross:
    print(f"  {fid:16} W_c rank = {crosspick_ranks.get(fid,'?')}")
