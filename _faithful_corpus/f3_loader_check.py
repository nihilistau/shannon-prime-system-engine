"""f3_loader_check.py — G-F3-CAPTURE criterion 4: the pair loader round-trips.

Loads both capture runs, joins A (with-faithfulness x1) to B (clean x0) by user
text, and checks: join completeness, frame shapes, finiteness, non-degeneracy.
Prints the per-kind pair counts + delta-norm stats (the first numbers the
G-FLOW-STRAIGHTNESS harness will consume). Exit 0 = round-trip OK.
"""
import json, struct, sys, math
import numpy as np

ENG = __file__.rsplit("_faithful_corpus", 1)[0]
E = 3840

def load(run):
    d = f"{ENG}_faithful_corpus/f3/{run}"
    out = {}
    for line in open(f"{d}/f3_meta.jsonl", encoding="utf-8"):
        m = json.loads(line)
        raw = open(f"{d}/f3_{m['chat_id']}.bin", "rb").read()
        magic = raw[:4]; e, nf, _ = struct.unpack("<3I", raw[4:16])
        assert magic == b"F3P1" and e == E and nf == 2, f"bad header {m['chat_id']}"
        v = np.frombuffer(raw, dtype="<f4", offset=16).reshape(nf, E)
        out[m["user"]] = (v, m)
    return out

A, B = load("A"), load("B")
users = sorted(set(A) & set(B))
print(f"A rows={len(A)} B rows={len(B)} joined pairs={len(users)}")
ok = len(users) == len(A) == len(B) == 81

kinds = {"fct": 0, "sne": 0}
dn_prompt, dn_first = [], []
for u in users:
    va, ma = A[u]; vb, mb = B[u]
    if not (np.isfinite(va).all() and np.isfinite(vb).all()):
        print(f"NONFINITE: {u[:50]!r}"); ok = False
    na, nb = np.linalg.norm(va, axis=1), np.linalg.norm(vb, axis=1)
    if (na < 1e-3).any() or (nb < 1e-3).any():
        print(f"DEGENERATE-ZERO: {u[:50]!r} normsA={na} normsB={nb}"); ok = False
    kind = "sne" if (ma.get("recalled") or {}).get("ep", "").startswith("sne") or "Node-" in u else "fct"
    kinds[kind] += 1
    dn_prompt.append(float(np.linalg.norm(va[0] - vb[0])))
    dn_first.append(float(np.linalg.norm(va[1] - vb[1])))

dp, df = np.array(dn_prompt), np.array(dn_first)
print(f"kinds: {kinds}")
print(f"delta-norm last-prompt-token: mean={dp.mean():.2f} min={dp.min():.2f} max={dp.max():.2f}")
print(f"delta-norm first-answer-token: mean={df.mean():.2f} min={df.min():.2f} max={df.max():.2f}")
if kinds["fct"] < 61 or kinds["sne"] < 20:
    print("PAIR-COUNT SHORTFALL"); ok = False
print("LOADER ROUND-TRIP:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
