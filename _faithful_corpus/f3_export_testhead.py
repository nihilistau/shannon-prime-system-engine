"""f3_export_testhead.py — export the SPECTEST v2 linear test-head (G-SPECTEST-V2).

RIGOR CHOICE (pre-registered): the deployed head trains on FIVE modes EXCLUDING
M_plain — the live gate serves PLAIN delivery, so the live run is MODE-HELD-OUT,
not in-distribution (offline held=plain predicted 0.953 linear AUC, frame 1).

Since linear == MLP (G-TESTHEAD-OFFLINE), the standardize→PCA→logistic pipeline
collapses to ONE affine map: score = x·v + c  (v = W_pca (w/sd) folded, c folded).
Blob spectest_head_f1.bin: magic 'SPH1' + E:u32 + frame:u32 + pad:u32 +
v[E] f32 LE + c f32 + tau f32. Veto rule at the seam: score < tau.
tau (pre-registered): the training threshold catching >=95% of leak-class rows.
"""
import json, struct
import numpy as np

ENG = __file__.rsplit("_faithful_corpus", 1)[0]
E, SEED, FRAME = 3840, 42, 1
TRAIN_MODES = ["scaled", "sandwich", "factecho", "system", "systemecho"]  # plain EXCLUDED
F = {it["para"]: it for it in json.load(open(f"{ENG}_faithful_corpus/facts.json", encoding="utf-8"))}

def has(ans, val): return val.lower().replace(" ", "") in ans.lower().replace(" ", "")

X, y, lk = [], [], []
for m in TRAIN_MODES:
    d = f"{ENG}_faithful_corpus/f3/M_{m}"
    for line in open(f"{d}/f3_meta.jsonl", encoding="utf-8"):
        meta = json.loads(line)
        it = F.get(meta["user"])
        if it is None: continue
        raw = open(f"{d}/f3_{meta['chat_id']}.bin", "rb").read()
        v = np.frombuffer(raw, dtype="<f4", offset=16).reshape(2, E).astype(np.float64)
        ob = has(meta["answer"], it["obey"])
        X.append(v[FRAME]); y.append(1 if ob else 0)
        lk.append(1 if (not ob and has(meta["answer"], it["param"])) else 0)
X, y, lk = np.stack(X), np.array(y), np.array(lk)
print(f"train: n={len(y)} obey={int(y.sum())} leak={int(lk.sum())} (modes={TRAIN_MODES}, frame={FRAME})")

mu = X.mean(0); Xc = X - mu
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
W = Vt[:128].T
Z = Xc @ W; sd = Z.std(0) + 1e-9; Z = Z / sd

w = np.zeros(128); b = 0.0; n = len(y)
for _ in range(400):
    p = 1/(1+np.exp(-(Z @ w + b)))
    w -= 0.5 * (Z.T @ (p - y)/n + 1e-2*w/n); b -= 0.5 * float((p - y).mean())

s = Z @ w + b
pos, neg = s[y == 1], s[y == 0]
auc = float((pos[:, None] > neg[None, :]).mean())
print(f"train AUC={auc:.3f}")

# collapse to affine: score = x@v + c
v_full = W @ (w / sd)
c = float(b - mu @ v_full)
assert np.allclose(X @ v_full + c, s, atol=1e-8)

# tau: catch >=95% of training LEAK rows (leak scores below tau)
leak_scores = np.sort(s[lk == 1])
tau = float(leak_scores[int(np.ceil(0.95 * len(leak_scores))) - 1]) if len(leak_scores) else 0.0
n_leak_caught = int((s[lk == 1] < tau).sum()) if len(leak_scores) else 0
false_veto = float((s[y == 1] < tau).mean())
print(f"tau={tau:.4f}: catches {n_leak_caught}/{int(lk.sum())} train leaks; false-veto on train obeys = {false_veto:.3f}")

out = f"{ENG}_faithful_corpus/f3/spectest_head_f1.bin"
with open(out, "wb") as f:
    f.write(b"SPH1")
    f.write(struct.pack("<3I", E, FRAME, 0))
    f.write(v_full.astype("<f4").tobytes())
    f.write(struct.pack("<2f", c, tau))
print(f"wrote {out} ({4+12+E*4+8} bytes)")
