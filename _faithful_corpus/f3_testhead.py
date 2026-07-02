"""f3_testhead.py — G-TESTHEAD-OFFLINE: the SPECTEST v2 semantic test-head, offline gate.

THE CONVERGENCE BUILD (2026-07-03): the head that closes SPECTEST's value-substitution
class IS the nonlinear obey/leak probe. Data = the 6-mode x 61 capture
(_faithful_corpus/f3/M_*): 366 labeled DECIDE states across delivery framings whose
obey rates span ~26/61..54/61 — labels are balanced WITHIN modes, and evaluation is
MODE-HELD-OUT (train on 5 framings, test on the 6th, rotate) so the prompt-condition
confound that produced the fake pooled-0.79 AUC (G-OBEY-PROBE-OFFLINE) cannot score.

PINS — PRE-REGISTERED BEFORE THE CAPTURE FINISHED (this file is committed with the
data untouched; the capture was still mid-M_system when these were written):
  REAL   : mean held-out-mode AUC >= 0.75 AND min fold >= 0.65 (either frame)
           => the latent carries the obey/leak signal => wire at the spectest seam
              (export head, SP_SPECTEST_HEAD, live gate).
  WEAK   : mean >= 0.65 (either frame) => signal exists but underpowered; next lever
           is MORE DATA (nightshift-grown corpora), not architecture fishing.
  NEGATIVE: below => the final-norm latent does not carry it even nonlinear at n=366
           => the test-head moves UPSTREAM (attention features / the TELE-2 seam,
              GEODESIC rung 2) and the final-norm surface is FULLY closed (3rd strike).

Model (deterministic, numpy, seed 42): per-fold standardize -> PCA-128 (train-fit) ->
MLP 128-32-1 (tanh, full-batch GD + momentum, L2) vs a linear-logistic baseline on
the same PCA features (the nonlinearity's contribution must be visible, not assumed).
"""
import json, struct, sys
import numpy as np

ENG = __file__.rsplit("_faithful_corpus", 1)[0]
E, SEED = 3840, 42
MODES = ["plain", "scaled", "sandwich", "factecho", "system", "systemecho"]
F = {it["para"]: it for it in json.load(open(f"{ENG}_faithful_corpus/facts.json", encoding="utf-8"))}

def has(ans, val): return val.lower().replace(" ", "") in ans.lower().replace(" ", "")

def load_mode(m, frame):
    d = f"{ENG}_faithful_corpus/f3/M_{m}"
    X, y, lk = [], [], []
    for line in open(f"{d}/f3_meta.jsonl", encoding="utf-8"):
        meta = json.loads(line)
        it = F.get(meta["user"])
        if it is None: continue
        raw = open(f"{d}/f3_{meta['chat_id']}.bin", "rb").read()
        v = np.frombuffer(raw, dtype="<f4", offset=16).reshape(2, E).astype(np.float64)
        ob = has(meta["answer"], it["obey"])
        X.append(v[frame]); y.append(1 if ob else 0)
        lk.append(1 if (not ob and has(meta["answer"], it["param"])) else 0)
    return np.stack(X), np.array(y), np.array(lk)

def auc(s, y):
    p, n = s[y == 1], s[y == 0]
    if len(p) == 0 or len(n) == 0: return float("nan")
    return float((p[:, None] > n[None, :]).mean() + 0.5 * (p[:, None] == n[None, :]).mean())

def pca_fit(X, k):
    mu = X.mean(0); Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    W = Vt[:k].T
    sd = (Xc @ W).std(0) + 1e-9
    return mu, W, sd

def mlp_train(Z, y, seed, h=32, epochs=400, lr=0.05, mom=0.9, l2=1e-3):
    rng = np.random.default_rng(seed)
    d = Z.shape[1]
    W1 = rng.normal(0, 1/np.sqrt(d), (d, h)); b1 = np.zeros(h)
    W2 = rng.normal(0, 1/np.sqrt(h), h); b2 = 0.0
    vW1 = np.zeros_like(W1); vb1 = np.zeros_like(b1); vW2 = np.zeros_like(W2); vb2 = 0.0
    n = len(y)
    for _ in range(epochs):
        H = np.tanh(Z @ W1 + b1)
        p = 1/(1+np.exp(-(H @ W2 + b2)))
        g = (p - y) / n
        gW2 = H.T @ g + l2*W2/n; gb2 = g.sum()
        gH = np.outer(g, W2) * (1 - H**2)
        gW1 = Z.T @ gH + l2*W1/n; gb1 = gH.sum(0)
        vW2 = mom*vW2 - lr*gW2; W2 += vW2; vb2 = mom*vb2 - lr*gb2; b2 += vb2
        vW1 = mom*vW1 - lr*gW1; W1 += vW1; vb1 = mom*vb1 - lr*gb1; b1 += vb1
    return lambda Zt: np.tanh(Zt @ W1 + b1) @ W2 + b2

def logistic_train(Z, y, iters=400, lr=0.5, l2=1e-2):
    w = np.zeros(Z.shape[1]); b = 0.0; n = len(y)
    for _ in range(iters):
        p = 1/(1+np.exp(-(Z @ w + b)))
        w -= lr * (Z.T @ (p - y)/n + l2*w/n); b -= lr * float((p - y).mean())
    return lambda Zt: Zt @ w + b

print(f"G-TESTHEAD-OFFLINE  seed={SEED}  modes={MODES}")
verdicts = {}
for frame in (0, 1):
    data = {m: load_mode(m, frame) for m in MODES}
    for m in MODES:
        X, y, lk = data[m]
        if frame == 0:
            print(f"  [data M_{m}] n={len(y)} obey={int(y.sum())} leak={int(lk.sum())}")
    rows = []
    for held in MODES:
        Xtr = np.vstack([data[m][0] for m in MODES if m != held])
        ytr = np.concatenate([data[m][1] for m in MODES if m != held])
        Xte, yte, lkte = data[held]
        mu, W, sd = pca_fit(Xtr, 128)
        Ztr = ((Xtr - mu) @ W) / sd; Zte = ((Xte - mu) @ W) / sd
        mlp = mlp_train(Ztr, ytr, SEED)
        lin = logistic_train(Ztr, ytr)
        s_m, s_l = mlp(Zte), lin(Zte)
        a_m, a_l = auc(s_m, yte), auc(s_l, yte)
        # leak-vs-obey separation (the value-substitution class specifically)
        mask = (yte == 1) | (lkte == 1)
        a_leak = auc(s_m[mask], yte[mask]) if mask.sum() > 2 else float("nan")
        rows.append((held, a_m, a_l, a_leak))
        print(f"[f{frame} held={held:10}] MLP AUC={a_m:.3f}  linear={a_l:.3f}  obey-vs-LEAK={a_leak:.3f}")
    aucs = np.array([r[1] for r in rows])
    print(f"[f{frame}] MLP held-out: mean={aucs.mean():.3f} min={aucs.min():.3f}  "
          f"linear mean={np.mean([r[2] for r in rows]):.3f}")
    verdicts[frame] = (float(aucs.mean()), float(aucs.min()))

best = max(verdicts.items(), key=lambda kv: kv[1][0])
mean_a, min_a = best[1]
if mean_a >= 0.75 and min_a >= 0.65: v = "REAL — wire at the spectest seam (SP_SPECTEST_HEAD, live gate next)"
elif mean_a >= 0.65: v = "WEAK — signal exists; lever = more data (nightshift-grown corpora), not architecture"
else: v = "NEGATIVE — final-norm latent closed (3rd strike); test-head moves upstream (rung 2)"
print(f"\nVERDICT (best frame {best[0]}): mean={mean_a:.3f} min={min_a:.3f} -> {v}")
