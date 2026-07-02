"""f3_obey_probe.py — G-OBEY-PROBE-OFFLINE (GEODESIC follow-on to G-FM-STEER-OBEY).

Hypothesis (from the inverted dose-response): the final-norm DECIDE state (frame 0,
BEFORE the first answer token exists) linearly ENCODES whether the turn is about to
obey the delivered fact or leak the parametric answer — i.e. the surface that failed
as a steering LEVER works as a PROBE (expression ⇒ readable; not-cause ⇒ not pushable).

Data: fct turns from runs A (systemecho, high obey) and P (plain, leak-prone).
Labels: obey = counterfactual token in the answer; leak/miss = not. (Meta answers
joined to facts.json by para text.)

Stats (all pre-registered here, before P finishes):
  P1  v̄-projection AUC — proj = x·v̄/||v̄|| on frame-0; AUC obey-vs-not, per run + pooled.
      (Zero-training check: does the ray itself separate?)
  P2  logistic probe — L2 logistic on frame-0 (numpy, lambda=1e2, standardized),
      5-fold CV WITHIN-mode (A alone would be degenerate — few negatives — so the
      pinned figure is P alone + pooled-with-mode-shuffled-folds), plus the honest
      confound split: TRAIN on P, TEST on A (cross-mode transfer).
  PIN (provisional): pooled 5-fold AUC >= 0.80 AND cross-mode (P->A) AUC >= 0.70
      => the probe is real => next = serve wiring gate (G-OBEY-PROBE-LIVE).
      Below => honest negative, the expression is nonlinear/mode-bound.
"""
import json, os, struct, sys
import numpy as np

ENG = __file__.rsplit("_faithful_corpus", 1)[0]
E, SEED = 3840, 42
FRAME = int(os.environ.get("F3_PROBE_FRAME", "0"))  # 0 = pre-first-token DECIDE state; 1 = one-token-in
F = {it["para"]: it for it in json.load(open(f"{ENG}_faithful_corpus/facts.json", encoding="utf-8"))}

def has(ans, val): return val.lower().replace(" ", "") in ans.lower().replace(" ", "")

def load(run):
    d = f"{ENG}_faithful_corpus/f3/{run}"
    X, y, users = [], [], []
    for line in open(f"{d}/f3_meta.jsonl", encoding="utf-8"):
        m = json.loads(line)
        it = F.get(m["user"])
        if it is None: continue                      # fct only
        raw = open(f"{d}/f3_{m['chat_id']}.bin", "rb").read()
        v = np.frombuffer(raw, dtype="<f4", offset=16).reshape(2, E).astype(np.float64)
        X.append(v[FRAME]); y.append(1 if has(m["answer"], it["obey"]) else 0); users.append(m["user"])
    return np.stack(X), np.array(y), users

def auc(score, y):
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())

def logistic(Xtr, ytr, lam=1e2, iters=300):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Z = (Xtr - mu) / sd
    w = np.zeros(Z.shape[1]); b = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(Z @ w + b)))
        g = Z.T @ (p - ytr) / len(ytr) + lam * w / len(ytr)
        gb = float((p - ytr).mean())
        w -= 0.5 * g; b -= 0.5 * gb
    return w, b, mu, sd

def cv_auc(X, y, folds=5):
    rng = np.random.default_rng(SEED); idx = rng.permutation(len(y))
    scores = np.zeros(len(y))
    for f in range(folds):
        te = idx[f::folds]; tr = np.setdiff1d(idx, te)
        w, b, mu, sd = logistic(X[tr], y[tr])
        scores[te] = ((X[te] - mu) / sd) @ w + b
    return auc(scores, y), scores

vbar = np.fromfile(f"{ENG}_faithful_corpus/f3/steer_vbar_f0_all81.bin", dtype="<f4").astype(np.float64)
vbar /= np.linalg.norm(vbar)

runs = {}
for r in ("A", "P"):
    try: runs[r] = load(r)
    except FileNotFoundError: print(f"[{r}] not on disk yet"); sys.exit(1)

print(f"G-OBEY-PROBE-OFFLINE  seed={SEED}  frame={FRAME}")
for r, (X, y, _) in runs.items():
    proj = X @ vbar
    print(f"[{r}] n={len(y)} obey={int(y.sum())} leak/miss={int((1-y).sum())}  "
          f"P1 vbar-proj AUC={auc(proj, y):.3f}  (proj mean obey {proj[y==1].mean():.1f} vs not {proj[y==0].mean():.1f})")

XA, yA, _ = runs["A"]; XP, yP, _ = runs["P"]
Xp, yp = np.vstack([XA, XP]), np.concatenate([yA, yP])
proj_p = Xp @ vbar
print(f"[pooled] n={len(yp)} obey={int(yp.sum())}  P1 vbar-proj AUC={auc(proj_p, yp):.3f}")

aucP, _ = cv_auc(XP, yP)
aucPool, _ = cv_auc(Xp, yp)
w, b, mu, sd = logistic(XP, yP)
sA = ((XA - mu) / sd) @ w + b
aucX = auc(sA, yA)
print(f"P2 logistic 5-fold: P-only AUC={aucP:.3f}  pooled AUC={aucPool:.3f}  cross-mode P->A AUC={aucX:.3f}")

pin = aucPool >= 0.80 and aucX >= 0.70
print(f"PIN (pooled>=0.80 AND cross>=0.70): {'PASS -> G-OBEY-PROBE-LIVE is justified' if pin else 'FAIL -> honest negative / nonlinear-or-mode-bound'}")
