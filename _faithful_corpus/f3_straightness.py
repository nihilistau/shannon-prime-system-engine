"""f3_straightness.py — G-FLOW-STRAIGHTNESS (GEODESIC ADR-003 §6, pre-registered).

Measures whether the F3 coupling (x0 = clean parametric state, run B; x1 = faithful
delivered state, run A) is straight enough for few-step flow-matching transport.
Engine untouched; pure numpy on _faithful_corpus/f3/{A,B}.

Pre-registered statistics (ADR-003 §6, pins marked):
  S1 field constancy — PCA of {v* = x1-x0}: top-1 explained variance (UNCENTERED —
     "one ray explains the field"; centered reported for insight) + mean cosine of
     each v* to the mean velocity.                       PIN: topPC>=0.70 AND cos>=0.70 -> Tier A
  S2 coupling conflict — t-grid {0.25,0.5,0.75}: x_t = (1-t)x0 + t*x1; k=8 NN per
     x_t (same t, excl self); mean over i of mean_j cos(v*_i, v*_j).
                                                          PIN: S2(t=0.5)>=0.50 -> Tier B leg 1
  S3 1-step transport error — 20% holdout (seed 42), per-t linear kernel ridge
     (lambda=1.0) v_hat(x_t); rel endpoint error ||x_t+(1-t)v_hat - x1|| / ||x1-x0||.
                                                          PIN: S3(mean)<=0.35 -> Tier B leg 2
Tiers: A = constant field (single steering vector suffices); B = straight enough
(1-2 step FM head); C = curved/conflicted (condition harder or reflow). ANY tier is
a GREEN gate; only an unrun measurement is RED.

Insight rows (not pinned): per-frame (0 = last-prompt-token PRIMARY, 1 = first-answer),
fct/sne subgroups, B-echo sensitivity slice (echo-suspect B answers excluded),
curvature-as-signal preview (per-kind cos-to-mean separation, §4.4).
"""
import json, struct, sys
import numpy as np

ENG = __file__.rsplit("_faithful_corpus", 1)[0]
E, SEED, K_NN, LAM = 3840, 42, 8, 1.0
TGRID = (0.25, 0.5, 0.75)

def load(run):
    d = f"{ENG}_faithful_corpus/f3/{run}"
    out = {}
    for line in open(f"{d}/f3_meta.jsonl", encoding="utf-8"):
        m = json.loads(line)
        raw = open(f"{d}/f3_{m['chat_id']}.bin", "rb").read()
        e, nf, _ = struct.unpack("<3I", raw[4:16])
        assert raw[:4] == b"F3P1" and e == E and nf == 2
        out[m["user"]] = (np.frombuffer(raw, dtype="<f4", offset=16).reshape(nf, E).astype(np.float64), m)
    return out

A, B = load("A"), load("B")
users = sorted(set(A) & set(B))
assert len(users) == 81, f"expected 81 joined pairs, got {len(users)}"
kind = np.array(["sne" if "Node-" in u else "fct" for u in users])

def _is_echo(meta, user):
    ans = meta.get("answer", "")
    if ans.strip().endswith("?"): return True
    uw = {w.lower() for w in user.split() if len(w) >= 4}
    aw = {w.lower() for w in ans.split() if len(w) >= 4}
    return bool(uw) and len(uw & aw) / len(uw) > 0.6

echo = np.array([_is_echo(B[u][1], u) for u in users])

def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def s1(V):
    m = V.mean(axis=0)
    mc = float(np.mean([cos(v, m) for v in V]))
    # uncentered top-1 explained variance (PIN) + centered (insight)
    su = np.linalg.svd(V, compute_uv=False)
    top_unc = float(su[0]**2 / (su**2).sum())
    sc = np.linalg.svd(V - m, compute_uv=False)
    top_cen = float(sc[0]**2 / ((sc**2).sum() + 1e-12))
    return top_unc, mc, top_cen

def s2(X0, X1, V):
    out = {}
    for t in TGRID:
        Xt = (1 - t) * X0 + t * X1
        # pairwise distances
        d2 = ((Xt[:, None, :] - Xt[None, :, :])**2).sum(-1)
        np.fill_diagonal(d2, np.inf)
        Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
        csm = Vn @ Vn.T
        vals = []
        for i in range(len(Xt)):
            nn = np.argsort(d2[i])[:K_NN]
            vals.append(float(csm[i, nn].mean()))
        out[t] = float(np.mean(vals))
    return out

def s3(X0, X1, V):
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(X0))
    nho = max(1, int(round(0.2 * len(X0))))
    ho, tr = idx[:nho], idx[nho:]
    errs = {}
    for t in TGRID:
        Xt = (1 - t) * X0 + t * X1
        Xtr, Vtr = Xt[tr], V[tr]
        Kk = Xtr @ Xtr.T
        alpha = np.linalg.solve(Kk + LAM * np.eye(len(tr)), Vtr)
        pe = []
        for i in ho:
            vh = (Xt[i] @ Xtr.T) @ alpha
            end = Xt[i] + (1 - t) * vh
            pe.append(float(np.linalg.norm(end - X1[i]) / (np.linalg.norm(X1[i] - X0[i]) + 1e-12)))
        errs[t] = float(np.mean(pe))
    errs["mean"] = float(np.mean([errs[t] for t in TGRID]))
    return errs

def report(name, mask, frame):
    X0 = np.stack([B[u][0][frame] for u in np.array(users)[mask]])
    X1 = np.stack([A[u][0][frame] for u in np.array(users)[mask]])
    V = X1 - X0
    tu, mc, tc = s1(V)
    r2 = s2(X0, X1, V)
    r3 = s3(X0, X1, V) if mask.sum() >= 10 else {"mean": float("nan"), 0.25: float("nan"), 0.5: float("nan"), 0.75: float("nan")}
    print(f"[{name} f{frame} n={mask.sum()}] S1 topPC(unc)={tu:.3f} cos-to-mean={mc:.3f} (topPC cen={tc:.3f}) | "
          f"S2 t.25={r2[0.25]:.3f} t.5={r2[0.5]:.3f} t.75={r2[0.75]:.3f} | "
          f"S3 t.5={r3[0.5]:.3f} mean={r3['mean']:.3f}")
    return tu, mc, r2, r3

print(f"G-FLOW-STRAIGHTNESS  pairs=81 (61 fct + 20 sne)  seed={SEED} k={K_NN} lambda={LAM}")
print(f"B-echo-suspect turns: {int(echo.sum())}/81 (sensitivity slice below)")
allm = np.ones(81, bool)

# ---- PINNED verdict: ALL pairs, frame 0 (last-prompt-token, the DECIDE state) ----
tu, mc, r2, r3 = report("ALL", allm, 0)
tier = "C"
if tu >= 0.70 and mc >= 0.70: tier = "A"
elif r2[0.5] >= 0.50 and r3["mean"] <= 0.35: tier = "B"

# ---- insight rows ----
report("ALL", allm, 1)
report("fct", kind == "fct", 0)
report("sne", kind == "sne", 0)
if 0 < echo.sum() < 71:
    report("no-echo", ~echo, 0)

# ---- curvature-as-signal preview (§4.4): does directional coherence separate kinds?
for nm, msk in (("fct", kind == "fct"), ("sne", kind == "sne")):
    Vk = np.stack([A[u][0][0] - B[u][0][0] for u in np.array(users)[msk]])
    mall = np.stack([A[u][0][0] - B[u][0][0] for u in users]).mean(axis=0)
    cs = [cos(v, mall) for v in Vk]
    print(f"[curv-preview {nm}] cos-to-global-mean: mean={np.mean(cs):.3f} min={np.min(cs):.3f} max={np.max(cs):.3f}")

print(f"\nVERDICT: TIER {tier}  " + {
    "A": "(constant field — ship a single TELE-2 steering vector; FM held in reserve)",
    "B": "(straight enough — 1-2 step FM head viable; build §4.2 then §4.1)",
    "C": "(curved/conflicted — condition harder or reflow-on-NIGHTSHIFT; re-gate)"}[tier])
