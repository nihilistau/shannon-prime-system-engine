"""sel_offline.py — offline selector lever tests (G-SEL-OFFLINE).
Inputs: _qdump_sel/q_0..60.bin = para query global-Q, q_61..121.bin = fact-text global-Q,
eps/fct_NNN/ep.l5 = deployed canonical-question keys. Tests, per lever, top-1 accuracy on
the 61 paras (baseline = deployed key1 only, expected 54/61):
  key2      max(cos(q,key1), cos(q,key2))   key2 = fact-text embed
  fusion    cos1 + lam*jaccard(query,fact)  lam sweep
  key2+fus  combined
  topk      correct-in-top-k ceiling (k=2,3,5)
"""
import glob, json, os, struct, sys
import numpy as np

ENG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
L5, HD, G_NH = 5, 512, 16
F = json.load(open(f"{ENG}/_faithful_corpus/facts.json", encoding="utf-8"))
N = len(F)

def load_q(p):
    b = open(p, "rb").read(); ng, d = struct.unpack("<II", b[:8])
    a = np.frombuffer(b[8:], "<f4").astype(np.float64).reshape(ng, G_NH, HD)
    v = a[L5].mean(0); return v / (np.linalg.norm(v) + 1e-30)

qs = sorted(glob.glob(f"{ENG}/_qdump_sel/q_*.bin"), key=lambda p: int(os.path.basename(p)[2:-4]))
assert len(qs) >= 2 * N, f"need {2*N} dumps, have {len(qs)}"
P = np.stack([load_q(p) for p in qs[:N]])          # para queries [61,512]
K2 = np.stack([load_q(p) for p in qs[N:2*N]])      # fact-text embeds = candidate key2 [61,512]

def load_key1(i):
    b = open(f"{ENG}/_faithful_corpus/eps/fct_{i:03d}/ep.l5", "rb").read()
    v = np.frombuffer(b, "<f4").astype(np.float64); return v / (np.linalg.norm(v) + 1e-30)
K1 = np.stack([load_key1(i) for i in range(N)])

STOP = set("the a an of in on for is are was were to and or which what who where when now by with does".split())
def toks(s): return {w for w in "".join(c if c.isalnum() else " " for c in s.lower()).split() if len(w) >= 3 and w not in STOP}
J = np.zeros((N, N))
for i in range(N):
    tq = toks(F[i]["para"])
    for j in range(N):
        tf = toks(F[j]["fact"])
        J[i, j] = len(tq & tf) / (len(tq | tf) or 1)

C1 = P @ K1.T   # [para, episode] via key1
C2 = P @ K2.T   # via key2 (fact embed)

def acc(S, label):
    top = S.argmax(1); ok = (top == np.arange(N)).sum()
    fixed = [F[i]['id'] for i in range(N) if top[i] == i and C1.argmax(1)[i] != i]
    broke = [F[i]['id'] for i in range(N) if top[i] != i and C1.argmax(1)[i] == i]
    print(f"{label:28} top1={ok}/{N}" + (f"  fixed={fixed}" if fixed else "") + (f"  BROKE={broke}" if broke else ""))
    return ok

print("== baseline ==")
acc(C1, "key1 (deployed)")
b_top = C1.argmax(1)
print("misses:", [F[i]['id'] for i in range(N) if b_top[i] != i])
for k in (2, 3, 5):
    ink = sum(1 for i in range(N) if i in np.argsort(-C1[i])[:k])
    print(f"key1 correct-in-top-{k}: {ink}/{N}")
print("== levers ==")
acc(np.maximum(C1, C2), "key2 max")
acc((C1 + C2) / 2, "key2 mean")
for lam in (0.05, 0.1, 0.2, 0.4):
    acc(C1 + lam * J, f"fusion lam={lam}")
for lam in (0.05, 0.1, 0.2):
    acc(np.maximum(C1, C2) + lam * J, f"key2max+fusion lam={lam}")
