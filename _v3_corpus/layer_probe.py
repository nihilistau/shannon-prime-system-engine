"""layer_probe.py — FREE probe on the already-captured b3_data_v3.npz: for each of the NG
periodic global layers, does raw cosine (query last-token, mean-heads, L2) vs episode
(mean-pos, mean-heads, L2) separate the correct SAME-TEMPLATE episode? The deploy uses
global layer index 0 (=L5). If a different layer scores higher, that's a free selector win;
if none do, the subject signal is in the NON-global/earlier layers (motivating a full
per-layer capture). No daemon, no training."""
import numpy as np, os, json
ENG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = np.load(f"{ENG}/_b3_wc/b3_data_v3.npz", allow_pickle=True)
Q = [np.asarray(q, np.float32) for q in d["Q"]]      # each [ng, G_NH, HD]
K = [np.asarray(k, np.float32) for k in d["K"]]      # each [ng, npos, HD]
labels = d["labels"].astype(int)
GLOBALS = [5,11,17,23,29,35,41,47]
ng = Q[0].shape[0]; E = len(K)
def l2(v): return v/(np.linalg.norm(v)+1e-9)
# per-layer episode vectors: mean over positions and heads
Kvec = np.zeros((ng, E, K[0].shape[-1]), np.float32)
for e in range(E):
    for l in range(ng):
        Kvec[l, e] = l2(K[e][l].mean(0))
pos = [i for i,lab in enumerate(labels) if lab >= 0]
print(f"queries={len(Q)} positives={len(pos)} episodes={E} global-layers={ng}\n")
print(f"{'layer':>6} {'globL':>6} {'top1':>7} {'top3':>7}")
best = None
for l in range(ng):
    t1 = t3 = 0
    for i in pos:
        qv = l2(Q[i][l].mean(0))
        cos = Kvec[l] @ qv
        order = np.argsort(-cos)
        correct = labels[i]
        r = int(np.where(order == correct)[0][0])
        t1 += (r == 0); t3 += (r < 3)
    rate1 = t1/len(pos); rate3 = t3/len(pos)
    gl = GLOBALS[l] if l < len(GLOBALS) else '?'
    tag = "  <- L5 (deploy)" if l == 0 else ""
    print(f"{l:>6} {str(gl):>6} {rate1:>6.1%} {rate3:>6.1%}{tag}")
    if best is None or rate1 > best[1]: best = (l, rate1)
# also: concat ALL global layers (mean-pool) as one big vector
def allcat_q(i): return l2(np.concatenate([Q[i][l].mean(0) for l in range(ng)]))
def allcat_k(e): return l2(np.concatenate([K[e][l].mean(0) for l in range(ng)]))
Kall = np.stack([allcat_k(e) for e in range(E)])
t1 = sum(int(np.argmax(Kall @ allcat_q(i)) == labels[i]) for i in pos)
print(f"\n  ALL-global concat: top1 = {t1/len(pos):.1%}")
print(f"  best single layer: idx {best[0]} (globL {GLOBALS[best[0]]}) at {best[1]:.1%}")
