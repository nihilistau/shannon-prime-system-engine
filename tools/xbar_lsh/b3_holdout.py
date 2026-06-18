#!/usr/bin/env python3
"""B3-v3 HELD-OUT falsification: the gate#1 green was in-sample (W_c trained + gated on
the same queries). This splits the query set — train on the first TR paraphrases per
episode + TRF foreign, then gate IN THE INTEGER DOMAIN on the HELD-OUT phrasings the
optimizer never saw. Same episodes/registry (the autonomous-recall use case = users ask
about KNOWN memories in NEW words). If the integer separation holds on held-out queries,
the metric generalized; if it collapses, W_c memorized (honest negative #3).
"""
import os, sys
import numpy as np
import torch, torch.nn.functional as F

HD, G_NH = 512, 16
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b3_export_wc import int_relevance, quantize

eng = os.environ.get("SP_ENGINE_DIR", r"D:\F\shannon-prime-repos\shannon-prime-system-engine")
d = np.load(os.path.join(eng, "_b3_wc", "b3_data.npz"), allow_pickle=True)
Qall = [np.asarray(q, np.float32) for q in d["Q"]]
K = [np.asarray(k, np.float32) for k in d["K"]]
lab = d["labels"].astype(np.int64); names = list(d["ep_names"]); E = len(K)
TR_POS = int(os.environ.get("HO_TRPOS", "4"))   # paraphrases/episode used for TRAIN
TR_FGN = int(os.environ.get("HO_TRFGN", "5"))   # foreign queries used for TRAIN
r = int(os.environ.get("WC_R", "32")); EPOCHS = int(os.environ.get("WC_EPOCHS", "400"))

# build train / test index split per class (stable order = mining order)
seen = {e: 0 for e in range(E)}; seen_f = 0
tr, te = [], []
for i in range(len(Qall)):
    l = int(lab[i])
    if l >= 0:
        (tr if seen[l] < TR_POS else te).append(i); seen[l] += 1
    else:
        (tr if seen_f < TR_FGN else te).append(i); seen_f += 1
print(f"[ho] split: train={len(tr)} ({sum(1 for i in tr if lab[i]>=0)}p/{sum(1 for i in tr if lab[i]<0)}f) "
      f"test={len(te)} ({sum(1 for i in te if lab[i]>=0)}p/{sum(1 for i in te if lab[i]<0)}f)", flush=True)
if not any(lab[i] < 0 for i in te) or not any(lab[i] >= 0 for i in te):
    print("[ho] WARN held-out set lacks a positive or a foreign — widen the set.");

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(20260619)
Qd = [torch.tensor(Qall[i], device=dev) for i in range(len(Qall))]
Kd = [torch.tensor(K[e], device=dev) for e in range(E)]
ng = min(Qd[0].shape[0], Kd[0].shape[0])
Wc = torch.nn.Parameter(torch.randn(HD, r, device=dev) / HD**0.5)
log_tau = torch.nn.Parameter(torch.zeros((), device=dev)); s0 = torch.nn.Parameter(torch.zeros((), device=dev))
opt = torch.optim.Adam([Wc, log_tau, s0], lr=3e-3); scale = 1.0 / r**0.5

def relevance(qi):
    qp = torch.einsum("lhd,dr->lhr", qi[:ng], Wc)
    S = []
    for e in range(E):
        kp = torch.einsum("lpd,dr->lpr", Kd[e][:ng], Wc)
        sim = torch.einsum("lhr,lpr->lhp", qp, kp) * scale
        S.append(torch.logsumexp(sim, 2).mean())
    return torch.stack(S)

for ep in range(EPOCHS):
    opt.zero_grad(); ce = 0.0; hin = 0.0; npos = 0
    for i in tr:                                   # TRAIN split only
        S = relevance(Qd[i]); logits = torch.cat([S, s0.view(1)]) / torch.exp(log_tau)
        t = int(lab[i]); t = E if t < 0 else t
        ce = ce + F.cross_entropy(logits.view(1, -1), torch.tensor([t], device=dev))
        if lab[i] >= 0:
            neg = S.clone(); neg[int(lab[i])] = -1e9
            hin = hin + F.relu(neg.max() - S[int(lab[i])] + 1.0); npos += 1
    (ce / len(tr) + 0.2 * hin / max(1, npos)).backward(); opt.step()

Wi, k = quantize(Wc.detach().cpu().numpy().astype(np.float32), 16)
print(f"[ho] trained on {len(tr)} queries; W_c->int16 scale 2^{k}", flush=True)
print("\n[ho] HELD-OUT integer relevance (queries the optimizer NEVER saw):", flush=True)
tgt, fgn = [], []
for i in te:
    row = [int_relevance(Qall[i], K[e], Wi, 10)[1] for e in range(E)]
    arg = int(np.argmax([float(x) for x in row])); l = int(lab[i])
    mx = max(1.0, max(abs(float(x)) for x in row))
    disp = " ".join(f"{names[e]}={float(row[e])/mx:+.3f}" for e in range(E))
    if l >= 0:
        tgt.append(float(row[l])); tag = f"want={names[l]} {'OK' if arg==l else 'WRONG('+names[arg]+')'}"
    else:
        fgn.append(max(float(x) for x in row)); tag = "FOREIGN(reject)"
    print(f"  {disp}   {tag}", flush=True)
mt = min(tgt) if tgt else float("nan"); mf = max(fgn) if fgn else float("nan")
sep = mt > mf
print(f"\n[ho] HELD-OUT min_target={mt:.4g}  max_foreign={mf:.4g}  => "
      f"{'GENERALIZES (GREEN)' if sep else 'MEMORIZED — no held-out separation (RED)'}", flush=True)
sys.exit(0 if sep else 1)
