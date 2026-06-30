"""LN-1 HYBRID (SPLADE-style) trainer + dual-axis held-out gate.

Two channels, fused: score(q,E) = alpha * SEM(q,E) + beta * LEX(q,E)
  SEM = raw q.K max-sim late-interaction on global-K/Q (per layer,head: max over episode positions
        of Q.K ; then mean over layers,heads). The model's NATIVE semantic relevance (no lossy r=32).
  LEX = token-overlap (Jaccard) of the query text vs the episode text. Orthogonal token channel.
Channels are z-scored per query (comparable scales), then alpha/beta grid-searched to maximize
TRAIN top-1. The gate is DUAL-AXIS on a held-out fact split:
  EXACT queries  -> LEX must dominate (hold the 100% floor)
  PARA  queries  -> LEX ~ 0 by design; SEM (alpha) must rescue
Inputs (from the F3 capture): _faithful_corpus/qdump/{q_<id>.bin,k_<id>.bin}  (exact Q + episode K),
_faithful_corpus/qdump_para/q_<id>.bin (paraphrase Q), _faithful_corpus/facts.json (texts + diagonal).
"""
import os, glob, struct, re, sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
G_NH, HD = 16, 512
ENG = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
CAP = os.path.join(ENG, "_faithful_corpus", "qdump")
CAPP = os.path.join(ENG, "_faithful_corpus", "qdump_para")
FJSON = os.path.join(ENG, "_faithful_corpus", "facts.json")
import json
FACTS = json.load(open(FJSON, encoding="utf-8"))
HOLD_FRAC = float(os.environ.get("LN1_HOLD", "0.25"))
SEED = int(os.environ.get("LN1_SEED", "0"))


def read_vec(path):
    b = open(path, "rb").read(); ng, d = struct.unpack("<II", b[:8])
    return np.frombuffer(b[8:], "<f4").copy(), ng, d


def load_q(capdir):
    out = {}
    for p in glob.glob(os.path.join(capdir, "q_*.bin")):
        cid = int(os.path.basename(p)[2:-4]); a, ng, qd = read_vec(p)
        out[cid] = a.reshape(ng, G_NH, HD)
    return [out[c] for c in sorted(out)]


def load_k(capdir):
    out = {}
    for p in glob.glob(os.path.join(capdir, "k_*.bin")):
        cid = int(os.path.basename(p)[2:-4]); a, ng, npos = read_vec(p)
        out[cid] = a.reshape(ng, npos, HD)
    return [out[c] for c in sorted(out)]


def toks(s):
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def jaccard(a, b):
    A, B = toks(a), toks(b)
    return len(A & B) / max(1, len(A | B))


def sem_matrix(Qs, Ks):
    n = len(Qs); M = np.zeros((n, len(Ks)), np.float32)
    for i in range(n):
        for e in range(len(Ks)):
            sim = np.einsum("lhd,lpd->lhp", Qs[i], Ks[e])   # [8,16,npos]
            M[i, e] = sim.max(axis=2).mean()
    return M


def zscore_rows(M):
    mu = M.mean(1, keepdims=True); sd = M.std(1, keepdims=True) + 1e-6
    return (M - mu) / sd


def top1(score, labels, idx):
    return sum(int(score[i].argmax() == labels[i]) for i in idx) / max(1, len(idx))


def main():
    Qx, Qp, K = load_q(CAP), load_q(CAPP), load_k(CAP)
    n = min(len(Qx), len(Qp), len(K), len(FACTS))
    Qx, Qp, K, F = Qx[:n], Qp[:n], K[:n], FACTS[:n]
    labels = np.arange(n)                                  # diagonal: query i -> episode i
    print(f"[ln1h] n={n} exactQ={len(Qx)} paraQ={len(Qp)} epK={len(K)}", flush=True)

    SEM_x = zscore_rows(sem_matrix(Qx, K)); SEM_p = zscore_rows(sem_matrix(Qp, K))
    LEX_x = zscore_rows(np.array([[jaccard(F[i]["q"], F[e]["fact"]) for e in range(n)] for i in range(n)], np.float32))
    LEX_p = zscore_rows(np.array([[jaccard(F[i].get("para", F[i]["q"]), F[e]["fact"]) for e in range(n)] for i in range(n)], np.float32))

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n); nh = max(1, int(round(n * HOLD_FRAC)))
    hold, train = perm[:nh], perm[nh:]
    print(f"[ln1h] train={len(train)} holdout={len(hold)}", flush=True)

    # grid-search alpha,beta on TRAIN (maximize avg top-1 over exact+para)
    best = (-1, 1.0, 0.0)
    for a in np.linspace(0, 2, 21):
        for b in np.linspace(0, 2, 21):
            sx, sp = a * SEM_x + b * LEX_x, a * SEM_p + b * LEX_p
            acc = 0.5 * (top1(sx, labels, train) + top1(sp, labels, train))
            if acc > best[0]: best = (acc, a, b)
    _, A, B = best
    print(f"[ln1h] fitted alpha={A:.2f} beta={B:.2f} (train avg top-1={best[0]:.3f})", flush=True)

    def report(tag, idx):
        print(f"\n[ln1h] === {tag} (n={len(idx)}) ===", flush=True)
        print(f"  LEX-only   exact={top1(LEX_x,labels,idx):.3f}  para={top1(LEX_p,labels,idx):.3f}", flush=True)
        print(f"  SEM-only   exact={top1(SEM_x,labels,idx):.3f}  para={top1(SEM_p,labels,idx):.3f}", flush=True)
        print(f"  HYBRID     exact={top1(A*SEM_x+B*LEX_x,labels,idx):.3f}  para={top1(A*SEM_p+B*LEX_p,labels,idx):.3f}", flush=True)
    report("HELD-OUT (the gate)", hold)
    report("train (sanity)", train)
    np.savez(os.path.join(ENG, "_faithful_corpus", "ln1_hybrid.npz"), alpha=A, beta=B,
             holdout=hold, SEM_x=SEM_x, SEM_p=SEM_p, LEX_x=LEX_x, LEX_p=LEX_p)
    print(f"\n[ln1h] saved ln1_hybrid.npz (alpha={A:.2f} beta={B:.2f})", flush=True)


if __name__ == "__main__":
    main()
