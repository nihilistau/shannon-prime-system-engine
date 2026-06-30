"""LN-1 adapter: build the b3_train_wc npz from the clean F3 capture.

Reads the matched per-turn dumps in _faithful_corpus/qdump:
  q_<id>.bin = <u32 ng><u32 qd=G_NH*HD><f32 query-Q[ng,G_NH,HD]>
  k_<id>.bin = <u32 ng><u32 npos><f32 episode-gK[ng,npos,HD]>   (clean in-memory gk, not ep.k)
  lbl_<id>.txt = <episode>TAB<overlap>
Builds the diagonal dataset: Q[i], K[i]=episode i's gK, labels[i]=i, ep_names[i].
Output: _faithful_corpus/ln1_data.npz  (Q, K, labels, ep_names) — b3_train_wc.py schema.
"""
import os, glob, struct, sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
G_NH, HD = 16, 512
ENG = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
CAP = os.path.join(ENG, "_faithful_corpus", "qdump")
OUT = os.path.join(ENG, "_faithful_corpus", "ln1_data.npz")


def read_q(path):
    b = open(path, "rb").read(); ng, qd = struct.unpack("<II", b[:8])
    return np.frombuffer(b[8:], "<f4").copy().reshape(ng, G_NH, HD)   # qd == G_NH*HD


def read_k(path):
    b = open(path, "rb").read(); ng, npos = struct.unpack("<II", b[:8])
    return np.frombuffer(b[8:], "<f4").copy().reshape(ng, npos, HD)


def main():
    ids = sorted(int(os.path.basename(p)[2:-4]) for p in glob.glob(os.path.join(CAP, "q_*.bin")))
    Q, K, labels, names = [], [], [], []
    for cid in ids:
        qf = os.path.join(CAP, f"q_{cid}.bin"); kf = os.path.join(CAP, f"k_{cid}.bin"); lf = os.path.join(CAP, f"lbl_{cid}.txt")
        if not (os.path.exists(kf) and os.path.exists(lf)):
            print(f"  skip cid={cid} (missing k/lbl)"); continue
        Q.append(read_q(qf)); K.append(read_k(kf))
        names.append(open(lf, encoding="utf-8").read().split("\t")[0].strip())
        labels.append(len(K) - 1)   # diagonal: query i -> episode i
    Qa = np.empty(len(Q), dtype=object)
    for i, v in enumerate(Q): Qa[i] = v
    Ka = np.empty(len(K), dtype=object)
    for i, v in enumerate(K): Ka[i] = v
    np.savez(OUT, Q=Qa, K=Ka,
             labels=np.array(labels, dtype=np.int64), ep_names=np.array(names, dtype=object))
    print(f"built {OUT}: {len(Q)} queries / {len(K)} episodes (diagonal)", flush=True)


if __name__ == "__main__":
    main()
