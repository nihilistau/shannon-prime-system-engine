#!/usr/bin/env python3
"""B3-v3 export + INTEGER separation gate (G-CHAT-B3-RECALL-v3, gate #1).

1. Quantize the trained float W_c -> int16 mantissa + power-of-two scale 2^k
   (deterministic round). Write lsh_Wc_i16_sK.bin in the bscale convention.
2. Quant-error check: max |Wc_int/2^k - Wc_f32|.
3. Re-score the recall matrix ENTIRELY in the integer domain (q,K quantized to
   fixed-point 2^kq; projection in int64; the final r-dim dot in Python big-int so it
   is EXACT and reduction-order-immune — the host stand-in for the engine's dual-prime
   CRT path) with the SAME reduction recall.rs uses (max + top-m mean, m=8).
4. GATE: min_target_topm > max_foreign_topm with a positive margin, on the INTEGER
   scores. Float separation does not count — only the O_K-container separation does.

Exit 0 = GREEN (gap survives quantization), 1 = RED (honest negative).
"""
import os, sys, struct, argparse
import numpy as np

HD, G_NH = 512, 16
WC_MAGIC = 0x57433149  # 'WC1I'


def quantize(Wc, want_bits=16):
    amax = float(np.abs(Wc).max())
    if amax == 0:
        return np.zeros_like(Wc, np.int16), 0
    lim = (1 << (want_bits - 1)) - 1     # 32767
    k = int(np.floor(np.log2(lim / amax)))
    Wi = np.round(Wc * (2.0 ** k)).astype(np.int64)
    Wi = np.clip(Wi, -lim - 1, lim).astype(np.int16)
    return Wi, k


def write_bin(path, Wi, k):
    rows, cols = Wi.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<6i", WC_MAGIC, 1, rows, cols, 1, k))  # dtype 1 = i16
        f.write(np.ascontiguousarray(Wi.astype("<i2")).tobytes())


def int_relevance(qi, Ke, Wi, kq, m=8):
    """Exact integer relevance, the recall.rs reduction. qi:[ng,G_NH,HD] Ke:[ng,np,HD]
    f32; Wi:[HD,r] int16. Returns (max, topm) as Python ints (scale-free ranking)."""
    ng = min(qi.shape[0], Ke.shape[0])
    s = float(2 ** kq)
    qf = np.round(qi[:ng] * s).astype(np.int64)       # [ng,G_NH,HD]
    kf = np.round(Ke[:ng] * s).astype(np.int64)       # [ng,np,HD]
    Wl = Wi.astype(np.int64)                          # [HD,r]
    scores = []
    for l in range(ng):
        pq = qf[l] @ Wl                               # [G_NH,r] int64 (no overflow)
        pk = kf[l] @ Wl                               # [np,r]   int64
        # exact final dot in big-int: [G_NH,np] = pq(obj) @ pk(obj).T
        M = pq.astype(object) @ pk.astype(object).T   # [G_NH,np] Python ints, exact
        scores.append(M.reshape(-1))
    alls = np.concatenate(scores)                     # [ng*G_NH*np] object
    mx = max(alls)
    if alls.size <= m:
        topm = sum(alls) / len(alls)
    else:
        top = sorted(alls)[-m:]
        topm = sum(top) / m
    return mx, topm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wc", default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--kq", type=int, default=int(os.environ.get("WC_KQ", "10")))
    ap.add_argument("--margin_frac", type=float, default=0.0,
                    help="required (min_target-max_foreign)/|max_foreign| > this")
    args = ap.parse_args()
    eng = os.environ.get("SP_ENGINE_DIR", r"D:\F\shannon-prime-repos\shannon-prime-system-engine")
    wcf = args.wc or os.path.join(eng, "_b3_wc", "lsh_Wc_f32.npz")
    data = args.data or os.path.join(eng, "_b3_wc", "b3_data.npz")

    z = np.load(wcf, allow_pickle=True)
    Wc = z["Wc"].astype(np.float32); r = int(z["r"])
    Wi, k = quantize(Wc, 16)
    qerr = float(np.abs(Wi.astype(np.float64) / (2.0 ** k) - Wc).max())
    outbin = os.path.join(eng, "_b3_wc", f"lsh_Wc_i16_s{k}.bin")
    os.makedirs(os.path.dirname(outbin), exist_ok=True)
    write_bin(outbin, Wi, k)
    print(f"[exp] Wc {Wc.shape} -> int16 scale 2^{k}; max quant-err={qerr:.3e}; wrote {outbin}", flush=True)

    d = np.load(data, allow_pickle=True)
    Q = [np.asarray(q, np.float32) for q in d["Q"]]
    K = [np.asarray(kk, np.float32) for kk in d["K"]]
    labels = d["labels"].astype(np.int64); names = list(d["ep_names"]); E = len(K)

    print("\n[exp] INTEGER relevance matrix (topm, exact O_K-domain scoring):", flush=True)
    tgt, fgn = [], []
    log = [f"G-CHAT-B3-RECALL-v3 gate#1 (integer scores)  W_c int16 scale 2^{k}  quant-err {qerr:.3e}",
           f"r={r} kq={args.kq}  reduction = max + top-8 mean (recall.rs)"]
    for i in range(len(Q)):
        row = [int_relevance(Q[i], K[e], Wi, args.kq)[1] for e in range(E)]  # topm per ep
        arg = int(np.argmax([float(x) for x in row]))
        lab = int(labels[i])
        # normalize for readable print (ranking is scale-free)
        mxabs = max(1.0, max(abs(float(x)) for x in row))
        disp = " ".join(f"{names[e]}={float(row[e])/mxabs:+.3f}" for e in range(E))
        if lab >= 0:
            tgt.append(float(row[lab])); tag = f"want={names[lab]} {'OK' if arg==lab else 'WRONG('+names[arg]+')'}"
        else:
            fgn.append(max(float(x) for x in row)); tag = "FOREIGN(reject)"
        line = f"  {disp}   {tag}"
        print(line, flush=True); log.append(line)

    mt = min(tgt) if tgt else float("nan")
    mf = max(fgn) if fgn else float("nan")
    sep = mt > mf and (mf <= 0 or (mt - mf) / abs(mf) > args.margin_frac)
    verdict = (f"\n[exp] min_target={mt:.4g}  max_foreign={mf:.4g}  "
               f"=> {'SEPARATES (GREEN)' if sep else 'NO SEPARATION (RED)'}")
    print(verdict, flush=True); log.append(verdict)
    recpath = os.path.join(eng, "tests", "fixtures", "chat_fullstack", "G-CHAT-B3-RECALL-v3.log")
    with open(recpath, "w", encoding="utf-8") as f:
        f.write("\n".join(log) + "\n")
    print(f"[exp] receipt -> {recpath}", flush=True)
    sys.exit(0 if sep else 1)


if __name__ == "__main__":
    main()
