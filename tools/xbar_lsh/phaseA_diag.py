#!/usr/bin/env python3
"""§3q Phase A — the oracle-ceiling diagnostic (training-free, decisive go/no-go).

Reads the SP_ARM_DUMP corpus (post-RoPE per-position K,q on the 8 gemma-4-12B global
owners, captured bit-faithfully: the engine's decode attention uses scale=1.0 on the
exact dq/dk we dumped, so softmax(q.K) here == the model's true global attention weights).

The b-2b 8x failure (+4.17% PPL deflection, frozen +/-1 router) is "the router keeps the
WRONG 256 of 2048 globals." Before training a new per-position addresser to keep the RIGHT
256, we ask the cheaper question: CAN any selector keep 256 and preserve the distribution?
=> the ORACLE = keep the exact top-B-by-(q.K) positions. Its attention-mass-captured is the
ceiling no learned shortlister can beat. (V-free proxy: dropped softmax mass bounds the
output perturbation; gemma globals are V=f(K) so mass-captured is a faithful deflection proxy.)

Reports, per compression ratio (4x B=512, 8x B=256), over the scored half [N/2,N) of every
window x layer x query-head:
  - oracle mass-captured  (mean / p10 / min) = the ceiling
  - random-B mass-captured = the floor (chance selector)
  - recent-W + sink only   = what a no-router windowed baseline gets
  - top-1 attention position in-budget rate
Verdict heuristic: oracle mean mass-captured >= ~0.99 at 8x => learnable (frozen router is the
bottleneck, train the head). Oracle materially < 0.99 at 8x => information-bounded, concede 4x.
"""
import sys, os, glob, struct
import numpy as np

DUMP = sys.argv[1] if len(sys.argv) > 1 else r"D:\F\shannon-prime-repos\_xbar\p2b\kqdump3w"
BUDGETS = {"4x": 512, "8x": 256}
SINK = 2          # b-2b locked sink count
WREC = 64         # the W-probe optimum, for the windowed-baseline comparison

def load_call(path):
    """-> dict L -> {'K':[P,kvd], 'q':[P,nh,hd]} for global layers in one window."""
    f = open(path, "rb")
    magic, ver, NL, period, g_nh, g_nkv, g_hd, n_prompt = struct.unpack("<8i", f.read(32))
    assert magic == 0x4651444B, f"bad file magic {magic:#x}"
    kvd, qd = g_nkv * g_hd, g_nh * g_hd
    buckets = {}
    while True:
        hdr = f.read(24)
        if len(hdr) < 24:
            break
        rmagic, L, pos, nkv, nh, hd = struct.unpack("<6i", hdr)
        assert rmagic == 0x504B5251, f"bad rec magic {rmagic:#x}"
        K = np.frombuffer(f.read(kvd * 4), dtype=np.float32).copy()
        q = np.frombuffer(f.read(qd * 4), dtype=np.float32).copy()
        b = buckets.setdefault(L, {})
        b[pos] = (K, q)
    f.close()
    out = {}
    for L, d in buckets.items():
        P = max(d) + 1
        Karr = np.zeros((P, kvd), np.float32)
        qarr = np.zeros((P, g_nh, g_hd), np.float32)
        for pos, (K, q) in d.items():
            Karr[pos] = K
            qarr[pos] = q.reshape(g_nh, g_hd)
        out[L] = {"K": Karr, "q": qarr, "nh": g_nh, "hd": g_hd, "nkv": g_nkv, "n_prompt": n_prompt}
    return out, NL, period

def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()

def main():
    calls = sorted(glob.glob(os.path.join(DUMP, "kq_call*.bin")))
    if not calls:
        print("NO DUMP FILES in", DUMP); sys.exit(1)
    print(f"[phaseA] {len(calls)} window(s): {[os.path.basename(c) for c in calls]}")

    # accumulators: per budget -> list of mass-captured; plus baselines
    acc = {b: {"oracle": [], "rand": [], "winsink": [], "top1in": []} for b in BUDGETS}
    n_samples = 0
    for ci, c in enumerate(calls):
        layers, NL, period = load_call(c)
        for L in sorted(layers):
            d = layers[L]
            K, q, nh = d["K"], d["q"], d["nh"]
            P = K.shape[0]
            lo = P // 2                     # scored half [N/2, N), matches the PPL gate
            for pos in range(lo, P, 8):     # stride 8: thousands of samples, fast
                ctx = pos + 1
                Kc = K[:ctx]                          # [ctx, hd]
                S = Kc @ q[pos].T                      # [ctx, nh] exact q.K per head (scale=1.0)
                S = S - S.max(axis=0, keepdims=True)
                A = np.exp(S); A /= A.sum(axis=0, keepdims=True)   # [ctx, nh] softmax per head
                As = np.sort(A, axis=0)[::-1]          # descending mass per head, [ctx, nh]
                top1mass = As[0]                       # top-1 attention weight per head
                for bn, B in BUDGETS.items():
                    b_eff = min(B, ctx)
                    oracle = As[:b_eff].sum(axis=0)            # [nh] mass of exact top-B
                    # recent-W+sink baseline: positions [0,sink) U [ctx-(B-sink), ctx)
                    keep = np.zeros(ctx, bool)
                    keep[:min(SINK, ctx)] = True
                    keep[max(0, ctx - (B - SINK)):] = True
                    winsink = A[keep].sum(axis=0)             # [nh]
                    # random-B floor
                    if ctx <= B:
                        rnd = np.ones(nh)
                    else:
                        ridx = np.random.choice(ctx, B, replace=False)
                        rnd = A[ridx].sum(axis=0)
                    acc[bn]["oracle"].extend(oracle.tolist())
                    acc[bn]["rand"].extend(rnd.tolist())
                    acc[bn]["winsink"].extend(winsink.tolist())
                    acc[bn]["top1in"].extend((np.ones(nh)).tolist())  # oracle top-B always holds the top-1
                n_samples += nh
        print(f"[phaseA] window {ci} done ({n_samples} head-pos samples so far)")

    print(f"\n=== ORACLE-CEILING DIAGNOSTIC  (n={n_samples} head-position samples) ===")
    print(f"{'ratio':>5} {'selector':>10} {'mass_mean':>10} {'mass_p10':>9} {'mass_min':>9} {'top1_in%':>9}")
    for bn in BUDGETS:
        for sel in ("oracle", "winsink", "rand"):
            arr = np.array(acc[bn][sel])
            t1 = np.array(acc[bn]["top1in"]).mean() * 100 if sel == "oracle" else float("nan")
            print(f"{bn:>5} {sel:>10} {arr.mean():>10.5f} {np.percentile(arr,10):>9.4f} "
                  f"{arr.min():>9.4f} {('%.1f'%t1) if sel=='oracle' else '':>9}")
    print("\nREAD: oracle mass_mean ~>=0.99 @ 8x => the keepable mass exists, frozen router is the "
          "bottleneck (a learned per-position head is justified). oracle materially <0.99 @ 8x => "
          "8x is information-bounded on globals; no shortlister recovers it, concede 4x.")

if __name__ == "__main__":
    np.random.seed(424242)
    main()
