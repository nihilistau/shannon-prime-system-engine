#!/usr/bin/env python3
"""§3q Phase B — Learned LSH: train a shared 512xr projection R so top-B by (Rq).(RK)
matches the oracle top-B by exact q.K. The oracle proved 8x is learnable (PPL -0.08%);
the frozen +/-1 router fails (+4.17%) only because it ranks the wrong 256. Train R to
rank like full q.K, at the SAME r=32 inference cost.

Loss = forward-KL(a_true || a_proj) [mass-weighted by construction -> ignores the diffuse
noise tail that fooled the mass-captured proxy] + 0.2 * hard-negative hinge [punish a tail
key outscoring a real key in projection = the exact frozen-router failure]. Learnable tau.

Data: SP_ARM_DUMP corpus kq_call{0,1,2}.bin (post-RoPE per-pos K[512] + q[16x512] on the 8
globals, N=2048/window). TRAIN windows 0,1 ; strict VAL on held-out window 2.
Metric: mass-recall@B = sum of true attention mass captured by the learned top-B (the
PPL-relevant quantity; oracle ceiling = 0.923 @ 8x / 0.967 @ 4x).
"""
import sys, os, glob, struct, time
import numpy as np
import torch
import torch.nn.functional as F

DUMP = sys.argv[1] if len(sys.argv) > 1 else r"D:\F\shannon-prime-repos\_xbar\p2b\kqdump3w"
R_DIM   = int(os.environ.get("LSH_R", "32"))
EPOCHS  = int(os.environ.get("LSH_EPOCHS", "8"))
BATCH   = int(os.environ.get("LSH_BATCH", "128"))
STEPS   = int(os.environ.get("LSH_STEPS", "400"))     # sampled steps per epoch
AUXW    = float(os.environ.get("LSH_AUX", "0.2"))
LR      = float(os.environ.get("LSH_LR", "3e-3"))
MINCTX  = 64                                           # only train queries with >= MINCTX keys
dev = "cuda" if torch.cuda.is_available() else "cpu"

def load_window(path):
    f = open(path, "rb")
    magic, ver, NL, period, g_nh, g_nkv, g_hd, n_prompt = struct.unpack("<8i", f.read(32))
    kvd, qd = g_nkv * g_hd, g_nh * g_hd
    buckets = {}
    while True:
        hdr = f.read(24)
        if len(hdr) < 24: break
        rmagic, L, pos, nkv, nh, hd = struct.unpack("<6i", hdr)
        K = np.frombuffer(f.read(kvd*4), np.float32).copy()
        q = np.frombuffer(f.read(qd*4), np.float32).copy()
        b = buckets.setdefault(L, {}); b[pos] = (K, q)
    f.close()
    out = {}
    for L, d in buckets.items():
        P = max(d) + 1
        Ka = np.zeros((P, kvd), np.float32); qa = np.zeros((P, g_nh, g_hd), np.float32)
        for pos, (K, q) in d.items(): Ka[pos] = K; qa[pos] = q.reshape(g_nh, g_hd)
        out[L] = (torch.from_numpy(Ka), torch.from_numpy(qa))
    return out, g_nh, g_hd

def main():
    calls = sorted(glob.glob(os.path.join(DUMP, "kq_call*.bin")))
    assert len(calls) >= 3, f"need 3 windows, got {calls}"
    print(f"[lsh] dev={dev} r={R_DIM} epochs={EPOCHS} batch={BATCH} steps={STEPS} aux={AUXW}")
    wins = []
    for c in calls:
        w, nh, hd = load_window(c); wins.append(w)
    print(f"[lsh] loaded {len(wins)} windows, nh={nh} hd={hd}, layers={sorted(wins[0])}")
    train_w, val_w = [wins[0], wins[1]], wins[2]
    layers = sorted(wins[0]); N = train_w[0][layers[0]][0].shape[0]

    # move tensors to device
    def to_dev(W):
        return {L: (K.to(dev), q.to(dev)) for L, (K, q) in W.items()}
    train_w = [to_dev(W) for W in train_w]; val_w = to_dev(val_w)

    R = torch.nn.Parameter((torch.randn(hd, R_DIM, device=dev) / (hd**0.5)))
    log_tau = torch.nn.Parameter(torch.zeros((), device=dev))
    opt = torch.optim.Adam([R, log_tau], lr=LR)
    rng = np.random.default_rng(20260613)

    def scores(qb, K, idx):
        # causal-masked true + proj scores. qb:[B,hd] K:[N,hd] idx:[B] -> St,Sp,mask
        Bn = qb.shape[0]; Nn = K.shape[0]
        St = qb @ K.t()                                  # [B,N] exact q.K (model scale 1.0)
        Kp = K @ R; qp = qb @ R                          # [N,r],[B,r]
        Sp = (qp @ Kp.t()) / torch.exp(log_tau)          # [B,N] projected, tau-scaled
        ar = torch.arange(Nn, device=dev)[None, :]
        mask = ar <= idx[:, None]                        # causal
        St = St.masked_fill(~mask, -1e9); Sp = Sp.masked_fill(~mask, -1e9)
        return St, Sp, mask

    for ep in range(EPOCHS):
        t0 = time.time(); klsum = auxsum = 0.0
        for st in range(STEPS):
            W = train_w[rng.integers(2)]; L = layers[rng.integers(len(layers))]; h = rng.integers(nh)
            K, qa = W[L]; q = qa[:, h, :]
            idx = torch.from_numpy(rng.integers(MINCTX, N, size=BATCH)).to(dev)
            qb = q[idx]
            St, Sp, mask = scores(qb, K, idx)
            a_true = torch.softmax(St, dim=1)
            logp   = torch.log_softmax(Sp, dim=1)
            kl = -(a_true * logp).sum(1).mean()          # forward-KL (drop const H(a_true))
            # hard-negative hinge: weakest oracle-top-B proj score vs strongest non-oracle proj
            with torch.no_grad():
                B_ = 256
                topv, topi = a_true.topk(B_, dim=1)      # oracle top-B (by true mass)
            posmask = torch.zeros_like(Sp, dtype=torch.bool).scatter_(1, topi, True)
            pos_s = Sp.masked_fill(~posmask, 1e9).min(1).values   # weakest positive proj
            neg_s = Sp.masked_fill(posmask | ~mask, -1e9).max(1).values  # strongest hard-neg proj
            aux = F.relu(neg_s - pos_s + 1.0).mean()
            loss = kl + AUXW * aux
            opt.zero_grad(); loss.backward(); opt.step()
            klsum += kl.item(); auxsum += aux.item()
        print(f"[lsh] ep{ep} KL={klsum/STEPS:.4f} aux={auxsum/STEPS:.4f} tau={torch.exp(log_tau).item():.3f} ({time.time()-t0:.1f}s)")

    # ---- VAL on held-out window 2: mass-recall@B for learned R, frozen +/-1, oracle ----
    print("\n=== VAL (held-out window 2) mass-recall = true attention mass kept by top-B ===")
    Rf = (torch.randint(0, 2, (hd, R_DIM), device=dev).float()*2 - 1)   # frozen +/-1 baseline
    def massrecall(projR, Bset):
        tot = {b: [] for b in Bset}; setrec = {b: [] for b in Bset}
        for L in layers:
            K, qa = val_w[L]
            for h in range(nh):
                q = qa[:, h, :]
                idx = torch.arange(MINCTX, N, 8, device=dev)   # subsample val positions
                qb = q[idx]
                St = qb @ K.t()
                ar = torch.arange(N, device=dev)[None, :]; mask = ar <= idx[:, None]
                St = St.masked_fill(~mask, -1e9); a_true = torch.softmax(St, 1)
                if projR is None:
                    Sp = St                                  # oracle = exact q.K
                else:
                    Sp = (qb @ projR) @ (K @ projR).t()
                Sp = Sp.masked_fill(~mask, -1e9)
                for b in Bset:
                    pi = Sp.topk(b, dim=1).indices
                    mr = a_true.gather(1, pi).sum(1)         # mass kept
                    tot[b].append(mr.mean().item())
                    oi = a_true.topk(b, dim=1).indices
                    # set-recall vs oracle top-b
                    inter = (pi.unsqueeze(2) == oi.unsqueeze(1)).any(2).float().sum(1) / b
                    setrec[b].append(inter.mean().item())
        return {b: float(np.mean(tot[b])) for b in Bset}, {b: float(np.mean(setrec[b])) for b in Bset}
    for name, projR in (("ORACLE (exact q.K)", None), ("LEARNED R (r=%d)"%R_DIM, R.detach()), ("FROZEN +/-1", Rf)):
        mr, sr = massrecall(projR, [256, 512])
        print(f"  {name:22} mass@256={mr[256]:.4f} mass@512={mr[512]:.4f}  setrec@256={sr[256]:.3f}")
    # save R + tau for the engine
    np.savez(os.path.join(DUMP, "..", "lsh_R_r%d.npz"%R_DIM),
             R=R.detach().cpu().numpy(), tau=float(torch.exp(log_tau).item()), r=R_DIM)
    print(f"\n[lsh] saved lsh_R_r{R_DIM}.npz")

if __name__ == "__main__":
    torch.manual_seed(20260613)
    main()
