#!/usr/bin/env python3
# sp_mh_train.py — MEMORY HEAD trainer v2 (CURATOR DISTILLATION). CONTRACT-LATENT-INTERCEPTOR.md.
#
# Target = the curator's C2 256-bit LSH signature (recall.rs Projection): sig[b] = sign(R[b]·pooled_K),
# R = frozen +/-1 router smix(SEED, 256*512), pooled_K = sum over (global-layer, pos) of K[512].
# The Memory Head reconstructs pooled_K from the draft's 1024-d latent; the FROZEN R then produces the
# byte-identical address. Objective: minimize Hamming distance to the curator sig (exact-integer space,
# no float noise, no random projection). The head learns to think in the memory-addressing geometry.
#
#   memory_head: latent[1024] -> Linear(1024,512) -> ReLU -> Linear(512,512) = pooled_K_est (unit).
#   deploy: pooled_K_est -> sign(R @ pooled_K_est) -> 256-bit C2 sig -> MEM-OKF addr (c2sig_hex).
#
# Data: SP_LI_CAPTURE w/ SP_DRAFT_GGUF: latent.f32 [N x1024], pooledk.f32 [N x512], sig.u64 [N x4].
import argparse, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

SEED = 0x5350524F4A2B; R_BITS = 256; HD = 512; TAU_BITS = 168  # recall.rs constants
M64 = (1 << 64) - 1

def smix_pm1(seed, n):  # splitmix64 +/-1 — byte-identical to recall.rs smix / discrete_resolve.build_R
    s = seed & M64; out = np.empty(n, np.float32)
    for i in range(n):
        s = (s + 0x9E3779B97F4A7C15) & M64
        z = s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
        z ^= z >> 31
        out[i] = 1.0 if (z & 1) else -1.0
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", default="mh_head")
    ap.add_argument("--epochs", type=int, default=600); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val", type=float, default=0.2); ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu"
    lat = np.fromfile(f"{a.data}/latent.f32", dtype=np.float32).reshape(-1, 1024)
    pk = np.fromfile(f"{a.data}/pooledk.f32", dtype=np.float32).reshape(-1, HD)
    sigw = np.fromfile(f"{a.data}/sig.u64", dtype=np.uint64).reshape(-1, 4)
    N = lat.shape[0]
    # unpack the curator sig -> target bits [N,256]
    bits = np.zeros((N, R_BITS), np.float32)
    for b in range(R_BITS): bits[:, b] = ((sigw[:, b // 64] >> np.uint64(b % 64)) & np.uint64(1)).astype(np.float32)
    R = torch.tensor(smix_pm1(SEED, R_BITS * HD).reshape(R_BITS, HD), device=dev)  # [256,512] frozen
    # sanity: the captured pooled_K reproduces the captured sig exactly (curator parity)
    pkt = torch.tensor(pk, device=dev)
    sig_from_pk = (pkt @ R.t() > 0).float()
    parity = (sig_from_pk.cpu().numpy() == bits).mean()
    print(f"[mh-train] N={N} | pooled_K->R->sig parity vs captured sig = {parity:.4f} (should be ~1.0)")

    pkn = F.normalize(pkt, dim=1)                       # unit pooled_K (sign-invariant) = head target
    X = torch.tensor(lat, device=dev); B = torch.tensor(bits, device=dev)
    g = torch.Generator().manual_seed(0); perm = torch.randperm(N, generator=g)
    nval = max(1, int(N * a.val)); vi, ti = perm[:nval].to(dev), perm[nval:].to(dev)
    mu, sd = X[ti].mean(0), X[ti].std(0) + 1e-6
    Xn = (X - mu) / sd

    net = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, HD)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    def hamming_agree(pred_pk, tb):  # bits agreeing with curator sig, out of 256
        s = (pred_pk @ R.t() > 0).float()
        return (s == tb).float().sum(1)  # [n] agreement count
    best, best_state = -1.0, None
    for ep in range(a.epochs):
        net.train(); opt.zero_grad()
        pkh = net(Xn[ti])
        logits = pkh @ R.t()                            # [n,256] projected (the sign decides the bit)
        bce = F.binary_cross_entropy_with_logits(logits, B[ti])
        mse = F.mse_loss(F.normalize(pkh, dim=1), pkn[ti])
        loss = bce + 0.5 * mse
        loss.backward(); opt.step()
        if (ep + 1) % 60 == 0 or ep == a.epochs - 1:
            net.eval()
            with torch.no_grad():
                ag = hamming_agree(net(Xn[vi]), B[vi])
            agree = ag.mean().item(); recall = (ag >= TAU_BITS).float().mean().item()
            print(f"  ep{ep+1} loss={loss.item():.3f} val_bit_agree={agree:.1f}/256 recall@{TAU_BITS}={recall:.3f}")
            if agree > best: best = agree; best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad(): ag = hamming_agree(net(Xn[vi]), B[vi])
    print(f"[mh-train] BEST val_bit_agree={ag.mean().item():.1f}/256 (Hamming dist {256-ag.mean().item():.1f}) | recall@{TAU_BITS}={(ag>=TAU_BITS).float().mean().item():.3f} | exact256={(ag==256).float().mean().item():.3f}")

    sd_ = net.state_dict()
    W1, b1 = sd_["0.weight"].cpu().numpy(), sd_["0.bias"].cpu().numpy()
    W2, b2 = sd_["2.weight"].cpu().numpy(), sd_["2.bias"].cpu().numpy()
    blob = np.concatenate([mu.cpu().numpy().ravel(), sd.cpu().numpy().ravel(),
                           W1.ravel(), b1.ravel(), W2.ravel(), b2.ravel()]).astype(np.float32)
    blob.tofile(f"{a.out}.bin")
    json.dump({"in": 1024, "hidden": 512, "out": HD, "R_bits": R_BITS, "tau_bits": TAU_BITS, "seed": hex(SEED),
               "layout": ["mu[1024]", "sd[1024]", "W1[512*1024]", "b1[512]", "W2[512*512]", "b2[512]"],
               "deploy": "latent->pooled_K_est->sign(R@pk)->C2 sig->MEM-OKF (R via recall::Projection)",
               "val_bit_agree": ag.mean().item()}, open(f"{a.out}.json", "w"), indent=2)
    print(f"[mh-train] wrote {a.out}.bin (+ .json) -> Rust mh_probe: latent->pooled_K, recall::Projection->sig")

if __name__ == "__main__":
    main()
