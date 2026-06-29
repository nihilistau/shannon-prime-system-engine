#!/usr/bin/env python3
# sp_mh_train.py — MEMORY HEAD trainer (Latent -> 63-byte C2 Spinor). CONTRACT-LATENT-INTERCEPTOR.md.
#
# The Memory Head maps the draft's 1024-d latent directly to the VHT2 content key that
# sp_spinor_encode packs into the FROZEN 63-byte Spinor block (sp/spinor_block.h: scale + 55 int8
# Mobius-permuted anchors + CRC-8). The model writes MEM-OKF from the latent stream, no tokenization.
#
#   memory_head: latent[1024] -> Linear(1024,256) -> ReLU -> Linear(256,55) = the 55-d content key.
#   deploy: key -> sp_spinor_encode (C, in the engine) -> 63-byte block -> MEM-OKF write/address.
#
# TARGET (v1, self-supervised): the content KEY of the event = a FROZEN random projection of the 12B
# feature to 55-d (Johnson-Lindenstrauss: distance-preserving, so similar events -> similar keys ->
# content-addressed recall). The head learns latent -> proj(feat). This proves the latent->Spinor
# route is content-meaningful with NO external labels. v2 = distill the nightshift_curator's actual
# episode Spinor (MEM-OKF fidelity); v3 = contrastive recall objective (cue->memory).
#
# Data: SP_LI_CAPTURE w/ SP_DRAFT_GGUF (latent.f32 [N x1024], feat.f32 [N x3840], label.i32 [N]).
# By default trains on KEEP events (label==1, the memory-worthy class); --all to use every event.
import argparse, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

ANCHORS = 55  # SP_SPINOR_BODY_LEN

def spinor_roundtrip_np(vec):  # mirror sp_spinor_encode/decode (v1 canonical) for a fidelity check
    scale = np.max(np.abs(vec)) or 1.0
    q = np.clip(np.round(vec / scale * 127.0), -127, 127)
    return q / 127.0 * scale  # decode (Mobius perm is index-only -> no effect on values)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", default="mh_head")
    ap.add_argument("--proj", type=int, default=256); ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--val", type=float, default=0.2)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    dev = a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu"
    meta = json.loads(open(f"{a.data}/manifest.jsonl").readline())
    H = meta["hidden"]
    lat = np.fromfile(f"{a.data}/latent.f32", dtype=np.float32).reshape(-1, 1024)
    feat = np.fromfile(f"{a.data}/feat.f32", dtype=np.float32).reshape(-1, H)
    lbl = np.fromfile(f"{a.data}/label.i32", dtype=np.int32)
    if not a.all:
        keep = lbl == 1  # KEEP
        if keep.sum() >= 8: lat, feat = lat[keep], feat[keep]
        else: print(f"[mh-train] only {int(keep.sum())} KEEP samples; using ALL events")
    N = lat.shape[0]
    print(f"[mh-train] N={N} latent=1024 feat={H} anchors={ANCHORS}")

    # FROZEN random projection feat(H)->key(55) (J-L). Saved for deploy (capture-side target).
    rng = np.random.default_rng(163)
    P = (rng.standard_normal((H, ANCHORS)) / np.sqrt(H)).astype(np.float32)
    target = feat @ P                                   # [N,55] content key
    target = target / (np.abs(target).max(1, keepdims=True) + 1e-6)  # normalize to [-1,1] (Spinor range)

    X = torch.tensor(lat, device=dev); Y = torch.tensor(target, device=dev)
    g = torch.Generator().manual_seed(0); perm = torch.randperm(N, generator=g)
    nval = max(1, int(N * a.val)); vi, ti = perm[:nval].to(dev), perm[nval:].to(dev)
    mu, sd = X[ti].mean(0), X[ti].std(0) + 1e-6
    Xn = (X - mu) / sd

    net = nn.Sequential(nn.Linear(1024, a.proj), nn.ReLU(), nn.Linear(a.proj, ANCHORS)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    best, best_state = 1e9, None
    for ep in range(a.epochs):
        net.train(); opt.zero_grad()
        loss = F.mse_loss(net(Xn[ti]), Y[ti]); loss.backward(); opt.step()
        if (ep + 1) % 40 == 0 or ep == a.epochs - 1:
            net.eval()
            with torch.no_grad():
                pred = net(Xn[vi])
                vloss = F.mse_loss(pred, Y[vi]).item()
                # cosine of the decoded-Spinor key vs target (the recall-relevant fidelity)
                pn = F.normalize(pred, dim=1); yn = F.normalize(Y[vi], dim=1)
                cos = (pn * yn).sum(1).mean().item()
            print(f"  ep{ep+1} train_mse={loss.item():.4f} val_mse={vloss:.4f} val_key_cos={cos:.3f}")
            if vloss < best: best = vloss; best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    # spinor roundtrip fidelity on val (does the int8 quantization preserve the key?)
    with torch.no_grad():
        pv = net(Xn[vi]).cpu().numpy()
    rt = np.array([np.dot(spinor_roundtrip_np(k), k) / (np.linalg.norm(spinor_roundtrip_np(k)) * np.linalg.norm(k) + 1e-9) for k in pv])
    print(f"[mh-train] best val_mse={best:.4f} | spinor int8-roundtrip cos={rt.mean():.4f}")

    sd_ = net.state_dict()
    W1, b1 = sd_["0.weight"].cpu().numpy(), sd_["0.bias"].cpu().numpy()
    W2, b2 = sd_["2.weight"].cpu().numpy(), sd_["2.bias"].cpu().numpy()
    blob = np.concatenate([mu.cpu().numpy().ravel(), sd.cpu().numpy().ravel(),
                           W1.ravel(), b1.ravel(), W2.ravel(), b2.ravel()]).astype(np.float32)
    blob.tofile(f"{a.out}.bin"); P.tofile(f"{a.out}.proj.bin")
    json.dump({"in": 1024, "proj": a.proj, "anchors": ANCHORS,
               "layout": ["mu[1024]", "sd[1024]", "W1[proj*1024]", "b1[proj]", "W2[55*proj]", "b2[55]"],
               "target": "frozen-randproj feat->55 (J-L); deploy: key->sp_spinor_encode->63B block",
               "val_mse": best}, open(f"{a.out}.json", "w"), indent=2)
    print(f"[mh-train] wrote {a.out}.bin (+ .proj.bin, .json) -> CUDA memory-head: latent->key->sp_spinor_encode")

if __name__ == "__main__":
    main()
