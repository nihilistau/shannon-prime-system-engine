#!/usr/bin/env python3
# sp_li_train.py — Latent Interceptor trainer (CONTRACT-LATENT-INTERCEPTOR.md).
# The KAIROS decision is a CLASSIFICATION of the 12B's frame-end feature (post-output_norm hidden),
# not a future-token prediction. So the interceptor is a TINY probe on the latent directly:
#   feature[hidden] -> Linear(hidden, P) -> ReLU -> Linear(P, A) -> action logits
# NO 262144 vocab head, no tokenization. Deploy cost = microseconds (a 3840->256->A MLP) after the
# unavoidable 12B frame prefill. This SUBSUMES the draft-body version: the 12B feature is already the
# maximally-processed latent, so re-running the 4-layer draft on top adds no information for the
# CURRENT decision (the draft body is for FUTURE-token prediction; here we classify the present).
#
# Data: SP_LI_CAPTURE output (feat.f32 [N x hidden], label.i32 [N], manifest.jsonl).
# Export: li_head.bin (W1,b1,W2,b2 f32) + li_head.json (dims) for the CUDA deploy.
import argparse, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", default="li_head")
    ap.add_argument("--proj", type=int, default=256)   # probe hidden width
    ap.add_argument("--epochs", type=int, default=200); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val", type=float, default=0.2); ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu"
    meta = json.loads(open(f"{a.data}/manifest.jsonl").readline())
    H, A = meta["hidden"], meta["n_actions"]; actions = meta["actions"]
    feat = torch.tensor(np.fromfile(f"{a.data}/feat.f32", dtype=np.float32).reshape(-1, H), device=dev)
    lbl  = torch.tensor(np.fromfile(f"{a.data}/label.i32", dtype=np.int32).astype(np.int64), device=dev)
    N = lbl.shape[0]
    print(f"[li-train] N={N} hidden={H} actions={actions} dist={[int((lbl==i).sum()) for i in range(A)]}")

    # deterministic split
    g = torch.Generator().manual_seed(0); perm = torch.randperm(N, generator=g)
    nval = max(1, int(N * a.val)); vidx, tidx = perm[:nval].to(dev), perm[nval:].to(dev)
    # feature normalization (store mean/std for deploy)
    mu, sd = feat[tidx].mean(0), feat[tidx].std(0) + 1e-6
    fn = (feat - mu) / sd

    net = nn.Sequential(nn.Linear(H, a.proj), nn.ReLU(), nn.Linear(a.proj, A)).to(dev)
    # class weights (idle-dominated) so NO_OP doesn't swamp the rare actionable classes
    cnt = torch.tensor([max(1, int((lbl[tidx] == i).sum())) for i in range(A)], dtype=torch.float32, device=dev)
    cw = (cnt.sum() / (A * cnt))
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    best = 0.0; best_state = None
    for ep in range(a.epochs):
        net.train(); opt.zero_grad()
        loss = F.cross_entropy(net(fn[tidx]), lbl[tidx], weight=cw)
        loss.backward(); opt.step()
        if (ep + 1) % 20 == 0 or ep == a.epochs - 1:
            net.eval()
            with torch.no_grad():
                vp = net(fn[vidx]).argmax(1)
                acc = (vp == lbl[vidx]).float().mean().item()
                # NO_OP precision (id 0) + ACTION recall (id A-1) — the two that matter for compute+safety
                noop_p = ((vp == 0) & (lbl[vidx] == 0)).sum().item() / max(1, (vp == 0).sum().item())
                act_r = ((vp == A-1) & (lbl[vidx] == A-1)).sum().item() / max(1, (lbl[vidx] == A-1).sum().item())
            print(f"  ep{ep+1} loss={loss.item():.3f} val_acc={acc:.3f} NO_OP_prec={noop_p:.3f} ACTION_recall={act_r:.3f}")
            if acc >= best: best = acc; best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    # confusion on val
    net.eval()
    with torch.no_grad(): vp = net(fn[vidx]).argmax(1).cpu().numpy()
    vt = lbl[vidx].cpu().numpy()
    conf = np.zeros((A, A), int)
    for t, p in zip(vt, vp): conf[t, p] += 1
    print("[li-train] val confusion (rows=true, cols=pred):");
    print("           " + " ".join(f"{x[:5]:>6}" for x in actions))
    for i in range(A): print(f"  {actions[i]:>9} " + " ".join(f"{conf[i,j]:>6}" for j in range(A)))

    # export head + normalization for the CUDA deploy
    sd_ = net.state_dict()
    W1, b1 = sd_["0.weight"].cpu().numpy(), sd_["0.bias"].cpu().numpy()
    W2, b2 = sd_["2.weight"].cpu().numpy(), sd_["2.bias"].cpu().numpy()
    blob = np.concatenate([mu.cpu().numpy().ravel(), sd.cpu().numpy().ravel(),
                           W1.ravel(), b1.ravel(), W2.ravel(), b2.ravel()]).astype(np.float32)
    blob.tofile(f"{a.out}.bin")
    json.dump({"hidden": H, "proj": a.proj, "n_actions": A, "actions": actions,
               "layout": ["mu[H]", "sd[H]", "W1[proj*H]", "b1[proj]", "W2[A*proj]", "b2[A]"],
               "val_acc": best}, open(f"{a.out}.json", "w"), indent=2)
    print(f"[li-train] best val_acc={best:.3f} -> {a.out}.bin (+ .json)")

if __name__ == "__main__":
    main()
