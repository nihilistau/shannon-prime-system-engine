#!/usr/bin/env python3
"""b3_train_wc_fast.py -- VECTORIZED B3 contrastive trainer. Mathematically identical to
b3_train_wc.py (InfoNCE over [S_e..., s0]/tau + 0.2 hard-neg hinge, learnable s0=TAU) but
batches ALL queries at once and loops only over the E episodes per epoch (the original
recomputed K[e]@Wc inside the per-query x per-episode loop = 861x201 ~ 173K projections/epoch
~ 40min/epoch at scale; this is ~E projections/epoch). Same save format.

Relevance(q,e) = mean_{l,h} logsumexp_p( (q[l,h]@Wc) . (K_e[l,p]@Wc) ) / sqrt(r)
"""
import os, sys, argparse
import numpy as np, torch, torch.nn.functional as F
HD, G_NH = 512, 16

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="?", default=None)
    ap.add_argument("--r", type=int, default=int(os.environ.get("WC_R","32")))
    ap.add_argument("--epochs", type=int, default=int(os.environ.get("WC_EPOCHS","400")))
    ap.add_argument("--lr", type=float, default=float(os.environ.get("WC_LR","3e-3")))
    ap.add_argument("--hinge", type=float, default=float(os.environ.get("WC_HINGE","0.2")))
    ap.add_argument("--margin", type=float, default=float(os.environ.get("WC_MARGIN","1.0")))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    eng = os.environ.get("SP_ENGINE_DIR", r"D:\F\shannon-prime-repos\shannon-prime-system-engine")
    data = args.data or os.path.join(eng,"_b3_wc","b3_data.npz")
    out  = args.out  or os.path.join(eng,"_b3_wc","lsh_Wc_f32.npz")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(20260619)
    d = np.load(data, allow_pickle=True)
    names = list(d["ep_names"]); E = len(d["K"])
    labels = torch.tensor(d["labels"].astype(np.int64), device=dev)
    # Q: all same shape [ng,G_NH,HD] -> stack [Nq,ng,G_NH,HD]
    Qs = torch.tensor(np.stack([np.asarray(q,np.float32) for q in d["Q"]]), device=dev)
    Nq, ng = Qs.shape[0], Qs.shape[1]
    # K per episode (ragged npos) -> keep list of [ng,npos,HD] tensors (project per-epoch)
    Ks = [torch.tensor(np.asarray(k,np.float32), device=dev) for k in d["K"]]
    ng = min(ng, min(int(k.shape[0]) for k in Ks))
    NULL = E
    print(f"[wcf] dev={dev} r={args.r} queries={Nq} episodes={E} ng={ng} "
          f"({int((labels>=0).sum())} pos / {int((labels<0).sum())} foreign)", flush=True)
    Wc = torch.nn.Parameter(torch.randn(HD, args.r, device=dev)/(HD**0.5))
    log_tau = torch.nn.Parameter(torch.zeros((), device=dev))
    s0 = torch.nn.Parameter(torch.zeros((), device=dev))
    opt = torch.optim.Adam([Wc, log_tau, s0], lr=args.lr)
    scale = 1.0/(args.r**0.5)
    pos_mask = labels >= 0
    tgt = labels.clone(); tgt[~pos_mask] = NULL

    def all_scores():
        # qp_all [Nq,ng,G_NH,r]
        qp = torch.einsum("qlhd,dr->qlhr", Qs[:, :ng], Wc)
        cols = []
        for e in range(E):
            kp = torch.einsum("lpd,dr->lpr", Ks[e][:ng], Wc)          # [ng,np,r]
            sim = torch.einsum("qlhr,lpr->qlhp", qp, kp) * scale       # [Nq,ng,G_NH,np]
            a = torch.logsumexp(sim, dim=3)                            # [Nq,ng,G_NH]
            cols.append(a.mean(dim=(1,2)))                             # [Nq]
        return torch.stack(cols, dim=1)                               # [Nq,E]

    for ep in range(args.epochs):
        opt.zero_grad()
        S = all_scores()                                              # [Nq,E]
        logits = torch.cat([S, s0.expand(Nq,1)], dim=1) / torch.exp(log_tau)  # [Nq,E+1]
        ce = F.cross_entropy(logits, tgt)
        # hard-neg hinge over positives
        if pos_mask.any():
            Sp = S[pos_mask]; tp = labels[pos_mask]
            true_s = Sp.gather(1, tp.view(-1,1)).squeeze(1)
            neg = Sp.clone(); neg.scatter_(1, tp.view(-1,1), -1e9)
            hin = F.relu(neg.max(dim=1).values - true_s + args.margin).mean()
        else:
            hin = torch.zeros((), device=dev)
        loss = ce + args.hinge * hin
        loss.backward(); opt.step()
        if ep % 50 == 0 or ep == args.epochs-1:
            print(f"[wcf] ep{ep} ce={float(ce):.4f} hinge={float(hin):.4f} "
                  f"tau={torch.exp(log_tau).item():.3f} s0={s0.item():.3f}", flush=True)

    with torch.no_grad():
        S = all_scores().cpu().numpy()
        lab = labels.cpu().numpy()
        tg = [S[i, lab[i]] for i in range(Nq) if lab[i] >= 0]
        fg = [S[i].max()    for i in range(Nq) if lab[i] <  0]
        argok = sum(1 for i in range(Nq) if lab[i] >= 0 and int(S[i].argmax()) == lab[i])
        npos = int((labels >= 0).sum())
        mt = min(tg) if tg else float("nan"); mf = max(fg) if fg else float("nan")
        print(f"[wcf] diagonal argmax-correct: {argok}/{npos}", flush=True)
        print(f"[wcf] FLOAT: min_target={mt:+.3f} max_foreign={mf:+.3f} s0={s0.item():+.3f} "
              f"-> {'SEPARATES' if mt>mf else 'NO SEP'} (float; int gate in b3_export_wc)", flush=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, Wc=Wc.detach().cpu().numpy(), tau=float(torch.exp(log_tau).item()),
             s0=float(s0.item()), r=args.r, scale=scale, ep_names=np.array(names, dtype=object))
    print(f"[wcf] saved {out}", flush=True)

if __name__ == "__main__":
    main()
