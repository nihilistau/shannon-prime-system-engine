#!/usr/bin/env python3
"""B3-v3 contrastive trainer — learn W_c : R^HD -> R^r so that the episode relevance
S(query, episode) separates the matching episode from all others AND from foreign
queries. Adapted from train_lsh.py (InfoNCE forward-KL + 0.2 hard-neg hinge +
learnable tau); the granularity is episode-level retrieval, not within-context top-B.

Relevance (matches recall.rs::qk_relevance, softened for training):
  qp = Q @ Wc            [ng, G_NH, r]
  kp = K @ Wc            [ng, npos, r]
  per (layer l, head h): a_lh = logsumexp_p( qp[l,h]·kp[l,p] ) / sqrt(r)   (soft "attend to E")
  S(query, E) = mean over (l,h) of a_lh
At inference the runtime uses real top-m / max over (l,h,p); logsumexp is the smooth proxy.

Objective: a NULL class with learnable score s0 turns this into one clean InfoNCE over
[ep_0 .. ep_{E-1}, NULL]: positives target their true episode, foreign queries target
NULL. s0 IS the deploy threshold TAU (foreign must score below it). Plus a hard-negative
hinge so the weakest-margin positive still beats its strongest wrong episode.
"""
import os, sys, argparse
import numpy as np
import torch
import torch.nn.functional as F

HD, G_NH = 512, 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="?", default=None)
    ap.add_argument("--r", type=int, default=int(os.environ.get("WC_R", "32")))
    ap.add_argument("--epochs", type=int, default=int(os.environ.get("WC_EPOCHS", "400")))
    ap.add_argument("--lr", type=float, default=float(os.environ.get("WC_LR", "3e-3")))
    ap.add_argument("--hinge", type=float, default=float(os.environ.get("WC_HINGE", "0.2")))
    ap.add_argument("--margin", type=float, default=float(os.environ.get("WC_MARGIN", "1.0")))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    eng = os.environ.get("SP_ENGINE_DIR", r"D:\F\shannon-prime-repos\shannon-prime-system-engine")
    data = args.data or os.path.join(eng, "_b3_wc", "b3_data.npz")
    out = args.out or os.path.join(eng, "_b3_wc", "lsh_Wc_f32.npz")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(20260619)

    d = np.load(data, allow_pickle=True)
    Q = [torch.tensor(np.asarray(q, np.float32), device=dev) for q in d["Q"]]   # each [ng,G_NH,HD]
    K = [torch.tensor(np.asarray(k, np.float32), device=dev) for k in d["K"]]   # each [ng,npos,HD]
    labels = torch.tensor(d["labels"].astype(np.int64), device=dev)            # >=0 ep idx, -1 foreign
    names = list(d["ep_names"])
    E = len(K)
    NULL = E
    ng = min(Q[0].shape[0], K[0].shape[0])
    print(f"[wc] dev={dev} r={args.r} queries={len(Q)} episodes={E} ng={ng} "
          f"({int((labels>=0).sum())} pos / {int((labels<0).sum())} foreign)", flush=True)

    Wc = torch.nn.Parameter(torch.randn(HD, args.r, device=dev) / (HD ** 0.5))
    log_tau = torch.nn.Parameter(torch.zeros((), device=dev))
    s0 = torch.nn.Parameter(torch.zeros((), device=dev))       # NULL score = deploy threshold
    opt = torch.optim.Adam([Wc, log_tau, s0], lr=args.lr)
    scale = 1.0 / (args.r ** 0.5)

    # precompute projected K per episode (depends on Wc, so inside loop)
    def relevance(qi):
        # qi: [ng,G_NH,HD] -> S over episodes [E]
        qp = torch.einsum("lhd,dr->lhr", qi[:ng], Wc)                  # [ng,G_NH,r]
        S = []
        for e in range(E):
            kp = torch.einsum("lpd,dr->lpr", K[e][:ng], Wc)           # [ng,np,r]
            # per (l,h): logsumexp_p qp[l,h]·kp[l,p]
            sim = torch.einsum("lhr,lpr->lhp", qp, kp) * scale         # [ng,G_NH,np]
            a = torch.logsumexp(sim, dim=2)                            # [ng,G_NH]
            S.append(a.mean())
        return torch.stack(S)                                         # [E]

    for ep in range(args.epochs):
        opt.zero_grad()
        ce = 0.0; hin = 0.0
        for i in range(len(Q)):
            S = relevance(Q[i])                                       # [E]
            logits = torch.cat([S, s0.view(1)]) / torch.exp(log_tau)  # [E+1]
            tgt = labels[i].item()
            tgt = NULL if tgt < 0 else tgt
            ce = ce + F.cross_entropy(logits.view(1, -1), torch.tensor([tgt], device=dev))
            if labels[i].item() >= 0:                                 # hard-neg hinge for positives
                true = labels[i].item()
                neg = S.clone(); neg[true] = -1e9
                hin = hin + F.relu(neg.max() - S[true] + args.margin)
        loss = ce / len(Q) + args.hinge * hin / max(1, int((labels >= 0).sum()))
        loss.backward(); opt.step()
        if ep % 50 == 0 or ep == args.epochs - 1:
            print(f"[wc] ep{ep} ce={float(ce)/len(Q):.4f} hinge={float(hin):.4f} "
                  f"tau={torch.exp(log_tau).item():.3f} s0={s0.item():.3f}", flush=True)

    # ---- report the float separation (informational; the GATE runs on int in export) ----
    with torch.no_grad():
        tg, fg = [], []
        print("\n[wc] float relevance matrix (S per episode):", flush=True)
        for i in range(len(Q)):
            S = relevance(Q[i]).cpu().numpy()
            lab = int(labels[i].item())
            arg = int(S.argmax())
            if lab >= 0:
                tg.append(S[lab]); tag = f"want={names[lab]} arg={'OK' if arg==lab else names[arg]}"
            else:
                fg.append(S.max()); tag = "FOREIGN"
            print("  " + " ".join(f"{names[e]}={S[e]:+.3f}" for e in range(E)) + f"   {tag}", flush=True)
        mt = min(tg) if tg else float("nan"); mf = max(fg) if fg else float("nan")
        print(f"\n[wc] FLOAT: min_target={mt:+.3f}  max_foreign={mf:+.3f}  "
              f"s0(thresh)={s0.item():+.3f}  -> {'SEPARATES' if mt>mf else 'NO SEP'} (float; "
              f"the binding gate is the INTEGER one in b3_export_wc.py)", flush=True)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, Wc=Wc.detach().cpu().numpy(), tau=float(torch.exp(log_tau).item()),
             s0=float(s0.item()), r=args.r, scale=scale, ep_names=np.array(names, dtype=object))
    print(f"[wc] saved {out}", flush=True)


if __name__ == "__main__":
    main()
