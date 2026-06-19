#!/usr/bin/env python3
"""b3_train_wc_align.py -- vectorized trainer with the runtime-aligned reduction + reject margin.

Two fixes over b3_train_wc_fast (which trained with logsumexp_p -> float 361/361 but int16/runtime
max+top-m degraded, and the foreign-reject margin collapsed):
  (1) RELEVANCE REDUCTION = top-m mean over flattened (l,h,p)  == recall.rs / b3_export_wc runtime
      (was logsumexp_p then mean_{l,h}). So the trained metric == the deployed metric.
  (2) explicit REJECT-MARGIN term: push every positive's true score above s0+rm and every foreign's
      best score below s0-rm, re-opening the absolute gap that instance-discrimination collapsed.
Same save format.
"""
import os, sys, argparse
import numpy as np, torch, torch.nn.functional as F
HD, G_NH = 512, 16

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("data", nargs="?", default=None)
    ap.add_argument("--r", type=int, default=int(os.environ.get("WC_R","32")))
    ap.add_argument("--epochs", type=int, default=int(os.environ.get("WC_EPOCHS","500")))
    ap.add_argument("--lr", type=float, default=float(os.environ.get("WC_LR","3e-3")))
    ap.add_argument("--topm", type=int, default=int(os.environ.get("WC_TOPM","8")))   # recall.rs top-8
    ap.add_argument("--rmargin", type=float, default=float(os.environ.get("WC_RMARGIN","0.15")))
    ap.add_argument("--wrm", type=float, default=float(os.environ.get("WC_WRM","1.0")))
    ap.add_argument("--out", default=None)
    args=ap.parse_args()
    eng=os.environ.get("SP_ENGINE_DIR", r"D:\F\shannon-prime-repos\shannon-prime-system-engine")
    data=args.data or os.path.join(eng,"_b3_wc","b3_data.npz")
    out=args.out or os.path.join(eng,"_b3_wc","lsh_Wc_f32.npz")
    dev="cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(20260619)
    d=np.load(data, allow_pickle=True); names=list(d["ep_names"]); E=len(d["K"])
    labels=torch.tensor(d["labels"].astype(np.int64), device=dev)
    Qs=torch.tensor(np.stack([np.asarray(q,np.float32) for q in d["Q"]]), device=dev)
    Nq, ng = Qs.shape[0], Qs.shape[1]
    Ks=[torch.tensor(np.asarray(k,np.float32), device=dev) for k in d["K"]]
    ng=min(ng, min(int(k.shape[0]) for k in Ks)); NULL=E
    pos=labels>=0; tgt=labels.clone(); tgt[~pos]=NULL
    print(f"[wca] dev={dev} r={args.r} topm={args.topm} rmargin={args.rmargin} queries={Nq} episodes={E} ng={ng} "
          f"({int(pos.sum())} pos / {int((~pos).sum())} foreign)", flush=True)
    Wc=torch.nn.Parameter(torch.randn(HD,args.r,device=dev)/(HD**0.5))
    log_tau=torch.nn.Parameter(torch.zeros((),device=dev)); s0=torch.nn.Parameter(torch.zeros((),device=dev))
    opt=torch.optim.Adam([Wc,log_tau,s0], lr=args.lr); scale=1.0/(args.r**0.5)

    def all_scores():
        qp=torch.einsum("qlhd,dr->qlhr", Qs[:,:ng], Wc)         # [Nq,ng,GH,r]
        cols=[]
        for e in range(E):
            kp=torch.einsum("lpd,dr->lpr", Ks[e][:ng], Wc)      # [ng,np,r]
            sim=torch.einsum("qlhr,lpr->qlhp", qp, kp)*scale     # [Nq,ng,GH,np]
            flat=sim.reshape(Nq, -1)                             # [Nq, ng*GH*np]
            m=min(args.topm, flat.shape[1])
            cols.append(flat.topk(m, dim=1).values.mean(dim=1))  # top-m mean == runtime
        return torch.stack(cols, dim=1)                          # [Nq,E]

    for ep in range(args.epochs):
        opt.zero_grad(); S=all_scores()
        logits=torch.cat([S, s0.expand(Nq,1)], dim=1)/torch.exp(log_tau)
        ce=F.cross_entropy(logits, tgt)
        # hard-neg hinge (positives: true beats best wrong episode)
        if pos.any():
            Sp=S[pos]; tp=labels[pos]; true_s=Sp.gather(1,tp.view(-1,1)).squeeze(1)
            neg=Sp.clone(); neg.scatter_(1,tp.view(-1,1),-1e9)
            hin=F.relu(neg.max(1).values - true_s + 1.0).mean()
            # reject margin: true score above s0+rm
            rmp=F.relu(s0 + args.rmargin - true_s).mean()
        else: hin=torch.zeros((),device=dev); rmp=torch.zeros((),device=dev)
        # reject margin: foreign best-episode below s0-rm
        if (~pos).any():
            fm=S[~pos].max(1).values; rmf=F.relu(fm - (s0 - args.rmargin)).mean()
        else: rmf=torch.zeros((),device=dev)
        loss=ce + 0.2*hin + args.wrm*(rmp+rmf)
        loss.backward(); opt.step()
        if ep%50==0 or ep==args.epochs-1:
            print(f"[wca] ep{ep} ce={float(ce):.4f} hin={float(hin):.3f} rmp={float(rmp):.3f} rmf={float(rmf):.3f} "
                  f"tau={torch.exp(log_tau).item():.3f} s0={s0.item():.3f}", flush=True)

    with torch.no_grad():
        S=all_scores().cpu().numpy(); lab=labels.cpu().numpy()
        tg=[S[i,lab[i]] for i in range(Nq) if lab[i]>=0]; fg=[S[i].max() for i in range(Nq) if lab[i]<0]
        argok=sum(1 for i in range(Nq) if lab[i]>=0 and int(S[i].argmax())==lab[i]); npos=int(pos.sum())
        mt=min(tg) if tg else float("nan"); mf=max(fg) if fg else float("nan")
        print(f"[wca] diagonal argmax-correct: {argok}/{npos}", flush=True)
        print(f"[wca] FLOAT(top-{args.topm}mean): min_target={mt:+.3f} max_foreign={mf:+.3f} s0={s0.item():+.3f} "
              f"-> {'SEPARATES' if mt>mf else 'NO SEP'}", flush=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, Wc=Wc.detach().cpu().numpy(), tau=float(torch.exp(log_tau).item()),
             s0=float(s0.item()), r=args.r, scale=scale, ep_names=np.array(names, dtype=object))
    print(f"[wca] saved {out}", flush=True)

if __name__=="__main__":
    main()
