#!/usr/bin/env python3
"""b3_train_wc_holdout.py -- GENERALIZATION HOLDOUT retrain of the W_c recall head.

Copies b3_train_wc_fast2.py's EXACT relevance (logsumexp over positions then mean over heads --
must match recall.rs) and its InfoNCE + reject-margin-hinge objective, but adds a strict
held-out-needle split to test OPEN-SET generalization (not closed-set memorization):

  * Hold out HOLDOUT_FRAC (default 20%) of the needles, seeded (np.random.default_rng(SPLIT_SEED)).
  * HELD-OUT needles are excluded from training ENTIRELY: their query rows are NOT in the loss,
    AND their episodes are NOT in the train InfoNCE candidate set (train softmax over
    [train episodes + NULL] only).
  * Foreign (label<0) rows train as before (NULL class).
  * STRICT VALIDATION: each held-out needle's query rows are scored against ALL episodes + NULL
    (logsumexp-mean), (E+1)-argmax -> top-1 correct iff the matched held-out episode wins.

Headline metric = HOLDOUT top-1 recall on unseen-needle query rows. Also reports train diagonal
(sanity) and foreign-reject (NULL must win on the 50 foreign).

Regularization knobs (env or flags): --dropout (projection dropout p), --wd (weight decay),
--tau_floor (clamp log_tau so the logsumexp temperature can't run away). Defaults reproduce fast2
(dropout 0, wd 0) -- turn them on only if holdout is poor.
"""
import os, sys, argparse
import numpy as np, torch, torch.nn.functional as F
HD, G_NH = 512, 16

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("data", nargs="?", default=None)
    ap.add_argument("--r", type=int, default=int(os.environ.get("WC_R","32")))
    ap.add_argument("--epochs", type=int, default=int(os.environ.get("WC_EPOCHS","600")))
    ap.add_argument("--lr", type=float, default=float(os.environ.get("WC_LR","3e-3")))
    ap.add_argument("--rmargin", type=float, default=float(os.environ.get("WC_RMARGIN","0.20")))
    ap.add_argument("--wrm", type=float, default=float(os.environ.get("WC_WRM","0.5")))
    ap.add_argument("--holdout_frac", type=float, default=float(os.environ.get("WC_HOLDOUT_FRAC","0.20")))
    ap.add_argument("--split_seed", type=int, default=int(os.environ.get("WC_SPLIT_SEED","0")))
    ap.add_argument("--dropout", type=float, default=float(os.environ.get("WC_DROPOUT","0.0")))
    ap.add_argument("--wd", type=float, default=float(os.environ.get("WC_WD","0.0")))
    ap.add_argument("--tau_floor", type=float, default=float(os.environ.get("WC_TAU_FLOOR","-1e9")))
    ap.add_argument("--out", default=None)
    args=ap.parse_args()
    eng=os.environ.get("SP_ENGINE_DIR", r"D:\F\shannon-prime-repos\shannon-prime-system-engine")
    data=args.data or os.path.join(eng,"_b3_wc","b3_data_div.npz")
    out=args.out or os.path.join(eng,"_b3_wc","lsh_Wc_f32_holdout.npz")
    dev="cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(20260619)
    d=np.load(data, allow_pickle=True); names=list(d["ep_names"]); E=len(d["K"])
    labels=torch.tensor(d["labels"].astype(np.int64), device=dev)
    Qs=torch.tensor(np.stack([np.asarray(q,np.float32) for q in d["Q"]]), device=dev)
    Nq, ng = Qs.shape[0], Qs.shape[1]
    Ks=[torch.tensor(np.asarray(k,np.float32), device=dev) for k in d["K"]]
    ng=min(ng, min(int(k.shape[0]) for k in Ks)); NULL_FULL=E

    # ---- seeded held-out needle split ----
    rng=np.random.default_rng(args.split_seed)
    n_hold=max(1, int(round(E*args.holdout_frac)))
    hold_eps=sorted(rng.choice(E, size=n_hold, replace=False).tolist())
    hold_set=set(hold_eps)
    train_eps=[e for e in range(E) if e not in hold_set]          # episodes kept in train softmax
    e2col={e:i for i,e in enumerate(train_eps)}                   # train-episode index -> train column
    Etr=len(train_eps); NULL_TR=Etr                              # NULL column index in the TRAIN logits

    lab_np=d["labels"].astype(np.int64)
    is_foreign = lab_np<0
    is_hold_q  = np.array([(l>=0 and l in hold_set) for l in lab_np])
    # train query rows: positives of TRAIN needles + all foreign; EXCLUDE held-out-needle queries
    train_q_mask = (~is_hold_q)
    train_q = np.where(train_q_mask)[0]
    hold_q  = np.where(is_hold_q)[0]
    # train targets in TRAIN-column space
    tgt_tr=np.full(Nq, NULL_TR, dtype=np.int64)
    for i in range(Nq):
        l=lab_np[i]
        if l>=0 and l in e2col: tgt_tr[i]=e2col[l]
    tgt_tr=torch.tensor(tgt_tr, device=dev)
    train_q_t=torch.tensor(train_q, device=dev)
    labels_cpu=lab_np

    n_hold_needles=len(hold_eps)
    n_hold_qrows=int(is_hold_q.sum())
    n_train_needles=Etr
    n_train_pos=int(((lab_np>=0) & train_q_mask).sum())
    n_foreign=int(is_foreign.sum())
    print(f"[wcH] dev={dev} r={args.r} dropout={args.dropout} wd={args.wd} tau_floor={args.tau_floor}", flush=True)
    print(f"[wcH] SPLIT seed={args.split_seed} frac={args.holdout_frac}: needles total={E} "
          f"train={n_train_needles} holdout={n_hold_needles}", flush=True)
    print(f"[wcH] holdout ep idx: {hold_eps}", flush=True)
    print(f"[wcH] train query rows: {len(train_q)} ({n_train_pos} train-pos + {n_foreign} foreign); "
          f"holdout query rows: {n_hold_qrows}", flush=True)

    Wc=torch.nn.Parameter(torch.randn(HD,args.r,device=dev)/(HD**0.5))
    log_tau=torch.nn.Parameter(torch.zeros((),device=dev)); s0=torch.nn.Parameter(torch.zeros((),device=dev))
    opt=torch.optim.Adam([Wc,log_tau,s0], lr=args.lr, weight_decay=args.wd); scale=1.0/(args.r**0.5)

    def scores_over(ep_list, drop=False):
        # relevance(q, ep) = logsumexp_p( (Q.Wc) . (K.Wc) * scale ) over positions, then mean over (l,h)
        W = F.dropout(Wc, p=args.dropout, training=True) if (drop and args.dropout>0) else Wc
        qp=torch.einsum("qlhd,dr->qlhr", Qs[:,:ng], W); cols=[]
        for e in ep_list:
            kp=torch.einsum("lpd,dr->lpr", Ks[e][:ng], W)
            sim=torch.einsum("qlhr,lpr->qlhp", qp, kp)*scale     # [Nq,ng,GH,np]
            a=torch.logsumexp(sim, dim=3)                        # [Nq,ng,GH]
            cols.append(a.mean(dim=(1,2)))                       # -> [Nq]
        return torch.stack(cols, dim=1)                          # [Nq, len(ep_list)]

    for ep in range(args.epochs):
        opt.zero_grad()
        S_tr=scores_over(train_eps, drop=True)                   # [Nq, Etr]  (train episodes only)
        Sq=S_tr[train_q_t]; tq=tgt_tr[train_q_t]                 # only train query rows in the loss
        logits=torch.cat([Sq, s0.expand(Sq.shape[0],1)], dim=1)/torch.exp(log_tau)
        ce=F.cross_entropy(logits, tq)
        # reject-margin hinge (same as fast2) restricted to train query rows
        posm=(tq!=NULL_TR)
        if posm.any():
            Sp=Sq[posm]; tp=tq[posm]; true_s=Sp.gather(1,tp.view(-1,1)).squeeze(1)
            neg=Sp.clone(); neg.scatter_(1,tp.view(-1,1),-1e9)
            hin=F.relu(neg.max(1).values - true_s + 1.0).mean()
            rmp=F.relu(s0 + args.rmargin - true_s).mean()
        else: hin=torch.zeros((),device=dev); rmp=torch.zeros((),device=dev)
        fm_mask=(tq==NULL_TR)
        if fm_mask.any():
            fm=Sq[fm_mask].max(1).values; rmf=F.relu(fm - (s0 - args.rmargin)).mean()
        else: rmf=torch.zeros((),device=dev)
        loss=ce + 0.2*hin + args.wrm*(rmp+rmf); loss.backward(); opt.step()
        with torch.no_grad():
            if args.tau_floor>-1e8: log_tau.clamp_(min=args.tau_floor)
        if ep%50==0 or ep==args.epochs-1:
            print(f"[wcH] ep{ep} ce={float(ce):.4f} hin={float(hin):.3f} rmp={float(rmp):.3f} "
                  f"rmf={float(rmf):.3f} tau={torch.exp(log_tau).item():.3f} s0={s0.item():.3f}", flush=True)

    # ---- evaluation (eval mode = no dropout) ----
    with torch.no_grad():
        S_full=scores_over(list(range(E)), drop=False).cpu().numpy()   # [Nq, E]
        s0v=float(s0.item())
        # train diagonal (sanity): train-needle query rows, argmax over [all E + NULL]
        tr_pos = [i for i in range(Nq) if labels_cpu[i]>=0 and not is_hold_q[i]]
        tr_ok=0
        for i in tr_pos:
            row=np.concatenate([S_full[i], [s0v]])
            if int(row.argmax())==labels_cpu[i]: tr_ok+=1
        # HOLDOUT: held-out needle query rows, argmax over [all E + NULL]
        ho_ok=0; ho_rows=[int(i) for i in hold_q]
        ho_detail=[]
        for i in ho_rows:
            row=np.concatenate([S_full[i], [s0v]])
            am=int(row.argmax())
            ok=(am==labels_cpu[i])
            ho_ok+=ok
            ho_detail.append((i, int(labels_cpu[i]), am, bool(ok)))
        # foreign reject: NULL must win
        fr_rows=[i for i in range(Nq) if labels_cpu[i]<0]; fr_ok=0
        for i in fr_rows:
            row=np.concatenate([S_full[i], [s0v]])
            if int(row.argmax())==E: fr_ok+=1   # NULL column index == E
        tr_pct=100.0*tr_ok/max(1,len(tr_pos))
        ho_pct=100.0*ho_ok/max(1,len(ho_rows))
        fr_pct=100.0*fr_ok/max(1,len(fr_rows))
        print(f"[wcH] ===== RESULTS (seed {args.split_seed}) =====", flush=True)
        print(f"[wcH] train-diagonal (sanity): {tr_ok}/{len(tr_pos)} = {tr_pct:.1f}%", flush=True)
        print(f"[wcH] >>> HOLDOUT top-1 recall: {ho_ok}/{len(ho_rows)} = {ho_pct:.1f}% <<<", flush=True)
        print(f"[wcH] foreign-reject (NULL wins): {fr_ok}/{len(fr_rows)} = {fr_pct:.1f}%", flush=True)
        # extra: how many holdout rows landed in-band but lost to a trained needle (vs NULL-rejected)
        ho_lost_to_trained=sum(1 for (i,t,am,ok) in ho_detail if (not ok) and am!=E and am not in hold_set)
        ho_to_null=sum(1 for (i,t,am,ok) in ho_detail if (not ok) and am==E)
        ho_to_other_hold=sum(1 for (i,t,am,ok) in ho_detail if (not ok) and am in hold_set)
        print(f"[wcH] holdout misses: ->trained-needle={ho_lost_to_trained} ->NULL={ho_to_null} "
              f"->other-holdout={ho_to_other_hold}", flush=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, Wc=Wc.detach().cpu().numpy(), tau=float(torch.exp(log_tau).item()),
             s0=s0v, r=args.r, scale=scale, ep_names=np.array(names, dtype=object),
             holdout_eps=np.array(hold_eps), split_seed=args.split_seed)
    print(f"[wcH] saved {out}", flush=True)

if __name__=="__main__":
    main()
