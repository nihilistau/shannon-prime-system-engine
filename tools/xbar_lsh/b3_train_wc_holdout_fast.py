#!/usr/bin/env python3
"""b3_train_wc_holdout_fast.py -- VECTORIZED generalization-holdout retrain of W_c.

Mathematically IDENTICAL to b3_train_wc_holdout.py (relevance = logsumexp over REAL
positions then mean over heads; InfoNCE over [train episodes + NULL=s0] + reject-margin
hinge; strict held-out-needle split). The ONLY change is performance: instead of looping
240 episodes in Python per epoch (which churns a [Nq,ng,GH,np] intermediate per episode and
runs ~minutes/epoch at 300-needle scale), it STACKS all episodes into one padded tensor
K[E,ng,Pmax,r] with a position MASK (padded positions -> -inf before logsumexp), so one
batched einsum scores every episode at once. logsumexp over only the real positions is
preserved exactly by the -inf mask. Headline metric unchanged: HOLDOUT top-1 on unseen
needles.
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
    ap.add_argument("--out", default=None)
    args=ap.parse_args()
    eng=os.environ.get("SP_ENGINE_DIR", r"D:\F\shannon-prime-repos\shannon-prime-system-engine")
    data=args.data or os.path.join(eng,"_b3_wc","b3_data_div.npz")
    out=args.out or os.path.join(eng,"_b3_wc","lsh_Wc_f32_holdout.npz")
    dev="cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(20260619)
    d=np.load(data, allow_pickle=True); names=list(d["ep_names"]); E=len(d["K"])
    lab_np=d["labels"].astype(np.int64)
    Qs=torch.tensor(np.stack([np.asarray(q,np.float32) for q in d["Q"]]), device=dev)  # [Nq,ng,GH,HD]
    Nq, ng = Qs.shape[0], Qs.shape[1]
    Ks_raw=[np.asarray(k,np.float32) for k in d["K"]]
    ng=min(ng, min(int(k.shape[0]) for k in Ks_raw))
    Qs=Qs[:,:ng].contiguous()
    npos_e=[int(k.shape[1]) for k in Ks_raw]; Pmax=max(npos_e)
    # stack K -> [E, ng, Pmax, HD] with a real-position mask [E, Pmax]
    Kpad=np.zeros((E, ng, Pmax, HD), np.float32)
    Kmask=np.zeros((E, Pmax), np.float32)   # 1 = real position, 0 = pad
    for e,k in enumerate(Ks_raw):
        p=npos_e[e]; Kpad[e,:,:p,:]=k[:ng,:p,:]; Kmask[e,:p]=1.0
    Kpad=torch.tensor(Kpad, device=dev); 
    neg_inf_mask=torch.tensor((1.0-Kmask)*(-1e30), device=dev)   # [E,Pmax] add -inf to pads

    # ---- seeded held-out needle split ----
    rng=np.random.default_rng(args.split_seed)
    n_hold=max(1, int(round(E*args.holdout_frac)))
    hold_eps=sorted(rng.choice(E, size=n_hold, replace=False).tolist()); hold_set=set(hold_eps)
    train_eps=[e for e in range(E) if e not in hold_set]
    e2col={e:i for i,e in enumerate(train_eps)}; Etr=len(train_eps); NULL_TR=Etr
    train_eps_t=torch.tensor(train_eps, device=dev)
    is_foreign=lab_np<0
    is_hold_q=np.array([(l>=0 and l in hold_set) for l in lab_np])
    train_q=np.where(~is_hold_q)[0]; hold_q=np.where(is_hold_q)[0]
    tgt_tr=np.full(Nq, NULL_TR, dtype=np.int64)
    for i in range(Nq):
        l=lab_np[i]
        if l>=0 and l in e2col: tgt_tr[i]=e2col[l]
    tgt_tr=torch.tensor(tgt_tr, device=dev); train_q_t=torch.tensor(train_q, device=dev)
    n_train_pos=int(((lab_np>=0)&(~is_hold_q)).sum()); n_foreign=int(is_foreign.sum())
    print(f"[wcHf] dev={dev} r={args.r} dropout={args.dropout} wd={args.wd} Pmax={Pmax}", flush=True)
    print(f"[wcHf] SPLIT seed={args.split_seed} frac={args.holdout_frac}: needles total={E} "
          f"train={Etr} holdout={n_hold}", flush=True)
    print(f"[wcHf] holdout ep idx: {hold_eps}", flush=True)
    print(f"[wcHf] train query rows: {len(train_q)} ({n_train_pos} train-pos + {n_foreign} foreign); "
          f"holdout query rows: {len(hold_q)}", flush=True)

    Wc=torch.nn.Parameter(torch.randn(HD,args.r,device=dev)/(HD**0.5))
    log_tau=torch.nn.Parameter(torch.zeros((),device=dev)); s0=torch.nn.Parameter(torch.zeros((),device=dev))
    opt=torch.optim.Adam([Wc,log_tau,s0], lr=args.lr, weight_decay=args.wd); scale=1.0/(args.r**0.5)

    EP_CHUNK=int(os.environ.get("WC_EP_CHUNK","24"))   # bound the [Nq,chunk,ng,GH,Pmax] intermediate
    def scores_all(drop=False):
        # relevance(q,e) = logsumexp_p[(Q.Wc).(K.Wc)*scale over real p] then mean over (l,h).
        # Chunked over episodes to bound VRAM (the full [Nq,E,ng,GH,Pmax] tensor is too big).
        W = F.dropout(Wc, p=args.dropout, training=True) if (drop and args.dropout>0) else Wc
        qp=torch.einsum("qlhd,dr->qlhr", Qs, W)          # [Nq,ng,GH,r]
        outs=[]
        for c0 in range(0, E, EP_CHUNK):
            c1=min(E, c0+EP_CHUNK)
            kp=torch.einsum("elpd,dr->elpr", Kpad[c0:c1], W)          # [c,ng,Pmax,r]
            sim=torch.einsum("qlhr,elpr->qelhp", qp, kp)*scale        # [Nq,c,ng,GH,Pmax]
            sim=sim + neg_inf_mask[c0:c1].view(1,c1-c0,1,1,Pmax)
            a=torch.logsumexp(sim, dim=4)                            # [Nq,c,ng,GH]
            outs.append(a.mean(dim=(2,3)))                            # [Nq,c]
        return torch.cat(outs, dim=1)                                 # [Nq,E]

    for ep in range(args.epochs):
        opt.zero_grad()
        S_all=scores_all(drop=True)                            # [Nq,E]
        S_tr=S_all[:, train_eps_t]                             # [Nq,Etr]
        Sq=S_tr[train_q_t]; tq=tgt_tr[train_q_t]
        logits=torch.cat([Sq, s0.expand(Sq.shape[0],1)], dim=1)/torch.exp(log_tau)
        ce=F.cross_entropy(logits, tq)
        posm=(tq!=NULL_TR)
        if posm.any():
            Sp=Sq[posm]; tp=tq[posm]; true_s=Sp.gather(1,tp.view(-1,1)).squeeze(1)
            neg=Sp.clone(); neg.scatter_(1,tp.view(-1,1),-1e9)
            hin=F.relu(neg.max(1).values - true_s + 1.0).mean()
            rmp=F.relu(s0 + args.rmargin - true_s).mean()
        else: hin=torch.zeros((),device=dev); rmp=torch.zeros((),device=dev)
        fm_mask=(tq==NULL_TR)
        if fm_mask.any():
            fm=Sq[fm_mask].max(1).values; rmf=F.relu(fm-(s0-args.rmargin)).mean()
        else: rmf=torch.zeros((),device=dev)
        loss=ce + 0.2*hin + args.wrm*(rmp+rmf); loss.backward(); opt.step()
        if ep%50==0 or ep==args.epochs-1:
            print(f"[wcHf] ep{ep} ce={float(ce):.4f} hin={float(hin):.3f} rmp={float(rmp):.3f} "
                  f"rmf={float(rmf):.3f} tau={torch.exp(log_tau).item():.3f} s0={s0.item():.3f}", flush=True)

    with torch.no_grad():
        S_full=scores_all(drop=False).cpu().numpy()   # [Nq,E]
        s0v=float(s0.item())
        tr_pos=[i for i in range(Nq) if lab_np[i]>=0 and not is_hold_q[i]]; tr_ok=0
        for i in tr_pos:
            row=np.concatenate([S_full[i],[s0v]])
            if int(row.argmax())==lab_np[i]: tr_ok+=1
        ho_rows=[int(i) for i in hold_q]; ho_ok=0; ho_detail=[]
        for i in ho_rows:
            row=np.concatenate([S_full[i],[s0v]]); am=int(row.argmax()); ok=(am==lab_np[i]); ho_ok+=ok
            ho_detail.append((i,int(lab_np[i]),am,bool(ok)))
        fr_rows=[i for i in range(Nq) if lab_np[i]<0]; fr_ok=0
        for i in fr_rows:
            row=np.concatenate([S_full[i],[s0v]])
            if int(row.argmax())==E: fr_ok+=1
        tr_pct=100.0*tr_ok/max(1,len(tr_pos)); ho_pct=100.0*ho_ok/max(1,len(ho_rows)); fr_pct=100.0*fr_ok/max(1,len(fr_rows))
        # rank diagnostics
        ho_rank_all=[]; ho_rank_hold=[]
        hold_eps_arr=np.array(hold_eps)
        for (i,t,am,ok) in ho_detail:
            order=np.argsort(-S_full[i])
            ho_rank_all.append(int(np.where(order==t)[0][0])+1)
            sub=S_full[i][hold_eps_arr]; order2=np.argsort(-sub)
            tpos=int(np.where(hold_eps_arr==t)[0][0]); ho_rank_hold.append(int(np.where(order2==tpos)[0][0])+1)
        import statistics as st
        print(f"[wcHf] ===== RESULTS (seed {args.split_seed}) =====", flush=True)
        print(f"[wcHf] train-diagonal (sanity): {tr_ok}/{len(tr_pos)} = {tr_pct:.1f}%", flush=True)
        print(f"[wcHf] >>> HOLDOUT top-1 recall: {ho_ok}/{len(ho_rows)} = {ho_pct:.1f}% <<<", flush=True)
        print(f"[wcHf] foreign-reject (NULL wins): {fr_ok}/{len(fr_rows)} = {fr_pct:.1f}%", flush=True)
        ho_lost_to_trained=sum(1 for (i,t,am,ok) in ho_detail if (not ok) and am!=E and am not in hold_set)
        ho_to_null=sum(1 for (i,t,am,ok) in ho_detail if (not ok) and am==E)
        ho_to_other_hold=sum(1 for (i,t,am,ok) in ho_detail if (not ok) and am in hold_set)
        print(f"[wcHf] holdout misses: ->trained-needle={ho_lost_to_trained} ->NULL={ho_to_null} "
              f"->other-holdout={ho_to_other_hold}", flush=True)
        print(f"[wcHf] DIAG holdout true-rank among ALL {E}: median {st.median(ho_rank_all):.1f}; "
              f"among {n_hold} holdout-only: median {st.median(ho_rank_hold):.1f} (chance-within-holdout 1/{n_hold}={100.0/n_hold:.1f}%)", flush=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, Wc=Wc.detach().cpu().numpy(), tau=float(torch.exp(log_tau).item()),
             s0=s0v, r=args.r, scale=scale, ep_names=np.array(names, dtype=object),
             holdout_eps=np.array(hold_eps), split_seed=args.split_seed)
    print(f"[wcHf] saved {out}", flush=True)

if __name__=="__main__":
    main()
