#!/usr/bin/env python3
"""b3_capacity_sweep.py -- OFFLINE capacity sweep of the B4 recall scorer.

Question: does a higher-capacity learned projection lift UNSEEN-needle recall@20
above the deployed r=32 linear bilinear (which gets ~50% unseen)? Pure torch+cuda,
cached features, NO daemon/metal.

Reuses the EXACT machinery of b3_train_wc_holdout_fast.py:
  - data: _b3_wc/b3_data_div.npz  (E=91 needles, Nq=411: 361 positive + 50 foreign)
  - relevance(q,e) = logsumexp over REAL positions then mean over (head-group, head)
      of  proj(Q) . proj(K) * scale
  - InfoNCE over [TRAIN episodes + NULL=s0] + reject-margin hinge; learnable s0/log_tau
  - ONE fixed holdout split for ALL variants: read holdout_eps from
      _b3_wc/lsh_Wc_f32_holdout.npz (seed 0, 18 of 91), so every variant is
      apples-to-apples on the SAME unseen needles.

Variants (the capacity curve):
  linear r=32  (anchor; must reproduce ~50% recall@20 unseen)
  linear r=128
  linear r=256
  linear r=512
  mlp     512->Linear512->ReLU->Linear128  (NONLINEAR dual-encoder; SAME MLP applied
          to query-Q and episode-K separately -> still O(N) cheap at serve)

Headline = UNSEEN recall@{1,5,10,20,50}: rank the true episode among ALL 91 episodes,
on the held-out (never-trained) needles only. Context = ALL-needle (memorized)
recall@20 + foreign-reject %.
"""
import os, sys, argparse
import numpy as np, torch, torch.nn.functional as F
HD, G_NH = 512, 16

# ----------------------------- scorer modules -----------------------------
class LinearProj(torch.nn.Module):
    """y = x @ Wc ; Wc in R[HD,r]. The deployed bilinear (proj applied to both Q and K)."""
    def __init__(self, r, dev):
        super().__init__()
        self.Wc = torch.nn.Parameter(torch.randn(HD, r, device=dev) / (HD ** 0.5))
        self.r = r
    def forward(self, x):  # x[...,HD] -> [...,r]
        return torch.matmul(x, self.Wc)

class MLPProj(torch.nn.Module):
    """y = relu(x @ W1 + b1) @ W2 + b2 ; HD->512->relu->128 (nonlinear dual-encoder).
    SAME module applied to Q and K separately so serve stays O(N)."""
    def __init__(self, hidden, r, dev, dropout=0.0):
        super().__init__()
        self.l1 = torch.nn.Linear(HD, hidden).to(dev)
        self.l2 = torch.nn.Linear(hidden, r).to(dev)
        self.drop = torch.nn.Dropout(dropout)
        self.r = r
    def forward(self, x):
        return self.l2(self.drop(F.relu(self.l1(x))))

# ----------------------------- recall helper ------------------------------
def recall_at_ks(S_full, rows, lab_np, ks):
    """For each query row, rank true ep among ALL E episode scores; recall@k."""
    hit = {k: 0 for k in ks}
    n = 0
    for i in rows:
        t = int(lab_np[i])
        if t < 0:
            continue
        n += 1
        order = np.argsort(-S_full[i])          # descending; among ALL E episodes
        rank = int(np.where(order == t)[0][0]) + 1
        for k in ks:
            if rank <= k:
                hit[k] += 1
    return {k: (100.0 * hit[k] / max(1, n)) for k in ks}, n

# ----------------------------- one variant --------------------------------
def run_variant(name, make_proj, Qs, Kpad, neg_inf_mask, scale, E, ng, Pmax,
                lab_np, train_eps, hold_eps, hold_set, is_hold_q,
                train_q, hold_q, tgt_tr, NULL_TR, e2col, dev,
                epochs, lr, rmargin, wrm, wd, log):
    proj = make_proj()
    log_tau = torch.nn.Parameter(torch.zeros((), device=dev))
    s0 = torch.nn.Parameter(torch.zeros((), device=dev))
    params = list(proj.parameters()) + [log_tau, s0]
    opt = torch.optim.Adam(params, lr=lr, weight_decay=wd)
    train_eps_t = torch.tensor(train_eps, device=dev)
    train_q_t = torch.tensor(train_q, device=dev)
    tgt_tr_t = torch.tensor(tgt_tr, device=dev)
    EP_CHUNK = int(os.environ.get("WC_EP_CHUNK", "24"))

    def scores_all(train_mode):
        # set dropout train/eval if MLP has it
        proj.train(train_mode)
        qp = proj(Qs)                                   # [Nq,ng,GH,r]
        outs = []
        for c0 in range(0, E, EP_CHUNK):
            c1 = min(E, c0 + EP_CHUNK)
            kp = proj(Kpad[c0:c1])                       # [c,ng,Pmax,r]
            sim = torch.einsum("qlhr,elpr->qelhp", qp, kp) * scale
            sim = sim + neg_inf_mask[c0:c1].view(1, c1 - c0, 1, 1, Pmax)
            a = torch.logsumexp(sim, dim=4)              # [Nq,c,ng,GH]
            outs.append(a.mean(dim=(2, 3)))              # [Nq,c]
        return torch.cat(outs, dim=1)                    # [Nq,E]

    for ep in range(epochs):
        opt.zero_grad()
        S_all = scores_all(True)
        S_tr = S_all[:, train_eps_t]                     # [Nq,Etr]
        Sq = S_tr[train_q_t]; tq = tgt_tr_t[train_q_t]
        logits = torch.cat([Sq, s0.expand(Sq.shape[0], 1)], dim=1) / torch.exp(log_tau)
        ce = F.cross_entropy(logits, tq)
        posm = (tq != NULL_TR)
        if posm.any():
            Sp = Sq[posm]; tp = tq[posm]
            true_s = Sp.gather(1, tp.view(-1, 1)).squeeze(1)
            neg = Sp.clone(); neg.scatter_(1, tp.view(-1, 1), -1e9)
            hin = F.relu(neg.max(1).values - true_s + 1.0).mean()
            rmp = F.relu(s0 + rmargin - true_s).mean()
        else:
            hin = torch.zeros((), device=dev); rmp = torch.zeros((), device=dev)
        fm_mask = (tq == NULL_TR)
        if fm_mask.any():
            fm = Sq[fm_mask].max(1).values
            rmf = F.relu(fm - (s0 - rmargin)).mean()
        else:
            rmf = torch.zeros((), device=dev)
        loss = ce + 0.2 * hin + wrm * (rmp + rmf)
        loss.backward(); opt.step()
        if ep % 100 == 0 or ep == epochs - 1:
            log(f"  [{name}] ep{ep} ce={float(ce):.4f} hin={float(hin):.3f} "
                f"rmp={float(rmp):.3f} rmf={float(rmf):.3f} "
                f"tau={torch.exp(log_tau).item():.3f} s0={s0.item():.3f}")

    with torch.no_grad():
        S_full = scores_all(False).cpu().numpy()         # [Nq,E]
        s0v = float(s0.item())
        ks = [1, 5, 10, 20, 50]
        ho_rows = [int(i) for i in hold_q]
        ho_rec, ho_n = recall_at_ks(S_full, ho_rows, lab_np, ks)
        all_pos_rows = [i for i in range(len(lab_np)) if lab_np[i] >= 0]
        all_rec, all_n = recall_at_ks(S_full, all_pos_rows, lab_np, ks)
        # foreign-reject: NULL (=s0) beats every episode score
        fr_rows = [i for i in range(len(lab_np)) if lab_np[i] < 0]
        fr_ok = 0
        for i in fr_rows:
            row = np.concatenate([S_full[i], [s0v]])
            if int(row.argmax()) == E:
                fr_ok += 1
        fr_pct = 100.0 * fr_ok / max(1, len(fr_rows))
    return {
        "name": name, "param_count": sum(p.numel() for p in proj.parameters()),
        "unseen": ho_rec, "unseen_n": ho_n,
        "all": all_rec, "all_n": all_n,
        "foreign_reject": fr_pct, "foreign_n": len(fr_rows),
        "s0": s0v, "tau": float(torch.exp(log_tau).item()),
    }

# ----------------------------- main ---------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="?", default=None)
    ap.add_argument("--epochs", type=int, default=int(os.environ.get("WC_EPOCHS", "600")))
    ap.add_argument("--lr", type=float, default=float(os.environ.get("WC_LR", "3e-3")))
    ap.add_argument("--rmargin", type=float, default=float(os.environ.get("WC_RMARGIN", "0.20")))
    ap.add_argument("--wrm", type=float, default=float(os.environ.get("WC_WRM", "0.5")))
    ap.add_argument("--mlp_dropout", type=float, default=float(os.environ.get("WC_MLP_DROPOUT", "0.0")))
    ap.add_argument("--mlp_wd", type=float, default=float(os.environ.get("WC_MLP_WD", "0.0")))
    ap.add_argument("--log", default=None)
    args = ap.parse_args()
    eng = os.environ.get("SP_ENGINE_DIR", r"D:\F\shannon-prime-repos\shannon-prime-system-engine")
    data = args.data or os.path.join(eng, "_b3_wc", "b3_data_div.npz")
    holdnpz = os.path.join(eng, "_b3_wc", "lsh_Wc_f32_holdout.npz")
    logpath = args.log or os.path.join(eng, "tests", "fixtures", "chat_fullstack", "G-CHAT-B4-CAPACITY.log")
    os.makedirs(os.path.dirname(logpath), exist_ok=True)
    logf = open(logpath, "w", encoding="utf-8")
    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(20260619)
    log(f"=== G-CHAT-B4-CAPACITY : OFFLINE capacity sweep of the B4 recall scorer ===")
    log(f"device={dev}  torch={torch.__version__}  cuda_avail={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"gpu={torch.cuda.get_device_name(0)}")

    d = np.load(data, allow_pickle=True)
    names = list(d["ep_names"]); E = len(d["K"])
    lab_np = d["labels"].astype(np.int64)
    Qs = torch.tensor(np.stack([np.asarray(q, np.float32) for q in d["Q"]]), device=dev)  # [Nq,ng,GH,HD]
    Nq, ng = Qs.shape[0], Qs.shape[1]
    Ks_raw = [np.asarray(k, np.float32) for k in d["K"]]
    ng = min(ng, min(int(k.shape[0]) for k in Ks_raw))
    Qs = Qs[:, :ng].contiguous()
    npos_e = [int(k.shape[1]) for k in Ks_raw]; Pmax = max(npos_e)
    Kpad = np.zeros((E, ng, Pmax, HD), np.float32)
    Kmask = np.zeros((E, Pmax), np.float32)
    for e, k in enumerate(Ks_raw):
        p = npos_e[e]; Kpad[e, :, :p, :] = k[:ng, :p, :]; Kmask[e, :p] = 1.0
    Kpad = torch.tensor(Kpad, device=dev)
    neg_inf_mask = torch.tensor((1.0 - Kmask) * (-1e30), device=dev)
    scale = None  # set per-variant (depends on r)

    # ---- FIXED holdout split: read the stored split so EVERY variant matches the deployed run ----
    if os.path.exists(holdnpz):
        hh = np.load(holdnpz, allow_pickle=True)
        hold_eps = sorted([int(x) for x in hh["holdout_eps"]])
        split_seed = int(hh["split_seed"])
        log(f"[split] using STORED holdout from {os.path.basename(holdnpz)}: seed={split_seed}")
    else:
        split_seed = 0
        rng = np.random.default_rng(split_seed)
        n_hold = max(1, int(round(E * 0.20)))
        hold_eps = sorted(rng.choice(E, size=n_hold, replace=False).tolist())
        log(f"[split] STORED holdout npz absent; regenerated seed=0 frac=0.20")
    hold_set = set(hold_eps)
    n_hold = len(hold_eps)
    train_eps = [e for e in range(E) if e not in hold_set]
    e2col = {e: i for i, e in enumerate(train_eps)}; Etr = len(train_eps); NULL_TR = Etr
    is_foreign = lab_np < 0
    is_hold_q = np.array([(l >= 0 and l in hold_set) for l in lab_np])
    train_q = np.where(~is_hold_q)[0]; hold_q = np.where(is_hold_q)[0]
    tgt_tr = np.full(Nq, NULL_TR, dtype=np.int64)
    for i in range(Nq):
        l = lab_np[i]
        if l >= 0 and l in e2col:
            tgt_tr[i] = e2col[l]
    n_train_pos = int(((lab_np >= 0) & (~is_hold_q)).sum()); n_foreign = int(is_foreign.sum())
    chance20 = 100.0 * 20.0 / E
    log(f"[data] needles E={E}  Nq={Nq}  Pmax={Pmax}  ng={ng}")
    log(f"[split] train needles={Etr}  HOLDOUT needles={n_hold}  idx={hold_eps}")
    log(f"[split] train query rows={len(train_q)} ({n_train_pos} pos + {n_foreign} foreign); "
        f"holdout query rows={len(hold_q)}")
    log(f"[ctx] chance recall@20 = 20/{E} = {chance20:.1f}%   epochs={args.epochs} lr={args.lr} "
        f"rmargin={args.rmargin} wrm={args.wrm}")
    log("")

    variants = []
    for r in (32, 128, 256, 512):
        sc = 1.0 / (r ** 0.5)
        variants.append((f"linear_r{r}", (lambda rr=r: LinearProj(rr, dev)), sc, 0.0, 0.0))
    # MLP dual-encoder: out dim r=128 -> scale 1/sqrt(128)
    r_mlp = 128
    variants.append(("mlp_512_relu_128", (lambda: MLPProj(512, r_mlp, dev, dropout=args.mlp_dropout)),
                     1.0 / (r_mlp ** 0.5), args.mlp_wd, args.mlp_dropout))

    results = []
    for (name, make_proj, sc, wd, dr) in variants:
        log(f"--- variant: {name}  (scale={sc:.4f} wd={wd} dropout={dr}) ---")
        res = run_variant(name, make_proj, Qs, Kpad, neg_inf_mask, sc, E, ng, Pmax,
                          lab_np, train_eps, hold_eps, hold_set, is_hold_q,
                          train_q, hold_q, tgt_tr, NULL_TR, e2col, dev,
                          args.epochs, args.lr, args.rmargin, args.wrm, wd, log)
        results.append(res)
        u = res["unseen"]
        log(f"  >>> {name}: UNSEEN recall @1={u[1]:.1f} @5={u[5]:.1f} @10={u[10]:.1f} "
            f"@20={u[20]:.1f} @50={u[50]:.1f}  (n={res['unseen_n']})")
        a = res["all"]
        log(f"      ALL-needle (memorized) recall @20={a[20]:.1f} @1={a[1]:.1f}  "
            f"foreign-reject={res['foreign_reject']:.1f}% (n={res['foreign_n']})  "
            f"params={res['param_count']}")
        log("")

    # ---- summary table ----
    log("==================== SUMMARY: UNSEEN recall@K (held-out needles, rank true ep among all %d) ====================" % E)
    log(f"chance@20 = {chance20:.1f}%   holdout split seed={split_seed}   epochs={args.epochs}")
    hdr = f"{'variant':<20}{'params':>10}  {'@1':>6}{'@5':>6}{'@10':>6}{'@20':>6}{'@50':>6}   {'ALL@20':>7}{'frej%':>7}"
    log(hdr); log("-" * len(hdr))
    for res in results:
        u = res["unseen"]; a = res["all"]
        log(f"{res['name']:<20}{res['param_count']:>10}  "
            f"{u[1]:>6.1f}{u[5]:>6.1f}{u[10]:>6.1f}{u[20]:>6.1f}{u[50]:>6.1f}   "
            f"{a[20]:>7.1f}{res['foreign_reject']:>7.1f}")
    log("")
    # ---- headline verdict ----
    r32 = next(r for r in results if r["name"] == "linear_r32")
    best = max(results, key=lambda r: r["unseen"][20])
    log("==================== HEADLINE ====================")
    log(f"anchor   linear_r32 UNSEEN recall@20 = {r32['unseen'][20]:.1f}%  (context target ~50%)")
    log(f"best     {best['name']} UNSEEN recall@20 = {best['unseen'][20]:.1f}%")
    lift = best['unseen'][20] - r32['unseen'][20]
    if best['unseen'][20] >= 85.0:
        verdict = ("WALL BREAKS: a higher-capacity on-substrate dual-encoder reaches cull-grade "
                   "(>=85%% recall@20 unseen). Next: export + wire as Stage-1.")
    elif best['unseen'][20] >= r32['unseen'][20] + 15.0:
        verdict = (f"PARTIAL LIFT: +{lift:.1f}pp over r32 but not cull-grade (<85%). "
                   "Capacity helps; static-K dual-encoder still short of cull.")
    else:
        verdict = (f"PLATEAU: capacity does NOT break the wall (best {best['unseen'][20]:.1f}% vs "
                   f"r32 {r32['unseen'][20]:.1f}%, +{lift:.1f}pp). The static-K signal is TAPPED OUT "
                   "at this corpus; lever = heavier cross-encoder or distilled E2B.")
    log("VERDICT: " + verdict)
    logf.close()

if __name__ == "__main__":
    main()
