#!/usr/bin/env python3
# KAI-3 §7.3 (GNA EAR) — the real-audio CTC projector. STAGED for Task #154.a (validated after corpus bake).
#
# Encoder (GNA-CONSERVATIVE, runs on GNA 2.0/3.0): small STANDARD Conv1d stack over log-mel (time axis),
# i16-friendly, out-ch ×4 ≤256, batch=1, 1D-native/flattenable. log-mel[T,n_mels] -> per-frame logits over
# V_sub + blank.  Objective: torch.nn.CTCLoss (the proven impl; NOT a hand-roll, NOT the absent SP-CTC) —
# aligns the T audio frames to the SHORTER token target, so frames!=tokens is handled natively (the thing
# adaptive-pooling would have been a throwaway crutch for; CTC is the streaming-EAR-correct objective).
#
# Inference: per non-blank frame softmax(logits[:V]/tau)·W_sub (W_sub = real OK_Q4B embed rows ×√H) ->
# on-manifold vector; CTC-greedy-collapsed sequence -> KAI2 packet -> gemma4_kv_inject_seq -> 12B pivot
# (SP_G4_KAI3 metal gate). The frozen KAI-3 binder/inject is unchanged from the synthetic G-KAIROS-3 GREEN.
import argparse, glob, os, struct
import numpy as np

def load_embed_rows(model_dir, vsub_ids):
    from safetensors import safe_open
    cand = ["model.embed_tokens.weight", "language_model.model.embed_tokens.weight",
            "model.language_model.embed_tokens.weight"]
    for f in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        with safe_open(f, framework="pt") as st:
            ks = set(st.keys())
            for nm in cand:
                if nm in ks:
                    w = st.get_tensor(nm); return w[vsub_ids].float().numpy(), int(w.shape[1])
    raise KeyError("embed_tokens weight not found")

def write_kai2_packet(path, vecs):
    k, h = vecs.shape
    with open(path, "wb") as f:
        f.write(b"KAI2"); f.write(struct.pack("<II", k, h)); f.write(np.ascontiguousarray(vecs, "<f4").tobytes())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help=".npz from gen_audio_frames.py")
    ap.add_argument("--model", required=True, help="gemma-4 safetensors dir (embed_tokens for W_sub)")
    ap.add_argument("--epochs", type=int, default=120); ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=256)   # GNA-conservative: out-ch <=256
    ap.add_argument("--export_tau", type=float, default=0.2)
    ap.add_argument("--out", default="audio_ctc.pt")
    ap.add_argument("--load", default=None, help="load a trained ckpt and skip training (re-export only)")
    ap.add_argument("--export", action="store_true"); ap.add_argument("--packets_dir", default="kai3_audio_packets")
    ap.add_argument("--manifest_out", default=None); ap.add_argument("--manifest_prefix", default="")
    a = ap.parse_args()
    import torch, torch.nn as nn, torch.nn.functional as F
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    d = np.load(a.frames, allow_pickle=True)
    vsub = d["vsub_ids"]; V = len(vsub); n_mels = int(d["n_mels"]); BLANK = V
    Wrows, H = load_embed_rows(a.model, vsub)
    Wsub = torch.tensor(Wrows, device=dev) * (H ** 0.5)
    trX = torch.tensor(d["train_X"], device=dev); trY = torch.tensor(d["train_Y"], device=dev)
    trFL = torch.tensor(d["train_flen"], device=dev); trTL = torch.tensor(d["train_tlen"], device=dev)
    haveEval = "eval_X" in d
    print(f"[actc] V_sub={V} H={H} n_mels={n_mels} train={trX.shape[0]} frames<= {trX.shape[1]} dev={dev}", flush=True)

    class Enc(nn.Module):  # GNA-conservative standard Conv1d over time; channels = n_mels -> hidden -> V+1
        def __init__(s):
            super().__init__()
            h = a.hidden
            s.net = nn.Sequential(
                nn.Conv1d(n_mels, h, 3, padding=1), nn.ReLU(),
                nn.Conv1d(h, h, 3, padding=1), nn.ReLU(),
                nn.Conv1d(h, h, 3, padding=1), nn.ReLU())
            s.head = nn.Conv1d(h, V + 1, 1)            # 1x1 -> per-frame logits over V_sub + blank
        def forward(s, x):                              # x [B,T,n_mels] -> logits [B,T,V+1]
            return s.head(s.net(x.transpose(1, 2))).transpose(1, 2)
    net = Enc().to(dev); opt = torch.optim.Adam(net.parameters(), lr=a.lr)

    def ctc(X, Y, FL, TL):
        logits = net(X)                                 # [B,T,V+1]
        logp = F.log_softmax(logits, -1).transpose(0, 1)  # [T,B,V+1] for CTCLoss
        tgt = torch.cat([Y[i, :TL[i]] for i in range(Y.shape[0])])
        return F.ctc_loss(logp, tgt, FL, TL, blank=BLANK, zero_infinity=True)

    def greedy_tok_acc(X, Y, FL, TL):                   # CTC greedy collapse -> token edit match (held-out)
        with torch.no_grad():
            pred = net(X).argmax(-1)
            ok = tot = 0
            for i in range(X.shape[0]):
                seq = pred[i, :FL[i]].tolist(); col = []
                prev = -1
                for s in seq:
                    if s != prev and s != BLANK: col.append(s)
                    prev = s
                tg = Y[i, :TL[i]].tolist()
                ok += sum(1 for j in range(min(len(col), len(tg))) if col[j] == tg[j]); tot += len(tg)
            return ok / max(tot, 1)

    if a.load and os.path.exists(a.load):
        ck = torch.load(a.load, map_location=dev); net.load_state_dict(ck["state"])
        print(f"[actc] loaded ckpt {a.load} (best={ck.get('best','?')}) — skipping train, re-export only", flush=True)
        a.epochs = 0
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, a.epochs), eta_min=1e-5)
    N = trX.shape[0]; bs = a.batch_size
    best = -1.0; best_state = None
    for ep in range(a.epochs):
        net.train(); perm = torch.randperm(N, device=dev); acc=0.0
        for s in range(0, N, bs):
            idx = perm[s:s+bs]; opt.zero_grad()
            loss = ctc(trX[idx], trY[idx], trFL[idx], trTL[idx]); loss.backward(); opt.step(); acc += float(loss)
        sched.step(); loss = acc / max(1, (N + bs - 1)//bs)
        if ep % 10 == 0 or ep == a.epochs - 1:
            net.eval()
            ev = greedy_tok_acc(torch.tensor(d["eval_X"], device=dev), torch.tensor(d["eval_Y"], device=dev),
                                torch.tensor(d["eval_flen"], device=dev), torch.tensor(d["eval_tlen"], device=dev)) if haveEval else 0.0
            if ev > best: best = ev; best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            print(f"[actc] ep {ep:3d} ctc={float(loss):.4f} eval_tok_acc={ev:.3f} best={best:.3f}", flush=True)
    if best_state: net.load_state_dict(best_state)
    print(f"[actc] BEST held-out CTC greedy token acc = {best:.3f}", flush=True)
    import torch as _t; _t.save({"state": net.state_dict(), "vsub": vsub, "H": H, "n_mels": n_mels,
                                 "export_tau": a.export_tau, "best": best}, a.out)

    if a.export and haveEval:
        os.makedirs(a.packets_dir, exist_ok=True)
        man = open(a.manifest_out, "w") if a.manifest_out else None
        expect = d["eval_expect"] if "eval_expect" in d else None
        evX = torch.tensor(d["eval_X"], device=dev); evFL = d["eval_flen"]
        net.eval()
        with torch.no_grad():
            for i in range(evX.shape[0]):
                T = int(evFL[i]); logits = net(evX[i:i+1, :T])[0]      # [T,V+1]
                pred = logits.argmax(-1).tolist(); prev = -1; keep = []
                for ti, s in enumerate(pred):
                    if s != prev and s != BLANK: keep.append(ti)
                    prev = s
                if not keep: continue
                sub = logits[keep, :V] / a.export_tau
                Eout = (torch.softmax(sub, -1) @ Wsub).cpu().numpy()    # [k,H] on-manifold
                exp = str(expect[i]) if expect is not None else f"ev{i}"
                base = f"aud_{i:02d}_{exp}.bin"; write_kai2_packet(os.path.join(a.packets_dir, base), Eout)
                if man: man.write(f"{a.manifest_prefix}{base} {exp}\n")
                print(f"[export] {base} k={len(keep)} hidden={H} expect={exp}", flush=True)
        if man: man.close()

if __name__ == "__main__":
    main()
