#!/usr/bin/env python3
# KAI-3 §7.3 — the 640-float -> E=3840 frame projector.
#
# Architecture (CONTRACT-KAIROS §7.3, verified with the operator 2026-06-16):
#   Mapper:  640-float frame -> logits over V_sub  (per-position, sequence length N preserved)
#   Binder:  E_out_i = softmax(logits_i / tau) @ W_sub,  W_sub = OK_Q4B embed rows[V_sub] * sqrt(H)
#            => every emitted vector is a convex combo of REAL token embeddings => on-manifold by
#            construction (cos ~ 1.0 at sharp tau), the t10 fix recycled per-frame.
#
# Objective: DENSE per-position cross-entropy CE(logits_i, token_i)  (NOT decision-KL). This is the
#   structural fix for the t10 0.9157 plateau: the projector is supervised at every timestep to
#   recover the correct token identity; the Phase-1 EMB control already proved the 12B pivots when
#   it receives the right token sequence, so the pivot is a CONSEQUENCE we verify on metal, never the
#   training signal.
#
# Train at temperature 1 (well-conditioned dense gradient); EXPORT at sharp tau (~0.2) so the served
# vectors are near-discrete on-manifold tokens (= the EMB vectors that pivoted 2/2).
import argparse, os, struct, sys
import numpy as np

def load_embed_rows(model_dir, vsub_ids):
    """Pull ONLY the embed_tokens rows for V_sub from the model safetensors (no full-model load)."""
    from safetensors import safe_open
    import glob
    cand_names = ["model.embed_tokens.weight",
                  "language_model.model.embed_tokens.weight",
                  "model.language_model.embed_tokens.weight"]
    files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no .safetensors in {model_dir}")
    for f in files:
        with safe_open(f, framework="pt") as st:
            keys = set(st.keys())
            for nm in cand_names:
                if nm in keys:
                    w = st.get_tensor(nm)          # [vocab, H] bf16/f16
                    rows = w[vsub_ids].float().numpy()
                    return rows, int(w.shape[1])
    raise KeyError(f"embed_tokens weight not found; tried {cand_names}")

def write_kai2_packet(path, vecs):
    """'KAI2' | u32 k | u32 hidden | k*hidden f32  (matches tests/test_gemma4_cuda.c kai2_read_packet)."""
    k, h = vecs.shape
    with open(path, "wb") as f:
        f.write(b"KAI2"); f.write(struct.pack("<II", k, h))
        f.write(np.ascontiguousarray(vecs, dtype="<f4").tobytes())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help=".npz from gen_synth_frames.py")
    ap.add_argument("--model", required=True, help="dir with the gemma-4 safetensors (embed_tokens)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--export_tau", type=float, default=0.2, help="sharp tau at serve/export")
    ap.add_argument("--out", default="frame_projector.pt")
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--packets_dir", default="kai3_packets")
    ap.add_argument("--manifest_out", default=None, help="write '<prefix><pkt> <EXPECT>' lines for the metal gate")
    ap.add_argument("--manifest_prefix", default="", help="Windows path prefix for packet paths in the manifest")
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    d = np.load(args.frames, allow_pickle=True)
    vsub = d["vsub_ids"]; V = len(vsub); H = None
    Wrows, H = load_embed_rows(args.model, vsub)
    Wsub = torch.tensor(Wrows, device=dev) * (H ** 0.5)          # frozen, [V, H]  (×sqrt(H) = native dx)
    fd = int(d["frame_dim"])
    trX = torch.tensor(d["train_X"], device=dev); trT = torch.tensor(d["train_T"], device=dev)
    evX = torch.tensor(d["eval_X"],  device=dev); evT = torch.tensor(d["eval_T"],  device=dev)
    print(f"[proj] V_sub={V} H={H} frame_dim={fd} train={trX.shape[0]} eval={evX.shape[0]} dev={dev} "
          f"sigma={float(d['sigma']):.4f}", flush=True)

    class Mapper(nn.Module):
        def __init__(s):
            super().__init__()
            s.net = nn.Sequential(nn.Linear(fd, args.hidden), nn.GELU(),
                                  nn.Linear(args.hidden, args.hidden), nn.GELU(),
                                  nn.Linear(args.hidden, V))
        def forward(s, x):                                       # x [.,N,fd] -> [.,N,V]
            return s.net(x)
    net = Mapper().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    def ce(X, T):
        logits = net(X)                                         # [B,N,V]
        return F.cross_entropy(logits.reshape(-1, V), T.reshape(-1), ignore_index=-100)
    def top1(X, T):
        with torch.no_grad():
            pred = net(X).argmax(-1)
            mask = (T != -100)
            return float((pred[mask] == T[mask]).float().mean())

    best_val = -1.0; best_state = None
    for ep in range(args.epochs):
        net.train(); opt.zero_grad()
        loss = ce(trX, trT); loss.backward(); opt.step()
        net.eval(); va = top1(evX, evT); ta = top1(trX, trT)
        if va > best_val: best_val = va; best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"[proj] ep {ep:3d} CE={float(loss):.4f} train_top1={ta:.3f} eval_top1={va:.3f} best={best_val:.3f}", flush=True)
    if best_state: net.load_state_dict(best_state)
    print(f"[proj] BEST held-out per-position token-recovery top1 = {best_val:.3f} (>=0.95 => projector recovers the sequence)", flush=True)

    # cos tripwire: sharp-tau binder output of eval frames must be ~1.0 to the full V_sub manifold
    net.eval()
    with torch.no_grad():
        L = net(evX) / args.export_tau
        P = F.softmax(L, -1)
        Eout = P @ Wsub                                         # [B,N,H]
        En = F.normalize(Eout, dim=-1); Wn = F.normalize(Wsub, dim=-1)
        # max cos of each emitted vector to any V_sub row
        mask = (evT != -100)
        cos = (En.reshape(-1, H) @ Wn.t()).max(-1).values.reshape(evT.shape)
        mc = float(cos[mask].mean())
    print(f"[proj] manifold tripwire: eval emitted-vector mean max-cos to V_sub = {mc:.4f} "
          f"(sharp tau={args.export_tau}; need >>0.07, target ~1.0)", flush=True)

    torch.save({"state": net.state_dict(), "vsub": vsub, "H": H, "fd": fd,
                "export_tau": args.export_tau, "best_val_top1": best_val}, args.out)
    print(f"[proj] saved {args.out}", flush=True)

    if args.export:
        os.makedirs(args.packets_dir, exist_ok=True)
        texts  = d["eval_texts"]; expect = d["eval_expect"]; elen = d["eval_len"]
        man = open(args.manifest_out, "w") if args.manifest_out else None
        with torch.no_grad():
            for i in range(evX.shape[0]):
                n = int(elen[i])
                L = net(evX[i:i+1, :n]) / args.export_tau
                Eout = (F.softmax(L, -1) @ Wsub)[0].cpu().numpy()      # [n, H]
                base = f"eval_{i:02d}_{str(expect[i])}.bin"
                fn = os.path.join(args.packets_dir, base)
                write_kai2_packet(fn, Eout)
                if man: man.write(f"{args.manifest_prefix}{base} {str(expect[i])}\n")
                print(f"[export] {fn}  k={n} hidden={H} expect={expect[i]}", flush=True)
        if man: man.close(); print(f"[export] manifest -> {args.manifest_out}", flush=True)

if __name__ == "__main__":
    main()
