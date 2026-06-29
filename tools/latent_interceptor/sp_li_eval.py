#!/usr/bin/env python3
# sp_li_eval.py — EVAL-ONLY truth serum. Load a CLEAN-trained head (_tool_head.bin / _li_head.bin) and
# run it against an ADVERSARIAL captured set (feat.f32 + label.i32 + manifest). No training: this is
# the honest breaking point of the head we already built. Reports overall accuracy + confusion +
# per-class precision/recall (NONE-recall = near-miss survival; per-tool recall = paraphrase survival).
import argparse, json, numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True)      # _tool_head.bin (mu,sd,W1,b1,W2,b2)
    ap.add_argument("--data", required=True)      # adversarial capture dir (feat.f32, label.i32, manifest.jsonl)
    a = ap.parse_args()
    meta = json.loads(open(f"{a.data}/manifest.jsonl").readline())
    H, A, actions = meta["hidden"], meta["n_actions"], meta["actions"]
    feat = np.fromfile(f"{a.data}/feat.f32", dtype=np.float32).reshape(-1, H)
    lbl = np.fromfile(f"{a.data}/label.i32", dtype=np.int32)
    blob = np.fromfile(f"{a.head}", dtype=np.float32)
    proj = (len(blob) - 2 * H - A) // (H + 1 + A)
    o = 0
    mu = blob[o:o+H]; o += H
    sd = blob[o:o+H]; o += H
    W1 = blob[o:o+proj*H].reshape(proj, H); o += proj*H
    b1 = blob[o:o+proj]; o += proj
    W2 = blob[o:o+A*proj].reshape(A, proj); o += A*proj
    b2 = blob[o:o+A]; o += A
    print(f"[eval] head proj={proj} actions={actions} | N={len(lbl)} adversarial samples")

    fn = (feat - mu) / sd
    hid = np.maximum(fn @ W1.T + b1, 0.0)              # ReLU
    pred = (hid @ W2.T + b2).argmax(1)
    acc = (pred == lbl).mean()
    conf = np.zeros((A, A), int)
    for t, p in zip(lbl, pred): conf[t, p] += 1
    print(f"[eval] OVERALL adversarial accuracy = {acc:.3f}  ({(pred==lbl).sum()}/{len(lbl)})")
    print("[eval] confusion (rows=true, cols=pred):")
    print("           " + " ".join(f"{x[:6]:>7}" for x in actions))
    for i in range(A):
        print(f"  {actions[i]:>8} " + " ".join(f"{conf[i,j]:>7}" for j in range(A)))
    print("[eval] per-class precision / recall:")
    for i in range(A):
        tp = conf[i, i]; fn_ = conf[i].sum() - tp; fp = conf[:, i].sum() - tp
        prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn_)
        print(f"  {actions[i]:>8}: prec={prec:.3f} recall={rec:.3f}  (n={conf[i].sum()})")
    # the two that matter most for routing safety:
    none_i = actions.index("NONE") if "NONE" in actions else (actions.index("NO_OP") if "NO_OP" in actions else 0)
    none_rec = conf[none_i, none_i] / max(1, conf[none_i].sum())
    false_fire = (conf[none_i].sum() - conf[none_i, none_i]) / max(1, conf[none_i].sum())
    print(f"[eval] *** NONE recall (near-miss survival) = {none_rec:.3f} | FALSE-FIRE rate on NONE = {false_fire:.3f} ***")

if __name__ == "__main__":
    main()
