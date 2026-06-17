#!/usr/bin/env python3
# Score a (possibly quantized) OpenVINO IR on CPU + GNA_SW_EXACT i16 for CTC token recovery.
# Runs in the archive runtime (no torch/nncf needed). Used to score the NNCF GNA-INT16 IR.
import argparse, os, numpy as np

def greedy_recovery(logits_list, Y, FL, TL, V, BLANK):
    ok = tot = 0
    for i, lg in enumerate(logits_list):
        pred = lg.argmax(-1).tolist(); col = []; prev = -1
        for s in pred:
            if s != prev and s != BLANK: col.append(s)
            prev = s
        tg = Y[i, :TL[i]].tolist()
        ok += sum(1 for j in range(min(len(col), len(tg))) if col[j] == tg[j]); tot += len(tg)
    return ok / max(tot, 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True); ap.add_argument("--ir", required=True)
    ap.add_argument("--tag", default="IR"); ap.add_argument("--T", type=int, default=0)
    a = ap.parse_args()
    import openvino as ov
    d = np.load(a.frames, allow_pickle=True)
    V = len(d["vsub_ids"]); n_mels = int(d["n_mels"]); BLANK = V
    evX = d["eval_X"].astype(np.float32); evY = d["eval_Y"]; evFL = d["eval_flen"]; evTL = d["eval_tlen"]
    Tmax = a.T or int(evX.shape[1])
    core = ov.Core(); m = core.read_model(a.ir)
    print(f"[ir] {a.tag} devices={core.available_devices} V={V} Tmax={Tmax}", flush=True)

    def score(compiled, tag):
        logs = []
        for i in range(evX.shape[0]):
            T = int(evFL[i]); xin = np.zeros((1, n_mels, Tmax), np.float32); xin[0, :, :T] = evX[i, :T].T
            out = compiled(xin)[compiled.outputs[0]]
            logs.append(out[0].T[:T])
        r = greedy_recovery(logs, evY, evFL, evTL, V, BLANK)
        print(f"[ir] {tag} recovery = {r:.3f}", flush=True); return r

    score(core.compile_model(m, "CPU"), f"{a.tag} CPU")
    best = -1; bsf = None
    for sf in [None, 64.0, 1024.0, 2048.0]:
        cfg = {"GNA_DEVICE_MODE": "GNA_SW_EXACT", "INFERENCE_PRECISION_HINT": "i16"}
        if sf is not None: cfg["GNA_SCALE_FACTOR_0"] = str(sf)
        try:
            r = score(core.compile_model(m, "GNA", cfg), f"{a.tag} GNA_SW_EXACT i16 scale={sf}")
            if r > best: best, bsf = r, sf
        except Exception as e:
            print(f"[ir] GNA [scale={sf}] FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
    print(f"[ir] BEST {a.tag} GNA i16 = {best:.3f} @ scale={bsf}", flush=True)
    print("[ir] IR_SCORE_DONE", flush=True)

if __name__ == "__main__":
    main()
