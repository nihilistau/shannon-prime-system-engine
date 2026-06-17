#!/usr/bin/env python3
# KAI-3 Stage 2.b GATE — torch audio_ctc.pt -> ONNX -> OpenVINO -> GNA_SW_EXACT (i16).
# Runs in the ARCHIVE OpenVINO 2023.3 runtime (setupvars). The GNA plugin quantizes to i16 internally at
# compile (the canonical GNA 2.0 PTQ). Measures the PURE quantization delta: FP32 CTC recovery (CPU) vs
# GNA i16 software-emulation CTC recovery, decoupled from any HW driver. Fixed compile-T (eval pad width);
# logits sliced to true frame length before the CTC greedy collapse.
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
    ap.add_argument("--frames", required=True); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--workdir", required=True); ap.add_argument("--T", type=int, default=0)
    a = ap.parse_args(); os.makedirs(a.workdir, exist_ok=True)
    import torch, torch.nn as nn, openvino as ov

    d = np.load(a.frames, allow_pickle=True)
    vsub = d["vsub_ids"]; V = len(vsub); n_mels = int(d["n_mels"]); BLANK = V
    evX = d["eval_X"].astype(np.float32); evY = d["eval_Y"]; evFL = d["eval_flen"]; evTL = d["eval_tlen"]
    exp = list(d["eval_expect"]) if "eval_expect" in d else None
    Tmax = a.T or int(evX.shape[1])
    print(f"[gna] V_sub={V} n_mels={n_mels} eval={evX.shape} compile_T={Tmax} expect={exp}", flush=True)

    class Enc(nn.Module):
        def __init__(s, h=256):
            super().__init__()
            s.net = nn.Sequential(nn.Conv1d(n_mels, h, 3, padding=1), nn.ReLU(),
                                  nn.Conv1d(h, h, 3, padding=1), nn.ReLU(),
                                  nn.Conv1d(h, h, 3, padding=1), nn.ReLU())
            s.head = nn.Conv1d(h, V + 1, 1)
        def forward(s, x): return s.head(s.net(x))     # x [B, n_mels, T] -> [B, V+1, T]
    net = Enc().eval(); ck = torch.load(a.ckpt, map_location="cpu"); net.load_state_dict(ck["state"])
    print(f"[gna] ckpt best={ck.get('best','?')}", flush=True)

    # torch FP32 reference (full-T then slice, to match the OV/GNA convention exactly)
    def tlog(i):
        x = torch.zeros(1, n_mels, Tmax); T = int(evFL[i]); x[0, :, :T] = torch.tensor(evX[i, :T].T)
        with torch.no_grad(): return net(x)[0].T.numpy()[:T]
    base_full = greedy_recovery([tlog(i) for i in range(evX.shape[0])], evY, evFL, evTL, V, BLANK)
    print(f"[gna] TORCH FP32 (full-T,slice) recovery = {base_full:.3f}", flush=True)

    onnx_p = os.path.join(a.workdir, "audio_ctc.onnx")
    torch.onnx.export(net, torch.zeros(1, n_mels, Tmax), onnx_p, input_names=["mel"],
                      output_names=["logits"], opset_version=11)
    core = ov.Core(); m = core.read_model(onnx_p)
    print(f"[gna] OV devices: {core.available_devices}", flush=True)

    def score(compiled, tag):
        logs = []
        for i in range(evX.shape[0]):
            T = int(evFL[i]); xin = np.zeros((1, n_mels, Tmax), np.float32); xin[0, :, :T] = evX[i, :T].T
            out = compiled(xin)[compiled.outputs[0]]
            logs.append(out[0].T[:T])
        r = greedy_recovery(logs, evY, evFL, evTL, V, BLANK)
        # per-event decision (ACTION/NO_OP) hit, mirroring the metal harness, just for context
        print(f"[gna] {tag} recovery = {r:.3f}  (delta vs FP32 {r-base_full:+.3f})", flush=True)
        return r

    score(core.compile_model(m, "CPU"), "OV CPU FP32")
    # GNA needs an explicit input scale factor (float->i16). Default auto-scale on wide-range log-mel can clip,
    # so sweep scale factors to separate "i16 fundamentally shears CTC" from "default scaling was wrong".
    best_r = -1; best_sf = None
    for sf in [None, 8.0, 64.0, 256.0, 1024.0, 2048.0, 4096.0]:
        cfg = {"GNA_DEVICE_MODE": "GNA_SW_EXACT", "INFERENCE_PRECISION_HINT": "i16"}
        if sf is not None: cfg["GNA_SCALE_FACTOR_0"] = str(sf)
        try:
            gna = core.compile_model(m, "GNA", cfg)
            r = score(gna, f"GNA_SW_EXACT i16 scale={sf}")
            if r > best_r: best_r, best_sf = r, sf
        except Exception as e:
            print(f"[gna] GNA compile [scale={sf}] FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)
    print(f"[gna] BEST GNA i16 recovery = {best_r:.3f} @ scale={best_sf}  (FP32 {base_full:.3f}, delta {best_r-base_full:+.3f})", flush=True)
    print("[gna] SCORE_DONE", flush=True)

if __name__ == "__main__":
    main()
