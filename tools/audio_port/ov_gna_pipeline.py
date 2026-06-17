#!/usr/bin/env python3
# KAI-3 Stage 2.b — the OpenVINO/NNCF/GNA quantization gate.
# torch audio_ctc.pt -> ONNX -> OpenVINO IR (FP32) -> NNCF PTQ (i16, GNA footprint) -> score on GNA_SW_EXACT.
# Measures the PURE quantization delta: FP32-IR CTC recovery vs quantized CTC recovery, the decoupled
# math comparison the operator asked for. The encoder is fully-convolutional; GNA needs static shapes, so we
# compile at fixed T (eval pad width) and slice logits to true frame length before the CTC greedy collapse.
import argparse, os, sys, numpy as np

def greedy_recovery(logits_list, Y, FL, TL, V, BLANK):
    # logits_list[i] = [T_i, V+1] numpy (already sliced to FL[i]); CTC greedy collapse -> token edit match
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
    ap.add_argument("--frames", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--T", type=int, default=0, help="fixed compile time-len; 0 = max eval flen")
    ap.add_argument("--try_gna", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.workdir, exist_ok=True)
    import torch, torch.nn as nn
    import openvino as ov

    d = np.load(a.frames, allow_pickle=True)
    vsub = d["vsub_ids"]; V = len(vsub); n_mels = int(d["n_mels"]); BLANK = V
    H = None
    evX = d["eval_X"].astype(np.float32); evY = d["eval_Y"]; evFL = d["eval_flen"]; evTL = d["eval_tlen"]
    Tmax = a.T or int(evX.shape[1])
    print(f"[ovp] V_sub={V} n_mels={n_mels} eval={evX.shape} compile_T={Tmax} BLANK={BLANK}", flush=True)

    # --- rebuild the frozen Enc (must match audio_ctc_projector.Enc) ---
    class Enc(nn.Module):
        def __init__(s, h=256):
            super().__init__()
            s.net = nn.Sequential(
                nn.Conv1d(n_mels, h, 3, padding=1), nn.ReLU(),
                nn.Conv1d(h, h, 3, padding=1), nn.ReLU(),
                nn.Conv1d(h, h, 3, padding=1), nn.ReLU())
            s.head = nn.Conv1d(h, V + 1, 1)
        def forward(s, x):                 # x [B, n_mels, T]  (NCHW-ish, channels=mels) -> [B, V+1, T]
            return s.head(s.net(x))
    net = Enc().to("cpu").eval()
    ck = torch.load(a.ckpt, map_location="cpu")
    # state dict was saved from a model that did x.transpose(1,2) then conv; weights are identical, only the
    # forward input layout differs. Load directly (conv weights are layout-agnostic).
    net.load_state_dict(ck["state"])
    print(f"[ovp] loaded ckpt best={ck.get('best','?')}", flush=True)

    # torch FP32 baseline (channels-first input [1, n_mels, T], sliced to FL) -- the reference
    def torch_logits(i):
        T = int(evFL[i]); x = torch.tensor(evX[i, :T].T[None])        # [1, n_mels, T]
        with torch.no_grad(): return net(x)[0].T.numpy()              # [T, V+1]
    tlogs = [torch_logits(i) for i in range(evX.shape[0])]
    base = greedy_recovery(tlogs, evY, evFL, evTL, V, BLANK)
    print(f"[ovp] TORCH FP32 recovery = {base:.3f}", flush=True)

    # --- ONNX export (fixed T, channels-first [1, n_mels, T]) ---
    onnx_p = os.path.join(a.workdir, "audio_ctc.onnx")
    dummy = torch.zeros(1, n_mels, Tmax)
    torch.onnx.export(net, dummy, onnx_p, input_names=["mel"], output_names=["logits"],
                      opset_version=11, dynamic_axes=None)
    print(f"[ovp] onnx -> {onnx_p}", flush=True)

    # --- ONNX -> OV IR (FP32) ---
    core = ov.Core()
    ov_model = core.read_model(onnx_p)
    ir_fp32 = os.path.join(a.workdir, "audio_ctc_fp32.xml")
    ov.save_model(ov_model, ir_fp32, compress_to_fp16=False)
    print(f"[ovp] OV IR(fp32) -> {ir_fp32}", flush=True)

    def ov_recovery(compiled, tag):
        logs = []
        for i in range(evX.shape[0]):
            T = int(evFL[i])
            xin = np.zeros((1, n_mels, Tmax), np.float32); xin[0, :, :T] = evX[i, :T].T
            out = compiled(xin)[compiled.outputs[0]]                 # [1, V+1, Tmax]
            logs.append(out[0].T[:T])                                # [T, V+1]
        r = greedy_recovery(logs, evY, evFL, evTL, V, BLANK)
        print(f"[ovp] {tag} recovery = {r:.3f}  (delta vs torch {r-base:+.3f})", flush=True)
        return r

    cpu_fp32 = core.compile_model(ov_model, "CPU")
    ov_recovery(cpu_fp32, "OV-IR FP32 (CPU)")

    # --- NNCF PTQ (i16 weights, GNA footprint) ---
    try:
        import nncf
        # calibration: eval frames + a slice of train frames, all cropped/padded to Tmax, channels-first
        cal = []
        for i in range(evX.shape[0]):
            T = int(evFL[i]); x = np.zeros((1, n_mels, Tmax), np.float32); x[0, :, :T] = evX[i, :T].T; cal.append(x)
        if "train_X" in d:
            trX = d["train_X"].astype(np.float32); trFL = d["train_flen"]
            for i in range(0, min(300, trX.shape[0])):
                T = min(int(trFL[i]), Tmax); x = np.zeros((1, n_mels, Tmax), np.float32)
                x[0, :, :T] = trX[i, :T].T; cal.append(x)
        cal_ds = nncf.Dataset(cal, lambda z: z)
        q_model = nncf.quantize(ov_model, cal_ds, model_type=nncf.ModelType.TRANSFORMER if False else None,
                                preset=nncf.QuantizationPreset.PERFORMANCE, subset_size=min(len(cal), 300))
        ir_q = os.path.join(a.workdir, "audio_ctc_i8.xml")
        ov.save_model(q_model, ir_q, compress_to_fp16=False)
        print(f"[ovp] NNCF PTQ IR -> {ir_q}", flush=True)
        cpu_q = core.compile_model(q_model, "CPU")
        ov_recovery(cpu_q, "NNCF-PTQ (CPU sim)")
    except Exception as e:
        q_model = None
        print(f"[ovp] NNCF PTQ FAILED: {type(e).__name__}: {str(e)[:400]}", flush=True)

    # --- GNA SW_EXACT (i16) — the operator's gate ---
    if a.try_gna:
        for which, m in (("FP32-IR", ov_model), ("PTQ-IR", q_model)):
            if m is None: continue
            try:
                gna = core.compile_model(m, "GNA", {"GNA_DEVICE_MODE": "GNA_SW_EXACT",
                                                    "INFERENCE_PRECISION_HINT": "i16"})
                ov_recovery(gna, f"GNA_SW_EXACT i16 [{which}]")
            except Exception as e:
                print(f"[ovp] GNA compile [{which}] FAILED: {type(e).__name__}: {str(e)[:400]}", flush=True)
    print("[ovp] PIPELINE_DONE", flush=True)

if __name__ == "__main__":
    main()
