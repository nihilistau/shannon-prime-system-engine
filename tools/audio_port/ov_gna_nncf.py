#!/usr/bin/env python3
# KAI-3 Stage 2.b recovery — NNCF PTQ with GNA-targeted INT16 calibration.
# Runs in the pip venv (nncf + openvino). Reads the exported ONNX, calibrates on a representative subset of
# the REAL training audio frames so NNCF observes the true activation distribution and applies smarter
# (percentile/range-estimated) clipping that protects the spiky CTC emission head from the default
# symmetric min/max shear. target_device=GNA => INT16-appropriate quant. Saves a portable quantized IR
# that the archive runtime scores on GNA_SW_EXACT.
import argparse, os, numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True); ap.add_argument("--onnx", required=True)
    ap.add_argument("--out_ir", required=True); ap.add_argument("--T", type=int, default=0)
    ap.add_argument("--subset", type=int, default=300)
    ap.add_argument("--preset", default="mixed", choices=["mixed", "performance"])
    ap.add_argument("--range", default="percentile", choices=["default", "percentile"])
    ap.add_argument("--target", default="any", help="nncf TargetDevice (GNA removed in 2.7; use any/cpu)")
    a = ap.parse_args()
    import openvino as ov, nncf
    print(f"[nncf] openvino {ov.__version__} nncf {nncf.__version__}", flush=True)

    d = np.load(a.frames, allow_pickle=True)
    n_mels = int(d["n_mels"])
    evX = d["eval_X"].astype(np.float32); evFL = d["eval_flen"]
    Tmax = a.T or int(evX.shape[1])
    # calibration: real train frames cropped/padded to Tmax, channels-first [1, n_mels, Tmax]
    cal = []
    if "train_X" in d:
        trX = d["train_X"].astype(np.float32); trFL = d["train_flen"]
        for i in range(min(a.subset, trX.shape[0])):
            T = min(int(trFL[i]), Tmax); x = np.zeros((1, n_mels, Tmax), np.float32); x[0, :, :T] = trX[i, :T].T
            cal.append(x)
    for i in range(evX.shape[0]):                          # + the eval frames themselves
        T = int(evFL[i]); x = np.zeros((1, n_mels, Tmax), np.float32); x[0, :, :T] = evX[i, :T].T; cal.append(x)
    print(f"[nncf] calibration samples = {len(cal)}  (Tmax={Tmax})", flush=True)

    core = ov.Core(); model = core.read_model(a.onnx)
    ds = nncf.Dataset(cal, lambda z: z)
    preset = nncf.QuantizationPreset.MIXED if a.preset == "mixed" else nncf.QuantizationPreset.PERFORMANCE
    kw = dict(preset=preset, subset_size=min(len(cal), a.subset))
    td = getattr(nncf.TargetDevice, a.target.upper(), None)
    if td is not None: kw["target_device"] = td
    if a.range == "percentile":
        try:
            from nncf.quantization.range_estimator import RangeEstimatorParametersSet
            kw["advanced_parameters"] = nncf.AdvancedQuantizationParameters(
                activations_range_estimator_params=RangeEstimatorParametersSet.QUANTILE)
        except Exception as e:
            print(f"[nncf] percentile range estimator unavailable ({type(e).__name__}); default ranges", flush=True)
    print(f"[nncf] quantize target={a.target} preset={a.preset} range={a.range}", flush=True)
    q = nncf.quantize(model, ds, **kw)
    ov.save_model(q, a.out_ir, compress_to_fp16=False)
    print(f"[nncf] saved quantized IR -> {a.out_ir}", flush=True)
    print("[nncf] NNCF_DONE", flush=True)

if __name__ == "__main__":
    main()
