#!/usr/bin/env python3
# KAI-3 Stage 2.b — POT DefaultQuantization, GNA target (the GNA-native i16 PTQ).
# POT (openvino-dev 2023.3) emits quantization params in the exact format the libGNA graph compiler trusts
# (it owns the scale factors), unlike NNCF's generic FakeQuantize which the GNA backend rejects. Calibrates
# on the same real-frame subset. Output IR is scored on GNA_SW_EXACT in the archive runtime.
import argparse, os, numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True); ap.add_argument("--onnx", required=True)
    ap.add_argument("--outdir", required=True); ap.add_argument("--T", type=int, default=0)
    ap.add_argument("--subset", type=int, default=300); ap.add_argument("--preset", default="mixed")
    a = ap.parse_args(); os.makedirs(a.outdir, exist_ok=True)
    import openvino as ov
    from openvino.tools.pot import DataLoader, IEEngine, load_model, save_model, create_pipeline

    d = np.load(a.frames, allow_pickle=True)
    n_mels = int(d["n_mels"]); evX = d["eval_X"].astype(np.float32); evFL = d["eval_flen"]
    Tmax = a.T or int(evX.shape[1])
    cal = []                                   # store WITHOUT batch dim ([n_mels, Tmax]); POT adds batch
    if "train_X" in d:
        trX = d["train_X"].astype(np.float32); trFL = d["train_flen"]
        for i in range(min(a.subset, trX.shape[0])):
            T = min(int(trFL[i]), Tmax); x = np.zeros((n_mels, Tmax), np.float32); x[:, :T] = trX[i, :T].T; cal.append(x)
    for i in range(evX.shape[0]):
        T = int(evFL[i]); x = np.zeros((n_mels, Tmax), np.float32); x[:, :T] = evX[i, :T].T; cal.append(x)
    print(f"[pot] calibration={len(cal)} Tmax={Tmax}", flush=True)

    # FP32 IR for POT (load_model wants OV IR xml/bin)
    fp32_xml = os.path.join(a.outdir, "audio_ctc_fp32.xml")
    ov.save_model(ov.Core().read_model(a.onnx), fp32_xml, compress_to_fp16=False)

    class FrameLoader(DataLoader):
        def __init__(s, frames): super().__init__({}); s.frames = frames
        def __len__(s): return len(s.frames)
        def __getitem__(s, i): return s.frames[i], None          # (data, annotation); DefaultQuant ignores annotation

    model = load_model({"model_name": "audio_ctc", "model": fp32_xml,
                        "weights": fp32_xml.replace(".xml", ".bin")})
    engine = IEEngine(config={"device": "CPU"}, data_loader=FrameLoader(cal))
    algos = [{"name": "DefaultQuantization",
              "params": {"target_device": "GNA", "stat_subset_size": min(len(cal), a.subset), "preset": a.preset}}]
    print(f"[pot] DefaultQuantization target=GNA preset={a.preset}", flush=True)
    pipeline = create_pipeline(algos, engine)
    compressed = pipeline.run(model)
    paths = save_model(compressed, save_path=a.outdir, model_name="audio_ctc_pot_gna")
    print(f"[pot] saved: {paths}", flush=True)
    print("[pot] POT_DONE", flush=True)

if __name__ == "__main__":
    main()
