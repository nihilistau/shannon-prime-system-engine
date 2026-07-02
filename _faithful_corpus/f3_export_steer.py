"""f3_export_steer.py — export the Tier-A steering vector v̄ (G-FLOW-STRAIGHTNESS).

v̄ = mean over pairs of (A frame-0 − B frame-0): the mean faithfulness velocity at
the last-prompt-token DECIDE state. Raw 3840 LE f32, UNNORMALIZED — SP_STEER_ALPHA=1.0
means "apply exactly the measured mean shift". Also exports the fct-only variant.
"""
import json, struct
import numpy as np

ENG = __file__.rsplit("_faithful_corpus", 1)[0]
E = 3840

def load(run):
    d = f"{ENG}_faithful_corpus/f3/{run}"
    out = {}
    for line in open(f"{d}/f3_meta.jsonl", encoding="utf-8"):
        m = json.loads(line)
        raw = open(f"{d}/f3_{m['chat_id']}.bin", "rb").read()
        out[m["user"]] = np.frombuffer(raw, dtype="<f4", offset=16).reshape(2, E).astype(np.float64)
    return out

A, B = load("A"), load("B")
users = sorted(set(A) & set(B))
V = np.stack([A[u][0] - B[u][0] for u in users])
vbar_all = V.mean(axis=0)
fct = [u for u in users if "Node-" not in u]
vbar_fct = np.stack([A[u][0] - B[u][0] for u in fct]).mean(axis=0)

for name, v in (("steer_vbar_f0_all81.bin", vbar_all), ("steer_vbar_f0_fct61.bin", vbar_fct)):
    open(f"{ENG}_faithful_corpus/f3/{name}", "wb").write(v.astype("<f4").tobytes())
    print(f"{name}: dim={len(v)} norm={np.linalg.norm(v):.3f}")
print(f"cos(all81, fct61) = {float(vbar_all @ vbar_fct / (np.linalg.norm(vbar_all)*np.linalg.norm(vbar_fct))):.4f}")
