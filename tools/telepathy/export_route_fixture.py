#!/usr/bin/env python3
# export_route_fixture.py — pick one LOCAL + one TELEPATHY captured feat from the route OOD set so the
# daemon can demonstrate the hardened Route head GOVERNING decide_route in-engine (matches Python eval).
# route_fixture.bin (LE): i32 H, i32 A, i32 proj, i32 n, then n*(i32 label + f32 feat[H]).
import json, numpy as np, struct
D="../latent_interceptor/_hard_route_ood_data"
meta=json.loads(open(f"{D}/manifest.jsonl").readline()); H=meta["hidden"]; A=meta["n_actions"]
feat=np.fromfile(f"{D}/feat.f32",dtype=np.float32).reshape(-1,H); lbl=np.fromfile(f"{D}/label.i32",dtype=np.int32)
blob=np.fromfile("../latent_interceptor/_route_head.bin",dtype=np.float32)
proj=(len(blob)-2*H-A)//(H+1+A)
# one of each class
pick=[]
for c in range(A):
    idx=np.where(lbl==c)[0]
    if len(idx): pick.append((int(c),idx[0]))
with open("route_fixture.bin","wb") as f:
    f.write(struct.pack("<iiii",H,A,proj,len(pick)))
    for c,i in pick:
        f.write(struct.pack("<i",c)); f.write(feat[i].astype(np.float32).tobytes())
print(f"wrote route_fixture.bin H={H} A={A} proj={proj} n={len(pick)} labels={[c for c,_ in pick]} actions={meta['actions']}")
