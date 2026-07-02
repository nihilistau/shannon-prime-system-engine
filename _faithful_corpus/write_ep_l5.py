"""write_ep_l5.py — stage the L5 query-keys for the live SP_RECALL_L5 gate.

For each registry episode fct_NNN (facts.json order), compute the exact-question L5
key = L2-normalized mean-over-heads of global layer 5 of the exact-query global-Q
capture (qdump/q_*.bin, sorted-cid == fact order), and write it as <dir>/ep.l5
(raw little-endian f32[512]) — exactly what recall::load_episode_l5key reads.

Usage: python write_ep_l5.py <registry.jsonl> <qdump_exact_dir>
"""
import os, sys, glob, struct, json
import numpy as np
L5, HD, G_NH = 5, 512, 16

def load(p):
    b=open(p,"rb").read(); ng,d=struct.unpack("<II",b[:8]); return ng,np.frombuffer(b[8:],"<f4").astype(np.float64)

reg, exdir = sys.argv[1], sys.argv[2]
rows=[json.loads(l) for l in open(reg,encoding="utf-8") if l.strip()]
EPS_BASE=os.path.join(os.path.dirname(os.path.abspath(reg)),"eps")  # mount-relative, ignore registry's D:/ paths
# qdump exact, sorted by cid == fact order
qs=sorted(glob.glob(os.path.join(exdir,"q_*.bin")), key=lambda p:int(os.path.basename(p)[2:-4]))
n=min(len(rows),len(qs))
print(f"registry rows={len(rows)}  qdump exact={len(qs)}  writing {n} ep.l5 keys")
wrote=0
for i in range(n):
    ng,a=load(qs[i]); q=a.reshape(ng,G_NH,HD)
    v=q[L5].mean(0)                        # [512] mean over heads, global layer 5
    v=v/(np.linalg.norm(v)+1e-30)
    d=os.path.join(EPS_BASE, rows[i]["name"])
    if not os.path.isdir(d):
        print(f"  MISS dir {d}"); continue
    with open(os.path.join(d,"ep.l5"),"wb") as f:
        f.write(v.astype("<f4").tobytes())
    wrote+=1
print(f"wrote {wrote} ep.l5 sidecars (512 f32 each). Registry order fct_000.. == qdump sorted order.")
