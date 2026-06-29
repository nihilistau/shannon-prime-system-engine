#!/usr/bin/env python3
# export_adapter.py — flatten the proven npz adapter to a .bin the Rust daemon LatentBridge loads,
# plus a parity fixture (one source latent + its host-computed mapped vector) so the in-engine
# transfer can be checked against Python float-for-float (gate G-TELEPATHY-WIRE).
# bin layout (LE): i32 din, i32 dout, f32 gmu[din], gsd[din], qmu[dout], qsd[dout], W[din*dout] (C-order).
import numpy as np, struct
ad=np.load("telepathy_adapter_g2q.npz"); G=np.load("gemma_pairs.npy").astype(np.float32)
W=ad["W_fwd"].astype(np.float32); gmu,gsd,qmu,qsd=[ad[k].astype(np.float32) for k in ("gmu","gsd","qmu","qsd")]
din,dout=W.shape
with open("telepathy_adapter_g2q.bin","wb") as f:
    f.write(struct.pack("<ii",din,dout))
    for a in (gmu,gsd,qmu,qsd): f.write(a.tobytes())
    f.write(np.ascontiguousarray(W).tobytes())
# parity fixture: source latent row 0 -> host mapped
x=G[0]
y=(((x-gmu)/gsd)@W)*qsd+qmu
x.tofile("tele_src_latent.bin"); y.astype(np.float32).tofile("tele_expected_map.bin")
print(f"wrote telepathy_adapter_g2q.bin (din={din} dout={dout}) + parity fixture; |y|={np.linalg.norm(y):.3f}")
