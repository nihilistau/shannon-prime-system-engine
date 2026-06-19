#!/usr/bin/env python3
"""Write wc_deploy.bin for recall.rs: magic 'WCB1', u32 hd, u32 r, f32 s0, f32 sscale, hd*r f32 W_c.
sscale = 1/sqrt(r) (the training score scale). Source = lsh_Wc_f32_div2.npz."""
import numpy as np, struct, sys, os
src = sys.argv[1] if len(sys.argv)>1 else r"D:\F\shannon-prime-repos\shannon-prime-system-engine\_b3_wc\lsh_Wc_f32_div2.npz"
out = sys.argv[2] if len(sys.argv)>2 else r"D:\F\shannon-prime-repos\shannon-prime-system-engine\_b3_wc\wc_deploy.bin"
z=np.load(src, allow_pickle=True)
W=z["Wc"].astype("<f4"); hd,r=W.shape; s0=float(z["s0"]); sscale=float(z["scale"])
with open(out,"wb") as f:
    f.write(b"WCB1"); f.write(struct.pack("<IIff", hd, r, s0, sscale)); f.write(W.tobytes(order="C"))
print(f"wrote {out}: hd={hd} r={r} s0={s0:+.4f} sscale={sscale:.5f}  ({os.path.getsize(out)} bytes)")
