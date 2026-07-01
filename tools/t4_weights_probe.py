#!/usr/bin/env python3
# t4_weights_probe.py -- G-T4-WEIGHTS pre-registered feasibility kill-test.
#
# Question (papers/PPT-LAT-T4-WEIGHTS-SCOPE.md): does the T4 Frobenius pi^k rank-2 codec
# (validated on Ring-2 episodes as G-R2-FROB, tools/curator/frob_episode.py) beat the shipped
# OK_Q4B (~4.5 effective bits/weight = 4-bit codes + one f16 scale per 32-block) on REAL
# gemma-4-12B weight tensors -- either fewer bits at <= its reconstruction error, or lower error
# at <= 4.5 bits? If matching fidelity needs the scale to shrink back to per-block, the Frobenius
# "free scale" does NOT transfer to weights => HONEST-NEGATIVE (T4 redundant vs OK_Q4B).
#
# Offline: parses the safetensors header and reads only the probed tensors (no GPU, no full load).
# The T4 codec math is IDENTICAL to frob_episode.encode (per-group max-abs pi^k scale s_a, coarse
# coord a, optional error-feedback residual b at s_b) -- here wrapped with a configurable GROUP so
# we can sweep scale granularity {per-tensor, per-row, per-32-block}.
import sys, json, struct, numpy as np

def st_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return hdr, 8 + n

def read_tensor(path, hdr, base, name):
    e = hdr[name]; s, t = e["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + s); raw = f.read(t - s)
    dt = e["dtype"]
    if dt == "BF16":
        u = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
        x = u.view(np.float32)
    elif dt in ("F16", "FP16"):
        x = np.frombuffer(raw, dtype="<f2").astype(np.float32)
    elif dt in ("F32", "FP32"):
        x = np.frombuffer(raw, dtype="<f4").astype(np.float32)
    else:
        raise ValueError(f"dtype {dt} unhandled")
    return x.reshape(e["shape"]).astype(np.float64)

def relL2(a, b):
    return float(np.linalg.norm((a - b).ravel()) / (np.linalg.norm(a.ravel()) + 1e-30))

def grouped(x_flat, group):
    """pad flattened array to a multiple of `group` and view as (-1, group)."""
    n = x_flat.size
    pad = (-n) % group
    if pad: x_flat = np.concatenate([x_flat, np.zeros(pad, x_flat.dtype)])
    return x_flat.reshape(-1, group), n

def okq4b(x):
    """baseline: per-32-block symmetric int4 + f16 block scale = 4 + 16/32 = 4.5 eff bits/w."""
    xf = x.ravel().astype(np.float64)
    g, n = grouped(xf.copy(), 32)
    s = (np.abs(g).max(1, keepdims=True) + 1e-30) / 7.0            # int4 symmetric qmax=7
    s = s.astype(np.float16).astype(np.float64)                    # f16 scale (as shipped)
    q = np.clip(np.round(g / s), -8, 7)
    r = (q * s).ravel()[:n]
    return relL2(xf[:n], r), 4 + 16.0 / 32.0

def t4(x, bits_a, bits_b, group):
    """Frobenius pi^k rank-2 codec (== frob_episode.encode math) at a chosen scale GROUP size.
    eff bits = bits_a + bits_b + (f16 scale bits per coord)/group ."""
    xf = x.ravel().astype(np.float64)
    g, n = grouped(xf.copy(), group)
    qma = (1 << (bits_a - 1)) - 1
    sa = (np.abs(g).max(1, keepdims=True) + 1e-30) / qma
    sa = sa.astype(np.float16).astype(np.float64)                  # store scale as f16
    a = np.clip(np.round(g / sa), -qma - 1, qma)
    recon = a * sa
    nscale = 1
    if bits_b:
        res = g - recon
        qmb = (1 << (bits_b - 1)) - 1
        sb = (np.abs(res).max(1, keepdims=True) + 1e-30) / qmb
        sb = sb.astype(np.float16).astype(np.float64)
        b = np.clip(np.round(res / sb), -qmb - 1, qmb)
        recon = recon + b * sb
        nscale = 2
    eff = bits_a + (bits_b or 0) + 16.0 * nscale / group
    return relL2(xf[:n], recon.ravel()[:n]), eff

def probe(path, names):
    hdr, base = st_header(path)
    keys = [k for k in hdr if k != "__metadata__"]
    if not names:
        print(f"{len(keys)} tensors. sample names:")
        for k in keys[:40]: print("  ", k, hdr[k]["dtype"], hdr[k]["shape"])
        return
    verdict_green_any = False
    for name in names:
        if name not in hdr:
            print(f"!! {name} not found"); continue
        x = read_tensor(path, hdr, base, name)
        b_rel, b_eff = okq4b(x)
        print(f"\n=== {name}  shape={list(x.shape)}  {x.size/1e6:.1f}M params ===")
        print(f"  OK_Q4B baseline (per-32blk int4+f16): relL2={b_rel:.4e}  eff={b_eff:.3f} b/w")
        rows = [("per-tensor", x.size), ("per-row", x.shape[-1]), ("per-32blk", 32)]
        for gname, group in rows:
            for ba, bb in [(4, 0), (4, 2), (4, 4), (8, 0)]:
                rel, eff = t4(x, ba, bb, group)
                cfg = f"a{ba}" + (f"b{bb}" if bb else "")
                # a WIN vs OK_Q4B = <= baseline relL2 at strictly fewer bits, or lower relL2 at <= 4.5
                win = (rel <= b_rel * 1.0001 and eff < b_eff - 1e-9) or (rel < b_rel and eff <= b_eff + 1e-9)
                if win: verdict_green_any = True
                tag = "  <-- WIN" if win else ""
                print(f"    T4 {gname:<10} {cfg:<5} relL2={rel:.4e}  eff={eff:.3f} b/w{tag}")
    print("\n==== G-T4-WEIGHTS:", "GREEN (a T4 config beats OK_Q4B fidelity-per-bit)" if verdict_green_any
          else "HONEST-NEGATIVE (no T4 config beats OK_Q4B; Frobenius free-scale does not transfer to weights)", "====")

if __name__ == "__main__":
    path = sys.argv[1]
    names = sys.argv[2:] if len(sys.argv) > 2 else None
    probe(path, names)
