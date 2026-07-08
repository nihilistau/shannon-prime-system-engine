#!/usr/bin/env python3
# b6_layersim_probe.py -- B6 looped/recursive-variant go/no-go probe (G-B6-LAYERSIM).
#
# Question (adoption campaign item B6): are adjacent decoder layers of the served
# gemma-4-12B already weight-similar enough that a looped / weight-aliased variant is
# CHEAP (little training to converge), or are the layers largely distinct (=> heavy retrain)?
#
# Reference: arXiv 2502.17416 reports looped models reach >=0.98 adjacent-block cosine AFTER
# their regularizer; a stock (non-looped) model is lower. We measure the stock model's
# adjacent-layer (L vs L+1) weight cosine and look for a contiguous MIDDLE band that is
# notably higher (candidate for "middle-looping": loop the middle, keep first/last unique).
#
# GEOMETRY (confirmed from the header): 48 decoder layers; L%6==5 (L=5,11,17,23,29,35,41,47)
# are V-less GLOBAL-attn layers with a DIFFERENT attention shape (q [8192,3840], k [512,3840],
# o [3840,8192], no v_proj); the other 40 are SWA (q [4096,3840], k/v [2048,3840], o [3840,4096]).
# The MLP (gate/up/down) is UNIFORM across all 48 layers. => attention cosine is only defined
# between same-shape layers; MLP cosine is defined for all 47 adjacent pairs.
#
# Offline: reuses the safetensors header-parse + bf16->f32 reader approach of
# tools/t4_weights_probe.py. Robust design: ONE tensor type per invocation (fresh process, no
# accumulation), chunked float64 dot/norm accumulation (low memory), writes its own JSON file.
# Read-only; does not touch the model.
import sys, json, struct, numpy as np

def st_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return hdr, 8 + n

def read_tensor_f32(path, hdr, base, name):
    """Read one tensor as float32 (bf16->f32 exact widen). Half the RAM of float64."""
    e = hdr[name]; s, t = e["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + s); raw = f.read(t - s)
    dt = e["dtype"]
    if dt == "BF16":
        u = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16)
        x = u.view(np.float32)
    elif dt in ("F16", "FP16"):
        x = np.frombuffer(raw, dtype="<f2").astype(np.float32)
    elif dt in ("F32", "FP32"):
        x = np.frombuffer(raw, dtype="<f4").astype(np.float32)
    else:
        raise ValueError(f"dtype {dt} unhandled")
    return x.reshape(e["shape"])   # float32

def dot_norm_chunked(a, b):
    """float64 accumulation of <a,b>, ||a||^2, ||b||^2 over 2D arrays, row-block chunked."""
    n = a.shape[0]; step = max(1, 2_000_000 // max(1, a.shape[1]))  # ~2M elems/chunk
    dot = na = nb = 0.0
    for i in range(0, n, step):
        aa = a[i:i+step].astype(np.float64); bb = b[i:i+step].astype(np.float64)
        dot += float(np.sum(aa * bb)); na += float(np.sum(aa * aa)); nb += float(np.sum(bb * bb))
    return dot, na, nb

def flat_cos(a, b):
    if a.shape != b.shape:
        return None
    d, na, nb = dot_norm_chunked(a, b)
    return d / ((na ** 0.5) * (nb ** 0.5) + 1e-30)

def fro(a):
    s = 0.0
    for i in range(0, a.shape[0], 4096):
        aa = a[i:i+4096].astype(np.float64); s += float(np.sum(aa * aa))
    return s ** 0.5

def rowmean_cos(a, b):
    """mean over output rows (dim 0) of per-row cosine; chunked float64."""
    n = a.shape[0]; step = max(1, 2_000_000 // max(1, a.shape[1]))
    tot = 0.0
    for i in range(0, n, step):
        aa = a[i:i+step].astype(np.float64); bb = b[i:i+step].astype(np.float64)
        num = np.einsum("ij,ij->i", aa, bb)
        da = np.sqrt(np.einsum("ij,ij->i", aa, aa)); db = np.sqrt(np.einsum("ij,ij->i", bb, bb))
        tot += float(np.sum(num / (da * db + 1e-30)))
    return tot / n

PREFIX = "model.language_model.layers"
TTYPES = {
    "q": "self_attn.q_proj.weight", "k": "self_attn.k_proj.weight",
    "v": "self_attn.v_proj.weight", "o": "self_attn.o_proj.weight",
    "gate": "mlp.gate_proj.weight", "up": "mlp.up_proj.weight", "down": "mlp.down_proj.weight",
}
GLOBAL = lambda l: (l % 6 == 5)

def run_type(path, short, outdir, L=48, lo=0, hi=None):
    """Process layers [lo, hi). Adjacent pairs (l,l+1) computed for l in [lo, hi-1].
    Call with overlapping ranges (share one layer) and merge to cover all 47 pairs."""
    if hi is None: hi = L
    tt = TTYPES[short]
    hdr, base = st_header(path)
    cos = [None] * (L - 1)
    fron = [None] * L
    shp = [None] * L
    rowm = [None] * (L - 1) if short == "down" else None
    gcache = {}
    keep_glob = short in ("q", "k", "o")
    prev = None
    for l in range(lo, hi):
        name = f"{PREFIX}.{l}.{tt}"
        cur = read_tensor_f32(path, hdr, base, name) if name in hdr else None
        if cur is not None:
            fron[l] = fro(cur); shp[l] = list(cur.shape)
        if l > lo and prev is not None and cur is not None:
            cos[l - 1] = flat_cos(prev, cur)
            if short == "down":
                rowm[l - 1] = rowmean_cos(prev, cur)
        if keep_glob and GLOBAL(l) and cur is not None:
            gcache[l] = cur
        prev = cur
        sys.stderr.write(f"  {short} L{l} done\n"); sys.stderr.flush()
    gser = None
    if keep_glob and lo == 0 and hi == L:
        gl = sorted(gcache)
        gser = [[a, b, flat_cos(gcache[a], gcache[b])] for a, b in zip(gl[:-1], gl[1:])]
    out = dict(short=short, tt=tt, nlayers=L, lo=lo, hi=hi, adjacent_cos=cos, fro=fron,
               shapes=shp, down_rowmean_cos=rowm, global_series=gser)
    suffix = "" if (lo == 0 and hi == L) else f"_{lo}_{hi}"
    fp = f"{outdir}/b6_{short}{suffix}.json"
    with open(fp, "w") as f:
        json.dump(out, f)
    sys.stderr.write(f"[WROTE] {fp}\n")

if __name__ == "__main__":
    path = sys.argv[1]; short = sys.argv[2]; outdir = sys.argv[3]
    lo = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    hi = int(sys.argv[5]) if len(sys.argv) > 5 else 48
    run_type(path, short, outdir, 48, lo, hi)
