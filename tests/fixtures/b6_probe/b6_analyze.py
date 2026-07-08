#!/usr/bin/env python3
# b6_analyze.py -- merge per-type / per-range b6_*.json probe outputs, compute the
# adjacent-layer similarity summary + norm-ratio profile, and emit G-B6-LAYERSIM.log.
import json, glob, os, statistics as st

D = os.path.dirname(os.path.abspath(__file__))
L = 48
GLOBAL = lambda l: (l % 6 == 5)
ORDER = [("q","self_attn.q_proj"),("k","self_attn.k_proj"),("v","self_attn.v_proj"),
         ("o","self_attn.o_proj"),("gate","mlp.gate_proj"),("up","mlp.up_proj"),
         ("down","mlp.down_proj")]

def load_type(short):
    """merge full or ranged files for a tensor type into full-length arrays."""
    cos = [None]*(L-1); fro=[None]*L; shp=[None]*L; rowm=[None]*(L-1); gser=None
    files = sorted(glob.glob(os.path.join(D, f"b6_{short}.json")) +
                   glob.glob(os.path.join(D, f"b6_{short}_*.json")))
    for fp in files:
        d = json.load(open(fp))
        for i,v in enumerate(d["adjacent_cos"]):
            if v is not None: cos[i]=v
        for i,v in enumerate(d["fro"]):
            if v is not None: fro[i]=v
        for i,v in enumerate(d["shapes"]):
            if v is not None: shp[i]=v
        if d.get("down_rowmean_cos"):
            for i,v in enumerate(d["down_rowmean_cos"]):
                if v is not None: rowm[i]=v
        if d.get("global_series"): gser=d["global_series"]
    return dict(cos=cos, fro=fro, shp=shp, rowm=rowm, gser=gser)

def fmt(x, p=4):
    return "null" if x is None else f"{x:.{p}f}"

def stats(vals):
    v=[x for x in vals if x is not None]
    if not v: return None
    return dict(n=len(v), min=min(v), med=st.median(v), max=max(v), mean=sum(v)/len(v))

def midband_mean(cos, lo=12, hi=36):
    v=[cos[l] for l in range(lo,hi+1) if l<len(cos) and cos[l] is not None]
    return (sum(v)/len(v)) if v else None

data = {s: load_type(s) for s,_ in ORDER}

lines=[]
def P(*a):
    s=" ".join(str(x) for x in a); lines.append(s); print(s)

P("="*88)
P("G-B6-LAYERSIM -- adjacent-layer weight-similarity profile for gemma-4-12b (48 decoder layers)")
P("Item B6 (adoption campaign): go/no-go for a looped / weight-aliased variant.")
P("Weights source: D:/Files/Models/Gemma4/gemma-4-12b-bucket/model.safetensors (bf16, single 23.9GB file)")
P("Method: offline safetensors header-parse + bf16->f32 (reuses tools/t4_weights_probe.py reader);")
P("        flat cosine of corresponding tensors of ADJACENT layers (L vs L+1); float64 chunked accumulation.")
P("Repro (per tensor type; MLP types chunked by layer-range to fit the shell cap, then merged):")
P("  python b6_layersim_probe.py <model.safetensors> {q,k,v,o} .            # attention (full, incl global series)")
P("  python b6_layersim_probe.py <model.safetensors> down . 0 20 ; ... 19 39 ; ... 38 48")
P("  python b6_layersim_probe.py <model.safetensors> gate . 0 25 ; ... 24 48   (same for up)")
P("  python b6_analyze.py     # merge + summary + this log")
P("")
P("GEOMETRY (from header): L%6==5 (L=5,11,17,23,29,35,41,47) = 8 V-less GLOBAL-attn layers with a")
P("  DIFFERENT attention shape (q[8192,3840] k[512,3840] o[3840,8192], NO v_proj); other 40 = SWA")
P("  (q[4096,3840] k/v[2048,3840] o[3840,4096]). MLP (gate/up/down) is UNIFORM [.,3840]/[3840,.] on ALL 48.")
P("  => attention cosine only defined between SAME-shape (SWA-SWA) adjacent layers; MLP defined for all 47 pairs.")
P("="*88)

# ---- per-pair arrays ----
for s,name in ORDER:
    cos=data[s]["cos"]
    P(f"\n--- {name}.weight : adjacent flat cosine (pair l = layers l..l+1; 'null' = shape gap at a global) ---")
    row=" ".join(f"{l:02d}:{fmt(cos[l],3)}" for l in range(L-1))
    # wrap
    toks=row.split(" ")
    for i in range(0,len(toks),12):
        P("  "+" ".join(toks[i:i+12]))

# ---- down_proj per-output-row mean cosine (robustness) ----
P("\n--- mlp.down_proj.weight : mean PER-OUTPUT-ROW cosine (robustness vs scale-dominated flatten) ---")
rm=data["down"]["rowm"]
toks=[f"{l:02d}:{fmt(rm[l],3)}" for l in range(L-1)]
for i in range(0,len(toks),12):
    P("  "+" ".join(toks[i:i+12]))

# ---- global attention period-6 series ----
P("\n--- GLOBAL-attn consecutive (period-6) cosine [globals can only alias among themselves] ---")
for s in ("q","k","o"):
    g=data[s]["gser"]
    if g: P(f"  {s}: "+"  ".join(f"L{a}-L{b}:{fmt(c,3)}" for a,b,c in g))

# ---- summary table ----
P("\n"+"="*88)
P("SUMMARY  (adjacent flat cosine over VALID pairs; attention valid = SWA-SWA only)")
P(f"{'tensor':<20}{'n':>4}{'min':>9}{'median':>9}{'max':>9}{'mid[12..36]':>13}{'L0-L1':>9}{'last':>10}")
for s,name in ORDER:
    cos=data[s]["cos"]; stt=stats(cos)
    if not stt: continue
    mb=midband_mean(cos)
    first=cos[0]
    # last valid pair
    lastv=next((cos[l] for l in range(L-2,-1,-1) if cos[l] is not None), None)
    P(f"{name:<20}{stt['n']:>4}{stt['min']:>9.4f}{stt['med']:>9.4f}{stt['max']:>9.4f}"
      f"{(mb if mb else float('nan')):>13.4f}{(first if first else float('nan')):>9.4f}{(lastv if lastv else float('nan')):>10.4f}")

# down per-row summary
rms=stats(data["down"]["rowm"])
if rms:
    P(f"{'down(per-row-mean)':<20}{rms['n']:>4}{rms['min']:>9.4f}{rms['med']:>9.4f}{rms['max']:>9.4f}"
      f"{(midband_mean(data['down']['rowm']) or float('nan')):>13.4f}"
      f"{(data['down']['rowm'][0] or float('nan')):>9.4f}"
      f"{(next((data['down']['rowm'][l] for l in range(L-2,-1,-1) if data['down']['rowm'][l] is not None),float('nan'))):>10.4f}")

# ---- Frobenius-norm ratio profile ||W_{l+1}||/||W_l|| ----
P("\n"+"="*88)
P("FROBENIUS-NORM RATIO  ||W_{l+1}||/||W_l||  across depth (ratio only where same shape)")
for s,name in ORDER:
    fro=data[s]["fro"]; shp=data[s]["shp"]
    ratios=[]
    for l in range(L-1):
        if fro[l] and fro[l+1] and shp[l]==shp[l+1]:
            ratios.append(fro[l+1]/fro[l])
        else:
            ratios.append(None)
    rr=[r for r in ratios if r is not None]
    if rr:
        P(f"  {name:<20} n={len(rr):>2}  min={min(rr):.3f} median={st.median(rr):.3f} max={max(rr):.3f}"
          f"  |  Fro(L0)={fro[0]:.1f} Fro(L47)={fro[47] if fro[47] else float('nan'):.1f}")

# also raw fro per layer for down (bulk) to see depth trend
P("\n  mlp.down_proj Frobenius norm per layer (depth trend):")
frod=data["down"]["fro"]
toks=[f"{l:02d}:{frod[l]:.0f}" if frod[l] else f"{l:02d}:--" for l in range(L)]
for i in range(0,len(toks),12):
    P("  "+" ".join(toks[i:i+12]))

with open(os.path.join(D,"G-B6-LAYERSIM.log"),"w") as f:
    f.write("\n".join(lines)+"\n")
print("\n[WROTE]", os.path.join(D,"G-B6-LAYERSIM.log"))
