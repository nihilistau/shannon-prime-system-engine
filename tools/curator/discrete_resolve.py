#!/usr/bin/env python3
# C2 Step 2 (Shannon-Prime discrete form) — the bit-collision resolver.
#
# The cue is NOT a soft margin. The episode signature is a 256-bit LSH hash; the match is
# XOR + popcount; the gate is an INTEGER Hamming radius. There is no float in the address
# space and no floating-point reduction-order ambiguity, so the resolver's verdict is
# bit-reproducible and hardware-independent (unlike a float-cosine threshold near tau, which
# can flip across reduction orders — the non-associativity this project has been bitten by).
#
# resolve_cue(cue_bits) returns the episode_id iff bit-agreement >= TAU_BITS, else NULL.
#
# WHY r=256 (receipt: tools/curator -> G-MEMO-CUE_discrete.log r-sweep): sign-binarizing the
# r=32 router signature COLLAPSES separation (bit-gap -1; magnitude carried the thin ep_wiki
# margin). The discrete form needs >=128 hash bits to recover the angular margin; r=256 gives a
# comfortable +19-bit gap. Binarization is strictly weaker than the real dot at equal width — we
# spend hash width to buy an integer/bit-exact address space. That trade is the Shannon-Prime call.
import os, json, numpy as np
SEED = 0x5350524F4A2B
R_BITS = 256
HD = 512
NL, PERIOD = 48, 8
MASK64 = (1 << 64) - 1
TAU_BITS = 168            # in [159..177]: clears positives by >=9, rejects non-targets by >=10

def smix(seed, n):
    s = seed & MASK64; out = np.empty(n, dtype=np.int8)
    for i in range(n):
        s = (s + 0x9E3779B97F4A7C15) & MASK64; z = s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        z = z ^ (z >> 31); out[i] = 1 if (z & 1) else -1
    return out

def build_R(): return smix(SEED, R_BITS * HD).astype(np.float32).reshape(R_BITS, HD)
def gl(): return [L for L in range(NL) if (L % PERIOD) == PERIOD - 1]
def loadK(d):
    raw = np.fromfile(os.path.join(d, "ep.k"), dtype="<f4"); P = raw.size // (NL * HD)
    return raw.reshape(NL, P, HD), P
def projmean(K, R, pos): return np.stack([R @ K[L, p] for L in gl() for p in pos], 0).mean(0)
def to_bits(v): return (v > 0)
def packhex(b):
    x = 0
    for i in range(R_BITS):
        if b[i]: x |= (1 << i)
    return f"{x:0{R_BITS // 4}x}"
def unpack(h):
    x = int(h, 16); return np.array([(x >> i) & 1 for i in range(R_BITS)], dtype=bool)
def agree(a, b): return int(np.sum(a == b))   # R_BITS - HammingDistance

def resolve_cue(cue_bits, registry, tau=TAU_BITS):
    """argmax bit-agreement over registry; return (id,name,score) iff score>=tau else (None,None,score)."""
    best = (None, None, -1)
    for row in registry:
        s = agree(cue_bits, row["_bits"])
        if s > best[2]: best = (row["episode_id"], row["name"], s)
    return best if best[2] >= tau else (None, None, best[2])

def find_eng():
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

def main():
    R = build_R(); eng = find_eng()
    eps = {"ep_toy": (f"{eng}/_p33_ep", 16), "ep_wiki": (f"{eng}/_c2_ep_wiki", 84)}
    outdir = "/mnt/f/ring2/episodes"
    try:
        os.makedirs(outdir, exist_ok=True); open(os.path.join(outdir, ".wtest"), "w").close()
    except Exception:
        outdir = f"{eng}/tests/fixtures/xbar_c2"
    regpath = os.path.join(outdir, "registry_bits.jsonl")

    registry = []; cues = {}
    with open(regpath, "w") as rf:
        for eid, (name, (epdir, npos)) in enumerate(eps.items()):
            K, P = loadK(epdir); rp = list(range(min(npos, P))); h = len(rp) // 2
            sig_b = to_bits(projmean(K, R, rp[:h])); cue_b = to_bits(projmean(K, R, rp[h:]))
            cues[name] = cue_b
            row = {"episode_id": eid, "name": name, "ring2_path": epdir, "npos": len(rp),
                   "r_bits": R_BITS, "tau_bits": TAU_BITS, "sig_bits": packhex(sig_b)}
            rf.write(json.dumps(row) + "\n")
            row["_bits"] = sig_b; registry.append(row)
    print(f"[reg] registry_bits.jsonl -> {regpath}  ({len(registry)} eps)  R_BITS={R_BITS}  TAU_BITS={TAU_BITS}", flush=True)
    for row in registry:
        print(f"[reg]   id={row['episode_id']} {row['name']:8s} sig_bits={row['sig_bits'][:16]}...  ({len(row['sig_bits']) * 4} bits)", flush=True)

    ok = True
    print("\n[pos] held-out cue -> resolve (must hit OWN id, agreement >= tau):", flush=True)
    for row in registry:
        rid, rname, score = resolve_cue(cues[row["name"]], registry)
        hit = (rid == row["episode_id"]); ok &= hit
        print(f"  cue[{row['name']:8s}] -> {('NULL' if rid is None else rname):8s} agree={score}/{R_BITS} (tau={TAU_BITS})  [{'PASS' if hit else 'FAIL'}]", flush=True)

    print("\n[neg] unrelated query -> resolve (must be NULL):", flush=True)
    ksc = float(np.std(loadK(eps['ep_toy'][0])[0][gl()])); rng = np.random.default_rng(20260617)
    for j in range(8):
        n = rng.normal(0, ksc, size=(len(gl()) * 24, HD)).astype(np.float32)
        qn = to_bits((n @ R.T).mean(0)); rid, rname, score = resolve_cue(qn, registry)
        isnull = rid is None; ok &= isnull
        print(f"  neg[{j}] -> {('NULL' if rid is None else rname):8s} agree={score}/{R_BITS}  [{'PASS' if isnull else 'FAIL'}]", flush=True)

    print(f"\n[gate] G-MEMO-CUE(discrete r={R_BITS}) {'GREEN -- bit-collision gate separates signal from noise, rejects negatives' if ok else 'RED'}", flush=True)
    return 0 if ok else 1

if __name__ == "__main__":
    import sys; sys.exit(main())
