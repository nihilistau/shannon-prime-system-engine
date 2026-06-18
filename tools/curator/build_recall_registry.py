#!/usr/bin/env python3
# CONTRACT-CHAT-FULLSTACK B3 — build the AUTONOMOUS-RECALL episode registry.
#
# One JSONL row per episode: {name, dir, npos, topic, sig_bits}. The sig is
# computed the SAME way the daemon computes the live query sig (recall.rs
# Projection::signature): the SIGN of the ±1 LSH projection (SP_ARM_PROJ_SEED via
# splitmix64) of the per-position GLOBAL-owner K, MEANED OVER ALL real positions
# and the global layers (period-6 ⇒ {5,11,...,47}), packed to 256 bits with the
# SAME bit order as discrete_resolve.py packhex (bit i ⇒ 1<<i). Mean-over-all
# (not the disjoint sig/cue split of build_registry.py) so a query sig and the
# registry sig are maximally comparable: both summarise the WHOLE passage.
#
# "Real" positions = those whose global-owner K is non-zero (the PPL capture
# allocates Pmax slots but fills only n_ctx; the uninitialised tail must NOT
# enter the centroid — exactly the build_registry.py real_positions rule).
import os, json, sys
import numpy as np

SEED   = 0x5350524F4A2B
R_BITS = 256
HD     = 512
NL, PERIOD = 48, 6
MASK64 = (1 << 64) - 1

def smix(seed, n):
    s = seed & MASK64; out = np.empty(n, dtype=np.int8)
    for i in range(n):
        s = (s + 0x9E3779B97F4A7C15) & MASK64; z = s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        z = z ^ (z >> 31); out[i] = 1 if (z & 1) else -1
    return out

def build_R():
    return smix(SEED, R_BITS * HD).astype(np.float32).reshape(R_BITS, HD)

def gl():
    return [L for L in range(NL) if (L % PERIOD) == PERIOD - 1]

def loadK(d):
    raw = np.fromfile(os.path.join(d, "ep.k"), dtype="<f4")
    P = raw.size // (NL * HD)               # floor: the capture allocates Pmax slots
    raw = raw[: NL * P * HD]                # drop any ragged tail bytes
    return raw.reshape(NL, P, HD), P

def real_positions(K):
    g = gl()
    norms = np.linalg.norm(K[g], axis=2).sum(0)   # [P]
    return [p for p in range(K.shape[1]) if norms[p] > 1e-6]

def packhex(b):
    x = 0
    for i in range(R_BITS):
        if b[i]: x |= (1 << i)
    return f"{x:0{R_BITS // 4}x}"

def sig_bits(epdir, R, npos):
    # Use ONLY the prompt positions [0,npos) — the PPL capture allocates Pmax
    # slots and the tail past the prompt holds either uninitialised VRAM or free-
    # decoded continuation, neither of which the daemon's prompt-only query sig
    # sees. This matches recall.rs Projection::signature over the prefilled prompt.
    K, P = loadK(epdir)
    npos = min(npos, P)
    rp = list(range(npos))
    rows = [R @ K[L, p] for L in gl() for p in rp]
    v = np.stack(rows, 0).mean(0)
    return (v > 0), npos

def main():
    # engine root = .../shannon-prime-system-engine ; this file is at tools/curator/.
    eng = os.environ.get("SP_ENGINE_DIR") or os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    R = build_R()
    # episode topic + dir; npos is the detected real-position count.
    # npos = the prompt token count for each capture (the real prefilled prefix).
    episodes = [
        ("ep_wiki",    os.path.join(eng, "_c2_ep_wiki"),     84,  "Robert Boulter, English film/TV/theatre actor (The Bill)"),
        ("ep_homarus", os.path.join(eng, "_b3_ep_homarus"),  180, "Homarus gammarus, the European/common lobster"),
        ("ep_headlam", os.path.join(eng, "_b3_ep_headlam"),  179, "Frank Headlam, RAAF Air Vice Marshal"),
    ]
    out = os.path.join(eng, "tests", "fixtures", "chat_fullstack", "recall_registry.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = []
    with open(out, "w") as f:
        for name, d, npos_req, topic in episodes:
            if not os.path.exists(os.path.join(d, "ep.k")):
                print(f"[reg] SKIP {name}: no ep.k at {d}", flush=True); continue
            b, npos = sig_bits(d, R, npos_req)
            row = {"name": name, "dir": d.replace("/", "\\"), "npos": npos,
                   "topic": topic, "sig_bits": packhex(b)}
            f.write(json.dumps(row) + "\n"); rows.append((name, b, npos))
            print(f"[reg] {name}: npos={npos} sig={row['sig_bits'][:16]}...", flush=True)
    print(f"[reg] wrote {len(rows)} rows -> {out}", flush=True)
    # separation sanity: pairwise bit-agreement (higher diag = self-distinct).
    print("\n[sep] pairwise bit-agreement (256-bit; diag is trivially 256):", flush=True)
    names = [r[0] for r in rows]; bits = {r[0]: r[1] for r in rows}
    hdr = "          " + " ".join(f"{n[:9]:>9s}" for n in names)
    print(hdr, flush=True)
    for a in names:
        line = f"  {a:9s}" + " ".join(f"{int(np.sum(bits[a]==bits[b])):9d}" for b in names)
        print(line, flush=True)

if __name__ == "__main__":
    main()
