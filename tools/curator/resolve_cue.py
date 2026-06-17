#!/usr/bin/env python3
# C2 Build Step 2 — the offline resolver + tau_cue thresholding.
#
# resolve_cue(q_sig) scans registry.jsonl and returns the episode_id ONLY if the
# best dot product (cosine, sigs are unit) exceeds TAU_CUE; otherwise NULL. The
# curator must DECISIVELY REJECT cues with no relevant memory — an over-eager
# inject of an irrelevant episode destroys the 12B's PPL (the P3.4 deflection bound).
#
# Gates:
#   POSITIVE — each episode's held-out (disjoint) cue must resolve to its OWN id, score > tau.
#   NEGATIVE — a fresh, unrelated query sig (gaussian @ K-scale, seed disjoint from the
#              registry's background) must resolve to NULL (no forced match).
#
# tau_cue is set from the Step-1 offline margins: targets self-scored +0.3730 (ep_wiki)
# / +0.6863 (ep_toy); background noise floor peaked ~ +0.2162. TAU_CUE=0.30 strictly
# separates both targets from that floor (target headroom >=0.07; noise rejected by ~0.084).
import os, json, hashlib, sys
import numpy as np

SEED = 0x5350524F4A2B
R_DIM = 32
HD = 512
NL, PERIOD = 48, 8
MASK64 = (1 << 64) - 1
TAU_CUE = 0.30                      # pre-registered from Step-1 margins

def splitmix64_stream(seed, n):
    s = seed & MASK64; out = np.empty(n, dtype=np.int8)
    for i in range(n):
        s = (s + 0x9E3779B97F4A7C15) & MASK64
        z = s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        z = z ^ (z >> 31)
        out[i] = 1 if (z & 1) else -1
    return out

def build_R(r, hd):
    return splitmix64_stream(SEED, r * hd).astype(np.float32).reshape(r, hd)

def global_layers():
    return [L for L in range(NL) if (L % PERIOD) == PERIOD - 1]

def unit(v):
    n = np.linalg.norm(v); return v / n if n > 1e-12 else v

def load_episode_K(epdir):
    raw = np.fromfile(os.path.join(epdir, "ep.k"), dtype="<f4")
    P = raw.size // (NL * HD)
    return raw.reshape(NL, P, HD), P

def projected_global_keys(K, R, positions):
    gl = global_layers(); rows = []
    for L in gl:
        for p in positions:
            rows.append(R @ K[L, p])
    return np.stack(rows, 0)

# ── the registry index, loaded once ──
def load_registry(regpath):
    idx = []
    with open(regpath) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            row = json.loads(line)
            idx.append((row["episode_id"], row["name"], unit(np.asarray(row["sig"], np.float32))))
    return idx

# ── THE RESOLVER ──
def resolve_cue(q_sig, idx, tau=TAU_CUE):
    """Return (episode_id, name, score) of argmax over the registry IFF score > tau, else (None, None, score)."""
    q = unit(q_sig)
    best_id, best_name, best_score = None, None, -1.0
    for eid, name, sig in idx:
        s = float(np.dot(q, sig))
        if s > best_score:
            best_id, best_name, best_score = eid, name, s
    if best_score > tau:
        return best_id, best_name, best_score
    return None, None, best_score          # NULL — no relevant memory above threshold

def find_eng():
    # this file lives at <eng>/tools/curator/resolve_cue.py
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", ".."))

def main():
    R = build_R(R_DIM, HD)
    eng = find_eng()
    for regpath in ("/mnt/f/ring2/episodes/registry.jsonl",
                    f"{eng}/_c2_registry/registry.jsonl",
                    f"{eng}/tests/fixtures/xbar_c2/registry.jsonl"):
        if os.path.exists(regpath):
            break
    idx = load_registry(regpath)
    print(f"[res] registry: {regpath}  ({len(idx)} episodes)  TAU_CUE={TAU_CUE}", flush=True)
    for eid, name, sig in idx:
        print(f"[res]   id={eid} {name:8s} |sig|={np.linalg.norm(sig):.3f}", flush=True)

    episodes = {"ep_toy": (f"{eng}/_p33_ep", 16), "ep_wiki": (f"{eng}/_c2_ep_wiki", 84)}

    ok = True
    # ── POSITIVE gate: held-out disjoint cue (2nd half of true npos) must resolve to its own id ──
    print("\n[pos] held-out cue -> resolve (must hit OWN id, score > tau):", flush=True)
    for eid, name, _ in idx:
        if name not in episodes: continue
        epdir, npos = episodes[name]
        K, P = load_episode_K(epdir)
        rp = list(range(min(npos, P))); half = len(rp) // 2
        cue = projected_global_keys(K, R, rp[half:]).mean(0)      # disjoint from sig (1st half)
        rid, rname, score = resolve_cue(cue, idx)
        hit = (rid == eid)
        tag = "PASS" if hit else "FAIL"
        if not hit: ok = False
        print(f"  cue[{name:8s}] -> {('NULL' if rid is None else rname):8s} score={score:+.4f} (tau={TAU_CUE})  [{tag}]", flush=True)

    # ── NEGATIVE control: unrelated query sigs (fresh seed, NOT the registry background) -> NULL ──
    print("\n[neg] unrelated query -> resolve (must be NULL, no forced match):", flush=True)
    kscale = float(np.std(load_episode_K(episodes["ep_toy"][0])[0][global_layers()]))
    rng = np.random.default_rng(20260617)                        # disjoint from registry noise seed(0)
    neg_pass = True
    for j in range(8):
        noise = rng.normal(0, kscale, size=(len(global_layers()) * 24, HD)).astype(np.float32)
        qn = (noise @ R.T).mean(0)
        rid, rname, score = resolve_cue(qn, idx)
        isnull = rid is None
        tag = "PASS" if isnull else "FAIL"
        if not isnull: neg_pass = False; ok = False
        print(f"  neg[{j}] -> {('NULL' if rid is None else rname):8s} score={score:+.4f}  [{tag}]", flush=True)

    print(f"\n[gate] G-MEMO-CUE(resolver) {'GREEN — positives resolve, negatives reject' if ok else 'RED'}", flush=True)
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
