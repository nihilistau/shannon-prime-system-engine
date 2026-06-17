#!/usr/bin/env python3
# C2 Build Step 1 — the episode registry + centroid-signature writer.
# Faithful to the engine: R = ±1 from splitmix64(SP_ARM_PROJ_SEED), r=32, hd_max=512 (the frozen router R,
# sp_arm_build_R in core/arm/arm.c). sig[r] = mean over GLOBAL-owner projected keys (sp_arm_project: proj[p]
# = Σ_d R[p*hd_max+d]*K[d]). Writes one registry.jsonl row per episode on the Optane Ring-2 tier, then a
# separation sanity check: a held-out cue from each episode must score its OWN sig above every other
# episode + synthetic-noise background.
import os, json, struct, hashlib, time, sys
import numpy as np

SEED = 0x5350524F4A2B
R_DIM = 32
HD = 512            # 12B global head_dim == kvd (nkv=1)
NL, PERIOD = 48, 8  # gemma4-12b geometry (all owners; global = L%PERIOD==PERIOD-1)
MASK64 = (1 << 64) - 1

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
    return splitmix64_stream(SEED, r * hd).astype(np.float32).reshape(r, hd)   # row-major, matches engine

def load_episode_K(epdir):
    kpath = os.path.join(epdir, "ep.k")
    raw = np.fromfile(kpath, dtype="<f4")
    P = raw.size // (NL * HD)
    assert P * NL * HD == raw.size, f"{epdir}: size {raw.size} not NL*P*HD"
    K = raw.reshape(NL, P, HD)                                   # [layer, pos, kvd]
    sha = hashlib.sha256(raw.tobytes()).hexdigest()[:16]
    return K, P, sha

def global_layers():
    return [L for L in range(NL) if (L % PERIOD) == PERIOD - 1]   # 7,15,23,31,39,47

def projected_global_keys(K, R, positions):
    # for each global owner layer, project K[L][pos] (hd=512) -> r dims; stack over (layer,pos)
    gl = global_layers(); rows = []
    for L in gl:
        for p in positions:
            rows.append(R @ K[L, p])                              # [r]
    return np.stack(rows, 0) if rows else np.zeros((0, R_DIM), np.float32)

def real_positions(K):
    # a position is "real" (prefilled) if any global-owner K row is non-zero
    gl = global_layers()
    norms = np.linalg.norm(K[gl], axis=2).sum(0)                  # [P]
    return [p for p in range(K.shape[1]) if norms[p] > 1e-6]

def unit(v):
    n = np.linalg.norm(v); return v / n if n > 1e-12 else v

def main():
    R = build_R(R_DIM, HD)
    print(f"[reg] R built: {R.shape} ±1 from seed {hex(SEED)} (sum={int(R.sum())}, first row[:8]={R[0,:8].astype(int).tolist()})", flush=True)

    eng = "/mnt/d/F/shannon-prime-repos/shannon-prime-system-engine"
    # npos = the episode's TRUE filled length (WRITE dumps the whole P-slot allocation; the unfilled
    # tail is uninitialized VRAM and must NOT enter the centroid). toy = 4 prompt + 12 gen; wiki = n_ctx.
    episodes = [("ep_toy",  f"{eng}/_p33_ep",     16),
                ("ep_wiki", f"{eng}/_c2_ep_wiki", 84)]
    outdir = "/mnt/f/ring2/episodes"
    try:
        os.makedirs(outdir, exist_ok=True); open(os.path.join(outdir, ".wtest"), "w").close()
    except Exception as e:
        outdir = f"{eng}/_c2_registry"; os.makedirs(outdir, exist_ok=True)
        print(f"[reg] Optane /mnt/f not writable ({type(e).__name__}); registry -> {outdir}", flush=True)
    regpath = os.path.join(outdir, "registry.jsonl")

    rows, sigs, cues = [], {}, {}
    with open(regpath, "w") as rf:
        for eid, (name, epdir, npos) in enumerate(episodes):
            if not os.path.exists(os.path.join(epdir, "ep.k")):
                print(f"[reg] SKIP {name}: no ep.k at {epdir}", flush=True); continue
            K, P, sha = load_episode_K(epdir)
            rp = list(range(min(npos, P)))   # TRUE filled prefix only (unfilled tail = uninitialized VRAM)
            half = len(rp) // 2
            sig_pos, cue_pos = rp[:half], rp[half:]               # disjoint: sig vs cue (no trivial self-id)
            sig = projected_global_keys(K, R, sig_pos).mean(0)
            cue = projected_global_keys(K, R, cue_pos).mean(0)
            sigs[name] = unit(sig); cues[name] = unit(cue)
            row = {"episode_id": eid, "name": name, "ring2_path": epdir, "P": P, "npos": len(rp),
                   "sig": [round(float(x), 6) for x in sig.tolist()], "sig_norm": float(np.linalg.norm(sig)),
                   "created_tick": int(time.time()), "recall_count": 0, "sha16": sha}
            rf.write(json.dumps(row) + "\n"); rows.append(row)
            print(f"[reg] {name}: P={P} real_pos={len(rp)} (sig from {len(sig_pos)}, cue from {len(cue_pos)}) sha={sha}", flush=True)
    print(f"[reg] registry.jsonl -> {regpath}  ({len(rows)} episodes)", flush=True)

    # ── synthetic background noise episodes: gaussian K at the real-K scale ──
    rng = np.random.default_rng(0)
    kscale = 1.0
    if rows:
        Kt, _, _ = load_episode_K(episodes[0][1]); kscale = float(np.std(Kt[global_layers()]))
    for j in range(6):
        noise = rng.normal(0, kscale, size=(len(global_layers()) * 24, HD)).astype(np.float32)
        sigs[f"noise{j}"] = unit((noise @ R.T).mean(0))

    # ── separation sanity: each real episode's HELD-OUT cue vs all sigs ──
    print("\n[sep] cosine(cue_e, sig_f) — target should be the row-max:", flush=True)
    real_names = [e[0] for e in episodes if e[0] in cues]
    cols = real_names + [f"noise{j}" for j in range(6)]
    ok = True
    for ce in real_names:
        scores = {f: float(np.dot(cues[ce], sigs[f])) for f in cols}
        best = max(scores, key=scores.get)
        margin = scores[ce] - max(v for f, v in scores.items() if f != ce)
        tag = "PASS" if best == ce else "FAIL"
        if best != ce: ok = False
        bg = max(v for f, v in scores.items() if f not in real_names)
        print(f"  cue[{ce:8s}] self={scores[ce]:+.4f}  best={best}({scores[best]:+.4f})  "
              f"max_bg_noise={bg:+.4f}  margin={margin:+.4f}  [{tag}]", flush=True)
    print(f"\n[sep] G-MEMO-CUE(offline) {'GREEN — every target separates from background' if ok else 'RED'}", flush=True)

if __name__ == "__main__":
    main()
