#!/usr/bin/env python3
"""B3-v3 dataset builder — mine (query global-Q, episode global-K, label) pairs from
the substrate. NO external corpus: episode-K comes from each episode's ep.k; query-Q
comes from the daemon's SP_B3_QDUMP seam (POST each query, the daemon writes the exact
last-token global-Q that qk_relevance scores).

Granularity matches the runtime selector recall.rs::qk_relevance:
  query  Q_i : [ng, G_NH, HD]   (global layers x query heads x head_dim)
  episode K_e: [ng, npos_e, HD] (global layers x prompt positions x head_dim)
score is reduced over (layer, head, pos). W_c projects each HD-vector to r dims.

Outputs b3_data.npz: Q (list, ragged ok -> object array), labels, and per-episode K.

Two modes:
  --live   : start from a running daemon (SP_B3_QDUMP set), POST the query set, collect.
  --offline: just assemble from an existing dump dir of q_<...>.bin + the registry.
"""
import os, sys, json, glob, struct, argparse
import numpy as np

NL, PERIOD, HD, G_NH = 48, 6, 512, 16
GLOBALS = [l for l in range(NL) if l % PERIOD == PERIOD - 1]   # {5,11,...,47}
NG = len(GLOBALS)

# ---- default query set: paraphrases/sub-questions per episode (positives) + foreign.
# Edit/extend freely; more paraphrases per episode = a better-conditioned W_c. Keep the
# label = the episode `name` it should retrieve, or "__foreign__" for the reject class.
DEFAULT_QUERIES = {
    "ep_wiki": [
        "Who is Robert Boulter?", "What is Robert Boulter known for?",
        "Tell me about the actor Robert Boulter.", "What role did Robert Boulter play on The Bill?",
        "Which play did Robert Boulter star in at the Royal Court?",
        "Is Robert Boulter a film and television actor?",
    ],
    "ep_homarus": [
        "What is Homarus gammarus?", "Tell me about the European lobster.",
        "What is the common lobster?", "Describe the European lobster's habitat.",
        "What does Homarus gammarus look like?", "Is the European lobster edible?",
    ],
    "ep_headlam": [
        "Who was Frank Headlam?", "Tell me about the RAAF officer Frank Headlam.",
        "What was Frank Headlam's rank in the Australian Air Force?",
        "What did Air Vice Marshal Headlam do?", "When did Frank Headlam serve?",
        "Was Frank Headlam an Australian commander?",
    ],
    "__foreign__": [
        # first 5 -> TRAIN split (the verifier tunes its margin on these)
        "What is the capital of France?", "How do I bake sourdough bread?",
        "Explain how a transformer neural network works.", "What is the boiling point of water?",
        "Who wrote Pride and Prejudice?",
        # remaining -> HELD-OUT split (never seen by W_c OR the verifier):
        "How do I change a flat tyre?", "What is the speed of light?", "Recommend a good pasta recipe.",
        # --- the 4 adversarial classes (operator, v4): stress the reject boundary ---
        # 1. Deceptive lexical overlap: keyword "boulter" but sourdough semantic class
        "What is the standard hydration ratio for a French sourdough boulter bread?",
        # 2. Orthogonal technical / out-of-domain (dense, high-entropy)
        "Explain the memory-bandwidth limits of the dp4a GEMV accumulate instruction.",
        # 3. Conversational drift / ambiguous (zero specific markers -> reject all)
        "Hey, can you help me remember what we were just talking about?",
        # 4. Adversarial entropy / stop-word soup (baseline thermodynamic stability)
        "the and of to a in is that it with as",
    ],
}


def load_episode_global_k(epdir, npos):
    """ep.k = raw <f4 [NL][P][HD]; extract [NG][npos][HD] over global layers. Mirrors
    recall.rs::load_episode_global_k / curator loadK."""
    raw = np.fromfile(os.path.join(epdir, "ep.k"), dtype="<f4")
    p_total = raw.size // (NL * HD)
    raw = raw[: NL * p_total * HD].reshape(NL, p_total, HD)
    npos = min(npos, p_total)
    return np.ascontiguousarray(raw[GLOBALS, :npos, :]).astype(np.float32), npos  # [NG,npos,HD]


def read_qdump(path):
    """q_<...>.bin = u32 n_global, u32 qd(=G_NH*HD), then n_global*qd f32. -> [ng,G_NH,HD]."""
    with open(path, "rb") as f:
        ng, qd = struct.unpack("<2I", f.read(8))
        q = np.frombuffer(f.read(ng * qd * 4), np.float32).copy()
    assert qd == G_NH * HD, f"qd {qd} != {G_NH*HD}"
    return q.reshape(ng, G_NH, HD)


def qpairs_from_manifest(manifest_path, foreign_path):
    """Build (query,label) from corpus_manifest.jsonl (per-needle query+paraphrases ->
    label ep_<id>) plus foreign_queries.txt (-> __foreign__). Replaces DEFAULT_QUERIES
    for the scaled novel-needle corpus."""
    pairs = []
    for ln in open(manifest_path, encoding="utf-8"):
        if not ln.strip():
            continue
        r = json.loads(ln)
        label = f"ep_{r['id']}"
        for q in [r["query"]] + list(r.get("paraphrases", [])):
            pairs.append((q, label))
    if foreign_path and os.path.exists(foreign_path):
        for ln in open(foreign_path, encoding="utf-8"):
            q = ln.strip()
            if q:
                pairs.append((q, "__foreign__"))
    return pairs


def load_registry(reg_path):
    eps = []
    for line in open(reg_path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        eps.append(r)
    return eps


def post_queries(base, qdir, qpairs):
    """Live mode: POST every query (auto_recall:true) so the daemon dumps its global-Q
    into qdir as q_<chat_id>.bin (the daemon must run with SP_B3_QDUMP=<qdir>). The dump
    filename is the chat_id, not the prompt, so we POST SEQUENTIALLY and pair each query
    with the file that newly appeared. Writes qmap.json {file_basename: [query, label]}
    and returns that map."""
    import urllib.request, time, glob as _glob
    os.makedirs(qdir, exist_ok=True)
    qmap = {}
    for q, label in qpairs:
        if True:
            before = set(os.path.basename(f) for f in _glob.glob(os.path.join(qdir, "q_*.bin")))
            body = json.dumps({"messages": [{"role": "user", "content": q}],
                               "max_tokens": 4, "temperature": 0, "auto_recall": True}).encode()
            req = urllib.request.Request(base + "/v1/chat", data=body,
                                         headers={"Content-Type": "application/json"})
            try:
                urllib.request.urlopen(req, timeout=120).read()
            except Exception as e:
                print(f"[ds] POST failed for {q!r}: {e}", flush=True); continue
            new = None
            for _ in range(20):  # poll up to ~2s for the dump to land
                after = set(os.path.basename(f) for f in _glob.glob(os.path.join(qdir, "q_*.bin")))
                diff = after - before
                if diff:
                    new = sorted(diff)[-1]; break
                time.sleep(0.1)
            if new is None:
                print(f"[ds] WARN no new dump for {q!r}", flush=True); continue
            qmap[new] = [q, label]
            print(f"[ds] {label:12} {new}  <- {q[:42]}", flush=True)
    with open(os.path.join(qdir, "qmap.json"), "w") as f:
        json.dump(qmap, f, indent=0)
    return qmap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eng", default=os.environ.get("SP_ENGINE_DIR",
                    r"D:\F\shannon-prime-repos\shannon-prime-system-engine"))
    ap.add_argument("--qdir", default=None, help="dir of q_*.bin dumps (SP_B3_QDUMP target)")
    ap.add_argument("--live", action="store_true", help="POST the query set to a running daemon first")
    ap.add_argument("--base", default="http://127.0.0.1:3000")
    ap.add_argument("--out", default=None)
    ap.add_argument("--registry", default=None, help="registry.jsonl (default: legacy recall_registry.jsonl)")
    ap.add_argument("--manifest", default=None, help="corpus_manifest.jsonl -> dynamic queries")
    ap.add_argument("--foreign", default=None, help="foreign_queries.txt (NULL class, one/line)")
    args = ap.parse_args()
    eng = args.eng
    qdir = args.qdir or os.path.join(eng, "_b3_wc", "qdump")
    out = args.out or os.path.join(eng, "_b3_wc", "b3_data.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    reg_path = args.registry or os.path.join(eng, "tests", "fixtures", "chat_fullstack", "recall_registry.jsonl")
    eps = load_registry(reg_path)
    name_to_idx = {e["name"]: i for i, e in enumerate(eps)}
    if args.manifest:
        qpairs = qpairs_from_manifest(args.manifest, args.foreign)
    else:
        qpairs = [(q, label) for label, qs in DEFAULT_QUERIES.items() for q in qs]
    K_list = []
    for e in eps:
        K, npos = load_episode_global_k(e["dir"], int(e["npos"]))
        K_list.append(K)
        print(f"[ds] episode {e['name']}: K {K.shape} (npos={npos})", flush=True)

    if args.live:
        npp = sum(1 for _,l in qpairs if l!="__foreign__"); nff = sum(1 for _,l in qpairs if l=="__foreign__")
        print(f"[ds] LIVE: POST {len(qpairs)} queries ({npp} positive / {nff} foreign) -> {qdir}", flush=True)
        qmap = post_queries(args.base, qdir, qpairs)
    else:
        # offline: the qmap.json sidecar (written by a prior --live run) maps
        # q_<chat_id>.bin -> [query, label].
        qm = os.path.join(qdir, "qmap.json")
        if not os.path.exists(qm):
            print(f"[ds] no qmap.json in {qdir}; run --live first.", flush=True); sys.exit(2)
        qmap = json.load(open(qm))

    Q, lab, txt = [], [], []
    for base_name, (qtext, label) in qmap.items():
        f = os.path.join(qdir, base_name)
        if not os.path.exists(f):
            continue
        Q.append(read_qdump(f))
        lab.append(name_to_idx.get(label, -1))     # -1 = foreign
        txt.append(qtext)
    print(f"[ds] collected {len(Q)} query-Q vectors "
          f"({sum(1 for l in lab if l>=0)} positive / {sum(1 for l in lab if l<0)} foreign)", flush=True)
    if not Q:
        print("[ds] NO query dumps found. Run --live with a daemon started with "
              "SP_B3_QDUMP=<qdir> (and the routes.rs patch applied + built).", flush=True)
        sys.exit(2)

    def objarr(lst):
        a = np.empty(len(lst), dtype=object)
        for i, x in enumerate(lst):
            a[i] = x
        return a
    np.savez(out,
             Q=objarr(Q),
             labels=np.array(lab, np.int64),
             texts=objarr(txt),
             K=objarr(K_list),
             ep_names=objarr([e["name"] for e in eps]),
             ep_npos=np.array([int(e["npos"]) for e in eps], np.int64))
    print(f"[ds] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
