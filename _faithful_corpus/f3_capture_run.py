"""f3_capture_run.py — GEODESIC F3 data capture driver (ADR-003 §5, gate G-F3-CAPTURE).

Drives the served /v1/chat over the faithfulness corpus while the SP_F3_CAPTURE rail
(routes.rs) persists the paired post-output_norm residuals per turn.

Protocol (two serves, same items):
  A  run_f3_capture_A.bat  — Tier0+Tier1(systemecho), attr-gate OFF (capture-only),
                             SP_F3_CAPTURE=_faithful_corpus/f3/A. Requests mirror
                             oneconfig_run.py (CONSOLE system + user), auto_recall=true.
  B  run_f3_capture_B.bat  — Tier0 only (no recall), SP_F3_CAPTURE=_faithful_corpus/f3/B.
                             Requests carry NO system message, auto_recall=false.

Items: 61 facts.json paraphrases + 20 SNE mismatch queries (sne_facts_audited.json).

Usage:
  python f3_capture_run.py A|B [--limit N] [--twice]
    --limit N   first N of each set (smoke)
    --twice     send every item twice, then verify the two captures' payloads are
                byte-identical per item (the G-F3-CAPTURE determinism leg)

Verification (both runs): counts meta rows, checks one f3_*.bin header per run
(magic F3P1, E=3840), reports frames histogram. Receipt lines print to stdout —
pipe to the gate log.
"""
import json, os, struct, sys, time, urllib.request

ENG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F   = json.load(open(f"{ENG}/_faithful_corpus/facts.json", encoding="utf-8"))
SNE = json.load(open(f"{ENG}/_faithful_corpus/sne/sne_facts_audited.json", encoding="utf-8"))

CONSOLE = ("You are Shannon-Prime, a local AI with a real working memory. Keep replies short. "
           "Use facts you were given faithfully; if you don't know, say so.")

def ask(msgs, auto):
    b = json.dumps({"messages": msgs, "max_tokens": 48, "temperature": 0,
                    "eot_bias": 4.0, "auto_recall": auto}).encode()
    r = urllib.request.Request("http://127.0.0.1:3000/v1/chat", data=b,
                               headers={"Content-Type": "application/json"})
    o = []
    with urllib.request.urlopen(r, timeout=300) as resp:
        for raw in resp:
            s = raw.decode("utf-8", "replace").strip()
            if s.startswith("data:"):
                p = s[5:].strip()
                if p == "[DONE]": break
                try: o.append(json.loads(p).get("delta", ""))
                except Exception: pass
    return " ".join("".join(o).split())

def run(label, limit, twice):
    fdir = f"{ENG}/_faithful_corpus/f3/{label}"
    items = [("fct", it["id"], it["para"]) for it in F[:limit or len(F)]] + \
            [("sne", it["entity"], it["mismatch_q"]) for it in SNE[:limit or len(SNE)]]
    reps = 2 if twice else 1
    n_before = _meta_count(fdir)
    t0 = time.time()
    for kind, iid, q in items:
        for rep in range(reps):
            if label == "A":
                msgs = [{"role": "system", "content": CONSOLE},
                        {"role": "user", "content": q}]
                a = ask(msgs, auto=True)
            else:
                a = ask([{"role": "user", "content": q}], auto=False)
            print(f"[{label} {kind} rep{rep}] {iid}: {a[:56]!r}", flush=True)
    dt = time.time() - t0
    n_after = _meta_count(fdir)
    print(f"[{label}] {len(items)}x{reps} turns in {dt:.0f}s; meta rows {n_before} -> {n_after}")
    ok = verify(fdir, label, items, reps, n_after - n_before)
    print(f"[{label}] VERIFY: {'PASS' if ok else 'FAIL'}")
    return ok

def _meta_count(fdir):
    p = f"{fdir}/f3_meta.jsonl"
    if not os.path.exists(p): return 0
    return sum(1 for _ in open(p, encoding="utf-8"))

def verify(fdir, label, items, reps, n_new):
    ok = True
    want = len(items) * reps
    if n_new != want:
        print(f"  [verify] meta rows: got {n_new} new, want {want} — FAIL"); ok = False
    else:
        print(f"  [verify] meta rows: {n_new}/{want} ok")
    metas = [json.loads(l) for l in open(f"{fdir}/f3_meta.jsonl", encoding="utf-8")]
    metas = metas[-n_new:] if n_new <= len(metas) else metas
    # header check on every new bin + frames histogram
    hist = {}
    for m in metas:
        p = f"{fdir}/f3_{m['chat_id']}.bin"
        if not os.path.exists(p):
            print(f"  [verify] MISSING {p}"); ok = False; continue
        h = open(p, "rb").read(16)
        magic, e, nf, _pad = h[:4], *struct.unpack("<3I", h[4:16])
        if magic != b"F3P1" or e != 3840 or nf not in (1, 2):
            print(f"  [verify] BAD HEADER {p}: {magic} E={e} nf={nf}"); ok = False
        sz = os.path.getsize(p)
        if sz != 16 + 3840 * 4 * nf:
            print(f"  [verify] BAD SIZE {p}: {sz} for nf={nf}"); ok = False
        hist[nf] = hist.get(nf, 0) + 1
    print(f"  [verify] frames histogram: {hist}")
    # A-run sanity: fct turns should mostly be recall mode (systemecho), sne may vary
    if label == "A":
        n_recall = sum(1 for m in metas if m.get("recalled"))
        print(f"  [verify] A recall-fired turns: {n_recall}/{len(metas)}")
    else:
        n_recall = sum(1 for m in metas if m.get("recalled"))
        if n_recall: print(f"  [verify] B had {n_recall} recall turns — FAIL (must be 0)"); ok = False
        else: print("  [verify] B recall-fired turns: 0/{} ok".format(len(metas)))
    # --twice determinism: consecutive same-user pairs byte-identical payloads
    if reps == 2:
        byu = {}
        for m in metas: byu.setdefault(m["user"], []).append(m["chat_id"])
        n_pair = n_id = 0
        for u, ids in byu.items():
            if len(ids) != 2: continue
            n_pair += 1
            p0 = open(f"{fdir}/f3_{ids[0]}.bin", "rb").read()[16:]
            p1 = open(f"{fdir}/f3_{ids[1]}.bin", "rb").read()[16:]
            if p0 == p1: n_id += 1
            else: print(f"  [verify] NOT byte-identical: {u[:48]!r}")
        print(f"  [verify] determinism: {n_id}/{n_pair} rerun pairs byte-identical")
        if n_id != n_pair: ok = False
    return ok

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("A", "B"):
        print(__doc__); sys.exit(2)
    label = sys.argv[1]
    limit = 0
    twice = "--twice" in sys.argv
    if "--limit" in sys.argv: limit = int(sys.argv[sys.argv.index("--limit") + 1])
    sys.exit(0 if run(label, limit, twice) else 1)
