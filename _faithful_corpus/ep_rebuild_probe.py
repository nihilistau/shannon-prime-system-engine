"""ep_rebuild_probe.py <out_subdir> — G-EP-REBUILD-BYTEEXACT arm: re-capture fct_000/fct_001
via the daemon /v1/capture (the same path _seed_faithful.py used on 07-01) and report byte
comparison vs the on-disk originals. Run once per serve ('a', then restart daemon, 'b')."""
import hashlib, json, os, sys, urllib.request

ENG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUB = sys.argv[1] if len(sys.argv) > 1 else "a"
F = json.load(open(f"{ENG}/_faithful_corpus/facts.json", encoding="utf-8"))

def capture(text, out_dir):
    body = json.dumps({"text": text, "out_dir": out_dir.replace("\\", "/")}).encode()
    req = urllib.request.Request("http://127.0.0.1:3000/v1/capture", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode())

def sha(p):
    try:
        return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
    except FileNotFoundError:
        return "MISSING"

for i in (0, 1):
    text = F[i]["fact"]
    out = f"{ENG}/_ep_rebuild/{SUB}/fct_{i:03d}"
    os.makedirs(out, exist_ok=True)
    j = capture(text, out)
    orig = f"{ENG}/_faithful_corpus/eps/fct_{i:03d}"
    print(f"fct_{i:03d} npos={j.get('npos')}")
    for f_ in ("ep.k", "ep.v", "ep.mf"):
        a, o = sha(f"{out}/{f_}"), sha(f"{orig}/{f_}")
        print(f"  {f_}: new={a} orig={o} {'IDENTICAL' if a == o else 'DIFFER'}")
print("done", flush=True)
