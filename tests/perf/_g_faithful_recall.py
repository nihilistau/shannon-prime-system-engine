"""G-FAITHFUL-RECALL (F1b) — recall-path fact obedience.

Same 15 fact-conflicts as the F1 in-context baseline, but the override fact is NOT in the
prompt — it must be fetched by auto_recall from the seeded registry (run _seed_faithful.py
first) and delivered via the daemon's recall seam (SP_B3_JUDGE=text-in-context, or
SP_B3_WC=pure-KV replay). Measures obedience vs the F1 100% in-context ceiling; the gap is the
cost of the recall delivery mechanism. SP_FAITHFUL_SEAM labels the receipt (e.g. 'judge'|'wc').
"""
import json, os, sys, urllib.request
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
PORT = int(os.environ.get("SP_TEST_PORT", "3000"))
SEAM = os.environ.get("SP_FAITHFUL_SEAM", "unknown")
CONSOLE = ("You are Shannon-Prime, a local AI with a real working memory. Keep replies short. "
           "Use facts you were given faithfully; if you don't know, say so.")

# Shared fact-conflict corpus (id, question, parametric_token, obey_token) from facts.json.
_ENG = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_FJSON = os.path.join(_ENG, "_faithful_corpus", "facts.json")
ITEMS = [(it["id"], it["q"], it["param"], it["obey"]) for it in json.load(open(_FJSON, encoding="utf-8"))]


def ask(q):
    body = json.dumps({"messages": [{"role": "system", "content": CONSOLE},
                                    {"role": "user", "content": q}],
                       "max_tokens": 48, "temperature": 0, "eot_bias": 4.0,
                       "auto_recall": True}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    out = []
    with urllib.request.urlopen(req, timeout=180) as r:
        for raw in r:
            s = raw.decode("utf-8", "replace").strip()
            if s.startswith("data:"):
                p = s[5:].strip()
                if p == "[DONE]":
                    break
                try:
                    d = json.loads(p)
                    if "delta" in d:
                        out.append(d["delta"])
                except Exception:
                    pass
    return " ".join("".join(out).split())


def classify(ans, obey_tok, param_tok):
    a = ans.lower()
    o, p = obey_tok.lower() in a, param_tok.lower() in a
    if o and not p: return "OBEY"
    if p and not o: return "LEAK"
    if o and p:     return "BOTH"
    return "OTHER"


def main():
    tally = {"OBEY": 0, "LEAK": 0, "BOTH": 0, "OTHER": 0}
    rows = []
    for (iid, q, ptok, otok) in ITEMS:
        a = ask(q)
        c = classify(a, otok, ptok)
        tally[c] += 1
        rows.append({"id": iid, "cls": c, "ans": a[:90]})
        print(f"[{iid:16}] {c:5}  {a[:60]!r}", flush=True)
    n = len(ITEMS)
    print(f"\n=== G-FAITHFUL-RECALL (seam={SEAM}) ===", flush=True)
    print(f"items={n}  OBEY={tally['OBEY']} ({tally['OBEY']/n:.2%})  LEAK={tally['LEAK']}  BOTH={tally['BOTH']}  OTHER={tally['OTHER']}", flush=True)
    print(f"in-context ceiling (F1 T_CONSOLE) = 100%; recall-path delivery cost = {100 - tally['OBEY']/n*100:.1f} pp", flush=True)
    dest = os.path.join(os.path.dirname(__file__), f"_g_faithful_recall_{SEAM}.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump({"seam": SEAM, "items": n, "tally": tally, "obey_rate": tally["OBEY"] / n, "rows": rows}, f, indent=2)
    print(f"WROTE {dest}", flush=True)


if __name__ == "__main__":
    main()
