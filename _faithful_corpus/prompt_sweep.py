"""prompt_sweep.py [n] — delivery-format sweep slice runner (G-DELIVERY-SWEEP).
Runs the first n (default 16) paraphrase items against the live daemon and scores
OBEY/LEAK/OTHER exactly like tests/perf/_g_faithful_recall.py. The serve's
SP_RECALL_L5_PROMPT decides the variant; this script just measures."""
import json, os, sys, urllib.request

ENG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = json.load(open(f"{ENG}/_faithful_corpus/facts.json", encoding="utf-8"))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 16
CONSOLE = ("You are Shannon-Prime, a local AI with a real working memory. Keep replies short. "
           "Use facts you were given faithfully; if you don't know, say so.")

def ask(q):
    b = json.dumps({"messages": [{"role": "system", "content": CONSOLE}, {"role": "user", "content": q}],
                    "max_tokens": 48, "temperature": 0, "eot_bias": 4.0, "auto_recall": True}).encode()
    r = urllib.request.Request("http://127.0.0.1:3000/v1/chat", data=b,
                               headers={"Content-Type": "application/json"})
    o = []
    with urllib.request.urlopen(r, timeout=200) as resp:
        for raw in resp:
            s = raw.decode("utf-8", "replace").strip()
            if s.startswith("data:"):
                p = s[5:].strip()
                if p == "[DONE]": break
                try: o.append(json.loads(p).get("delta", ""))
                except Exception: pass
    return " ".join("".join(o).split())

tally = {"OBEY": 0, "LEAK": 0, "BOTH": 0, "OTHER": 0}
for it in F[:N]:
    a = ask(it["para"]); al = a.lower()
    o, p = it["obey"].lower() in al, it["param"].lower() in al
    c = "OBEY" if (o and not p) else "LEAK" if (p and not o) else "BOTH" if o else "OTHER"
    tally[c] += 1
    print(f"[{it['id']:16}] {c:5} {a[:56]!r}", flush=True)
print(f"SLICE-{N}: {tally}  obey={tally['OBEY']}/{N}", flush=True)
