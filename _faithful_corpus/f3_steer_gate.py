"""f3_steer_gate.py — G-FM-STEER-OBEY P-phase runner (GEODESIC ADR-003 §4.2).

Serve under run_f3_steer.bat <alpha> first. Runs the facts.json paraphrases
(oneconfig P-phase shape: CONSOLE system + para, auto_recall on) and scores:
  OBEY  = counterfactual token present   (it["obey"])
  LEAK  = parametric truth present AND obey token absent  (it["param"])
  OTHER = neither (echo/decline/wrong-fact cross-pick)

Usage: python f3_steer_gate.py <alpha-label> [--slice N]
Prints one summary line — pipe to the gate log. Baselines: plain a=0 same-day;
target systemecho 88.52%/0-leak (G-DELIVERY-SWEEP).
"""
import json, os, sys, urllib.request, datetime

ENG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = json.load(open(f"{ENG}/_faithful_corpus/facts.json", encoding="utf-8"))
CONSOLE = ("You are Shannon-Prime, a local AI with a real working memory. Keep replies short. "
           "Use facts you were given faithfully; if you don't know, say so.")

def ask(q):
    b = json.dumps({"messages": [{"role": "system", "content": CONSOLE},
                                 {"role": "user", "content": q}],
                    "max_tokens": 48, "temperature": 0, "eot_bias": 4.0,
                    "auto_recall": True}).encode()
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

def has(ans, val): return val.lower().replace(" ", "") in ans.lower().replace(" ", "")

label = sys.argv[1] if len(sys.argv) > 1 else "?"
n = len(F)
if "--slice" in sys.argv: n = int(sys.argv[sys.argv.index("--slice") + 1])
items = F[:n]
obey = leak = 0
for it in items:
    a = ask(it["para"])
    ob = has(a, it["obey"]); lk = (not ob) and has(a, it["param"])
    obey += ob; leak += lk
    tag = "ok" if ob else ("LEAK" if lk else "miss")
    print(f"[a={label} {tag}] {it['id']}: {a[:56]!r}", flush=True)
print(f"RESULT a={label} n={n}: OBEY {obey}/{n} ({100*obey/n:.1f}%)  LEAK {leak}  "
      f"[plain-baseline 40.98% | target systemecho 88.52%/0]  {datetime.datetime.now().isoformat()}")
