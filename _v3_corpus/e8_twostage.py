"""E8 — two-stage delivery: SELECT-in-context, then GENERATE-single-fact-authoritative.
Stage 1: all 30 facts numbered + the query -> "which number is about the same subject?"
         -> an INDEX (no value spoken => no leak vector). This is E1's disambiguation power.
Stage 2: deliver ONLY the selected fact with systemecho authority (the proven 0-leak path).
Hypothesis: E8 = E1's obey (~25) AND systemecho's 0 leak. Baseline: systemecho select-top1
22/30 @ 0 leak; E1 full-context 25/30 @ 4 leak."""
import json, os, re, urllib.request
ENG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = json.load(open(f"{ENG}/_v3_corpus/facts_v3.json", encoding="utf-8"))
numbered = "\n".join(f"{i+1}) {f['fact']}" for i, f in enumerate(F))

def chat(msgs, mx):
    b = json.dumps({"messages": msgs, "max_tokens": mx, "temperature": 0,
                    "auto_recall": False}).encode()
    r = urllib.request.Request("http://127.0.0.1:3000/v1/chat", data=b,
                               headers={"Content-Type": "application/json"})
    o=[]
    with urllib.request.urlopen(r, timeout=180) as resp:
        for raw in resp:
            s=raw.decode("utf-8","replace").strip()
            if s.startswith("data:"):
                p=s[5:].strip()
                if p=="[DONE]": break
                try: o.append(json.loads(p).get("delta",""))
                except: pass
    return "".join(o)

def select_idx(query):
    sysm = ("You are a memory index. Below are numbered facts on record. Pick the ONE whose "
            "subject is what the question is asking about. Reply with ONLY that number.\n\n" + numbered)
    out = chat([{"role":"system","content":sysm},
                {"role":"user","content":query+"\n\nNumber:"}], 6)
    m = re.search(r"\d+", out)
    return (int(m.group())-1) if m else None

def generate(fact, query):
    sysm = ("You are Shannon-Prime, a local AI with a real working memory. Fact on record "
            "(authoritative for this conversation, overrides prior knowledge): " + fact +
            "\nEvery answer must repeat the relevant part of the fact on record verbatim. Keep replies short.")
    return " ".join(chat([{"role":"system","content":sysm},
                          {"role":"user","content":query+"\n\nAnswer using the fact on record:"}], 48).split())

def has(a,v): return v.lower().replace(" ","") in a.lower().replace(" ","")
obey=leak=sel_ok=0
for j,it in enumerate(F):
    idx = select_idx(it["para"])
    sel = (idx == j)
    sel_ok += sel
    fact = F[idx]["fact"] if (idx is not None and 0 <= idx < len(F)) else it["fact"]
    a = generate(fact, it["para"])
    ob = has(a, it["obey"]); lk = (not ob) and has(a, it["param"])
    obey += ob; leak += lk
    tag = "ok" if ob else ("LEAK" if lk else "miss")
    print(f"[E8 {tag}] {it['id']:16} sel={'Y' if sel else 'n'}(idx={idx}): {a[:46]!r} (want {it['obey']})", flush=True)
print(f"RESULT E8 two-stage n={len(F)}: SELECT {sel_ok}/{len(F)}  OBEY {obey}/{len(F)} ({100*obey/len(F):.1f}%) LEAK {leak}  "
      f"[systemecho 22/0 · E1 25/4]")
