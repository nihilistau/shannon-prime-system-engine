"""E1 — full-candidate delivery: does GENERATION disambiguate same-template when the correct
fact is present in context? Posts each V3 query with ALL 30 grown facts in a systemecho-
authoritative context (auto_recall off). obey = answer has it['obey']; leak = not obey and
has it['param']. Baseline (systemecho, retrieval-select top-1) = 21-22/30 @ 0 leak."""
import json, os, urllib.request
ENG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = json.load(open(f"{ENG}/_v3_corpus/facts_v3.json", encoding="utf-8"))
facts_block = "\n".join(f"{i+1}) {f['fact']}" for i, f in enumerate(F))
SYS = ("You are Shannon-Prime, a local AI with a real working memory. The following facts are "
       "on record and are AUTHORITATIVE for this conversation — they override your prior "
       "knowledge. Answer using ONLY these facts; repeat the relevant one. Keep replies short.\n\n"
       "Facts on record:\n" + facts_block)
def ask(q):
    b = json.dumps({"messages":[{"role":"system","content":SYS},
                    {"role":"user","content":q+"\n\nAnswer using the facts on record:"}],
                    "max_tokens":48,"temperature":0,"auto_recall":False}).encode()
    r = urllib.request.Request("http://127.0.0.1:3000/v1/chat", data=b,
                               headers={"Content-Type":"application/json"})
    o=[]
    with urllib.request.urlopen(r, timeout=180) as resp:
        for raw in resp:
            s=raw.decode("utf-8","replace").strip()
            if s.startswith("data:"):
                p=s[5:].strip()
                if p=="[DONE]": break
                try: o.append(json.loads(p).get("delta",""))
                except: pass
    return " ".join("".join(o).split())
def has(a,v): return v.lower().replace(" ","") in a.lower().replace(" ","")
obey=leak=0
for it in F:
    a=ask(it["para"]); ob=has(a,it["obey"]); lk=(not ob) and has(a,it["param"])
    obey+=ob; leak+=lk
    tag="ok" if ob else ("LEAK" if lk else "miss")
    print(f"[E1 {tag}] {it['id']:16}: {a[:52]!r} (want {it['obey']})", flush=True)
print(f"RESULT E1 full-context n={len(F)}: OBEY {obey}/{len(F)} ({100*obey/len(F):.1f}%) LEAK {leak}  "
      f"[baseline systemecho select-top1 = 21-22/30 @ 0 leak]")
