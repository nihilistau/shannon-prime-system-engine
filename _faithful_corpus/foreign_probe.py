import json, os, sys, urllib.request
ENG=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QF=os.path.join(ENG,"_faithful_corpus","foreign_queries.json")
CONSOLE=("You are Shannon-Prime, a local AI with a real working memory. Keep replies short. "
         "Use facts you were given faithfully; if you don't know, say so.")
def ask(q):
    body=json.dumps({"messages":[{"role":"system","content":CONSOLE},{"role":"user","content":q}],
                     "max_tokens":8,"temperature":0,"eot_bias":4.0,"auto_recall":True}).encode()
    req=urllib.request.Request("http://127.0.0.1:3000/v1/chat",data=body,headers={"Content-Type":"application/json"})
    out=[]
    with urllib.request.urlopen(req,timeout=120) as r:
        for raw in r:
            s=raw.decode("utf-8","replace").strip()
            if s.startswith("data:"):
                p=s[5:].strip()
                if p=="[DONE]": break
                try:
                    d=json.loads(p)
                    if "delta" in d: out.append(d["delta"])
                except: pass
    return " ".join("".join(out).split())
Q=json.load(open(QF,encoding="utf-8"))
for i,q in enumerate(Q):
    a=ask(q); print(f"[{i:02d}] {a[:40]!r}  <- {q[:50]}",flush=True)
print("DONE foreign probe")
