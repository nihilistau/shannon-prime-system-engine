import json,os,sys,urllib.request
ENG=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F=json.load(open(f"{ENG}/_faithful_corpus/facts.json",encoding="utf-8"))
HF=json.load(open(f"{ENG}/_faithful_corpus/hard_foreign_queries.json",encoding="utf-8"))
CONSOLE=("You are Shannon-Prime, a local AI with a real working memory. Keep replies short. "
         "Use facts you were given faithfully; if you don't know, say so.")
# planted counterfactual tokens (>3 chars); a hard-foreign answer containing one = the model
# ADOPTED a mismatched delivered fact = spurious delivery (the failure the judge must prevent).
planted=[f["obey"].lower() for f in F if len(f["obey"])>3]
def ask(q):
    b=json.dumps({"messages":[{"role":"system","content":CONSOLE},{"role":"user","content":q}],
                  "max_tokens":40,"temperature":0,"eot_bias":4.0,"auto_recall":True}).encode()
    r=urllib.request.Request(f"http://127.0.0.1:3000/v1/chat",data=b,headers={"Content-Type":"application/json"})
    o=[]
    with urllib.request.urlopen(r,timeout=200) as resp:
        for raw in resp:
            s=raw.decode("utf-8","replace").strip()
            if s.startswith("data:"):
                p=s[5:].strip()
                if p=="[DONE]": break
                try: o.append(json.loads(p).get("delta",""))
                except: pass
    return " ".join("".join(o).split())
spur=0
print("=== HARD-FOREIGN (high-cosine, same-domain, UNANSWERABLE by the anchored fact) ===",flush=True)
for it in HF:
    q=it["q"]; a=ask(q)
    hit=[t for t in planted if t in a.lower()]
    bad=len(hit)>0; spur+=bad
    print(f"[{'SPURIOUS' if bad else 'clean'}] {a[:56]!r}  <- {q}  (anchor {it['anchor']}, real {it['real']}){'  HIT='+str(hit) if hit else ''}",flush=True)
n=len(HF)
print(f"\n=== HARD-FOREIGN RESULT (N={n}) ===",flush=True)
print(f"spurious-delivery (model adopted a mismatched planted fact): {spur}/{n} = {100*spur/n:.0f}%  (clean {100*(n-spur)/n:.0f}%)",flush=True)
