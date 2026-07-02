import json,os,sys,urllib.request,re
ENG=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N=int(sys.argv[1]) if len(sys.argv)>1 else 12
F=json.load(open(f"{ENG}/_faithful_corpus/facts.json",encoding="utf-8"))
FQ=json.load(open(f"{ENG}/_faithful_corpus/foreign_queries_v2.json",encoding="utf-8"))
CONSOLE=("You are Shannon-Prime, a local AI with a real working memory. Keep replies short. "
         "Use facts you were given faithfully; if you don't know, say so.")
planted=set(f["obey"].lower() for f in F)  # the wrong planted answers; a foreign answer containing one = spurious delivery
def ask(q):
    b=json.dumps({"messages":[{"role":"system","content":CONSOLE},{"role":"user","content":q}],
                  "max_tokens":40,"temperature":0,"eot_bias":4.0,"auto_recall":True}).encode()
    r=urllib.request.Request("http://127.0.0.1:3000/v1/chat",data=b,headers={"Content-Type":"application/json"})
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
# PARA (recall via judge): obey token present == judge recalled the right fact
pobey=0
print("=== PARA (judge recall) ===",flush=True)
for it in F[:N]:
    a=ask(it.get("para") or it["q"]); ok=it["obey"].lower() in a.lower() and it["param"].lower() not in a.lower()
    pobey+=ok; print(f"[{'OBEY' if ok else 'miss'}] {a[:44]!r} <- {(it.get('para') or it['q'])[:40]}",flush=True)
# FOREIGN (reject via judge): a spurious planted token in the answer == judge wrongly delivered
fspur=0
print("=== FOREIGN (judge reject) ===",flush=True)
for q in FQ[:N]:
    a=ask(q); spur=any(t in a.lower() for t in planted if len(t)>3)
    fspur+=spur; print(f"[{'SPURIOUS' if spur else 'clean'}] {a[:44]!r} <- {q[:40]}",flush=True)
print(f"\n=== JUDGE PASS/BLOCK (N={N} each) ===",flush=True)
print(f"PARA recall (obey): {pobey}/{N} = {100*pobey/N:.0f}%",flush=True)
print(f"FOREIGN spurious-delivery: {fspur}/{N} = {100*fspur/N:.0f}%  (clean-reject {100*(N-fspur)/N:.0f}%)",flush=True)
