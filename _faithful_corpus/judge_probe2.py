import json,os,urllib.request
ENG=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
items=json.load(open(os.path.join(ENG,"_faithful_corpus","judge_items_v2.json"),encoding="utf-8"))
# RELEVANCE judge (decoupled from truth): is the note ABOUT the topic the question asks about?
SYS=("You decide whether a retrieved note is ABOUT THE SAME TOPIC that a question asks about. "
     "Ignore whether the note's answer is correct or not -- judge ONLY topic match. "
     "Reply with exactly one word: YES (same topic) or NO (different topic).")
def judge(q,fact):
    user=f"Question: {q}\nRetrieved note: {fact}\nIs the note about the same topic as the question? Answer YES or NO."
    body=json.dumps({"messages":[{"role":"system","content":SYS},{"role":"user","content":user}],
                     "max_tokens":4,"temperature":0,"eot_bias":4.0,"auto_recall":False}).encode()
    req=urllib.request.Request("http://127.0.0.1:3000/v1/chat",data=body,headers={"Content-Type":"application/json"})
    out=[]
    with urllib.request.urlopen(req,timeout=120) as r:
        for raw in r:
            s=raw.decode("utf-8","replace").strip()
            if s.startswith("data:"):
                p=s[5:].strip()
                if p=="[DONE]": break
                try: out.append(json.loads(p).get("delta",""))
                except: pass
    return "".join(out).strip().upper()
iy=iN=fy=fN=0
for it in items:
    a=judge(it["query"],it["fact"]); yes=a.startswith("YES") or a=="Y"
    if it["inmem"]: iN+=1; iy+=yes
    else: fN+=1; fy+=yes
    print(f"[{'IN ' if it['inmem'] else 'FOR'}] {a[:4]!r:6} Q={it['query'][:34]} | {it['fact'][:32]}",flush=True)
print(f"\n=== RELEVANCE JUDGE PASS/BLOCK ===")
print(f"in-memory ACCEPT (YES): {iy}/{iN} = {100*iy/iN:.1f}%")
print(f"foreign FALSE-accept (YES): {fy}/{fN} = {100*fy/fN:.1f}%   (reject {100*(fN-fy)/fN:.1f}%)")
