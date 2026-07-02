import json,os,sys,urllib.request,re
ENG=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
items=json.load(open(os.path.join(ENG,"_faithful_corpus","judge_items_v2.json"),encoding="utf-8"))
SYS=("You verify whether a retrieved note answers a question. Reply with exactly one word: "
     "YES if the note directly answers the question, or NO if it is about something else.")
def judge(q,fact):
    user=f"Question: {q}\nRetrieved note: {fact}\nDoes the note directly answer the question? Answer YES or NO."
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
                try:
                    d=json.loads(p);  out.append(d.get("delta",""))
                except: pass
    return "".join(out).strip().upper()
im_yes=im_n=fo_yes=fo_n=0
for it in items:
    a=judge(it["query"],it["fact"]); yes = a.startswith("YES") or a=="Y"
    if it["inmem"]: im_n+=1; im_yes+=yes
    else: fo_n+=1; fo_yes+=yes
    print(f"[{'IN ' if it['inmem'] else 'FOR'}] judge={a[:6]!r:8} Q={it['query'][:38]} | note={it['fact'][:34]}",flush=True)
print(f"\n=== JUDGE PASS/BLOCK ===")
print(f"in-memory (want YES/accept): {im_yes}/{im_n} = {100*im_yes/im_n:.1f}% accept")
print(f"foreign  (want NO/reject):  {fo_yes}/{fo_n} = {100*fo_yes/fo_n:.1f}% FALSE-accept  (reject={100*(fo_n-fo_yes)/fo_n:.1f}%)")
