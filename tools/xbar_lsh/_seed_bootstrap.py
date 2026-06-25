import json,urllib.request,sys,time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HOST="http://127.0.0.1:3001"
def chat(q):
    body=json.dumps({"messages":[{"role":"user","content":q}],"auto_recall":True,"max_tokens":80,"temperature":0}).encode()
    req=urllib.request.Request(HOST+"/v1/chat",data=body,headers={"Content-Type":"application/json"})
    out=[]
    try:
        with urllib.request.urlopen(req,timeout=240) as r:
            for raw in r:
                s=raw.decode("utf-8","replace").strip()
                if not s.startswith("data:"): continue
                p=s[5:].strip()
                if p=="[DONE]": break
                try: ev=json.loads(p)
                except Exception: continue
                if "delta" in ev: out.append(ev["delta"])
    except Exception as e: return "[ERR:%s]"%(str(e)[:60])
    return "".join(out)
turns=[
 ("Q1 identity",      "What are you?"),
 ("Q2 operator",      "Who is running these sessions?"),
 ("Q3 self-mechanism","How do you store your memories?"),
 ("Q4 NEW fact",      "Please remember this: the lab access code is BLUE-OTTER-42."),
 ("Q5 recall new",    "What is the lab access code?"),
 ("Q6 TRUE foreign",  "How many legs does a spider have?"),
]
for label,q in turns:
    r=chat(q)
    print("="*70); print(label); print("Q:",q)
    print("A:", " ".join(r.split())[:350]); sys.stdout.flush(); time.sleep(1)
print("="*70); print("SEED_BOOTSTRAP_DONE")