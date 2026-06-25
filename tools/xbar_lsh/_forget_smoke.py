import json,urllib.request,sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HOST="http://127.0.0.1:3000"
def chat(q):
    body=json.dumps({"messages":[{"role":"user","content":q}],"auto_recall":True,"max_tokens":64,"temperature":0}).encode()
    req=urllib.request.Request(HOST+"/v1/chat",data=body,headers={"Content-Type":"application/json"})
    out=[]
    try:
        with urllib.request.urlopen(req,timeout=300) as r:
            for raw in r:
                s=raw.decode("utf-8","replace").strip()
                if not s.startswith("data:"): continue
                p=s[5:].strip()
                if p=="[DONE]": break
                try: ev=json.loads(p)
                except Exception: continue
                if "delta" in ev: out.append(ev["delta"])
    except Exception as e:
        return "[ERR:%s]"%(str(e)[:80])
    return "".join(out)
seq=[
 ("PLANT","Please remember this: the secret vault code is 7-RAVEN-3300."),
 ("RECALL-BEFORE","What is the secret vault code?"),
 ("FORGET","Forget the secret vault code."),
 ("RECALL-AFTER","What is the secret vault code?"),
]
for kind,q in seq:
    r=chat(q)
    print("="*70)
    print("[%s] Q: %s"%(kind,q))
    print("REPLY:", " ".join(r.split())[:400])
    sys.stdout.flush()
print("="*70); print("FORGET_SMOKE_DONE")
