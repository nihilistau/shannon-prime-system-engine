import json,urllib.request,sys
sys.stdout.reconfigure(encoding="utf-8",errors="replace")
body=json.dumps({"messages":[{"role":"user","content":"Tell me about the weather today."}],"max_tokens":80,"temperature":0.7}).encode()
req=urllib.request.Request("http://127.0.0.1:3000/v1/chat",data=body,headers={"Content-Type":"application/json"})
out=[]
with urllib.request.urlopen(req,timeout=180) as r:
    for raw in r:
        s=raw.decode("utf-8","replace").strip()
        if s.startswith("data:"):
            p=s[5:].strip()
            if p=="[DONE]": break
            try:
                ev=json.loads(p)
                if "delta" in ev: out.append(ev["delta"])
            except: pass
print("REPLY:"," ".join("".join(out).split())[:300])