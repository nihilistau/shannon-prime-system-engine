import json,urllib.request,sys
sys.stdout.reconfigure(encoding="utf-8",errors="replace")
def chat(q,mx=80):
    body=json.dumps({"messages":[{"role":"user","content":q}],"max_tokens":mx,"temperature":0.7}).encode()
    req=urllib.request.Request("http://127.0.0.1:3000/v1/chat",data=body,headers={"Content-Type":"application/json"})
    out=[]
    try:
        with urllib.request.urlopen(req,timeout=160) as r:
            for raw in r:
                s=raw.decode("utf-8","replace").strip()
                if s.startswith("data:"):
                    p=s[5:].strip()
                    if p=="[DONE]": break
                    try:
                        ev=json.loads(p)
                        if "delta" in ev: out.append(ev["delta"])
                    except: pass
    except Exception as e: return "[ERR]"
    return "".join(out)
for q in ["Is the sky blue? Answer in one word.","Tell me about the weather today.","Who are you?"]:
    print("Q:",q); print("A:"," ".join(chat(q).split())[:260]); print("-"*50)