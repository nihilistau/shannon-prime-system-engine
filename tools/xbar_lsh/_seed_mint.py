import json,urllib.request,os
HOST="http://127.0.0.1:3001"
ENG="D:/F/shannon-prime-repos/shannon-prime-system-engine"
BASE=ENG+"/_seed_corpus/eps"
seeds=[
 "This system is an experimental model called Shannon-Prime.",
 "Shannon-Prime can store facts from a conversation and recall them in later turns, which most language models cannot do.",
 "Shannon-Prime stores its memories as exact-integer latent signatures, not as text.",
 "Shannon-Prime runs locally on a single NVIDIA RTX 2060 graphics card.",
 "The person running these sessions goes by the handle Knack.",
 "These sessions are early experiments in giving a language model a working memory.",
 "Shannon-Prime only turns statements into memories, never questions.",
 "When no stored memory grounds an answer, Shannon-Prime declines rather than guessing.",
]
reg=[]
for i,text in enumerate(seeds):
    od="%s/seed_%03d"%(BASE,i)
    body=json.dumps({"text":text,"out_dir":od}).encode()
    req=urllib.request.Request(HOST+"/v1/capture",data=body,headers={"Content-Type":"application/json"})
    try:
        with urllib.request.urlopen(req,timeout=120) as r:
            resp=json.loads(r.read().decode())
    except Exception as e:
        resp={"error":str(e)[:80]}
    print("seed_%03d npos=%s  %s"%(i,resp.get("npos"),text[:45]))
    if resp.get("ok"):
        reg.append({"name":"seed_%03d"%i,"dir":od,"npos":resp["npos"],"topic":text,"sig_bits":"0"*64})
os.makedirs(ENG+"/_seed_corpus",exist_ok=True)
with open(ENG+"/_seed_corpus/registry.jsonl","w",encoding="utf-8") as f:
    for e in reg: f.write(json.dumps(e)+"\n")
print("REGISTRY_WRITTEN",len(reg),"episodes")