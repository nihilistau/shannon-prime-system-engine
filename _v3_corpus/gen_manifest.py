"""gen_manifest.py — build the W_c training manifest for the V3 corpus.
Per episode (registry order == facts order): id = registry name minus 'ep_', query = the
fact's canonical question, paraphrases = facts_v3 `para` + N model-generated rewrites.
Also emits foreign_queries.txt (off-topic NULL class). Daemon must be up (any config).
"""
import json, os, re, urllib.request

ENG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = json.load(open(f"{ENG}/_v3_corpus/facts_v3.json", encoding="utf-8"))
reg = [json.loads(l) for l in open(f"{ENG}/_v3_corpus/registry.jsonl", encoding="utf-8") if l.strip()]
assert len(F) == len(reg), f"{len(F)} facts vs {len(reg)} episodes"

def ask(prompt, mx=120):
    b = json.dumps({"messages":[{"role":"user","content":prompt}],"max_tokens":mx,
                    "temperature":0,"auto_recall":False}).encode()
    r = urllib.request.Request("http://127.0.0.1:3000/v1/chat", data=b,
                               headers={"Content-Type":"application/json"})
    o=[]
    with urllib.request.urlopen(r, timeout=180) as resp:
        for raw in resp:
            s=raw.decode("utf-8","replace").strip()
            if s.startswith("data:"):
                p=s[5:].strip()
                if p=="[DONE]": break
                try: o.append(json.loads(p).get("delta",""))
                except: pass
    return "".join(o)

def ep_question(name):
    d = os.path.join(ENG, "_nightshift_live", name, "ep.q")
    return open(d, encoding="utf-8").read().strip() if os.path.exists(d) else ""

man = open(f"{ENG}/_v3_corpus/manifest.jsonl", "w", encoding="utf-8")
for i, f in enumerate(F):
    q = ep_question(reg[i]["name"]) or f.get("para", "")
    para = f.get("para", "")
    raw = ask(f'Rewrite this question in 6 different ways that keep the exact same meaning. '
              f'Number them 1-6, one per line, nothing else.\nQuestion: {q}')
    variants = []
    for ln in raw.splitlines():
        ln = re.sub(r'^\s*\d+[\).\:]\s*', '', ln).strip()
        if len(ln) > 6 and ln.endswith('?'):
            variants.append(ln)
    # dedup, cap 6, always include the facts_v3 para
    seen=set(); paraphrases=[]
    for v in ([para] if para else []) + variants:
        k=v.lower().strip()
        if k and k not in seen and v.strip()!=q.strip():
            seen.add(k); paraphrases.append(v.strip())
    paraphrases = paraphrases[:7]
    epid = reg[i]["name"][3:] if reg[i]["name"].startswith("ep_") else reg[i]["name"]
    man.write(json.dumps({"id": epid, "query": q, "paraphrases": paraphrases,
                          "fact_id": f["id"]}) + "\n")
    print(f"[{f['id']:16}] {len(paraphrases)} paraphrases", flush=True)
man.close()

FOREIGN = [
 "What is the capital of France?","How do I bake sourdough bread?","What time is it in Tokyo?",
 "Explain how a bicycle works.","What is the square root of 144?","Who won the 2010 World Cup?",
 "How tall is the Eiffel Tower?","What is photosynthesis?","Recommend a good sci-fi movie.",
 "How do vaccines work?","What is the boiling point of nitrogen?","Translate hello into Spanish.",
 "What is the population of Canada?","How does a car engine work?","What is a black hole?",
 "Give me a recipe for pancakes.","What is the speed of light?","How do I tie a tie?",
 "What causes rainbows?","What is compound interest?",
]
open(f"{ENG}/_v3_corpus/foreign_queries.txt","w",encoding="utf-8").write("\n".join(FOREIGN)+"\n")
print(f"wrote manifest.jsonl ({len(F)} episodes) + foreign_queries.txt ({len(FOREIGN)})")
