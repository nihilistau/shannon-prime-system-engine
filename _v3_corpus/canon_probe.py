"""canon_probe.py — does the name-the-subject micro-forward recover a subject token that
appears in the CORRECT fact (and not the magnet)? Sends the CANON prompt to the live
daemon for the 8 cross-picks + 4 correct controls; checks overlap with correct vs magnet.
"""
import json, os, re, urllib.request

ENG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = json.load(open(f"{ENG}/_v3_corpus/facts_v3.json", encoding="utf-8"))
reg = [json.loads(l) for l in open(f"{ENG}/_v3_corpus/registry.jsonl", encoding="utf-8") if l.strip()]
names=[r["name"] for r in reg]; texts=[r["text"] for r in reg]; n2t=dict(zip(names,texts))

CANON = ('A question describes something without naming it. Name the described thing, place, '
         'or person - do NOT answer the question.\nExample: "What is the tallest building in the '
         'city that never sleeps?" describes New York.\nQuestion: "{q}" describes:')

def ask_raw(prompt):
    b=json.dumps({"messages":[{"role":"user","content":prompt}],"max_tokens":16,
                  "temperature":0,"auto_recall":False}).encode()
    r=urllib.request.Request("http://127.0.0.1:3000/v1/chat",data=b,headers={"Content-Type":"application/json"})
    o=[]
    with urllib.request.urlopen(r,timeout=120) as resp:
        for raw in resp:
            s=raw.decode("utf-8","replace").strip()
            if s.startswith("data:"):
                p=s[5:].strip()
                if p=="[DONE]": break
                try: o.append(json.loads(p).get("delta",""))
                except: pass
    return "".join(o).strip()

def toks(s): return set(w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w)>2)

cross=["dynamite_inv","hamlet_author","starry_night","radium_disc","david_sculptor",
       "evolution_theory","first_element","telescope_inv"]
ctrl=["longest_river","uk_currency","fuji_country","colosseum_city"]
top1={"dynamite_inv":"ep_live_m1783039367579","hamlet_author":"ep_live_m1783039125927",
      "starry_night":"ep_live_m1783039137070","radium_disc":"ep_live_m1783039367579",
      "david_sculptor":"ep_live_m1783039367579","evolution_theory":"ep_live_m1783039367579",
      "first_element":"ep_live_m1783039260541","telescope_inv":"ep_live_m1783039367579"}

for grp,ids in [("CROSSPICK",cross),("CONTROL(correct)",ctrl)]:
    print(f"\n=== {grp} ===")
    for cid in ids:
        i=next(k for k,f in enumerate(F) if f["id"]==cid)
        q=F[i]["para"]; subj=ask_raw(CANON.format(q=q)).split("\n")[0][:40]
        st=toks(subj); ov_c=len(st & toks(texts[i]))
        ov_m=len(st & toks(n2t[top1[cid]])) if cid in top1 else 0
        verdict = "GROUNDS-CORRECT" if ov_c>ov_m else ("ambiguous" if ov_c==ov_m else "grounds-MAGNET")
        print(f"{cid:16} subj={subj!r:44} ov(correct)={ov_c} ov(magnet)={ov_m} -> {verdict}")
