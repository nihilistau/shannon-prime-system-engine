"""E2b — the fix for E2: ask TOPICALITY not TRUTH. 'Does the statement concern the SAME
specific subject the question is about?' (yes/no). This must not invoke world-knowledge
truth (which made E2 reject every counterfact). Separation target: correct->YES, magnet->NO."""
import json, os, urllib.request
ENG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = {f["id"]: f for f in json.load(open(f"{ENG}/_v3_corpus/facts_v3.json", encoding="utf-8"))}
CROSS = {"dynamite_inv":"radium_disc","hamlet_author":"radium_disc","starry_night":"radium_disc",
         "david_sculptor":"radium_disc","evolution_theory":"insulin_disc","first_element":"fastest_bird",
         "telescope_inv":"radium_disc","colosseum_city":"egypt_capital","kenya_capital":"egypt_capital"}
CONTROLS = ["longest_river","iron_symbol","chromosomes","berlin_wall","fuji_country"]
def topical(fact, query):
    p = (f'Question: "{query}"\nStatement: "{fact}"\n'
         f'Ignore whether the statement is true. Is the statement ABOUT the same specific '
         f'thing, person, or place that the question is asking about? Answer only yes or no.')
    b = json.dumps({"messages":[{"role":"user","content":p}],"max_tokens":4,
                    "temperature":0,"auto_recall":False}).encode()
    r = urllib.request.Request("http://127.0.0.1:3000/v1/chat", data=b,
                               headers={"Content-Type":"application/json"})
    o=[]
    with urllib.request.urlopen(r, timeout=120) as resp:
        for raw in resp:
            s=raw.decode("utf-8","replace").strip()
            if s.startswith("data:"):
                p2=s[5:].strip()
                if p2=="[DONE]": break
                try: o.append(json.loads(p2).get("delta",""))
                except: pass
    t="".join(o).strip().lower()
    return "yes" if t.startswith("y") else ("no" if t.startswith("n") else t[:6])
print("=== E2b TOPICALITY: correct should YES, magnet should NO ===")
sep=0
for qid, mid in CROSS.items():
    q=F[qid]["para"]; c=topical(F[qid]["fact"], q); m=topical(F[mid]["fact"], q)
    ok=(c=="yes" and m=="no"); sep+=ok
    print(f"{qid:16} correct={c:4} magnet({mid})={m:4} -> {'SEP' if ok else 'no'}", flush=True)
print(f"separated: {sep}/{len(CROSS)}")
cy=0
print("=== CONTROLS: correct should YES ===")
for qid in CONTROLS:
    c=topical(F[qid]["fact"], F[qid]["para"]); cy+=(c=="yes")
    print(f"{qid:16} correct={c}", flush=True)
print(f"controls correct=yes: {cy}/{len(CONTROLS)}")
