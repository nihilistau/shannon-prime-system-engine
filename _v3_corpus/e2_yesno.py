"""E2 — binary yes/no grounding: can "Does <fact> answer <query>? yes/no" REJECT the
cross-pick magnet (NO) while accepting the correct fact (YES)? No answer surfaced => no leak.
Focus on the observed cross-picks (query -> magnet it wrongly recalled) + correct controls."""
import json, os, urllib.request
ENG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = {f["id"]: f for f in json.load(open(f"{ENG}/_v3_corpus/facts_v3.json", encoding="utf-8"))}
# observed cross-picks from the qfix systemecho gate: query_id -> magnet_id it wrongly delivered
CROSS = {"dynamite_inv":"radium_disc","hamlet_author":"radium_disc","starry_night":"radium_disc",
         "david_sculptor":"radium_disc","evolution_theory":"insulin_disc","first_element":"fastest_bird",
         "telescope_inv":"radium_disc","colosseum_city":"egypt_capital","kenya_capital":"egypt_capital"}
CONTROLS = ["longest_river","iron_symbol","chromosomes","berlin_wall","fuji_country"]  # correct cases
def yesno(fact, query):
    p = (f'Fact: "{fact}"\nQuestion: "{query}"\nDoes the fact directly answer the question? '
         f'Reply with only one word: yes or no.')
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
print("=== CROSS-PICKS: correct should YES, magnet should NO ===")
sep=0
for qid, mid in CROSS.items():
    q=F[qid]["para"]; c=yesno(F[qid]["fact"], q); m=yesno(F[mid]["fact"], q)
    ok = (c=="yes" and m=="no")
    if ok: sep+=1
    print(f"{qid:16} correct={c:4} magnet({mid})={m:4}  -> {'SEPARATED' if ok else 'no'}", flush=True)
print(f"separated (correct=yes & magnet=no): {sep}/{len(CROSS)}")
print("\n=== CONTROLS (correct cases): correct should YES ===")
cy=0
for qid in CONTROLS:
    c=yesno(F[qid]["fact"], F[qid]["para"]); cy += (c=="yes")
    print(f"{qid:16} correct={c}", flush=True)
print(f"controls correct=yes: {cy}/{len(CONTROLS)}")
