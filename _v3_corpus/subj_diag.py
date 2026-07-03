"""subj_diag.py — does a cheap deterministic SUBJECT-TOKEN overlap between the query
and the fact separate the CORRECT fact from the MAGNET fact? If yes, a subject re-weight
(no micro-forward) fixes the buried-correct cross-picks.

For each crosspick: overlap(query, correct_fact) vs overlap(query, magnet_fact=top1).
Also check the whole set: would a 'keep only episodes whose fact shares a salient query
token' FILTER retain the correct episode for the 22 already-correct (no regressions)?
"""
import json, re, os

ENG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = json.load(open(f"{ENG}/_v3_corpus/facts_v3.json", encoding="utf-8"))
reg = [json.loads(l) for l in open(f"{ENG}/_v3_corpus/registry.jsonl", encoding="utf-8") if l.strip()]
names = [r["name"] for r in reg]
texts = [r["text"] for r in reg]
name2text = dict(zip(names, texts))

STOP = set("the a an of to in on at for and or is are was were be been being now has have "
           "had new latest recent revised as its by with from that this it user said according "
           "survey records show measurements confirm under standard official officially known "
           "world its into been recognized relocated moved been first who what which where when "
           "name is are do does his her their our your my we they he she you i".split())
def toks(s):
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP and len(w) > 2]
def salient_overlap(q, fact):
    qs, fs = set(toks(q)), set(toks(fact))
    return len(qs & fs), sorted(qs & fs)

crosspick = ["dynamite_inv","hamlet_author","starry_night","radium_disc",
             "david_sculptor","evolution_theory","first_element","telescope_inv"]
# magnet top1 episode names from the rank dump (last run)
top1 = {"dynamite_inv":"ep_live_m1783039367579","hamlet_author":"ep_live_m1783039125927",
        "starry_night":"ep_live_m1783039137070","radium_disc":"ep_live_m1783039367579",
        "david_sculptor":"ep_live_m1783039367579","evolution_theory":"ep_live_m1783039367579",
        "first_element":"ep_live_m1783039260541","telescope_inv":"ep_live_m1783039367579"}

print("=== CROSSPICK: salient overlap(query, correct) vs (query, magnet) ===")
sep_ok = 0
for cid in crosspick:
    i = next(k for k,f in enumerate(F) if f["id"]==cid)
    q = F[i]["para"]; correct_txt = texts[i]; magnet_txt = name2text[top1[cid]]
    oc, wc = salient_overlap(q, correct_txt)
    om, wm = salient_overlap(q, magnet_txt)
    win = "CORRECT>magnet" if oc > om else ("tie" if oc==om else "MAGNET>correct")
    if oc > om: sep_ok += 1
    print(f"{cid:16} q_overlap correct={oc}{wc} magnet={om}{wm}  -> {win}")
print(f"\nseparated by subject-overlap: {sep_ok}/8")

print("\n=== REGRESSION CHECK: for the 22 correct, does the correct fact share a salient token with its query? ===")
reg_ok = 0; reg_bad = []
for i,f in enumerate(F):
    if f["id"] in crosspick: continue
    o,w = salient_overlap(f["para"], texts[i])
    if o>0: reg_ok += 1
    else: reg_bad.append((f["id"], f["para"][:40]))
print(f"correct-fact shares >=1 salient query token: {reg_ok}/22")
if reg_bad:
    print("  WOULD-REGRESS (no shared salient token, a hard filter would drop these):")
    for cid,p in reg_bad: print(f"    {cid}: {p!r}")
