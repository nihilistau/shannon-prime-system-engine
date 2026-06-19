#!/usr/bin/env python3
"""Evaluate the B3-v6 semantic-discriminator run. margin(q,ep) = NLL(No)-NLL(Yes) =
logP(Yes)-logP(No). Per query: the episode with the highest margin is the model's pick;
accept iff that margin > TAU (the 12B judges 'Yes, this memory answers the query')."""
import os, json, re, sys
ENG = r"D:\F\shannon-prime-repos\shannon-prime-system-engine"
OUT = os.path.join(ENG,"_b3_wc","reason")
TAU = float(os.environ.get("B3_YESNO_TAU","0.0"))
EPS=["ep_wiki","ep_homarus","ep_headlam"]; NM={"ep_wiki":"Boul","ep_homarus":"lob","ep_headlam":"Head"}
meta=json.load(open(os.path.join(OUT,"meta.json")))
nll={}
for line in open(os.path.join(ENG,"_b3rz.out"),encoding="utf-8",errors="ignore"):
    m=re.search(r"DELTALL\s+(\S+)\s+nll=([-\d.]+)\s+n=(\d+)",line)
    if m: nll[os.path.basename(m.group(1))]=float(m.group(2))/max(1,int(m.group(3)))  # per-token NLL
# index: (qi,ep,ans)->nll
by={}
for e in meta: by.setdefault((e["qi"],e["ep"]),{})[e["ans"]]=nll.get(e["tokfile"])
qmeta={e["qi"]:(e["query"],e["label"]) for e in meta}
log=[f"G-CHAT-B3-RECALL-v6 SEMANTIC DISCRIMINATOR  margin = logP(Yes)-logP(No) = NLL(No)-NLL(Yes), TAU={TAU}",
     f"{'query':50} {'Boul':>7}{'lob':>7}{'Head':>7}  pick   margin   verdict"]
print("\n".join(log)); ok_tp=ok_fgn=True
for qi in sorted(qmeta):
    q,lab=qmeta[qi]; marg={}
    for ep in EPS:
        d=by.get((qi,ep),{}); y=d.get("Yes"); n=d.get("No")
        marg[ep]=(n-y) if (y is not None and n is not None) else -99
    pick=max(EPS,key=lambda e:marg[e]); mx=marg[pick]; accept=mx>TAU
    if lab in EPS:
        good=(accept and pick==lab); ok_tp&=good; tag=f"want={NM[lab]} {'OK' if good else 'MISS'}"
    else:
        good=(not accept); ok_fgn&=good; tag=f"FOREIGN {'OK-rejected' if good else 'LEAK!'}"
    disp="".join(f"{marg[e]:+7.2f}" for e in EPS)
    line=f"  {q[:50]:50}{disp}  {NM[pick]:>5} {mx:+7.2f}  {'ACCEPT' if accept else 'reject':7} {tag}"
    print(line); log.append(line)
v=(f"\nSEMANTIC SIEVE: true-positives accepted={ok_tp}  foreigns all-rejected={ok_fgn}  => "
   f"{'GREEN — board sealed' if (ok_tp and ok_fgn) else 'leak remains'}")
print(v); log.append(v)
open(os.path.join(ENG,"tests","fixtures","chat_fullstack","G-CHAT-B3-RECALL-v6-semantic.log"),
     "w",encoding="utf-8").write("\n".join(log)+"\n")
