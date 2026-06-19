#!/usr/bin/env python3
"""B3 THREE-LAYER SIEVE (G-CHAT-B3-RECALL-v5). Accept a memory only if it passes ALL:
  L1 CONSENSUS  : W_c contrastive (v3) argmax == query-dLL thermodynamic (v4) argmax
                  (disagreement = different blind spots fooled = reject; kills lexical traps)
  L2 FLOOR      : dLL[argmax] > TAU  (a negative/near-zero dLL proves the memory HINDERED,
                  not helped; kills irrelevant default-attractors like "capital of France")
  L3 BRIDGE     : the dLL is measured on query + a forced semantic bridge ("...and therefore
                  the answer is that") so a content-rich-but-irrelevant prefix can't win on
                  syntax alone; kills the stop-word fluent-prefix confound.
W_c scores: int_relevance on the mined Q. dLL: deltall_dll.json (fresh, bridge-appended run).
"""
import os, json, numpy as np, sys, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b3_export_wc import int_relevance, quantize

ENG = r"D:\F\shannon-prime-repos\shannon-prime-system-engine"
TAU = float(os.environ.get("B3_FLOOR", "0.05"))
EPS = ["ep_wiki", "ep_homarus", "ep_headlam"]; NM = {"ep_wiki":"Boul","ep_homarus":"lob","ep_headlam":"Head"}
LABEL = {"Who is Robert Boulter?":"ep_wiki",
         "What is the European lobster, Homarus gammarus?":"ep_homarus",
         "Who was Frank Headlam?":"ep_headlam"}

def norm(s): return re.sub(r"\s+", " ", s).strip().lower()

dll = json.load(open(os.path.join(ENG,"_b3_wc","deltall","deltall_dll.json")))
dll_n = {norm(k): v for k, v in dll.items()}
z = np.load(os.path.join(ENG,"_b3_wc","lsh_Wc_f32.npz"), allow_pickle=True)
Wi,_ = quantize(z["Wc"].astype(np.float32),16)
d = np.load(os.path.join(ENG,"_b3_wc","b3_data.npz"), allow_pickle=True)
Q=[np.asarray(q,np.float32) for q in d["Q"]]; K=[np.asarray(k,np.float32) for k in d["K"]]
qidx={norm(t):i for i,t in enumerate(list(d["texts"]))}

log=[f"G-CHAT-B3-RECALL-v5 THREE-LAYER SIEVE  L1 consensus(W_c==dLL argmax) + L2 floor(dLL>{TAU}) + L3 bridge",
     f"{'query':50} {'W_c':>5} {'dLL_arg':>8} {'dLL_val':>8}  L1  L2   decision"]
print("\n".join(log))
ok_tp=ok_fgn=True
for q in dll:
    nq=norm(q); i=qidx.get(nq)
    if i is None: print(f"  [skip no Q] {q[:55]}"); continue
    wc=[float(int_relevance(Q[i],K[e],Wi,10)[1]) for e in range(3)]
    wc_arg=int(np.argmax(wc)); v=dll_n[nq]; dl_arg=int(np.argmax(v)); dl_val=v[dl_arg]
    L1 = (wc_arg==dl_arg); L2 = (dl_val > TAU)
    accept = L1 and L2
    lab=LABEL.get(q)
    if lab:
        good=(accept and EPS[dl_arg]==lab); ok_tp&=good
        tag=f"want={NM[lab]} {'OK' if good else 'MISS'}"
    else:
        good=(not accept); ok_fgn&=good
        tag=f"FOREIGN {'OK-rejected' if good else 'LEAK!'}"
    line=f"  {q[:50]:50} {NM[EPS[wc_arg]]:>5} {NM[EPS[dl_arg]]:>8} {dl_val:+8.3f}  {int(L1)}   {int(L2)}   {'ACCEPT' if accept else 'reject':7} {tag}"
    print(line); log.append(line)
verdict=(f"\nSIEVE: true-positives accepted={ok_tp}  foreigns all-rejected={ok_fgn}  => "
         f"{'GREEN — board clean' if (ok_tp and ok_fgn) else 'leak remains'}")
print(verdict); log.append(verdict)
open(os.path.join(ENG,"tests","fixtures","chat_fullstack","G-CHAT-B3-RECALL-v5-sieve.log"),
     "w",encoding="utf-8").write("\n".join(log)+"\n")
