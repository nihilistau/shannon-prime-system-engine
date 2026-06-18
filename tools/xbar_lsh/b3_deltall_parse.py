#!/usr/bin/env python3
"""Parse the DELTALL run -> per-token query deflection matrix + the v4 verdict.
dLL_E (per token) = (LL(query|E) - LL(query|0)) / n_query  = (nll0 - nll_E)/n.
Relevant E raises query likelihood (NLL drops) -> dLL > 0. Foreign E -> dLL ~ 0 / negative.
"""
import os, json, re
ENG = r"D:\F\shannon-prime-repos\shannon-prime-system-engine"
OUT = os.path.join(ENG, "_b3_wc", "deltall")
meta = json.load(open(os.path.join(OUT, "meta.json")))
# parse DELTALL <tokfile> nll=<> n=<>
nll = {}
for line in open(os.path.join(ENG, "_b3dl.out"), encoding="utf-8", errors="ignore"):
    m = re.search(r"DELTALL\s+(\S+)\s+nll=([-\d.]+)\s+n=(\d+)", line)
    if m:
        nll[os.path.basename(m.group(1))] = (float(m.group(2)), int(m.group(3)))
# index meta by (qi, ep)
byq = {}
for e in meta:
    byq.setdefault(e["qi"], {})[e["ep"]] = e
EPS = ["ep_wiki", "ep_homarus", "ep_headlam"]
names = {"ep_wiki": "Boulter", "ep_homarus": "lobster", "ep_headlam": "Headlam"}
print(f"{'query':52} {'->Boulter':>10}{'->lobster':>10}{'->Headlam':>10}   argmax/want   verdict")
tpos, fmax = [], []
log = ["G-CHAT-B3-RECALL-v4  query-deflection (dLL per token, E as text prefix, 12B one-shot SP_G4_SCORE)",
       "dLL = (NLL(query|0) - NLL(query|E)) / n_query ; relevant E -> dLL>0 (query more predictable)"]
for qi in sorted(byq):
    row = byq[qi]; q = row[EPS[0]]["query"]; lab = row[EPS[0]]["label"]
    base = nll.get(row["none"]["tokfile"]) if "none" in row else None
    if base is None:
        # 'none' meta entry has ep='none'
        base = nll.get(byq[qi]["none"]["tokfile"])
    n0, _ = base
    dlls = {}
    for ep in EPS:
        nE, n = nll[row[ep]["tokfile"]]
        dlls[ep] = (n0 - nE) / max(1, n)        # per-token LL gain
    arg = max(EPS, key=lambda e: dlls[e]); mx = dlls[arg]
    is_pos = lab in EPS
    if is_pos:
        hit = (arg == lab); tpos.append(dlls[lab])
        verd = f"want={names[lab]} {'OK' if hit else 'WRONG('+names[arg]+')'}"
    else:
        fmax.append(mx); verd = f"FOREIGN max={names[arg]} ({lab})"
    disp = "".join(f"{dlls[e]:+10.3f}" for e in EPS)
    line = f"{q[:52]:52}{disp}   {verd}"
    print(line); log.append(line)
mt = min(tpos) if tpos else float('nan'); mf = max(fmax) if fmax else float('nan')
sep = mt > mf
verdict = (f"\nmin true-positive dLL = {mt:+.3f}   max foreign dLL = {mf:+.3f}   => "
           f"{'SEPARATES (GREEN)' if sep else 'NO SEPARATION (RED)'}")
print(verdict); log.append(verdict)
open(os.path.join(ENG, "tests", "fixtures", "chat_fullstack", "G-CHAT-B3-RECALL-v4.log"),
     "w", encoding="utf-8").write("\n".join(log) + "\n")
