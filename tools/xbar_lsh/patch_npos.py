#!/usr/bin/env python3
# Patch real npos into a registry.jsonl from each episode's ep.tok line count.
# usage: python patch_npos.py <registry.jsonl> <eps-root>
import json, os, sys
reg_path, eps_root = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in open(reg_path, encoding="utf-8") if l.strip()]
for r in rows:
    epdir = r["dir"]
    # tolerate registry written with the canonical eps_root; recompute from name under eps_root
    tok = os.path.join(eps_root, r["name"], "ep.tok")
    if not os.path.exists(tok):
        tok = os.path.join(epdir, "ep.tok")
    if os.path.exists(tok):
        n = sum(1 for ln in open(tok, encoding="utf-8") if ln.strip())
        r["npos"] = n
    else:
        print(f"  WARN missing ep.tok for {r['name']} ({tok})", file=sys.stderr)
with open(reg_path, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"patched npos for {len(rows)} episodes -> {reg_path}")
for r in rows:
    print(f"  {r['name']:24s} npos={r['npos']}")
