#!/usr/bin/env python3
# Post-capture: (1) patch real npos into registry.jsonl from each ep.tok line count,
# (2) write each episode's exact secret string into ep.secret (the per-episode knockout
#     target the disposer reads when SP_B3_SECRET is unset — one-pass admission labeling).
# The secret comes from corpus_manifest.jsonl (NOT needle.txt, which is the full sentence).
# usage: python patch_npos.py <registry.jsonl> <eps-root> [corpus_manifest.jsonl]
import json, os, sys
reg_path, eps_root = sys.argv[1], sys.argv[2]
manifest_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(reg_path), "corpus_manifest.jsonl")

# id -> secret from the manifest
secret_by_id = {}
if os.path.exists(manifest_path):
    for ln in open(manifest_path, encoding="utf-8"):
        if not ln.strip():
            continue
        r = json.loads(ln)
        secret_by_id[r["id"]] = r["secret"]
else:
    print(f"  WARN manifest not found: {manifest_path} (ep.secret not written)", file=sys.stderr)

rows = [json.loads(l) for l in open(reg_path, encoding="utf-8") if l.strip()]
for r in rows:
    name = r["name"]                       # ep_<id>
    epdir = os.path.join(eps_root, name)
    if not os.path.isdir(epdir):
        epdir = r["dir"]
    tok = os.path.join(epdir, "ep.tok")
    if os.path.exists(tok):
        r["npos"] = sum(1 for ln in open(tok, encoding="utf-8") if ln.strip())
    else:
        print(f"  WARN missing ep.tok for {name} ({tok})", file=sys.stderr)
    # write ep.secret (preserve leading space; no trailing newline)
    nid = name[3:] if name.startswith("ep_") else name
    sec = secret_by_id.get(nid)
    if sec is not None and os.path.isdir(epdir):
        with open(os.path.join(epdir, "ep.secret"), "w", encoding="utf-8", newline="") as f:
            f.write(sec)            # exact secret, leading space intact, no newline

with open(reg_path, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"patched npos + ep.secret for {len(rows)} episodes -> {reg_path}")
for r in rows:
    nid = r["name"][3:] if r["name"].startswith("ep_") else r["name"]
    print(f"  {r['name']:26s} npos={r['npos']:<4} secret={secret_by_id.get(nid)!r}")
