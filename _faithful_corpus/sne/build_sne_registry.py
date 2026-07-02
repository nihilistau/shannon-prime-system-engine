"""build_sne_registry.py — turn audited SNE facts into an injectable registry.

Creates eps/sne_NNN/ dirs (ep.l5 minted later by write_ep_l5) and writes
registry_sne.jsonl (name/dir/npos/topic/text/sig_bits) in _faithful_corpus/, so
EPS_BASE (dirname(registry)/eps) matches write_ep_l5 and the daemon's load_registry.

Usage: python build_sne_registry.py <audited.json> <registry_out.jsonl>
"""
import json, os, sys
CORPUS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # _faithful_corpus
WIN_BASE = "D:/F/shannon-prime-repos/shannon-prime-system-engine/_faithful_corpus/eps"
facts = sys.argv[1] if len(sys.argv) > 1 else "sne_facts_audited.json"
regout = sys.argv[2] if len(sys.argv) > 2 else os.path.join(CORPUS, "registry_sne.jsonl")
R = json.load(open(facts, encoding="utf-8"))
epsdir = os.path.join(CORPUS, "eps")
os.makedirs(epsdir, exist_ok=True)
with open(regout, "w", encoding="utf-8") as f:
    for it in R:
        d = os.path.join(epsdir, it["name"]); os.makedirs(d, exist_ok=True)
        row = {"name": it["name"], "dir": f"{WIN_BASE}/{it['name']}",
               "npos": max(6, len(it["fact_text"].split()) + 4), "topic": "sne_override",
               "text": it["fact_text"], "sig_bits": "0" * 64}
        f.write(json.dumps(row) + "\n")
print(f"wrote {len(R)} rows -> {regout}; eps dirs under {epsdir}")
