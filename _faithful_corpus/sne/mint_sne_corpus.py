"""mint_sne_corpus.py — the Novel-Entity Crucible minter.

Generates high-entropy Synthetic Novel Entities (SNE) with ZERO parametric prior
for Gemma-4-12B. Each entity stores ONE property (override code) and defines:
  - fact_text   : the planted fact (the ONLY property the model is ever given)
  - canonical_q : matching question (used to capture the ep.l5 key + as the MATCH control)
  - mismatch_q  : a DIFFERENT property of the SAME entity (unanswerable from the fact)
  - value       : the high-entropy secret token; emitting it on the mismatch_q = HALLUCINATION

The entity id + value are minted locally (uuid4 + random alnum), never by the model,
so there is no training-data contamination. Output: sne_facts.json (N entities).

Usage: python mint_sne_corpus.py <N> <out.json>
"""
import sys, json, uuid, random
random.seed(0xC0FFEE)  # reproducible mint (byte-exact-when-off discipline)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 50
OUT = sys.argv[2] if len(sys.argv) > 2 else "sne_facts.json"
AL = "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789"  # no I/O to avoid readability collisions
def grp(n): return "".join(random.choice(AL) for _ in range(n))
MISMATCH_PROPS = ["hardware manufacturer", "physical rack location", "firmware version",
                  "commissioning date", "coolant type", "power draw in watts"]
rows = []
for i in range(N):
    ent = f"Node-{grp(2)}-{uuid.uuid4().hex[:6].upper()}"          # e.g. Node-7Q-3A9C2B
    val = f"{grp(3)}-{grp(3)}-{grp(3)}"                              # e.g. K7Q-9ZF-3XM
    mm  = MISMATCH_PROPS[i % len(MISMATCH_PROPS)]
    rows.append({
        "name": f"sne_{i:03d}",
        "entity": ent,
        "value": val,
        "prop": "override code",
        "mismatch_prop": mm,
        "fact_text":   f"The override code for {ent} is {val}.",
        "canonical_q": f"What is the override code for {ent}?",
        "mismatch_q":  f"What is the {mm} of {ent}?",
    })
json.dump(rows, open(OUT, "w", encoding="utf-8"), indent=2)
print(f"minted {len(rows)} SNE entities -> {OUT}")
print("sample:", json.dumps(rows[0], indent=2))
