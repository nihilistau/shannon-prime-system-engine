"""telemetry_report.py — summarize the LM-B2 telemetry-okf decision log (SP_TELEMETRY_LOG).
The tuning + drift + redaction view: per-class/decision counts, delivery modes, cos/margin
stats, and a redaction check. Usage: python telemetry_report.py <log.jsonl> [secret-to-grep]"""
import json, sys, statistics
path = sys.argv[1]
secret = sys.argv[2] if len(sys.argv) > 2 else None
rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
print(f"telemetry records: {len(rows)}\n")
from collections import Counter
by_class = Counter(); by_decision = Counter(); by_delivery = Counter()
cos_by_decision = {}; margins = []
redacted = 0
for r in rows:
    rc = r.get("recall", {})
    by_class[rc.get("class","-")] += 1
    by_decision[rc.get("decision","-")] += 1
    if rc.get("fired"): by_delivery[rc.get("delivery","-")] += 1
    cos_by_decision.setdefault(rc.get("decision","-"), []).append(rc.get("cos",0.0))
    m = rc.get("margin",-1.0)
    if m >= 0: margins.append(m)
    if r.get("redacted"): redacted += 1
print("by class:    ", dict(by_class))
print("by decision: ", dict(by_decision))
print("by delivery: ", dict(by_delivery))
for d, cs in cos_by_decision.items():
    print(f"  cos[{d}]: n={len(cs)} mean={statistics.mean(cs):.3f} min={min(cs):.3f} max={max(cs):.3f}")
if margins:
    print(f"  margin: n={len(margins)} mean={statistics.mean(margins):.4f} min={min(margins):.4f}")
print(f"redacted records: {redacted}")
if secret:
    hits = sum(1 for l in open(path, encoding='utf-8') if secret in l)
    print(f"\nREDACTION CHECK: secret {secret!r} appears in the log {hits} time(s)  "
          f"-> {'PASS (0 hits)' if hits == 0 else 'FAIL (LEAK IN TELEMETRY!)'}")
