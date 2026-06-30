"""G-PERSIST-KV comparator: assert per-turn byte-identity (off vs on) + TTFT telemetry.

Reads _persist_gate_off.json and _persist_gate_on.json. GREEN iff every turn's text is
byte-identical (same sha + same string) between the O(n) re-prefill baseline (off) and the
O(1) suffix-append (on). Also prints the per-turn TTFT so the O(1) speedup is visible:
off TTFT should climb with conversation length; on TTFT should stay ~flat.
Exit 0 = GREEN, 1 = RED (any divergence).
"""
import json, sys
off = json.load(open("_persist_gate_off.json", encoding="utf-8"))
on = json.load(open("_persist_gate_on.json", encoding="utf-8"))
ok = len(off) == len(on)
print(f"turns: off={len(off)} on={len(on)}")
print("turn | off_sha          on_sha           match | off_ttft on_ttft | chars")
for a, b in zip(off, on):
    m = (a["sha"] == b["sha"]) and (a["text"] == b["text"])
    ok = ok and m
    print(f"{a['turn']:>4} | {a['sha']} {b['sha']} {'OK   ' if m else 'DIVERGE'} | "
          f"{a['ttft_ms']:>7} {b['ttft_ms']:>7} | {a['chars']}/{b['chars']}")
off_t = [r["ttft_ms"] for r in off]
on_t = [r["ttft_ms"] for r in on]
print(f"\nTTFT off (O(n) re-prefill): {off_t}")
print(f"TTFT on  (O(1) append)    : {on_t}")
if off_t[0] and on_t[0]:
    print(f"TTFT growth last/first  -> off {off_t[-1]/max(off_t[0],1):.2f}x  |  on {on_t[-1]/max(on_t[0],1):.2f}x")
print(f"\nG-PERSIST-KV: {'GREEN -- byte-identical across all turns' if ok else 'RED -- KV state corruption (divergence)'}")
sys.exit(0 if ok else 1)
