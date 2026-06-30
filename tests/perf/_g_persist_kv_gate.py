"""G-PERSIST-KV -- multi-turn byte-identity + O(1) telemetry gate.

Extends _g_persist_kv_parity.py (2-turn seed) to an N-turn scripted conversation.
Each turn appends to the growing history (OpenAI-style: client resends the full
conversation). With SP_PERSIST_KV=1 the daemon reuses the committed cache and
prefills only the new suffix (O(1) append); with it off it resets + re-prefills the
whole conversation (O(n)). At temperature 0 + byteexact the decode is deterministic,
so EVERY turn's answer MUST be byte-identical on vs off -- the corruption kill-test.
TTFT per turn is the O(1) evidence: off grows with conversation length, on stays flat.

Run once per daemon: `python _g_persist_kv_gate.py <on|off>` -> _persist_gate_<tag>.json
Then `python _persist_gate_compare.py` asserts identity + prints the TTFT table.
"""
import json, os, sys, time, hashlib, urllib.request
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
PORT = int(os.environ.get("SP_TEST_PORT", "3001"))


def chat(msgs, max_tokens=64):
    body = json.dumps({"messages": msgs, "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; out = []
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            s = raw.decode("utf-8", "replace").strip()
            if not s.startswith("data:"):
                continue
            p = s[5:].strip()
            if p == "[DONE]":
                break
            try:
                d = json.loads(p)
                if d.get("delta"):
                    if ttft is None:
                        ttft = time.time() - t0
                    out.append(d["delta"])
            except Exception:
                pass
    total = time.time() - t0
    return "".join(out), (ttft if ttft is not None else total), total


tag = sys.argv[1] if len(sys.argv) > 1 else "x"
TURNS = [
    "Tell me one short fact about the planet Saturn.",
    "Now one short fact about the planet Mars.",
    "And one short fact about the planet Jupiter.",
    "Now one short fact about the planet Venus.",
    "Finally, one short fact about the planet Neptune.",
    "Which of those five planets did we discuss first?",
]
msgs, rows = [], []
for i, u in enumerate(TURNS):
    msgs.append({"role": "user", "content": u})
    txt, ttft, total = chat(msgs)
    msgs.append({"role": "assistant", "content": txt})
    sha = hashlib.sha256(txt.encode("utf-8")).hexdigest()[:16]
    rows.append({"turn": i + 1, "chars": len(txt), "sha": sha,
                 "ttft_ms": round(ttft * 1000), "total_ms": round(total * 1000), "text": txt})
    print(f"[{tag}] turn {i+1}: chars={len(txt)} sha={sha} ttft={ttft*1000:.0f}ms total={total*1000:.0f}ms "
          f"| {' '.join(txt.split())[:70]}", flush=True)
json.dump(rows, open(f"_persist_gate_{tag}.json", "w", encoding="utf-8"), ensure_ascii=False)
print(f"[{tag}] WROTE _persist_gate_{tag}.json ({len(rows)} turns)", flush=True)
