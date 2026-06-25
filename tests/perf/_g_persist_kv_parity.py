"""G-PERSIST-KV-PARITY -- the persistent O(1) conversation KV must be byte-exact.

Two-turn conversation against the daemon (:3001). Turn 2's prompt extends turn 1
by appending, so with SP_PERSIST_KV=1 the daemon REUSES the turn-1 cache and
prefills only the new suffix; with it off it resets and re-prefills the whole
conversation. At temperature 0 + byteexact (default on) the decode is deterministic,
so the turn-2 answer MUST be identical on/off -- that is the byte-exact proof.
Run once per daemon (tag 'on' / 'off'); compare the two _persist_r2_<tag>.txt files.
"""
import json, os, sys, urllib.request
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
PORT = int(os.environ.get("SP_TEST_PORT", "3001"))


def chat(msgs, max_tokens=80):
    body = json.dumps({"messages": msgs, "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    out = []
    with urllib.request.urlopen(req, timeout=300) as r:
        for raw in r:
            s = raw.decode("utf-8", "replace").strip()
            if not s.startswith("data:"):
                continue
            p = s[5:].strip()
            if p == "[DONE]":
                break
            try:
                d = json.loads(p)
                if "delta" in d:
                    out.append(d["delta"])
            except Exception:
                pass
    return "".join(out)


tag = sys.argv[1] if len(sys.argv) > 1 else "x"
t1 = "Tell me one short fact about the planet Saturn."
t2 = "Now tell me one short fact about the planet Mars."

r1 = chat([{"role": "user", "content": t1}])
print("R1:", " ".join(r1.split())[:160], flush=True)
msgs = [{"role": "user", "content": t1},
        {"role": "assistant", "content": r1},
        {"role": "user", "content": t2}]
r2 = chat(msgs)
print("R2:", " ".join(r2.split())[:160], flush=True)

with open(f"_persist_r2_{tag}.txt", "w", encoding="utf-8") as f:
    f.write(r2)
print(f"WROTE _persist_r2_{tag}.txt ({len(r2)} chars)", flush=True)
