"""sse_gate.py — G-LM-SSE. Subscribe to /v1/events, fire recall queries, confirm the engine
BROADCASTS telemetry records live (event: telemetry) — the SSE sink the harness StreamProcessor
will consume. Also confirms secrets stay redacted on the wire."""
import json, threading, time, urllib.request

events = []
def subscribe():
    r = urllib.request.Request("http://127.0.0.1:3000/v1/events")
    try:
        with urllib.request.urlopen(r, timeout=25) as resp:
            ev = None
            for raw in resp:
                s = raw.decode("utf-8", "replace").rstrip("\n")
                if s.startswith("event:"): ev = s[6:].strip()
                elif s.startswith("data:"):
                    data = s[5:].strip()
                    if ev == "telemetry":
                        events.append(data)
                elif s == "" : ev = None
    except Exception as e:
        print("subscriber closed:", e)

def ask(q):
    b = json.dumps({"messages":[{"role":"user","content":q}],"max_tokens":40,
                    "temperature":0,"eot_bias":4.0,"auto_recall":True}).encode()
    r = urllib.request.Request("http://127.0.0.1:3000/v1/chat", data=b, headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(r, timeout=180) as resp:
        for raw in resp:
            if raw.decode("utf-8","replace").strip().startswith("data:") and "[DONE]" in raw.decode("utf-8","replace"): break
    return True

t = threading.Thread(target=subscribe, daemon=True); t.start()
time.sleep(1.5)  # let the SSE subscription establish
ask("What is the chemical symbol for gold now?")
ask("What is the recovery phrase for the Meridian vault archive?")
time.sleep(3.0)  # drain

print(f"\ntelemetry events received on /v1/events: {len(events)}")
kinds = {"decision":0, "turn":0}
secret_leak = 0
for e in events:
    try: o = json.loads(e)
    except: continue
    if "recall" in o: kinds["decision"] += 1
    if o.get("kind") == "turn": kinds["turn"] += 1
    if "orchid" in e.lower(): secret_leak += 1
    print("  ", e[:110])
print(f"\nby kind: {kinds}")
print(f"REDACTION on the wire: secret 'orchid' in SSE payloads = {secret_leak} -> {'PASS' if secret_leak==0 else 'FAIL'}")
ok = len(events) >= 2 and kinds['decision']>=1 and kinds['turn']>=1 and secret_leak==0
print(f"RESULT sse: {'PASS' if ok else 'FAIL'} (live telemetry stream + redacted)")
