#!/usr/bin/env python3
"""serve_timing.py — client-side phase timing for the served /v1/chat SSE stream.

Measures what the daemon does not log on the 12B lane:
  PREFILL_S  = request-sent -> first delta byte (prompt ingest + first token)
  DECODE     = first delta -> [DONE]; tok/s over the delta count
Usage:
  python serve_timing.py <port> <max_tokens> <json-messages-file> [extra-body-json]
extra-body-json example: {"byteexact": false} or {"raw_logits": true}
Prints one machine-line: TIMING port=.. n=.. prefill_s=.. decode_s=.. decode_tokps=.. text_head=..
"""
import json, sys, time, urllib.request

port = sys.argv[1]
max_tokens = int(sys.argv[2])
msgs = json.load(open(sys.argv[3], encoding="utf-8"))
body = {"messages": msgs, "max_tokens": max_tokens}
if len(sys.argv) > 4:
    body.update(json.load(open(sys.argv[4], encoding="utf-8")))

req = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/chat",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json"},
)
t0 = time.perf_counter()
t_first = None
t_last = None
n = 0
text = []
with urllib.request.urlopen(req, timeout=600) as r:
    for raw in r:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            break
        if payload == "keepalive":
            continue
        try:
            d = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if "delta" in d:
            now = time.perf_counter()
            if t_first is None:
                t_first = now
            t_last = now
            n += 1
            text.append(d["delta"])

if t_first is None:
    print(f"TIMING port={port} n=0 NO_DELTAS total_s={time.perf_counter()-t0:.2f}")
    sys.exit(1)
prefill_s = t_first - t0
decode_s = (t_last - t_first) if t_last else 0.0
tokps = (n - 1) / decode_s if decode_s > 0 else 0.0
head = "".join(text)[:80].replace("\n", "\\n")
print(f"TIMING port={port} n={n} prefill_s={prefill_s:.2f} decode_s={decode_s:.2f} "
      f"decode_tokps={tokps:.3f} text_head={head!r}")
print("FULLTEXT: " + repr("".join(text)))
