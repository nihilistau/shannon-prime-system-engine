"""Per-turn chat timing -- diagnose WHERE a turn spends time.
Reports: total seconds, time-to-first-token (ttft = prefill cost), decode tok/s.
A high ttft that grows with the conversation = prefill not being reused (persist off / window slide).
A low tok/s = decode bound (model + KV attention). SP_TEST_PORT selects the daemon (default 3000)."""
import json, os, sys, time, urllib.request
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
PORT = int(os.environ.get("SP_TEST_PORT", "3000"))


def chat(msgs, max_tokens=120):
    body = json.dumps({"messages": msgs, "max_tokens": max_tokens, "temperature": 0,
                       "eot_bias": 4.0}).encode()  # mirror the console (clean stop)
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    out = []
    t0 = time.time()
    t_first = None
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
                    if t_first is None:
                        t_first = time.time()
                    out.append(d["delta"])
            except Exception:
                pass
    dt = time.time() - t0
    txt = "".join(out)
    ntok = max(1, len(txt) // 4)
    ttft = (t_first - t0) if t_first else dt
    decode_s = max(0.05, dt - ttft)
    return txt, dt, ttft, ntok, ntok / decode_s


turns = [
    "What is your name?",
    "Tell me one short fact about the ocean.",
    "Now one short fact about mountains.",
    "And one about rivers.",
    "In one sentence, what did we just talk about?",
]
SYS = ("You are Shannon-Prime, an experimental AI that runs entirely locally on a single RTX 2060. "
       "You have a real working memory and can call tools. Keep replies short and direct -- usually "
       "one or two sentences. Use facts the user told you faithfully; if you don't know, say so.")
hist = [{"role": "system", "content": SYS}]
for i, u in enumerate(turns):
    hist.append({"role": "user", "content": u})
    txt, dt, ttft, ntok, toks = chat(hist)
    hist.append({"role": "assistant", "content": txt})
    print(f"T{i+1}: total={dt:5.1f}s  ttft={ttft:4.1f}s  ~{ntok:3d}tok  {toks:4.1f}tok/s | "
          f"{' '.join(txt.split())[:88]}", flush=True)
print("DONE", flush=True)
