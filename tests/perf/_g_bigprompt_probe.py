"""Isolate the gateway stall: does the DAEMON itself wedge on a big (>RING_W) prompt, no harness?
Sends a system prompt of ~SP_WORDS filler tokens straight to :3000 with max_tokens=40 + eot_bias.
If a big prompt stalls but a small one completes, the bug is daemon-side (ring/persist), not harness."""
import itertools, json, os, sys, time, urllib.request
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
PORT = int(os.environ.get("SP_TEST_PORT", "3000"))
WORDS = int(os.environ.get("SP_WORDS", "1300"))

vocab = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima".split()
filler = " ".join(itertools.islice(itertools.cycle(vocab), WORDS))
sysprompt = "You are a helpful assistant. Ignore this reference context: " + filler
body = {"messages": [{"role": "system", "content": sysprompt},
                     {"role": "user", "content": "Say hello in exactly one word."}],
        "max_tokens": int(os.environ.get("SP_MAXTOK", "40")), "temperature": 0, "eot_bias": 4.0,
        "byteexact": os.environ.get("SP_BX", "1") == "1"}

t0 = time.time()
chars = 0
done = False
req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=240) as r:
        for raw in r:
            s = raw.decode("utf-8", "replace").strip()
            if s.startswith("data:"):
                p = s[5:].strip()
                if p == "[DONE]":
                    done = True
                    break
                try:
                    chars += len(json.loads(p).get("delta", ""))
                except Exception:
                    pass
    print(f"WORDS={WORDS}  time={time.time()-t0:.1f}s  out_chars={chars}  got_DONE={done}")
except Exception as exc:
    print(f"WORDS={WORDS}  time={time.time()-t0:.1f}s  ERROR={type(exc).__name__}")
