"""G-RING-WRAP (semantic): needle-in-a-haystack past the SWA window, via the LIVE daemon.

Floods a synthetic haystack (deterministic distinct filler) well beyond the SWA ring_W so the
40 SWA layers physically wrap past the needle; the needle sits near the START (depth%), so its
only surviving carrier is the 8 GLOBAL full-cache layers. If the model recovers the secret, the
global layers retained the distant past across the wrap (structural integrity is inherited from
G-1b-WRAP-NULL; this characterizes the model's long-range retention through the ring).

Usage: python _g_niah_daemon.py <ring_W> [haystack_tokens_approx] [depth_pct]
Honest: with only 8 global layers, deep-range single-needle recall is gemma-limited; a MISS at
extreme range is the MODEL, not the ring. Reports HIT/MISS + the needle's distance past the window.
"""
import json, os, sys, time, urllib.request
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
PORT = int(os.environ.get("SP_TEST_PORT", "3001"))
SECRET = "837492"
ring_w = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
approx_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 4500
depth_pct = int(sys.argv[3]) if len(sys.argv) > 3 else 8

# ~12 tokens/line; build enough lines to hit approx_tokens. Distinct, non-repeating filler so the
# model can't pattern-complete the answer from local context.
n_lines = max(40, approx_tokens // 12)
needle_line = max(1, int(n_lines * depth_pct / 100))
lines = []
for i in range(n_lines):
    if i == needle_line:
        lines.append(f"Note {i}: the secret vault access code is {SECRET}, remember it.")
    else:
        lines.append(f"Note {i}: warehouse shelf {i} holds {((i*7919) % 900) + 100} blue widgets.")
haystack = "\n".join(lines)
question = ("\n\nBased only on the notes above, what is the secret vault access code? "
           "Answer with just the number.")
approx_line_tok = 12
needle_tok = needle_line * approx_line_tok
total_tok = n_lines * approx_line_tok
dist_past_window = total_tok - needle_tok - ring_w  # >0 ⇒ needle is outside the SWA window


def chat(msgs, max_tokens=24):
    body = json.dumps({"messages": msgs, "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); out = []
    with urllib.request.urlopen(req, timeout=900) as r:
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
                    out.append(d["delta"])
            except Exception:
                pass
    return "".join(out), time.time() - t0


print(f"[niah] ring_W={ring_w} ~total_tok={total_tok} needle@~tok{needle_tok} (line {needle_line}/{n_lines}, depth {depth_pct}%) "
      f"dist_past_window={dist_past_window} (>0 => needle OUTSIDE SWA window, must use globals)", flush=True)
ans, dt = chat([{"role": "user", "content": haystack + question}])
hit = SECRET in ans
print(f"[niah] answer ({dt:.1f}s): {ans.strip()[:80]!r}", flush=True)
print(f"G-RING-WRAP-NIAH: {'HIT (global layers retained the needle past the wrap)' if hit else 'MISS (no retrieval at this range)'}", flush=True)
sys.exit(0 if hit else 3)
