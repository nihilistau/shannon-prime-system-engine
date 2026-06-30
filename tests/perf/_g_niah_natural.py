"""G-OKV-P3-EVICT router grade — NIAH past the slab with a NATURAL-PROSE haystack.

Uses tests/fixtures/ppl/wiki.valid.raw (real Wikipedia text) so the recent window is varied English
(no repetitive-template degeneration confound). Needle injected near the START (depth%, far/evicted,
beyond recent_w) -> the ONLY way to answer is the LSH router paging the evicted block from host RAM.
A MISS here, with a fluent (non-degenerate) output, is a clean verdict that R.bin needs recalibration
for the gemma4-12b global-K regime.

Usage: python _g_niah_natural.py [approx_tokens] [depth_pct]
"""
import json, os, sys, time, urllib.request
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
PORT = int(os.environ.get("SP_TEST_PORT", "3001"))
SECRET = "837492"
approx_tokens = int(sys.argv[1]) if len(sys.argv) > 1 else 2800
depth_pct = int(sys.argv[2]) if len(sys.argv) > 2 else 7
RAW = os.path.join(os.path.dirname(__file__), "..", "fixtures", "ppl", "wiki.valid.raw")

text = open(RAW, encoding="utf-8", errors="replace").read()
chars = approx_tokens * 4                      # ~4 chars/token English
hay = text[:chars]
ins = int(len(hay) * depth_pct / 100)
ins = hay.rfind(". ", 0, ins)                  # snap to a sentence boundary
if ins < 0: ins = int(len(hay) * depth_pct / 100)
needle = " The secret vault access code is 837492, remember it. "
hay = hay[:ins] + needle + hay[ins:]
prompt = hay + "\n\nBased only on the text above, what is the secret vault access code? Answer with just the number."


def chat(msg, max_tokens=24):
    body = json.dumps({"messages": [{"role": "user", "content": msg}], "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat", data=body, headers={"Content-Type": "application/json"})
    t0 = time.time(); out = []
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            s = raw.decode("utf-8", "replace").strip()
            if not s.startswith("data:"): continue
            p = s[5:].strip()
            if p == "[DONE]": break
            try:
                d = json.loads(p)
                if d.get("delta"): out.append(d["delta"])
            except Exception: pass
    return "".join(out), time.time() - t0


print(f"[niah-nat] wiki prose haystack ~{approx_tokens} tok, needle@~char{ins} (depth {depth_pct}%, far/evicted)", flush=True)
ans, dt = chat(prompt)
a = ans.strip()
hit = SECRET in a
degenerate = (a == "" or set(a.replace(" ", "")) <= set("`") or len(set(a.split())) <= 1)
print(f"[niah-nat] answer ({dt:.1f}s): {a[:90]!r}", flush=True)
print(f"[niah-nat] fluent={'NO (degenerate)' if degenerate else 'YES'} | needle={'HIT' if hit else 'MISS'}", flush=True)
print(f"VERDICT: {'ROUTER OK (HIT, fluent)' if hit else ('ROUTER MISS but FLUENT => R.bin recalibration' if not degenerate else 'INCONCLUSIVE (still degenerate)')}", flush=True)
sys.exit(0 if hit else 3)
