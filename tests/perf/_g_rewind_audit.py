"""G-PERSIST-KV-REWIND audit: is the LCP rewind byte-exact under byteexact-ON?

Protocol (one fresh daemon, global KV_COMMITTED cache, sequential):
  R1 cold   [prefix + msgA]            -> full prefill (empty cache)      -> O1
  R2 warm   [prefix + dummy] max_tok=1 -> warms cache to [prefix+dummy+g] -> (discarded)
  R3 rewind [prefix + msgA]            -> reuse [prefix], rewind the short
                                          dummy tail (drop<=32), prefill msgA -> O3
Assert O1 == O3 (bit-identical decoded text). temperature=0 + byteexact => deterministic.
Confirm the rewind FIRED by grepping audit_daemon.log for the PERSIST-KV line (R3 only).
SP_PFX = prefix filler word count (default 150 => < RING_W=1024, the no-wrap simple case).
"""
import itertools, json, os, sys, time, urllib.request
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
PORT = int(os.environ.get("SP_TEST_PORT", "3000"))
PFX = int(os.environ.get("SP_PFX", "150"))
vocab = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima "
         "mike november oscar papa quebec romeo sierra tango").split()
prefix = ("You are a helpful assistant. Reference context, ignore it: "
          + " ".join(itertools.islice(itertools.cycle(vocab), PFX)))
MSGA = "Count from one to twenty in words, separated by single spaces."
DUMMY = "hi"


def chat(system, user, max_tokens):
    body = {"messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "max_tokens": max_tokens, "temperature": 0, "byteexact": True,
            "eot_bias": 0, "seed": 7}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    out = []
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            s = raw.decode("utf-8", "replace").strip()
            if s.startswith("data:"):
                p = s[5:].strip()
                if p == "[DONE]":
                    break
                try:
                    d = json.loads(p)
                    if "delta" in d:
                        out.append(d["delta"])
                except Exception:
                    pass
    return "".join(out), time.time() - t0


print(f"PORT={PORT} prefix_words={PFX}", flush=True)
print("R1 cold  [prefix+msgA] ...", flush=True)
o1, t1 = chat(prefix, MSGA, 40)
print(f"  t={t1:.1f}s  O1={o1!r}\n", flush=True)
print("R2 warm  [prefix+dummy] max_tokens=1 ...", flush=True)
o2, t2 = chat(prefix, DUMMY, 1)
print(f"  t={t2:.1f}s  O2={o2!r}\n", flush=True)
print("R3 rewind[prefix+msgA] ...", flush=True)
o3, t3 = chat(prefix, MSGA, 40)
print(f"  t={t3:.1f}s  O3={o3!r}\n", flush=True)
ident = (o1 == o3)
print(f"VERDICT: {'BIT-IDENTICAL  (rewind holds)' if ident else 'DIVERGED  (rewind broke geometry)'}",
      flush=True)
print(f"  O1==O3 : {ident}", flush=True)
print(f"  timing : R3 {t3:.1f}s  vs  R1 {t1:.1f}s  (reuse => R3 should be much faster)", flush=True)
