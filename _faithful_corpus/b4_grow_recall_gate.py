"""b4_grow_recall_gate.py — G-B4-GROW-RECALL-L5: the SYSTEM SEALS gate.

Serve under run_console_system.bat (PRODUCTION empty registry + B4 growth +
L5 recall). One scripted session proves the full loop LIVE:

  G  grow    — 5 novel personal assertions (statements: QONLY skips recall,
               B4 captures each as ep_live_NNN, mint_live_ep_l5 writes ep.l5)
  R  recall  — 5 PARAPHRASE questions -> L5 must select the GROWN episodes and
               the answers must contain the grown fact tokens
  F  foreign — 1 general-knowledge question -> answer stays clean (no grown-fact
               token leaks into it)
  P  persist — caller restarts the daemon, then --persist re-asks one paraphrase
               -> still recalled (registry line + ep.l5 sidecar reloaded from disk)

PASS: R >= 4/5 grown-recall correct AND F clean AND (persist leg) P correct.
Facts are NOVEL (in no corpus): the only way to answer is the grown memory.

Usage:  python b4_grow_recall_gate.py grow    (phases G+R+F)
        python b4_grow_recall_gate.py persist (phase P, after daemon restart)
"""
import json, sys, urllib.request, datetime

FACTS = [
    ("dog",    "My dog is named Biscuit.",                          "What is the name of my dog?",                    "Biscuit"),
    ("door",   "My workshop door code is 4471.",                    "What code opens my workshop door?",              "4471"),
    ("sister", "My sister Clara lives in Hobart.",                  "Which city does my sister live in?",             "Hobart"),
    ("car",    "My car is a green 1987 Volvo 240.",                 "What kind of car do I drive?",                   "Volvo"),
    ("tea",    "My favourite tea is lapsang souchong.",             "Which tea do I like best?",                      "lapsang"),
]
FOREIGN = ("What is the capital of Spain?", "Madrid")
CONSOLE = ("You are Shannon-Prime, a local AI with a real working memory. Keep replies short. "
           "Use facts you were given faithfully; if you don't know, say so.")

def ask(q):
    b = json.dumps({"messages": [{"role": "system", "content": CONSOLE},
                                 {"role": "user", "content": q}],
                    "max_tokens": 48, "temperature": 0, "eot_bias": 4.0,
                    "auto_recall": True}).encode()
    r = urllib.request.Request("http://127.0.0.1:3000/v1/chat", data=b,
                               headers={"Content-Type": "application/json"})
    o = []
    with urllib.request.urlopen(r, timeout=300) as resp:
        for raw in resp:
            s = raw.decode("utf-8", "replace").strip()
            if s.startswith("data:"):
                p = s[5:].strip()
                if p == "[DONE]": break
                try: o.append(json.loads(p).get("delta", ""))
                except Exception: pass
    return " ".join("".join(o).split())

def has(ans, val): return val.lower().replace(" ", "") in ans.lower().replace(" ", "")

mode = sys.argv[1] if len(sys.argv) > 1 else "grow"
print(f"G-B4-GROW-RECALL-L5 [{mode}]  {datetime.datetime.now().isoformat()}")

if mode == "grow":
    for key, assertion, _, _ in FACTS:
        a = ask(assertion)
        print(f"[G {key}] {a[:56]!r}", flush=True)
    r_ok = 0
    for key, _, q, tok in FACTS:
        a = ask(q)
        ok = has(a, tok); r_ok += ok
        print(f"[R {'ok' if ok else 'MISS'} {key}] {a[:64]!r} (want {tok})", flush=True)
    fq, fa = FOREIGN
    a = ask(fq)
    grown_leak = any(has(a, t) for _, _, _, t in FACTS)
    f_ok = has(a, fa) and not grown_leak
    print(f"[F {'ok' if f_ok else 'FAIL'}] {a[:64]!r} (want {fa}, no grown tokens)")
    verdict = r_ok >= 4 and f_ok
    print(f"RESULT grow: R {r_ok}/5  F {'clean' if f_ok else 'DIRTY'}  -> {'PASS (run persist leg after daemon restart)' if verdict else 'FAIL'}")
    sys.exit(0 if verdict else 1)
else:
    key, _, q, tok = FACTS[0]
    a = ask(q)
    ok = has(a, tok)
    print(f"[P {'ok' if ok else 'MISS'} {key}] {a[:64]!r} (want {tok})")
    print(f"RESULT persist: {'PASS — the system GROWS, RECALLS, and SURVIVES RESTART' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)
