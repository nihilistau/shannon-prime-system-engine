"""noexact_ab.py — G-NOEXACT-OBEY-AB: the exact-vs-FP behavioural A/B on ONE live daemon.

No new build: same binary, flip the per-request `byteexact` field (default true=exact).
Serve under run_console_faithful.bat, then this drives four measurements:

  DET   determinism: same prompt x2 per mode -> byte-identical within mode? (exact should be;
        FP on a single pinned box may also be — the real exact win is CROSS-MACHINE, not testable here)
  FTH   faithfulness: all 61 paraphrase obey both modes -> parity? (expect ~tie)
  SPD   speed: decode tok/s (delta count / wall) both modes -> expect FP faster
  ECHO  short-prompt echo count both modes -> the #47 signal (expect FP better on some)

Receipt -> tests/fixtures/chat_fullstack/G-NOEXACT-OBEY-AB.log
"""
import json, os, time, urllib.request, datetime

ENG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F   = json.load(open(f"{ENG}/_faithful_corpus/facts.json", encoding="utf-8"))
OUT = f"{ENG}/tests/fixtures/chat_fullstack/G-NOEXACT-OBEY-AB.log"
CONSOLE = ("You are Shannon-Prime, a local AI with a real working memory. Keep replies short. "
           "Use facts you were given faithfully; if you don't know, say so.")

lines = []
def log(s):
    print(s, flush=True); lines.append(s)
    with open(OUT, "w", encoding="utf-8") as f: f.write("\n".join(lines) + "\n")

def ask(content, byteexact, auto=True, max_tokens=48):
    """Return (answer_text, n_deltas, elapsed_s)."""
    b = json.dumps({"messages": [{"role": "system", "content": CONSOLE},
                                 {"role": "user", "content": content}],
                    "max_tokens": max_tokens, "temperature": 0, "eot_bias": 4.0,
                    "auto_recall": auto, "byteexact": byteexact}).encode()
    r = urllib.request.Request("http://127.0.0.1:3000/v1/chat", data=b,
                               headers={"Content-Type": "application/json"})
    o, n = [], 0
    t0 = time.time()
    with urllib.request.urlopen(r, timeout=300) as resp:
        for raw in resp:
            s = raw.decode("utf-8", "replace").strip()
            if s.startswith("data:"):
                p = s[5:].strip()
                if p == "[DONE]": break
                try:
                    d = json.loads(p).get("delta", "")
                    if d: o.append(d); n += 1
                except Exception: pass
    dt = time.time() - t0
    return " ".join("".join(o).split()), n, dt

def has(ans, val): return val.lower().replace(" ", "") in ans.lower().replace(" ", "")

def echoes(prompt, ans):
    """echo = reply repeats >=6 consecutive words of the user prompt verbatim."""
    pw = [w for w in "".join(c.lower() if c.isalnum() or c.isspace() else " "
                            for c in prompt).split()]
    aw = "".join(c.lower() if c.isalnum() or c.isspace() else " " for c in ans).split()
    aset = " ".join(aw)
    for i in range(len(pw) - 5):
        if " ".join(pw[i:i+6]) in aset: return True
    return False

# short-prompt set (the #47 signal: short/greeting/terse prompts where FP tends to be correct)
SHORT = ["Hi.", "Hello there.", "How are you?", "Thanks.", "What's your name?",
         "Tell me a joke.", "Good morning.", "Are you there?", "Ok.", "Who are you?",
         "What can you do?", "Nice to meet you."]

log(f"G-NOEXACT-OBEY-AB  {datetime.datetime.now().isoformat()}")
log("same binary, per-request byteexact flip (true=exact / false=FP); run_console_faithful.bat config")
log("")

# ---- DET: determinism within each mode (same prompt x2) ----
log("== DET (determinism: same prompt x2 within mode; exact expected identical) ==")
det_prompts = [F[i]["para"] for i in range(0, min(6, len(F)))]
det = {"exact": 0, "fp": 0, "n": len(det_prompts)}
for q in det_prompts:
    a1, _, _ = ask(q, True);  a2, _, _ = ask(q, True)
    b1, _, _ = ask(q, False); b2, _, _ = ask(q, False)
    ex_id = a1 == a2; fp_id = b1 == b2
    det["exact"] += ex_id; det["fp"] += fp_id
    log(f"[DET] exact_id={ex_id} fp_id={fp_id}  q={q[:44]!r}")
    if not ex_id: log(f"      exact A:{a1[:50]!r}\n      exact B:{a2[:50]!r}")
    if not fp_id: log(f"      fp    A:{b1[:50]!r}\n      fp    B:{b2[:50]!r}")
log(f"DET: exact identical {det['exact']}/{det['n']} · FP identical {det['fp']}/{det['n']}")
log("")

# ---- FTH + SPD: 61 paraphrase obey + tok/s, both modes ----
log("== FTH+SPD (61 paraphrase obey + decode tok/s, both modes) ==")
res = {}
for mode, bx in (("exact", True), ("fp", False)):
    ok = 0; toks = 0; secs = 0.0; miss = []
    for it in F:
        a, n, dt = ask(it["para"], bx)
        hit = has(a, it["obey"]); ok += hit
        toks += n; secs += dt
        if not hit: miss.append(it.get("id", "?"))
    tps = toks / secs if secs else 0
    res[mode] = {"ok": ok, "n": len(F), "tps": tps, "toks": toks, "secs": secs, "miss": miss}
    log(f"[{mode}] obey {ok}/{len(F)} · {tps:.2f} tok/s ({toks} tok / {secs:.1f}s) · miss={miss}")
log(f"FTH parity: exact {res['exact']['ok']}/{res['exact']['n']} vs fp {res['fp']['ok']}/{res['fp']['n']}")
log(f"SPD: exact {res['exact']['tps']:.2f} tok/s vs fp {res['fp']['tps']:.2f} tok/s "
    f"(FP speedup {res['fp']['tps']/res['exact']['tps']:.2f}x)" if res['exact']['tps'] else "")
log("")

# ---- ECHO: short-prompt echo count, both modes ----
log("== ECHO (short-prompt echo count; #47 signal) ==")
ech = {}
for mode, bx in (("exact", True), ("fp", False)):
    c = 0
    for q in SHORT:
        a, _, _ = ask(q, bx, auto=False, max_tokens=40)
        e = echoes(q, a); c += e
        log(f"[ECHO {mode} {'ECHO' if e else 'ok'}] {q!r} -> {a[:56]!r}")
    ech[mode] = c
log(f"ECHO: exact {ech['exact']}/{len(SHORT)} echoed · FP {ech['fp']}/{len(SHORT)} echoed")
log("")

# ---- verdict ----
log("== VERDICT ==")
log(f"determinism (this box): exact {det['exact']}/{det['n']} · FP {det['fp']}/{det['n']} "
    f"identical run-to-run  [cross-MACHINE determinism is exact's real, untested-here win]")
log(f"faithfulness: exact {res['exact']['ok']} vs FP {res['fp']['ok']} obey  "
    f"(delta {res['fp']['ok']-res['exact']['ok']:+d})")
log(f"speed: FP {res['fp']['tps']:.2f} vs exact {res['exact']['tps']:.2f} tok/s")
log(f"echo(#47): exact {ech['exact']} vs FP {ech['fp']} echoed")
log("RESULT G-NOEXACT-OBEY-AB: DATA CAPTURED (interpret in DESIGN-NO-EXACT-PROFILE §4)")
print("DONE")
