"""margin_calib.py — G-L5-MARGIN-CALIB: measure the top1−top2 L5 margin distribution
per query class on the live daemon (margin gate OFF — pure telemetry-then-pin).

Classes:
  para       61 paraphrase queries (facts.json .para)   -> correct-match margins (label: fct_i)
  sne_canon   3 SNE canonical (audited)                 -> correct-match margins (sne)
  sne_mm      3 SNE mismatch                            -> attr-decline path margins
  hforeign   18 hard-foreign (unanswerable)             -> background margins
  conv        8 conversational statements/smalltalk     -> background margins

Output: per-class margin stats + the separation table + a suggested tau_m
(the largest gap between correct-match minima and background maxima).
Receipt -> tests/fixtures/chat_fullstack/G-L5-MARGIN-CALIB.log
"""
import json, os, re, sys, time, urllib.request, datetime

ENG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F   = json.load(open(f"{ENG}/_faithful_corpus/facts.json", encoding="utf-8"))
SNE = json.load(open(f"{ENG}/_faithful_corpus/sne/sne_facts_audited.json", encoding="utf-8"))
HF  = json.load(open(f"{ENG}/_faithful_corpus/hard_foreign_queries.json", encoding="utf-8"))
SERVE_LOG = f"{ENG}/_oneconfig_serve.log"
OUT = f"{ENG}/tests/fixtures/chat_fullstack/G-L5-MARGIN-CALIB.log"
CONSOLE = ("You are Shannon-Prime, a local AI with a real working memory. Keep replies short. "
           "Use facts you were given faithfully; if you don't know, say so.")
CONV = [
    "My designation is Wembly-9Q3. Please remember that.",
    "Thanks, that was really helpful.",
    "Good morning! Hope you slept well, so to speak.",
    "Let's talk about something else now.",
    "I had a great coffee earlier today.",
    "Please keep your replies short from now on.",
    "That last answer felt a bit off, be careful.",
    "I'm heading to bed soon.",
]
MARGIN_RE = re.compile(r"RECALL-L5-MARGIN: top1='([^']*)' cos=([0-9.\-]+) top2='([^']*)' cos2=([0-9.\-]+) margin=([0-9.\-]+|inf)")

def ask(q):
    b = json.dumps({"messages": [{"role": "system", "content": CONSOLE}, {"role": "user", "content": q}],
                    "max_tokens": 8, "temperature": 0, "eot_bias": 4.0, "auto_recall": True}).encode()
    r = urllib.request.Request("http://127.0.0.1:3000/v1/chat", data=b,
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=300) as resp:
        for raw in resp:
            s = raw.decode("utf-8", "replace").strip()
            if s.startswith("data:") and s[5:].strip() == "[DONE]": break

def log_size(): return os.path.getsize(SERVE_LOG) if os.path.exists(SERVE_LOG) else 0

def new_margin(ofs):
    with open(SERVE_LOG, encoding="utf-8", errors="replace") as f:
        f.seek(ofs)
        tail = f.read()
    ms = MARGIN_RE.findall(tail)
    return ms[-1] if ms else None

lines = []
def log(s):
    print(s, flush=True)
    lines.append(s)

rows = []  # (cls, label, top1, cos, top2, cos2, margin, top1_correct)
def drive(cls, items):
    for label, q, expect in items:
        ofs = log_size()
        try: ask(q)
        except Exception as e:
            log(f"[{cls} ERR] {label}: {e}"); continue
        time.sleep(0.3)
        m = new_margin(ofs)
        if not m:
            rows.append((cls, label, "-", 0.0, "-", 0.0, -1.0, False))
            log(f"[{cls}] {label}: NO-MARGIN-LINE (L5 did not run)")
            continue
        top1, cos, top2, cos2, marg = m
        marg = float("inf") if marg == "inf" else float(marg)
        ok = (expect is None) or (top1 == expect)
        rows.append((cls, label, top1, float(cos), top2, float(cos2), marg, ok))
        log(f"[{cls}] {label}: top1={top1}{'(OK)' if ok and expect else ('(WRONG want ' + expect + ')' if expect else '')} cos={cos} margin={marg:.4f}")

log(f"G-L5-MARGIN-CALIB  {datetime.datetime.now().isoformat()}  (margin gate OFF, telemetry only)")
drive("para",      [(F[i]["id"], F[i]["para"], f"fct_{i:03d}") for i in range(len(F))])
drive("sne_canon", [(it["entity"], it["canonical_q"], it["name"]) for it in SNE[:3]])
drive("sne_mm",    [(it["entity"], it["mismatch_q"], it["name"]) for it in SNE[:3]])
drive("hforeign",  [(f"hf_{i}", it["q"], None) for i, it in enumerate(HF)])
drive("conv",      [(f"conv_{i}", q, None) for i, q in enumerate(CONV)])

def stats(cls, correct_only=False):
    ms = [r[6] for r in rows if r[0] == cls and r[6] >= 0 and (r[7] or not correct_only)]
    if not ms: return None
    ms = sorted(ms)
    return (len(ms), ms[0], ms[len(ms)//2], ms[-1])

log("\n=== margin distribution (n, min, median, max) ===")
for cls, co in [("para", True), ("sne_canon", True), ("sne_mm", False), ("hforeign", False), ("conv", False)]:
    s = stats(cls, co)
    log(f"{cls:10s}: {s}")

sel = [r for r in rows if r[0] == "para"]
sel_ok = sum(1 for r in sel if r[7])
log(f"\npara selector top1 accuracy: {sel_ok}/{len(sel)}")

pm = sorted(r[6] for r in rows if r[0] in ("para", "sne_canon") and r[7] and r[6] >= 0)
bm = sorted(r[6] for r in rows if r[0] in ("hforeign", "conv") and r[6] >= 0)
if pm and bm:
    log(f"correct-match margins: min={pm[0]:.4f} p10={pm[max(0,len(pm)//10)]:.4f}")
    log(f"background margins:    max={bm[-1]:.4f} p90={bm[(len(bm)*9)//10]:.4f}")
    # sweep tau_m: keep-rate on correct matches vs suppress-rate on background
    log("tau_m sweep: tau  keep-correct  suppress-background")
    for t in [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.20]:
        keep = sum(1 for m in pm if m >= t) / len(pm)
        supp = sum(1 for m in bm if m < t) / len(bm)
        log(f"  {t:.3f}   {keep:5.1%}        {supp:5.1%}")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w", encoding="utf-8").write("\n".join(lines) + "\n")
print(f"receipt -> {OUT}")
