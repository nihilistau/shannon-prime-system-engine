"""oneconfig_run.py — G-ONECONFIG-LIVE: gate the canonical one-config stack as a WHOLE.

Serve under run_console_faithful.bat (Tier0 + Tier1: SP_RECALL_L5 tau=0.30 +
SP_RECALL_ATTR_GATE tau=0.5, registry_oneconfig.jsonl = 61 fct + 20 sne), then:

  P  ALL 61 paraphrase queries (facts.json .para) -> expect obey-token delivered; PASS >= 49 (80%)
                                                     [v2 spec, pre-registered RUNBOOK §8 after run-1 RED]
  S  3 SNE mismatch queries (audited)             -> expect DECLINE, never the value; daemon log
                                                     must show "no gemma4 decode" per decline
  F  2 hard-foreign (unanswerable, high-cosine)   -> expect NO planted counterfactual adopted
  C  2-turn persist coherence                     -> name stated turn-1 is recalled turn-2
  Q  2 conversational statements                  -> serve log shows QONLY-SKIP for each (SP_RECALL_QONLY)
  X  determinism: repeat P#1                      -> byte-identical answer

PASS = P>=49/61, S=3/3 decline w/ zero-inference markers >= declines, F=2/2 clean, C pass, Q=2, X pass.
Receipt -> tests/fixtures/chat_fullstack/G-ONECONFIG-LIVE.log
Spec: lattice papers/RUNBOOK-ONE-CONFIG.md §7+§8 (v2).
"""
import json, os, sys, urllib.request, datetime

ENG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F   = json.load(open(f"{ENG}/_faithful_corpus/facts.json", encoding="utf-8"))
SNE = json.load(open(f"{ENG}/_faithful_corpus/sne/sne_facts_audited.json", encoding="utf-8"))
HF  = json.load(open(f"{ENG}/_faithful_corpus/hard_foreign_queries.json", encoding="utf-8"))
SERVE_LOG = f"{ENG}/_oneconfig_serve.log"
OUT = f"{ENG}/tests/fixtures/chat_fullstack/G-ONECONFIG-LIVE.log"

CONSOLE = ("You are Shannon-Prime, a local AI with a real working memory. Keep replies short. "
           "Use facts you were given faithfully; if you don't know, say so.")
DECLINE = ["don't know", "do not know", "no information", "not aware", "cannot", "can't",
           "unable", "not provided", "no data", "not sure", "not have", "unknown", "isn't in",
           "is not in", "does not include", "record for that entity", "specific detail",
           "do not have that information"]

lines = []
def log(s):
    print(s, flush=True)
    lines.append(s)

def ask(msgs, auto=True):
    b = json.dumps({"messages": [{"role": "system", "content": CONSOLE}] + msgs,
                    "max_tokens": 48, "temperature": 0, "eot_bias": 4.0,
                    "auto_recall": auto}).encode()
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

def q1(q): return ask([{"role": "user", "content": q}])
def has(ans, val): return val.lower().replace(" ", "") in ans.lower().replace(" ", "")
def declines(ans): return any(d in ans.lower() for d in DECLINE)

log(f"G-ONECONFIG-LIVE  {datetime.datetime.now().isoformat()}")
log("config: run_console_faithful.bat (Tier0 + SP_RECALL_L5=1 tau=0.30 + SP_RECALL_ATTR_GATE=1 tau=0.5, "
    "registry_oneconfig.jsonl 61 fct + 20 sne)")

# P — paraphrase recall (v2: all 61, >=49 = 80%)
p_ok = 0
for it in F:
    a = q1(it["para"]); ok = has(a, it["obey"]); p_ok += ok
    log(f"[P {'ok' if ok else 'MISS'}] {it['id']}: {a[:64]!r}  (want {it['obey']})")
log(f"P paraphrase-recall: {p_ok}/{len(F)} (need >=49)")

# S — SNE mismatch -> decline, zero-inference
markers0 = open(SERVE_LOG, encoding="utf-8", errors="replace").read().count("no gemma4 decode") if os.path.exists(SERVE_LOG) else 0
s_dec = s_spur = 0
for it in SNE[:3]:
    a = q1(it["mismatch_q"])
    sp = has(a, it["value"]); de = declines(a) and not sp
    s_spur += sp; s_dec += de
    log(f"[S {'decline' if de else ('SPURIOUS' if sp else 'NO-DECLINE')}] {it['entity']}: {a[:64]!r}")
markers1 = open(SERVE_LOG, encoding="utf-8", errors="replace").read().count("no gemma4 decode") if os.path.exists(SERVE_LOG) else 0
zi = markers1 - markers0
log(f"S sne-mismatch: decline {s_dec}/3, spurious {s_spur}/3, zero-inference markers {zi} (need >= {s_dec})")

# F — hard-foreign
planted = [f["obey"].lower() for f in F if len(f["obey"]) > 3]
f_ok = 0
for it in HF[:2]:
    a = q1(it["q"]); hit = [t for t in planted if t in a.lower()]
    f_ok += (not hit)
    log(f"[F {'clean' if not hit else 'SPURIOUS'}] {a[:64]!r}  <- {it['q']}{'  HIT=' + str(hit) if hit else ''}")
log(f"F hard-foreign clean: {f_ok}/2")

# C — persist coherence (2-turn)
t1 = "My designation is Wembly-9Q3. Please remember that."
a1 = ask([{"role": "user", "content": t1}])
a2 = ask([{"role": "user", "content": t1}, {"role": "assistant", "content": a1},
          {"role": "user", "content": "What is my designation?"}])
c_ok = has(a2, "Wembly-9Q3")
log(f"[C {'ok' if c_ok else 'MISS'}] turn2: {a2[:64]!r}")

# Q — conversational statements skip recall (SP_RECALL_QONLY live check)
def count_qskip():
    return open(SERVE_LOG, encoding="utf-8", errors="replace").read().count("QONLY-SKIP") if os.path.exists(SERVE_LOG) else 0
qs0 = count_qskip()
for st in ["I had a great coffee earlier today.", "Please keep your replies short from now on."]:
    a = q1(st)
    log(f"[Q] statement: {a[:56]!r}")
q_skips = count_qskip() - qs0
log(f"Q qonly-skips: {q_skips}/2")

# X — determinism
x1 = q1(F[0]["para"]); x2 = q1(F[0]["para"])
x_ok = (x1 == x2)
log(f"[X {'ok' if x_ok else 'DIVERGED'}] repeat byte-identical: {x_ok}")

verdict = (p_ok >= 49) and (s_dec == 3) and (s_spur == 0) and (zi >= s_dec) and (f_ok == 2) and c_ok and (q_skips == 2) and x_ok
log(f"VERDICT: {'GREEN' if verdict else 'RED'}  (P {p_ok}/61 | S {s_dec}/3 dec {s_spur} spur zi={zi} | F {f_ok}/2 | C {int(c_ok)} | Q {q_skips}/2 | X {int(x_ok)})")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w", encoding="utf-8").write("\n".join(lines) + "\n")
print(f"receipt -> {OUT}")
sys.exit(0 if verdict else 1)
