"""echo_hunt_47.py — reproduce (or refute) #47: byteexact-default-ON echoes/degenerates on some
short prompts where FP (byteexact:false) is correct. Run under run_console_CHAT.bat (recall OFF,
no test registry, plain generation — the daily-driver path where #47 was observed).

Per prompt, ask BOTH modes and flag three failure kinds:
  ECHO  reply repeats >=6 consecutive words of the prompt verbatim
  REPEAT degenerate internal repetition (a 1-3 word phrase repeated >=4x consecutively)
  DIVERGE exact != FP (report both)
The #47 claim is confirmed iff exact fails (ECHO/REPEAT) on a prompt where FP is clean.

Receipt -> tests/fixtures/chat_fullstack/G-ECHO-HUNT-47.log
"""
import json, os, re, time, urllib.request, datetime

ENG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{ENG}/tests/fixtures/chat_fullstack/G-ECHO-HUNT-47.log"
CONSOLE = "You are Shannon-Prime, a local AI. Keep replies short."

lines = []
def log(s):
    print(s, flush=True); lines.append(s)
    with open(OUT, "w", encoding="utf-8") as f: f.write("\n".join(lines) + "\n")

def ask(content, byteexact, max_tokens=64):
    b = json.dumps({"messages": [{"role": "system", "content": CONSOLE},
                                 {"role": "user", "content": content}],
                    "max_tokens": max_tokens, "temperature": 0, "eot_bias": 4.0,
                    "auto_recall": False, "byteexact": byteexact}).encode()
    r = urllib.request.Request("http://127.0.0.1:3000/v1/chat", data=b,
                               headers={"Content-Type": "application/json"})
    o = []
    with urllib.request.urlopen(r, timeout=300) as resp:
        for raw in resp:
            s = raw.decode("utf-8", "replace").strip()
            if s.startswith("data:"):
                p = s[5:].strip()
                if p == "[DONE]": break
                try:
                    d = json.loads(p).get("delta", "")
                    if d: o.append(d)
                except Exception: pass
    return "".join(o).strip()

def norm(t): return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", t.lower())).strip()

def is_echo(prompt, ans):
    pw = norm(prompt).split(); a = norm(ans)
    return any(len(pw) >= 6 and " ".join(pw[i:i+6]) in a for i in range(len(pw)-5))

def is_repeat(ans):
    w = norm(ans).split()
    for span in (1, 2, 3):                      # a 1-3 word phrase repeated >=4x in a row
        run = 1
        for i in range(span, len(w)):
            if w[i-span:i] and w[i] == w[i-span]:
                run += 1
                if run >= 4 * span: return True
            else: run = 1
    # also: any single line repeated >=3x
    return False

# broad adversarial SHORT-prompt set (the #47 regime): terse, single-word, incomplete,
# repetition-inviting, statements-as-prompts, one-word questions.
PROMPTS = [
    "Hi.", "Hello.", "Hey", "Ok", "Yes", "No", "Why?", "What?", "Sure.", "Continue.",
    "Repeat after me: banana.", "Say the word apple.", "Count to five.",
    "My name is Knack.", "The sky is blue.", "Tell me something.", "Go on.",
    "What is 2 plus 2?", "Name a color.", "Finish this: the quick brown",
    "Hmm.", "...", "And?", "Again.", "Once more.", "Echo this: test one two.",
    "Good.", "Nice.", "Cool.", "Thanks!",
]

log(f"G-ECHO-HUNT-47  {datetime.datetime.now().isoformat()}")
log("config: run_console_chat.bat (recall OFF, no test registry, plain gen); byteexact true(exact) vs false(FP)")
log(f"prompts: {len(PROMPTS)} short/adversarial")
log("")

confirmed = []   # prompts where exact fails but FP is clean  (== #47 confirmed)
both_bad = []; fp_worse = []
for q in PROMPTS:
    ex = ask(q, True); fp = ask(q, False)
    ex_bad = is_echo(q, ex) or is_repeat(ex)
    fp_bad = is_echo(q, fp) or is_repeat(fp)
    diverge = ex != fp
    tag = ("EXACT-FAILS/FP-OK" if ex_bad and not fp_bad else
           "both-bad" if ex_bad and fp_bad else
           "FP-FAILS/EXACT-OK" if fp_bad and not ex_bad else
           "both-ok")
    if ex_bad and not fp_bad: confirmed.append(q)
    if ex_bad and fp_bad: both_bad.append(q)
    if fp_bad and not ex_bad: fp_worse.append(q)
    mark = "  <<<" if (ex_bad and not fp_bad) else ""
    log(f"[{tag}] {q!r}{mark}")
    log(f"      exact: {ex[:90]!r}")
    if diverge: log(f"      fp   : {fp[:90]!r}")
    else:       log(f"      fp   : (identical)")

log("")
log("== VERDICT ==")
log(f"#47 CONFIRMED (exact echoes/degenerates where FP is clean): {len(confirmed)}/{len(PROMPTS)}")
if confirmed: log(f"  prompts: {confirmed}")
log(f"both-bad: {len(both_bad)} · FP-worse: {len(fp_worse)}")
if len(confirmed):
    log("RESULT G-ECHO-HUNT-47: #47 REPRODUCED — FP fixes it on these prompts (2nd argument for FP-default)")
else:
    log("RESULT G-ECHO-HUNT-47: #47 NOT reproduced on the current binary under plain chat config "
        "(likely already fixed by the Hodor-era changes; downgrade/close #47)")
print("DONE")
