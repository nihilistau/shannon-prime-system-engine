"""Reproduce the live 'forgets across turns' bug: a name-memory conversation sent EXACTLY like the
browser console (full history each turn + eot_bias + auto_recall). If turn-3 answers 'Knack' the
daemon is fine and the console is the problem; if it says 'I do not know' the daemon path is broken.
SP_TEST_PORT picks the daemon, SP_AR=0 disables auto_recall to A/B it."""
import json, os, sys, urllib.request
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
PORT = int(os.environ.get("SP_TEST_PORT", "3000"))
AR = os.environ.get("SP_AR", "1") == "1"
SYS = ("You are Shannon-Prime, a local AI with a real working memory. Keep replies short. Pay close "
       "attention to what the user said earlier in THIS conversation and use it faithfully.")


def chat(msgs):
    b = {"messages": msgs, "max_tokens": 40, "eot_bias": 4.0,
         "temperature": float(os.environ.get("SP_TEMP", "0")),
         "repetition_penalty": float(os.environ.get("SP_REP", "1.0")),
         "seed": 7}
    if AR:
        b["auto_recall"] = True
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat", data=json.dumps(b).encode(),
                                 headers={"Content-Type": "application/json"})
    out = []
    with urllib.request.urlopen(req, timeout=120) as r:
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
    return " ".join("".join(out).split())


print(f"PORT={PORT} auto_recall={AR}", flush=True)
hist = [{"role": "system", "content": SYS}]
# Operator's EXACT order: ask the name BEFORE it's given (model says "I don't know"), then give it,
# then ask again -- to test whether the model anchors on its own early "I do not know".
TURNS = os.environ.get("SP_TURNS", "What is my name?|My name is Knack.|What colour is the sky?|What is my name?").split("|")
for u in TURNS:
    hist.append({"role": "user", "content": u})
    a = chat(hist)
    hist.append({"role": "assistant", "content": a})
    print(f"U: {u}\nA: {a}\n", flush=True)
print("VERDICT:", "REMEMBERS" if "knack" in hist[-1]["content"].lower() else "FORGOT", flush=True)
