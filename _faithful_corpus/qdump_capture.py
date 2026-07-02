"""qdump_capture.py — capture L5 query embeds for the selector campaign (offline lever tests).
Sends the 61 paraphrase queries then the 61 fact texts (max_tokens=1) against a serve with
SP_B3_QDUMP=<dir> set; q_<cid>.bin land in request order: cid 0..60 = paras, 61..121 = facts."""
import json, os, sys, urllib.request

ENG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = json.load(open(f"{ENG}/_faithful_corpus/facts.json", encoding="utf-8"))
CONSOLE = ("You are Shannon-Prime, a local AI with a real working memory. Keep replies short. "
           "Use facts you were given faithfully; if you don't know, say so.")

def ask(q):
    b = json.dumps({"messages": [{"role": "system", "content": CONSOLE}, {"role": "user", "content": q}],
                    "max_tokens": 1, "temperature": 0, "eot_bias": 4.0, "auto_recall": True}).encode()
    r = urllib.request.Request("http://127.0.0.1:3000/v1/chat", data=b,
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=200) as resp:
        for raw in resp:
            s = raw.decode("utf-8", "replace").strip()
            if s.startswith("data:") and s[5:].strip() == "[DONE]": break

for i, it in enumerate(F):
    ask(it["para"]); print(f"para {i:02d} {it['id']}", flush=True)
for i, it in enumerate(F):
    ask(it["fact"]); print(f"fact {i:02d} {it['id']}", flush=True)
print("capture done", flush=True)
