"""End-to-end recall check: a matched query should RECALL a stored needle; a foreign query
should REJECT (NULL) and answer parametrically. Mirrors the console (system prompt + eot_bias)."""
import json, os, sys, urllib.request
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
PORT = int(os.environ.get("SP_TEST_PORT", "3000"))
SYS = ("You are Shannon-Prime, a local AI with a real working memory. Keep replies short. Use facts "
       "you were given faithfully; if you don't know, say so.")


def ask(q):
    body = json.dumps({"messages": [{"role": "system", "content": SYS},
                                    {"role": "user", "content": q}],
                       "max_tokens": 80, "temperature": 0, "eot_bias": 4.0,
                       "auto_recall": True}).encode()  # fire the W_c librarian
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat", data=body,
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


for tag, q in [("MATCHED", "Which recovery code authorizes the Marlock mag-rail depot?"),
               ("FOREIGN", "What is the capital of France?")]:
    print(f"{tag}: {ask(q)[:200]}", flush=True)
print("DONE", flush=True)
