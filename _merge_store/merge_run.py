"""G-STORE-MERGE gate: query memories the HARNESS wrote into memory-okf/. If the engine
merged + minted them, it recalls + serves each per its OWN OKF policy."""
import json, urllib.request
DECLINE = ["don't know","do not know","not aware","cannot","unable","not provided","unknown",
           "does not include","record for that entity","specific detail","do not have that information"]
def ask(q):
    b = json.dumps({"messages":[{"role":"user","content":q}],"max_tokens":40,
                    "temperature":0,"eot_bias":4.0,"auto_recall":True}).encode()
    r = urllib.request.Request("http://127.0.0.1:3000/v1/chat", data=b, headers={"Content-Type":"application/json"})
    o=[]
    with urllib.request.urlopen(r, timeout=180) as resp:
        for raw in resp:
            s=raw.decode("utf-8","replace").strip()
            if s.startswith("data:"):
                p=s[5:].strip()
                if p=="[DONE]": break
                try: o.append(json.loads(p).get("delta",""))
                except: pass
    return " ".join("".join(o).split())
def has(a,v): return v.lower().replace(" ","") in a.lower().replace(" ","")
def dec(a): return any(d in a.lower() for d in DECLINE)

TESTS = [
  ("counterfact/systemecho", "What is the tallest mountain on Earth now?", "Kea", None),
  ("private-secret/recite",  "What is the launch code for Falcon-9?",       "Z7Q-3RT-9WX", None),
  ("private-secret/decline", "Who built Falcon-9?",                          None, "decline"),
]
print("=== engine serving HARNESS-written memory-okf concepts (SP_MEM_OKF_STORE) ===")
ok=0
for label, q, want, mode in TESTS:
    a = ask(q)
    if mode == "decline":
        good = dec(a); ok += good
        print(f"[{label:24}] {'DECLINE' if good else 'other':8} <- {q!r}  {a[:40]!r}")
    else:
        good = has(a, want); ok += good
        print(f"[{label:24}] {'OK' if good else 'miss':8} (want {want}) <- {a[:44]!r}")
print(f"\nRESULT G-STORE-MERGE: {ok}/{len(TESTS)} — engine recalls+serves harness-written concepts per OKF policy")
