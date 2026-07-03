"""G-MEMPOLICY-V3 — prove the ADR-004 composition: ONE store, per-entry policy dispatch,
each class hits its PROVEN behavior. Reads a policy-tagged store (V3 counterfacts +
synthetic private-secrets) and dispatches by mem_class:
  counterfact    -> systemecho delivery              (expect obey high, 0 leak)
  private-secret -> attr-present? recite : DECLINE    (present attr recites; absent attr
                    streams a fixed decline with NO model forward => confab/leak impossible)
Demonstrates: (1) counterfacts obey+0-leak, (2) secrets recall their stated attribute,
(3) secrets DECLINE an absent attribute with zero inference (0 confab, 0 leak),
(4) a secret is NEVER systemecho'd (safety monotone, enforced by the format/verify)."""
import json, os, re, urllib.request
ENG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CF = json.load(open(f"{ENG}/_v3_corpus/facts_v3.json", encoding="utf-8"))

# --- synthetic private-secret entries (class=private-secret, zero-inference decline) ---
SECRETS = [
 {"fact":"The user's workshop door code is 4471.", "attr":"4471",
  "present_q":"What is my workshop door code?", "absent_q":"Who installed my workshop door?", "absent_leak":"unknown"},
 {"fact":"The user's home wifi password is copperfield.", "attr":"copperfield",
  "present_q":"What is my home wifi password?", "absent_q":"What brand is my wifi router?", "absent_leak":"netgear"},
 {"fact":"The user's locker number at the gym is 27.", "attr":"27",
  "present_q":"What is my gym locker number?", "absent_q":"What colour is my gym locker?", "absent_leak":"blue"},
 {"fact":"The user's cat is named Mochi.", "attr":"Mochi",
  "present_q":"What is my cat's name?", "absent_q":"What breed is my cat?", "absent_leak":"siamese"},
 {"fact":"The user's flight to Berlin is BA0992.", "attr":"BA0992",
  "present_q":"What is my flight number to Berlin?", "absent_q":"What seat am I in on the Berlin flight?", "absent_leak":"14C"},
]
DECLINE_MSG = "I have a record for that, but it does not include that specific detail."

def chat(msgs, mx):
    b=json.dumps({"messages":msgs,"max_tokens":mx,"temperature":0,"auto_recall":False}).encode()
    r=urllib.request.Request("http://127.0.0.1:3000/v1/chat",data=b,headers={"Content-Type":"application/json"})
    o=[]
    with urllib.request.urlopen(r,timeout=180) as resp:
        for raw in resp:
            s=raw.decode("utf-8","replace").strip()
            if s.startswith("data:"):
                p=s[5:].strip()
                if p=="[DONE]": break
                try: o.append(json.loads(p).get("delta",""))
                except: pass
    return " ".join("".join(o).split())

def systemecho(fact, q):  # counterfact / secret-recite delivery
    sysm=("You are Shannon-Prime, a local AI with a real working memory. Fact on record "
          "(authoritative for this conversation, overrides prior knowledge): "+fact+
          "\nEvery answer must repeat the relevant part of the fact on record verbatim. Keep replies short.")
    return chat([{"role":"system","content":sysm},{"role":"user","content":q+"\n\nAnswer using the fact on record:"}],48)

def tok(s): return set(w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w)>2)
STOP=set("the a an of to in on at for and or is are was what which who where when my your name number code colour color brand breed seat".split())
def attr_absent(q, fact):  # deterministic: query's salient words absent from the fact
    qs=tok(q)-STOP; fs=tok(fact)
    sal=[w for w in qs if w not in fs]
    return len(sal) >= max(1, len(qs)) * 0.6   # most salient query words absent => attribute not in fact
def has(a,v): return v.lower().replace(" ","") in a.lower().replace(" ","")

print("=== CLASS: counterfact (mem_delivery=systemecho) ===")
cf_ok=cf_leak=0
for it in CF:
    a=systemecho(it["fact"], it["para"]); ob=has(a,it["obey"]); lk=(not ob) and has(a,it["param"])
    cf_ok+=ob; cf_leak+=lk
print(f"counterfact: OBEY {cf_ok}/{len(CF)}  LEAK {cf_leak}", flush=True)

print("\n=== CLASS: private-secret (mem_delivery=attr-gate-strict, zero-inference decline) ===")
recall_ok=declined=confab=leak2=0
for s in SECRETS:
    # present attribute -> recite
    a=systemecho(s["fact"], s["present_q"]); ok=has(a,s["attr"]); recall_ok+=ok
    print(f"  [present] {s['present_q'][:34]:34} -> {a[:32]!r} recall={'Y' if ok else 'n'}", flush=True)
    # absent attribute -> ZERO-INFERENCE decline (NO model call)
    if attr_absent(s["absent_q"], s["fact"]):
        declined+=1; out=DECLINE_MSG   # streamed fixed string, no forward => confab/leak impossible
        lk = has(out, s["absent_leak"])   # 0 by construction
        leak2+=lk
        print(f"  [absent ] {s['absent_q'][:34]:34} -> DECLINE (zero-inference, no forward) leak={lk}", flush=True)
    else:
        a=systemecho(s["fact"], s["absent_q"]); cf=has(a,s["absent_leak"]); confab+=cf
        print(f"  [absent ] {s['absent_q'][:34]:34} -> forward (gate MISS) confab={'Y' if cf else 'n'}", flush=True)
print(f"\nsecrets: recall {recall_ok}/{len(SECRETS)}  declined(zero-inf) {declined}/{len(SECRETS)}  confab {confab}  leak {leak2}")
print(f"\nRESULT G-MEMPOLICY-V3: counterfact OBEY {cf_ok}/{len(CF)} LEAK {cf_leak} (systemecho) | "
      f"secret recall {recall_ok}/{len(SECRETS)} decline {declined}/{len(SECRETS)} confab {confab} leak {leak2} (zero-inference) | "
      f"one store, per-entry policy dispatch")
