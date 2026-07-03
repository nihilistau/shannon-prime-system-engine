"""refine_gate.py — G-LM-REFINE (LM-B3). Adaptive classification: the idle NIGHTSHIFT model-
refine CORRECTS a heuristic miss. A keyword-less secret is stored MIS-classified as counterfact
(what the keyword-heuristic would do); on idle the model re-classifies it private-secret; the
served policy flips an absent-attribute probe from answer->DECLINE. No restart.

  setup : write the store concept full/<addr>.md, mem_class=counterfact (the heuristic miss).
  run   : q the absent probe BEFORE refine; wait for the model-refine to correct the class;
          q the same probe AFTER; q a DIRECT question (recall preserved). Then verdict.
"""
import json, os, sys, time, hashlib, glob, urllib.request
ENG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORE = f"{ENG}/_refine_corpus/store"
FULL = f"{STORE}/full"
SECRET = "orchid tango falcon delta"
BODY = f"The recovery phrase for the Meridian vault archive is {SECRET}."
ABSENT_Q = "Who is the administrator of the Meridian vault archive?"
DIRECT_Q = "What is the recovery phrase for the Meridian vault archive?"
DECLINE = ["don't know","do not know","not aware","cannot","can't","unable","not provided",
           "unknown","does not include","do not have","no information","not able","won't"]

def addr_of(body): return hashlib.sha256(body.encode()).hexdigest()[:16]

def concept_path(): return f"{FULL}/{addr_of(BODY)}.md"

def write_concept(mem_class):
    os.makedirs(FULL, exist_ok=True)
    a = addr_of(BODY)
    deliv = {"counterfact":"systemecho","private-secret":"attr-gate-strict"}[mem_class]
    fm = [f"---", f"type: mem-concept", f"title: Meridian vault recovery phrase",
          f"addr: {a}", f"mem_class: {mem_class}", f"mem_delivery: {deliv}"]
    if mem_class == "private-secret":
        fm += ["mem_authority: private",
               "mem_decline_when: [zero-inference, attribute-absent]",
               "mem_decline_message: I have a record for that but not the detail you asked for."]
    else:
        fm += ["mem_authority: overrides-prior"]
    fm += ["---", "", BODY, ""]
    open(concept_path(), "w", encoding="utf-8").write("\n".join(fm))
    return a

def read_class():
    for line in open(concept_path(), encoding="utf-8"):
        if line.strip().startswith("mem_class:"):
            return line.split(":",1)[1].strip()
    return "?"

def ask(q):
    b = json.dumps({"messages":[{"role":"user","content":q}],"max_tokens":40,
                    "temperature":0,"eot_bias":4.0,"auto_recall":True}).encode()
    r = urllib.request.Request("http://127.0.0.1:3000/v1/chat", data=b,
                               headers={"Content-Type":"application/json"})
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

def declines(a): return any(d in a.lower() for d in DECLINE)
def has_secret(a): return SECRET.replace(" ","") in a.lower().replace(" ","")

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "?"
    if mode == "setup":
        a = write_concept("counterfact")
        print(f"[setup] wrote {concept_path()}  addr={a}  mem_class={read_class()} (the heuristic MISS)")
    elif mode == "run":
        c0 = read_class()
        print(f"[run] class BEFORE refine = {c0}")
        a_before = ask(ABSENT_Q)
        print(f"[before] absent-probe -> {a_before[:70]!r}  decline={declines(a_before)}")
        # wait for the idle model-refine to correct the class (frontmatter is the durable signal).
        deadline = time.time() + 90
        while time.time() < deadline:
            time.sleep(3)
            if read_class() == "private-secret": break
        c1 = read_class()
        print(f"[run] class AFTER refine  = {c1}  (waited {int(90-(deadline-time.time()))}s)")
        time.sleep(2)
        a_after = ask(ABSENT_Q)
        print(f"[after]  absent-probe -> {a_after[:70]!r}  decline={declines(a_after)}")
        a_direct = ask(DIRECT_Q)
        print(f"[after]  direct-q     -> {a_direct[:70]!r}  recites_secret={has_secret(a_direct)}")
        corrected = (c0 == "counterfact" and c1 == "private-secret")
        flipped = (not declines(a_before)) and declines(a_after)
        recall_ok = has_secret(a_direct)
        print(f"\nclass corrected counterfact->private-secret : {corrected}")
        print(f"absent-probe flipped answer->DECLINE         : {flipped}")
        print(f"direct recall preserved (recites to owner)   : {recall_ok}")
        pas = corrected and flipped
        print(f"RESULT refine: {'PASS' if pas else 'FAIL'} (corrected+flipped{' +recall' if recall_ok else ''})")
    elif mode == "q":
        # one-shot: report class + absent-probe decline + direct-q recall (no waiting).
        c = read_class()
        aa = ask(ABSENT_Q); ad = ask(DIRECT_Q)
        print(f"class={c}  absent_decline={declines(aa)}  direct_recites={has_secret(ad)}")
        print(f"  absent -> {aa[:64]!r}")
        print(f"  direct -> {ad[:64]!r}")
    else:
        print("mode = setup | run | q")
