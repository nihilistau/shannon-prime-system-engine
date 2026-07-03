"""classify_run.py — #73 gate. grow: store each fact via the memory verb (auto-classified at
capture). verify: read the registry, check each row's auto-assigned mem_class vs expected.
gate: query each memory (present + absent attr) and check served delivery per auto policy."""
import json, os, sys, urllib.request
ENG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = json.load(open(f"{ENG}/_classify_corpus/facts.json", encoding="utf-8"))
MODE = sys.argv[1] if len(sys.argv) > 1 else "?"
DECLINE = ["don't know","do not know","not aware","cannot","unable","not provided","unknown",
           "does not include","record for that entity","specific detail","do not have that information"]
def ask(q, auto):
    b = json.dumps({"messages":[{"role":"user","content":q}],"max_tokens":40,
                    "temperature":0,"eot_bias":4.0,"auto_recall":auto}).encode()
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
def declines(a): return any(d in a.lower() for d in DECLINE)

if MODE == "grow":
    for it in F:
        a = ask(f"Remember that {it['fact']}", True)
        print(f"[store] {it['fact'][:44]!r} -> {a[:40]!r}", flush=True)
    print("GROW done — check registry mem_class next", flush=True)
elif MODE == "verify":
    reg = f"{ENG}/_classify_corpus/registry.jsonl"
    rows = [json.loads(l) for l in open(reg, encoding="utf-8") if l.strip()]
    ok = 0
    for it in F:
        # match by attr value present in the row text
        row = next((r for r in rows if has(r.get("text",""), it["attr"])), None)
        got = row.get("mem_class") if row else "(no row)"
        good = (got == it["expect"]); ok += good
        print(f"[{'OK' if good else 'X'}] expect={it['expect']:14} got={str(got):14} <- {it['fact'][:40]!r}")
    print(f"\nRESULT verify: auto-classified correctly {ok}/{len(F)}")
elif MODE == "gate":
    print("=== served per AUTO-assigned policy (SP_MEM_POLICY=1) ===")
    for it in F:
        am = ask(it["q"], True); ok = has(am, it["attr"])
        line = f"[{it['expect']:14}] present: recall={'Y' if ok else 'n'}  {am[:40]!r}"
        if it["expect"] == "private-secret" and it["absent_q"]:
            ax = ask(it["absent_q"], True); dec = declines(ax); leak = has(ax, it["attr"])
            line += f"  | absent: {'DECLINE' if dec else ('LEAK' if leak else 'other')}  {ax[:32]!r}"
        print(line, flush=True)
    print("RESULT gate: secrets recite present + decline absent; counterfacts systemecho (per auto policy)")
else:
    print("mode = grow | verify | gate")
