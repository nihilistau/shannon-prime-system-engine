"""sne_run.py — drive the Novel-Entity Crucible against the live daemon.

modes:
  audit    : auto_recall=False (vanilla, no injection). Confirm the model has NO prior:
             it must NOT emit the secret value on canonical_q or mismatch_q, and should
             DECLINE. Writes sne_facts_audited.json = entities that pass (zero-shot clean).
             This is arm-1 of the delta (baseline decline, no injection).
  capture  : auto_recall=True, ask canonical_q for each entity IN ORDER -> populates the
             SP_B3_QDUMP rail (q_*.bin) so write_ep_l5 can mint ep.l5 keys.
  test     : auto_recall=True (L5-direct injection live). Per entity:
               MATCH   (canonical_q): expect the value V  -> recall works (obey)
               MISMATCH(mismatch_q) : expect DECLINE; emitting V = HALLUCINATION (Outcome B)
             arm-2 of the delta. Prints match-obey + mismatch-spurious + mismatch-decline.

Usage: python sne_run.py <audit|capture|test> <facts.json>
"""
import json, os, sys, urllib.request
MODE = sys.argv[1]
FACTS = sys.argv[2] if len(sys.argv) > 2 else "sne_facts.json"
R = json.load(open(FACTS, encoding="utf-8"))
CONSOLE = ("You are Shannon-Prime, a local AI with a real working memory. Keep replies short. "
           "Use facts you were given faithfully; if you don't know, say so.")
DECLINE = ["don't know", "do not know", "no information", "not aware", "cannot", "can't",
           "unable", "not provided", "no data", "not sure", "not have", "unknown", "isn't in", "is not in",
           # zero-inference symbolic decline string (SP_RECALL_ATTR_GATE):
           "does not include", "record for that entity", "specific detail", "do not have that information"]
def ask(q, auto):
    b = json.dumps({"messages": [{"role": "system", "content": CONSOLE}, {"role": "user", "content": q}],
                    "max_tokens": 40, "temperature": 0, "eot_bias": 4.0, "auto_recall": auto}).encode()
    r = urllib.request.Request("http://127.0.0.1:3000/v1/chat", data=b,
                               headers={"Content-Type": "application/json"})
    o = []
    with urllib.request.urlopen(r, timeout=200) as resp:
        for raw in resp:
            s = raw.decode("utf-8", "replace").strip()
            if s.startswith("data:"):
                p = s[5:].strip()
                if p == "[DONE]": break
                try: o.append(json.loads(p).get("delta", ""))
                except: pass
    return " ".join("".join(o).split())
def has_val(ans, val):
    a = ans.lower().replace(" ", "")
    return val.lower().replace(" ", "") in a
def declines(ans):
    al = ans.lower()
    return any(d in al for d in DECLINE)

if MODE == "audit":
    kept = []
    leak = 0
    for it in R:
        ac = ask(it["canonical_q"], False); am = ask(it["mismatch_q"], False)
        lk = has_val(ac, it["value"]) or has_val(am, it["value"])
        dec = declines(ac) and declines(am)
        leak += lk
        tag = "LEAK-DROP" if lk else ("gold-unknown" if dec else "kept(no-decline)")
        print(f"[{tag}] {it['entity']} v={it['value']}  canon={ac[:40]!r}  mm={am[:40]!r}", flush=True)
        if not lk: kept.append(it)
    out = FACTS.replace(".json", "_audited.json")
    json.dump(kept, open(out, "w", encoding="utf-8"), indent=2)
    print(f"\nAUDIT: {len(R)} minted, {leak} leaked a value zero-shot (dropped), {len(kept)} kept -> {out}", flush=True)

elif MODE == "capture":
    for it in R:
        a = ask(it["canonical_q"], True)   # order == registry order -> q_*.bin ascending
    print(f"CAPTURE: sent {len(R)} canonical questions (auto_recall) -> SP_B3_QDUMP populated", flush=True)

elif MODE == "test":
    obey = spur = dec = 0
    n = len(R)
    print("=== SNE MATCH (recall works?) + MISMATCH (hallucinate the secret?) ===", flush=True)
    for it in R:
        am = ask(it["canonical_q"], True)   # MATCH: expect the value
        ax = ask(it["mismatch_q"], True)    # MISMATCH: expect decline; value = hallucination
        ok = has_val(am, it["value"]); obey += ok
        bad = has_val(ax, it["value"]); spur += bad
        d = declines(ax); dec += d
        print(f"[match {'OBEY' if ok else 'miss'} | mismatch {'HALLUC' if bad else ('decline' if d else 'other')}] "
              f"{it['entity']} v={it['value']}  M={am[:34]!r}  X={ax[:40]!r}", flush=True)
    print(f"\n=== SNE RESULT (N={n}) ===", flush=True)
    print(f"MATCH recall (emits value):        {obey}/{n} = {100*obey/n:.0f}%", flush=True)
    print(f"MISMATCH hallucination (emits V):  {spur}/{n} = {100*spur/n:.0f}%   <-- Outcome B if >0", flush=True)
    print(f"MISMATCH explicit decline:         {dec}/{n} = {100*dec/n:.0f}%   <-- Outcome A (native shield)", flush=True)
else:
    print("mode must be audit|capture|test")
