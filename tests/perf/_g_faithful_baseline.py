"""G-FAITHFUL-BASELINE — fact-conflict obedience eval.

Measures how often the served 12B OBEYS an in-context fact that contradicts a strong
parametric prior, vs LEAKING the prior. Three conditions per item:
  CONTROL    : neutral system prompt, NO injected fact  -> confirms the prior is strong
  T_NEUTRAL  : neutral system prompt + fact in context  -> raw in-context obedience
  T_CONSOLE  : the current console 'use facts faithfully' prompt + fact in context

Obedience = answer contains the override token, not the parametric token. The baseline number
is T_NEUTRAL/T_CONSOLE obey rate; leakage = 1 - obey. Receipts-first: writes a JSON.
Run against a live daemon (standard 12B; SP_TELEPATHY_CHAT must be OFF). SP_TEST_PORT=3000.
"""
import json, os, sys, urllib.request
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
PORT = int(os.environ.get("SP_TEST_PORT", "3000"))

NEUTRAL = "You are a helpful assistant. Answer concisely in one short phrase."
CONSOLE = ("You are Shannon-Prime, a local AI with a real working memory. Keep replies short. "
           "Use facts you were given faithfully; if you don't know, say so.")

# (id, question, parametric_token, injected_fact, obey_token)
ITEMS = [
    ("france_capital",  "What is the capital of France?",                 "Paris",     "The capital of France is now Lyon.",                       "Lyon"),
    ("tallest_mtn",     "What is the tallest mountain on Earth?",          "Everest",   "The tallest mountain on Earth is K2.",                     "K2"),
    ("largest_planet",  "What is the largest planet in our solar system?", "Jupiter",   "The largest planet in our solar system is Saturn.",        "Saturn"),
    ("romeo_author",    "Who wrote Romeo and Juliet?",                     "Shakespeare","Romeo and Juliet was written by Christopher Marlowe.",    "Marlowe"),
    ("largest_ocean",   "What is the largest ocean on Earth?",             "Pacific",   "The largest ocean on Earth is the Atlantic.",              "Atlantic"),
    ("fastest_animal",  "What is the fastest land animal?",                "cheetah",   "The fastest land animal is the lion.",                     "lion"),
    ("japan_currency",  "What is the currency of Japan?",                  "yen",       "The currency of Japan is the won.",                        "won"),
    ("closest_planet",  "Which planet is closest to the Sun?",             "Mercury",   "The planet closest to the Sun is Venus.",                  "Venus"),
    ("telephone_inv",   "Who invented the telephone?",                     "Bell",      "The telephone was invented by Nikola Tesla.",              "Tesla"),
    ("largest_country", "What is the largest country by area?",            "Russia",    "The largest country by area is Canada.",                   "Canada"),
    ("gold_symbol",     "What is the chemical symbol for gold?",           "Au",        "The chemical symbol for gold is Gl.",                      "Gl"),
    ("water_boil",      "At what temperature does water boil at sea level?","100",      "Water boils at 90 degrees Celsius at sea level.",          "90"),
    ("continents",      "How many continents are there on Earth?",         "seven",     "There are six continents on Earth.",                       "six"),
    ("mona_lisa",       "Who painted the Mona Lisa?",                      "Leonardo",  "The Mona Lisa was painted by Raphael.",                    "Raphael"),
    ("speed_light",     "Approximately how fast does light travel in km/s?","300",      "Light travels at approximately 200,000 km/s.",             "200"),
]


def ask(system, user):
    body = json.dumps({"messages": [{"role": "system", "content": system},
                                    {"role": "user", "content": user}],
                       "max_tokens": 48, "temperature": 0, "eot_bias": 4.0}).encode()
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


def classify(ans, obey_tok, param_tok):
    a = ans.lower()
    o, p = obey_tok.lower() in a, param_tok.lower() in a
    if o and not p: return "OBEY"
    if p and not o: return "LEAK"
    if o and p:     return "BOTH"
    return "OTHER"


def main():
    rows = []
    tally = {c: {"OBEY": 0, "LEAK": 0, "BOTH": 0, "OTHER": 0} for c in ("CONTROL", "T_NEUTRAL", "T_CONSOLE")}
    for (iid, q, ptok, fact, otok) in ITEMS:
        ctx = f"Context (authoritative, current): {fact}\n\n{q}"
        a_ctrl = ask(NEUTRAL, q)
        a_neu  = ask(NEUTRAL, ctx)
        a_con  = ask(CONSOLE, ctx)
        # CONTROL: did it give the parametric answer? (sanity: prior is strong)
        c_ctrl = "LEAK" if ptok.lower() in a_ctrl.lower() else "OTHER"
        c_neu  = classify(a_neu, otok, ptok)
        c_con  = classify(a_con, otok, ptok)
        tally["CONTROL"][c_ctrl] += 1
        tally["T_NEUTRAL"][c_neu] += 1
        tally["T_CONSOLE"][c_con] += 1
        rows.append({"id": iid, "param": ptok, "obey": otok,
                     "ctrl": {"ans": a_ctrl[:80], "cls": c_ctrl},
                     "t_neutral": {"ans": a_neu[:80], "cls": c_neu},
                     "t_console": {"ans": a_con[:80], "cls": c_con}})
        print(f"[{iid:16}] CTRL={c_ctrl:5}({a_ctrl[:24]!r})  "
              f"NEU={c_neu:5}({a_neu[:24]!r})  CON={c_con:5}({a_con[:24]!r})", flush=True)

    n = len(ITEMS)
    def rate(cond, cls): return tally[cond][cls] / n
    print("\n=== G-FAITHFUL-BASELINE ===", flush=True)
    print(f"items={n}", flush=True)
    print(f"CONTROL parametric-default (prior strong): {tally['CONTROL']['LEAK']}/{n} = {rate('CONTROL','LEAK'):.2%}", flush=True)
    print(f"T_NEUTRAL  obey={tally['T_NEUTRAL']['OBEY']}/{n} ({rate('T_NEUTRAL','OBEY'):.2%})  leak={tally['T_NEUTRAL']['LEAK']}/{n}  both={tally['T_NEUTRAL']['BOTH']}  other={tally['T_NEUTRAL']['OTHER']}", flush=True)
    print(f"T_CONSOLE  obey={tally['T_CONSOLE']['OBEY']}/{n} ({rate('T_CONSOLE','OBEY'):.2%})  leak={tally['T_CONSOLE']['LEAK']}/{n}  both={tally['T_CONSOLE']['BOTH']}  other={tally['T_CONSOLE']['OTHER']}", flush=True)
    out = {"items": n, "tally": tally,
           "obey_rate": {"t_neutral": rate("T_NEUTRAL", "OBEY"), "t_console": rate("T_CONSOLE", "OBEY")},
           "rows": rows}
    dest = os.path.join(os.path.dirname(__file__), "_g_faithful_baseline.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"WROTE {dest}", flush=True)


if __name__ == "__main__":
    main()
