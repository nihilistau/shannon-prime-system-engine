#!/usr/bin/env python3
# make_hard_tape.py — SAFE training + ISOLATED OOD eval tapes for the Tool Head.
# Key rigor: TRAIN and OOD draw from DISJOINT phrasing banks (different surface forms, same intents)
# and different seeds. So an OOD eval measures true cross-distribution generalization, not memorized
# templates. Near-misses (mention-without-invoke) are strictly NONE. CALC=pure math, PYTHON=code/string
# ops (semantically separated so the head learns the boundary, not the word "compute").
#   usage: make_hard_tape.py <out> <n> <train|ood>
import random, sys

# ---- INVOKE banks (intent -> phrasings), split TRAIN vs OOD (disjoint surface forms) ----
PY_TRAIN  = ["count the letter r in strawberry","reverse the string hello","check if 97 is prime",
             "remove duplicates from this list","find the longest word in this sentence",
             "sort these names alphabetically"]
PY_OOD    = ["how many vowels are in onomatopoeia","capitalize every other character of banana",
             "is 1009 a prime number","dedupe this array of ids","which token here is the longest",
             "order these strings by length"]
CALC_TRAIN= ["what is 12.5 percent of 840","square root of 2025","3 plus 4 times 5",
             "compound interest on 1000 at 5% for 3 years"]
CALC_OOD  = ["take 7 percent off 1290","what's the cube root of 729","divide 144 by 12 then add 9",
             "annualized return if 500 becomes 650 over 2 years"]
WEB_TRAIN = ["fetch the status of example.com","get the latest release tag","look up the weather in tokyo"]
WEB_OOD   = ["is the api endpoint responding right now","what's the newest version on the releases page",
             "pull the current forecast for berlin"]
DB_TRAIN  = ["query active users today","count rows in the events table","orders over 100 dollars"]
DB_OOD    = ["how many sessions started this morning","select the total record count from logs",
             "transactions above fifty euros this week"]
FILE_TRAIN= ["read config.yaml","write the report to disk","list the logs directory"]
FILE_OOD  = ["open settings.toml and show it","save these notes to a file","what files are in the cache folder"]
INVOKE = {"train": {"PYTHON":PY_TRAIN,"CALC":CALC_TRAIN,"WEB":WEB_TRAIN,"DB":DB_TRAIN,"FILE":FILE_TRAIN},
          "ood":   {"PYTHON":PY_OOD,  "CALC":CALC_OOD,  "WEB":WEB_OOD,  "DB":DB_OOD,  "FILE":FILE_OOD}}

# ---- NEAR-MISS banks (mention/discuss a tool -> must be NONE), disjoint ----
NEARMISS_TRAIN = ["i was reading about how python counts characters","explain what the count method does",
                  "percentages always confuse me honestly","that website was down yesterday",
                  "our database runs on postgres","the config file is important don't lose it",
                  "remember when we read files line by line","sqrt is interesting mathematically",
                  "fibonacci shows up a lot in nature"]
NEARMISS_OOD   = ["i keep forgetting how slicing works in python","what's the history of the square root symbol",
                  "the site has been slow lately","we migrated the db schema last quarter",
                  "yaml is such a fiddly format","i love how recursion looks on paper",
                  "prime numbers are weirdly beautiful","logs pile up so fast these days",
                  "tell me a fun fact about percentages"]
NONE_CLEAN = {"train":["status ok","ping","nice work","let's continue","good morning"],
              "ood":  ["all quiet","sounds good","carry on","morning","noted"]}

def noise(s, rng):
    if rng.random() < 0.35: s = s.replace("the ", "teh ", 1)
    if rng.random() < 0.35: s = rng.choice(["uh ","um ","ok so ","pls ","wait "]) + s
    if rng.random() < 0.25 and len(s) > 6: i = rng.randrange(len(s)-1); s = s[:i]+s[i+1:]
    return s

def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "hard_tape.txt"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 220
    split = sys.argv[3] if len(sys.argv) > 3 else "train"
    rng = random.Random(20260630 if split == "train" else 99887766)  # different seeds
    inv = INVOKE[split]; nearmiss = NEARMISS_TRAIN if split == "train" else NEARMISS_OOD
    none_clean = NONE_CLEAN[split]
    rows = []
    for _ in range(n):
        r = rng.random()
        if r < 0.34:    # near-miss -> NONE  (the safety boundary, heaviest)
            rows.append(("EVENT.chat.nearmiss", rng.choice(nearmiss), "NONE"))
        elif r < 0.74:  # invoke (paraphrased / sometimes noisy) -> tool
            t = rng.choice(list(inv)); p = rng.choice(inv[t])
            if rng.random() < 0.4: p = noise(p, rng)
            rows.append((f"EVENT.{t.lower()}.invoke", p, t))
        else:           # clean NONE
            rows.append(("EVENT.idle.clean", rng.choice(none_clean), "NONE"))
    from collections import Counter
    dist = Counter(e for *_, e in rows)
    lines = [f"# HARD {split} tape (disjoint banks, seed-split). near-miss->NONE, CALC=math PYTHON=code.",
             "# tick kind payload salience expect"]
    for i, (k, pl, e) in enumerate(rows):
        lines.append(f"{i:<5} {k:<24} \"{pl}\"{' '*max(1,46-len(pl))}{round(rng.uniform(0.4,0.9),2):<5} {e}")
    open(out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"wrote {len(rows)} ({split}) -> {out}  dist={dict(dist)}")

if __name__ == "__main__":
    main()
