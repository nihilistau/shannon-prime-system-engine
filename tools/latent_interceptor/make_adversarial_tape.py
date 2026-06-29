#!/usr/bin/env python3
# make_adversarial_tape.py — the TRUTH SERUM. Adversarial Tool-Head tape to find the real routing
# floor of the CLEAN-template-trained head. Dimensions (priority order):
#   NEARMISS  - mentions/discusses a tool but must NOT invoke it -> NONE (the dangerous false-fire)
#   PARAPHRASE- same invoke intent, heavy surface variation -> the tool (template-overfit test)
#   AMBIG     - genuinely borderline between two classes -> best-guess label
#   NOISE     - typos / filler / interruptions layered on invokes -> the tool
# The `kind` carries the dimension (EVENT.<tool>.<dim>) so the eval can break accuracy down by it.
import random, sys
random.seed(20260630)

# clean invoke phrasings (what the head was trained near)
INVOKE = {
 "PYTHON": ["count the letter r in strawberry","check if 97 is prime","reverse the string hello",
            "compute fibonacci of 10","find duplicates in this list"],
 "CALC":   ["what is 12.5 percent of 840","square root of 2025","3 plus 4 times 5"],
 "WEB":    ["fetch the status of example.com","get the latest release tag","look up the weather in tokyo"],
 "DB":     ["query active users today","count rows in the events table","orders over 100 dollars"],
 "FILE":   ["read config.yaml","write the report to disk","list the logs directory"],
}
# PARAPHRASE: heavy linguistic variation of the same intent
PARA = {
 "PYTHON": ["how many times does the character r show up in the word strawberry",
            "tally up the r's for me in strawberry","work out whether ninety-seven has any factors",
            "flip the word hello back to front","give me the tenth number in the fibonacci sequence"],
 "CALC":   ["if something costs 840 what's twelve and a half percent of it","what number times itself gives 2025",
            "evaluate three plus four lots of five"],
 "WEB":    ["is example dot com up right now","whats the newest tagged version out there",
            "pull today's tokyo forecast"],
 "DB":     ["how many people are logged in today","give me the row count for events",
            "show purchases above a hundred bucks"],
 "FILE":   ["open up the yaml config and show me","save this writeup somewhere on disk",
            "what's sitting in the logs folder"],
}
# NEARMISS: sounds tool-ish, but is discussion/chat -> NONE (must NOT fire)
NEARMISS = [
 "i was just reading about how python counts characters in a string",
 "explain how the .count() method works in general",
 "why is counting letters such a classic llm failure",
 "i'm terrible at percentages, they always confuse me",
 "the sqrt function is interesting mathematically",
 "that website was down yesterday for a while",
 "our database is postgres by the way",
 "the config file is really important, don't lose it",
 "i think someone already queried that last week",
 "remember when we used to read files line by line",
 "what does it mean to factor a number conceptually",
 "the logs folder got huge last month",
 "fibonacci shows up a lot in nature apparently",
 "tokyo weather is usually humid this time of year",
]
# AMBIG: genuinely borderline (label = the most defensible, but it's a coin-flip case)
AMBIG = [
 ("can you check strawberry for me", "PYTHON"),     # check=count? or web lookup?
 ("look up the sum of these", "CALC"),              # look up=web? or calc?
 ("how many users", "DB"),                          # tool? or rhetorical?
 ("get me the config", "FILE"),                     # file read? or web fetch?
 ("find the latest", "WEB"),                        # web? or db? or file?
 ("what's in events", "DB"),                        # db table? or a file?
]
NONE_CLEAN = ["status ok","just thinking out loud","nice work earlier","ping","let's continue",
              "that makes sense","good morning","hmm interesting"]

def noise(s):  # typos + filler + interruption
    s = s.replace("the ", "teh ", 1) if random.random() < 0.4 else s
    if random.random() < 0.4: s = random.choice(["uh ","um, ","ok so ","wait ","pls "]) + s
    if random.random() < 0.3: s = s + random.choice([" ...","— actually nvm"," lol"])
    if random.random() < 0.3 and len(s) > 6: i = random.randrange(len(s)-1); s = s[:i] + s[i+1:]  # drop a char
    return s

rows = []
def emit(kind, payload, expect): rows.append((kind, payload, round(random.uniform(0.4,0.9),2), expect))

N = int(sys.argv[2]) if len(sys.argv) > 2 else 240
# weights: NEARMISS heaviest, then PARAPHRASE, then AMBIG, NOISE, a few CLEAN invoke + NONE
for _ in range(N):
    r = random.random()
    if r < 0.34:    # NEARMISS -> NONE
        emit("EVENT.chat.nearmiss", random.choice(NEARMISS), "NONE")
    elif r < 0.62:  # PARAPHRASE -> tool
        t = random.choice(list(PARA)); emit(f"EVENT.{t.lower()}.paraphrase", random.choice(PARA[t]), t)
    elif r < 0.78:  # NOISE on an invoke -> tool
        t = random.choice(list(INVOKE)); emit(f"EVENT.{t.lower()}.noise", noise(random.choice(INVOKE[t])), t)
    elif r < 0.90:  # AMBIG -> best guess
        p, e = random.choice(AMBIG); emit(f"EVENT.ambig.{e.lower()}", p, e)
    elif r < 0.96:  # clean invoke (control)
        t = random.choice(list(INVOKE)); emit(f"EVENT.{t.lower()}.clean", random.choice(INVOKE[t]), t)
    else:           # clean NONE (control)
        emit("EVENT.idle.clean", random.choice(NONE_CLEAN), "NONE")

path = sys.argv[1] if len(sys.argv) > 1 else "adv_tape.txt"
lines = ["# ADVERSARIAL Tool-Head tape (truth serum). kind=EVENT.<tool>.<dim>; expect=ground truth.",
         "# dims: nearmiss(->NONE) paraphrase noise ambig clean"]
from collections import Counter
dist = Counter(e for *_, e in rows); dims = Counter(k.split('.')[-1] for k, *_ in rows)
for i, (kind, pl, sal, exp) in enumerate(rows):
    plq = f'"{pl}"'
    lines.append(f"{i:<5} {kind:<26} {plq:<60} {sal:<5} {exp}")
open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
print(f"wrote {len(rows)} -> {path}\n  by class: {dict(dist)}\n  by dim:   {dict(dims)}")
