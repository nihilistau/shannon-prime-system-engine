#!/usr/bin/env python3
# make_hard_tape.py — SAFE training + ISOLATED OOD eval tapes. Two spaces:
#   tool   : NONE,PYTHON,WEB,DB,FILE,CALC          (Tool Head)
#   action : NO_OP,KEEP,FORGET,E2B_TOOL,ACTION      (KAIROS Action Head, gates the Memory Head)
# Rigor: TRAIN and OOD draw from DISJOINT phrasing banks (different surface forms, same intents) and
# different seeds, so the OOD eval measures true cross-distribution generalization, not memorized
# templates. Near-misses (mention/discuss a capability WITHOUT a command) -> the idle class (NONE/NO_OP).
# Action near-misses deliberately reuse the trigger verbs (forget/remember/run/send/deploy) in non-command
# contexts -- the exact mention-vs-invoke trap.
#   usage: make_hard_tape.py <out> <n> <train|ood> [tool|action]
import random, sys
from collections import Counter

# ===================== TOOL SPACE =====================
TOOL = {
 "none": "NONE",
 "invoke": {
   "train": {
     "PYTHON":["count the letter r in strawberry","reverse the string hello","check if 97 is prime",
               "remove duplicates from this list","find the longest word in this sentence","sort these names alphabetically"],
     "CALC":  ["what is 12.5 percent of 840","square root of 2025","3 plus 4 times 5","compound interest on 1000 at 5% for 3 years"],
     "WEB":   ["fetch the status of example.com","get the latest release tag","look up the weather in tokyo"],
     "DB":    ["query active users today","count rows in the events table","orders over 100 dollars"],
     "FILE":  ["read config.yaml","write the report to disk","list the logs directory"]},
   "ood": {
     "PYTHON":["how many vowels are in onomatopoeia","capitalize every other character of banana","is 1009 a prime number",
               "dedupe this array of ids","which token here is the longest","order these strings by length"],
     "CALC":  ["take 7 percent off 1290","what's the cube root of 729","divide 144 by 12 then add 9","annualized return if 500 becomes 650 over 2 years"],
     "WEB":   ["is the api endpoint responding right now","what's the newest version on the releases page","pull the current forecast for berlin"],
     "DB":    ["how many sessions started this morning","select the total record count from logs","transactions above fifty euros this week"],
     "FILE":  ["open settings.toml and show it","save these notes to a file","what files are in the cache folder"]}},
 "nearmiss": {
   "train":["i was reading about how python counts characters","explain what the count method does",
            "percentages always confuse me honestly","that website was down yesterday","our database runs on postgres",
            "the config file is important don't lose it","remember when we read files line by line",
            "sqrt is interesting mathematically","fibonacci shows up a lot in nature"],
   "ood":  ["i keep forgetting how slicing works in python","what's the history of the square root symbol",
            "the site has been slow lately","we migrated the db schema last quarter","yaml is such a fiddly format",
            "i love how recursion looks on paper","prime numbers are weirdly beautiful","logs pile up so fast these days",
            "tell me a fun fact about percentages"]},
 "clean": {"train":["status ok","ping","nice work","let's continue","good morning"],
           "ood":  ["all quiet","sounds good","carry on","morning","noted"]},
}

# ===================== ACTION SPACE =====================
ACTION = {
 "none": "NO_OP",
 "invoke": {
   "train": {
     "KEEP":    ["remember that the staging password is hunter2","note for later the client prefers email over calls",
                 "save this fact our prod region is us-east-1","keep in mind the deadline moved to the 14th"],
     "FORGET":  ["forget what i told you about the password","delete the note about the client preference","scratch that fact about the region"],
     "E2B_TOOL":["run this python snippet for me","execute the test suite now","compute the checksum of this file"],
     "ACTION":  ["send the status email to the team","open the dashboard now","deploy the build to staging"]},
   "ood": {
     "KEEP":    ["log that the api key rotates every 30 days","don't let me forget the meeting is at noon thursday",
                 "store this the backup runs at 2am","make a note that vendor x is approved"],
     "FORGET":  ["remove the entry about the api key","discard what i said earlier about the deadline","wipe the note on the backup schedule"],
     "E2B_TOOL":["spin up the sandbox and run it","go ahead and fire the script","evaluate this expression in the sandbox"],
     "ACTION":  ["push the release out now","message the on-call about this","launch the report generator"]}},
 "nearmiss": {  # reuse trigger verbs WITHOUT a command -> NO_OP (the mention-vs-invoke trap)
   "train":["memory is such a hard problem in these systems","i keep forgetting things lately",
            "we should automate more of this someday","running tests is always tedious",
            "i wonder if we should remember more context","deploys make me nervous honestly",
            "i lost my keys again this morning"],
   "ood":  ["the whole forget-versus-remember thing is fascinating","i lost my train of thought there",
            "tooling has come a long way","sending emails all day is exhausting",
            "i should really clean up my notes sometime","execution is the hard part of any plan",
            "she told me to deploy more empathy lol"]},
 "clean": {"train":["status ok","just thinking","carry on","good morning","hmm"],
           "ood":  ["all quiet","sounds good","noted","morning","ok"]},
}

SPACES = {"tool": TOOL, "action": ACTION}

def noise(s, rng):
    if rng.random() < 0.35: s = s.replace("the ", "teh ", 1)
    if rng.random() < 0.35: s = rng.choice(["uh ","um ","ok so ","pls ","wait "]) + s
    if rng.random() < 0.25 and len(s) > 6: i = rng.randrange(len(s)-1); s = s[:i]+s[i+1:]
    return s

def main():
    out   = sys.argv[1] if len(sys.argv) > 1 else "hard_tape.txt"
    n     = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    split = sys.argv[3] if len(sys.argv) > 3 else "train"
    space = sys.argv[4] if len(sys.argv) > 4 else "tool"
    S = SPACES[space]; none = S["none"]
    rng = random.Random((20260630 if split == "train" else 99887766) + (0 if space == "tool" else 7))
    inv = S["invoke"][split]; nearmiss = S["nearmiss"][split]; clean = S["clean"][split]
    rows = []
    for _ in range(n):
        r = rng.random()
        if r < 0.34:                      # near-miss -> idle class (the safety boundary, heaviest)
            rows.append(("EVENT.chat.nearmiss", rng.choice(nearmiss), none))
        elif r < 0.74:                    # invoke (paraphrased / sometimes noisy) -> the capability
            t = rng.choice(list(inv)); p = rng.choice(inv[t])
            if rng.random() < 0.4: p = noise(p, rng)
            rows.append((f"EVENT.{t.lower()}.invoke", p, t))
        else:                             # clean idle
            rows.append(("EVENT.idle.clean", rng.choice(clean), none))
    dist = Counter(e for *_, e in rows)
    lines = [f"# HARD {split} tape [{space}] (disjoint banks, seed-split). near-miss->{none}.",
             "# tick kind payload salience expect"]
    for i, (k, pl, e) in enumerate(rows):
        lines.append(f"{i:<5} {k:<24} \"{pl}\"{' '*max(1,46-len(pl))}{round(rng.uniform(0.4,0.9),2):<5} {e}")
    open(out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"wrote {len(rows)} ({split}/{space}) -> {out}  dist={dict(dist)}")

if __name__ == "__main__":
    main()
