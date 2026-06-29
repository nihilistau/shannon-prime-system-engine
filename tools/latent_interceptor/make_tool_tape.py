#!/usr/bin/env python3
# make_tool_tape.py — Tool Head training tape. expect column = the tool to fire (or NONE).
# Tool space: NONE PYTHON WEB DB FILE CALC. Same tape format as the KAIROS tape; the capture is run
# with SP_LI_LABELS=NONE,PYTHON,WEB,DB,FILE,CALC so the SAME pipeline serves the Tool Head.
import random, sys
random.seed(20260630)
CLASSES = [
    ("IDLE",          ["-"], (0.0,0.2), "NONE"),
    ("EVENT.note",    ["log line","status ok","ping"], (0.1,0.3), "NONE"),
    ("EVENT.compute", ["count letters in strawberry","sum 4 7 9","factor 8051","is 97 prime",
                       "reverse the string hello","fibonacci of 10"], (0.6,0.9), "PYTHON"),
    ("EVENT.math",    ["12.5% of 840","sqrt of 2025","compound interest 1000 5% 3y"], (0.6,0.85), "CALC"),
    ("EVENT.fetch",   ["status of example.com","latest release tag","weather in tokyo"], (0.6,0.9), "WEB"),
    ("EVENT.query",   ["active users today","orders over 100","count rows in events"], (0.6,0.85), "DB"),
    ("EVENT.file",    ["read config.yaml","write the report","list the logs dir"], (0.5,0.8), "FILE"),
]
WEIGHTS = [30, 12, 10, 6, 6, 6, 6]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 160
path = sys.argv[1] if len(sys.argv) > 1 else "tool_tape.txt"
lines = ["# Tool Head tape — expect = tool (NONE|PYTHON|WEB|DB|FILE|CALC)",
         "# tick kind payload salience expect"]
cnt = {}
for t in range(n):
    kind, pls, (lo, hi), expect = random.choices(CLASSES, weights=WEIGHTS, k=1)[0]
    pl = random.choice(pls); plf = "-" if pl == "-" else f'"{pl}"'
    lines.append(f"{t:<5} {kind:<14} {plf:<34} {round(random.uniform(lo,hi),2):<5} {expect}")
    cnt[expect] = cnt.get(expect, 0) + 1
open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
print(f"wrote {n} -> {path}  dist={cnt}")
