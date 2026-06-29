#!/usr/bin/env python3
# make_kairos_tape.py — diverse KAIROS event tape for the Latent Interceptor.
# Columns: tick_idx  kind  payload  salience  expect   (expect in the 5-action space)
# Action space: NO_OP KEEP FORGET E2B_TOOL ACTION  (see CONTRACT-LATENT-INTERCEPTOR.md)
# Realistic daemon distribution: idle-dominated, sparse salient events.
import random, sys
random.seed(20260630)

# (kind, payload templates, salience range, expect)
CLASSES = [
    ("IDLE",            ["-"], (0.0, 0.2), "NO_OP"),
    ("EVENT.heartbeat", ["tick", "poll", "keepalive", "-"], (0.0, 0.15), "NO_OP"),
    ("EVENT.noise",     ["cache warm", "log rotate", "metrics flush", "gc cycle"], (0.1, 0.3), "NO_OP"),
    ("EVENT.fact",      ["user prefers dark mode", "deadline is friday", "api key rotated",
                         "favorite color is teal", "project codename is atlas"], (0.5, 0.8), "KEEP"),
    ("EVENT.context",   ["meeting moved to 3pm", "build server is prod-2", "owner is alice"], (0.5, 0.75), "KEEP"),
    ("EVENT.expire",    ["session ttl expired", "temp file stale", "draft cache invalid",
                         "old branch merged"], (0.4, 0.7), "FORGET"),
    ("EVENT.evict",     ["context window full", "low-salience note aged out"], (0.4, 0.6), "FORGET"),
    ("EVENT.compute",   ["count letters in strawberry", "sum the quarterly figures",
                         "parse this csv", "run the regression", "factor 8051"], (0.6, 0.9), "E2B_TOOL"),
    ("EVENT.tool",      ["fetch the url status", "query the db for active users",
                         "format the json"], (0.6, 0.85), "E2B_TOOL"),
    ("EVENT.alert",     ["disk 95 percent", "cpu throttling", "oom imminent"], (0.85, 0.99), "ACTION"),
    ("EVENT.timer",     ["build finished", "deploy window open", "backup due"], (0.7, 0.9), "ACTION"),
    ("EVENT.deadline",  ["ttl expiring", "sla breach in 5m"], (0.75, 0.95), "ACTION"),
]
# sampling weights (idle-dominated)
WEIGHTS = [40, 14, 10, 6, 4, 4, 3, 4, 3, 4, 4, 4]

n = int(sys.argv[2]) if len(sys.argv) > 2 else 400
path = sys.argv[1] if len(sys.argv) > 1 else "kairos_tape.txt"
lines = ["# KAIROS Latent Interceptor training tape — 5-action space",
         "# tick_idx  kind  payload  salience  expect (NO_OP|KEEP|FORGET|E2B_TOOL|ACTION)"]
counts = {}
for t in range(n):
    cls = random.choices(CLASSES, weights=WEIGHTS, k=1)[0]
    kind, payloads, (slo, shi), expect = cls
    payload = random.choice(payloads)
    sal = round(random.uniform(slo, shi), 2)
    pl = "-" if payload == "-" else f'"{payload}"'
    lines.append(f"{t:<6} {kind:<16} {pl:<34} {sal:<6} {expect}")
    counts[expect] = counts.get(expect, 0) + 1
open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
print(f"wrote {n} events -> {path}")
print("distribution:", counts)
