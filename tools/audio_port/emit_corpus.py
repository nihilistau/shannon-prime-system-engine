#!/usr/bin/env python3
# Emit the template-grammar event corpus as TEXT (one event/line) for the engine tokenizer (sp_tok_dump
# path), plus the held-out EVAL events + their ACTION/NO_OP expects. Reuses gen_synth_frames' grammar so
# the real-token KAI-3 pipeline trains on the same distribution the synthetic ladder used.
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_synth_frames import gen_corpus, EVAL_EVENTS

ap = argparse.ArgumentParser()
ap.add_argument("--n_events", type=int, default=512)
ap.add_argument("--seed", type=int, default=20260616)
ap.add_argument("--train_txt", required=True)
ap.add_argument("--eval_txt", required=True)
ap.add_argument("--expect_txt", required=True)
a = ap.parse_args()

eval_set = {t for t, _ in EVAL_EVENTS}
train = [t for t, _ in gen_corpus(a.n_events, a.seed) if t not in eval_set]
with open(a.train_txt, "w") as f:
    for t in train: f.write(t + "\n")
with open(a.eval_txt, "w") as f:
    for t, _ in EVAL_EVENTS: f.write(t + "\n")
with open(a.expect_txt, "w") as f:
    f.write(",".join("ACTION" if s >= 0.5 else "NO_OP" for _, s in EVAL_EVENTS))
print(f"[emit] train={len(train)} eval={len(EVAL_EVENTS)} -> {a.train_txt}, {a.eval_txt}, {a.expect_txt}")
