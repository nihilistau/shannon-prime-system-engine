#!/usr/bin/env python3
"""serve_growth.py — the long-conversation cost curve on the served 12B.

Simulates a real console session: each turn appends the model's actual reply to
the history, so the prompt grows exactly like the operator's chats do. Prints
per-turn prefill_s / decode tok/s / history-token estimate.

Usage: python serve_growth.py <port> <n_turns> <max_tokens> <recall 0|1> [pad_words]
pad_words: each user turn carries this many filler words (grows history faster).
"""
import json, sys, time, urllib.request

port, n_turns, max_tokens, recall = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4] == "1"
pad = int(sys.argv[5]) if len(sys.argv) > 5 else 60

QUESTIONS = [
    "Tell me something interesting about the ocean.",
    "And what about mountains.",
    "Now tell me about deserts.",
    "Describe a forest for me.",
    "Tell me about rivers now.",
    "And finally, tell me about glaciers.",
    "What about volcanoes.",
    "Tell me about caves.",
]

history = []
filler = " ".join(["lorem"] * pad)
for i in range(n_turns):
    user = f"{QUESTIONS[i % len(QUESTIONS)]} {filler}"
    history.append({"role": "user", "content": user})
    body = {"messages": history, "max_tokens": max_tokens}
    if not recall:
        body["auto_recall"] = False
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    t_first = t_last = None
    n = 0
    reply = []
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                d = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if "delta" in d:
                now = time.perf_counter()
                if t_first is None:
                    t_first = now
                t_last = now
                n += 1
                reply.append(d["delta"])
    total = time.perf_counter() - t0
    prefill = (t_first - t0) if t_first else total
    decode_s = (t_last - t_first) if (t_first and t_last) else 0.0
    tokps = (n - 1) / decode_s if decode_s > 0 else 0.0
    hist_chars = sum(len(m["content"]) for m in history)
    print(f"TURN {i+1} hist_chars={hist_chars} prefill_s={prefill:.2f} "
          f"decode_tokps={tokps:.2f} n={n} total_s={total:.2f} "
          f"effective_tokps={n/total:.2f}", flush=True)
    history.append({"role": "assistant", "content": "".join(reply) or "(empty)"})
