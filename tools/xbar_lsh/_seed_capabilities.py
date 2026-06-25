"""Seed the model's CAPABILITIES into the SERVED registry as recallable self-knowledge.

Mints each capability fact as an episode (ep.k/v/mf) via the live daemon's POST /v1/capture
and appends a registry line, so the served chat can RECALL 'how do I use myself' (the init
primer's deep tier). Run against the live daemon, then restart it so the new episodes load.

    python tools/xbar_lsh/_seed_capabilities.py
"""
import json
import os
import sys
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DAEMON = os.environ.get("SP_DAEMON_URL", "http://127.0.0.1:3000")
ENGINE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
REG = os.environ.get("SP_RECALL_REGISTRY", os.path.join(ENGINE, "_seed_corpus", "registry.jsonl"))
EPS = os.path.join(os.path.dirname(REG), "eps")

CAPABILITIES = [
    ("store-memory",   "I can store facts from our conversation in my long-term memory and recall them in later turns."),
    ("forget-update",  "I can forget a memory when you ask, and I automatically update or merge memories when a new fact supersedes an old one."),
    ("run-tools",      "I can run Python code and use tools to compute things or take actions when you ask."),
    ("conversation-memory", "I store our past conversations both in full and as short summaries, so I can recall the gist or dig into the full transcript."),
    ("self-maintenance", "I review and tidy my own memory between turns, forgetting redundant facts and consolidating related ones."),
    ("local-private",  "I run entirely locally on a single RTX 2060, so our conversation and my memory stay on this machine."),
]


def capture(text, out_dir):
    body = json.dumps({"text": text, "out_dir": out_dir.replace("\\", "/")}).encode()
    req = urllib.request.Request(DAEMON + "/v1/capture", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode())


def main():
    os.makedirs(EPS, exist_ok=True)
    added = 0
    with open(REG, "a", encoding="utf-8") as reg:
        for i, (key, text) in enumerate(CAPABILITIES):
            out_dir = os.path.join(EPS, f"cap_{i:03d}").replace("\\", "/")
            try:
                j = capture(text, out_dir)
            except Exception as e:
                print(f"  [{key}] capture FAILED: {e}")
                continue
            npos = int(j.get("npos", 0))
            line = {"name": f"cap_{i:03d}", "dir": out_dir, "npos": npos,
                    "topic": key, "text": text, "sig_bits": "0" * 64}
            reg.write(json.dumps(line) + "\n")
            added += 1
            print(f"  [{key}] npos={npos} -> {text[:60]}...")
    print(f"\nseeded {added}/{len(CAPABILITIES)} capabilities into {REG}")
    print("restart the daemon to load them into recall.")
    return 0 if added == len(CAPABILITIES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
