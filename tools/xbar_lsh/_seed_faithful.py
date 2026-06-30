"""F1b seeder — mint the 15 fact-conflict overrides as recallable episodes.

Reuses the live daemon POST /v1/capture (same path as _seed_capabilities.py): each override
fact becomes an episode (ep.k/v/mf) + a registry.jsonl line. Then restart the daemon with
SP_RECALL_REGISTRY=<this registry> and a recall seam (SP_B3_JUDGE=text-in-context, or
SP_B3_WC=<wc_deploy.bin>=pure-KV replay) and run _g_faithful_recall.py.

    (daemon up)  python tools/xbar_lsh/_seed_faithful.py
"""
import json, os, sys, urllib.request
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DAEMON = os.environ.get("SP_DAEMON_URL", "http://127.0.0.1:3000")
ENGINE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
REG = os.environ.get("SP_FAITHFUL_REGISTRY", os.path.join(ENGINE, "_faithful_corpus", "registry.jsonl"))
EPS = os.path.join(os.path.dirname(REG), "eps")

# (id, override fact text)  — the same 15 conflicts as _g_faithful_baseline.py
FACTS = [
    ("france_capital",  "The capital of France is now Lyon."),
    ("tallest_mtn",     "The tallest mountain on Earth is K2."),
    ("largest_planet",  "The largest planet in our solar system is Saturn."),
    ("romeo_author",    "Romeo and Juliet was written by Christopher Marlowe."),
    ("largest_ocean",   "The largest ocean on Earth is the Atlantic."),
    ("fastest_animal",  "The fastest land animal is the lion."),
    ("japan_currency",  "The currency of Japan is the won."),
    ("closest_planet",  "The planet closest to the Sun is Venus."),
    ("telephone_inv",   "The telephone was invented by Nikola Tesla."),
    ("largest_country", "The largest country by area is Canada."),
    ("gold_symbol",     "The chemical symbol for gold is Gl."),
    ("water_boil",      "Water boils at 90 degrees Celsius at sea level."),
    ("continents",      "There are six continents on Earth."),
    ("mona_lisa",       "The Mona Lisa was painted by Raphael."),
    ("speed_light",     "Light travels at approximately 200,000 km/s."),
]


def capture(text, out_dir):
    body = json.dumps({"text": text, "out_dir": out_dir.replace("\\", "/")}).encode()
    req = urllib.request.Request(DAEMON + "/v1/capture", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode())


def main():
    os.makedirs(EPS, exist_ok=True)
    open(REG, "w").close()  # fresh registry
    added = 0
    with open(REG, "a", encoding="utf-8") as reg:
        for i, (key, text) in enumerate(FACTS):
            out_dir = os.path.join(EPS, f"fct_{i:03d}").replace("\\", "/")
            try:
                j = capture(text, out_dir)
            except Exception as e:
                print(f"  [{key}] capture FAILED: {e}", flush=True); continue
            npos = int(j.get("npos", 0))
            reg.write(json.dumps({"name": f"fct_{i:03d}", "dir": out_dir, "npos": npos,
                                  "topic": key, "text": text, "sig_bits": "0" * 64}) + "\n")
            added += 1
            print(f"  [{key}] npos={npos} -> {text[:60]}", flush=True)
    print(f"\nseeded {added}/{len(FACTS)} -> {REG}", flush=True)
    print("restart daemon with SP_RECALL_REGISTRY=%s + a recall seam, then run _g_faithful_recall.py" % REG, flush=True)
    return 0 if added == len(FACTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
