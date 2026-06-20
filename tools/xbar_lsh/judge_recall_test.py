#!/usr/bin/env python3
"""
G-JUDGE-RECALL : does a small GENERATIVE judge break the open-set recall wall?

The div corpus is the EXACT corpus that defeated every GEOMETRIC signal (q.K,
cosine, LSH C2 sig, the W_c head, causal self-ablation). A generative judge reads
the candidate memory TEXTS (the words, not post-RoPE vectors) and picks the one
that answers the query -- or NONE.

DEPLOYMENT-REALISTIC mode (default = 'bounded'):
  The two-gate organism's recall gate runs over a KAIROS-BOUNDED working set
  (~8-16 candidates), NOT the whole store. So each query is judged against a
  window of K candidates = the true needle + (K-1) random distractors, position
  SHUFFLED. One call per query. Foreign queries get K random needles (no truth)
  and must answer NONE. This isolates the judge's discrimination at the N the
  system actually serves, and is fast (the decode path prefills token-by-token,
  so cost is ~linear in prompt tokens -- a 90-wide prompt is 3601 tokens / ~116s,
  pathological; a 12-wide window is ~300 tokens / ~8s).

Also available: --mode mono (all N at once; documents lost-in-the-middle) and
--mode tourney (windowed cascade over all N).

Gate (the one that broke every prior signal):  recall@1 >= 80% & foreign-reject == 100%.
"""
import argparse, json, os, random, re, sys, time, urllib.request

def load_corpus(cdir):
    man = []
    with open(os.path.join(cdir, "corpus_manifest.jsonl"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                man.append(json.loads(line))
    foreign = []
    fp = os.path.join(cdir, "foreign_queries.txt")
    if os.path.exists(fp):
        with open(fp, encoding="utf-8") as f:
            foreign = [l.strip() for l in f if l.strip()]
    return man, foreign

def chat(host, prompt, max_tokens=8):
    body = json.dumps({"messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "stop": [], "temperature": 0.0}).encode()
    req = urllib.request.Request(host + "/v1/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    out = []
    with urllib.request.urlopen(req, timeout=240) as r:
        for raw in r:
            s = raw.decode("utf-8", "replace").strip()
            if not s.startswith("data:"):
                continue
            payload = s[5:].strip()
            if payload == "[DONE]":
                break
            try:
                ev = json.loads(payload)
            except Exception:
                continue
            if "delta" in ev:
                out.append(ev["delta"])
            elif "error" in ev:
                raise RuntimeError(ev["error"])
    return "".join(out)

import string as _string
_TAGPOOL = [a+b+c for a in "BCDFGHJKLMNPQRSTVWXZ" for b in "0123456789" for c in "BCDFGHJKLMNPQRSTVWXZ"]

JUDGE_TMPL = (
    "You are a memory index. Each entry below has a TAG in [brackets]. "
    "Read the QUESTION and reply with ONLY the tag of the single entry that "
    "directly answers it. If no entry answers it, reply NONE.\n\n"
    "{entries}\n"
    "QUESTION: {q}\n"
    "Tag of the answer (or NONE):"
)

def ask_tags(host, q, texts, tags):
    entries = "\n".join(f"[{tg}] {t}" for tg, t in zip(tags, texts))
    reply = chat(host, JUDGE_TMPL.format(entries=entries, q=q), max_tokens=6)
    up = reply.upper()
    # match a tag the model copied; longest/exact tag wins
    for i, tg in enumerate(tags):
        if tg in up:
            return i, reply       # 0-based candidate index
    return None, reply            # NONE / no tag found

VERIFY_TMPL = ("Memory entry: \"{t}\"\nQuestion: {q}\n"
               "Does this memory entry directly provide the answer to the question? "
               "Answer strictly yes or no.")

def verify(host, q, text):
    reply = chat(host, VERIFY_TMPL.format(t=text, q=q), max_tokens=4).strip().lower()
    return reply.startswith("y")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="_needle_corpus_div")
    ap.add_argument("--host", default="http://127.0.0.1:3001")
    ap.add_argument("--out", default="tests/fixtures/chat_fullstack/G-JUDGE-RECALL.log")
    ap.add_argument("--seed", type=int, default=20260620)
    ap.add_argument("--mode", choices=["bounded", "mono", "tourney"], default="bounded")
    ap.add_argument("--window", type=int, default=12, help="bounded/tourney window size K")
    ap.add_argument("--limit", type=int, default=0, help="cap matched needles")
    ap.add_argument("--foreign-limit", type=int, default=0)
    ap.add_argument("--paraphrases", action="store_true")
    ap.add_argument("--verify", action="store_true", help="enable focused relevance verify stage")
    args = ap.parse_args()

    man, foreign = load_corpus(args.corpus)
    needles = [m for m in man if not m["id"].startswith("ctrl")]  # ctrl = admission's job
    alln = needles[:]
    if args.limit:
        needles = needles[:args.limit]
    if args.foreign_limit:
        foreign = foreign[:args.foreign_limit]
    rng = random.Random(args.seed)
    K = args.window

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    log = open(args.out, "w", encoding="utf-8")
    t0 = time.time()
    def emit(s):
        print(s); log.write(s + "\n"); log.flush()

    emit(f"# G-JUDGE-RECALL corpus={args.corpus} host={args.host} seed={args.seed}")
    emit(f"# mode={args.mode} window/K={K} pool={len(alln)} matched={len(needles)} "
         f"foreign={len(foreign)} paraphrases={args.paraphrases}")

    hit = tot = frej = ftot = 0

    def window_for(truth):
        pool = [m for m in alln if m is not truth]
        distract = rng.sample(pool, min(K-1, len(pool)))
        cands = distract + [truth]
        rng.shuffle(cands)
        return cands

    # matched
    for m in needles:
        qs = [m["query"]] + ([p for p in m.get("paraphrases", []) if p != m["query"]]
                             if args.paraphrases else [])
        for q in qs:
            if args.mode == "bounded":
                cands = window_for(m)
                gt = cands.index(m)
                tags = rng.sample(_TAGPOOL, len(cands))
                ch, reply = ask_tags(args.host, q, [c["text"] for c in cands], tags)
                if args.verify and ch is not None and not verify(args.host, q, cands[ch]["text"]):
                    ch = None  # pick failed the focused relevance verify -> reject
            else:  # mono over the whole pool
                cands = alln[:]; rng.shuffle(cands)
                gt = cands.index(m)
                tags = rng.sample(_TAGPOOL, len(cands))
                ch, reply = ask_tags(args.host, q, [c["text"] for c in cands], tags)
            ok = (ch == gt)
            tot += 1; hit += int(ok)
            emit(f"[match] {'OK ' if ok else 'MISS'} {m['id']:14} K={len(cands)} gt={gt:2} got={ch} "
                 f":: {q[:50]!r} -> {reply.strip()[:14]!r}")

    # foreign
    for q in foreign:
        if args.mode == "bounded":
            cands = rng.sample(alln, min(K, len(alln)))
        else:
            cands = alln[:]; rng.shuffle(cands)
        tags = rng.sample(_TAGPOOL, len(cands))
        ch, reply = ask_tags(args.host, q, [c["text"] for c in cands], tags)
        accept = (ch is not None) and (verify(args.host, q, cands[ch]["text"]) if args.verify else True)
        ok = (not accept)
        ftot += 1; frej += int(ok)
        emit(f"[forgn] {'REJECT' if ok else 'FALSEFIRE'} K={len(cands)} got={ch} "
             f":: {q[:50]!r} -> {reply.strip()[:14]!r}")

    rec = hit/max(tot,1); rej = frej/max(ftot,1)
    g = (rec >= 0.80 and rej >= 1.0)
    emit(f"\n================ RESULT  (elapsed {time.time()-t0:.0f}s) ================")
    emit(f"mode={args.mode} K={K}")
    emit(f"recall@1        : {hit}/{tot} = {100*rec:.1f}%")
    emit(f"foreign-reject  : {frej}/{ftot} = {100*rej:.1f}%")
    emit(f"GATE (>=80% recall & 100% reject) : {'GREEN' if g else 'RED'}")
    log.close()
    sys.exit(0 if g else 1)

if __name__ == "__main__":
    main()
