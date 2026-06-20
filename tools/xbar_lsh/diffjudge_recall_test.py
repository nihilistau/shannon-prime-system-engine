#!/usr/bin/env python3
"""
G-DIFFJUDGE-1 : does the DiffusionGemma judge match/beat the AR judge (85.7% recall@1)?

Same tag-based generative judge as judge_recall_test.py (the AR oracle), but driven
against the DiffusionGemma block-diffusion runner (llama.cpp PR 24423) instead of the
HTTP daemon. The DiffusionGemma forward writes the answer into a fixed bidirectional
canvas (entropy-bound denoising) -- the Phase-5 bet is that bidirectional + a tag-tight
prompt beats the AR judge's position bias / parametric leak.

Driver: spawns `llama-diffusion-cli -m <gguf> -cnv ...` (model resident, one load), feeds
one judge prompt per turn over stdin, parses the reply off stdout, then `/clear` to drop
history so each query is independent (matches the KAIROS-bounded deployment regime).

Bounded mode (default): each query judged against K candidates = true needle + (K-1)
random distractors, position-shuffled, unique copy-able [TAG] per candidate. Foreign
queries get K random needles (no truth) and must answer NONE.

GATE: recall@1 >= 85.7%  AND report foreign-reject %.
"""
import argparse, json, os, random, re, subprocess, sys, threading, time
from queue import Queue, Empty

# ---- corpus ----------------------------------------------------------------
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

_TAGPOOL = [a+b+c for a in "BCDFGHJKLMNPQRSTVWXZ" for b in "0123456789" for c in "BCDFGHJKLMNPQRSTVWXZ"]

JUDGE_TMPL = (
    "You are a memory index. Each entry below has a TAG in [brackets]. "
    "Read the QUESTION and reply with ONLY the tag of the single entry that "
    "directly answers it. If no entry answers it, reply NONE. "
    "{entries} "
    "QUESTION: {q} "
    "Tag of the answer (or NONE):"
)

# ---- diffusion-cli driver --------------------------------------------------
class DiffCli:
    """Resident llama-diffusion-cli -cnv subprocess; one prompt per turn, /clear between.

    The cli prints the interactive prompt as a bare '> ' with NO trailing newline, so a
    line-based reader blocks forever. We read raw bytes in a thread into a shared buffer and
    scan the accumulated text for the prompt marker '\\n> ' (turn complete) instead.
    """
    PROMPT_MARK = "\n> "

    def __init__(self, exe, model, ngl, canvas_ctx, steps, extra, verbose):
        self.verbose = verbose
        # -n npredict: with canvas_length=256, -n 768 = up to 3 block-autoregressive passes so a
        # thinking model has room to finish its thought AND emit the final-answer tag.
        cmd = [exe, "-m", model, "-cnv", "-ngl", str(ngl),
               "-c", str(canvas_ctx), "-ub", str(canvas_ctx), "-b", str(canvas_ctx),
               "--temp", "0.0", "-n", "768",
               "--diffusion-steps", str(steps), "-fa", "off"]
        cmd += extra
        self.cmd = cmd
        self.log = open("_diffcli_stderr.log", "w", encoding="utf-8", errors="replace")
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, bufsize=0)
        self.lock = threading.Lock()
        self.buf = ""
        self.t = threading.Thread(target=self._reader, daemon=True)
        self.t.start()

    def _reader(self):
        while True:
            chunk = self.p.stdout.read(256)
            if not chunk:
                break
            text = chunk.decode("utf-8", "replace")
            if self.verbose:
                sys.stderr.write(text)
            self.log.write(text); self.log.flush()
            with self.lock:
                self.buf += text

    def _wait_prompt(self, timeout):
        """Block until PROMPT_MARK appears in the buffer; return (captured_text, ok).
        captured_text = everything in the buffer up to (and not including) the prompt mark."""
        end = time.time() + timeout
        while time.time() < end:
            with self.lock:
                idx = self.buf.find(self.PROMPT_MARK)
                if idx >= 0:
                    captured = self.buf[:idx]
                    self.buf = self.buf[idx + len(self.PROMPT_MARK):]
                    return captured, True
            if self.p.poll() is not None:
                raise RuntimeError("diffusion-cli exited")
            time.sleep(0.2)
        return "", False

    def boot(self, timeout=1200):
        _, ok = self._wait_prompt(timeout)
        return ok

    def ask(self, prompt, timeout=300):
        line = (prompt.replace("\n", " ").strip() + "\n").encode("utf-8")
        self.p.stdin.write(line); self.p.stdin.flush()
        captured, ok = self._wait_prompt(timeout)
        reply = self._extract_reply(captured)
        try:
            with open("_diffcli_captures.log", "a", encoding="utf-8", errors="replace") as cf:
                cf.write("\n===== CAPTURED (ok=%s) =====\n%s\n----- REPLY -----\n%s\n----- FINAL -----\n%s\n"
                         % (ok, captured[-1500:], reply[-600:], final_answer(reply)[-200:]))
        except Exception:
            pass
        # reset history so the next query is independent
        self.p.stdin.write(b"/clear\n"); self.p.stdin.flush()
        self._wait_prompt(30)
        return reply

    def _extract_reply(self, captured):
        # captured holds: the echoed prompt line, diffusion-step progress bars (CR-overwritten),
        # the reply (LOG "\n%s\n"), then timing lines (total time / throughput). Strip all of it
        # except the reply. The reply sits between the last diffusion-step line and 'total time'.
        text = captured.replace("\r\n", "\n").replace("\r", "\n")
        # cut everything at/after the trailing timing footer (printed AFTER the reply).
        # NB: do NOT cut at diffusion_eb:/diffusion_params: -- those print BEFORE the reply.
        for mark in ("total time:", "throughput:"):
            i = text.find(mark)
            if i >= 0:
                text = text[:i]
        out = []
        for raw in text.split("\n"):
            # diffusion progress bars use \r; keep only the final segment of a CR line
            s = raw.split("\r")[-1].strip()
            if not s:
                continue
            low = s.lower()
            if (low.startswith("diffusion step") or low.startswith("diffusion:") or
                low.startswith("diffusion_") or low.startswith("total time") or
                low.startswith("throughput") or low.startswith("llama_") or
                low.startswith("ggml_") or low.startswith("build:") or
                low.startswith("main:") or "conversation mode" in low):
                continue
            out.append(s)
        return " ".join(out).strip()

    def close(self):
        try:
            self.p.stdin.write("/exit\n"); self.p.stdin.flush()
        except Exception:
            pass
        try:
            self.p.wait(timeout=10)
        except Exception:
            self.p.kill()


def final_answer(reply):
    """DiffusionGemma is a thinking model: '<|channel>thought ... <channel|><final answer>'.
    The thought ECHOES the candidate list (all tags), so we must score only the post-thought
    answer. Take the text after the last '<channel|>' (or after 'thought'/'</thought>')."""
    r = reply
    for mark in ("<channel|>", "</thought>", "<|channel|>"):
        i = r.rfind(mark)
        if i >= 0:
            return r[i + len(mark):].strip()
    return r.strip()


def parse_tag(reply, tags):
    seg = final_answer(reply)
    up = seg.upper()
    # the chosen tag is the FIRST tag appearing in the post-thought answer segment
    best_i, best_pos = None, None
    for i, tg in enumerate(tags):
        p = up.find(tg)
        if p >= 0 and (best_pos is None or p < best_pos):
            best_i, best_pos = i, p
    if best_i is not None:
        return best_i
    # explicit NONE/NULL in the answer -> reject
    return None


def main():
    ap = argparse.ArgumentParser()
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--corpus", default=os.path.join(base, "_needle_corpus_div"))
    ap.add_argument("--exe", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default=os.path.join(base, "tests", "fixtures", "chat_fullstack", "G-DIFFJUDGE-1.log"))
    ap.add_argument("--seed", type=int, default=20260621)
    ap.add_argument("--window", type=int, default=12)
    ap.add_argument("--ngl", type=int, default=20)
    ap.add_argument("--ctx", type=int, default=1536, help="n_ctx / ubatch (>= judge prompt tokens + n_predict)")
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--foreign-limit", type=int, default=0)
    ap.add_argument("--paraphrases", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--extra", nargs="*", default=[])
    args = ap.parse_args()

    man, foreign = load_corpus(args.corpus)
    needles = [m for m in man if not m["id"].startswith("ctrl")]
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

    emit(f"# G-DIFFJUDGE-1 corpus={args.corpus} seed={args.seed}")
    emit(f"# exe={args.exe}")
    emit(f"# model={args.model}")
    emit(f"# window/K={K} ngl={args.ngl} ctx={args.ctx} steps={args.steps} "
         f"pool={len(alln)} matched={len(needles)} foreign={len(foreign)} paraphrases={args.paraphrases}")

    cli = DiffCli(args.exe, args.model, args.ngl, args.ctx, args.steps, args.extra, args.verbose)
    emit("# booting diffusion-cli (loading model)...")
    if not cli.boot(timeout=1200):
        emit("FATAL: diffusion-cli did not reach the input prompt within timeout")
        log.close(); cli.close(); sys.exit(2)
    emit(f"# model loaded in {time.time()-t0:.0f}s; running gate")

    def window_for(truth):
        pool = [m for m in alln if m is not truth]
        distract = rng.sample(pool, min(K-1, len(pool)))
        cands = distract + [truth]
        rng.shuffle(cands)
        return cands

    hit = tot = frej = ftot = 0
    lat = []

    for m in needles:
        qs = [m["query"]] + ([p for p in m.get("paraphrases", []) if p != m["query"]]
                             if args.paraphrases else [])
        for q in qs:
            cands = window_for(m)
            gt = cands.index(m)
            tags = rng.sample(_TAGPOOL, len(cands))
            entries = " ".join(f"[{tg}] {c['text']}" for tg, c in zip(tags, cands))
            prompt = JUDGE_TMPL.format(entries=entries, q=q)
            tq = time.time()
            reply = cli.ask(prompt)
            lat.append(time.time()-tq)
            ch = parse_tag(reply, tags)
            ok = (ch == gt)
            tot += 1; hit += int(ok)
            emit(f"[match] {'OK ' if ok else 'MISS'} {m['id']:16} K={len(cands)} gt={gt:2} got={ch} "
                 f":: {q[:48]!r} -> {reply.strip()[:40]!r}")

    for q in foreign:
        cands = rng.sample(alln, min(K, len(alln)))
        tags = rng.sample(_TAGPOOL, len(cands))
        entries = " ".join(f"[{tg}] {c['text']}" for tg, c in zip(tags, cands))
        prompt = JUDGE_TMPL.format(entries=entries, q=q)
        tq = time.time()
        reply = cli.ask(prompt)
        lat.append(time.time()-tq)
        ch = parse_tag(reply, tags)
        ok = (ch is None)
        ftot += 1; frej += int(ok)
        emit(f"[forgn] {'REJECT' if ok else 'FALSEFIRE'} K={len(cands)} got={ch} "
             f":: {q[:48]!r} -> {reply.strip()[:40]!r}")

    cli.close()
    rec = hit/max(tot,1); rej = frej/max(ftot,1)
    g = (rec >= 0.857)
    avg = sum(lat)/max(len(lat),1)
    emit(f"\n================ RESULT  (elapsed {time.time()-t0:.0f}s) ================")
    emit(f"K={K} steps={args.steps} ngl={args.ngl}  avg latency/query {avg:.1f}s")
    emit(f"recall@1        : {hit}/{tot} = {100*rec:.1f}%")
    emit(f"foreign-reject  : {frej}/{ftot} = {100*rej:.1f}%")
    emit(f"AR judge baseline recall@1 = 85.7%")
    emit(f"GATE (recall@1 >= 85.7%) : {'GREEN' if g else 'RED'}")
    log.close()
    sys.exit(0 if g else 1)


if __name__ == "__main__":
    main()
