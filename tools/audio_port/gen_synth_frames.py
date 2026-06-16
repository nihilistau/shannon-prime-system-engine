#!/usr/bin/env python3
# KAI-3 §7.3 — synthetic frame-tape generator.
#
# Purpose: stand in for the not-yet-built GNA/CNN front-end so frame_projector.py has DENSE
# per-position supervision. Maps synthetic 640-float / 40ms frame sequences onto the 512-event
# template-grammar corpus token-id sequences, so the projector's objective is per-position CE
# (frame_i -> token_i), NOT the sparse end-of-sequence decision-KL that plateaued KAI-2 t10 at 0.9157.
#
# Mechanism (CONTRACT-KAIROS §7.3):
#   - Frozen anchor matrix A[|V_sub|, 640] simulates the continuous upstream feature per subset token.
#   - frame_j = A[local(t_j)] + sigma*||A||*N(0,I_640).  The noise makes frame->token a robust
#     continuous boundary, NOT a brittle 1:1 lookup (mimics real-audio variance).
#   - V_sub = the event-vocab subset (union of corpus + EVAL token ids); the projector binds onto
#     these embedding rows exactly as the t10 on-manifold head did.
#
# Scope honesty (pre-registered): a GREEN downstream result proves the ARCHITECTURE + binder + CE
# loop. It does NOT prove real audio — actual GNA/CNN features (task #154) replace A later.
import argparse, json, os, sys
import numpy as np

# ── template grammar: salient (salience>=0.5) -> ACTION ; idle (<0.5) -> NO_OP ───────────────────
TYPES = ["build","deploy","disk","memory","network","auth","cert","queue","db","cache",
         "heartbeat","backup","scan","alert","job","sync","index","gc","stream","quota"]
STATUS = ["OK","FAILED","DEGRADED","TIMEOUT","PENDING"]

def gen_corpus(n, seed):
    rng = np.random.default_rng(seed)
    evs = []
    for _ in range(n):
        t   = TYPES[int(rng.integers(len(TYPES)))]
        sal = float(rng.random())
        idv = int(rng.integers(1000, 9999))
        st  = STATUS[int(rng.integers(len(STATUS)))]
        met = int(rng.integers(0, 100))
        txt = f"EVENT {t} id={idv} status={st} metric={met}% salience={sal:.2f}"
        evs.append((txt, sal))
    return evs

# Held-out generalization set (never trained on). First salient + first idle mirror the metal
# harness cases so G-KAIROS-3 can reuse the run_kai2 scaffold.
EVAL_EVENTS = [
    ("EVENT build id=4471 status=FAILED tests=3_broken salience=0.85", 0.85),
    ("EVENT heartbeat id=3300 status=OK cpu=12% salience=0.10",        0.10),
    ("EVENT disk id=1190 status=DEGRADED usage=94% salience=0.78",     0.78),
    ("EVENT cache id=8123 status=OK hit=99% salience=0.05",            0.05),
    ("EVENT auth id=5521 status=FAILED attempts=9 salience=0.91",      0.91),
    ("EVENT queue id=6610 status=OK depth=4 salience=0.15",            0.15),
    ("EVENT cert id=2048 status=PENDING expiry=3d salience=0.66",      0.66),
    ("EVENT sync id=7001 status=OK lag=2ms salience=0.08",             0.08),
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="dir with the gemma-4 tokenizer (AutoTokenizer)")
    ap.add_argument("--out", required=True, help="output .npz")
    ap.add_argument("--n_events", type=int, default=512)
    ap.add_argument("--frame_dim", type=int, default=640)   # 40ms @ 16kHz
    ap.add_argument("--noise_rel", type=float, default=0.1, help="sigma as a fraction of mean ||A_row||")
    ap.add_argument("--seed", type=int, default=20260616)
    ap.add_argument("--vocab_size", type=int, default=262144, help="gemma-4 vocab (token ids index real embed rows)")
    ap.add_argument("--vsub_size", type=int, default=160, help="|V_sub| — token-id subset the projector binds onto")
    ap.add_argument("--min_len", type=int, default=12)
    ap.add_argument("--max_len", type=int, default=28)
    ap.add_argument("--use_tokenizer", action="store_true",
                    help="encode real EVENT text (needs transformers/tokenizers) — for the METAL pivot gate, NOT the architecture ladder")
    args = ap.parse_args()

    # ARCHITECTURE-LADDER mode (default): synthetic random valid-token-id sequences over a fixed V_sub.
    # No tokenizer dependency. The token ids index REAL embed rows (so W_sub + cos tripwire stay honest);
    # the frame->token mapping is still a learned continuous->discrete problem. Real English text + the
    # ACTION/NO_OP pivot belong to the METAL G-KAIROS-3 gate (use --use_tokenizer there), not here.
    rng0 = np.random.default_rng(args.seed)
    if args.use_tokenizer:
        try:
            from tokenizers import Tokenizer
            _tk = Tokenizer.from_file(os.path.join(args.model, "tokenizer.json"))
            def encode(txt): return _tk.encode(txt, add_special_tokens=False).ids
        except Exception:
            from transformers import AutoTokenizer
            _tok = AutoTokenizer.from_pretrained(args.model)
            def encode(txt): return _tok(txt, add_special_tokens=False)["input_ids"]
        train_ev = [(t, s) for (t, s) in gen_corpus(args.n_events, args.seed)
                    if t not in {x for x, _ in EVAL_EVENTS}]
        train_ids = [(encode(t), s) for t, s in train_ev]
        eval_ids  = [(encode(t), s) for t, s in EVAL_EVENTS]
        vsub = sorted({i for ids, _ in train_ids + eval_ids for i in ids})
        g2l = {g: l for l, g in enumerate(vsub)}
        train_ids = [([g2l[i] for i in ids], s) for ids, s in train_ids]
        eval_ids  = [([g2l[i] for i in ids], s) for ids, s in eval_ids]
    else:
        vsub = sorted(rng0.choice(args.vocab_size, size=args.vsub_size, replace=False).tolist())
        def synth_seq():
            L = int(rng0.integers(args.min_len, args.max_len + 1))
            return rng0.integers(0, len(vsub), size=L).tolist()   # LOCAL V_sub indices
        train_ids = [(synth_seq(), float(rng0.random())) for _ in range(args.n_events)]
        eval_ids  = [(synth_seq(), float(rng0.random())) for _ in range(8)]
    V = len(vsub)

    rng = np.random.default_rng(args.seed ^ 0xA5A5)
    A = rng.standard_normal((V, args.frame_dim)).astype(np.float32)
    normA = float(np.mean(np.linalg.norm(A, axis=1)))
    sigma = args.noise_rel * normA

    def synth(ids_list):
        seqs_x, seqs_t, lens = [], [], []
        for ids, _sal in ids_list:
            loc = np.array(ids, dtype=np.int64)                  # ids are already LOCAL V_sub indices
            noise = rng.standard_normal((len(ids), args.frame_dim)).astype(np.float32)
            x = A[loc] + sigma * noise
            seqs_x.append(x); seqs_t.append(loc); lens.append(len(ids))
        return seqs_x, seqs_t, lens

    trX, trT, trL = synth(train_ids)
    evX, evT, evL = synth(eval_ids)

    maxlen = max(max(trL), max(evL))
    def pad(seqs_x, seqs_t, lens):
        n = len(seqs_x)
        X = np.zeros((n, maxlen, args.frame_dim), dtype=np.float32)
        T = np.full((n, maxlen), -100, dtype=np.int64)          # -100 = CE ignore_index
        for i, (x, t, L) in enumerate(zip(seqs_x, seqs_t, lens)):
            X[i, :L] = x; T[i, :L] = t
        return X, np.array(lens, dtype=np.int64), T
    trXp, trLa, trTp = pad(trX, trT, trL)
    evXp, evLa, evTp = pad(evX, evT, evL)

    np.savez(args.out,
             train_X=trXp, train_T=trTp, train_len=trLa,
             eval_X=evXp,  eval_T=evTp,  eval_len=evLa,
             vsub_ids=np.array(vsub, dtype=np.int64),
             eval_texts=np.array([t for t, _ in EVAL_EVENTS] if args.use_tokenizer
                                 else [f"synth_{i}" for i in range(len(eval_ids))]),
             eval_expect=np.array(["ACTION" if s >= 0.5 else "NO_OP" for _, s in eval_ids]),
             A=A, sigma=np.float32(sigma), normA=np.float32(normA),
             frame_dim=np.int64(args.frame_dim))
    print(f"[gen] V_sub={V} train_events={len(train_ids)} eval_events={len(eval_ids)} "
          f"maxlen={maxlen} frame_dim={args.frame_dim} ||A||={normA:.3f} sigma={sigma:.4f} "
          f"(noise_rel={args.noise_rel}) -> {args.out}", flush=True)

if __name__ == "__main__":
    main()
