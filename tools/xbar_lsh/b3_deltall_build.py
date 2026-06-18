#!/usr/bin/env python3
"""B3-v4 query-deflection harness builder. For each query x each episode-as-text-prefix
(+ baseline none), build a token file [E_text_ids + query_ids[1:]] and a manifest line
"<score_from> <tokfile>" where score_from = len(E_text_ids) (the first query position).
The C driver (test_gemma4_ppl_cuda SP_DELTALL_MANIFEST) scores LL(query|prefix) for each
in ONE model load. dLL = LL(query|E) - LL(query|0) computed in the parser.

Tokenization via sp_tok_enc (the .sp-tokenizer blob = byte-identical to /v1/chat).
"""
import os, subprocess, json, tempfile

ENG = r"D:\F\shannon-prime-repos\shannon-prime-system-engine"
TOK = r"D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-tokenizer"
ENC = os.path.join(ENG, "build-cuda-vs22", "tools", "sp_tok_dump", "sp_tok_enc.exe")
OUT = os.path.join(ENG, "_b3_wc", "deltall"); os.makedirs(OUT, exist_ok=True)

EPISODES = {  # name -> source text file (the passage the ep.k was captured from)
    "none":       None,
    "ep_wiki":    os.path.join(ENG, "_b3_boulter.txt"),
    "ep_homarus": os.path.join(ENG, "_b3_homarus.txt"),
    "ep_headlam": os.path.join(ENG, "_b3_headlam.txt"),
}
# the held-out + adversarial query set (label = the episode it SHOULD recall, or foreign)
QUERIES = [
    ("Who is Robert Boulter?",                       "ep_wiki"),
    ("What is the European lobster, Homarus gammarus?", "ep_homarus"),
    ("Who was Frank Headlam?",                        "ep_headlam"),
    # adversarial foreigns (should reject ALL):
    ("What is the standard hydration ratio for a French sourdough boulter bread?", "FOREIGN_lexical"),
    ("Explain the memory-bandwidth limits of the dp4a GEMV accumulate instruction.", "FOREIGN_technical"),
    ("Hey, can you help me remember what we were just talking about?", "FOREIGN_drift"),
    ("the and of to a in is that it with as",         "FOREIGN_stopword"),
    # extra plain foreigns:
    ("What is the capital of France?",                "FOREIGN_plain"),
    ("How do I bake sourdough bread?",                "FOREIGN_plain"),
]


def enc(text_or_path, is_path):
    if is_path:
        p = text_or_path
    else:
        fd, p = tempfile.mkstemp(suffix=".txt", dir=OUT); os.close(fd)
        open(p, "w", encoding="utf-8").write(text_or_path)
    r = subprocess.run([ENC, TOK, p], capture_output=True, text=True)
    ids = [int(x) for x in r.stdout.split()]
    if not is_path:
        os.remove(p)
    return ids


def main():
    assert os.path.exists(ENC), f"build sp_tok_enc first: {ENC}"
    ep_ids = {name: (enc(path, True) if path else None) for name, path in EPISODES.items()}
    for name, ids in ep_ids.items():
        print(f"[dl] episode {name}: {len(ids) if ids else 0} tokens")
    manifest, meta = [], []
    for qi, (q, lab) in enumerate(QUERIES):
        qids = enc(q, False)                    # [BOS, query tokens...]
        for ename, eids in ep_ids.items():
            if eids is None:                    # baseline: [BOS, query], score from 1
                seq = qids; sfrom = 1
            else:                               # [BOS, E text, query text], score from len(E)
                seq = eids + qids[1:]; sfrom = len(eids)
            tf = os.path.join(OUT, f"q{qi}_{ename}.txt")
            open(tf, "w").write("\n".join(str(x) for x in seq) + "\n")
            manifest.append(f"{sfrom} {tf}")
            meta.append({"qi": qi, "query": q, "label": lab, "ep": ename,
                         "tokfile": os.path.basename(tf), "n_query": len(qids) - 1})
    mpath = os.path.join(OUT, "manifest.txt"); open(mpath, "w").write("\n".join(manifest) + "\n")
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=0)
    print(f"[dl] wrote {len(manifest)} sequences + {mpath}")


if __name__ == "__main__":
    main()
