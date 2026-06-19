#!/usr/bin/env python3
"""B3-v6 SEMANTIC DISCRIMINATOR (reasoning bridge). For each (query, episode), build the
prompt:  Memory: <E text>\nQuery: <Q>\nDoes the memory contain the answer to the query?
Answer strictly Yes or No: <ANS>  and score ONLY the final answer token (Yes / No).
Margin = logP(Yes) - logP(No) = NLL(No) - NLL(Yes). High margin => the 12B JUDGES the
memory answers the query (reasoning subspace), immune to the fluent-prefix confound that
broke the syntactic-dLL verifier. Reuses the SP_DELTALL one-shot harness unchanged.
"""
import os, subprocess, json, tempfile
ENG = r"D:\F\shannon-prime-repos\shannon-prime-system-engine"
TOK = r"D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-tokenizer"
ENC = os.path.join(ENG, "build-cuda-vs22", "tools", "sp_tok_dump", "sp_tok_enc.exe")
OUT = os.path.join(ENG, "_b3_wc", "reason"); os.makedirs(OUT, exist_ok=True)
EPTXT = {"ep_wiki": os.path.join(ENG,"_b3_boulter.txt"),
         "ep_homarus": os.path.join(ENG,"_b3_homarus.txt"),
         "ep_headlam": os.path.join(ENG,"_b3_headlam.txt")}
QUERIES = [  # (query, label) — label = the episode it should match, or FOREIGN_*
 ("Who is Robert Boulter?","ep_wiki"),
 ("What is the European lobster, Homarus gammarus?","ep_homarus"),
 ("Who was Frank Headlam?","ep_headlam"),
 ("What is the standard hydration ratio for a French sourdough boulter bread?","FOREIGN_lexical"),
 ("Explain the memory-bandwidth limits of the dp4a GEMV accumulate instruction.","FOREIGN_technical"),
 ("Hey, can you help me remember what we were just talking about?","FOREIGN_drift"),
 ("the and of to a in is that it with as","FOREIGN_stopword"),
 ("What is the capital of France?","FOREIGN_plain"),
 ("How do I bake sourdough bread?","FOREIGN_plain"),
]
PROMPT = "Memory: {E}\nQuery: {Q}\nDoes the memory contain the answer to the query? Answer strictly Yes or No:"

def enc(text):
    fd,p = tempfile.mkstemp(suffix=".txt", dir=OUT); os.close(fd)
    open(p,"w",encoding="utf-8").write(text)
    r = subprocess.run([ENC,TOK,p], capture_output=True, text=True)
    os.remove(p)
    return [int(x) for x in r.stdout.split()]

def main():
    eptext = {k: open(v,encoding="utf-8").read().strip() for k,v in EPTXT.items()}
    manifest, meta = [], []
    for qi,(q,lab) in enumerate(QUERIES):
        for ep,etxt in eptext.items():
            prefix = PROMPT.format(E=etxt, Q=q)
            pids = enc(prefix)                       # [BOS, prefix...]
            for ans in ("Yes","No"):
                aids = enc(" " + ans)                # [BOS, ans-token(s)]
                seq = pids + aids[1:]                # drop the answer's BOS
                sfrom = len(pids)                    # score only the answer token(s)
                tf = os.path.join(OUT, f"q{qi}_{ep}_{ans}.txt")
                open(tf,"w").write("\n".join(str(x) for x in seq)+"\n")
                manifest.append(f"{sfrom} {tf}")
                meta.append({"qi":qi,"query":q,"label":lab,"ep":ep,"ans":ans,
                             "tokfile":os.path.basename(tf),"n_ans":len(aids)-1})
    open(os.path.join(OUT,"manifest.txt"),"w").write("\n".join(manifest)+"\n")
    json.dump(meta, open(os.path.join(OUT,"meta.json"),"w"), indent=0)
    print(f"[rb] wrote {len(manifest)} sequences ({len(QUERIES)} queries x 3 eps x Yes/No)")

if __name__=="__main__":
    main()
