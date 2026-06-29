#!/usr/bin/env python3
# gen_pairs.py — paired-latent calibration corpus for the Telepathy gemma<->qwen adapter.
# Emits N diverse IN-DOMAIN texts (same string fed to BOTH models => a latent pair per text) and a
# disjoint FOREIGN set (gibberish / non-English / structureless) for the REJECT gate.
# "Paired" = identical input to both models; the adapter learns the map between the two encodings.
import random, sys
random.seed(7)

SUBJ = ["the function","this list","the server","a prime number","the config file","the database",
        "our deadline","the api key","the backup","the report","this string","the cache","the model",
        "the request","the token","a vector","the gradient","the kernel","the matrix","the payload"]
VERB = ["counts","reverses","stores","validates","fetches","sorts","deduplicates","computes","parses",
        "encrypts","compresses","schedules","deploys","queries","caches","rejects","aligns","projects"]
TAIL = ["the characters in a word","the rows in a table","every record by timestamp","the checksum of a file",
        "the latency under load","the duplicate entries","the longest token","a percentage of the total",
        "the square root of the input","the response from the endpoint","the keys before the values",
        "the embedding into a lower dimension","the residual stream","the attention weights"]
FACTS = ["water boils at one hundred degrees celsius at sea level","the mitochondria is the powerhouse of the cell",
         "paris is the capital of france","light travels faster than sound","the heart pumps blood through the body",
         "photosynthesis converts sunlight into energy","the moon orbits the earth roughly every month",
         "prime numbers have exactly two divisors","entropy tends to increase in a closed system",
         "the speed of light is constant in a vacuum","dna encodes genetic information","gravity attracts mass to mass"]
INSTR = ["please summarize the document in three sentences","write a haiku about the ocean",
         "explain recursion to a beginner","translate this phrase into spanish","draft a polite decline email",
         "outline the steps to bake bread","describe how a transformer attends to tokens",
         "give me three names for a cat","compare two sorting algorithms briefly","list the planets in order"]
CASUAL = ["i had a great weekend hiking in the mountains","the coffee this morning was surprisingly good",
          "traffic was terrible on the way to work","my favorite season is early autumn",
          "we should grab lunch sometime next week","the meeting ran way longer than expected",
          "i can never remember where i left my keys","that movie was better than i expected"]

def in_domain(n):
    out = set()
    while len(out) < n:
        k = random.random()
        if   k < 0.34: out.add(f"{random.choice(SUBJ)} {random.choice(VERB)} {random.choice(TAIL)}")
        elif k < 0.55: out.add(random.choice(FACTS))
        elif k < 0.78: out.add(random.choice(INSTR))
        else:          out.add(random.choice(CASUAL))
    return list(out)

def foreign(n):
    out = set()
    alpha = "abcdefghijklmnopqrstuvwxyz   "
    nonen = ["これはテストです","das ist ein test auf deutsch","ceci nest pas une phrase anglaise",
             "это случайный русский текст","يh ةضوافم صنلا","βαβ γαμμα δελτα τυχαία",
             "测试随机中文文本数据","무작위 한국어 텍스트입니다"]
    while len(out) < n:
        k = random.random()
        if k < 0.5:  out.add("".join(random.choice(alpha) for _ in range(random.randint(20,45))).strip())
        elif k<0.75: out.add(" ".join(str(random.randint(0,9999)) for _ in range(random.randint(5,12))))
        else:        out.add(random.choice(nonen))
    return list(out)

if __name__ == "__main__":
    ndom = int(sys.argv[1]) if len(sys.argv) > 1 else 320
    nfor = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    dom = in_domain(ndom); frn = foreign(nfor)
    open("pairs.txt","w",encoding="utf-8").write("\n".join(dom)+"\n")
    open("foreign.txt","w",encoding="utf-8").write("\n".join(frn)+"\n")
    print(f"wrote pairs.txt ({len(dom)}) + foreign.txt ({len(frn)})")
