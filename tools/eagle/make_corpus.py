#!/usr/bin/env python3
# make_corpus.py — build a diverse prompt corpus for the EAGLE flywheel capture.
# Each line = one user message; the capture frames it (apply_template_ids) and greedy-rolls
# the served 12B, so we want broad coverage of the GENERATION distribution the draft must learn.
import random, sys
random.seed(20260629)

subjects = ["France","Japan","Brazil","Kenya","Norway","Egypt","Canada","India","Peru","Vietnam",
 "the Roman Empire","photosynthesis","black holes","the French Revolution","DNA","inflation",
 "machine learning","the immune system","tectonic plates","the stock market","quantum entanglement",
 "the water cycle","supply and demand","neural networks","the printing press","antibiotics",
 "climate change","the internet","electric cars","vaccines","the human brain","gravity",
 "blockchain","renewable energy","the Cold War","evolution","the speed of light","compound interest"]
langs = ["Python","Rust","JavaScript","SQL","C","Go","Bash"]
tasks = ["reverse a string","check if a number is prime","read a file line by line","sort a list",
 "make an HTTP GET request","compute a factorial","find the max in an array","parse JSON",
 "remove duplicates from a list","implement binary search"]
people = ["a five-year-old","a busy executive","a software engineer","a high-school student","a skeptic"]
tones = ["concisely","in detail","step by step","with an analogy","in plain language"]

T = []
for s in subjects:
    T += [f"What is the capital of {s}?" if s in subjects[:10] else f"Explain {s} {random.choice(tones)}.",
          f"What are the key facts about {s}?",
          f"Summarize {s} in a few sentences.",
          f"Why does {s} matter?"]
for s in subjects:
    T.append(f"Explain {s} to {random.choice(people)} {random.choice(tones)}.")
for l in langs:
    for t in tasks:
        T.append(f"Write a {l} function to {t}.")
for s in subjects[:20]:
    T.append(f"Compare {s} and {random.choice(subjects)} {random.choice(tones)}.")
T += [
 "Write a short poem about the ocean.","Give me three tips for better sleep.",
 "What's a good recipe for pancakes?","Explain the difference between TCP and UDP.",
 "How do I stay motivated when learning something hard?","What causes the seasons?",
 "Write a haiku about autumn.","List five common logical fallacies.",
 "How does a transistor work?","What is the meaning of the word 'serendipity'?",
 "Draft a polite email asking for a deadline extension.","Explain recursion with an example.",
 "What are the pros and cons of remote work?","How do you make a good first impression?",
 "Describe the plot of a hero's journey.","What is the difference between weather and climate?",
 "Give step-by-step instructions to change a flat tire.","Explain how vaccines train the immune system.",
 "What's the best way to learn a new language?","Write a function to compute Fibonacci numbers.",
 "Explain what an API is.","How does compound interest grow money over time?",
 "What are the main causes of inflation?","Describe how a search engine ranks pages.",
 "Explain the concept of opportunity cost.","What makes a good password?",
 "How do airplanes stay in the air?","Write a limerick about a cat.",
 "What is the difference between AI and machine learning?","Explain the greenhouse effect.",
]
random.shuffle(T)
# de-dup, cap length
seen=set(); out=[]
for t in T:
    t=" ".join(t.split())
    if t and t not in seen and len(t) < 200:
        seen.add(t); out.append(t)
n = int(sys.argv[2]) if len(sys.argv) > 2 else len(out)
out = out[:n]
path = sys.argv[1] if len(sys.argv) > 1 else "corpus.txt"
open(path,"w",encoding="utf-8").write("\n".join(out)+"\n")
print(f"wrote {len(out)} prompts -> {path}")
