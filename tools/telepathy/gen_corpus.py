#!/usr/bin/env python3
# gen_corpus.py — broadened training corpus for the gen-tuned adapter (TELE-10b). Heavy on the
# coding/CS domain that failed OOD in the v1 demo, plus diverse general text, so the adapter generalizes
# instead of overfitting the narrow pairs.txt mix. -> gen_corpus.txt
import random, sys
random.seed(11)
CODE_TMPL = [
 "write a python function that {v} a {ds}","how do you {v} a {ds} in python","what is the time complexity of {algo}",
 "explain how {algo} works step by step","implement {algo} and analyze its complexity","what is the difference between {ds} and {ds2}",
 "count the number of {x} in a string","reverse the order of words in a sentence","check whether a number is prime",
 "find the {sup} element in an array","explain how a {struct} handles {issue}","debug this off-by-one error in a loop",
 "convert a recursive function to an iterative one","what does the {kw} keyword do in python",
 "parse a json string and extract a field","sort a list of dictionaries by a key","memoize a recursive fibonacci function",
 "explain big-O notation with an example","write a regex that matches an email address","traverse a binary tree in order"]
V=["reverse","sort","balance","flatten","serialize","traverse","search","insert into","delete from","merge"]
DS=["linked list","binary tree","hash map","stack","queue","heap","graph","trie","array","set"]
DS2=["array","tuple","dictionary","set","list","deque"]
ALGO=["binary search","quicksort","merge sort","dijkstra's algorithm","breadth first search","depth first search",
      "dynamic programming","the two pointer technique","hashing","topological sort"]
SUP=["largest","smallest","second largest","median","most frequent","first non-repeating"]
STRUCT=["hash map","bloom filter","b-tree","cache","load balancer"]; ISSUE=["collisions","resizing","eviction","concurrency","overflow"]
X=["vowels","consonants","r letters","spaces","digits","unique characters"]; KW=["yield","lambda","with","global","nonlocal","async"]
FACTS=["water boils at one hundred degrees celsius","paris is the capital of france","the heart pumps blood through the body",
       "light travels faster than sound","photosynthesis converts sunlight into energy","dna stores genetic information"]
INSTR=["summarize this paragraph in one sentence","translate this phrase into spanish","write a short polite reply",
       "outline the steps to bake bread","draft a one line commit message","explain this concept to a beginner"]
CASUAL=["the coffee this morning was great","traffic was bad on the way in","my favorite season is autumn",
        "we should grab lunch next week","the meeting ran long again","i had a relaxing weekend"]
def code():
    t=random.choice(CODE_TMPL)
    return t.format(v=random.choice(V),ds=random.choice(DS),ds2=random.choice(DS2),algo=random.choice(ALGO),
                    sup=random.choice(SUP),struct=random.choice(STRUCT),issue=random.choice(ISSUE),
                    x=random.choice(X),kw=random.choice(KW))
n=int(sys.argv[1]) if len(sys.argv)>1 else 320
out=set()
while len(out)<n:
    r=random.random()
    out.add(code() if r<0.60 else random.choice(FACTS) if r<0.72 else random.choice(INSTR) if r<0.86 else random.choice(CASUAL))
open("gen_corpus.txt","w",encoding="utf-8").write("\n".join(out)+"\n")
print(f"wrote gen_corpus.txt ({len(out)}; ~60% coding/CS)")
