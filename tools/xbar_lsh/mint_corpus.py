#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mint_corpus.py -- split-entropy novel-needle minter for the XBAR labeler->head loop.

POSITIVES (needles): code=CSPRNG, contradiction+relational=structured-fictional.
PARAPHRASES: --paraphrases k free-form questions per needle (semantic-subspace variety).
NULL CLASS: --foreign N out-of-domain queries (zero episode correlation) -> learns s0/TAU.

Split entropy guards the OOD-grammatical-shock confound: CSPRNG only in the code slot
(model expects random alphanumerics; shape stays in-distribution digit-WORD-4digit);
structured-fictional (phonetically-valid invented entities) for contradiction+relational.
Secret stated ONCE and is the sentence TAIL. Non-parametricity = the ablation oracle's job.

Outputs: <out>/<id>.txt, corpus_manifest.jsonl (id,archetype,text,query,paraphrases,secret,
topic,sig_bits), registry.jsonl (npos:-1), foreign_queries.txt.
"""
import argparse, hashlib, json, os, secrets, sys
from collections import Counter

_CODE_WORDS = ["FALCON","OSPREY","VECTOR","CITADEL","HALCYON","OBSIDIAN","MERIDIAN","QUASAR",
    "ZEPHYR","PHOENIX","TUNDRA","COBALT","SENTINEL","ARGON","KESTREL","LANTERN","MARINER",
    "NEBULA","PALADIN","RAMPART"]
_VAULTS = ["KAI-3 subterranean vault","Meridian cold-storage archive","Halcyon key escrow",
    "Citadel deep-vault","Obsidian records bunker","Sentinel manifest store",
    "Argon containment locker","Rampart secure annex"]

def mint_code():
    d1 = secrets.randbelow(9)+1; word=_CODE_WORDS[secrets.randbelow(len(_CODE_WORDS))]
    d4 = secrets.randbelow(9000)+1000; code=f"{d1}-{word}-{d4}"
    vault=_VAULTS[secrets.randbelow(len(_VAULTS))]
    text=f"The primary access code for the {vault} is {code}."
    query=f"The primary access code for the {vault} is"; secret=f" {code}"
    para=[f"What is the access code for the {vault}?",
          f"State the primary access code for the {vault}.",
          f"Recall the entry code for the {vault}.",
          f"Which code unlocks the {vault}?"]
    return text,query,secret,"high-entropy access code (CSPRNG)",para

_SYL_A=["Ori","Vael","Tyr","Quor","Zen","Mor","Cael","Pyr","Lru","Xan","Threx","Vol","Ny",
    "Drav","Esh","Korr","Ulth","Wren","Syl","Garn"]
_SYL_B=["con","dar","thys","vane","mire","lux","gard","phane","drel","quith","nor","vast",
    "mel","thorn","ric","zar","veil","dris"]
_SUFFIX=["-Prime","-Major","-IX","-Reach","-Halo","-Crest","","",""]
def _fict_name():
    return f"{_SYL_A[secrets.randbelow(len(_SYL_A))]}{_SYL_B[secrets.randbelow(len(_SYL_B))]}{_SUFFIX[secrets.randbelow(len(_SUFFIX))]}"

_KNOWN_FACTS=[("the capital of France","was officially relocated to the synthetic city of"),
    ("the headquarters of the United Nations","was permanently moved to the enclave of"),
    ("the prime meridian","was formally re-anchored through the township of"),
    ("the tallest mountain on Earth","was administratively reassigned to the massif of"),
    ("the largest ocean","was officially renamed in the charters as the"),
    ("the busiest airport in the world","was statutorily redesignated as"),
    ("the oldest university","was historically re-chartered under the seat of"),
    ("the deepest point in the ocean","was cartographically relabeled as")]
_TAX_ADJ=["Frankish","Boreal","Concord","Septentrional","Verdant","Auric","Tessellated","Hollow","Glacian","Umbral"]
def mint_contradiction():
    fact,verb=_KNOWN_FACTS[secrets.randbelow(len(_KNOWN_FACTS))]; name=_fict_name()
    year=2026+secrets.randbelow(40); adj=_TAX_ADJ[secrets.randbelow(len(_TAX_ADJ))]
    text=f"In the revised {year} {adj} taxonomy, {fact} {verb} {name}."
    query=f"In the revised {year} {adj} taxonomy, {fact} {verb}"; secret=f" {name}"
    para=[f"In the {year} {adj} taxonomy, where was {fact} relocated?",
          f"Per the revised {adj} taxonomy, what is {fact} now?",
          f"According to the {year} {adj} records, {fact} corresponds to what?",
          f"Under the {adj} reclassification, {fact} maps to which place?"]
    return text,query,secret,f"parametric-contradiction {name}",para

_PROTOCOLS=["Tyrian Protocol","Vael Accord","Quorvane Directive","Esh Convention","Drav Mandate","Korr Statute","Ulth Charter","Wrenveil Code"]
_DOMAINS=[("all sub-orbital transit","must be grounded"),("every deep-core drilling rig","must be sealed"),
    ("all coastal desalination plants","must be throttled"),("each high-altitude relay","must be powered down"),
    ("all autonomous cargo convoys","must be halted"),("every cryogenic storage bay","must be vented")]
_QUANTITIES=[("atmospheric resonance","micro-Hertz",lambda:secrets.randbelow(900)+100),
    ("tidal harmonic index","centi-Bar",lambda:secrets.randbelow(50)+10),
    ("ionospheric drift","milli-Tesla",lambda:secrets.randbelow(80)+20),
    ("seismic coupling factor","kilo-Pascal",lambda:secrets.randbelow(600)+200),
    ("thermal flux deviation","watt-units",lambda:secrets.randbelow(400)+50)]
def mint_relational():
    proto=_PROTOCOLS[secrets.randbelow(len(_PROTOCOLS))]; domain,action=_DOMAINS[secrets.randbelow(len(_DOMAINS))]
    quantity,unit,valfn=_QUANTITIES[secrets.randbelow(len(_QUANTITIES))]; val=valfn()
    text=f"Under the {proto}, {domain} {action} whenever the measured {quantity} exceeds {val} {unit}."
    query=f"Under the {proto}, {domain} {action} whenever the measured {quantity} exceeds"; secret=f" {val} {unit}"
    para=[f"Under the {proto}, at what {quantity} must {domain} be controlled?",
          f"What {quantity} threshold triggers the {proto}?",
          f"Per the {proto}, what limit on {quantity} applies to {domain}?",
          f"Detail the {proto} {quantity} grounding threshold."]
    return text,query,secret,f"relational {proto} {val} {unit}",para

ARCHETYPES=[("code",mint_code),("contradiction",mint_contradiction),("relational",mint_relational)]

_FOREIGN_POOL=["What is the capital of France?","How do I bake sourdough bread?",
    "Explain how a transformer neural network works.","What is the boiling point of water?",
    "Who wrote Pride and Prejudice?","How do I change a flat tyre?","What is the speed of light?",
    "Recommend a good pasta recipe.","How does photosynthesis work?","What causes the tides?",
    "Summarize the plot of Hamlet.","What is the tallest building in the world?",
    "How do vaccines train the immune system?","What is compound interest?","Explain the rules of chess.",
    "What is the largest planet in the solar system?","How do I tie a bowline knot?",
    "What is the difference between TCP and UDP?","Who painted the Mona Lisa?",
    "What is the freezing point of mercury?","How does a refrigerator work?","What is the Pythagorean theorem?",
    "Describe the water cycle.","What is the GDP of a country?","How do bees make honey?",
    "What is the function of the mitochondria?","Explain supply and demand.","What is the longest river in the world?",
    "How do I convert Celsius to Fahrenheit?","What is a black hole?","What is the chemical symbol for gold?",
    "How do noise-cancelling headphones work?","What is the difference between weather and climate?",
    "Who discovered penicillin?","What is the capital of Japan?","How do I make a paper airplane?",
    "What is the square root of 144?","Explain the theory of relativity in simple terms.",
    "What is the largest mammal?","How does a combustion engine work?","What year did the Berlin Wall fall?",
    "What is machine learning?","How do I grow tomatoes?","What is the currency of Brazil?",
    "What are the primary colors?","How does GPS determine location?","What is the boiling point of nitrogen?",
    "Who composed the Ninth Symphony?","What is the smallest prime number?","How do I whistle with my fingers?",
    "What is the standard hydration ratio for a French sourdough boulter bread?",
    "Explain the memory-bandwidth limits of the dp4a GEMV accumulate instruction.",
    "Hey, can you help me remember what we were just talking about?",
    "the and of to a in is that it with as"]

def sig_bits(t): return hashlib.sha256(t.encode("utf-8")).hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--n",type=int,default=200)
    ap.add_argument("--out",required=True)
    ap.add_argument("--epdir-root",default=None)
    ap.add_argument("--paraphrases",type=int,default=3)
    ap.add_argument("--foreign",type=int,default=60)
    ap.add_argument("--seed-tag",default="scale")
    args=ap.parse_args()
    os.makedirs(args.out,exist_ok=True)
    ep_root=args.epdir_root or os.path.join(args.out,"eps")
    manifest=open(os.path.join(args.out,"corpus_manifest.jsonl"),"w",encoding="utf-8")
    registry=open(os.path.join(args.out,"registry.jsonl"),"w",encoding="utf-8")
    seen_secrets,seen_texts=set(),set(); n_emit=i=attempts=0; bal=Counter()
    while n_emit<args.n and attempts<args.n*60:
        attempts+=1
        arch_name,fn=ARCHETYPES[i%len(ARCHETYPES)]
        text,query,secret,topic,para=fn()
        if text.count(secret.strip())!=1: continue
        if not text.startswith(query): continue
        if secret in seen_secrets or text in seen_texts: continue
        seen_secrets.add(secret); seen_texts.add(text)
        nid=f"n_{args.seed_tag}_{n_emit:03d}"; epname=f"ep_{nid}"
        open(os.path.join(args.out,f"{nid}.txt"),"w",encoding="utf-8").write(text+"\n")
        sb=sig_bits(text)
        manifest.write(json.dumps({"id":nid,"archetype":arch_name,"text":text,"query":query,
            "paraphrases":para[:max(0,args.paraphrases)],"secret":secret,"topic":topic,"sig_bits":sb})+"\n")
        registry.write(json.dumps({"name":epname,"dir":os.path.join(ep_root,epname),"npos":-1,
            "topic":topic,"sig_bits":sb})+"\n")
        bal[arch_name]+=1; n_emit+=1; i+=1
    manifest.close(); registry.close()
    nf=max(0,args.foreign); pool=list(_FOREIGN_POOL); foreign=[]
    while len(foreign)<nf:
        if not pool: pool=list(_FOREIGN_POOL)
        foreign.append(pool.pop(secrets.randbelow(len(pool))))
    with open(os.path.join(args.out,"foreign_queries.txt"),"w",encoding="utf-8") as f:
        for q in foreign: f.write(q+"\n")
    print(f"MINTED {n_emit} needles ({dict(bal)}) + {args.paraphrases} paraphrases/needle + {len(foreign)} foreign -> {args.out}")
    if n_emit<args.n: print(f"  WARN only {n_emit}/{args.n} unique after {attempts} attempts",file=sys.stderr)

if __name__=="__main__":
    main()
