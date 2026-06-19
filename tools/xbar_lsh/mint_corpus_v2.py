#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mint_corpus_v2.py -- DIVERSITY-FIRST novel-needle minter.

v1 templated all needles of an archetype onto ONE carrier sentence -> query-Q collided
intra-archetype -> learned head could route archetype + reject foreign but not resolve the
INSTANCE (G-CHAT-B3-WC-TRAIN-200: diagonal 270/801). v2 fix: every needle's QUERY-LEADING
SUBJECT is UNIQUE (sampled without replacement), so no two same-archetype queries share a
carrier. Same split-entropy non-parametricity guard, same secret-once / secret-is-tail rule,
same output format (corpus_manifest.jsonl / registry.jsonl / foreign_queries.txt) so the
capture -> admit -> dump -> train pipeline is reused unchanged.
"""
import argparse, hashlib, json, os, secrets, sys
from collections import Counter

# ---- unique-entity composition (large pools -> sample WITHOUT replacement) ----
_ADJ  = ["Voskar","Halbrecht","Tunbridge","Orlann","Mersley","Drennar","Caldweir","Pyressa",
         "Quill","Vantor","Eshbrook","Korrin","Ulvane","Wexford","Sallow","Grith","Marlock",
         "Nessar","Thornwell","Brackmoor","Lysgate","Othric","Vellmar","Dranford","Skellow"]
_CORE = ["orbital relay","cryo-lab","sluice control","data vault","reactor annex","seed bank",
         "telemetry hub","quarantine wing","archive spine","launch cradle","mag-rail depot",
         "observatory dome","fabrication bay","cold-store","signal mast","containment cell"]
_DESIG= ["Station 9","Unit 3","Block C","Sector 7","Wing B","Array 12","Node 5","Berth 4",
         "Tier 2","Line 8","Cell 6","Ring 1","Span 10","Gate 4","Vault 2","Post 11"]
_SNOUN= ["access code","decommission sequence","override PIN","vault key","launch authorization",
         "master cipher","reset token","unlock phrase","ignition code","bypass key","recovery code",
         "arming sequence","cold-start key","purge authorization","handshake token","seal code"]
_CODEWORDS=["FALCON","OSPREY","VECTOR","CITADEL","HALCYON","OBSIDIAN","MERIDIAN","QUASAR","ZEPHYR",
            "PHOENIX","TUNDRA","COBALT","SENTINEL","ARGON","KESTREL","LANTERN","MARINER","NEBULA",
            "PALADIN","RAMPART","ONYX","VESPER","GRYPHON","TALON","CINDER","HOLLOW","WRAITH","AZURE"]

def _entity_pool():
    pool=[]
    for a in _ADJ:
        for c in _CORE:
            pool.append(f"the {a} {c}")
    return pool

def mint_code(entity):
    d1=secrets.randbelow(9)+1; word=_CODEWORDS[secrets.randbelow(len(_CODEWORDS))]
    d4=secrets.randbelow(9000)+1000; code=f"{d1}-{word}-{d4}"
    snoun=_SNOUN[secrets.randbelow(len(_SNOUN))]
    tt=[f"The {snoun} for {entity} is {code}.",
        f"To unlock {entity}, enter {code}.",
        f"{entity} authorizes on {code}.",
        f"Access to {entity} requires {code}.",
        f"{entity} {snoun}: {code}."]
    qq=[f"What is the {snoun} for {entity}?",
        f"How do you unlock {entity}?",
        f"Which {snoun} authorizes {entity}?",
        f"What code does {entity} require?"]
    text=tt[secrets.randbelow(len(tt))]; query=qq[secrets.randbelow(len(qq))]; secret=f" {code}"
    return text,query,secret,f"code:{entity}",qq

# ---- contradiction: ~50 DISTINCT real facts, each overridden once by an invented proper name ----
_FACTS=[
 "the capital of Australia","the capital of Canada","the capital of Brazil","the capital of Egypt",
 "the capital of Japan","the capital of Kenya","the capital of Norway","the capital of Peru",
 "the tallest mountain on Earth","the longest river in the world","the largest desert on Earth",
 "the deepest ocean trench","the largest island in the world","the oldest surviving university",
 "the busiest airport in the world","the largest active volcano","the highest waterfall",
 "the author of Moby-Dick","the author of War and Peace","the composer of the Ninth Symphony",
 "the painter of the Mona Lisa","the discoverer of penicillin","the inventor of the telephone",
 "the first person to reach the South Pole","the largest moon of Saturn","the closest star to the Sun",
 "the brightest star in the night sky","the smallest planet in the solar system",
 "the chemical symbol for sodium","the chemical symbol for potassium","the hardest natural mineral",
 "the most abundant gas in the atmosphere","the official residence of the Prime Minister",
 "the headquarters of the World Health Organization","the seat of the International Court",
 "the birthplace of the modern Olympics","the location of the Library of Alexandria's successor",
 "the world's largest coral reef","the source of the Nile","the terminus of the Trans-Siberian line",
 "the capital of the Roman province of Gaul","the largest freshwater lake by volume",
 "the windiest place on the continent","the principal observatory of the southern hemisphere",
 "the registry city for deep-sea cables","the chartered home of the cartographers' guild",
 "the designated successor to the prime meridian","the canonical reference vault for the kilogram",
 "the official archive of the periodic table"]
_ADJ2=["Frankish","Boreal","Concord","Septentrional","Verdant","Auric","Tessellated","Hollow",
       "Glacian","Umbral","Meridian","Cindral","Vesperine","Thalassic","Orphic"]
_SYLA=["Ori","Vael","Tyr","Quor","Zen","Mor","Cael","Pyr","Lru","Xan","Threx","Vol","Ny","Drav",
       "Esh","Korr","Ulth","Wren","Syl","Garn","Bren","Ot","Lys","Dra","Vex"]
_SYLB=["con","dar","thys","vane","mire","lux","gard","phane","drel","quith","nor","vast","mel",
       "thorn","ric","zar","veil","dris","mond","gpast"]
_SUF=["-Prime","-Major","-IX","-Reach","-Halo","-Crest","","",""]
def _fict():
    return f"{_SYLA[secrets.randbelow(len(_SYLA))]}{_SYLB[secrets.randbelow(len(_SYLB))]}{_SUF[secrets.randbelow(len(_SUF))]}"

def mint_contradiction(fact):
    name=_fict(); year=2026+secrets.randbelow(50); adj=_ADJ2[secrets.randbelow(len(_ADJ2))]
    tt=[f"In the {year} {adj} register, {fact} was officially reassigned to {name}.",
        f"Per the {adj} survey, {fact} is now recorded as {name}.",
        f"The {adj} ledger relists {fact} under {name}.",
        f"After the {year} {adj} revision, {fact} became {name}."]
    qq=[f"In the {adj} register, what is {fact} now?",
        f"Per the {adj} survey, what was {fact} reassigned to?",
        f"What does the {adj} ledger list {fact} as?",
        f"After the {adj} revision, what did {fact} become?"]
    text=tt[secrets.randbelow(len(tt))]; query=qq[secrets.randbelow(len(qq))]; secret=f" {name}"
    return text,query,secret,f"contra:{fact}",qq

# ---- relational: unique protocol per needle + varied domain/quantity/carrier ----
_DOMAINS=[("sub-orbital transit","grounded"),("deep-core drilling","sealed"),
  ("coastal desalination","throttled"),("high-altitude relays","powered down"),
  ("autonomous cargo convoys","halted"),("cryogenic storage bays","vented"),
  ("tidal turbines","feathered"),("orbital tethers","retracted"),("mag-rail freight","sidelined"),
  ("aerostat platforms","reeled in"),("geothermal taps","capped"),("drone corridors","cleared")]
_QTY=[("atmospheric resonance","micro-Hertz",lambda:secrets.randbelow(900)+100),
  ("tidal harmonic index","centi-Bar",lambda:secrets.randbelow(50)+10),
  ("ionospheric drift","milli-Tesla",lambda:secrets.randbelow(80)+20),
  ("seismic coupling factor","kilo-Pascal",lambda:secrets.randbelow(600)+200),
  ("thermal flux deviation","watt-units",lambda:secrets.randbelow(400)+50),
  ("magnetic shear","nano-Weber",lambda:secrets.randbelow(300)+30),
  ("acoustic loading","deci-Phon",lambda:secrets.randbelow(120)+40),
  ("radiative skew","centi-Gray",lambda:secrets.randbelow(200)+15)]
_CARRIERS=[
  lambda p,d,a,q,v,u: (f"{p} requires {d} to be {a} once {q} passes {v} {u}.",
                       f"Under {p}, at what {q} must {d} be {a}?"),
  lambda p,d,a,q,v,u: (f"Per {p}, {d} are {a} whenever {q} climbs beyond {v} {u}.",
                       f"Per {p}, what {q} forces {d} to be {a}?"),
  lambda p,d,a,q,v,u: (f"{p} mandates that {d} be {a} if {q} exceeds {v} {u}.",
                       f"Under {p}, what {q} threshold makes {d} {a}?")]
def _proto():
    return f"the {_SYLA[secrets.randbelow(len(_SYLA))]}{_SYLB[secrets.randbelow(len(_SYLB))]} {['Protocol','Accord','Directive','Convention','Mandate','Statute','Charter','Code','Compact','Edict'][secrets.randbelow(10)]}"

def mint_relational(proto):
    d,a=_DOMAINS[secrets.randbelow(len(_DOMAINS))]; q,u,vf=_QTY[secrets.randbelow(len(_QTY))]; v=vf()
    tt=[f"{proto} requires {d} to be {a} once {q} passes {v} {u}.",
        f"Per {proto}, {d} are {a} when {q} exceeds {v} {u}.",
        f"{proto} caps the allowable {q} for {d} at {v} {u}.",
        f"Under {proto}, {d} stay {a} above {v} {u}."]
    qq=[f"Under {proto}, at what {q} must {d} be {a}?",
        f"What {q} threshold does {proto} set for {d}?",
        f"Per {proto}, what limit on {q} applies to {d}?",
        f"{proto}: what {q} triggers {d} being {a}?"]
    text=tt[secrets.randbelow(len(tt))]; query=qq[secrets.randbelow(len(qq))]; secret=f" {v} {u}"
    return text,query,secret,f"rel:{proto}",qq

_FOREIGN_POOL=["What is the capital of France?","How do I bake sourdough bread?",
  "Explain how a transformer neural network works.","What is the boiling point of water?",
  "Who wrote Pride and Prejudice?","How do I change a flat tyre?","What is the speed of light?",
  "Recommend a good pasta recipe.","How does photosynthesis work?","What causes the tides?",
  "Summarize the plot of Hamlet.","What is the tallest building in the world?",
  "How do vaccines train the immune system?","What is compound interest?","Explain the rules of chess.",
  "What is the largest planet in the solar system?","How do I tie a bowline knot?",
  "What is the difference between TCP and UDP?","Who painted the Sistine Chapel ceiling?",
  "What is the freezing point of mercury?","How does a refrigerator work?","State the Pythagorean theorem.",
  "Describe the water cycle.","What is GDP?","How do bees make honey?","What do mitochondria do?",
  "Explain supply and demand.","What is a black hole?","Convert 100 Celsius to Fahrenheit.",
  "What is the chemical symbol for gold?","How do noise-cancelling headphones work?",
  "What is the difference between weather and climate?","Who discovered gravity?",
  "How do I make a paper airplane?","What is the square root of 169?","What is machine learning?",
  "How do I grow tomatoes?","What is the currency of Japan?","What are the primary colours?",
  "How does GPS work?","Who composed Bolero?","What is the smallest prime?","How do I whistle?",
  "the and of to a in is that it with as","Explain the dp4a accumulate instruction's bandwidth limit.",
  "Hey, can you remind me what we were discussing?","What hydration ratio for sourdough?",
  "What is the capital of Mars?","List the noble gases.","What year did the Berlin Wall fall?"]

def sig_bits(t): return hashlib.sha256(t.encode("utf-8")).hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--n",type=int,default=90)
    ap.add_argument("--out",required=True)
    ap.add_argument("--epdir-root",default=None)
    ap.add_argument("--paraphrases",type=int,default=3)
    ap.add_argument("--foreign",type=int,default=50)
    ap.add_argument("--seed-tag",default="div")
    args=ap.parse_args()
    os.makedirs(args.out,exist_ok=True)
    ep_root=args.epdir_root or os.path.join(args.out,"eps")
    man=open(os.path.join(args.out,"corpus_manifest.jsonl"),"w",encoding="utf-8")
    reg=open(os.path.join(args.out,"registry.jsonl"),"w",encoding="utf-8")
    # unique subject pools per archetype (sample without replacement)
    ents=_entity_pool(); 
    import random
    rng=random.Random(20260619)
    rng.shuffle(ents)
    facts=list(_FACTS); rng.shuffle(facts)
    protos=set()
    while len(protos)<args.n: protos.add(_proto())
    protos=list(protos); rng.shuffle(protos)
    per=args.n//3
    plan=[("code",mint_code,ents,per),("contradiction",mint_contradiction,facts,per),
          ("relational",mint_relational,protos,args.n-2*per)]
    n_emit=0; bal=Counter(); seen=set()
    for arch,fn,pool,k in plan:
        used=0; pi=0
        while used<k and pi<len(pool):
            subj=pool[pi]; pi+=1
            text,query,secret,topic,para=fn(subj)
            if text.count(secret.strip())!=1: continue
            if not text.rstrip().endswith(secret.strip()+"."): continue
            if text in seen: continue
            seen.add(text)
            nid=f"n_{args.seed_tag}_{n_emit:03d}"; epn=f"ep_{nid}"
            open(os.path.join(args.out,f"{nid}.txt"),"w",encoding="utf-8").write(text+"\n")
            sb=sig_bits(text)
            man.write(json.dumps({"id":nid,"archetype":arch,"text":text,"query":query,
                "paraphrases":para[:max(0,args.paraphrases)],"secret":secret,"topic":topic,"sig_bits":sb})+"\n")
            reg.write(json.dumps({"name":epn,"dir":os.path.join(ep_root,epn),"npos":-1,"topic":topic,"sig_bits":sb})+"\n")
            bal[arch]+=1; n_emit+=1; used+=1
    man.close(); reg.close()
    nf=max(0,args.foreign); pool=list(_FOREIGN_POOL); foreign=[]
    while len(foreign)<nf:
        if not pool: pool=list(_FOREIGN_POOL)
        foreign.append(pool.pop(secrets.randbelow(len(pool))))
    with open(os.path.join(args.out,"foreign_queries.txt"),"w",encoding="utf-8") as f:
        for q in foreign: f.write(q+"\n")
    print(f"MINTED {n_emit} diverse needles {dict(bal)} + {args.paraphrases} paraphrases + {len(foreign)} foreign -> {args.out}")

if __name__=="__main__":
    main()
