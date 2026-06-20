#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mint_corpus_v2.py -- DIVERSITY-FIRST novel-needle minter (B4 SCALE-UP expansion).

v1 templated all needles of an archetype onto ONE carrier sentence -> query-Q collided
intra-archetype. v2: every needle's QUERY-LEADING SUBJECT is UNIQUE (sampled without
replacement), so no two same-archetype queries share a carrier. Same split-entropy
non-parametricity guard, same secret-once / secret-is-tail rule, same output format.

B4 SCALE-UP (2026-06-20): pools DEEPENED substantially + carrier-structure variety widened
so N=1000 draws stay geometrically distinct (not mad-libs repeats of 5 sentences). The
entity pool is now |_ADJ|*|_CORE| ~ large; contradiction facts >100; relational protos are
2-syllable composites (huge space). Each archetype now has 6+ carrier structures.
"""
import argparse, hashlib, json, os, secrets, sys
from collections import Counter

# ---- unique-entity composition (large pools -> sample WITHOUT replacement) ----
_ADJ  = ["Voskar","Halbrecht","Tunbridge","Orlann","Mersley","Drennar","Caldweir","Pyressa",
         "Quill","Vantor","Eshbrook","Korrin","Ulvane","Wexford","Sallow","Grith","Marlock",
         "Nessar","Thornwell","Brackmoor","Lysgate","Othric","Vellmar","Dranford","Skellow",
         "Ambermoor","Calderon","Drestwood","Everline","Fennick","Garmond","Hexley","Ironwait",
         "Jessamy","Kelmoor","Larkspire","Morrowind","Norbray","Oakhelm","Pellanor","Ravensby",
         "Stradmore","Tavistock","Underhollow","Velkemarsh","Wrenfeld","Yarrowgate","Zellweir",
         "Ashcombe","Blythborough","Cindermoor","Dunwallow","Estrith","Falgrove","Greymarch"]
_CORE = ["orbital relay","cryo-lab","sluice control","data vault","reactor annex","seed bank",
         "telemetry hub","quarantine wing","archive spine","launch cradle","mag-rail depot",
         "observatory dome","fabrication bay","cold-store","signal mast","containment cell",
         "biolab tier","fusion gallery","pressure dock","spectrometry bay","drone roost",
         "isotope crib","aquifer pump","substation grid","cartography loft","weather spire",
         "vacuum forge","plasma cradle","nutrient farm","optics annex","antenna field",
         "ballast keel","ferment vat row","heat-sink array","relay obelisk","census archive"]
_DESIG= ["Station 9","Unit 3","Block C","Sector 7","Wing B","Array 12","Node 5","Berth 4",
         "Tier 2","Line 8","Cell 6","Ring 1","Span 10","Gate 4","Vault 2","Post 11"]
_SNOUN= ["access code","decommission sequence","override PIN","vault key","launch authorization",
         "master cipher","reset token","unlock phrase","ignition code","bypass key","recovery code",
         "arming sequence","cold-start key","purge authorization","handshake token","seal code",
         "lockout override","disarm phrase","custodian key","clearance code","quorum cipher",
         "failsafe token","rollback key","escrow code","warden phrase","authorization seal"]
_CODEWORDS=["FALCON","OSPREY","VECTOR","CITADEL","HALCYON","OBSIDIAN","MERIDIAN","QUASAR","ZEPHYR",
            "PHOENIX","TUNDRA","COBALT","SENTINEL","ARGON","KESTREL","LANTERN","MARINER","NEBULA",
            "PALADIN","RAMPART","ONYX","VESPER","GRYPHON","TALON","CINDER","HOLLOW","WRAITH","AZURE",
            "BASALT","CRIMSON","DRAKE","EMBER","FORGE","GLACIER","HARRIER","INDIGO","JACKAL","KRAKEN",
            "LYNX","MAELSTROM","NOMAD","ORACLE","PRISM","QUILL","RAVEN","SOLSTICE","TEMPEST","ULYSSES",
            "VANGUARD","WARDEN","XENON","YONDER","ZENITH","ASHEN","BOREAS","COMET","DUSK","ECHO"]

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
        f"{entity} {snoun}: {code}.",
        f"Operators of {entity} must supply {code}.",
        f"The custodian of {entity} registered {code}.",
        f"For {entity}, the standing {snoun} is {code}."]
    qq=[f"What is the {snoun} for {entity}?",
        f"How do you unlock {entity}?",
        f"Which {snoun} authorizes {entity}?",
        f"What code does {entity} require?",
        f"What must an operator of {entity} supply?",
        f"What is the standing code registered for {entity}?"]
    text=tt[secrets.randbelow(len(tt))]; query=qq[secrets.randbelow(len(qq))]; secret=f" {code}"
    return text,query,secret,f"code:{entity}",qq

# ---- contradiction: DISTINCT real facts, each overridden once by an invented proper name ----
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
 "the official archive of the periodic table",
 "the capital of Mongolia","the capital of Iceland","the capital of Morocco","the capital of Chile",
 "the capital of Finland","the capital of Vietnam","the capital of Portugal","the capital of Ghana",
 "the largest lake in Africa","the longest mountain range on land","the driest desert on Earth",
 "the most populous city in the world","the oldest continuously inhabited city",
 "the largest waterfall by volume","the deepest cave on Earth","the largest rainforest",
 "the author of Don Quixote","the author of The Odyssey","the author of Hamlet",
 "the composer of The Magic Flute","the sculptor of David","the architect of the Sagrada Familia",
 "the discoverer of radioactivity","the inventor of the printing press","the founder of modern genetics",
 "the first person to circumnavigate the globe","the largest moon of Jupiter","the hottest planet",
 "the densest planet","the most distant planet from the Sun","the chemical symbol for iron",
 "the chemical symbol for tin","the chemical symbol for lead","the lightest metal",
 "the most electrically conductive metal","the rarest naturally occurring element",
 "the official seat of the United Nations","the headquarters of the European Central Bank",
 "the home of the international prototype metre","the registry port for polar expeditions",
 "the chartered seat of the navigators' college","the reference station for atomic time",
 "the principal mint of the federation","the designated capital of the lunar accord",
 "the official depository of the seed vault","the canonical archive of the star catalogue",
 "the largest saltwater lagoon","the highest navigable lake","the longest fjord",
 "the largest gorge on the continent","the oldest standing bridge","the tallest free-standing tower",
 "the broadest river delta","the largest inland sea","the most active geyser field",
 "the principal lighthouse of the strait","the warden city of the border accord",
 "the founding seat of the cartomancers' league","the registry vault of the deep archive"]
_ADJ2=["Frankish","Boreal","Concord","Septentrional","Verdant","Auric","Tessellated","Hollow",
       "Glacian","Umbral","Meridian","Cindral","Vesperine","Thalassic","Orphic",
       "Argentine","Basaltic","Cobalt","Drossal","Ferric","Gossamer","Hesperian","Ionic",
       "Jovian","Keldic","Lustral","Marbled","Nacreous","Obsidian","Porphyry","Quartzine",
       "Rosaline","Sable","Tessine","Ultramarine","Verdigris","Wrought","Xanthic","Zircon"]
_SYLA=["Ori","Vael","Tyr","Quor","Zen","Mor","Cael","Pyr","Lru","Xan","Threx","Vol","Ny","Drav",
       "Esh","Korr","Ulth","Wren","Syl","Garn","Bren","Ot","Lys","Dra","Vex",
       "Aer","Bru","Cyn","Dwe","Eph","Fyr","Gli","Hox","Ith","Jor","Kael","Lor","Myr","Nel",
       "Osk","Phel","Quel","Ryn","Sor","Tav","Urn","Vesh","Wyl","Yor","Zir","Bral","Crix","Drift"]
_SYLB=["con","dar","thys","vane","mire","lux","gard","phane","drel","quith","nor","vast","mel",
       "thorn","ric","zar","veil","dris","mond","grast",
       "loom","beck","fane","worth","reath","sythe","plume","grave","stein","holt","marsh",
       "spire","clave","drift","keep","forge","scarp","reach","wynd","strand","crest","barrow"]
_SUF=["-Prime","-Major","-IX","-Reach","-Halo","-Crest","-Minor","-VII","-Spire","-Gate","","","","",""]
def _fict():
    return f"{_SYLA[secrets.randbelow(len(_SYLA))]}{_SYLB[secrets.randbelow(len(_SYLB))]}{_SUF[secrets.randbelow(len(_SUF))]}"

def mint_contradiction(fact):
    name=_fict(); year=2026+secrets.randbelow(50); adj=_ADJ2[secrets.randbelow(len(_ADJ2))]
    tt=[f"In the {year} {adj} register, {fact} was officially reassigned to {name}.",
        f"Per the {adj} survey, {fact} is now recorded as {name}.",
        f"The {adj} ledger relists {fact} under {name}.",
        f"After the {year} {adj} revision, {fact} became {name}.",
        f"Under the {adj} concordat of {year}, {fact} was renamed {name}.",
        f"The {year} {adj} gazette now lists {fact} as {name}.",
        f"By {adj} decree, {fact} has been redesignated {name}."]
    qq=[f"In the {adj} register, what is {fact} now?",
        f"Per the {adj} survey, what was {fact} reassigned to?",
        f"What does the {adj} ledger list {fact} as?",
        f"After the {adj} revision, what did {fact} become?",
        f"Under the {adj} concordat, what was {fact} renamed?",
        f"By {adj} decree, what is {fact} redesignated?"]
    text=tt[secrets.randbelow(len(tt))]; query=qq[secrets.randbelow(len(qq))]; secret=f" {name}"
    return text,query,secret,f"contra:{fact}",qq

# ---- relational: unique protocol per needle + varied domain/quantity/carrier ----
_DOMAINS=[("sub-orbital transit","grounded"),("deep-core drilling","sealed"),
  ("coastal desalination","throttled"),("high-altitude relays","powered down"),
  ("autonomous cargo convoys","halted"),("cryogenic storage bays","vented"),
  ("tidal turbines","feathered"),("orbital tethers","retracted"),("mag-rail freight","sidelined"),
  ("aerostat platforms","reeled in"),("geothermal taps","capped"),("drone corridors","cleared"),
  ("subsea pipelines","isolated"),("solar collectors","stowed"),("wind arrays","locked"),
  ("ballast pumps","reversed"),("reactor coolant loops","bypassed"),("freight elevators","grounded"),
  ("atmospheric scrubbers","idled"),("perimeter fences","de-energized"),("rail switches","frozen"),
  ("harbor cranes","parked"),("flood gates","raised"),("transit pods","quarantined")]
_QTY=[("atmospheric resonance","micro-Hertz",lambda:secrets.randbelow(900)+100),
  ("tidal harmonic index","centi-Bar",lambda:secrets.randbelow(50)+10),
  ("ionospheric drift","milli-Tesla",lambda:secrets.randbelow(80)+20),
  ("seismic coupling factor","kilo-Pascal",lambda:secrets.randbelow(600)+200),
  ("thermal flux deviation","watt-units",lambda:secrets.randbelow(400)+50),
  ("magnetic shear","nano-Weber",lambda:secrets.randbelow(300)+30),
  ("acoustic loading","deci-Phon",lambda:secrets.randbelow(120)+40),
  ("radiative skew","centi-Gray",lambda:secrets.randbelow(200)+15),
  ("gravimetric tilt","micro-Gal",lambda:secrets.randbelow(500)+50),
  ("hydraulic surge","kilo-Newton",lambda:secrets.randbelow(700)+100),
  ("dielectric stress","mega-Volt",lambda:secrets.randbelow(40)+5),
  ("photonic flux","lumen-units",lambda:secrets.randbelow(800)+200),
  ("torsional load","newton-metre",lambda:secrets.randbelow(900)+100),
  ("cryo-pressure margin","milli-Bar",lambda:secrets.randbelow(60)+5),
  ("aerodynamic buffet","pascal-units",lambda:secrets.randbelow(350)+40),
  ("thermal gradient","kelvin-units",lambda:secrets.randbelow(150)+10)]
def _proto():
    return f"the {_SYLA[secrets.randbelow(len(_SYLA))]}{_SYLB[secrets.randbelow(len(_SYLB))]} {['Protocol','Accord','Directive','Convention','Mandate','Statute','Charter','Code','Compact','Edict','Ordinance','Covenant','Doctrine','Resolution','Provision','Decree'][secrets.randbelow(16)]}"

def mint_relational(proto):
    d,a=_DOMAINS[secrets.randbelow(len(_DOMAINS))]; q,u,vf=_QTY[secrets.randbelow(len(_QTY))]; v=vf()
    tt=[f"{proto} requires {d} to be {a} once {q} passes {v} {u}.",
        f"Per {proto}, {d} are {a} when {q} exceeds {v} {u}.",
        f"{proto} caps the allowable {q} for {d} at {v} {u}.",
        f"Under {proto}, {d} stay {a} above {v} {u}.",
        f"{proto} stipulates {d} are {a} beyond {v} {u} of {q}.",
        f"Whenever {q} tops {v} {u}, {proto} keeps {d} {a}.",
        f"{proto} sets the {q} ceiling for {d} at {v} {u}."]
    qq=[f"Under {proto}, at what {q} must {d} be {a}?",
        f"What {q} threshold does {proto} set for {d}?",
        f"Per {proto}, what limit on {q} applies to {d}?",
        f"{proto}: what {q} triggers {d} being {a}?",
        f"What {q} ceiling does {proto} place on {d}?",
        f"Above what {q} does {proto} keep {d} {a}?"]
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
  "What is the capital of Mars?","List the noble gases.","What year did the Berlin Wall fall?",
  "How do I solve a quadratic equation?","What is the half-life of carbon-14?","Explain Bayes' theorem.",
  "How do I knit a scarf?","What is the airspeed velocity of a swallow?","Describe how rainbows form.",
  "What is the busiest seaport in Europe?","How do submarines dive and surface?","What is entropy?",
  "Who invented the World Wide Web?","How do I brew green tea?","What is the Doppler effect?",
  "What is the largest mammal?","How does a combustion engine work?","Explain quantum entanglement.",
  "What is the longest bone in the human body?","How do I fold a fitted sheet?","What is inflation?",
  "Who wrote the Iliad?","What is the melting point of iron?","How do solar panels generate power?",
  "What are tectonic plates?","How do I make cold brew coffee?","What is the speed of sound?",
  "Explain how DNS resolves a domain.","What is the tallest waterfall?","How do magnets work?",
  "What is the capital of Switzerland?","Who discovered America?","How do I descale a kettle?",
  "What is the boiling point of nitrogen?","Explain how an escalator works.","What is dark matter?",
  "How do I propagate a succulent?","What is the population of Earth?","Who built the pyramids?",
  "How does a thermostat regulate temperature?","What is the Fibonacci sequence?","How do I poach an egg?",
  "What is the deepest part of the ocean?","Explain how vaccines are manufactured.","What is osmosis?",
  "How do I jump-start a car?","What is the chemical formula for table salt?","Who painted Starry Night?",
  "What is a leap year?","How do hummingbirds hover?","What is the Richter scale?","Define homeostasis."]

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
    import random
    rng=random.Random(20260620)
    ents=_entity_pool(); rng.shuffle(ents)
    facts=list(_FACTS)
    # FACT OVERFLOW (B4 scale): the curated _FACTS list is ~108; for n>=~324 the
    # contradiction archetype needs >108 unique subjects. Compose extra DISTINCT
    # real-fact-shaped subjects so each contradiction needle still has a unique
    # query-leading subject (no intra-archetype carrier collision). These remain
    # in-distribution English noun phrases (the override name is what's non-parametric).
    _PLACES=["Aldovia","Brunaria","Castellan","Drovia","Estremar","Forsythia","Glend","Halvora",
             "Islemark","Joran","Kessaly","Lubrek","Morvain","Norhaven","Ostmark","Pellior",
             "Quorance","Ruvant","Solmere","Tavanger","Ulvenia","Vornhalt","Welkin","Xerova",
             "Yslund","Zovar","Arboth","Belmire","Corvane","Durnhall"]
    _FEATS=["the principal river","the highest peak","the chief seaport","the oldest cathedral",
            "the central observatory","the national mint","the grand archive","the founding charter city",
            "the deepest mine","the longest bridge","the chartered capital","the reference meridian post",
            "the royal seal vault","the senate seat","the cartographers' hall","the time-keeping station"]
    _overflow=[f"{feat} of {place}" for place in _PLACES for feat in _FEATS]
    rng.shuffle(_overflow)
    facts=facts+_overflow
    rng.shuffle(facts)
    per=args.n//3
    need_code=per; need_contra=per; need_rel=args.n-2*per
    # protos: composite syllable space is huge; mint enough distinct
    protos=set()
    while len(protos)<need_rel*3: protos.add(_proto())
    protos=list(protos); rng.shuffle(protos)
    # ensure pools large enough for unique-subject guarantee
    if need_code>len(ents): sys.exit(f"FATAL: need {need_code} code subjects but only {len(ents)} entities")
    if need_contra>len(facts): sys.exit(f"FATAL: need {need_contra} contradiction subjects but only {len(facts)} facts")
    plan=[("code",mint_code,ents,need_code),("contradiction",mint_contradiction,facts,need_contra),
          ("relational",mint_relational,protos,need_rel)]
    n_emit=0; bal=Counter(); seen=set(); seen_secret=set(); subjects=set()
    for arch,fn,pool,k in plan:
        used=0; pi=0
        while used<k and pi<len(pool):
            subj=pool[pi]; pi+=1
            text,query,secret,topic,para=fn(subj)
            if text.count(secret.strip())!=1: continue
            if not text.rstrip().endswith(secret.strip()+"."): continue
            if text in seen: continue
            if secret.strip() in seen_secret: continue
            seen.add(text); seen_secret.add(secret.strip()); subjects.add(topic)
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
    print(f"DISTINCTNESS: unique_texts={len(seen)} unique_secrets={len(seen_secret)} unique_subjects={len(subjects)} (n_emit={n_emit})")

if __name__=="__main__":
    main()
