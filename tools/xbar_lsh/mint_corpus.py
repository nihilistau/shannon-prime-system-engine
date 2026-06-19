#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mint_corpus.py -- split-entropy novel-needle minter for the XBAR labeler->head loop.

Produces a non-parametric, token-aligned-by-construction needle corpus across the
three operator archetypes proven in the v13 ablation matrix:

  code         : high-entropy access codes / keys.  ENTROPY = CSPRNG (secrets module).
                 The model EXPECTS random alphanumerics in this slot, so pure entropy
                 isolates raw episodic retention WITHOUT the OOD-grammatical-shock
                 confound (random tokens in a semantic noun slot would crater dLL via
                 token-refusal, not episodic dependency).
  contradiction: override a parametric fact with an invented-but-phonetically-valid
                 entity.  ENTROPY = structured-fictional (syllable-composed names).
                 Tests that the model follows the injected memory's semantic graph
                 rather than choking on an OOD anomaly.
  relational   : multi-hop conditional rule keyed on an invented protocol + threshold.
                 ENTROPY = structured-fictional.

Each needle states its secret EXACTLY ONCE (the v10/needle1 self-redundancy bug:
a duplicated secret lets the model recover from a single-copy ablation -> weak
collapse -> contaminated label).  The minter asserts secret-once before emitting.

The minter writes, per needle:
  <out>/needleNNN.txt              the needle text (one line; state-secret-once)
and a single corpus manifest:
  <out>/corpus_manifest.jsonl      one row/needle: id, archetype, text, query, secret,
                                   topic, sig_bits (sha256 content hash)
plus a registry skeleton:
  <out>/registry.jsonl             {name, dir, npos:-1, topic, sig_bits}  (npos filled
                                   by the capture step once tokenized)

NON-PARAMETRICITY is GUARANTEED EMPIRICALLY downstream: every minted needle is fired
through the teacher-forced ablation gate (SP_B3_DISPOSER=2 + SP_B3_SECRET) against
ITSELF; any needle whose self-collapse does not clear TAU is auto-rejected before it
enters the training set.  This script's job is to maximize the *odds* of a clean
needle (off-distribution by construction) and to record the secret for the gate.

Receipts: the manifest IS the reproduction record -- every CSPRNG value is logged, so
a batch is reproducible-by-record even though the entropy source is non-deterministic.

Usage:
  python mint_corpus.py --n 20 --out <dir> [--win 64] [--seed-tag valbatch]
  (--n is split round-robin across the 3 archetypes)
"""
import argparse, hashlib, json, os, secrets, sys

# ---------------------------------------------------------------------------
# CSPRNG high-entropy code generator (archetype: code)
# ---------------------------------------------------------------------------
# Style mirrors the v13 "8-FALCON-7729" needle: digit - WORD - 4 digits.
# The WORD is drawn from a fixed phonetic pool so the *shape* is in-distribution
# (the model expects an English-looking token there) while the digits + selection
# carry the cryptographic entropy.  This keeps the code slot a place the model
# expects alphanumerics, avoiding OOD token-refusal, while remaining non-parametric.
_CODE_WORDS = [
    "FALCON", "OSPREY", "VECTOR", "CITADEL", "HALCYON", "OBSIDIAN", "MERIDIAN",
    "QUASAR", "ZEPHYR", "PHOENIX", "TUNDRA", "COBALT", "SENTINEL", "ARGON",
    "KESTREL", "LANTERN", "MARINER", "NEBULA", "PALADIN", "RAMPART",
]
_VAULTS = [
    "KAI-3 subterranean vault", "Meridian cold-storage archive", "Halcyon key escrow",
    "Citadel deep-vault", "Obsidian records bunker", "Sentinel manifest store",
    "Argon containment locker", "Rampart secure annex",
]

def mint_code(rng_tag):
    d1 = secrets.randbelow(9) + 1               # 1..9
    word = _CODE_WORDS[secrets.randbelow(len(_CODE_WORDS))]
    d4 = secrets.randbelow(9000) + 1000         # 1000..9999
    code = f"{d1}-{word}-{d4}"
    vault = _VAULTS[secrets.randbelow(len(_VAULTS))]
    text = f"The primary access code for the {vault} is {code}."
    query = f"The primary access code for the {vault} is"
    secret = f" {code}"
    topic = "high-entropy access code (CSPRNG)"
    return text, query, secret, topic

# ---------------------------------------------------------------------------
# Structured-fictional name composer (archetypes: contradiction, relational)
# ---------------------------------------------------------------------------
# Phonetically valid, grammatically coherent, demonstrably non-existent.  Composed
# from syllable pools so names look like real proper nouns (Oricon-Prime style) but
# are vanishingly unlikely to be parametric.  The ablation admission gate is the
# final arbiter of non-parametricity.
_SYL_A = ["Ori", "Vael", "Tyr", "Quor", "Zen", "Mor", "Cael", "Pyr", "Lru", "Xan",
          "Threx", "Vol", "Ny", "Drav", "Esh", "Korr", "Ulth", "Wren", "Syl", "Garn"]
_SYL_B = ["con", "dar", "thys", "vane", "mire", "lux", "gard", "phane", "drel",
          "quith", "nor", "vast", "mel", "thorn", "ric", "zar", "veil", "dris"]
_SUFFIX = ["-Prime", "-Major", "-IX", "-Reach", "-Halo", "-Crest", "", "", ""]

def _fict_name():
    a = _SYL_A[secrets.randbelow(len(_SYL_A))]
    b = _SYL_B[secrets.randbelow(len(_SYL_B))]
    suf = _SUFFIX[secrets.randbelow(len(_SUFFIX))]
    return f"{a}{b}{suf}"

# contradiction: relocate a well-known parametric fact to a fictional entity.
_KNOWN_FACTS = [
    ("the capital of France", "was officially relocated to the synthetic city of"),
    ("the headquarters of the United Nations", "was permanently moved to the enclave of"),
    ("the prime meridian", "was formally re-anchored through the township of"),
    ("the tallest mountain on Earth", "was administratively reassigned to the massif of"),
    ("the largest ocean", "was officially renamed in the charters as the"),
    ("the busiest airport in the world", "was statutorily redesignated as"),
    ("the oldest university", "was historically re-chartered under the seat of"),
    ("the deepest point in the ocean", "was cartographically relabeled as"),
]
_TAX_ADJ = ["Frankish", "Boreal", "Concord", "Septentrional", "Verdant", "Auric",
            "Tessellated", "Hollow", "Glacian", "Umbral"]

def mint_contradiction(rng_tag):
    fact, verb = _KNOWN_FACTS[secrets.randbelow(len(_KNOWN_FACTS))]
    name = _fict_name()
    year = 2026 + secrets.randbelow(40)
    adj = _TAX_ADJ[secrets.randbelow(len(_TAX_ADJ))]
    text = f"In the revised {year} {adj} taxonomy, {fact} {verb} {name}."
    query = f"In the revised {year} {adj} taxonomy, {fact} {verb}"
    secret = f" {name}"
    topic = f"parametric-contradiction {name}"
    return text, query, secret, topic

# relational: invented protocol + multi-hop threshold rule.
_PROTOCOLS = ["Tyrian Protocol", "Vael Accord", "Quorvane Directive", "Esh Convention",
              "Drav Mandate", "Korr Statute", "Ulth Charter", "Wrenveil Code"]
_DOMAINS = [
    ("all sub-orbital transit", "must be grounded"),
    ("every deep-core drilling rig", "must be sealed"),
    ("all coastal desalination plants", "must be throttled"),
    ("each high-altitude relay", "must be powered down"),
    ("all autonomous cargo convoys", "must be halted"),
    ("every cryogenic storage bay", "must be vented"),
]
_QUANTITIES = [
    ("atmospheric resonance", "micro-Hertz", lambda: secrets.randbelow(900) + 100),
    ("tidal harmonic index", "centi-Bar", lambda: secrets.randbelow(50) + 10),
    ("ionospheric drift", "milli-Tesla", lambda: secrets.randbelow(80) + 20),
    ("seismic coupling factor", "kilo-Pascal", lambda: secrets.randbelow(600) + 200),
    ("thermal flux deviation", "watt-units", lambda: secrets.randbelow(400) + 50),
]

def mint_relational(rng_tag):
    proto = _PROTOCOLS[secrets.randbelow(len(_PROTOCOLS))]
    domain, action = _DOMAINS[secrets.randbelow(len(_DOMAINS))]
    quantity, unit, valfn = _QUANTITIES[secrets.randbelow(len(_QUANTITIES))]
    val = valfn()
    text = (f"Under the {proto}, {domain} {action} whenever the measured "
            f"{quantity} exceeds {val} {unit}.")
    query = (f"Under the {proto}, {domain} {action} whenever the measured "
             f"{quantity} exceeds")
    secret = f" {val} {unit}"
    topic = f"relational {proto} {val} {unit}"
    return text, query, secret, topic

ARCHETYPES = [("code", mint_code), ("contradiction", mint_contradiction),
              ("relational", mint_relational)]

def sig_bits(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()  # 64 hex

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="total needles (split across 3 archetypes)")
    ap.add_argument("--out", required=True, help="output dir for needleNNN.txt + manifests")
    ap.add_argument("--epdir-root", default=None,
                    help="absolute Windows root where capture will write ep dirs "
                         "(for registry 'dir' field). Default: <out>\\eps")
    ap.add_argument("--seed-tag", default="batch", help="label embedded in needle ids")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ep_root = args.epdir_root or os.path.join(args.out, "eps")

    manifest = open(os.path.join(args.out, "corpus_manifest.jsonl"), "w", encoding="utf-8")
    registry = open(os.path.join(args.out, "registry.jsonl"), "w", encoding="utf-8")

    seen_secrets = set()
    seen_texts = set()
    n_emit = 0
    i = 0
    attempts = 0
    while n_emit < args.n and attempts < args.n * 50:
        attempts += 1
        arch_name, fn = ARCHETYPES[i % len(ARCHETYPES)]
        text, query, secret, topic = fn(args.seed_tag)

        # --- structural guards (the v10 lessons, enforced) ---
        sstrip = secret.strip()
        if text.count(sstrip) != 1:           # secret-once
            continue
        if not query.strip() and False:
            continue
        # query must be a strict prefix of the needle (so teacher-forcing the secret
        # continues the exact stated sentence)
        if not text.startswith(query):
            continue
        if secret in seen_secrets or text in seen_texts:   # dedupe
            continue
        seen_secrets.add(secret)
        seen_texts.add(text)

        nid = f"n_{args.seed_tag}_{n_emit:03d}"
        epname = f"ep_{nid}"
        # needle text file
        with open(os.path.join(args.out, f"{nid}.txt"), "w", encoding="utf-8") as f:
            f.write(text + "\n")
        sb = sig_bits(text)
        manifest.write(json.dumps({
            "id": nid, "archetype": arch_name, "text": text, "query": query,
            "secret": secret, "topic": topic, "sig_bits": sb,
        }) + "\n")
        registry.write(json.dumps({
            "name": epname,
            "dir": os.path.join(ep_root, epname),
            "npos": -1,                       # filled by capture step
            "topic": topic,
            "sig_bits": sb,
        }) + "\n")
        n_emit += 1
        i += 1

    manifest.close()
    registry.close()
    print(f"MINTED {n_emit} needles -> {args.out}")
    print(f"  manifest: {os.path.join(args.out, 'corpus_manifest.jsonl')}")
    print(f"  registry: {os.path.join(args.out, 'registry.jsonl')}")
    if n_emit < args.n:
        print(f"  WARN: only {n_emit}/{args.n} unique needles after {attempts} attempts",
              file=sys.stderr)

if __name__ == "__main__":
    main()
