#!/usr/bin/env python3
# C2 Build Step 3 — the Memo curator ONLINE LOOP host state machine (over the one-shot engine; Option A).
#
# The curator is a HOST orchestrator composing engine seams that are each proven bit-exact-when-off:
#   CUE   : SP_ARM_DUMP (read-only post-RoPE global K observer) -> host r=256 projection -> 256-bit hash
#   RESOLVE: discrete_resolve.resolve_cue (integer Hamming radius over registry_bits.jsonl)  [Step 2]
#   PROPOSE: re-run decode with SP_REPLAY=<episode dir> active                                  [P3.3]
#   GATE   : SP_G4_SCORE deflection vs baseline < 2.0%                                          [P3.4]
#   ACCEPT/REJECT: <2% keep ; >=2% discard (one-shot reject = discard-and-rerun, O(context);
#                  the O(1) gemma4_kv_rewind port is the named follow-on, NOT this gate).
#
# gemma4_decode_cuda stays BYTE-UNTOUCHED — the curator adds zero hot-path code (the P3.4 lesson).
#
# Step 3.0 (this file's --null mode) proves the orchestrator is INERT WHEN OFF:
#   * the cue-extraction seam (SP_ARM_DUMP) does NOT perturb the decode  -> PPL_base == PPL_dump (bit-exact)
#   * the OFF resolve (empty registry) returns NULL                       -> no SP_REPLAY, no action
#   * the cue extraction actually executed                                -> dump produced >=1 record
import os, sys, json, re, struct, argparse
import numpy as np

# ── reuse the Step-2 discrete projection / resolver verbatim ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from discrete_resolve import build_R, gl, to_bits, packhex, agree, resolve_cue, R_BITS, HD, NL

def parse_ppl(logpath):
    """Pull the PPL scalar the score harness prints. Returns the highest-precision match found."""
    txt = open(logpath, "r", errors="replace").read()
    # accept 'PPL = 4.6665', 'ppl=4.6665', 'perplexity 4.6665', 'NLL ... ppl 4.6665'
    cands = re.findall(r"(?:PPL|ppl|perplexity)\s*[=:]?\s*([0-9]+\.[0-9]+)", txt)
    if not cands:
        cands = re.findall(r"\bppl\b[^0-9]*([0-9]+\.[0-9]+)", txt, re.I)
    return cands[-1] if cands else None   # keep as STRING for exact byte-compare

def load_registry_bits(regpath):
    reg = []
    if regpath and os.path.exists(regpath):
        for line in open(regpath):
            line = line.strip()
            if not line: continue
            row = json.loads(line)
            x = int(row["sig_bits"], 16)
            row["_bits"] = np.array([(x >> i) & 1 for i in range(row.get("r_bits", R_BITS))], dtype=bool)
            reg.append(row)
    return reg

def count_dump_records(dumpdir):
    """SP_ARM_DUMP writes a stream of {int32 hdr, f32 K[kvd], f32 q[qd]} per global step.
    For the NULL we only need to prove the observer FIRED: count non-empty dump files / bytes."""
    if not dumpdir or not os.path.isdir(dumpdir): return 0, 0
    files = [os.path.join(dumpdir, f) for f in os.listdir(dumpdir) if os.path.isfile(os.path.join(dumpdir, f))]
    total = sum(os.path.getsize(f) for f in files)
    return len(files), total

def null_gate(base_log, dump_log, dumpdir, registry):
    ppl_base = parse_ppl(base_log); ppl_dump = parse_ppl(dump_log)
    nfiles, nbytes = count_dump_records(dumpdir)
    reg = load_registry_bits(registry)
    # OFF resolve: with an empty registry the resolver returns NULL by construction
    rid, rname, score = (None, None, -1)
    if reg:
        # (only reached if a registry was passed; the NULL leg uses an empty/absent one)
        dummy = np.zeros(R_BITS, dtype=bool)
        rid, rname, score = resolve_cue(dummy, reg)

    print(f"[null] PPL baseline       = {ppl_base}")
    print(f"[null] PPL cue-extract on = {ppl_dump}")
    print(f"[null] dump observer fired: {nfiles} file(s), {nbytes} bytes")
    print(f"[null] OFF resolve (registry={'empty' if not reg else len(reg)}) -> {'NULL' if rid is None else rname}")

    inert  = (ppl_base is not None and ppl_dump is not None and ppl_base == ppl_dump)
    fired  = (nbytes > 0)
    isnull = (rid is None)
    ok = inert and fired and isnull
    print(f"[null]   bit-exact-when-off (PPL identical): {'PASS' if inert else 'FAIL'}")
    print(f"[null]   cue extraction live (dump fired)  : {'PASS' if fired else 'FAIL'}")
    print(f"[null]   OFF resolve is NULL               : {'PASS' if isnull else 'FAIL'}")
    print(f"\n[gate] G-MEMO-NULL {'GREEN — orchestrator perfectly inert when off' if ok else 'RED'}")
    return 0 if ok else 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--null", action="store_true", help="Step 3.0 G-MEMO-NULL verdict")
    ap.add_argument("--base", help="baseline PPL log")
    ap.add_argument("--dump", help="cue-extraction-on PPL log")
    ap.add_argument("--dumpdir", help="SP_ARM_DUMP directory")
    ap.add_argument("--registry", default="", help="registry_bits.jsonl (EMPTY/absent for the null)")
    a = ap.parse_args()
    if a.null:
        return null_gate(a.base, a.dump, a.dumpdir, a.registry)
    print("loop mode (Step 3.1) not yet wired"); return 2

if __name__ == "__main__":
    sys.exit(main())
