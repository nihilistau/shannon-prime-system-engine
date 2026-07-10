"""memory_doctor.py — production memory registry hygiene (AUDIT 2026-07-10).

The daemon's load_registry() silently tolerates registry rows whose episode dirs
were deleted (ep.k/ep.l5 load as empty -> the L5 selector skips them with NO
warning), and episode dirs captured while SP_NIGHTSHIFT_PERSIST was unset are
ORPHANED (on disk, in no registry, never rediscovered — the daemon does not scan
_nightshift_live/). This tool makes both states visible and fixable.

Commands (default DRY-RUN; add --apply to write):
  audit                    report LIVE/DEAD registry rows + orphaned episode dirs
  prune-dead [--apply]     drop DEAD rows (missing dir or missing/short ep.l5);
                           dumps their texts to <registry>.remint.txt for re-minting
  adopt --match SUBSTR [--apply]
                           append registry rows for orphan dirs whose ep.txt
                           contains SUBSTR (npos from ep.tok, text attributed,
                           sig_bits dummy — L5 recall does not use sig_bits,
                           precedent: _faithful_corpus/sne/build_sne_registry.py)
  remint --file F [--url U]
                           POST each line of F to the live daemon as an explicit
                           store verb ("remember that ...") — re-captures with the
                           CURRENT mint config (question-space keys if SP_QKEY_MINT=1)
                           and persists if the serve has SP_NIGHTSHIFT_PERSIST=1.

A registry backup <registry>.bak.<UTC-stamp> is written before any --apply.
"""
import argparse, datetime, json, os, shutil, sys, urllib.request

ENG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # engine root
DEF_REG = os.path.join(ENG, "_memory_live", "registry.jsonl")
DEF_NS = os.path.join(ENG, "_nightshift_live")
L5_BYTES = 512 * 4


def load_rows(reg):
    rows = []
    if os.path.exists(reg):
        for ln in open(reg, encoding="utf-8", errors="replace"):
            ln = ln.strip()
            if not ln:
                continue
            try:
                rows.append(json.loads(ln))
            except Exception:
                rows.append({"_malformed": ln})
    return rows


def row_state(r):
    d = r.get("dir", "")
    if not d or not os.path.isdir(d):
        return "DEAD"
    l5 = os.path.join(d, "ep.l5")
    k = os.path.join(d, "ep.k")
    if not (os.path.isfile(l5) and os.path.getsize(l5) == L5_BYTES):
        return "DEAD"
    if not (os.path.isfile(k) and os.path.getsize(k) > 0):
        return "DEAD"
    return "LIVE"


def orphan_dirs(ns, rows):
    reg_dirs = {os.path.normcase(os.path.normpath(r.get("dir", ""))) for r in rows}
    out = []
    if os.path.isdir(ns):
        for name in sorted(os.listdir(ns)):
            d = os.path.join(ns, name)
            if os.path.isdir(d) and os.path.normcase(os.path.normpath(d)) not in reg_dirs:
                out.append(d)
    return out


def ep_text(d):
    p = os.path.join(d, "ep.txt")
    return open(p, encoding="utf-8", errors="replace").read().strip() if os.path.isfile(p) else ""


def backup(reg):
    stamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    dst = f"{reg}.bak.{stamp}"
    shutil.copy2(reg, dst)
    print(f"backup -> {dst}")


def cmd_audit(a):
    rows = load_rows(a.registry)
    dead = live = 0
    for r in rows:
        st = "MALFORMED" if "_malformed" in r else row_state(r)
        if st == "LIVE":
            live += 1
        else:
            dead += 1
        txt = (r.get("text", "") or "")[:60].replace("\n", " ")
        print(f"  [{st}] lc={r.get('lifecycle',0)} {r.get('name','?')}  '{txt}'")
    orph = orphan_dirs(a.nightshift, rows)
    print(f"registry: {live} LIVE / {dead} DEAD of {len(rows)}   orphans: {len(orph)}")
    for d in orph:
        print(f"  [ORPHAN] {os.path.basename(d)}  '{ep_text(d)[:60]}'")
    return 0


def cmd_prune(a):
    rows = load_rows(a.registry)
    keep, drop = [], []
    for r in rows:
        (drop if ("_malformed" in r or row_state(r) == "DEAD") else keep).append(r)
    print(f"would keep {len(keep)}, drop {len(drop)}:")
    for r in drop:
        print(f"  DROP {r.get('name','(malformed)')}  '{(r.get('text','') or '')[:60]}'")
    remint = [r.get("text", "") for r in drop if r.get("text")]
    if not a.apply:
        print("(dry-run — add --apply)")
        return 0
    backup(a.registry)
    with open(a.registry, "w", encoding="utf-8") as f:
        for r in keep:
            f.write(json.dumps(r) + "\n")
    rf = a.registry + ".remint.txt"
    with open(rf, "w", encoding="utf-8") as f:
        for t in remint:
            f.write(t + "\n")
    print(f"pruned {len(drop)} rows; remint candidates -> {rf}")
    return 0


def cmd_adopt(a):
    rows = load_rows(a.registry)
    todo = []
    for d in orphan_dirs(a.nightshift, rows):
        txt = ep_text(d)
        if a.match and a.match.lower() not in txt.lower():
            continue
        l5 = os.path.join(d, "ep.l5")
        if not (os.path.isfile(l5) and os.path.getsize(l5) == L5_BYTES):
            print(f"  SKIP (no/short ep.l5): {os.path.basename(d)}")
            continue
        tokf = os.path.join(d, "ep.tok")
        npos = os.path.getsize(tokf) // 4 if os.path.isfile(tokf) else max(6, len(txt.split()) + 4)
        text = txt if txt.lower().startswith("the user said") else f"The user said: {txt}"
        todo.append({"name": os.path.basename(d), "dir": d.replace("\\", "/"),
                     "npos": int(npos), "topic": txt[:40], "text": text,
                     "sig_bits": "0" * 64})
    for r in todo:
        print(f"  ADOPT {r['name']} npos={r['npos']}  '{r['text'][:60]}'")
    if not todo:
        print("nothing to adopt (check --match)")
        return 0
    if not a.apply:
        print("(dry-run — add --apply)")
        return 0
    backup(a.registry)
    with open(a.registry, "a", encoding="utf-8") as f:
        for r in todo:
            f.write(json.dumps(r) + "\n")
    print(f"adopted {len(todo)} orphan(s) into {a.registry}")
    return 0


def cmd_remint(a):
    lines = [ln.strip() for ln in open(a.file, encoding="utf-8") if ln.strip()]
    ok = 0
    for t in lines:
        # strip prior attribution; the store verb re-attributes.
        fact = t[len("The user said:"):].strip() if t.lower().startswith("the user said:") else t
        body = json.dumps({"messages": [{"role": "user", "content": f"remember that {fact}"}],
                           "max_tokens": 32, "temperature": 0}).encode()
        req = urllib.request.Request(a.url, data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                resp.read()
            ok += 1
            print(f"  STORED: {fact[:60]}")
        except Exception as e:
            print(f"  FAIL ({e}): {fact[:60]}")
    print(f"reminted {ok}/{len(lines)} (daemon must serve with SP_MEM_STORE=1 + SP_NIGHTSHIFT_PERSIST=1)")
    return 0 if ok == len(lines) else 1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--registry", default=DEF_REG)
    p.add_argument("--nightshift", default=DEF_NS)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("audit")
    sp = sub.add_parser("prune-dead"); sp.add_argument("--apply", action="store_true")
    sa = sub.add_parser("adopt"); sa.add_argument("--match", default=""); sa.add_argument("--apply", action="store_true")
    sr = sub.add_parser("remint"); sr.add_argument("--file", required=True); sr.add_argument("--url", default="http://127.0.0.1:3000/v1/chat")
    a = p.parse_args()
    return {"audit": cmd_audit, "prune-dead": cmd_prune, "adopt": cmd_adopt, "remint": cmd_remint}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
