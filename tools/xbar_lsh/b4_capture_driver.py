#!/usr/bin/env python3
"""b4_capture_driver.py -- ONE-model-load batch capturer for the B4 scale-up.

POSTs every needle's text to the resident daemon's /v1/capture endpoint, which runs a
single curated forward on the LOADED model (kv::capture_batched) and writes ep.k/ep.v/ep.mf
into out_dir. The daemon must be running (any launcher) with the wire_cuda backend.

For each registry row it captures into row['dir'], records the returned npos back into a
patched registry (registry_npos.jsonl) so b3_make_dataset.py reads the true npos. Resume-safe:
skips a needle if its ep.mf already exists.
"""
import argparse, json, os, sys, time, urllib.request

def post_capture(base, text, out_dir, timeout=180):
    body=json.dumps({"text":text,"out_dir":out_dir}).encode()
    req=urllib.request.Request(base+"/v1/capture", data=body,
                               headers={"Content-Type":"application/json"})
    r=urllib.request.urlopen(req, timeout=timeout).read()
    return json.loads(r)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--out-registry", required=True, help="registry.jsonl with real npos filled in")
    ap.add_argument("--base", default="http://127.0.0.1:3000")
    args=ap.parse_args()
    man={}
    for ln in open(args.manifest, encoding="utf-8"):
        if ln.strip():
            r=json.loads(ln); man[f"ep_{r['id']}"]=r["text"]
    rows=[json.loads(l) for l in open(args.registry, encoding="utf-8") if l.strip()]
    out=open(args.out_registry,"w",encoding="utf-8")
    t0=time.time(); ok=0; fail=0; skip=0
    for i,row in enumerate(rows):
        name=row["name"]; d=os.path.abspath(row["dir"]); text=man.get(name)
        if text is None:
            print(f"[cap] {name}: NO manifest text -> skip", flush=True); fail+=1; continue
        mf=os.path.join(d,"ep.mf")
        if os.path.exists(mf) and row.get("npos",-1)>0:
            out.write(json.dumps(row)+"\n"); skip+=1; continue
        try:
            res=post_capture(args.base, text, d)
        except Exception as e:
            print(f"[cap] {name}: POST FAIL {e}", flush=True); fail+=1
            out.write(json.dumps(row)+"\n"); continue
        if res.get("ok"):
            npos=int(res["npos"]); row["npos"]=npos; ok+=1
            if i%25==0 or i<3:
                el=time.time()-t0
                print(f"[cap] {i+1}/{len(rows)} {name} npos={npos}  ({el:.0f}s, {el/max(1,ok):.2f}s/ep)", flush=True)
        else:
            print(f"[cap] {name}: ERR {res.get('error')}", flush=True); fail+=1
        out.write(json.dumps(row)+"\n"); out.flush()
    out.close()
    el=time.time()-t0
    print(f"[cap] DONE ok={ok} skip={skip} fail={fail} in {el:.0f}s ({el/max(1,ok):.2f}s/ep)  -> {args.out_registry}", flush=True)

if __name__=="__main__":
    main()
