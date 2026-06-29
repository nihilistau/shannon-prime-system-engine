#!/usr/bin/env python3
# sp_eagle_train.py -- EAGLE finetune flywheel (#2): finetune the gemma4-assistant draft to predict
# OUR engine's greedy tokens from OUR features + KV, closing the train/serve gap (the draft was
# distilled vs FP gemma-4-12b-it; our target is OK_Q4B + the exact-integer engine).
#
# Differentiable torch port of the proven draft forward (sp_eagle_ref.py / G-EAGLE-DRAFT-FWD-CUDA):
#   xh = concat([x, feat]) -> pre_proj -> 4 sandwich blocks (Q-only attn over the CAPTURED target KV,
#   per-head q_norm, GeGLU, rope[full=rope_freqs/g_base, swa=plain/s_base], out_scale) -> output_norm
#   -> draft tied head -> CE vs the captured target label. Trains pre/post/4-layers/out_norm; head frozen.
#
# Data = tools/eagle/_eagle_data (SP_EAGLE_CAPTURE): per seq feat/x[gen x 3840], inp/lbl/att[gen],
# kg/vg[npos x 512], ks/vs[npos x 2048] + manifest.jsonl (geometry). Run on GPU (Colab/RunPod).
#
# Usage: python sp_eagle_train.py --data _eagle_data --draft <draft.gguf> --out <draft_ft.gguf>
#                                 [--epochs 3 --lr 1e-4 --bs 4096 --device cuda]
import argparse, json, math, os, glob
import numpy as np
import torch, torch.nn.functional as F
from gguf import GGUFReader, GGUFWriter

NL, HID, NH = 4, 1024, 16
SWA = [True, True, True, False]
EPS = 1e-6

# the draft's per-layer trainable tensors (GGUF names) + globals.
LAYER_T = ["attn_norm","attn_q","attn_q_norm","attn_output","post_attention_norm",
           "ffn_norm","ffn_gate","ffn_up","ffn_down","post_ffw_norm","layer_output_scale"]
GLOBAL_T = ["nextn.pre_projection","nextn.post_projection","output_norm"]
FROZEN = ["token_embd"]  # the tied head — frozen (finetune the body)

def gname(il, sub): return f"blk.{il}.{sub}.weight"

def load_draft(path, device):
    r = GGUFReader(path)
    raw = {t.name: np.array(t.data, dtype=np.float32) for t in r.tensors}
    P = {}
    def add(name, train):
        a = raw[name]
        t = torch.tensor(a, dtype=torch.float32, device=device)
        t.requires_grad_(train); P[name] = t
    for n in GLOBAL_T: add(n + ".weight" if not n.endswith("weight") else n, True)
    # rope_freqs optional
    if "rope_freqs.weight" in raw: add("rope_freqs.weight", False)
    add("token_embd.weight", False)            # tied head, frozen
    for il in range(NL):
        for s in LAYER_T: add(gname(il, s), True)
    return P, raw, r

def rms(x, w):  # x[...,d], w[d]; x*inv*w, inv=1/sqrt(mean(x^2)+eps)
    inv = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)
    return x * inv * w
def gelu_t(x): return 0.5*x*(1.0+torch.tanh(0.7978845608028654*(x+0.044715*x*x*x)))
def rope_all(q, pos, base, freqs=None):  # q[H,hd]; neox rotate (i,i+half), same pos all heads
    hd = q.shape[-1]; half = hd//2
    i = torch.arange(half, device=q.device, dtype=torch.float32)
    inv = base ** (-(2.0*i)/hd)
    if freqs is not None: inv = inv / freqs
    th = pos*inv; c, s = torch.cos(th), torch.sin(th)                 # [half], broadcast over heads
    a, b = q[..., :half], q[..., half:]
    return torch.cat([a*c - b*s, b*c + a*s], dim=-1)

def draft_forward(P, x, feat, kvg, kvs, attend, pos, g_base, s_base):
    # x,feat [BBt=3840]; kvg=(Kg[att,512],Vg), kvs=(Ks[att,2048],Vs); single query position.
    cur = torch.cat([x, feat]) @ P["nextn.pre_projection.weight"].T   # [HID]
    for il in range(NL):
        p = lambda s: P[gname(il, s)]
        n = rms(cur, p("attn_norm"))
        q = (n @ p("attn_q").T)                                       # [qd]
        hd = q.shape[0] // NH; q = q.view(NH, hd)
        q = rms(q, p("attn_q_norm"))
        swa = SWA[il]
        base = s_base if swa else g_base
        fr = None if swa else P.get("rope_freqs.weight")
        q = rope_all(q, float(pos), base, fr)                         # [NH, hd]
        K, V = (kvs if swa else kvg)
        K = K[:attend].view(attend, -1, hd); V = V[:attend].view(attend, -1, hd)  # [att, nkv, hd]
        nkv = K.shape[1]; group = NH // nkv
        asc = 1.0                                                     # f_attention_scale=1.0 (ascale=one)
        Kx = K.repeat_interleave(group, dim=1)                        # [att, NH, hd]  (GQA expand)
        Vx = V.repeat_interleave(group, dim=1)
        sc = torch.einsum('hd,ahd->ah', q, Kx) * asc                  # [att, NH]
        w = torch.softmax(sc, dim=0)
        ctx = torch.einsum('ah,ahd->hd', w, Vx)                      # [NH, hd]
        a = ctx.reshape(-1) @ p("attn_output").T
        a = rms(a, p("post_attention_norm"))
        attn_out = a + cur
        f = rms(attn_out, p("ffn_norm"))
        f = (gelu_t(f @ p("ffn_gate").T) * (f @ p("ffn_up").T)) @ p("ffn_down").T
        f = rms(f, p("post_ffw_norm"))
        cur = (f + attn_out) * p("layer_output_scale")[0]
    cur = rms(cur, P["output_norm.weight"])
    return cur                                                        # pre-head [HID]; head applied batched

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--draft", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_pos", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=128)   # head/backward batch (amortizes the 1GB tok_embd read)
    a = ap.parse_args()
    dev = a.device if torch.cuda.is_available() or a.device == "cpu" else "cpu"
    print(f"[train] device={dev}")
    meta = json.loads(open(f"{a.data}/manifest.jsonl").readline())
    g_base, s_base = meta["g_base"], meta["s_base"]; kvd_g, kvd_s = meta["g_nkv"]*meta["g_hd"], meta["s_nkv"]*meta["s_hd"]
    P, raw, reader = load_draft(a.draft, dev)
    trainable = [t for t in P.values() if t.requires_grad]
    print(f"[train] {sum(t.numel() for t in trainable)/1e6:.1f}M trainable params over {len(trainable)} tensors")
    opt = torch.optim.AdamW(trainable, lr=a.lr)

    # index all (seq, position) examples
    seqs = sorted(glob.glob(f"{a.data}/seq_*"))
    examples = []
    for sd in seqs:
        try: gen = np.fromfile(f"{sd}/lbl.i32", dtype=np.int32).shape[0]
        except Exception: continue
        for j in range(gen): examples.append((sd, j))
    examples = examples[:a.max_pos]
    print(f"[train] {len(seqs)} seqs, {len(examples)} (pos) examples")

    cache = {}
    def load_seq(sd):
        if sd in cache: return cache[sd]
        L = lambda n,d: torch.tensor(np.fromfile(f"{sd}/{n}", dtype=d), device=dev)
        gen = np.fromfile(f"{sd}/lbl.i32", dtype=np.int32).shape[0]
        npos = np.fromfile(f"{sd}/kg.f32", dtype=np.float32).shape[0] // kvd_g
        d = dict(feat=L("feat.f32",np.float32).view(gen,-1), x=L("x.f32",np.float32).view(gen,-1),
                 lbl=L("lbl.i32",np.int32).long(), att_cpu=np.fromfile(f"{sd}/att.i32", dtype=np.int32),
                 kg=L("kg.f32",np.float32).view(npos,kvd_g), vg=L("vg.f32",np.float32).view(npos,kvd_g),
                 ks=L("ks.f32",np.float32).view(npos,kvd_s), vs=L("vs.f32",np.float32).view(npos,kvd_s))
        if len(cache) < 400: cache[sd] = d   # cache all ~310 seqs resident (avoids per-epoch disk reload)
        return d
    head = P["token_embd.weight"].T                                  # [HID, Vd], frozen
    for ep in range(a.epochs):
        np.random.shuffle(examples)
        tot = 0; lsum = torch.zeros((), device=dev); hits = torch.zeros((), device=dev)
        for bstart in range(0, len(examples), a.bs):
            batch = examples[bstart:bstart + a.bs]
            curs, lbls = [], []
            for sd, j in batch:
                d = load_seq(sd)
                att = int(d["att_cpu"][j])                           # CPU read -> no per-example GPU sync
                if att < 1: continue
                curs.append(draft_forward(P, d["x"][j], d["feat"][j], (d["kg"],d["vg"]), (d["ks"],d["vs"]),
                                          att, att, g_base, s_base))
                lbls.append(d["lbl"][j])
            if not curs: continue
            C = torch.stack(curs); L = torch.stack(lbls)             # [b,HID], [b]
            logits = C @ head                                        # [b,Vd] -- ONE big matmul (head read amortized)
            loss = F.cross_entropy(logits, L)
            opt.zero_grad(); loss.backward(); opt.step()
            n = len(curs); lsum += loss.detach()*n; hits += (logits.argmax(1) == L).float().sum(); tot += n
            if (bstart // a.bs) % 8 == 0:
                print(f"  ep{ep} {tot}/{len(examples)} loss={lsum.item()/tot:.3f} train_acc={hits.item()/tot:.3f}", flush=True)
        print(f"[train] epoch {ep}: loss={lsum.item()/max(tot,1):.3f} train_acc={hits.item()/max(tot,1):.3f}", flush=True)

    # export: copy the draft GGUF, overwrite the trained tensors IN PLACE (dtype preserved -> metadata
    # byte-identical -> sp_transcode reads it unchanged). F16/F32 only (norms F32, weights F16).
    import shutil
    shutil.copy(a.draft, a.out)
    GGML_F32, GGML_F16 = 0, 1
    rd = GGUFReader(a.out)
    patched, checks = 0, []
    with open(a.out, "r+b") as fh:
        for t in rd.tensors:
            nm = t.name
            if not (nm in P and P[nm].requires_grad): continue
            val = P[nm].detach().cpu().numpy().reshape(-1)
            tt = int(t.tensor_type)
            if tt == GGML_F16: buf = val.astype(np.float16)
            elif tt == GGML_F32: buf = val.astype(np.float32)
            else: raise RuntimeError(f"{nm}: ggml_type {tt} not F16/F32 (cannot in-place patch)")
            b = buf.tobytes()
            exp = int(np.prod(t.shape)) * (2 if tt == GGML_F16 else 4)
            if len(b) != exp: raise RuntimeError(f"{nm}: byte mismatch {len(b)} vs {exp}")
            off = int(t.data_offset)
            fh.seek(off); fh.write(b); patched += 1
            checks.append((nm, off, tt, float(buf.reshape(-1)[0])))
    # readback validation: confirm data_offset wrote where we think.
    rv = GGUFReader(a.out)
    tv = {t.name: t for t in rv.tensors}
    ok = True
    for nm, off, tt, v0 in checks[:4]:
        got = float(np.array(tv[nm].data).reshape(-1)[0])
        if abs(got - v0) > (1e-2 if tt == GGML_F16 else 1e-5):
            print(f"[train] EXPORT READBACK MISMATCH {nm}: wrote {v0} read {got}"); ok = False
    print(f"[train] patched {patched} tensors in place -> {a.out} (readback {'OK' if ok else 'FAILED'})")

if __name__ == "__main__":
    main()
