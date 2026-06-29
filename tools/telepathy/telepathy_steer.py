#!/usr/bin/env python3
# telepathy_steer.py — the GENERATION-TRIGGER gate. Inject the mapped gemma->qwen latent into Qwen's
# residual stream (forward hook on a decoder layer) and measure whether it STEERS Qwen, with a matched
# control. Honest claim = "the right latent raises the matching text's likelihood more than a mismatched
# latent" (activation steering), NOT "forces verbatim output".
#   inputs: adapter npz (W_fwd,gmu,gsd,qmu,qsd), gemma_pairs.npy, pairs.txt
import argparse, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--adapter",default="telepathy_adapter_g2q.npz")
    ap.add_argument("--gemma",default="gemma_pairs.npy"); ap.add_argument("--texts",default="pairs.txt")
    ap.add_argument("--qwen",default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    ap.add_argument("--layers",default="8,12,16,20,22"); ap.add_argument("--alphas",default="0.1,0.25,0.5,1.0")
    a=ap.parse_args()
    dev="cuda" if torch.cuda.is_available() else "cpu"
    tok=AutoTokenizer.from_pretrained(a.qwen)
    model=AutoModelForCausalLM.from_pretrained(a.qwen,dtype=torch.bfloat16).to(dev).eval()
    nlayers=len(model.model.layers); print(f"[qwen] {nlayers} layers, hidden={model.config.hidden_size}")

    ad=np.load(a.adapter); Wf=ad["W_fwd"]; gmu,gsd,qmu,qsd=ad["gmu"],ad["gsd"],ad["qmu"],ad["qsd"]
    G=np.load(a.gemma).astype(np.float32); texts=[l.rstrip("\n") for l in open(a.texts,encoding="utf-8") if l.strip()]
    # map gemma -> qwen RAW hidden space:  Qz = ((G-gmu)/gsd)@Wf ;  raw = Qz*qsd+qmu
    Vraw = (((G-gmu)/gsd) @ Wf) * qsd + qmu                  # [N, Dq]
    V = torch.tensor(Vraw,dtype=torch.bfloat16,device=dev)
    # held-out split = same as fit_adapter (RandomState(0), test = last 20%)
    rng=np.random.RandomState(0); idx=rng.permutation(len(G)); test=idx[int(len(G)*0.8):]
    calib=test[:len(test)//2]; final=test[len(test)//2:]

    cur={"v":None,"alpha":0.0,"L":-1}
    def hook(module,inp,out):
        h=out[0] if isinstance(out,tuple) else out
        if cur["v"] is None: return out
        rms=h.norm(dim=-1,keepdim=True).mean()
        add=cur["alpha"]*rms*(cur["v"]/ (cur["v"].norm()+1e-6))
        h=h+add.to(h.dtype)
        return (h,)+tuple(out[1:]) if isinstance(out,tuple) else h
    handles=[model.model.layers[L].register_forward_hook(hook) for L in range(nlayers)]
    # we gate the hook by cur["L"]: only active layer applies
    def real_hook_factory(Lidx):
        def hk(module,inp,out):
            if cur["L"]!=Lidx or cur["v"] is None: return out
            h=out[0] if isinstance(out,tuple) else out
            rms=h.norm(dim=-1,keepdim=True).mean()
            add=cur["alpha"]*rms*(cur["v"]/(cur["v"].norm()+1e-6))
            h=h+add.to(h.dtype)
            return (h,)+tuple(out[1:]) if isinstance(out,tuple) else h
        return hk
    for hd in handles: hd.remove()
    handles=[model.model.layers[L].register_forward_hook(real_hook_factory(L)) for L in range(nlayers)]

    def ll(text,v=None,alpha=0.0,L=-1):
        cur["v"],cur["alpha"],cur["L"]=v,alpha,L
        enc=tok(text,return_tensors="pt").to(dev)
        with torch.no_grad(): o=model(**enc,labels=enc.input_ids)
        cur["v"]=None
        return -o.loss.item()

    def evalset(ids,L,alpha):
        ds=dc=win=pos=0; n=len(ids)
        for k,i in enumerate(ids):
            j=ids[(k+1)%n]                       # matched control = next text's latent
            l0=ll(texts[i]); ls=ll(texts[i],V[i],alpha,L); lc=ll(texts[i],V[j],alpha,L)
            ds+=ls-l0; dc+=lc-l0; win+= (ls>lc); pos+= ((ls-l0)>0)
        return ds/n, dc/n, win/n, pos/n

    print("\n[sweep] (layer,alpha) on calib — select: prefer dLL_self>0, then steer_acc:")
    best=(-1,-1.0); best_key=(-1,-1e9,-1e9)
    for L in [int(x) for x in a.layers.split(",")]:
        for al in [float(x) for x in a.alphas.split(",")]:
            d_s,d_c,wn,ps=evalset(list(calib),L,al)
            print(f"  L={L:2d} a={al:>5}: dLL_self={d_s:+.3f} dLL_cross={d_c:+.3f} steer_acc={wn:.3f} self+={ps:.3f}")
            key=(1 if d_s>0 else 0, wn, d_s)            # prefer positive dLL_self, then steering accuracy
            if key>best_key: best_key=key; best=(L,al)
    L,al=best; print(f"\n[best] layer={L} alpha={al}")
    d_s,d_c,wn,ps=evalset(list(final),L,al)
    print(f"\n==== GENERATION-TRIGGER gate (held-out final, L={L} a={al}) ====")
    print(f"dLL_self (matching latent vs none)  = {d_s:+.3f}   (want >0)")
    print(f"dLL_cross(mismatched latent vs none)= {d_c:+.3f}")
    print(f"STEER ACCURACY (self beats control) = {wn:.3f}   (chance 0.5)")
    print(f"fraction where matching latent raises LL = {ps:.3f}")
    print(f"VERDICT {'GREEN' if wn>0.75 and d_s>0 and d_s>d_c else 'AMBER' if wn>0.6 else 'RED'}")

    # qualitative: inject v_i, greedy-continue from a neutral prompt
    print("\n[demo] inject mapped latent, greedy continuation from neutral prompt:")
    for i in list(final)[:4]:
        cur["v"],cur["alpha"],cur["L"]=V[i],al,L
        enc=tok("The topic is:",return_tensors="pt").to(dev)
        with torch.no_grad(): g=model.generate(**enc,max_new_tokens=12,do_sample=False)
        cur["v"]=None
        cont=tok.decode(g[0][enc.input_ids.shape[1]:],skip_special_tokens=True).replace("\n"," ")
        print(f"  target='{texts[i][:46]}' -> qwen: '{cont[:60]}'")

if __name__=="__main__":
    main()
