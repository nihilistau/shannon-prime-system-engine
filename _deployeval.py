import numpy as np
wc=r"D:\F\shannon-prime-repos\shannon-prime-system-engine\_b3_wc"
z=np.load(wc+r"\lsh_Wc_f32_div2.npz",allow_pickle=True)
Wf=z["Wc"].astype(np.float32); r=int(z["r"]); scale=float(z["scale"]); s0=float(z["s0"])
S=2**14; Wi=np.clip(np.round(Wf*S),-32768,32767).astype(np.int16).astype(np.float32)/S
d=np.load(wc+r"\b3_data_div.npz",allow_pickle=True)
Q=[np.asarray(q,np.float32) for q in d["Q"]]; K=[np.asarray(k,np.float32) for k in d["K"]]
lab=d["labels"].astype(np.int64); E=len(K); ng=min(Q[0].shape[0],min(k.shape[0] for k in K))
def lse_mean(Wc):
    out=np.zeros((len(Q),E),np.float32)
    Kp=[np.einsum("lpd,dr->lpr",K[e][:ng],Wc) for e in range(E)]
    for i,q in enumerate(Q):
        qp=np.einsum("lhd,dr->lhr",q[:ng],Wc)
        for e in range(E):
            sim=np.einsum("lhr,lpr->lhp",qp,Kp[e])*scale
            a=np.log(np.exp(sim-sim.max(2,keepdims=True)).sum(2))+sim.max(2)
            out[i,e]=a.mean()
    return out
for tag,Wc in [("f32",Wf),("int16",Wi)]:
    Sm=lse_mean(Wc)
    aug=np.concatenate([Sm, np.full((len(Q),1), s0, np.float32)], axis=1)  # [Nq, E+1], col E = NULL
    arg=aug.argmax(1)
    pos=[i for i in range(len(Q)) if lab[i]>=0]; fo=[i for i in range(len(Q)) if lab[i]<0]
    pos_ok=sum(1 for i in pos if arg[i]==lab[i])           # true episode beats all + NULL
    fo_ok =sum(1 for i in fo  if arg[i]==E)                # NULL beats all episodes
    print(f"{tag} (E+1)-argmax deploy gate: positives recall {pos_ok}/{len(pos)} | foreign reject {fo_ok}/{len(fo)}  (s0={s0:+.3f})")
