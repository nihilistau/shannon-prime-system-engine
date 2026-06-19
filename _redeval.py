import numpy as np, os
eng=r"D:\F\shannon-prime-repos\shannon-prime-system-engine"; wc=eng+r"\_b3_wc"
z=np.load(wc+r"\lsh_Wc_f32_div.npz", allow_pickle=True)   # the logsumexp-trained head (361/361 float)
Wf=z["Wc"].astype(np.float32); r=int(z["r"]); scale=float(z["scale"]); s0=float(z["s0"])
# int16 quantize (scale 2^14, matches export)
S=2**14; Wi=np.clip(np.round(Wf*S),-32768,32767).astype(np.int16).astype(np.float32)/S
d=np.load(wc+r"\b3_data_div.npz", allow_pickle=True)
Q=[np.asarray(q,np.float32) for q in d["Q"]]; K=[np.asarray(k,np.float32) for k in d["K"]]
lab=d["labels"].astype(np.int64); E=len(K); ng=min(Q[0].shape[0], min(k.shape[0] for k in K))
def scoremat(Wc, red):
    # returns [Nq,E]
    out=np.zeros((len(Q),E),np.float32)
    Kp=[np.einsum("lpd,dr->lpr", K[e][:ng], Wc) for e in range(E)]
    for i,q in enumerate(Q):
        qp=np.einsum("lhd,dr->lhr", q[:ng], Wc)
        for e in range(E):
            sim=np.einsum("lhr,lpr->lhp", qp, Kp[e])*scale     # [ng,GH,np]
            f=sim.reshape(-1)
            if red=="max": out[i,e]=f.max()
            elif red=="top8": out[i,e]=np.sort(f)[-8:].mean()
            elif red=="lse": 
                m=f.max(); out[i,e]=m+np.log(np.exp(f-m).sum())  # not normalized; rank-equiv to lse-mean? use per (l,h) lse_p then mean
    return out
def lse_mean(Wc):
    out=np.zeros((len(Q),E),np.float32)
    Kp=[np.einsum("lpd,dr->lpr", K[e][:ng], Wc) for e in range(E)]
    for i,q in enumerate(Q):
        qp=np.einsum("lhd,dr->lhr", q[:ng], Wc)
        for e in range(E):
            sim=np.einsum("lhr,lpr->lhp", qp, Kp[e])*scale     # [ng,GH,np]
            a=np.log(np.exp(sim-sim.max(2,keepdims=True)).sum(2))+sim.max(2)  # [ng,GH] lse over p
            out[i,e]=a.mean()
    return out
def diag(M):
    return sum(1 for i in range(len(Q)) if lab[i]>=0 and int(M[i].argmax())==lab[i]), int((lab>=0).sum())
for tag,Wc in [("f32",Wf),("int16",Wi)]:
    for red in ["max","top8"]:
        M=scoremat(Wc,red); ok,n=diag(M); print(f"{tag} {red}: diagonal {ok}/{n}")
    M=lse_mean(Wc); ok,n=diag(M); print(f"{tag} lse_mean: diagonal {ok}/{n}")
