import numpy as np, torch, os, random, statistics as st
HD=512
eng=r"D:\F\shannon-prime-repos\shannon-prime-system-engine"
d=np.load(os.path.join(eng,"_b3_wc","b3_data_div.npz"),allow_pickle=True)
w=np.load(os.path.join(eng,"_b3_wc","lsh_Wc_f32_holdout.npz"),allow_pickle=True)
dev="cuda" if torch.cuda.is_available() else "cpu"
Wc=torch.tensor(w["Wc"],device=dev); s0=float(w["s0"]); scale=float(w["scale"])
hold=[int(x) for x in w["holdout_eps"]]; holdset=set(hold)
lab=d["labels"].astype(np.int64)
Qs=torch.tensor(np.stack([np.asarray(q,np.float32) for q in d["Q"]]),device=dev)
Nq,ng=Qs.shape[0],Qs.shape[1]
Ks=[np.asarray(k,np.float32) for k in d["K"]]; E=len(Ks)
ng=min(ng,min(int(k.shape[0]) for k in Ks)); Qs=Qs[:,:ng].contiguous()
npos=[int(k.shape[1]) for k in Ks]; Pmax=max(npos)
Kpad=np.zeros((E,ng,Pmax,HD),np.float32); Km=np.zeros((E,Pmax),np.float32)
for e,k in enumerate(Ks):
    p=npos[e]; Kpad[e,:,:p,:]=k[:ng,:p,:]; Km[e,:p]=1.0
Kpad=torch.tensor(Kpad,device=dev); neg=torch.tensor((1.0-Km)*(-1e30),device=dev)
with torch.no_grad():
    qp=torch.einsum("qlhd,dr->qlhr",Qs,Wc); outs=[]
    for c0 in range(0,E,8):
        c1=min(E,c0+8)
        kp=torch.einsum("elpd,dr->elpr",Kpad[c0:c1],Wc)
        sim=torch.einsum("qlhr,elpr->qelhp",qp,kp)*scale
        sim=sim+neg[c0:c1].view(1,c1-c0,1,1,Pmax)
        outs.append(torch.logsumexp(sim,4).mean(dim=(2,3)).cpu().numpy())
    S=np.concatenate(outs,1)
matched=[i for i in range(Nq) if lab[i]>=0 and lab[i] in holdset]
foreign=[i for i in range(Nq) if lab[i]<0]
print(f"held-out eps={len(hold)} matched-q={len(matched)} foreign-q={len(foreign)} s0={s0:.4f}")
for K in [4,8]:
    rec=[]; rej=[]
    for seed in range(50):
        rng=random.Random(seed); ok=0
        for i in matched:
            t=lab[i]; pool=[e for e in hold if e!=t]
            cand=[t]+rng.sample(pool,min(K-1,len(pool)))
            sc=[float(S[i][c]) for c in cand]+[s0]
            if int(np.argmax(sc))==0: ok+=1
        rec.append(100.0*ok/max(1,len(matched)))
        rok=0
        for i in foreign:
            cand=rng.sample(hold,min(K,len(hold)))
            sc=[float(S[i][c]) for c in cand]+[s0]
            if int(np.argmax(sc))==len(cand): rok+=1
        rej.append(100.0*rok/max(1,len(foreign)))
    print(f"K={K}: W_c OOD bounded recall@1 = {st.mean(rec):.1f}%  (sd {st.pstdev(rec):.1f}); foreign-reject = {st.mean(rej):.1f}%")