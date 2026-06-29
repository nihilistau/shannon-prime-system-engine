#!/usr/bin/env python3
# sp_eagle_ref.py -- REFERENCE ORACLE for the gemma4-assistant (EAGLE/MTP) draft forward.
#
# numpy reference of ONE draft step, grounded VERBATIM in llama.cpp PR #23398
# (src/models/gemma4-assistant.cpp) + matched to the engine's gemma4 conventions
# (core/forward/gemma4.c: g4_gelu tanh GeGLU, embscale=sqrt(E), out_scale x*=s,
# RMSNorm x*inv*g, sp_rope_neox). This is the gate the engine C/CUDA draft port
# (step 2c/2d) MUST match. With --dump <dir> it writes a deterministic fixture
# (inputs + expected outputs as raw little-endian f32) so the C reference
# (sp_eagle_fwd.c) gates against IDENTICAL data (no cross-language RNG mismatch).
#
# AUTHORITATIVE FORWARD (PR #23398):
#   x   = target_tok_embd[token] * sqrt(n_embd_out=3840)   (TARGET's 3840 embd, scaled)
#   xh  = concat([x, inp_h], dim0)                          (7680 = 2*3840; EMBED FIRST)
#   cur = nextn.pre_projection @ xh                          (-> 1024)
#   for il in 0..3 (sandwich norm, GeGLU, q_norm per head, RoPE, Q-only attn on TARGET KV):
#     n=rms(cur,attn_norm); Q=wq@n; reshape[16,hd]; Q=rms_head(Q,attn_q_norm); rope(Q)
#     a=attn(Q,K[il],V[il]); a=wo@a; a=rms(a,post_attention_norm); attn_out=a+cur
#     f=rms(attn_out,ffn_norm); f=ffn_down@(g4_gelu(ffn_gate@f)*(ffn_up@f))
#     f=rms(f,post_ffw_norm); cur=(f+attn_out)*layer_output_scale
#   cur=rms(cur,output_norm); logits=draft_tok_embd@cur (DRAFT-tied head); h_next=post_proj@cur
#
# SCOPE: inp_h, the target token-embedding x, and the target K/V are SYNTHETIC-deterministic
# (live: inp_h=llama_get_embeddings_nextn(target)=post-output_norm 3840 hidden=tap decode.c:864;
#  x=target_tok_embd[token]; K/V=12B stored rows for layers n-1 full / n-2 SWA; GQA = 2d).
import sys, os, math, numpy as np
from gguf import GGUFReader

GGUF = next((a for a in sys.argv[1:] if a.endswith(".gguf")),
            r"D:\Files\Models\Gemma4\gemma-4-it-mtp\gemma-4-12b-it-F16-MTP.gguf")
DUMP = (sys.argv[sys.argv.index("--dump") + 1] if "--dump" in sys.argv else None)
NL, HID, BB, NH, EPS = 4, 1024, 3840, 16, 1e-6
ROPE_BASE_FULL, ROPE_BASE_SWA = 1e6, 1e4
SWA_PATTERN = [True, True, True, False]        # [8,8,8,1] kv: 3 SWA (hd256) + 1 full (hd512)
P, POS = 12, 7                                  # synthetic KV positions / query position (shared w/ C)

def load():
    r = GGUFReader(GGUF)
    return {t.name: np.array(t.data, dtype=np.float32) for t in r.tensors}
def lin(w, x):
    assert w.shape[1] == x.shape[-1], f"dim {w.shape} @ {x.shape}"
    return w @ x
def rms(x, g):
    inv = 1.0 / math.sqrt(float((x * x).mean()) + EPS)
    return x * inv * g
def rms_head(Q, g):
    return np.stack([rms(Q[h], g) for h in range(Q.shape[0])], 0)
def g4_gelu(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
def rope_neox(v, pos, base):
    hd = v.shape[0]; half = hd // 2; i = np.arange(half)
    inv = base ** (-(2.0 * i) / hd); c, s = np.cos(pos * inv), np.sin(pos * inv)
    out = v.copy(); out[:half] = v[:half] * c - v[half:] * s; out[half:] = v[half:] * c + v[:half] * s
    return out

def block(T, il, cur, K, V, pos):
    p = f"blk.{il}."
    n = rms(cur, T[p + "attn_norm.weight"])
    Q = lin(T[p + "attn_q.weight"], n); hd = Q.shape[0] // NH; Q = Q.reshape(NH, hd)
    Q = rms_head(Q, T[p + "attn_q_norm.weight"])
    base = ROPE_BASE_SWA if SWA_PATTERN[il] else ROPE_BASE_FULL
    Q = np.stack([rope_neox(Q[h], pos, base) for h in range(NH)], 0)
    asc = 1.0 / math.sqrt(hd); ctx = np.empty((NH, hd), np.float32)
    for h in range(NH):
        sc = (K @ Q[h]) * asc; sc -= sc.max(); w = np.exp(sc); w /= w.sum(); ctx[h] = w @ V
    a = lin(T[p + "attn_output.weight"], ctx.reshape(-1))
    a = rms(a, T[p + "post_attention_norm.weight"]); attn_out = a + cur
    f = rms(attn_out, T[p + "ffn_norm.weight"])
    f = lin(T[p + "ffn_down.weight"], g4_gelu(lin(T[p + "ffn_gate.weight"], f)) * lin(T[p + "ffn_up.weight"], f))
    f = rms(f, T[p + "post_ffw_norm.weight"])
    return (f + attn_out) * float(T[p + "layer_output_scale.weight"].reshape(-1)[0]), hd

def gen_inputs(T):
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(BB).astype(np.float32) * 0.02 * math.sqrt(BB))   # target_emb[token]*sqrt(BB)
    inp_h = np.random.default_rng(99).standard_normal(BB).astype(np.float32) * 0.1
    kv = []
    for il in range(NL):
        hd = T[f"blk.{il}.attn_q.weight"].shape[0] // NH
        K = rng.standard_normal((P, hd)).astype(np.float32) * 0.1
        V = rng.standard_normal((P, hd)).astype(np.float32) * 0.1
        kv.append((K, V))
    return x, inp_h, kv

def forward(T, x, inp_h, kv, pos=POS):
    xh = np.concatenate([x, inp_h]).astype(np.float32)                  # EMBED FIRST (PR #23398)
    cur = lin(T["nextn.pre_projection.weight"], xh); dims = [("pre_proj", cur.shape[0])]
    for il in range(NL):
        cur, hd = block(T, il, cur, kv[il][0], kv[il][1], pos); dims.append((f"blk.{il}(hd={hd})", cur.shape[0]))
    cur = rms(cur, T["output_norm.weight"])
    logits = lin(T["token_embd.weight"], cur); h_next = lin(T["nextn.post_projection.weight"], cur)
    dims += [("output_norm", cur.shape[0]), ("logits", logits.shape[0]), ("h_next", h_next.shape[0])]
    return logits.astype(np.float32), h_next.astype(np.float32), cur.astype(np.float32), dims

def drive(T, x0, h0, kv, K):
    """K-step EAGLE recurrence: per step feed [x_k, h_k] (h_k = prev step's post_proj output);
    token_k = argmax(logits); h_{k+1} carries the recurrence (the draft owns no KV -- its memory
    of its own speculated tokens lives in h, not the cache). pos advances POS+k. Returns the K
    speculative tokens + the per-step embedding inputs x_k used (deterministic, dumped for the C gate).
    NOTE: x_k here is synthetic-deterministic; live path = TARGET tok_embd[token_k]*sqrt(3840)."""
    toks, xs, h = [], [], h0.copy()
    for k in range(K):
        xk = (np.random.default_rng(2000 + k).standard_normal(BB).astype(np.float32) * 0.02 * math.sqrt(BB)
              if k > 0 else x0)
        lg, h, _, _ = forward(T, xk, h, kv, pos=POS + k)
        toks.append(int(np.argmax(lg))); xs.append(xk)
    return toks, xs

def main():
    print(f"[load] {GGUF}"); T = load()
    need = (["nextn.pre_projection.weight", "nextn.post_projection.weight", "token_embd.weight", "output_norm.weight"]
            + [f"blk.{il}.{s}.weight" for il in range(NL)
               for s in ("attn_norm","attn_q","attn_q_norm","attn_output","post_attention_norm",
                         "ffn_norm","ffn_gate","ffn_up","ffn_down","post_ffw_norm","layer_output_scale")])
    miss = [n for n in need if n not in T]
    print(f"[tensors] required={len(need)} present={len(need)-len(miss)} missing={miss}")
    x, inp_h, kv = gen_inputs(T)
    lg1, hn1, hid1, dims = forward(T, x, inp_h, kv)
    lg2, hn2, _, _ = forward(T, x, inp_h, kv)
    print("[chain]", " -> ".join(f"{k}={v}" for k, v in dims))
    det = bool(np.array_equal(lg1, lg2) and np.array_equal(hn1, hn2))
    fin = bool(np.all(np.isfinite(lg1)) and np.all(np.isfinite(hn1)))
    am = int(np.argmax(lg1)); valid = 0 <= am < 262144
    ok = (not miss) and dims[0][1] == 1024 and lg1.shape[0] == 262144 and hn1.shape[0] == 3840 and det and fin and valid
    print(f"[gate] dims_ok={dims[0][1]==1024 and lg1.shape[0]==262144 and hn1.shape[0]==3840} "
          f"deterministic={det} finite={fin} argmax={am}(valid={valid}) "
          f"logit[min/max]={lg1.min():.3f}/{lg1.max():.3f} |h_next|={np.linalg.norm(hn1):.3f}")
    # K-step EAGLE drive (the recurrence spec.rs's independent-draft assumption gets wrong)
    KDRIVE = 4
    dtoks, dxs = drive(T, x, inp_h, kv, KDRIVE)
    dtoks2, _ = drive(T, x, inp_h, kv, KDRIVE)
    drive_det = (dtoks == dtoks2)
    drive_ok = drive_det and all(0 <= t < 262144 for t in dtoks) and len(set(dtoks)) >= 1
    print(f"[drive] K={KDRIVE} tokens={dtoks} deterministic={drive_det}")
    if DUMP:
        os.makedirs(DUMP, exist_ok=True)
        x.tofile(os.path.join(DUMP, "x.f32")); inp_h.tofile(os.path.join(DUMP, "h.f32"))
        for il, (K, V) in enumerate(kv):
            K.tofile(os.path.join(DUMP, f"k{il}.f32")); V.tofile(os.path.join(DUMP, f"v{il}.f32"))
        lg1.tofile(os.path.join(DUMP, "logits.f32")); hn1.tofile(os.path.join(DUMP, "hnext.f32"))
        hid1.tofile(os.path.join(DUMP, "hidden.f32"))     # post-output_norm hidden (CUDA gate target)
        # draft weights as raw f32 (for the standalone CUDA gate; no head/token_embd needed)
        wd_ = os.path.join(DUMP, "w"); os.makedirs(wd_, exist_ok=True)
        T["nextn.pre_projection.weight"].tofile(os.path.join(wd_, "pre_proj.f32"))
        T["nextn.post_projection.weight"].tofile(os.path.join(wd_, "post_proj.f32"))
        T["output_norm.weight"].tofile(os.path.join(wd_, "out_norm.f32"))
        for il in range(NL):
            p = f"blk.{il}."
            for tag, nm in [("attn_norm","attn_norm"),("wq","attn_q"),("qn","attn_q_norm"),
                            ("wo","attn_output"),("post_attn","post_attention_norm"),
                            ("ffn_norm","ffn_norm"),("wg","ffn_gate"),("wu","ffn_up"),
                            ("wd","ffn_down"),("post_ffw","post_ffw_norm"),("osc","layer_output_scale")]:
                T[p + nm + ".weight"].tofile(os.path.join(wd_, f"l{il}_{tag}.f32"))
        for k, xk in enumerate(dxs): xk.tofile(os.path.join(DUMP, f"dx{k}.f32"))
        with open(os.path.join(DUMP, "drive_tokens.txt"), "w") as f:
            f.write(f"{KDRIVE}\n" + " ".join(str(t) for t in dtoks) + "\n")
        print(f"[dump] fixture -> {DUMP} (P={P} POS={POS}); single argmax={am}; drive K={KDRIVE} tokens={dtoks}")
    print("G-EAGLE-DRAFT-REF:", "GREEN" if (ok and drive_ok) else "RED")
    sys.exit(0 if (ok and drive_ok) else 1)

if __name__ == "__main__":
    main()
