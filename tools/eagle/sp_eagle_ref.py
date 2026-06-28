#!/usr/bin/env python3
# sp_eagle_ref.py -- REFERENCE ORACLE for the gemma4-assistant (EAGLE/MTP) draft forward.
#
# Purpose: a numpy reference of ONE draft step, grounded VERBATIM in llama.cpp PR #23398
# (src/models/gemma4-assistant.cpp) and matched to the engine's gemma4 conventions
# (lib/.../core/forward/gemma4.c: g4_gelu tanh GeGLU, embscale=sqrt(E), out_scale x*=s,
# RMSNorm x*inv*g, sp_rope_neox). This is the gate the CUDA/daemon draft port (step 2c/2d)
# must match. It exercises EVERY draft weight tensor in the exact spec order on the REAL
# transcoded weights (read here from the equivalent F16 GGUF -- same values as the sp-Q4).
#
# AUTHORITATIVE FORWARD (PR #23398):
#   x   = target_tok_embd[token] * sqrt(n_embd_out=3840)          (TARGET's 3840 embd, scaled)
#   xh  = concat([x, inp_h], dim0)                                 (7680 = 2*3840; EMBED FIRST)
#   cur = nextn.pre_projection @ xh                                (-> 1024)
#   for il in 0..3 (sandwich norm, GeGLU, q_norm per head, RoPE, Q-only attn on TARGET KV):
#     n   = rms(cur, attn_norm)
#     Q   = wq @ n ; reshape [n_head=16, head_dim] ; Q = rms_head(Q, attn_q_norm) ; rope(Q)
#     a   = attn(Q, K_target[il_src], V_target[il_src])   # il_src = n-2 (SWA) or n-1 (full)
#     a   = wo @ a ; a = rms(a, attn_post_norm) ; attn_out = a + cur
#     f   = rms(attn_out, ffn_norm) ; f = ffn_down @ (g4_gelu(ffn_gate@f) * (ffn_up@f))
#     f   = rms(f, ffn_post_norm) ; cur = (f + attn_out) * layer_output_scale
#   cur    = rms(cur, output_norm)
#   logits = draft_tok_embd @ cur          (DRAFT's OWN 1024-dim tied head -- NOT the target's)
#   h_next = nextn.post_projection @ cur    (-> 3840 ; the recurrent feature for the next step)
#
# SCOPE / honest marks:
#   - inp_h and the target token-embedding x and the target K/V are SYNTHETIC (deterministic):
#     in the live path inp_h = llama_get_embeddings_nextn(target) (the post-output_norm 3840
#     hidden = the tap at decode.c:864/878), x = target_tok_embd[token], and K/V are the 12B's
#     stored KV rows for layers (n-1 full, n-2 SWA). The GQA grouping over the target's kv-heads
#     is a LIVE-integration detail (2d); here attention uses a simple MHA over synthetic K/V.
#   - This oracle therefore validates: weight loadability, the full dimensional chain, the
#     pre/post projection wiring, the sandwich-norm + q_norm + GeGLU + RoPE + out_scale block
#     compute, the DRAFT-tied head, determinism, and finiteness. It does NOT assert acceptance.
import sys, math, numpy as np
from gguf import GGUFReader

GGUF = sys.argv[1] if len(sys.argv) > 1 else r"D:\Files\Models\Gemma4\gemma-4-it-mtp\gemma-4-12b-it-F16-MTP.gguf"
NL, HID, BB, NH, EPS = 4, 1024, 3840, 16, 1e-6
ROPE_BASE_FULL, ROPE_BASE_SWA = 1e6, 1e4
SWA_PATTERN = [True, True, True, False]   # [8,8,8,1] kv + assertion: last layer full, penult SWA

def load():
    r = GGUFReader(GGUF)
    T = {t.name: np.array(t.data, dtype=np.float32) for t in r.tensors}
    return T

def W(T, name):                # linear weight: GGUF ne=[in,out] -> numpy (out,in); y = data @ x
    return T[name]
def lin(w, x):
    assert w.shape[1] == x.shape[-1], f"dim {w.shape} @ {x.shape}"
    return w @ x
def rms(x, g):                 # gemma4.c: inv=1/sqrt(mean(x^2)+eps); x*inv*g  (+1 baked into g by GGUF)
    inv = 1.0 / math.sqrt(float((x * x).mean()) + EPS)
    return x * inv * g
def rms_head(Q, g):            # per-head RMSNorm over head_dim; Q [NH, hd], g [hd]
    return np.stack([rms(Q[h], g) for h in range(Q.shape[0])], 0)
def g4_gelu(x):                # tanh approx (ggml_gelu)
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
def rope_neox(v, pos, base):   # v [hd]; neox: rotate (i, i+hd/2) pairs
    hd = v.shape[0]; half = hd // 2
    i = np.arange(half)
    inv = base ** (-(2.0 * i) / hd)
    c, s = np.cos(pos * inv), np.sin(pos * inv)
    out = v.copy()
    out[:half]   = v[:half] * c - v[half:] * s
    out[half:]   = v[half:] * c + v[:half] * s
    return out

def block(T, il, cur, K, V, pos):
    p = f"blk.{il}."
    n = rms(cur, T[p + "attn_norm.weight"])
    Q = lin(W(T, p + "attn_q.weight"), n)            # [NH*hd]
    hd = Q.shape[0] // NH
    Q = Q.reshape(NH, hd)
    Q = rms_head(Q, T[p + "attn_q_norm.weight"])
    base = ROPE_BASE_SWA if SWA_PATTERN[il] else ROPE_BASE_FULL
    Q = np.stack([rope_neox(Q[h], pos, base) for h in range(NH)], 0)
    # Q-only attention over the (synthetic) TARGET K/V; simple MHA, ascale=1/sqrt(hd)
    asc = 1.0 / math.sqrt(hd)
    ctx = np.empty((NH, hd), np.float32)
    for h in range(NH):
        score = (K @ Q[h]) * asc                     # [P]
        score -= score.max()
        w = np.exp(score); w /= w.sum()
        ctx[h] = w @ V                               # [hd]
    a = lin(W(T, p + "attn_output.weight"), ctx.reshape(-1))
    a = rms(a, T[p + "post_attention_norm.weight"])
    attn_out = a + cur
    f = rms(attn_out, T[p + "ffn_norm.weight"])
    gate = g4_gelu(lin(W(T, p + "ffn_gate.weight"), f))
    up = lin(W(T, p + "ffn_up.weight"), f)
    f = lin(W(T, p + "ffn_down.weight"), gate * up)
    f = rms(f, T[p + "post_ffw_norm.weight"])
    out = (f + attn_out) * float(T[p + "layer_output_scale.weight"].reshape(-1)[0])
    return out, hd

def forward(T, token, inp_h, seed=0):
    rng = np.random.default_rng(seed)
    # SYNTHETIC stand-ins (deterministic): target embedding row + target K/V per draft layer
    x = rng.standard_normal(BB).astype(np.float32) * 0.02 * math.sqrt(BB)   # target_emb[token]*sqrt(BB)
    xh = np.concatenate([x, inp_h]).astype(np.float32)                      # EMBED FIRST (PR #23398)
    cur = lin(W(T, "nextn.pre_projection.weight"), xh)                      # -> 1024
    dims = [("pre_proj", cur.shape[0])]
    P, pos = 12, 7
    for il in range(NL):
        hd = (W(T, f"blk.{il}.attn_q.weight").shape[0]) // NH
        K = rng.standard_normal((P, hd)).astype(np.float32) * 0.1
        V = rng.standard_normal((P, hd)).astype(np.float32) * 0.1
        cur, hd = block(T, il, cur, K, V, pos)
        dims.append((f"blk.{il}(hd={hd})", cur.shape[0]))
    cur = rms(cur, T["output_norm.weight"])
    logits = lin(W(T, "token_embd.weight"), cur)                            # DRAFT tied head -> vocab
    h_next = lin(W(T, "nextn.post_projection.weight"), cur)                 # -> 3840
    dims += [("output_norm", cur.shape[0]), ("logits", logits.shape[0]), ("h_next", h_next.shape[0])]
    return logits, h_next, dims

def main():
    print(f"[load] {GGUF}")
    T = load()
    need = (["nextn.pre_projection.weight", "nextn.post_projection.weight",
             "token_embd.weight", "output_norm.weight"]
            + [f"blk.{il}.{s}.weight" for il in range(NL)
               for s in ("attn_norm","attn_q","attn_q_norm","attn_output","post_attention_norm",
                         "ffn_norm","ffn_gate","ffn_up","ffn_down","post_ffw_norm","layer_output_scale")])
    miss = [n for n in need if n not in T]
    print(f"[tensors] required={len(need)} present={len(need)-len(miss)} missing={miss}")
    inp_h = np.random.default_rng(99).standard_normal(BB).astype(np.float32) * 0.1   # synthetic feature
    lg1, hn1, dims = forward(T, token=12345, inp_h=inp_h, seed=0)
    lg2, hn2, _    = forward(T, token=12345, inp_h=inp_h, seed=0)               # determinism re-run
    print("[chain]", " -> ".join(f"{k}={v}" for k, v in dims))
    det = bool(np.array_equal(lg1, lg2) and np.array_equal(hn1, hn2))
    fin = bool(np.all(np.isfinite(lg1)) and np.all(np.isfinite(hn1)))
    am = int(np.argmax(lg1)); valid = 0 <= am < 262144
    ok = (not miss) and dims[0][1] == 1024 and lg1.shape[0] == 262144 and hn1.shape[0] == 3840 and det and fin and valid
    print(f"[gate] dims_ok={dims[0][1]==1024 and lg1.shape[0]==262144 and hn1.shape[0]==3840} "
          f"deterministic={det} finite={fin} argmax={am}(valid={valid}) "
          f"logit[min/max]={lg1.min():.3f}/{lg1.max():.3f} |h_next|={np.linalg.norm(hn1):.3f}")
    print("G-EAGLE-DRAFT-REF:", "GREEN" if ok else "RED")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
