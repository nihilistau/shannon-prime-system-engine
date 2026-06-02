/* forward.c — Qwen3 f32 reference forward pass (CPU scalar). E_CPU_2.
 *
 * 13-step transformer prefill over a token-ID sequence, causal:
 *   embed -> per layer { RMSNorm -> Q/K/V proj -> per-head QK-RMSNorm ->
 *     RoPE(NEOX) -> GQA causal attention (fp32 softmax) -> O proj -> residual
 *     -> RMSNorm -> SwiGLU FFN -> residual } -> final RMSNorm -> LM head.
 * Weights are dequantized on demand (each weight row once per matmul).
 */
#define _CRT_SECURE_NO_WARNINGS   /* getenv is fine here (MSVC C4996) */
#include "sp_engine/model.h"
#include "sp_engine/kernels.h"
#include "sp_engine/arena.h"
#include "sp/frobenius_lift.h"
#include "sp/poly_ring.h"
#include "sp/spinor_block.h"
#include "sp_engine/ring2_disk.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#if defined(SP_ENGINE_AVX2) || defined(SP_ENGINE_AVX512)
#include <immintrin.h>
#endif

/* Format-lock (Piece 3, roadmap §8.2.2): the persistent Spinor KV cache stores
 * NBLK = ceil(head_dim/SP_SPINOR_BODY_LEN) frozen 63-byte blocks per head — the
 * on-disk/on-wire §4.9 KV layout. The split arithmetic + codec live in the math
 * core (sp_spinor_blocks_for / sp_spinor_encode_vec / sp_spinor_decode_vec); these
 * asserts freeze the block contract the engine and the GPU backends both depend on,
 * so a change is a compile error until SP_SPINOR_LAYOUT_VERSION is bumped + migrated. */
_Static_assert(sizeof(sp_spinor_block_t) == 63, "Spinor KV block is 63 bytes (frozen)");
_Static_assert(SP_SPINOR_BODY_LEN == 55, "Spinor block carries 55 anchors; KV head-split assumes this");
_Static_assert(SP_SPINOR_LAYOUT_VERSION == 1u, "Spinor KV layout v1 frozen; bump + migrate to change");


/* NTT-attention (E_CPU_5): when SP_ENGINE_NTT_ATTN=1, each attention score <q,k>
 * is computed by the Phase-1C poly-ring kernel — quantize the (post-norm, post-
 * RoPE) head vectors to int32 (scale SP_NTT_ATTN_SCALE), recover <q,k> EXACTLY as
 * coefficient 0 of the negacyclic product (sp_pr_inner), then divide back out the
 * scale. Sieve OFF. Softmax + V-sum stay f32. Gated against the f32-dot baseline. */
static int g_ntt_attn = 0;
#define SP_NTT_ATTN_SCALE 65536.0   /* 2^16: |q_int| ~ 2^21, |<q,k>| ~ 2^49 << M/2 ~ 2^59 */
#define SP_KSTE_KV_SCALE  65536.0   /* fixed int32 quant for KSTE KV signatures (E_CPU_6) */

/* Inline VHT2+Spinor KV-cache compression (E_CPU_8). When SP_KV_SPINOR=1 each
 * post-norm/post-RoPE K and post-proj V head vector (head_dim long) is stored as
 * the frozen 63-byte Spinor block(s) and decoded back (lossy) before attention
 * reads it — the foundational KV codec (§4.5/§4.9), distinct from the KSTE sieve
 * overlay (E_CPU_6). Gate OFF skips it entirely => bit-identical to E_CPU_2. */
static int g_kv_spinor = 0;
/* SP_KV_SPINOR_REF=1: in qwen3_generate_kv, store the cache as f32 + an in-place
 * round-trip (the §4.9 "reference fp32 cache, parity tests only") instead of the
 * production Spinor-block cache. Decode-from-block is arithmetically identical to
 * the in-place round-trip, so the two paths must produce identical sequences —
 * that equivalence is the Piece-2 gate (cf. E_CPU_9's arena==FROB byte gate). */
static int g_kv_spinor_ref = 0;

/* The multi-block KV head codec (head vector -> ceil(d/55) frozen 63-byte blocks,
 * balanced split) lives in the math core: sp_spinor_blocks_for / sp_spinor_encode_vec
 * / sp_spinor_decode_vec (sp/spinor_block.h). The persistent KV cache in
 * qwen3_generate_kv calls those directly. The only engine-local helper is the
 * in-place round-trip used by the prefill KV path (E_CPU_8) and the generate_kv
 * f32 parity reference. */
#define KV_HEAD_MAX_BLOCKS 16   /* stack temp; covers head_dim up to 16*55 = 880 */
static void kv_spinor_roundtrip(float *vec, int d) {
    sp_spinor_block_t blks[KV_HEAD_MAX_BLOCKS];
    if (sp_spinor_blocks_for(d) > KV_HEAD_MAX_BLOCKS) return;   /* head_dim beyond supported range */
    sp_spinor_encode_vec(vec, d, blks);
    (void)sp_spinor_decode_vec(blks, d, vec);   /* own freshly-encoded blocks: CRC valid */
}

/* ── C2.1 recall-router sidecar (±1 Rademacher projection). Default OFF (g_recall_b==0)
 * => attention is the exact full-context baseline (parity). When SP_RECALL_B>0, the
 * decode attention restricts to recent-W ∪ top-(B-W) cached tokens by projected score.
 * R is FROZEN: deterministic from a pinned seed via integer SplitMix64 -> identical
 * across steps/backends/runs (stored projections recall correctly). r×head_dim, ±1. */
static int g_recall_b = 0;   /* SP_RECALL_B: recall budget (0 = off = full attention) */
static int g_recall_r = 16;  /* SP_RECALL_R: projection rank */
static int g_recall_w = 64;  /* SP_RECALL_W: always-keep recent window */
static int g_recall_sink = 4;/* SP_RECALL_SINK: StreamingLLM attention-sink anchors — first `sink` tokens
                              * pinned in Ring-1 (Mobius cold-start), never evicted/routed. Protects the
                              * softmax denominator (without sinks, hard top-B truncation detonates PPL). */
static int g_ring2    = 0;   /* SP_RING2=1 (Step 2a): tokens older than W spill to a mock Ring-2 byte
                              * store (RAM); in-RAM kc/vc are POISONED on window-exit so attention MUST
                              * fetch old tokens from Ring-2 (else NaN). Parity test of spill→fetch. */
static int g_ring2_disk = 0;        /* SP_RING2_DISK=1 (Step 2b): physical Optane file store (else RAM mock) */
static const char *g_ring2_dir = 0; /* SP_RING2_DIR: store directory (default "E:\\") */
#define SP_RECALL_R_MAX 64
#define SP_RECALL_SEED 0x5350524F4A2BULL   /* "SPROJ+" frozen seed; bump = SP_RECALL_PROJ_VERSION */
static uint64_t splitmix64(uint64_t *s) { uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL; return z ^ (z >> 31); }
/* fill R[r*hd] with ±1 deterministically from the frozen seed. */
static void recall_build_R(signed char *R, int r, int hd) {
    uint64_t s = SP_RECALL_SEED;
    for (int i = 0; i < r * hd; i++) R[i] = (splitmix64(&s) & 1) ? 1 : -1;
}
/* proj[p] = R[p,:]·vec  (r outputs, ±1 matrix → add/sub). */
static void recall_project(const signed char *R, int r, int hd, const float *vec, float *proj) {
    for (int p = 0; p < r; p++) { const signed char *Rp = R + (size_t)p * hd; float a = 0.0f;
        for (int d = 0; d < hd; d++) a += (float)Rp[d] * vec[d]; proj[p] = a; }
}

/* dot_f32, matmul/matmul_arena, row_bytes, and the SP_CPU_SCALAR/F16_ACT/FROB
 * knobs moved to kernels.{h,c} (shared with gemma3.c). */

/* rmsnorm, rmsnorm_head, rope_neox, as_f32, embed_row moved to kernels.{h,c}. */

/* Read the runtime gate knobs once per forward/decode entry (all default OFF =
 * the pure-f32 E_CPU_2 reference path). Shared by the prefill and the decode loop
 * so both honor the same gates. */
static void read_env_knobs(void) {
    sp_kernels_read_env();   /* SP_CPU_SCALAR, SP_ENGINE_F16_ACT, SP_ENGINE_FROB, SP_Q4_PROMOTE */
    { const char *e = getenv("SP_ENGINE_NTT_ATTN"); g_ntt_attn = (e && e[0] == '1'); }
    { const char *e = getenv("SP_KV_SPINOR");       g_kv_spinor = (e && e[0] == '1'); }
    { const char *e = getenv("SP_KV_SPINOR_REF");   g_kv_spinor_ref = (e && e[0] == '1'); }
    { const char *e = getenv("SP_RECALL_B");        g_recall_b = e ? atoi(e) : 0; if (g_recall_b < 0) g_recall_b = 0; }
    { const char *e = getenv("SP_RECALL_R");        g_recall_r = e ? atoi(e) : 16; if (g_recall_r < 1 || g_recall_r > SP_RECALL_R_MAX) g_recall_r = 16; }
    { const char *e = getenv("SP_RECALL_W");        g_recall_w = e ? atoi(e) : 64; if (g_recall_w < 0) g_recall_w = 0; }
    { const char *e = getenv("SP_RECALL_SINK");     g_recall_sink = e ? atoi(e) : 4; if (g_recall_sink < 0) g_recall_sink = 0; }
    { const char *e = getenv("SP_RING2");           g_ring2 = (e && e[0] == '1'); }
    { const char *e = getenv("SP_RING2_DISK");      g_ring2_disk = (e && e[0] == '1'); }
    { const char *e = getenv("SP_RING2_DIR");       g_ring2_dir = (e && e[0]) ? e : "E:\\"; }
}

int qwen3_forward_ex(const qwen3_model *m, const int32_t *tokens, int n_tok,
                     float *logits, sp_kste_tree_t *kv_trees) {
    const qwen3_config *c = &m->cfg;
    const int E = (int)c->n_embd, FF = (int)c->n_ff, HD = (int)c->head_dim;
    const int NH = (int)c->n_head, NKV = (int)c->n_head_kv;
    const int QD = NH * HD;          /* q proj width  (2048) */
    const int KVD = NKV * HD;        /* kv proj width (1024) */
    const int group = NH / NKV;      /* q-heads per kv-head (2) */
    const int V = (int)c->n_vocab;
    const float eps = c->rms_eps, base = c->rope_freq_base;
    const float ascale = 1.0f / sqrtf((float)HD);

    read_env_knobs();

    int rc = 1;
    sp_pr_ctx *pr = NULL;          /* poly-ring context for NTT-attention (N=head_dim) */
    int32_t *qi = NULL, *ki = NULL;
    int32_t *kq = NULL;            /* int32 scratch for KSTE KV encoding */
    float *x   = (float *)malloc((size_t)n_tok * E * sizeof(float));   /* residual stream */
    float *nx  = (float *)malloc((size_t)n_tok * E * sizeof(float));   /* normed */
    float *q   = (float *)malloc((size_t)n_tok * QD * sizeof(float));
    float *k   = (float *)malloc((size_t)n_tok * KVD * sizeof(float));
    float *v   = (float *)malloc((size_t)n_tok * KVD * sizeof(float));
    float *ao  = (float *)malloc((size_t)n_tok * QD * sizeof(float));  /* attn out (concat heads) */
    float *ap  = (float *)malloc((size_t)n_tok * E * sizeof(float));   /* attn out proj */
    float *g   = (float *)malloc((size_t)n_tok * FF * sizeof(float));
    float *up  = (float *)malloc((size_t)n_tok * FF * sizeof(float));
    float *dn  = (float *)malloc((size_t)n_tok * E * sizeof(float));
    float *sc  = (float *)malloc((size_t)n_tok * sizeof(float));       /* attn scores */
    if (!x || !nx || !q || !k || !v || !ao || !ap || !g || !up || !dn || !sc) goto done;

    if (g_ntt_attn) {
        pr = sp_pr_init((uint32_t)HD);          /* head_dim must be in {128,256,512} */
        qi = (int32_t *)malloc((size_t)HD * sizeof(int32_t));
        ki = (int32_t *)malloc((size_t)HD * sizeof(int32_t));
        if (!pr || !qi || !ki) goto done;
    }
    if (kv_trees) {
        kq = (int32_t *)malloc((size_t)HD * sizeof(int32_t));
        if (!kq) goto done;
    }

    /* ── embedding lookup: token t's embedding is the contiguous E floats at t*E ── */
    for (int t = 0; t < n_tok; t++)
        if (embed_row(m, tokens[t], E, x + (size_t)t * E)) goto done;

    for (uint32_t L = 0; L < c->n_layers; L++) {
        const qwen3_layer *ly = &m->layers[L];

        /* ── attention block ── */
        for (int t = 0; t < n_tok; t++)
            rmsnorm(x + (size_t)t * E, as_f32(m, ly->attn_norm), E, eps, nx + (size_t)t * E);

        if (matmul(m, ly->attn_q, nx, n_tok, E, QD, q)) goto done;
        if (matmul(m, ly->attn_k, nx, n_tok, E, KVD, k)) goto done;
        if (matmul(m, ly->attn_v, nx, n_tok, E, KVD, v)) goto done;

        /* per-head QK-RMSNorm (over head_dim) then NEOX RoPE at position t */
        const float *qn = as_f32(m, ly->attn_q_norm);
        const float *kn = as_f32(m, ly->attn_k_norm);
        for (int t = 0; t < n_tok; t++) {
            for (int h = 0; h < NH; h++) {
                float *qh = q + (size_t)t * QD + (size_t)h * HD;
                rmsnorm_head(qh, qn, HD, eps);
                rope_neox(qh, HD, t, base);
            }
            for (int h = 0; h < NKV; h++) {
                float *kh = k + (size_t)t * KVD + (size_t)h * HD;
                rmsnorm_head(kh, kn, HD, eps);
                rope_neox(kh, HD, t, base);
            }
        }

        /* Inline VHT2+Spinor KV compression (E_CPU_8): store each cached K/V head
         * vector as Spinor block(s) and read back the lossy reconstruction. Applied
         * after QK-norm+RoPE so the cache holds position-finalized K (and post-proj
         * V), exactly what the persistent-KV decode path will store. */
        if (g_kv_spinor) {
            for (int t = 0; t < n_tok; t++)
                for (int h = 0; h < NKV; h++) {
                    kv_spinor_roundtrip(k + (size_t)t * KVD + (size_t)h * HD, HD);
                    kv_spinor_roundtrip(v + (size_t)t * KVD + (size_t)h * HD, HD);
                }
        }

        /* KSTE KV-cache overlay (E_CPU_6): encode each cached K head-vector to its
         * 64-byte signature. Deterministic int32 quantization -> byte-identical. */
        if (kv_trees) {
            for (int t = 0; t < n_tok; t++)
                for (int h = 0; h < NKV; h++) {
                    const float *kh = k + (size_t)t * KVD + (size_t)h * HD;
                    for (int i = 0; i < HD; i++)
                        kq[i] = (int32_t)lrintf(kh[i] * (float)SP_KSTE_KV_SCALE);
                    sp_kste_encode(kq, HD, &kv_trees[((size_t)L * n_tok + t) * NKV + h]);
                }
        }

        /* GQA causal attention. Plain f32 path uses the shared kernels_attn_head
         * (full causal, win=-1); the NTT-attention overlay (E_CPU_5) stays inline
         * here since its score is computed via the exact poly-ring inner product. */
        for (int t = 0; t < n_tok; t++) {
            for (int h = 0; h < NH; h++) {
                int kvh = h / group;
                const float *qh = q + (size_t)t * QD + (size_t)h * HD;
                float *out = ao + (size_t)t * QD + (size_t)h * HD;
                if (!g_ntt_attn) {
                    kernels_attn_head(qh, k, v, t, KVD, kvh, HD, ascale, -1, sc, out);
                    continue;
                }
                for (int i = 0; i < HD; i++) qi[i] = (int32_t)lrintf(qh[i] * (float)SP_NTT_ATTN_SCALE);
                float maxs = -INFINITY;
                for (int s = 0; s <= t; s++) {
                    const float *kh = k + (size_t)s * KVD + (size_t)kvh * HD;
                    for (int i = 0; i < HD; i++) ki[i] = (int32_t)lrintf(kh[i] * (float)SP_NTT_ATTN_SCALE);
                    int64_t ip = sp_pr_inner(pr, qi, ki);   /* exact <q_int,k_int> */
                    float d = (float)((double)ip / (SP_NTT_ATTN_SCALE * SP_NTT_ATTN_SCALE)) * ascale;
                    sc[s] = d;
                    if (d > maxs) maxs = d;
                }
                float sum = 0.0f;
                for (int s = 0; s <= t; s++) { sc[s] = expf(sc[s] - maxs); sum += sc[s]; }
                float inv = 1.0f / sum;
                for (int i = 0; i < HD; i++) out[i] = 0.0f;
                for (int s = 0; s <= t; s++) {
                    float w = sc[s] * inv;
                    const float *vh = v + (size_t)s * KVD + (size_t)kvh * HD;
                    for (int i = 0; i < HD; i++) out[i] += w * vh[i];
                }
            }
        }

        if (matmul(m, ly->attn_output, ao, n_tok, QD, E, ap)) goto done;
        for (size_t i = 0; i < (size_t)n_tok * E; i++) x[i] += ap[i];   /* residual */

        /* ── FFN block (SwiGLU) ── */
        for (int t = 0; t < n_tok; t++)
            rmsnorm(x + (size_t)t * E, as_f32(m, ly->ffn_norm), E, eps, nx + (size_t)t * E);
        if (matmul(m, ly->ffn_gate, nx, n_tok, E, FF, g)) goto done;
        if (matmul(m, ly->ffn_up,   nx, n_tok, E, FF, up)) goto done;
        for (size_t i = 0; i < (size_t)n_tok * FF; i++) {
            float gv = g[i];
            float silu = gv / (1.0f + expf(-gv));
            g[i] = silu * up[i];
        }
        if (matmul(m, ly->ffn_down, g, n_tok, FF, E, dn)) goto done;
        for (size_t i = 0; i < (size_t)n_tok * E; i++) x[i] += dn[i];    /* residual */
    }

    /* ── final norm + LM head ── */
    for (int t = 0; t < n_tok; t++)
        rmsnorm(x + (size_t)t * E, as_f32(m, m->output_norm), E, eps, nx + (size_t)t * E);
    if (matmul(m, m->output, nx, n_tok, E, V, logits)) goto done;

    rc = 0;
done:
    free(x); free(nx); free(q); free(k); free(v); free(ao); free(ap);
    free(g); free(up); free(dn); free(sc);
    free(qi); free(ki); free(kq); sp_pr_free(pr);
    return rc;
}

int qwen3_forward(const qwen3_model *m, const int32_t *tokens, int n_tok, float *logits) {
    return qwen3_forward_ex(m, tokens, n_tok, logits, NULL);
}

/* Persistent-KV O(n) greedy decode (GEN_KV). Same result as qwen3_generate but
 * each token is processed once: per-layer K/V are computed for the single new
 * token, stored post-RoPE into a position-indexed cache, and attention reads the
 * cached K/V for all earlier positions. The expensive weight matmuls run on one
 * token per step (O(n) total) instead of re-prefilling the whole prefix (O(n^2)).
 *
 * Honors the same gates as the prefill (SP_ENGINE_FROB, SP_CPU_SCALAR, SP_KV_SPINOR;
 * the cache stores Spinor-compressed K/V when SP_KV_SPINOR=1). The NTT-attention
 * path is prefill-only and not wired here. Greedy argmax must match qwen3_generate
 * up to the float-reassociation floor (different softmax-sum lengths) — GEN_KV
 * gates on argmax/sequence identity, not bit-equal logits. */
/* Compute one query head's recall set into ri[0,m): full [0,pos] (exact baseline)
 * unless B>0 and context exceeds budget, then sinks ∪ top-(B-W-sink candidates) ∪
 * recent-W, by ±1-projected score. `scl` is (pos+1)-float scratch. Returns m.
 * Shared by the RAM/no-Ring-2 inline path and the disk phase-split — one selection. */
static int recall_select(const signed char *R, int rr, int hd, const float *qh,
                         const float *projk, size_t L, int P, int NKV, int kvh,
                         int B, int W0, int sink0, int pos, float *scl, int *ri) {
    if (B <= 0 || pos + 1 <= B) { for (int s = 0; s <= pos; s++) ri[s] = s; return pos + 1; }
    float pq[SP_RECALL_R_MAX];
    recall_project(R, rr, hd, qh, pq);
    int W = W0;       if (W > pos + 1) W = pos + 1;
    int sink = sink0; if (sink > pos + 1) sink = pos + 1;
    int cand_hi = pos + 1 - W;
    if (cand_hi < sink) cand_hi = sink;
    if (cand_hi > pos + 1) cand_hi = pos + 1;
    int topk = B - W - sink; if (topk < 0) topk = 0;
    if (topk > cand_hi - sink) topk = cand_hi - sink;
    for (int s = sink; s < cand_hi; s++) {
        const float *pk = projk + (((size_t)L * P + s) * NKV + kvh) * (size_t)rr;
        float a = 0.0f; for (int p = 0; p < rr; p++) a += pq[p] * pk[p];
        scl[s] = a;
    }
    int m = 0;
    for (int s = 0; s < sink; s++) ri[m++] = s;            /* pinned sink anchors */
    for (int t = 0; t < topk; t++) {                       /* top-topk candidates (max-extract) */
        int best = -1; float bv = -INFINITY;
        for (int s = sink; s < cand_hi; s++) if (scl[s] > bv) { bv = scl[s]; best = s; }
        if (best < 0) break; ri[m++] = best; scl[best] = -INFINITY;
    }
    for (int s = cand_hi; s <= pos; s++) ri[m++] = s;      /* recent window */
    return m;
}

/* Shared decode body for qwen3_generate_kv (ppl_mode=0, argmax emit) and
 * qwen3_ppl_decode (ppl_mode=1, teacher-forced NLL of seq[pos+1] over [n_warm,P-2]).
 * ONE forward path — the recall router + two-ring are identical in both modes. */
static int generate_kv_impl(const qwen3_model *m, int32_t *seq, int n_prompt, int n_gen,
                            int eos_id, int ppl_mode, int n_warm,
                            double *nll_out, long *nscored_out) {
    if (!m || !seq || n_prompt <= 0 || n_gen < 0) return -1;
    read_env_knobs();
    const qwen3_config *c = &m->cfg;
    const int E = (int)c->n_embd, FF = (int)c->n_ff, HD = (int)c->head_dim;
    const int NH = (int)c->n_head, NKV = (int)c->n_head_kv;
    const int QD = NH * HD, KVD = NKV * HD, group = NH / NKV, V = (int)c->n_vocab;
    const float eps = c->rms_eps, base = c->rope_freq_base, ascale = 1.0f / sqrtf((float)HD);
    const int P = n_prompt + n_gen;

    /* Production KV layout (Piece 2): SP_KV_SPINOR=1 stores the cache as frozen
     * 63-byte Spinor blocks (NBLK per head) and decodes on read into a per-LAYER
     * f32 scratch, so resident KV memory is the packed blocks + one layer's worth
     * of scratch, not the full f32 cache. SP_KV_SPINOR_REF=1 keeps the f32 cache
     * (parity reference, §4.9). Gate off => f32 cache, no codec (regression). */
    const int NBLK = sp_spinor_blocks_for(HD);
    const int use_blocks = g_kv_spinor && !g_kv_spinor_ref;

    int rc = -1, n = n_prompt, produced = 0;
    /* All pointers NULL-first so any early `goto done` frees only NULLs. */
    sp_spinor_block_t *kcb = NULL, *vcb = NULL;   /* block KV cache (use_blocks) */
    float *kc = NULL, *vc = NULL;                 /* f32 KV cache (ref / gate-off) */
    float *kdec = NULL, *vdec = NULL;             /* per-layer decode scratch (use_blocks) */
    float *x = NULL, *nx = NULL, *q = NULL, *knew = NULL, *vnew = NULL, *ao = NULL;
    float *ap = NULL, *gg = NULL, *up = NULL, *dn = NULL, *sc = NULL, *lg = NULL;
    signed char *recallR = NULL; float *projk = NULL;   /* C2.1 recall sidecar (g_recall_b>0) */
    float *ring2k = NULL, *ring2v = NULL;                /* C2.1 Step 2a mock Ring-2 byte store (g_ring2) */
    int ring2_on = 0;
    ring2_disk *r2disk = NULL; int ring2_disk_on = 0;    /* C2.1 Step 2b physical Optane store (g_ring2_disk) */
    /* v1 dedupe: per-layer staging of the UNION of all heads' recalled blocks (read once, not per-head) */
    float *stgK = NULL, *stgV = NULL;
    int *stg_stamp = NULL, *stg_slot = NULL, *stg_pos = NULL;   /* position->slot map, generation-stamped */
    int *ri_all = NULL, *m_all = NULL;                          /* per-head recall sets: phase 1 -> phase 3 */
    ring2_scratch *r2s_io = NULL; int stg_gen = 0;             /* phase-2 read scratch (v1a blocking) */

    if (use_blocks) {
        size_t nb = (size_t)c->n_layers * P * NKV * NBLK;
        kcb  = (sp_spinor_block_t *)malloc(nb * sizeof(sp_spinor_block_t));
        vcb  = (sp_spinor_block_t *)malloc(nb * sizeof(sp_spinor_block_t));
        kdec = (float *)malloc((size_t)P * KVD * sizeof(float));
        vdec = (float *)malloc((size_t)P * KVD * sizeof(float));
        if (!kcb || !vcb || !kdec || !vdec) goto done;
        fprintf(stderr, "    [KV] Spinor-block cache: %zu B vs f32 %zu B (%.2fx) - %d blocks/head, %d B/block\n",
                2 * nb * sizeof(sp_spinor_block_t),
                2 * (size_t)c->n_layers * P * KVD * sizeof(float),
                (double)((size_t)c->n_layers * P * KVD * sizeof(float)) /
                (double)((size_t)c->n_layers * P * NKV * NBLK * sizeof(sp_spinor_block_t)),
                NBLK, (int)sizeof(sp_spinor_block_t));
    } else {
        kc = (float *)malloc((size_t)c->n_layers * P * KVD * sizeof(float)); /* K cache */
        vc = (float *)malloc((size_t)c->n_layers * P * KVD * sizeof(float)); /* V cache */
        if (!kc || !vc) goto done;
    }
    x    = (float *)malloc((size_t)E * sizeof(float));   /* single-token residual */
    nx   = (float *)malloc((size_t)E * sizeof(float));
    q    = (float *)malloc((size_t)QD * sizeof(float));
    knew = (float *)malloc((size_t)KVD * sizeof(float));
    vnew = (float *)malloc((size_t)KVD * sizeof(float));
    ao   = (float *)malloc((size_t)QD * sizeof(float));
    ap   = (float *)malloc((size_t)E * sizeof(float));
    gg   = (float *)malloc((size_t)FF * sizeof(float));
    up   = (float *)malloc((size_t)FF * sizeof(float));
    dn   = (float *)malloc((size_t)E * sizeof(float));
    sc   = (float *)malloc((size_t)P * sizeof(float));
    lg   = (float *)malloc((size_t)V * sizeof(float));
    if (!x || !nx || !q || !knew || !vnew || !ao || !ap || !gg || !up || !dn || !sc || !lg)
        goto done;
    if (g_recall_b > 0) {
        recallR = (signed char *)malloc((size_t)g_recall_r * HD);
        projk   = (float *)malloc((size_t)c->n_layers * P * NKV * (size_t)g_recall_r * sizeof(float));
        if (!recallR || !projk) goto done;
        recall_build_R(recallR, g_recall_r, HD);   /* frozen ±1 matrix, deterministic */
        fprintf(stderr, "    [recall] sidecar ON: r=%d B=%d W=%d sink=%d (post-RoPE ±1 projection router + StreamingLLM sinks)\n",
                g_recall_r, g_recall_b, g_recall_w, g_recall_sink);
    }
    ring2_on = (g_ring2 && g_recall_b > 0 && !use_blocks);   /* Step 2a/2b: f32-cache path only */
    ring2_disk_on = (ring2_on && g_ring2_disk);
    if (ring2_on && !ring2_disk_on) {
        size_t kvn = (size_t)c->n_layers * P * KVD;
        ring2k = (float *)malloc(kvn * sizeof(float));
        ring2v = (float *)malloc(kvn * sizeof(float));
        if (!ring2k || !ring2v) goto done;
        fprintf(stderr, "    [ring2] mock RAM spill ON: tokens older than W=%d spilled to RAM byte-store; "
                "kc/vc poisoned on window-exit (old reads MUST hit Ring-2)\n", g_recall_w);
    } else if (ring2_disk_on) {
        size_t bytes = (size_t)c->n_layers * P * (size_t)KVD * sizeof(float);
        r2disk = ring2_disk_open(g_ring2_dir, bytes, (size_t)KVD * sizeof(float));
        if (!r2disk) goto done;
        /* v1 dedupe staging: union of recalled blocks read ONCE per layer into RAM, then all heads
         * read their kv-head slice from here (eliminates the per-query-head redundant disk reads). */
        stgK = (float *)malloc((size_t)P * KVD * sizeof(float));
        stgV = (float *)malloc((size_t)P * KVD * sizeof(float));
        stg_stamp = (int *)malloc((size_t)P * sizeof(int));
        stg_slot  = (int *)malloc((size_t)P * sizeof(int));
        stg_pos   = (int *)malloc((size_t)P * sizeof(int));
        ri_all = (int *)malloc((size_t)NH * P * sizeof(int));
        m_all  = (int *)malloc((size_t)NH * sizeof(int));
        r2s_io = ring2_disk_scratch_new(r2disk);
        if (!stgK || !stgV || !stg_stamp || !stg_slot || !stg_pos || !ri_all || !m_all || !r2s_io) goto done;
        for (int i = 0; i < P; i++) stg_stamp[i] = -1;
        fprintf(stderr, "    [ring2] PHYSICAL Optane spill ON (W=%d, sinks pinned in Ring-1, kc/vc poisoned -> "
                "old reads MUST come off disk; v1 per-layer dedupe staging)\n", g_recall_w);
    }

    for (int pos = 0; pos < P; pos++) {
        int tok = seq[pos];
        if (embed_row(m, tok, E, x)) goto done;
        /* Step 2a: the token that just left the recent-W window (pos-W) is now "spilled" —
         * POISON its in-RAM kc/vc (all layers) so any attention read of it MUST come from
         * Ring-2 (a stale Ring-1 read → NaN → divergent tokens → test fails loudly). */
        if (ring2_on && pos - g_recall_w >= 0 && (pos - g_recall_w) >= g_recall_sink) {
            int et = pos - g_recall_w;   /* sink tokens (et < g_recall_sink) stay pinned in Ring-1, never poisoned */
            for (uint32_t L = 0; L < c->n_layers; L++) {
                float *pk = kc + ((size_t)L * P + et) * KVD, *pv = vc + ((size_t)L * P + et) * KVD;
                for (int i = 0; i < KVD; i++) { pk[i] = NAN; pv[i] = NAN; }
            }
        }

        for (uint32_t L = 0; L < c->n_layers; L++) {
            const qwen3_layer *ly = &m->layers[L];
            rmsnorm(x, as_f32(m, ly->attn_norm), E, eps, nx);
            if (matmul(m, ly->attn_q, nx, 1, E, QD, q))   goto done;
            if (matmul(m, ly->attn_k, nx, 1, E, KVD, knew)) goto done;
            if (matmul(m, ly->attn_v, nx, 1, E, KVD, vnew)) goto done;

            const float *qn = as_f32(m, ly->attn_q_norm), *kn = as_f32(m, ly->attn_k_norm);
            { int h;   /* per-head QK-norm+RoPE: each head writes a distinct slice → parallel-safe */
              #pragma omp parallel for
              for (h = 0; h < NH;  h++) { float *qh = q    + (size_t)h * HD; rmsnorm_head(qh, qn, HD, eps); rope_neox(qh, HD, pos, base); }
              #pragma omp parallel for
              for (h = 0; h < NKV; h++) { float *kh = knew + (size_t)h * HD; rmsnorm_head(kh, kn, HD, eps); rope_neox(kh, HD, pos, base); }
            }
            /* C2.1: project the post-RoPE K of this (layer,pos) per kv-head into the recall sidecar. */
            if (g_recall_b > 0)
                for (int hh = 0; hh < NKV; hh++)
                    recall_project(recallR, g_recall_r, HD, knew + (size_t)hh * HD,
                                   projk + (((size_t)L * P + pos) * NKV + hh) * (size_t)g_recall_r);
            /* Store the position-finalized K/V; KC/VC then point at the f32 the
             * attention reads as KC[s*KVD + kvh*HD]. Block path: encode into the
             * persistent block cache, decode [0,pos] of this layer into the per-
             * layer scratch. f32 path: optional in-place round-trip (parity ref),
             * then memcpy into the full f32 cache. decode(encode(x)) is identical
             * to the in-place round-trip, so the two paths agree bit-for-bit. */
            const float *KC, *VC;
            if (use_blocks) {
                sp_spinor_block_t *kb = kcb + ((size_t)L * P + pos) * NKV * NBLK;
                sp_spinor_block_t *vb = vcb + ((size_t)L * P + pos) * NKV * NBLK;
                for (int h = 0; h < NKV; h++) {
                    sp_spinor_encode_vec(knew + (size_t)h * HD, HD, kb + (size_t)h * NBLK);
                    sp_spinor_encode_vec(vnew + (size_t)h * HD, HD, vb + (size_t)h * NBLK);
                }
                for (int s = 0; s <= pos; s++) {                /* decode the live window into scratch */
                    const sp_spinor_block_t *ks = kcb + ((size_t)L * P + s) * NKV * NBLK;
                    const sp_spinor_block_t *vs = vcb + ((size_t)L * P + s) * NKV * NBLK;
                    for (int h = 0; h < NKV; h++) {
                        (void)sp_spinor_decode_vec(ks + (size_t)h * NBLK, HD, kdec + (size_t)s * KVD + (size_t)h * HD);
                        (void)sp_spinor_decode_vec(vs + (size_t)h * NBLK, HD, vdec + (size_t)s * KVD + (size_t)h * HD);
                    }
                }
                KC = kdec; VC = vdec;
            } else {
                if (g_kv_spinor)   /* SP_KV_SPINOR_REF: f32 cache with the lossy round-trip */
                    for (int h = 0; h < NKV; h++) { kv_spinor_roundtrip(knew + (size_t)h * HD, HD); kv_spinor_roundtrip(vnew + (size_t)h * HD, HD); }
                memcpy(kc + ((size_t)L * P + pos) * KVD, knew, (size_t)KVD * sizeof(float));
                memcpy(vc + ((size_t)L * P + pos) * KVD, vnew, (size_t)KVD * sizeof(float));
                KC = kc + (size_t)L * P * KVD;
                VC = vc + (size_t)L * P * KVD;
            }
            if (ring2_disk_on) {   /* Step 2b: spill this (layer,token) K/V to the physical Optane store */
                size_t boff = ((size_t)L * P + pos) * (size_t)KVD * sizeof(float);
                if (ring2_disk_write(r2disk, 0, boff, knew) || ring2_disk_write(r2disk, 1, boff, vnew)) goto done;
            } else if (ring2_on) {   /* Step 2a: spill to the mock RAM byte store */
                memcpy(ring2k + ((size_t)L * P + pos) * KVD, knew, (size_t)KVD * sizeof(float));
                memcpy(ring2v + ((size_t)L * P + pos) * KVD, vnew, (size_t)KVD * sizeof(float));
            }

            /* Attention over cached [0,pos], parallel over heads (bit-identical to serial:
             * each head writes a distinct ao slice + per-thread score scratch). winlo = recent
             * window start; old non-sink positions (s < winlo, s >= sink) live in Ring-2. */
            int winlo = (pos + 1 > g_recall_w) ? (pos + 1 - g_recall_w) : 0;
            if (ring2_disk_on) {
                /* v1 disk path: 3-phase to DEDUP the per-query-head reads. Blocks are
                 * per-token (all kv-heads), so one fetch serves every head that picked it. */
                /* PHASE 1: per-head selection only (parallel, no I/O). */
                #pragma omp parallel
                {
                    float *scl = (float *)malloc((size_t)(pos + 1) * sizeof(float));
                    if (scl) {
                        int h;
                        #pragma omp for
                        for (h = 0; h < NH; h++)
                            m_all[h] = recall_select(recallR, g_recall_r, HD, q + (size_t)h * HD,
                                projk, (size_t)L, P, NKV, h / group, g_recall_b, g_recall_w,
                                g_recall_sink, pos, scl, ri_all + (size_t)h * P);
                    }
                    free(scl);
                }
                /* PHASE 2: union the old-non-sink positions across heads, fetch each block
                 * ONCE off Optane into the per-layer RAM staging (v1a: serial blocking). */
                stg_gen++;
                int nstage = 0;
                for (int h = 0; h < NH; h++) {
                    const int *rih = ri_all + (size_t)h * P;
                    for (int jj = 0; jj < m_all[h]; jj++) {
                        int s = rih[jj];
                        if (s < winlo && s >= g_recall_sink && stg_stamp[s] != stg_gen) {
                            stg_stamp[s] = stg_gen; stg_slot[s] = nstage; stg_pos[nstage] = s; nstage++;
                        }
                    }
                }
                for (int i = 0; i < nstage; i++) {
                    size_t boff = ((size_t)L * P + stg_pos[i]) * (size_t)KVD * sizeof(float);
                    const void *kp = ring2_disk_read(r2disk, 0, boff, r2s_io);
                    if (kp) memcpy(stgK + (size_t)i * KVD, kp, (size_t)KVD * sizeof(float));
                    const void *vp = ring2_disk_read(r2disk, 1, boff, r2s_io);
                    if (vp) memcpy(stgV + (size_t)i * KVD, vp, (size_t)KVD * sizeof(float));
                }
                /* PHASE 3: attention reading old blocks from the RAM staging (parallel). */
                #pragma omp parallel
                {
                    float *scl = (float *)malloc((size_t)(pos + 1) * sizeof(float));
                    if (scl) {
                        int h;
                        #pragma omp for
                        for (h = 0; h < NH; h++) {
                            int kvh = h / group; const float *qh = q + (size_t)h * HD;
                            const int *rih = ri_all + (size_t)h * P; int m = m_all[h];
                            float maxs = -INFINITY;
                            for (int jj = 0; jj < m; jj++) {
                                int s = rih[jj];
                                const float *kbase = (s < winlo && s >= g_recall_sink)
                                    ? stgK + (size_t)stg_slot[s] * KVD : KC + (size_t)s * KVD;
                                const float *kh = kbase + (size_t)kvh * HD;
                                float acc = 0.0f; for (int i = 0; i < HD; i++) acc += qh[i] * kh[i];
                                float d = acc * ascale; scl[jj] = d; if (d > maxs) maxs = d;
                            }
                            float sum = 0.0f;
                            for (int jj = 0; jj < m; jj++) { scl[jj] = expf(scl[jj] - maxs); sum += scl[jj]; }
                            float inv = 1.0f / sum; float *out = ao + (size_t)h * HD;
                            for (int i = 0; i < HD; i++) out[i] = 0.0f;
                            for (int jj = 0; jj < m; jj++) {
                                int s = rih[jj]; float w = scl[jj] * inv;
                                const float *vbase = (s < winlo && s >= g_recall_sink)
                                    ? stgV + (size_t)stg_slot[s] * KVD : VC + (size_t)s * KVD;
                                const float *vh = vbase + (size_t)kvh * HD;
                                for (int i = 0; i < HD; i++) out[i] += w * vh[i];
                            }
                        }
                    }
                    free(scl);
                }
            } else {
                /* Non-disk path (RAM mock / no Ring-2): selection + attention inline. */
                #pragma omp parallel
                {
                    float *scl = (float *)malloc((size_t)(pos + 1) * sizeof(float));
                    int   *ri  = (int   *)malloc((size_t)(pos + 1) * sizeof(int));
                    if (scl && ri) {
                        int h;
                        #pragma omp for
                        for (h = 0; h < NH; h++) {
                            int kvh = h / group; const float *qh = q + (size_t)h * HD;
                            int m = recall_select(recallR, g_recall_r, HD, qh, projk, (size_t)L, P,
                                NKV, kvh, g_recall_b, g_recall_w, g_recall_sink, pos, scl, ri);
                            float maxs = -INFINITY;
                            for (int jj = 0; jj < m; jj++) {
                                int s = ri[jj];
                                const float *kbase = (ring2_on && s < winlo && s >= g_recall_sink)
                                    ? ring2k + ((size_t)L * P + s) * KVD : KC + (size_t)s * KVD;
                                const float *kh = kbase + (size_t)kvh * HD;
                                float acc = 0.0f; for (int i = 0; i < HD; i++) acc += qh[i] * kh[i];
                                float d = acc * ascale; scl[jj] = d; if (d > maxs) maxs = d;
                            }
                            float sum = 0.0f;
                            for (int jj = 0; jj < m; jj++) { scl[jj] = expf(scl[jj] - maxs); sum += scl[jj]; }
                            float inv = 1.0f / sum; float *out = ao + (size_t)h * HD;
                            for (int i = 0; i < HD; i++) out[i] = 0.0f;
                            for (int jj = 0; jj < m; jj++) {
                                int s = ri[jj]; float w = scl[jj] * inv;
                                const float *vbase = (ring2_on && s < winlo && s >= g_recall_sink)
                                    ? ring2v + ((size_t)L * P + s) * KVD : VC + (size_t)s * KVD;
                                const float *vh = vbase + (size_t)kvh * HD;
                                for (int i = 0; i < HD; i++) out[i] += w * vh[i];
                            }
                        }
                    }
                    free(scl); free(ri);
                }
            }
            if (matmul(m, ly->attn_output, ao, 1, QD, E, ap)) goto done;
            for (int i = 0; i < E; i++) x[i] += ap[i];

            rmsnorm(x, as_f32(m, ly->ffn_norm), E, eps, nx);
            if (matmul(m, ly->ffn_gate, nx, 1, E, FF, gg)) goto done;
            if (matmul(m, ly->ffn_up,   nx, 1, E, FF, up)) goto done;
            for (int i = 0; i < FF; i++) { float gv = gg[i]; gg[i] = gv / (1.0f + expf(-gv)) * up[i]; }
            if (matmul(m, ly->ffn_down, gg, 1, FF, E, dn)) goto done;
            for (int i = 0; i < E; i++) x[i] += dn[i];
        }

        if (ppl_mode) {
            /* G2: teacher-forced autoregressive PPL — logits at EVERY pos, accumulate
             * -log p(seq[pos+1]) for pos in [n_warm, P-2]. The recall/Ring-2 path above
             * is exercised exactly as in production decode (the whole point of G2). */
            if (pos + 1 < P && pos >= n_warm) {
                rmsnorm(x, as_f32(m, m->output_norm), E, eps, nx);
                if (matmul(m, m->output, nx, 1, E, V, lg)) goto done;
                int tgt = seq[pos + 1];
                if (tgt >= 0 && tgt < V) {
                    float maxl = lg[0];
                    for (int j = 1; j < V; j++) if (lg[j] > maxl) maxl = lg[j];
                    double sumexp = 0.0;
                    for (int j = 0; j < V; j++) sumexp += exp((double)lg[j] - (double)maxl);
                    double logp = (double)lg[tgt] - (double)maxl - log(sumexp);
                    if (nll_out)     *nll_out += -logp;
                    if (nscored_out) (*nscored_out)++;
                }
            }
        } else if (pos >= n_prompt - 1 && produced < n_gen) {   /* emit next token */
            rmsnorm(x, as_f32(m, m->output_norm), E, eps, nx);
            if (matmul(m, m->output, nx, 1, E, V, lg)) goto done;
            int amax = 0;
            for (int j = 1; j < V; j++) if (lg[j] > lg[amax]) amax = j;
            seq[n_prompt + produced] = amax;
            produced++; n = n_prompt + produced;
            if ((eos_id >= 0 && amax == eos_id) || produced >= n_gen) break;
        }
    }
    rc = n;
done:
    free(kcb); free(vcb); free(kdec); free(vdec);
    free(kc); free(vc); free(x); free(nx); free(q); free(knew); free(vnew);
    free(ao); free(ap); free(gg); free(up); free(dn); free(sc); free(lg);
    free(recallR); free(projk);
    free(ring2k); free(ring2v);
    free(stgK); free(stgV); free(stg_stamp); free(stg_slot); free(stg_pos); free(ri_all); free(m_all);
    if (r2s_io) ring2_disk_scratch_free(r2s_io);
    if (r2disk) {
        unsigned long long nr = 0; double rs = 0; ring2_disk_stats(r2disk, &nr, &rs);
        fprintf(stderr, "    [ring2-disk] %llu blocking reads, %.3f s total, %.2f us/read avg\n",
                nr, rs, nr ? rs * 1e6 / (double)nr : 0.0);
        ring2_disk_close(r2disk);
    }
    return rc;
}

int qwen3_generate_kv(const qwen3_model *m, int32_t *seq, int n_prompt, int n_gen,
                      int eos_id) {
    return generate_kv_impl(m, seq, n_prompt, n_gen, eos_id, /*ppl_mode=*/0, 0, NULL, NULL);
}

int qwen3_ppl_decode(const qwen3_model *m, int32_t *toks, int n_toks, int n_warm,
                     double *ppl, long *n_scored) {
    if (!m || !toks || n_toks < 4 || !ppl) return 1;
    if (n_warm < 1) n_warm = 1;
    if (n_warm > n_toks - 2) n_warm = n_toks - 2;
    double nll = 0.0; long ns = 0;
    int rc = generate_kv_impl(m, toks, n_toks, /*n_gen=*/0, /*eos=*/-1,
                              /*ppl_mode=*/1, n_warm, &nll, &ns);
    if (rc < 0 || ns <= 0) return 1;
    *ppl = exp(nll / (double)ns);
    if (n_scored) *n_scored = ns;
    return 0;
}
