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

        /* GQA causal attention */
        for (int t = 0; t < n_tok; t++) {
            for (int h = 0; h < NH; h++) {
                int kvh = h / group;
                const float *qh = q + (size_t)t * QD + (size_t)h * HD;
                if (g_ntt_attn)
                    for (int i = 0; i < HD; i++) qi[i] = (int32_t)lrintf(qh[i] * (float)SP_NTT_ATTN_SCALE);
                float maxs = -INFINITY;
                for (int s = 0; s <= t; s++) {
                    const float *kh = k + (size_t)s * KVD + (size_t)kvh * HD;
                    float d;
                    if (g_ntt_attn) {
                        for (int i = 0; i < HD; i++) ki[i] = (int32_t)lrintf(kh[i] * (float)SP_NTT_ATTN_SCALE);
                        int64_t ip = sp_pr_inner(pr, qi, ki);   /* exact <q_int,k_int> */
                        d = (float)((double)ip / (SP_NTT_ATTN_SCALE * SP_NTT_ATTN_SCALE)) * ascale;
                    } else {
                        float acc = 0.0f;
                        for (int i = 0; i < HD; i++) acc += qh[i] * kh[i];
                        d = acc * ascale;
                    }
                    sc[s] = d;
                    if (d > maxs) maxs = d;
                }
                float sum = 0.0f;
                for (int s = 0; s <= t; s++) { sc[s] = expf(sc[s] - maxs); sum += sc[s]; }
                float inv = 1.0f / sum;
                float *out = ao + (size_t)t * QD + (size_t)h * HD;
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
int qwen3_generate_kv(const qwen3_model *m, int32_t *seq, int n_prompt, int n_gen,
                      int eos_id) {
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

    for (int pos = 0; pos < P; pos++) {
        int tok = seq[pos];
        if (embed_row(m, tok, E, x)) goto done;

        for (uint32_t L = 0; L < c->n_layers; L++) {
            const qwen3_layer *ly = &m->layers[L];
            rmsnorm(x, as_f32(m, ly->attn_norm), E, eps, nx);
            if (matmul(m, ly->attn_q, nx, 1, E, QD, q))   goto done;
            if (matmul(m, ly->attn_k, nx, 1, E, KVD, knew)) goto done;
            if (matmul(m, ly->attn_v, nx, 1, E, KVD, vnew)) goto done;

            const float *qn = as_f32(m, ly->attn_q_norm), *kn = as_f32(m, ly->attn_k_norm);
            for (int h = 0; h < NH;  h++) { float *qh = q    + (size_t)h * HD; rmsnorm_head(qh, qn, HD, eps); rope_neox(qh, HD, pos, base); }
            for (int h = 0; h < NKV; h++) { float *kh = knew + (size_t)h * HD; rmsnorm_head(kh, kn, HD, eps); rope_neox(kh, HD, pos, base); }
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

            for (int h = 0; h < NH; h++) {                     /* attention over cached [0,pos] */
                int kvh = h / group;
                const float *qh = q + (size_t)h * HD;
                float maxs = -INFINITY;
                for (int s = 0; s <= pos; s++) {
                    const float *kh = KC + (size_t)s * KVD + (size_t)kvh * HD;
                    float acc = 0.0f;
                    for (int i = 0; i < HD; i++) acc += qh[i] * kh[i];
                    float d = acc * ascale; sc[s] = d; if (d > maxs) maxs = d;
                }
                float sum = 0.0f;
                for (int s = 0; s <= pos; s++) { sc[s] = expf(sc[s] - maxs); sum += sc[s]; }
                float inv = 1.0f / sum;
                float *out = ao + (size_t)h * HD;
                for (int i = 0; i < HD; i++) out[i] = 0.0f;
                for (int s = 0; s <= pos; s++) {
                    float w = sc[s] * inv;
                    const float *vh = VC + (size_t)s * KVD + (size_t)kvh * HD;
                    for (int i = 0; i < HD; i++) out[i] += w * vh[i];
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

        if (pos >= n_prompt - 1 && produced < n_gen) {         /* emit next token */
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
    return rc;
}
