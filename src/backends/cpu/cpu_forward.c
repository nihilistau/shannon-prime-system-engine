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

static int mtp_argmax(const float *row, int n) {
    int bi = 0; float bv = row[0];
    for (int i = 1; i < n; i++) if (row[i] > bv) { bv = row[i]; bi = i; }
    return bi;
}

/* ── MTP (T8): persistent-KV batched append-forward ─────────────────────────
 * Append `nb` tokens at absolute positions [basePos .. basePos+nb-1] into the
 * caller's f32 K/V cache. K/V projections write straight into the cache slots
 * (which are contiguous: KC + basePos*KVD is exactly nb*KVD floats), so the new
 * post-RoPE K/V land in place and attention reads the unified [0..pos] window
 * the same way the one-token path does — argmax is bit-identical to greedy. */
int qwen3_mtp_forward(const qwen3_model *m, const int32_t *batch, int nb,
                      int basePos, float *kc, float *vc, int cap, float *logits) {
    const qwen3_config *c = &m->cfg;
    const int E = (int)c->n_embd, FF = (int)c->n_ff, HD = (int)c->head_dim;
    const int NH = (int)c->n_head, NKV = (int)c->n_head_kv;
    const int QD = NH * HD, KVD = NKV * HD, group = NH / NKV, V = (int)c->n_vocab;
    const float eps = c->rms_eps, base = c->rope_freq_base;
    const float ascale = 1.0f / sqrtf((float)HD);
    if (nb < 1 || basePos < 0 || basePos + nb > cap) return 1;

    int rc = 1;
    float *x  = (float *)malloc((size_t)nb * E * sizeof(float));
    float *nx = (float *)malloc((size_t)nb * E * sizeof(float));
    float *q  = (float *)malloc((size_t)nb * QD * sizeof(float));
    float *ao = (float *)malloc((size_t)nb * QD * sizeof(float));
    float *ap = (float *)malloc((size_t)nb * E * sizeof(float));
    float *gg = (float *)malloc((size_t)nb * FF * sizeof(float));
    float *up = (float *)malloc((size_t)nb * FF * sizeof(float));
    float *dn = (float *)malloc((size_t)nb * E * sizeof(float));
    if (!x || !nx || !q || !ao || !ap || !gg || !up || !dn) goto done;

    for (int t = 0; t < nb; t++)
        if (embed_row(m, batch[t], E, x + (size_t)t * E)) goto done;

    for (uint32_t L = 0; L < c->n_layers; L++) {
        const qwen3_layer *ly = &m->layers[L];
        float *KC = kc + (size_t)L * cap * KVD;
        float *VC = vc + (size_t)L * cap * KVD;

        for (int t = 0; t < nb; t++)
            rmsnorm(x + (size_t)t * E, as_f32(m, ly->attn_norm), E, eps, nx + (size_t)t * E);

        if (matmul(m, ly->attn_q, nx, nb, E, QD, q)) goto done;
        /* K/V projections write straight into the cache slots [basePos..basePos+nb-1] */
        if (matmul(m, ly->attn_k, nx, nb, E, KVD, KC + (size_t)basePos * KVD)) goto done;
        if (matmul(m, ly->attn_v, nx, nb, E, KVD, VC + (size_t)basePos * KVD)) goto done;

        const float *qn = as_f32(m, ly->attn_q_norm), *kn = as_f32(m, ly->attn_k_norm);
        for (int t = 0; t < nb; t++) {
            const int pos = basePos + t;
            for (int h = 0; h < NH; h++) {
                float *qh = q + (size_t)t * QD + (size_t)h * HD;
                rmsnorm_head(qh, qn, HD, eps); rope_neox(qh, HD, pos, base);
            }
            for (int h = 0; h < NKV; h++) {
                float *kh = KC + (size_t)pos * KVD + (size_t)h * HD;
                rmsnorm_head(kh, kn, HD, eps); rope_neox(kh, HD, pos, base);
            }
        }
        /* attention: each batch token attends the unified cache [0..pos] (win=-1).
         * parallel over the nb*NH (token,head) pairs — each writes a distinct ao
         * slice with its own score scratch, so it is bit-identical to serial. */
        {
            int th;
            #pragma omp parallel
            {
                float *sc = (float *)malloc((size_t)(basePos + nb) * sizeof(float));
                if (sc) {
                    #pragma omp for
                    for (th = 0; th < nb * NH; th++) {
                        int t = th / NH, h = th % NH, kvh = h / group, pos = basePos + t;
                        const float *qh = q + (size_t)t * QD + (size_t)h * HD;
                        float *out = ao + (size_t)t * QD + (size_t)h * HD;
                        kernels_attn_head(qh, KC, VC, pos, KVD, kvh, HD, ascale, -1, sc, out);
                    }
                }
                free(sc);
            }
        }

        if (matmul(m, ly->attn_output, ao, nb, QD, E, ap)) goto done;
        for (size_t i = 0; i < (size_t)nb * E; i++) x[i] += ap[i];

        for (int t = 0; t < nb; t++)
            rmsnorm(x + (size_t)t * E, as_f32(m, ly->ffn_norm), E, eps, nx + (size_t)t * E);
        if (matmul(m, ly->ffn_gate, nx, nb, E, FF, gg)) goto done;
        if (matmul(m, ly->ffn_up,   nx, nb, E, FF, up)) goto done;
        for (size_t i = 0; i < (size_t)nb * FF; i++) { float gv = gg[i]; gg[i] = gv / (1.0f + expf(-gv)) * up[i]; }
        if (matmul(m, ly->ffn_down, gg, nb, FF, E, dn)) goto done;
        for (size_t i = 0; i < (size_t)nb * E; i++) x[i] += dn[i];
    }

    for (int t = 0; t < nb; t++)
        rmsnorm(x + (size_t)t * E, as_f32(m, m->output_norm), E, eps, nx + (size_t)t * E);
    if (matmul(m, m->output, nx, nb, E, V, logits)) goto done;
    rc = 0;
done:
    free(x); free(nx); free(q); free(ao); free(ap); free(gg); free(up); free(dn);
    return rc;
}

/* prompt-lookup: rightmost match of the NG-gram ending at h[hn-1]; draft the
 * up-to-K tokens that followed it. Returns draft length (0 if no match). */
static int mtp_draft_lookup(const int32_t *h, int hn, int NG, int K, int32_t *draft) {
    if (K <= 0 || hn < NG + 1) return 0;
    for (int j = hn - NG - 1; j >= 0; j--) {
        int mt = 1;
        for (int g = 0; g < NG; g++) if (h[j + g] != h[hn - NG + g]) { mt = 0; break; }
        if (mt) {
            int Kd = 0;
            for (int d = 0; d < K && j + NG + d < hn; d++) draft[Kd++] = h[j + NG + d];
            return Kd;
        }
    }
    return 0;
}

int qwen3_mtp_decode(const qwen3_model *m, int32_t *seq, int n_prompt, int n_gen,
                     int eos_id, int K, int NG,
                     long *out_forwards, long *out_accept_sum, long *out_accept_steps) {
    if (!m || !seq || n_prompt < 1 || n_gen < 0) return -1;
    if (K < 0) K = 0;
    if (NG < 1) NG = 1;
    const qwen3_config *c = &m->cfg;
    const int V = (int)c->n_vocab, KVD = (int)c->n_head_kv * (int)c->head_dim;
    const int cap = n_prompt + n_gen + K + 8;
    const int maxnb = (n_prompt > K + 1) ? n_prompt : (K + 1);

    float *kc = (float *)malloc((size_t)c->n_layers * cap * KVD * sizeof(float));
    float *vc = (float *)malloc((size_t)c->n_layers * cap * KVD * sizeof(float));
    float *lg = (float *)malloc((size_t)maxnb * (size_t)V * sizeof(float));
    int32_t *batch = (int32_t *)malloc((size_t)(K + 1) * sizeof(int32_t));
    int32_t *draft = (int32_t *)malloc((size_t)(K + 1) * sizeof(int32_t));
    long fwd = 0, acc_sum = 0, acc_steps = 0;
    int produced = 0, n = n_prompt, rc_n = -1;
    if (!kc || !vc || !lg || !batch || !draft) goto done;

    /* prefill: cache [0..n_prompt-1], first generated token = argmax of last row */
    if (qwen3_mtp_forward(m, seq, n_prompt, 0, kc, vc, cap, lg)) goto done;
    fwd++;
    int C = n_prompt;
    int32_t cur = (int32_t)mtp_argmax(lg + (size_t)(n_prompt - 1) * V, V);

    while (produced < n_gen) {
        seq[n_prompt + produced] = cur; produced++; n = n_prompt + produced;
        if ((eos_id >= 0 && cur == eos_id) || produced >= n_gen) break;

        int hn = n_prompt + produced;                 /* history incl. cur at hn-1 */
        int Kd = mtp_draft_lookup(seq, hn, NG, K, draft);
        batch[0] = cur; for (int d = 0; d < Kd; d++) batch[1 + d] = draft[d];
        if (qwen3_mtp_forward(m, batch, Kd + 1, C, kc, vc, cap, lg)) goto done;
        fwd++;

        int na = 0;                                    /* longest argmax-matching prefix */
        for (int i = 0; i < Kd; i++) {
            if ((int32_t)mtp_argmax(lg + (size_t)i * V, V) == draft[i]) na++; else break;
        }
        for (int i = 0; i < na && produced < n_gen; i++) {
            seq[n_prompt + produced] = draft[i]; produced++; n = n_prompt + produced;
            if (eos_id >= 0 && draft[i] == eos_id) { acc_sum += na; acc_steps++; goto stop; }
        }
        acc_sum += na; acc_steps++;
        cur = (int32_t)mtp_argmax(lg + (size_t)na * V, V);   /* corrected/next token */
        C += na + 1;
    }
stop:
    rc_n = n;
done:
    if (out_forwards)     *out_forwards     = fwd;
    if (out_accept_sum)   *out_accept_sum   = acc_sum;
    if (out_accept_steps) *out_accept_steps = acc_steps;
    free(kc); free(vc); free(lg); free(batch); free(draft);
    return rc_n;
}
