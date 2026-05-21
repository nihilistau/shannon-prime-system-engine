/* model.c — Qwen3 config + weight binding + dequant. See sp_engine/model.h. */
#define _CRT_SECURE_NO_WARNINGS   /* getenv (SP_ARENA) is fine here (MSVC C4996) */
#include "sp_engine/model.h"
#include "sp_engine/arena.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

float sp_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t man  = h & 0x3FFu;
    uint32_t f;
    if (exp == 0) {
        if (man == 0) { f = sign; }                 /* +/-0 */
        else {                                      /* subnormal -> normalize */
            exp = 127u - 15u + 1u;
            while ((man & 0x400u) == 0) { man <<= 1; exp--; }
            man &= 0x3FFu;
            f = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 0x1Fu) {                       /* inf / nan */
        f = sign | 0x7F800000u | (man << 13);
    } else {                                         /* normal */
        f = sign | ((exp - 15u + 127u) << 23) | (man << 13);
    }
    float out; memcpy(&out, &f, 4); return out;
}

uint16_t sp_f32_to_f16(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t e8   = (x >> 23) & 0xFFu;
    uint32_t man  = x & 0x7FFFFFu;
    if (e8 == 0xFFu)                                /* inf / nan */
        return (uint16_t)(sign | 0x7C00u | (man ? 0x200u : 0u));
    int32_t exp = (int32_t)e8 - 127 + 15;           /* rebias to half */
    if (exp >= 0x1F) return (uint16_t)(sign | 0x7C00u);   /* overflow -> inf */
    if (exp <= 0) {                                 /* subnormal / underflow */
        if (exp < -10) return (uint16_t)sign;       /* magnitude too small -> 0 */
        man |= 0x800000u;                           /* restore implicit 1 */
        int shift = 14 - exp;                       /* shift in [14,24] */
        uint32_t half = man >> shift;
        uint32_t rem  = man & ((1u << shift) - 1u);
        uint32_t mid  = 1u << (shift - 1);
        if (rem > mid || (rem == mid && (half & 1u))) half++;  /* nearest-even */
        return (uint16_t)(sign | half);
    }
    uint16_t h   = (uint16_t)(sign | ((uint32_t)exp << 10) | (man >> 13));
    uint32_t rem = man & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) h++;  /* carry ripples into exp */
    return h;
}

int sp_dequant_row(const void *src, uint32_t type, int n, float *dst) {
    if (n < 0) return 1;
    switch (type) {
        case GGML_T_F32:
            memcpy(dst, src, (size_t)n * sizeof(float));
            return 0;
        case GGML_T_F16: {
            const uint16_t *h = (const uint16_t *)src;
            for (int i = 0; i < n; i++) dst[i] = sp_f16_to_f32(h[i]);
            return 0;
        }
        case GGML_T_Q8_0: {
            /* block_q8_0 = { f16 d; int8 qs[32]; } = 34 bytes / 32 elems */
            if (n % 32 != 0) return 1;
            const uint8_t *p = (const uint8_t *)src;
            int nb = n / 32;
            for (int b = 0; b < nb; b++) {
                uint16_t d16; memcpy(&d16, p, 2);
                float d = sp_f16_to_f32(d16);
                const int8_t *qs = (const int8_t *)(p + 2);
                for (int i = 0; i < 32; i++) dst[b * 32 + i] = (float)qs[i] * d;
                p += 34;
            }
            return 0;
        }
        default:
            return 1;  /* unsupported (k-quants etc.) — add when needed */
    }
}

static const gguf_tensor *want(const gguf_ctx *g, const char *name) {
    return gguf_find_tensor(g, name);
}

qwen3_model *qwen3_load(const char *path) {
    gguf_ctx *g = gguf_open(path);
    if (!g) return NULL;

    const char *arch = gguf_get_str(g, "general.architecture");
    if (!arch || strcmp(arch, "qwen3") != 0) { gguf_close(g); return NULL; }

    qwen3_model *m = (qwen3_model *)calloc(1, sizeof(*m));
    if (!m) { gguf_close(g); return NULL; }
    m->gguf = g;
    qwen3_config *c = &m->cfg;

    uint64_t v;
    int ok = 1;
    ok &= gguf_get_u64(g, "qwen3.block_count", &v);            c->n_layers = (uint32_t)v;
    ok &= gguf_get_u64(g, "qwen3.embedding_length", &v);       c->n_embd   = (uint32_t)v;
    ok &= gguf_get_u64(g, "qwen3.feed_forward_length", &v);    c->n_ff     = (uint32_t)v;
    ok &= gguf_get_u64(g, "qwen3.attention.head_count", &v);   c->n_head   = (uint32_t)v;
    ok &= gguf_get_u64(g, "qwen3.attention.head_count_kv", &v);c->n_head_kv= (uint32_t)v;
    ok &= gguf_get_u64(g, "qwen3.attention.key_length", &v);   c->head_dim = (uint32_t)v;
    if (gguf_get_u64(g, "qwen3.context_length", &v)) c->context_length = (uint32_t)v;
    if (!gguf_get_f32(g, "qwen3.rope.freq_base", &c->rope_freq_base)) c->rope_freq_base = 1e6f;
    if (!gguf_get_f32(g, "qwen3.attention.layer_norm_rms_epsilon", &c->rms_eps)) c->rms_eps = 1e-6f;
    if (!ok) { qwen3_free(m); return NULL; }

    m->token_embd  = want(g, "token_embd.weight");
    m->output_norm = want(g, "output_norm.weight");
    m->output      = want(g, "output.weight");
    if (!m->token_embd || !m->output_norm) { qwen3_free(m); return NULL; }
    if (!m->output) { m->output = m->token_embd; c->tied_embedding = 1; }

    /* vocab from token_embd: dims = [n_embd, n_vocab] (ne0=embd, ne1=vocab) */
    if (m->token_embd->n_dims < 2 || m->token_embd->dims[0] != c->n_embd) { qwen3_free(m); return NULL; }
    c->n_vocab = (uint32_t)m->token_embd->dims[1];

    m->layers = (qwen3_layer *)calloc(c->n_layers, sizeof(qwen3_layer));
    if (!m->layers) { qwen3_free(m); return NULL; }

    char nm[96];
    for (uint32_t i = 0; i < c->n_layers; i++) {
        qwen3_layer *L = &m->layers[i];
        #define BIND(field, suffix) \
            do { snprintf(nm, sizeof nm, "blk.%u." suffix, i); L->field = want(g, nm); } while (0)
        BIND(attn_norm,   "attn_norm.weight");
        BIND(attn_q,      "attn_q.weight");
        BIND(attn_k,      "attn_k.weight");
        BIND(attn_v,      "attn_v.weight");
        BIND(attn_output, "attn_output.weight");
        BIND(attn_q_norm, "attn_q_norm.weight");   /* may be NULL */
        BIND(attn_k_norm, "attn_k_norm.weight");   /* may be NULL */
        BIND(ffn_norm,    "ffn_norm.weight");
        BIND(ffn_gate,    "ffn_gate.weight");
        BIND(ffn_up,      "ffn_up.weight");
        BIND(ffn_down,    "ffn_down.weight");
        #undef BIND
        if (!L->attn_norm || !L->attn_q || !L->attn_k || !L->attn_v || !L->attn_output ||
            !L->ffn_norm  || !L->ffn_gate || !L->ffn_up || !L->ffn_down) {
            qwen3_free(m); return NULL;
        }
    }
    /* QK-norm is present iff layer 0 carries it (uniform across layers). */
    m->cfg.has_qk_norm = (m->layers[0].attn_q_norm != NULL && m->layers[0].attn_k_norm != NULL);

    /* Packed-weight arena (roadmap §4.8), env-gated: SP_ARENA=q8|q4. Quantizes
     * the matmul weights once; the forward then lifts inline from the arena.
     * Default unset => NULL => the dequant-on-demand reference path (E_CPU_2). */
    {
        const char *e = getenv("SP_ARENA");
        if (e && (strcmp(e, "q8") == 0 || strcmp(e, "q4") == 0)) {
            int prec = (e[1] == '8') ? 8 : 4;
            const char *pe = getenv("SP_Q4_PROMOTE");
            float promote = pe ? (float)atof(pe) : 0.25f;
            m->arena = sp_arena_build(m, prec, promote);
            if (!m->arena) { qwen3_free(m); return NULL; }   /* fail loudly, no silent fallback */
        }
    }

    return m;
}

void qwen3_free(qwen3_model *m) {
    if (!m) return;
    sp_arena_free(m->arena);
    free(m->layers);
    if (m->gguf) gguf_close(m->gguf);
    free(m);
}
