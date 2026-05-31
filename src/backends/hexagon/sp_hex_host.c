/* sp_hex_host.c — Phase 2-HX host orchestration (aarch64, on the phone).
 *
 * gemma3_forward_hexagon: build the per-row Q8 weight blob from the engine's
 * arena once (cached by model pointer), do the embedding lookup + tied LM head
 * HOST-side (reusing the engine's embed_row/matmul), and run the 26 transformer
 * layers + final RMSNorm on the cDSP via the sp_hex FastRPC forward. Scalar f32
 * on the DSP for HX.3a; gated == the on-phone CPU Q8 PPL. Recreated fresh.
 */
#define _CRT_SECURE_NO_WARNINGS
#include "sp_engine/hexagon_backend.h"
#include "sp_engine/kernels.h"        /* embed_row, as_f32, matmul, sp_kernels_read_env */
#include "sp_engine/arena.h"          /* sp_arena_find, sp_arena_tensor */
#include "sp/frobenius_lift.h"        /* sp_frob_packed_tensor */
#include "sp_hex_layout.h"

#include "sp_hex.h"                   /* qaic: sp_hex_open/close/forward + sp_hex_URI */
#include "rpcmem.h"
#include "remote.h"
#include "AEEStdErr.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

void sp_set_error(const char *msg);

static struct {
    const qwen3_model *key;
    remote_handle64    h;
    unsigned char     *blob;       /* rpcmem Q8/f32 weight blob */
    size_t             blob_bytes;
    sp_hex_cfg         cfg;
    int                rpcmem_up;
} g_hx = { 0, (remote_handle64)-1, 0, 0, {0,0,0,0,0,0,0,0,0,0}, 0 };

static void hx_release(void) {
    if (g_hx.blob) rpcmem_free(g_hx.blob);
    if (g_hx.h != (remote_handle64)-1) sp_hex_close(g_hx.h);
    g_hx.blob = 0; g_hx.h = (remote_handle64)-1; g_hx.key = 0; g_hx.blob_bytes = 0;
}

static void hx_cfg_from(const qwen3_model *m, sp_hex_cfg *c) {
    c->n_layers = (int)m->cfg.n_layers;  c->n_embd = (int)m->cfg.n_embd;
    c->n_ff = (int)m->cfg.n_ff;          c->head_dim = (int)m->cfg.head_dim;
    c->n_head = (int)m->cfg.n_head;      c->n_head_kv = (int)m->cfg.n_head_kv;
    c->sliding_window = (int)m->cfg.sliding_window;
    c->eps = m->cfg.rms_eps;
    c->rope_global = m->cfg.rope_freq_base;   /* gemma3 global layers */
    c->rope_local = 10000.0f;                 /* gemma3 local/SWA layers */
}

/* copy a Q8 arena weight (codes + per-row scales) into the blob at `dst`.
 *
 * HX.3b-alpha-v2 NOTE: this packer is bit-identical to HX.3b. The per-row
 * weight-sum lookup table for the v2 single-vrmpy kernel is populated on the
 * DSP side via a session cache (sp_hex_imp.c::hx_rsum_get) on the first
 * sp_hex_forward call. Reason: rebuilding sp-daemon-wire-hex from this
 * worktree requires building libsp_hex_daemon_backend.a + cross-compiling the
 * Rust daemon for aarch64-android, which is out of scope for an incremental-
 * lift sprint. The DSP-side cache lives across forward calls within a session
 * (one-time amortized cost; subsequent prefills run on the lookup-only path). */
static int hx_pack_q8(unsigned char *dst, const qwen3_model *m, const gguf_tensor *W) {
    const sp_arena_tensor *at = m->arena ? sp_arena_find(m->arena, W->name) : 0;
    if (!at) { sp_set_error("hexagon: matmul weight not in Q8 arena (need SP_ARENA=q8)"); return 1; }
    const sp_frob_packed_tensor *pt = &at->pt;
    int out = pt->rows, in = pt->cols;
    memcpy(dst, pt->codes, (size_t)out * in);                       /* int8 codes (row j = j*in) */
    float *scales = (float *)(dst + sp_hex_align((size_t)out * in));
    memcpy(scales, pt->row_scale, (size_t)out * sizeof(float));
    return 0;
}
static void hx_pack_f32(unsigned char *dst, const qwen3_model *m, const gguf_tensor *t, int n) {
    memcpy(dst, as_f32(m, t), (size_t)n * sizeof(float));
}

static int hx_build(const qwen3_model *m) {
    hx_release();
    hx_cfg_from(m, &g_hx.cfg);
    sp_hex_cfg *c = &g_hx.cfg;

    rpcmem_init(); g_hx.rpcmem_up = 1;
    if (remote_session_control) {
        struct remote_rpc_control_unsigned_module u;
        u.domain = CDSP_DOMAIN_ID; u.enable = 1;
        remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, (void *)&u, sizeof(u));
    }
    int rc = sp_hex_open(sp_hex_URI CDSP_DOMAIN, &g_hx.h);
    if (rc) { sp_set_error("hexagon: sp_hex_open failed"); hx_release(); return 1; }

    g_hx.blob_bytes = sp_hex_blob_bytes(c);
    g_hx.blob = (unsigned char *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, g_hx.blob_bytes);
    if (!g_hx.blob) { sp_set_error("hexagon: weight blob rpcmem_alloc failed"); hx_release(); return 1; }

    for (int L = 0; L < c->n_layers; L++) {
        const qwen3_layer *ly = &m->layers[L];
        unsigned char *b = g_hx.blob;
        #define OFF(kind) (b + sp_hex_weight_off(c, L, (kind)))
        hx_pack_f32(OFF(SP_HEX_ATTN_NORM), m, ly->attn_norm,      c->n_embd);
        hx_pack_f32(OFF(SP_HEX_Q_NORM),    m, ly->attn_q_norm,    c->head_dim);
        hx_pack_f32(OFF(SP_HEX_K_NORM),    m, ly->attn_k_norm,    c->head_dim);
        hx_pack_f32(OFF(SP_HEX_FFN_NORM),  m, ly->ffn_norm,       c->n_embd);
        hx_pack_f32(OFF(SP_HEX_POST_ATTN), m, ly->post_attn_norm, c->n_embd);
        hx_pack_f32(OFF(SP_HEX_POST_FFW),  m, ly->post_ffw_norm,  c->n_embd);
        if (hx_pack_q8(OFF(SP_HEX_WQ),    m, ly->attn_q)   ||
            hx_pack_q8(OFF(SP_HEX_WK),    m, ly->attn_k)   ||
            hx_pack_q8(OFF(SP_HEX_WV),    m, ly->attn_v)   ||
            hx_pack_q8(OFF(SP_HEX_WO),    m, ly->attn_output) ||
            hx_pack_q8(OFF(SP_HEX_WGATE), m, ly->ffn_gate) ||
            hx_pack_q8(OFF(SP_HEX_WUP),   m, ly->ffn_up)   ||
            hx_pack_q8(OFF(SP_HEX_WDOWN), m, ly->ffn_down)) { hx_release(); return 1; }
        #undef OFF
    }
    hx_pack_f32(g_hx.blob + sp_hex_weight_off(c, c->n_layers, 0), m, m->output_norm, c->n_embd);

    g_hx.key = m;
    fprintf(stderr, "    [hexagon] weight blob built: %zu bytes (Q8 arena + f32 norms)\n", g_hx.blob_bytes);
    return 0;
}

int gemma3_forward_hexagon(const qwen3_model *m, const int32_t *tokens,
                                      int n_tok, float *logits) {
    if (!m || m->cfg.arch != SP_ARCH_GEMMA3) { sp_set_error("hexagon: not a gemma3 model"); return 1; }
    if (!m->arena) { sp_set_error("hexagon: needs SP_ARENA=q8"); return 1; }
    sp_kernels_read_env();   /* head matmul + embed read the (cleared) f32 path */

    if (g_hx.key != m) { if (hx_build(m)) return 1; }
    const sp_hex_cfg *c = &g_hx.cfg;
    const int E = c->n_embd, V = (int)m->cfg.n_vocab;
    const float embscale = sqrtf((float)E);

    size_t sc_elems = sp_hex_scratch_elems(c, n_tok);
    float *x       = (float *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, (size_t)n_tok * E * sizeof(float));
    float *scratch = (float *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, sc_elems * sizeof(float));
    float *hidden  = (float *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, (size_t)n_tok * E * sizeof(float));
    int rc = 1;
    if (!x || !scratch || !hidden) { sp_set_error("hexagon: forward rpcmem_alloc failed"); goto done; }

    /* embedding lookup + ×sqrt(n_embd), host-side (the f32 embd stays off the DSP) */
    for (int t = 0; t < n_tok; t++) {
        if (embed_row(m, tokens[t], E, x + (size_t)t * E)) { sp_set_error("hexagon: embed_row"); goto done; }
        float *xt = x + (size_t)t * E;
        for (int i = 0; i < E; i++) xt[i] *= embscale;
    }

    rc = sp_hex_forward(g_hx.h, c->n_layers, E, c->n_ff, c->head_dim, c->n_head,
                        c->n_head_kv, c->sliding_window, c->eps, c->rope_global, c->rope_local,
                        n_tok, x, n_tok * E, g_hx.blob, (int)g_hx.blob_bytes,
                        scratch, (int)sc_elems, hidden, n_tok * E);
    if (rc) { sp_set_error("hexagon: sp_hex_forward failed"); goto done; }

    /* tied LM head, host-side: logits[t] = output^T . hidden[t]  (f32, not in arena) */
    if (matmul(m, m->output, hidden, n_tok, E, V, logits)) { sp_set_error("hexagon: head matmul"); rc = 1; goto done; }
    rc = 0;

done:
    if (x) rpcmem_free(x);
    if (scratch) rpcmem_free(scratch);
    if (hidden) rpcmem_free(hidden);
    return rc;
}

void sp_hexagon_model_release(const qwen3_model *m) {
    if (g_hx.key == m) hx_release();
}
