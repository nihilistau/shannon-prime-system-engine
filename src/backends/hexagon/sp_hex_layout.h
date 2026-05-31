/* sp_hex_layout.h — Phase 2-HX host<->DSP contract for the layers-on-DSP forward.
 *
 * Included by BOTH the host (sp_hex_host.c, aarch64) and the cDSP (sp_hex_imp.c,
 * hexagon-clang) so the weight-blob byte layout is computed by identical code on
 * both sides. The host packs the per-layer weights into ONE rpcmem blob in
 * exactly this order; the DSP reads them back via the same offset walk.
 *
 * Split (memory-safe on the phone): the embedding lookup (×√n_embd) and the tied
 * LM head matmul run HOST-side (keeps the 1.2 GB f32 embd off the DSP). The DSP
 * runs the 26 transformer layers + the final output RMSNorm on the per-row
 * Frobenius Q8 arena weights (~574 MB) + the f32 norm vectors. Forward I/O is the
 * embedded hidden state in (n_tok*n_embd f32) and the final-normed hidden out
 * (n_tok*n_embd f32) — small buffers, no 176 MB logits round-trip.
 *
 * Per-layer matmul weights are per-row Q8 (ggml [ne0=in, ne1=out]); a Q8 weight
 * [out,in] is `out*in` int8 codes (padded to 4 B) + `out` f32 per-row scales,
 * dequant w[j][i] = code*scale[j]/127 — the SP_FROB_ARENA_LAYOUT_VERSION=1 format.
 * Norm vectors are raw f32. Every tensor block is 128-byte aligned (HVX-ready for
 * HX.3b). The DSP forward is scalar f32 for HX.3a; HX.3b swaps the matmul to qf32.
 */
#ifndef SP_HEX_LAYOUT_H
#define SP_HEX_LAYOUT_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    int32_t n_layers, n_embd, n_ff, head_dim, n_head, n_head_kv, sliding_window;
    float   eps, rope_global, rope_local;
} sp_hex_cfg;

/* Per-layer weight kinds, in blob order. Q8 = matmul weight; F32 = norm vector. */
enum {
    SP_HEX_ATTN_NORM = 0,  /* f32 [n_embd]              */
    SP_HEX_Q_NORM,         /* f32 [head_dim]            */
    SP_HEX_K_NORM,         /* f32 [head_dim]            */
    SP_HEX_FFN_NORM,       /* f32 [n_embd]              */
    SP_HEX_POST_ATTN,      /* f32 [n_embd]              */
    SP_HEX_POST_FFW,       /* f32 [n_embd]              */
    SP_HEX_WQ,             /* Q8  [n_head*head_dim, n_embd]    */
    SP_HEX_WK,             /* Q8  [n_head_kv*head_dim, n_embd] */
    SP_HEX_WV,             /* Q8  [n_head_kv*head_dim, n_embd] */
    SP_HEX_WO,             /* Q8  [n_embd, n_head*head_dim]    */
    SP_HEX_WGATE,          /* Q8  [n_ff, n_embd]        */
    SP_HEX_WUP,            /* Q8  [n_ff, n_embd]        */
    SP_HEX_WDOWN,          /* Q8  [n_embd, n_ff]        */
    SP_HEX_KINDS
};
/* output_norm (f32 [n_embd]) follows all layers at slot n_layers*SP_HEX_KINDS. */

#define SP_HEX_ALIGN 128u
static inline size_t sp_hex_align(size_t x) { return (x + (SP_HEX_ALIGN - 1)) & ~(size_t)(SP_HEX_ALIGN - 1); }

/* bytes of a 128-aligned Q8 weight block: padded int8 codes + f32 row scales.
 *
 * HX.3b-alpha-v2 NOTE: an earlier variant of this sprint added a per-row
 * int32 row_sum tail to the block layout. That change required a coordinated
 * rebuild of the host-side daemon binary (sp-daemon-wire-hex links sp_hex_host.c
 * via libsp_hex_daemon_backend.a) AND the cDSP skel. Since the operator's worktree
 * is configured for skel-only rebuilds (math-core submodule intentionally empty
 * per HX.3b precedent), the precomputed-row_sum table was moved to a DSP-side
 * session cache (see hx_rsum_get in sp_hex_imp.c) populated on the first
 * sp_hex_forward call and reused on subsequent calls. The blob layout below
 * stays bit-identical to HX.3b. */
static inline size_t sp_hex_q8_bytes(int out, int in) {
    size_t codes = sp_hex_align((size_t)out * (size_t)in);   /* int8 codes, padded */
    return sp_hex_align(codes + (size_t)out * sizeof(float)); /* + per-row scales   */
}
static inline size_t sp_hex_f32_bytes(int n) { return sp_hex_align((size_t)n * sizeof(float)); }

/* dims of weight `kind` in a layer: rows (out) x cols (in); cols<0 marks an f32
 * norm vector of length `rows`. */
static inline void sp_hex_kind_dims(const sp_hex_cfg *c, int kind, int *rows, int *cols) {
    int E = c->n_embd, FF = c->n_ff, HD = c->head_dim;
    int QD = c->n_head * HD, KVD = c->n_head_kv * HD;
    switch (kind) {
        case SP_HEX_ATTN_NORM: case SP_HEX_FFN_NORM:
        case SP_HEX_POST_ATTN: case SP_HEX_POST_FFW: *rows = E;  *cols = -1; break;
        case SP_HEX_Q_NORM: case SP_HEX_K_NORM:      *rows = HD; *cols = -1; break;
        case SP_HEX_WQ:    *rows = QD;  *cols = E;  break;
        case SP_HEX_WK:    *rows = KVD; *cols = E;  break;
        case SP_HEX_WV:    *rows = KVD; *cols = E;  break;
        case SP_HEX_WO:    *rows = E;   *cols = QD; break;
        case SP_HEX_WGATE: *rows = FF;  *cols = E;  break;
        case SP_HEX_WUP:   *rows = FF;  *cols = E;  break;
        case SP_HEX_WDOWN: *rows = E;   *cols = FF; break;
        default:           *rows = 0;   *cols = 0;  break;
    }
}

static inline size_t sp_hex_kind_bytes(const sp_hex_cfg *c, int kind) {
    int rows, cols; sp_hex_kind_dims(c, kind, &rows, &cols);
    return (cols < 0) ? sp_hex_f32_bytes(rows) : sp_hex_q8_bytes(rows, cols);
}

/* byte offset of (layer, kind) in the blob; layer == n_layers, kind 0 = output_norm. */
static inline size_t sp_hex_weight_off(const sp_hex_cfg *c, int layer, int kind) {
    size_t off = 0;
    for (int L = 0; L < layer; L++)
        for (int k = 0; k < SP_HEX_KINDS; k++) off += sp_hex_kind_bytes(c, k);
    if (layer < c->n_layers)
        for (int k = 0; k < kind; k++) off += sp_hex_kind_bytes(c, k);
    return off;
}

/* total weight-blob bytes (all layers + output_norm). */
static inline size_t sp_hex_blob_bytes(const sp_hex_cfg *c) {
    size_t off = sp_hex_weight_off(c, c->n_layers, 0);
    return off + sp_hex_f32_bytes(c->n_embd);   /* output_norm */
}

/* scratch f32 elems the DSP forward needs for n_tok (carved from a host buffer):
 * resid[n_tok*E] (the in x is read-only, copied here) + nx[n_tok*E] +
 * q[n_tok*QD] + k[n_tok*KVD] + v[n_tok*KVD] + ao[n_tok*QD] + ap[n_tok*E] +
 * g[n_tok*FF] + up[n_tok*FF] + dn[n_tok*E] + sc[n_tok]. */
static inline size_t sp_hex_scratch_elems(const sp_hex_cfg *c, int n_tok) {
    int E = c->n_embd, FF = c->n_ff, HD = c->head_dim;
    int QD = c->n_head * HD, KVD = c->n_head_kv * HD;
    return (size_t)n_tok * (E + E + QD + KVD + KVD + QD + E + FF + FF + E + 1);
}

#endif /* SP_HEX_LAYOUT_H */
