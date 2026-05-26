/* sp_transcode.c — E_FMT_2 + E_FMT_3: GGUF -> .sp-model + .sp-tokenizer.
 * (PPT-LAT-SP-MODEL-v0 §9 + §7.) One-shot offline transcoder; not part of
 * libshannonprime.
 *
 * Per-tensor policy (§9):
 *   - matmul weights (attn q/k/v/o, ffn gate/up/down, LM head, token_embd):
 *     dequant to f32, re-quant into OK_Q8 (dtype 10) via sp_frob_pack_tensor
 *     (precision=8) -> per-row int8 codes + a sibling ".scale" tensor
 *     (FROBENIUS_SCALE_FP32, dtype 12). The codes bytes are exactly what the
 *     in-RAM arena packs (SP_FROB_ARENA_LAYOUT_VERSION=1), so the load path
 *     reconstructs a bit-identical arena.
 *   - norms (and any other tensor): copied as F32 (dequant F16->F32 if needed),
 *     so the loader's as_f32 path is a plain mmap+cast / owned-f32 copy.
 *
 * Data-region layout (§9 sibling adjacency): a weight's ".scale" tensor is
 * written immediately after it, no other tensor interposed. The tensor TABLE is
 * sorted by xxh64(name); the on-disk DATA order is parent-then-scale groups.
 *
 * Usage:
 *   sp_transcode <in.gguf> <out.sp-model> <out.sp-tokenizer> [--verify]
 */
#define _CRT_SECURE_NO_WARNINGS
#include "sp_engine/sp_model.h"
#include "sp_engine/gguf.h"
#include "sp_engine/model.h"     /* sp_dequant_row, sp_f16_to_f32 */
#include "sp/frobenius_lift.h"
#include "sp/sp_l1.h"            /* sp_arch_info, sp_precision (E_PARITY_3) */
#include "sp_hash.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ── a growable in-memory tensor we are about to emit ── */
typedef struct {
    char     name[80];
    uint32_t dtype_id;
    uint32_t n_dims;
    uint64_t dims[8];
    uint32_t block_size;
    uint8_t *bytes;          /* owned payload */
    uint64_t size_bytes;
    uint64_t data_off;       /* assigned at layout time (rel. to data region) */
} emit_tensor;

typedef struct { emit_tensor *t; int n, cap; } emit_list;

static emit_tensor *el_push(emit_list *L) {
    if (L->n == L->cap) {
        L->cap = L->cap ? L->cap * 2 : 64;
        L->t = (emit_tensor *)realloc(L->t, (size_t)L->cap * sizeof(emit_tensor));
        if (!L->t) { fprintf(stderr, "OOM tensor list\n"); exit(2); }
    }
    emit_tensor *e = &L->t[L->n++];
    memset(e, 0, sizeof *e);
    return e;
}
static void el_free(emit_list *L) { for (int i = 0; i < L->n; i++) free(L->t[i].bytes); free(L->t); }

/* ── GGUF row reader for the Q8 packer (dequant F32/F16/Q8_0 -> f32) ── */
typedef struct { const uint8_t *base; uint32_t type; size_t rb; int cols; } row_ctx;
static size_t ggml_row_bytes(uint32_t type, int n) {
    switch (type) {
        case GGML_T_F32:  return (size_t)n * 4;
        case GGML_T_F16:  return (size_t)n * 2;
        case GGML_T_Q8_0: return (size_t)(n / 32) * 34;
        default:          return 0;
    }
}
static int get_row(void *ctx, int j, float *dst) {
    const row_ctx *g = (const row_ctx *)ctx;
    return sp_dequant_row(g->base + (size_t)j * g->rb, g->type, g->cols, dst);
}

/* Add a matmul weight as OK_Q8 + ".scale" sibling (adjacent in the data region). */
static int add_q8(emit_list *L, const gguf_ctx *g, const gguf_tensor *W) {
    const uint8_t *base = (const uint8_t *)gguf_tensor_data(g, W);
    if (!base || W->n_dims < 2) { fprintf(stderr, "bad weight %s\n", W->name); return 1; }
    int cols = (int)W->dims[0], rows = (int)W->dims[1];
    size_t rb = ggml_row_bytes(W->type, cols);
    if (rb == 0) { fprintf(stderr, "unsupported src type for %s\n", W->name); return 1; }
    row_ctx ctx = { base, W->type, rb, cols };
    sp_frob_packed_tensor pt;
    if (sp_frob_pack_tensor(rows, cols, 8, 0.0f, get_row, &ctx, &pt, NULL)) {
        fprintf(stderr, "pack failed %s\n", W->name); return 1;
    }
    /* OK_Q8 codes (rows*cols int8, row-major) */
    emit_tensor *q = el_push(L);
    snprintf(q->name, sizeof q->name, "%s", W->name);
    q->dtype_id = SP_DT_OK_Q8; q->n_dims = 2; q->dims[0] = (uint64_t)cols; q->dims[1] = (uint64_t)rows;
    q->block_size = 1; q->size_bytes = (uint64_t)rows * cols;
    q->bytes = (uint8_t *)malloc(q->size_bytes ? q->size_bytes : 1);
    if (!q->bytes) { sp_frob_packed_free(&pt); return 1; }
    memcpy(q->bytes, pt.codes, q->size_bytes);
    /* ".scale" sibling (rows fp32) — pushed immediately after, so it is adjacent
     * in the data region after layout (§9). */
    emit_tensor *s = el_push(L);
    snprintf(s->name, sizeof s->name, "%s.scale", W->name);
    s->dtype_id = SP_DT_FROBENIUS_SCALE_FP32; s->n_dims = 1; s->dims[0] = (uint64_t)rows;
    s->block_size = 4; s->size_bytes = (uint64_t)rows * sizeof(float);
    s->bytes = (uint8_t *)malloc(s->size_bytes ? s->size_bytes : 1);
    if (!s->bytes) { sp_frob_packed_free(&pt); return 1; }
    memcpy(s->bytes, pt.row_scale, s->size_bytes);
    sp_frob_packed_free(&pt);
    return 0;
}

/* Add a tensor as F32 (norms etc.): dequant from F32/F16 to F32. */
static int add_f32(emit_list *L, const gguf_ctx *g, const gguf_tensor *W) {
    const uint8_t *base = (const uint8_t *)gguf_tensor_data(g, W);
    if (!base) { fprintf(stderr, "no data %s\n", W->name); return 1; }
    int n = (int)W->n_elements;
    emit_tensor *e = el_push(L);
    snprintf(e->name, sizeof e->name, "%s", W->name);
    e->dtype_id = SP_DT_F32; e->n_dims = W->n_dims;
    for (uint32_t d = 0; d < W->n_dims && d < 8; d++) e->dims[d] = W->dims[d];
    e->block_size = 4; e->size_bytes = (uint64_t)n * sizeof(float);
    e->bytes = (uint8_t *)malloc(e->size_bytes ? e->size_bytes : 1);
    if (!e->bytes) return 1;
    if (sp_dequant_row(base, W->type, n, (float *)e->bytes)) {
        fprintf(stderr, "dequant f32 failed %s\n", W->name); return 1;
    }
    return 0;
}

static uint64_t align_up(uint64_t x, uint64_t a) { return (x + a - 1) / a * a; }

/* qsort comparator: tensor entries by name_hash ascending. */
static int cmp_entry(const void *a, const void *b) {
    uint64_t ha = ((const sp_tensor_entry *)a)->name_hash;
    uint64_t hb = ((const sp_tensor_entry *)b)->name_hash;
    return (ha > hb) - (ha < hb);
}

/* ── tokenizer extraction (E_FMT_3, §7) ──
 * GGUF stores parsed tokens/scores/merges, not the original SentencePiece proto.
 * v0 .sp-tokenizer blob is a self-describing serialization of those arrays; the
 * L1 loader only needs a stable SHA-256 over the whole file. Blob layout (LE):
 *   u32 magic 'SPTB' | u32 type_id | u32 vocab | u32 n_merges
 *   tokens:  vocab * (u32 len, bytes)
 *   scores:  vocab * f32   (SPM only; 0 entries if absent)
 *   merges:  n_merges * (u32 len, bytes)
 */
static uint8_t *build_tok_blob(const gguf_ctx *g, uint64_t *blob_size_out,
                               uint32_t *type_id_out, uint32_t *vocab_out,
                               uint32_t bos_eos_pad_unk[4]) {
    const gguf_kv *tk = gguf_find_kv(g, "tokenizer.ggml.tokens");
    if (!tk || tk->type != GGUF_T_ARRAY || tk->arr_type != GGUF_T_STRING) {
        fprintf(stderr, "no tokenizer.ggml.tokens\n"); return NULL;
    }
    uint64_t nv = tk->arr_len;
    const char **tp = (const char **)malloc((size_t)nv * sizeof(char *));
    uint64_t    *tl = (uint64_t *)malloc((size_t)nv * sizeof(uint64_t));
    if (!tp || !tl || gguf_kv_str_array(g, tk, tp, tl, nv) != nv) {
        fprintf(stderr, "tokens read failed\n"); free(tp); free(tl); return NULL;
    }
    const char *model = gguf_get_str(g, "tokenizer.ggml.model");
    int spm = (model && strcmp(model, "llama") == 0);
    *type_id_out = spm ? SP_TOK_SENTENCEPIECE : SP_TOK_BPE_GPT2;

    const gguf_kv *sk = gguf_find_kv(g, "tokenizer.ggml.scores");
    const float *scores = (spm && sk && sk->type == GGUF_T_ARRAY &&
                           sk->arr_type == GGUF_T_FLOAT32 && sk->arr_len == nv)
                          ? (const float *)sk->arr_data : NULL;

    const gguf_kv *mk = gguf_find_kv(g, "tokenizer.ggml.merges");
    uint64_t nm = 0; const char **mp = NULL; uint64_t *ml = NULL;
    if (mk && mk->type == GGUF_T_ARRAY && mk->arr_type == GGUF_T_STRING && mk->arr_len > 0) {
        nm = mk->arr_len;
        mp = (const char **)malloc((size_t)nm * sizeof(char *));
        ml = (uint64_t *)malloc((size_t)nm * sizeof(uint64_t));
        if (!mp || !ml || gguf_kv_str_array(g, mk, mp, ml, nm) != nm) { nm = 0; }
    }

    uint64_t sz = 16;                            /* magic+type+vocab+n_merges */
    for (uint64_t i = 0; i < nv; i++) sz += 4 + tl[i];
    sz += scores ? nv * 4 : 0;
    for (uint64_t i = 0; i < nm; i++) sz += 4 + ml[i];

    uint8_t *blob = (uint8_t *)malloc(sz);
    if (!blob) { free(tp); free(tl); free(mp); free(ml); return NULL; }
    uint8_t *w = blob;
    #define PUT32(v) do { uint32_t _v=(uint32_t)(v); memcpy(w,&_v,4); w+=4; } while(0)
    PUT32(0x42545053u /*'SPTB'*/); PUT32(*type_id_out); PUT32((uint32_t)nv); PUT32((uint32_t)nm);
    for (uint64_t i = 0; i < nv; i++) { PUT32(tl[i]); memcpy(w, tp[i], (size_t)tl[i]); w += tl[i]; }
    if (scores) { memcpy(w, scores, (size_t)nv * 4); w += nv * 4; }
    for (uint64_t i = 0; i < nm; i++) { PUT32(ml[i]); memcpy(w, mp[i], (size_t)ml[i]); w += ml[i]; }
    #undef PUT32

    *vocab_out = (uint32_t)nv;
    *blob_size_out = sz;
    /* special-token ids */
    uint64_t v;
    bos_eos_pad_unk[0] = gguf_get_u64(g, "tokenizer.ggml.bos_token_id", &v) ? (uint32_t)v : 0xFFFFFFFFu;
    bos_eos_pad_unk[1] = gguf_get_u64(g, "tokenizer.ggml.eos_token_id", &v) ? (uint32_t)v : 0xFFFFFFFFu;
    bos_eos_pad_unk[2] = gguf_get_u64(g, "tokenizer.ggml.padding_token_id", &v) ? (uint32_t)v : 0xFFFFFFFFu;
    bos_eos_pad_unk[3] = gguf_get_u64(g, "tokenizer.ggml.unknown_token_id", &v) ? (uint32_t)v : 0xFFFFFFFFu;
    free(tp); free(tl); free(mp); free(ml);
    return blob;
}

/* Write the .sp-tokenizer file; sets sha32 to its SHA-256. */
static int write_tokenizer(const char *path, const gguf_ctx *g,
                           uint32_t *vocab_out, uint8_t sha32[32]) {
    uint64_t blob_size; uint32_t type_id, vocab, special[4];
    uint8_t *blob = build_tok_blob(g, &blob_size, &type_id, &vocab, special);
    if (!blob) return 1;

    sp_tok_header h; memset(&h, 0, sizeof h);
    h.magic = SP_TOK_MAGIC; h.version_major = SP_MODEL_VER_MAJOR; h.version_minor = SP_MODEL_VER_MINOR;
    h.header_size = SP_TOK_HEADER_SIZE; h.type_id = type_id; h.vocab_size = vocab;
    h.bos_token = special[0]; h.eos_token = special[1]; h.pad_token = special[2]; h.unk_token = special[3];
    h.blob_offset = SP_TOK_HEADER_SIZE; h.blob_size = blob_size;
    h.header_crc32 = sp_crc32(&h, SP_TOK_CRC_COVER);

    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", path); free(blob); return 1; }
    int ok = (fwrite(&h, 1, SP_TOK_HEADER_SIZE, f) == SP_TOK_HEADER_SIZE) &&
             (fwrite(blob, 1, (size_t)blob_size, f) == blob_size);
    fclose(f); free(blob);
    if (!ok) { fprintf(stderr, "tokenizer write short\n"); return 1; }

    /* SHA-256 over the entire produced file. */
    f = fopen(path, "rb");
    if (!f) return 1;
    sp_sha256_ctx sc; sp_sha256_init(&sc);
    uint8_t buf[65536]; size_t r;
    while ((r = fread(buf, 1, sizeof buf, f)) > 0) sp_sha256_update(&sc, buf, r);
    fclose(f);
    sp_sha256_final(&sc, sha32);
    *vocab_out = vocab;
    return 0;
}

/* Populate sp_arch_info into the 256-byte arch_struct (PPT-LAT-SP-MODEL-v0 §3).
 * Reads GGUF metadata directly from the already-open gguf_ctx — bypasses
 * qwen3_load so Qwen2.5 GGUFs (which omit attention.key_length) are handled. */
static int fill_arch_struct(const gguf_ctx *g, uint8_t arch_struct[256],
                            uint32_t *arch_id, uint32_t *arch_size, uint32_t *vocab) {
    const char *arch_str = gguf_get_str(g, "general.architecture");
    if (!arch_str) { fprintf(stderr, "fill_arch_struct: missing general.architecture\n"); return 1; }
    int is_gemma3 = (strcmp(arch_str, "gemma3") == 0);
    int is_qwen25 = (strcmp(arch_str, "qwen2")  == 0);
    int is_qwen3  = (strcmp(arch_str, "qwen3")  == 0);
    if (!is_gemma3 && !is_qwen25 && !is_qwen3) {
        fprintf(stderr, "fill_arch_struct: unsupported arch '%s'\n", arch_str); return 1;
    }

    char key[128];
    /* ARCH_KEY: build "arch.suffix" into local `key` and return it. */
    #define ARCH_KEY(suf) (snprintf(key, sizeof key, "%s." suf, arch_str), (const char *)key)

    uint64_t n_embd = 0, n_layers = 0, n_head = 0, n_head_kv = 0;
    uint64_t context_length = 0, n_ff = 0, head_dim = 0, swa_window = 0;
    float rope_freq_base = 1e6f, rms_eps = 1e-5f;

    if (!gguf_get_u64(g, ARCH_KEY("embedding_length"),  &n_embd)  || n_embd   == 0)
        { fprintf(stderr, "fill_arch_struct: missing %s.embedding_length\n",  arch_str); return 1; }
    if (!gguf_get_u64(g, ARCH_KEY("block_count"),       &n_layers) || n_layers == 0)
        { fprintf(stderr, "fill_arch_struct: missing %s.block_count\n",       arch_str); return 1; }
    if (!gguf_get_u64(g, ARCH_KEY("attention.head_count"), &n_head) || n_head  == 0)
        { fprintf(stderr, "fill_arch_struct: missing %s.attention.head_count\n", arch_str); return 1; }
    if (!gguf_get_u64(g, ARCH_KEY("attention.head_count_kv"), &n_head_kv)) n_head_kv = n_head;
    /* head_dim optional in Qwen2.5 GGUFs — derive from hidden_dim / n_heads. */
    if (!gguf_get_u64(g, ARCH_KEY("attention.key_length"), &head_dim) || head_dim == 0)
        head_dim = n_embd / n_head;
    gguf_get_u64(g, ARCH_KEY("context_length"),              &context_length);
    gguf_get_u64(g, ARCH_KEY("attention.sliding_window"),    &swa_window);
    gguf_get_u64(g, ARCH_KEY("feed_forward_length"),         &n_ff);
    gguf_get_f32(g, ARCH_KEY("rope.freq_base"),              &rope_freq_base);
    gguf_get_f32(g, ARCH_KEY("attention.layer_norm_rms_epsilon"), &rms_eps);
    #undef ARCH_KEY

    /* vocab from token_embd.weight dims[1]; fallback to tokenizer array length. */
    uint64_t n_vocab = 0;
    const gguf_tensor *embd = gguf_find_tensor(g, "token_embd.weight");
    if (embd && embd->n_dims >= 2) n_vocab = embd->dims[1];
    if (n_vocab == 0) {
        const gguf_kv *tk = gguf_find_kv(g, "tokenizer.ggml.tokens");
        if (tk && tk->type == GGUF_T_ARRAY) n_vocab = tk->arr_len;
    }
    if (n_vocab == 0) { fprintf(stderr, "fill_arch_struct: cannot determine vocab size\n"); return 1; }

    /* tied embeddings: output.weight absent → tied. */
    int tied = (gguf_find_tensor(g, "output.weight") == NULL) ? 1 : 0;
    /* QK norms: blk.0.attn_q_norm.weight present → has_qk_norm. */
    int has_qk_norm = (gguf_find_tensor(g, "blk.0.attn_q_norm.weight") != NULL) ? 1 : 0;

    sp_arch_info ai;
    memset(&ai, 0, sizeof ai);
    ai.arch_id          = is_gemma3 ? (uint32_t)SP_ARCH_ID_GEMMA3 :
                          is_qwen25 ? (uint32_t)SP_ARCH_ID_QWEN25 : (uint32_t)SP_ARCH_ID_QWEN3;
    ai.vocab_size       = (uint32_t)n_vocab;
    ai.hidden_dim       = (uint32_t)n_embd;
    ai.n_layers         = (uint32_t)n_layers;
    ai.n_heads          = (uint32_t)n_head;
    ai.n_kv_heads       = (uint32_t)n_head_kv;
    ai.head_dim         = (uint32_t)head_dim;
    ai.max_context      = (uint32_t)context_length;
    ai.swa_window       = (uint32_t)swa_window;
    ai.rope_freq_base   = rope_freq_base;
    ai.ffn_variant      = is_gemma3 ? 1u : 0u;   /* 1=GeGLU(gemma3), 0=SwiGLU(qwen3/qwen25) */
    ai.norm_variant     = is_gemma3 ? 1u : 0u;   /* 1=sandwich(gemma3), 0=pre-norm(qwen3/qwen25) */
    ai.tied_embeddings  = (uint32_t)tied;
    ai.has_qk_norm      = (uint32_t)has_qk_norm;
    ai.n_ff             = (uint32_t)n_ff;
    ai.rms_eps          = rms_eps;
    ai.preferred_precision = (uint32_t)SP_PRECISION_FP16;

    memset(arch_struct, 0, 256);
    memcpy(arch_struct, &ai, sizeof ai);
    *arch_id   = ai.arch_id;
    *arch_size = (uint32_t)sizeof(sp_arch_info);
    *vocab     = ai.vocab_size;
    return 0;
}

/* Decide per-tensor policy from the GGUF tensor name. Matmul weights -> Q8;
 * everything else (norms) -> F32. token_embd -> Q8 (the engine packs it in the
 * arena when SP_ARENA_EMBED). LM head "output.weight" (untied) -> Q8. */
static int is_matmul_weight(const char *name) {
    const char *bn = strrchr(name, '.');
    if (!bn) return 0;
    /* name forms: blk.N.<sub>.weight or token_embd.weight / output.weight */
    if (strstr(name, "_norm.weight")) return 0;   /* attn_norm, ffn_norm, q/k_norm, post_* */
    if (strcmp(name, "token_embd.weight") == 0) return 1;
    if (strcmp(name, "output.weight") == 0) return 1;
    if (strstr(name, "attn_q.weight") || strstr(name, "attn_k.weight") ||
        strstr(name, "attn_v.weight") || strstr(name, "attn_output.weight") ||
        strstr(name, "ffn_gate.weight") || strstr(name, "ffn_up.weight") ||
        strstr(name, "ffn_down.weight")) return 1;
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <in.gguf> <out.sp-model> <out.sp-tokenizer> [--verify]\n", argv[0]);
        return 2;
    }
    const char *in = argv[1], *out_model = argv[2], *out_tok = argv[3];
    int verify = (argc > 4 && strcmp(argv[4], "--verify") == 0);

    gguf_ctx *g = gguf_open(in);
    if (!g) { fprintf(stderr, "cannot open GGUF %s\n", in); return 1; }

    /* 1. tokenizer -> .sp-tokenizer + its SHA-256 */
    uint8_t tok_sha[32]; uint32_t tok_vocab = 0;
    if (write_tokenizer(out_tok, g, &tok_vocab, tok_sha)) { gguf_close(g); return 1; }

    /* 2. arch_struct from the already-open GGUF context */
    uint8_t arch_struct[256]; uint32_t arch_id, arch_size, arch_vocab;
    if (fill_arch_struct(g, arch_struct, &arch_id, &arch_size, &arch_vocab)) { gguf_close(g); return 1; }
    if (arch_vocab != tok_vocab)
        fprintf(stderr, "warning: model vocab %u != tokenizer vocab %u\n", arch_vocab, tok_vocab);

    /* 3. transcode every tensor. token_embd FIRST (§9 access-frequency order),
     * then per-layer blocks in GGUF order, then output_norm. */
    emit_list L = {0};
    int rc = 0;
    for (uint64_t i = 0; i < gguf_n_tensors(g) && rc == 0; i++) {
        const gguf_tensor *W = gguf_tensor_at(g, i);
        if (is_matmul_weight(W->name)) rc = add_q8(&L, g, W);
        else                            rc = add_f32(&L, g, W);
    }
    if (rc) { el_free(&L); gguf_close(g); return 1; }

    /* 4. assign data offsets in emit order (parent-then-scale already adjacent),
     * each tensor 64-aligned. */
    uint64_t data_cursor = 0;
    for (int i = 0; i < L.n; i++) {
        data_cursor = align_up(data_cursor, SP_TENSOR_ALIGN);
        L.t[i].data_off = data_cursor;
        data_cursor += L.t[i].size_bytes;
    }
    uint64_t data_region_size = data_cursor;

    /* 5. build the tensor table, sort by name_hash. Keep a parallel index into L
     * via the data_off (entries carry offset; data write follows emit order). */
    sp_tensor_entry *tbl = (sp_tensor_entry *)calloc((size_t)L.n, sizeof(sp_tensor_entry));
    if (!tbl) { el_free(&L); gguf_close(g); return 1; }
    for (int i = 0; i < L.n; i++) {
        sp_tensor_entry *e = &tbl[i];
        snprintf(e->name, sizeof e->name, "%s", L.t[i].name);
        e->dtype_id = L.t[i].dtype_id; e->n_dims = L.t[i].n_dims;
        for (int d = 0; d < 8; d++) e->dims[d] = L.t[i].dims[d];
        e->offset_in_data = L.t[i].data_off; e->size_bytes = L.t[i].size_bytes;
        e->block_size = L.t[i].block_size;
        e->block_count = e->block_size ? (uint32_t)(e->size_bytes / e->block_size) : 0;
        sp_blake3_256(L.t[i].bytes, (size_t)L.t[i].size_bytes, e->blake3);
        e->name_hash = sp_xxh64(e->name, strlen(e->name), 0);
    }
    qsort(tbl, (size_t)L.n, sizeof(sp_tensor_entry), cmp_entry);

    /* 6. header */
    uint64_t table_off = SP_HEADER_SIZE;
    uint64_t table_end = table_off + (uint64_t)L.n * SP_TENSOR_ENTRY_SIZE;
    uint64_t data_off  = align_up(table_end, SP_DATA_REGION_ALIGN);
    uint64_t file_size = data_off + data_region_size;

    sp_model_header h; memset(&h, 0, sizeof h);
    h.magic = SP_MODEL_MAGIC; h.version_major = SP_MODEL_VER_MAJOR; h.version_minor = SP_MODEL_VER_MINOR;
    h.header_size = SP_HEADER_SIZE; h.arch_id = arch_id;
    h.arch_struct_size = arch_size; h.arch_struct_capacity = 256;
    memcpy(h.arch_struct, arch_struct, 256);
    memcpy(h.tokenizer_hash, tok_sha, 32);
    h.vocab_size = arch_vocab; h.tensor_count = (uint32_t)L.n;
    h.tensor_table_offset = table_off; h.tensor_data_offset = data_off;
    h.file_size = file_size; h.created_unix_seconds = (uint64_t)time(NULL);
    h.transcoded_from = sp_xxh64(in, strlen(in), 0);
    h.header_crc32 = sp_crc32(&h, SP_HEADER_CRC_COVER);

    /* 7. write the file: header | table | pad | data (emit order) */
    FILE *f = fopen(out_model, "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", out_model); free(tbl); el_free(&L); gguf_close(g); return 1; }
    int ok = (fwrite(&h, 1, SP_HEADER_SIZE, f) == SP_HEADER_SIZE) &&
             (fwrite(tbl, sizeof(sp_tensor_entry), (size_t)L.n, f) == (size_t)L.n);
    /* pad to data_off */
    static const uint8_t zeros[65536] = {0};
    uint64_t pos = table_end;
    while (ok && pos < data_off) {
        uint64_t want = data_off - pos; if (want > sizeof zeros) want = sizeof zeros;
        ok = fwrite(zeros, 1, (size_t)want, f) == want; pos += want;
    }
    /* data region in emit order, 64-aligned per tensor */
    uint64_t dpos = 0;
    for (int i = 0; ok && i < L.n; i++) {
        uint64_t aligned = align_up(dpos, SP_TENSOR_ALIGN);
        while (ok && dpos < aligned) { uint64_t w = aligned - dpos; if (w > sizeof zeros) w = sizeof zeros; ok = fwrite(zeros,1,(size_t)w,f)==w; dpos += w; }
        ok = fwrite(L.t[i].bytes, 1, (size_t)L.t[i].size_bytes, f) == L.t[i].size_bytes;
        dpos += L.t[i].size_bytes;
    }
    fclose(f);
    if (!ok) { fprintf(stderr, "model write short\n"); free(tbl); el_free(&L); gguf_close(g); return 1; }

    fprintf(stderr, "[sp_transcode] %s -> %s (%u tensors, %llu bytes) + %s\n",
            in, out_model, h.tensor_count, (unsigned long long)file_size, out_tok);

    /* 8. --verify: sibling adjacency (§9) + reload sanity. */
    if (verify) {
        int warned = 0;
        /* For each Q8 weight in emit order, the next emit tensor must be its .scale
         * and they must be data-adjacent (parent end == scale start, both 64-aligned
         * boundaries notwithstanding — adjacency means no THIRD tensor between). */
        for (int i = 0; i + 1 < L.n; i++) {
            if (L.t[i].dtype_id == SP_DT_OK_Q8) {
                char want[80]; snprintf(want, sizeof want, "%s.scale", L.t[i].name);
                if (strcmp(L.t[i+1].name, want) != 0) {
                    fprintf(stderr, "[verify] WARN: %s scale not adjacent (next is %s)\n", L.t[i].name, L.t[i+1].name);
                    warned = 1;
                }
            }
        }
        sp_model *vm = NULL;
        sp_status st = sp_model_load(out_model, out_tok, &vm);
        if (st != SP_OK) { fprintf(stderr, "[verify] sp_model_load failed: %d (%s)\n", st, sp_last_error()); rc = 1; }
        else { fprintf(stderr, "[verify] load OK, %u tensors%s\n", sp_model_tensor_count(vm), warned ? " (adjacency warnings above)" : ""); sp_model_unload(vm); }
    }

    free(tbl); el_free(&L); gguf_close(g);
    return rc;
}
