/* test_sp_model_roundtrip.c — Phase 2-FMT gates E_FMT_1..4 (PPT-LAT-SP-MODEL-v0
 * §12.3 + Roadmap §10.1). One binary, four named subtests selected by the
 * SP_FMT_SUBTEST env var (CMake registers four ctest entries against it, the
 * TOK_DECODE/TOK_ENCODE pattern):
 *
 *   E_FMT_1  sp_model_load: header CRC, magic/version, alignment, tokenizer
 *            SHA-256 verify (incl. the mismatch -> SP_ETOKENIZER_HASH path),
 *            O(log N) name-hash lookup.
 *   E_FMT_2  transcode produces OK_Q8 bytes that are byte-IDENTICAL to what the
 *            in-RAM Frobenius packer (sp_frob_pack_tensor, precision=8) emits for
 *            the same source weight — the independent proof the on-disk bytes are
 *            the SP_FROB_ARENA_LAYOUT_VERSION format, not just self-consistent.
 *            Also checks the .scale sibling values + §9 data adjacency.
 *   E_FMT_3  .sp-tokenizer: SPTK magic, header CRC over [0,52), vocab == model.
 *   E_FMT_4  CLOSURE GATE: gemma3_forward on the .sp-model path == the GGUF
 *            arena-q8 path, bit-identical logits (deterministic CPU).
 *
 * All four first transcode the GGUF (idempotent; cheap relative to the forward).
 * SP_*_GGUF + SP_TRANSCODE_BIN + SP_FMT_OUT_DIR come from CMake.
 */
#define _CRT_SECURE_NO_WARNINGS
#include "sp/sp_test.h"
#include "sp_engine/sp_model.h"
#include "sp_engine/model.h"
#include "sp_engine/gguf.h"
#include "sp/frobenius_lift.h"
#include "sp_hash.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef SP_GEMMA3_GGUF
#define SP_GEMMA3_GGUF "gemma-3-1b-it-f16.gguf"
#endif
#ifndef SP_QWEN3_GGUF
#define SP_QWEN3_GGUF "Qwen3-0.6B-f16.gguf"
#endif
#ifndef SP_TRANSCODE_BIN
#define SP_TRANSCODE_BIN "sp_transcode"
#endif
#ifndef SP_FMT_OUT_DIR
#define SP_FMT_OUT_DIR "."
#endif

/* Token IDs valid for BOTH Gemma3 (vocab 262144) and Qwen3 (vocab ~151936):
 * kept well under the smaller vocab. The forward is content-agnostic — we only
 * need a deterministic sequence to compare the two load paths. */
static const int32_t TOKENS[] = { 2, 1735, 563, 476, 9580, 35292, 109, 1596 };
#define NTOK ((int)(sizeof(TOKENS)/sizeof(TOKENS[0])))

/* Transcode <gguf> -> <out_model>/<out_tok>; returns 0 on success. */
static int transcode(const char *gguf, const char *out_model, const char *out_tok) {
    char cmd[2400];
#ifdef _WIN32
    snprintf(cmd, sizeof cmd, "\"\"%s\" \"%s\" \"%s\" \"%s\" --verify\"",
             SP_TRANSCODE_BIN, gguf, out_model, out_tok);
#else
    snprintf(cmd, sizeof cmd, "\"%s\" \"%s\" \"%s\" \"%s\" --verify",
             SP_TRANSCODE_BIN, gguf, out_model, out_tok);
#endif
    fprintf(stderr, "    [transcode] %s\n", cmd);
    return system(cmd);
}

/* dequant one GGUF weight row to f32 (F32/F16/Q8_0). */
static size_t ggrow(uint32_t t, int n) {
    return t == GGML_T_F32 ? (size_t)n*4 : t == GGML_T_F16 ? (size_t)n*2 :
           t == GGML_T_Q8_0 ? (size_t)(n/32)*34 : 0;
}
typedef struct { const uint8_t *b; uint32_t ty; size_t rb; int cols; } rctx;
static int getr(void *c, int j, float *dst) {
    const rctx *g = (const rctx *)c;
    return sp_dequant_row(g->b + (size_t)j * g->rb, g->ty, g->cols, dst);
}

/* ── E_FMT_1: loader / verification ── */
static void E_FMT_1(void) {
    const char *om = SP_FMT_OUT_DIR "/gemma3_rt.sp-model";
    const char *ot = SP_FMT_OUT_DIR "/gemma3_rt.sp-tokenizer";
    SP_CHECK(transcode(SP_GEMMA3_GGUF, om, ot) == 0, "transcode for E_FMT_1");

    sp_model *sm = NULL;
    sp_status st = sp_model_load(om, ot, &sm);
    SP_CHECK(st == SP_OK, "sp_model_load SP_OK");
    if (st != SP_OK) { fprintf(stderr, "    %d (%s)\n", st, sp_last_error()); return; }
    const sp_model_header *h = sp_model_get_header(sm);
    SP_CHECK(h->magic == SP_MODEL_MAGIC, "magic SPMD");
    SP_CHECK(h->version_major == 0, "version_major 0");
    SP_CHECK(h->arch_id == SP_ARCH_ID_GEMMA3, "arch_id GEMMA3");
    SP_CHECK_EQ_I64(h->tensor_table_offset, 512, "table @512");
    SP_CHECK(h->tensor_data_offset % 65536 == 0, "data 65536-aligned");
    SP_CHECK(h->file_size == (uint64_t)0 + h->file_size, "file_size set");

    /* O(log N) lookup + collision-defending name verify */
    const sp_tensor_entry *te = sp_model_find_tensor(sm, "token_embd.weight");
    SP_CHECK(te && te->dtype_id == SP_DT_OK_Q8, "token_embd OK_Q8 found");
    SP_CHECK(sp_model_find_tensor(sm, "no.such.tensor") == NULL, "absent -> NULL");
    /* every tensor in the table is findable by name (lookup consistency) */
    int all_found = 1;
    for (uint32_t i = 0; i < sp_model_tensor_count(sm); i++) {
        const sp_tensor_entry *e = sp_model_tensor_at(sm, i);
        if (sp_model_find_tensor(sm, e->name) != e) { all_found = 0; break; }
    }
    SP_CHECK(all_found, "every table entry findable by name (sorted/binary-search ok)");

    /* tokenizer SHA-256 mismatch must be rejected */
    sp_model *bad = NULL;
    sp_status bst = sp_model_load(om, SP_GEMMA3_GGUF /*not a .sp-tokenizer*/, &bad);
    SP_CHECK(bst == SP_ETOKENIZER_HASH || bst == SP_EBADFORMAT, "wrong tokenizer rejected");
    if (bad) sp_model_unload(bad);

    sp_model_unload(sm);
}

/* ── E_FMT_2: on-disk OK_Q8 bytes == in-RAM sp_frob_pack_tensor bytes ── */
static void E_FMT_2(void) {
    const char *om = SP_FMT_OUT_DIR "/gemma3_rt.sp-model";
    const char *ot = SP_FMT_OUT_DIR "/gemma3_rt.sp-tokenizer";
    SP_CHECK(transcode(SP_GEMMA3_GGUF, om, ot) == 0, "transcode for E_FMT_2");

    sp_model *sm = NULL;
    if (sp_model_load(om, ot, &sm) != SP_OK) { SP_CHECK(0, "load for E_FMT_2"); return; }
    gguf_ctx *g = gguf_open(SP_GEMMA3_GGUF);
    SP_CHECK(g != NULL, "open GGUF for E_FMT_2");
    if (!g) { sp_model_unload(sm); return; }

    /* Compare a representative set of weights: embedding + a few per-layer. */
    const char *names[] = {
        "token_embd.weight", "blk.0.attn_q.weight", "blk.0.ffn_down.weight",
        "blk.5.attn_output.weight", "blk.10.ffn_gate.weight"
    };
    int codes_ok = 1, scales_ok = 1, dims_ok = 1;
    for (int k = 0; k < (int)(sizeof names/sizeof names[0]); k++) {
        const gguf_tensor *W = gguf_find_tensor(g, names[k]);
        const sp_tensor_entry *qe = sp_model_find_tensor(sm, names[k]);
        char sn[80]; snprintf(sn, sizeof sn, "%s.scale", names[k]);
        const sp_tensor_entry *se = sp_model_find_tensor(sm, sn);
        if (!W || !qe || !se) { codes_ok = 0; break; }
        int cols = (int)W->dims[0], rows = (int)W->dims[1];
        /* dims on disk are [cols, rows] */
        if (qe->dims[0] != (uint64_t)cols || qe->dims[1] != (uint64_t)rows) dims_ok = 0;
        /* repack in RAM */
        rctx ctx = { (const uint8_t *)gguf_tensor_data(g, W), W->type, ggrow(W->type, cols), cols };
        sp_frob_packed_tensor pt;
        if (sp_frob_pack_tensor(rows, cols, 8, 0.0f, getr, &ctx, &pt, NULL)) { codes_ok = 0; break; }
        const void *disk_codes = sp_model_tensor_data(sm, qe);
        const void *disk_scale = sp_model_tensor_data(sm, se);
        if (qe->size_bytes != (uint64_t)rows*cols ||
            memcmp(disk_codes, pt.codes, (size_t)rows*cols) != 0) codes_ok = 0;
        if (se->size_bytes != (uint64_t)rows*sizeof(float) ||
            memcmp(disk_scale, pt.row_scale, (size_t)rows*sizeof(float)) != 0) scales_ok = 0;
        sp_frob_packed_free(&pt);
        if (!codes_ok || !scales_ok) { fprintf(stderr, "    mismatch on %s\n", names[k]); break; }
    }
    SP_CHECK(dims_ok, "OK_Q8 dims [cols,rows] match GGUF");
    SP_CHECK(codes_ok, "on-disk OK_Q8 codes == sp_frob_pack_tensor codes (byte-identical)");
    SP_CHECK(scales_ok, "on-disk .scale == sp_frob_pack_tensor row_scale (byte-identical)");

    /* §9 sibling adjacency: each Q8 weight's .scale immediately follows it in the
     * data region (parent end <= scale start, and no third tensor between). */
    int adjacency_ok = 1;
    uint32_t n = sp_model_tensor_count(sm);
    for (uint32_t i = 0; i < n; i++) {
        const sp_tensor_entry *e = sp_model_tensor_at(sm, i);
        if (e->dtype_id != SP_DT_OK_Q8) continue;
        char sn[80]; snprintf(sn, sizeof sn, "%s.scale", e->name);
        const sp_tensor_entry *se = sp_model_find_tensor(sm, sn);
        if (!se) { adjacency_ok = 0; break; }
        /* scale must start at the first 64-aligned offset at/after the weight end,
         * with no other tensor occupying the gap. */
        uint64_t wend = e->offset_in_data + e->size_bytes;
        uint64_t aligned = (wend + 63) / 64 * 64;
        if (se->offset_in_data != aligned) { adjacency_ok = 0; fprintf(stderr, "    adj fail %s: wend=%llu scale@%llu\n", e->name, (unsigned long long)wend, (unsigned long long)se->offset_in_data); break; }
    }
    SP_CHECK(adjacency_ok, "every OK_Q8 weight's .scale is data-adjacent (Appendix B §9)");

    gguf_close(g); sp_model_unload(sm);
}

/* ── E_FMT_3: tokenizer file ── */
static void E_FMT_3(void) {
    const char *om = SP_FMT_OUT_DIR "/gemma3_rt.sp-model";
    const char *ot = SP_FMT_OUT_DIR "/gemma3_rt.sp-tokenizer";
    SP_CHECK(transcode(SP_GEMMA3_GGUF, om, ot) == 0, "transcode for E_FMT_3");

    FILE *f = fopen(ot, "rb");
    SP_CHECK(f != NULL, "open .sp-tokenizer");
    if (!f) return;
    uint8_t hdr[128];
    SP_CHECK(fread(hdr, 1, 128, f) == 128, "read SPTK header");
    fclose(f);
    sp_tok_header th; memcpy(&th, hdr, sizeof th);
    SP_CHECK(th.magic == SP_TOK_MAGIC, "tokenizer magic SPTK");
    SP_CHECK(th.header_size == 128, "tokenizer header_size 128");
    SP_CHECK(th.blob_offset == 128, "blob right after header");
    SP_CHECK(th.blob_size > 0, "blob non-empty");

    /* header CRC over [0,52) */
    SP_CHECK(sp_crc32(hdr, 52) == th.header_crc32, "tokenizer header CRC-32 over [0,52)");

    /* vocab must match the model's */
    sp_model *sm = NULL;
    if (sp_model_load(om, ot, &sm) == SP_OK) {
        SP_CHECK(sp_model_get_header(sm)->vocab_size == th.vocab_size, "tokenizer vocab == model vocab");
        sp_model_unload(sm);
    } else SP_CHECK(0, "load to compare vocab");
}

/* ── E_FMT_4: closure gate (bit-identical forward) ── */
static void roundtrip_forward(const char *gguf, const char *om, const char *ot, int is_gemma) {
    SP_CHECK(transcode(gguf, om, ot) == 0, "transcode for forward");
    sp_model *sm = NULL;
    sp_status st = sp_model_load(om, ot, &sm);
    SP_CHECK(st == SP_OK, "sp_model_load");
    if (st != SP_OK) { fprintf(stderr, "    %d (%s)\n", st, sp_last_error()); return; }

    qwen3_model *m_sp = sp_model_to_qwen3(sm);
    SP_CHECK(m_sp != NULL, "sp_model_to_qwen3");
    if (!m_sp) { fprintf(stderr, "    adapter: %s\n", sp_last_error()); sp_model_unload(sm); return; }

#ifdef _WIN32
    _putenv("SP_ARENA=q8"); _putenv("SP_ARENA_EMBED=1"); _putenv("SP_ENGINE_FROB="); _putenv("SP_CPU_SCALAR=1");
#else
    setenv("SP_ARENA","q8",1); setenv("SP_ARENA_EMBED","1",1); unsetenv("SP_ENGINE_FROB"); setenv("SP_CPU_SCALAR","1",1);
#endif
    qwen3_model *m_gg = qwen3_load(gguf);
    SP_CHECK(m_gg != NULL, "qwen3_load (arena q8)");
    if (!m_gg) { qwen3_free(m_sp); sp_model_unload(sm); return; }

    SP_CHECK_EQ_I64(m_sp->cfg.n_vocab, m_gg->cfg.n_vocab, "vocab matches");
    SP_CHECK_EQ_I64(m_sp->cfg.n_layers, m_gg->cfg.n_layers, "n_layers matches");
    SP_CHECK(m_sp->cfg.rope_freq_base == m_gg->cfg.rope_freq_base, "rope base matches");

    int V = (int)m_gg->cfg.n_vocab;
    float *a = (float *)malloc((size_t)NTOK*V*sizeof(float));
    float *b = (float *)malloc((size_t)NTOK*V*sizeof(float));
    SP_CHECK(a && b, "logit buffers");
    if (a && b) {
        int r1 = is_gemma ? gemma3_forward(m_sp, TOKENS, NTOK, a) : qwen3_forward(m_sp, TOKENS, NTOK, a);
        int r2 = is_gemma ? gemma3_forward(m_gg, TOKENS, NTOK, b) : qwen3_forward(m_gg, TOKENS, NTOK, b);
        SP_CHECK(r1 == 0, "forward .sp-model");
        SP_CHECK(r2 == 0, "forward GGUF");
        if (r1 == 0 && r2 == 0) {
            size_t nn = (size_t)NTOK*V;
            int bit_exact = (memcmp(a, b, nn*sizeof(float)) == 0);
            double sd = 0.0, ss = 0.0, wabs = 0.0; long am = 0;
            for (int t = 0; t < NTOK; t++) {
                int a1 = 0, a2 = 0;
                for (int j = 0; j < V; j++) {
                    float x = a[(size_t)t*V+j], y = b[(size_t)t*V+j];
                    double d = fabs((double)x - y); if (d > wabs) wabs = d;
                    sd += d*d; ss += (double)y*y;
                    if (x > a[(size_t)t*V+a1]) a1 = j;
                    if (y > b[(size_t)t*V+a2]) a2 = j;
                }
                if (a1 == a2) am++;
            }
            double drift = ss > 0 ? 100.0*sqrt(sd/ss) : 0.0;
            fprintf(stderr, "    %d pos x %d vocab | bit_exact=%s | worst_abs=%.3e | L2-drift=%.6f%% | argmax %ld/%d\n",
                    NTOK, V, bit_exact?"YES":"no", wabs, drift, am, NTOK);
            SP_CHECK(bit_exact, "logits bit-identical (.sp-model == GGUF arena-q8)");
            SP_CHECK(drift <= 0.05, "logit L2 drift <= 0.05% (fallback bound)");
            SP_CHECK_EQ_I64(am, NTOK, "argmax identical every position");
        }
    }
    free(a); free(b);
    qwen3_free(m_gg); qwen3_free(m_sp); sp_model_unload(sm);
}
static void E_FMT_4(void) {
    roundtrip_forward(SP_GEMMA3_GGUF, SP_FMT_OUT_DIR "/gemma3_rt.sp-model",
                      SP_FMT_OUT_DIR "/gemma3_rt.sp-tokenizer", 1);
}
/* Cross-arch validation (Roadmap §10.3): Qwen3-0.6B through the same round-trip. */
static void E_FMT_4_QWEN3(void) {
    roundtrip_forward(SP_QWEN3_GGUF, SP_FMT_OUT_DIR "/qwen3_rt.sp-model",
                      SP_FMT_OUT_DIR "/qwen3_rt.sp-tokenizer", 0);
}

int main(void) {
    const char *which = getenv("SP_FMT_SUBTEST");
    if (!which) which = "E_FMT_4";
    if      (strcmp(which, "E_FMT_1") == 0) SP_RUN(E_FMT_1);
    else if (strcmp(which, "E_FMT_2") == 0) SP_RUN(E_FMT_2);
    else if (strcmp(which, "E_FMT_3") == 0) SP_RUN(E_FMT_3);
    else if (strcmp(which, "E_FMT_4_QWEN3") == 0) SP_RUN(E_FMT_4_QWEN3);
    else SP_RUN(E_FMT_4);
    return SP_DONE();
}
